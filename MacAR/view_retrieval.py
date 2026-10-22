"""Object-region SIFT or DINOv2 retrieval over virtual camera views.

The virtual-view localizer can also crop live ALIKED extraction to the
detected object; matched coordinates are restored before XYZ transfer/PnP.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from object_roi import GSObjectDetector, reference_rgb_mask
from patch_view_rerank import GRID_SIDE, SHORTLIST, square_mask, rerank, select_diverse


MODES = ("sift", "dino-global", "dino-spatial", "dino-patch")
DEFAULT_ROI = (0.10, 0.10, 0.80, 0.80)  # x, y, width, height, fractions


def crop_box(image, roi):
    """Return a bounded normalized-ROI crop without changing the source image."""
    if roi is None:
        return image
    x, y, width, height = roi
    rows, cols = image.shape[:2]
    x0 = int(np.floor(x * cols))
    y0 = int(np.floor(y * rows))
    x1 = int(np.ceil((x + width) * cols))
    y1 = int(np.ceil((y + height) * rows))
    return image[max(0, y0):min(rows, y1), max(0, x0):min(cols, x1)]


def object_bbox(mask, margin=0.08):
    """Bounding box from the virtual-view geometry validity mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    left, top = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    right, bottom = min(width, int(x1 + dx)), min(height, int(y1 + dy))
    return (left / width, top / height, (right - left) / width,
            (bottom - top) / height)


def square_rgb(image, side=224):
    """Preserve crop aspect ratio; pad with a neutral background."""
    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)),
                                 max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((side, side, 3), 127, dtype=np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class DINOv2Encoder:
    """Frozen DINOv2 ViT-S/14 with global or spatially pooled patch output."""

    model_name = "dinov2_vits14_reg"
    # The current main branch uses Python 3.10-only annotations. This official
    # revision precedes that change and loads on this project's Python 3.9.
    repository = "facebookresearch/dinov2:b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"

    def __init__(self, mode, device="auto", model=None):
        if mode not in {"dino-global", "dino-spatial", "dino-patch"}:
            raise ValueError(f"Unknown DINOv2 descriptor: {mode}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("DINOv2 retrieval requires PyTorch") from exc
        self.torch = torch
        self.mode = mode
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if model is None:
            try:
                model = torch.hub.load(self.repository, self.model_name,
                                       trust_repo=True, skip_validation=True)
            except Exception as exc:
                raise RuntimeError(
                    "DINOv2 could not be loaded. Connect once for the pinned official "
                    "torch.hub model/weights download, or use --retrieval-mode sift. "
                    f"Original error: {exc}"
                ) from exc
        self.model = model.eval().to(device)
        self.lock = threading.Lock()
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    def encode(self, bgr):
        rgb = square_rgb(bgr)
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        # The live coarse viewer and background relocalizer may infer at once.
        with self.lock, torch.inference_mode():
            features = self.model.forward_features(tensor)
            cls = features["x_norm_clstoken"][0]
            if self.mode in {"dino-global", "dino-patch"}:
                vector = cls
            else:
                patches = features["x_norm_patchtokens"][0]
                side = int(round(patches.shape[0] ** 0.5))
                if side * side != patches.shape[0] or side % 2:
                    raise RuntimeError("Unexpected DINOv2 patch layout")
                patches = patches.reshape(side, side, -1)
                half = side // 2
                quadrants = [patches[y:y + half, x:x + half].mean((0, 1))
                             for y in (0, half) for x in (0, half)]
                vector = torch.cat((cls * 0.5, *quadrants))
            vector = vector.float().cpu().numpy()
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-12:
            raise RuntimeError("DINOv2 returned an invalid embedding")
        return (vector / norm).astype(np.float32)

    def encode_tokens(self, bgr):
        """Global vector + spatial patches from the SAME cropped-image forward."""
        torch = self.torch
        rgb = square_rgb(bgr)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()/255.
        tensor = (tensor-self.mean)/self.std
        with self.lock, torch.inference_mode():
            features = self.model.forward_features(tensor)
            cls = features['x_norm_clstoken'][0].float().cpu().numpy()
            patches = features['x_norm_patchtokens'][0].float().cpu().numpy()
        side = int(round(np.sqrt(len(patches))))
        if side != GRID_SIDE or not np.isfinite(patches).all() or not np.isfinite(cls).all():
            raise RuntimeError('Unexpected/invalid DINO patch layout')
        pooled = patches.reshape(GRID_SIDE, GRID_SIDE, -1)
        norm = float(np.linalg.norm(cls))
        if norm < 1e-8:
            raise RuntimeError('Invalid DINO global embedding')
        pooled /= np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
        return (cls/norm).astype(np.float32), pooled.astype(np.float32)

    def patch_grid(self, bgr, max_side=448):
        """L2-normalized dense patch descriptors on an aspect-preserving grid."""
        height, width = bgr.shape[:2]
        scale = max_side / max(height, width)
        columns = max(2, round(width * scale / 14))
        rows = max(2, round(height * scale / 14))
        resized = cv2.resize(bgr, (columns * 14, rows * 14),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        with self.lock, torch.inference_mode():
            patches = self.model.forward_features(tensor)["x_norm_patchtokens"][0]
            patches = torch.nn.functional.normalize(patches.float(), dim=-1)
            values = patches.cpu().numpy().reshape(rows, columns, -1)
        if not np.isfinite(values).all():
            raise RuntimeError("DINOv2 returned invalid patch features")
        return values


class ViewRetrievalIndex:
    """One interchangeable coarse-view index; scores are not pose confidence."""

    def __init__(self, views_path: Path, views, *, mode="sift", crop=False,
                 roi=DEFAULT_ROI, ratio=0.78, device="auto", encoder=None,
                 auto_roi=False):
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if not 0 < ratio < 1:
            raise ValueError("SIFT ratio must be in (0, 1)")
        if len(roi) != 4 or any(not np.isfinite(v) for v in roi):
            raise ValueError("ROI must be four finite fractions")
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("ROI must fit inside the image")
        self.views_path = Path(views_path)
        self.views = views
        self.mode = mode
        if auto_roi and mode == "sift":
            raise ValueError("Automatic object ROI requires DINOv2 retrieval")
        self.crop = crop or auto_roi
        self.auto_roi = auto_roi
        self.roi = tuple(float(v) for v in roi)
        self.ratio = ratio
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.rank_lock = threading.Lock()
        self.encoder = (encoder if encoder is not None else DINOv2Encoder(mode, device)
                        if mode != "sift" else None)
        self.reference = self._build_index()
        self.detector = (GSObjectDetector(self.encoder, self.views_path, self.views)
                         if auto_roi else None)
        self.last_query_bbox = None
        self.last_detection = None
        self.last_query_tokens = None
        self.query_max_width = None  # Photo analysis only; live defaults are unchanged.

    def _reference_crop(self, image, view):
        if not self.crop:
            return image
        mask = reference_rgb_mask(image)
        bbox = object_bbox(mask)
        if bbox is None:
            raise RuntimeError(f"Empty reference object mask: {view['image_name']}")
        return crop_box(image, bbox)

    def _cache_path(self):
        manifest = []
        for view in self.views:
            for name in (view["image_name"],):
                if name is None:
                    continue
                file = self.views_path / str(name)
                stat = file.stat()
                manifest.append((str(name), stat.st_size, stat.st_mtime_ns))
        key = json.dumps(("v4-patch16-masked" if self.mode == 'dino-patch' else "v2-rgb-mask", self.mode, self.crop,
                          getattr(self.encoder, "model_name", None),
                          getattr(self.encoder, "repository", None), manifest),
                         sort_keys=True).encode()
        return self.views_path / ".view_retrieval_cache" / (hashlib.sha256(key).hexdigest() + ".npz")

    def _build_index(self):
        if self.mode == 'dino-patch':
            return self._build_patch_index()
        if self.mode != "sift":
            cache = self._cache_path()
            if cache.exists():
                with np.load(cache, allow_pickle=False) as data:
                    values = data["embeddings"]
                norms = (np.linalg.norm(values.astype(np.float64), axis=1)
                         if values.ndim == 2 else np.array([]))
                if (values.ndim == 2 and values.shape[0] == len(self.views)
                    and np.isfinite(values).all()
                    and np.all(np.abs(norms - 1.0) < 1e-3)):
                    return values
        result = []
        for view in self.views:
            image = cv2.imread(str(self.views_path / str(view["image_name"])))
            if image is None:
                raise RuntimeError(f"Cannot read reference image: {view['image_name']}")
            image = self._reference_crop(image, view)
            if self.mode == "sift":
                _, descriptor = self.sift.detectAndCompute(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
                result.append(descriptor)
            else:
                result.append(self.encoder.encode(image))
        if self.mode == "sift":
            if not any(d is not None and len(d) >= 2 for d in result):
                raise RuntimeError("No reference images have SIFT descriptors")
            return result
        values = np.stack(result).astype(np.float32)
        cache.parent.mkdir(exist_ok=True)
        temporary = cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, embeddings=values)
        temporary.replace(cache)
        return values

    def _build_patch_index(self):
        cache = self._cache_path()
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as data:
                    values, patches, masks = data['embeddings'], data['patches'], data['masks']
                if (values.ndim != 2 or values.shape[0] != len(self.views) or
                    patches.shape != (len(self.views), GRID_SIDE, GRID_SIDE, values.shape[1]) or
                    masks.shape != patches.shape[:3] or
                    not np.isfinite(values).all() or not np.isfinite(patches).all() or
                    not np.all(np.abs(np.linalg.norm(values, axis=1)-1.) < 1e-3)):
                    raise ValueError('Invalid patch index')
                self.reference_patches, self.reference_masks = patches, masks.astype(bool)
                return values
            except (OSError, ValueError, KeyError):
                pass
        vectors, grids, masks = [], [], []
        for view in self.views:
            image = cv2.imread(str(self.views_path/view['image_name']))
            if image is None:
                raise RuntimeError(f"Cannot read reference image: {view['image_name']}")
            mask = reference_rgb_mask(image)
            bbox = object_bbox(mask) if self.crop else None
            cropped, cropped_mask = crop_box(image, bbox), crop_box(mask, bbox)
            vector, patches = self.encoder.encode_tokens(cropped)
            vectors.append(vector)
            grids.append(patches)
            masks.append(square_mask(cropped_mask))
        values = np.stack(vectors).astype(np.float32)
        self.reference_patches = np.stack(grids).astype(np.float32)
        self.reference_masks = np.stack(masks).astype(bool)
        try:
            cache.parent.mkdir(exist_ok=True)
            temporary = cache.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, embeddings=values, patches=self.reference_patches,
                                masks=self.reference_masks)
            temporary.replace(cache)
        except OSError:
            pass  # A read-only dataset still works; the RAM index is valid.
        return values

    def rank(self, image, top_k=None, excluded=(), include_zero=False):
        """Return [(score, -tie_break, index)], query feature count, full ranking."""
        with self.rank_lock:
            ranked, count, all_ranked, _ = self._rank(
                image, top_k, excluded, include_zero, preview=False)
            return ranked, count, all_ranked

    def rank_debug(self, image, top_k=None, excluded=(), include_zero=False,
                   capture_query_tokens=False):
        """Rank and return the exact square RGB input used by the embedding."""
        with self.rank_lock:
            result = self._rank(image, top_k, excluded, include_zero, preview=True)
            if capture_query_tokens:
                # Snapshot before the viewer can rank a different frame on this index.
                result[3]['_query_tokens'] = self.last_query_tokens
            return result

    def reference_preview(self, index):
        view = self.views[index]
        image = cv2.imread(str(self.views_path / view["image_name"]))
        if image is None:
            raise RuntimeError(f"Cannot read reference image: {view['image_name']}")
        image = self._reference_crop(image, view)
        return cv2.cvtColor(square_rgb(image), cv2.COLOR_RGB2BGR)

    def _query_image(self, image):
        limit = self.query_max_width
        if limit and image.shape[1] > limit:
            height = max(1, round(image.shape[0] * limit / image.shape[1]))
            return cv2.resize(image, (limit, height), interpolation=cv2.INTER_AREA)
        return image

    def detect_query(self, image):
        """Fixed-reference ROI detection uses the same photo analysis scale.

        The returned bbox and compact mask stay normalized to the full image.
        No resized pixel coordinates are passed to fine matching or PnP.
        """
        return self.detector.detect(self._query_image(image))

    def _rank(self, image, top_k, excluded, include_zero, preview):
        started = time.perf_counter()
        image = self._query_image(image)
        self.last_query_tokens = None
        input_shape = image.shape[:2]
        detection = None
        bbox = None
        if self.crop:
            if self.auto_roi:
                bbox, detection = self.detector.detect(image)
            else:
                bbox = self.roi
            self.last_query_bbox = bbox
            self.last_detection = detection
            if bbox is None:
                debug = dict(bbox=None, detection=detection,
                             query_preview=None, reference_preview=None,
                             full_preview=cv2.cvtColor(square_rgb(image), cv2.COLOR_RGB2BGR)
                             if preview else None)
                return [], 0, [], debug
            image = crop_box(image, bbox)
        else:
            self.last_query_bbox = None
            self.last_detection = None
        debug = dict(bbox=bbox, detection=detection,
                     query_preview=(cv2.cvtColor(square_rgb(image), cv2.COLOR_RGB2BGR)
                                    if preview else None), reference_preview=None,
                     full_preview=None)
        excluded = set(excluded)
        ranked = []
        if self.mode == "sift":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptor = self.sift.detectAndCompute(gray, None)
            if descriptor is None:
                return [], 0, [], debug
            for index, reference in enumerate(self.reference):
                if index in excluded or reference is None or len(reference) < 2:
                    continue
                neighbors = self.matcher.knnMatch(descriptor, reference, k=2)
                accepted = [pair[0].distance for pair in neighbors if len(pair) == 2
                            and pair[0].distance < self.ratio * pair[1].distance]
                if accepted or include_zero:
                    ranked.append((len(accepted), -float(np.mean(accepted))
                                   if accepted else -float("inf"), index))
            count = len(keypoints)
        else:
            if self.mode == 'dino-patch':
                query, patches = self.encoder.encode_tokens(image)
            else:
                query = self.encoder.encode(image)
            # The reference set is small enough that float64 accumulation avoids
            # Accelerate/BLAS float32 overflow warnings observed on macOS.
            scores = np.einsum("ij,j->i", self.reference, query,
                               dtype=np.float64, optimize=False)
            ranked = [(float(score), 0.0, index) for index, score in enumerate(scores)
                      if index not in excluded]
            count = 0
        ranked.sort(reverse=True)
        if self.mode == 'dino-patch':
            global_ranked = list(ranked)
            # Use the detector's object mask, cropped by exactly the embedding ROI.
            # Without a detector, valid crop pixels exclude letterbox padding.
            full_mask = (detection or {}).get('mask')
            if full_mask is None:
                mask = np.ones(image.shape[:2], np.uint8)
            else:
                full_mask = cv2.resize(full_mask.astype(np.uint8),
                                      (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = crop_box(full_mask, bbox)
            query_mask = square_mask(mask)
            # Shared only with the same engine's registered-photo selection;
            # keep arrays out of public crop/debug JSON.
            self.last_query_tokens = (query, patches, query_mask)
            rerank_started = time.perf_counter()
            shortlist = ranked[:max(SHORTLIST, top_k or 0)]
            ranked, rows = rerank(shortlist, patches, query_mask,
                                  self.reference_patches, self.reference_masks)
            patch_top1 = ranked[0][2] if ranked else None
            selected = select_diverse(ranked, self.views, min(top_k or 3, len(ranked)))
            selected_ids = [entry[2] for entry in selected]
            ranked = selected + [entry for entry in ranked if entry[2] not in selected_ids]
            for row in rows:
                row['image_name'] = self.views[row['index']]['image_name']
                row['fine_selected'] = row['index'] in selected_ids
            debug['patch_rerank'] = dict(
                global_top1_index=global_ranked[0][2] if global_ranked else None,
                reranked_top1_index=patch_top1,
                selected_indices=selected_ids,
                selected_images=[self.views[index]['image_name'] for index in selected_ids],
                diversity_angle_deg=35.,
                shortlist=len(shortlist), fine_candidates=top_k,
                patch_grid=GRID_SIDE, query_foreground_patches=int(query_mask.sum()),
                rerank_ms=(time.perf_counter()-rerank_started)*1000., rows=rows,
                retrieval_ms=(time.perf_counter()-started)*1000.)
        if preview and ranked:
            debug['reference_preview'] = self.reference_preview(ranked[0][2])
            debug['reference_name'] = self.views[ranked[0][2]]['image_name']
        return ranked[:top_k] if top_k is not None else ranked, count, ranked, debug


def draw_crop_comparison(debug, *, mode="dino-global"):
    """Show the exact 224x224 live/reference inputs to coarse embedding."""
    panel = np.full((290, 470, 3), 28, np.uint8)
    live = debug.get('query_preview')
    reference = debug.get('reference_preview')
    if live is not None:
        panel[42:266, 8:232] = live
    elif debug.get('full_preview') is not None:
        panel[42:266, 8:232] = debug['full_preview']
    if reference is not None:
        panel[42:266, 238:462] = reference
    live_label = f"LIVE {mode}" if live is not None else "LIVE - NOT EMBEDDED"
    cv2.putText(panel, live_label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (220, 220, 220), 1, cv2.LINE_AA)
    name = debug.get('reference_name', 'NO VIEW')
    cv2.putText(panel, f"3DGS {name}", (238, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (220, 220, 220), 1, cv2.LINE_AA)
    detection = debug.get('detection') or {}
    status = detection.get('reason', 'fixed/full ROI')
    cv2.putText(panel, str(status), (8, 282), cv2.FONT_HERSHEY_SIMPLEX,
                .43, (0, 180, 255) if status != 'detected' else (50, 220, 50),
                1, cv2.LINE_AA)
    return panel
