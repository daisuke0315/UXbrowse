"""GS-constrained appearance selection, plus isolated comparison helpers.

Keep the coarse GS direction gate for gs-real. The direct arm deliberately
does not have this independent gate, and must be audited for false poses.
"""
import hashlib
from macar_paths import DATA_DIR
import inspect
import json
import os
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np

from patch_view_rerank import SHORTLIST, rerank, square_mask


class RegisteredAppearanceIndex:
    """Immutable real-photo crop tokens; never supplies or modifies XYZ."""

    def __init__(self, map_path, registered, encoder, *, cache_root=None, progress=None):
        self.encoder = encoder
        self.names = sorted(registered)
        root = Path(map_path).resolve()
        summary = json.loads((root/'summary.json').read_text())
        # Current exported/enriched maps can use the COLMAP dataset layout
        # without an explicit RGB directory. Honour custom-directory maps too.
        image_dir = (Path(summary['image_dir']) if summary.get('image_dir') else
                     Path(summary['input_files']['images']).parents[2]/'images')
        sources = []
        for name in self.names:
            paths = [Path(registered[name]['path']), image_dir/name]
            sources.append((name, [(str(p.resolve()), p.stat().st_size,
                                     p.stat().st_mtime_ns) for p in paths]))
        # Include token/mask preprocessing code, so cache cannot outlive its semantics.
        from view_retrieval import DINOv2Encoder, square_rgb
        code = [hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                for f in (DINOv2Encoder, square_rgb, square_mask)]
        identity = json.dumps(('registered-dino-v1', str(root), encoder.model_name,
                              encoder.repository, code, sources), sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        cache_root = (Path(cache_root) if cache_root is not None else
                      DATA_DIR/'.model_cache/registered_dino')
        self.cache_path = cache_root/(key+'.npz')
        notify = progress or (lambda **_: None)
        notify(appearance_loaded=0, appearance_total=len(self.names), appearance_cache='loading')
        if self.cache_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as a:
                    if str(a['identity']) != identity or a['names'].tolist() != self.names:
                        raise ValueError('Registered DINO cache identity mismatch')
                    self.embeddings, self.patches, self.masks = [a[k].copy()
                        for k in ('embeddings','patches','masks')]
                self._validate()
                notify(appearance_loaded=len(self.names), appearance_total=len(self.names),
                       appearance_cache='loaded')
                return
            except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile):
                notify(appearance_cache='rebuilding')
        vectors, patches, masks = [], [], []
        for number, name in enumerate(self.names, 1):
            image = cv2.imread(str(image_dir/name))
            if image is None:
                raise ValueError(f'Cannot read registered RGB: {image_dir/name}')
            with np.load(registered[name]['path'], allow_pickle=False) as a:
                box, mask = a['crop_box'].copy(), a['object_mask'].copy()
            if box.shape != (4,) or not np.isfinite(box).all() or not np.equal(box, np.round(box)).all():
                raise ValueError(f'Invalid registered crop/mask: {name}')
            x0,y0,x1,y1 = box.astype(int)
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                raise ValueError(f'Invalid registered crop/mask: {name}')
            crop = image[y0:y1,x0:x1]
            if crop.shape[:2] != mask.shape or not mask.any():
                raise ValueError(f'Invalid registered crop/mask: {name}')
            vector, grid = encoder.encode_tokens(crop)
            vectors.append(vector); patches.append(grid); masks.append(square_mask(mask))
            notify(appearance_loaded=number, appearance_total=len(self.names),
                   appearance_cache='building')
        self.embeddings = np.stack(vectors).astype(np.float32)
        self.patches = np.stack(patches).astype(np.float32)
        self.masks = np.stack(masks).astype(bool)
        self._validate()
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix='.npz', delete=False) as file:
                temporary = Path(file.name)
                np.savez_compressed(file, identity=identity, names=np.asarray(self.names),
                    embeddings=self.embeddings, patches=self.patches, masks=self.masks)
            os.replace(temporary, self.cache_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        notify(appearance_loaded=len(self.names), appearance_total=len(self.names),
               appearance_cache='built')

    def _validate(self):
        n = len(self.names)
        if (self.embeddings.ndim != 2 or self.embeddings.shape[0] != n or
            self.patches.shape != (n,16,16,self.embeddings.shape[1]) or
            self.masks.shape != (n,16,16) or self.masks.dtype != bool or
            not np.isfinite(self.embeddings).all() or not np.isfinite(self.patches).all() or
            not np.all(np.abs(np.linalg.norm(self.embeddings,axis=1)-1.) < 1e-3)):
            raise ValueError('Invalid registered DINO cache arrays')

    def select(self, views, registered, coarse, query_tokens, *, budget, max_angle):
        from registered_aliked_live import rotation_angle
        vector, patches, mask = query_tokens
        rankings = {}
        for _, _, index in coarse:
            rotation = np.asarray(views[index]['world_to_camera'])[:3,:3]
            eligible = {name for name, item in registered.items()
                        if rotation_angle(item['rotation'],rotation) <= max_angle}
            rankings[index] = rank_real_references(self.names, self.embeddings,
                self.patches, self.masks, vector, patches, mask, eligible)
        return select_appearance_references(views, registered, coarse, rankings,
            mode='gs-real', budget=budget, max_angle=max_angle)


def rank_real_references(names, embeddings, patches, masks, query, query_patches,
                         query_mask, eligible):
    """Filter eligibility BEFORE global Top10 and patch reranking."""
    scores = np.einsum('ij,j->i', embeddings, query, dtype=np.float64, optimize=False)
    shortlist = sorted(((float(score), 0., i) for i, score in enumerate(scores)
                        if names[i] in eligible), reverse=True)[:SHORTLIST]
    if not shortlist:
        return []
    ranked, _ = rerank(shortlist, query_patches, query_mask, patches, masks)
    return [(names[index], float(score)) for score, _, index in ranked]


def select_appearance_references(views, registered, coarse, rankings, *,
                                 mode, per_view=2, budget=6, max_angle=55.):
    """Return exact ranked real photos in the production (name, view, score, angle) format."""
    from registered_aliked_live import rotation_angle
    if mode not in ('real-direct', 'gs-real'):
        raise ValueError('Unknown appearance shortlist mode')
    rotations = [np.asarray(view['world_to_camera'], np.float64)[:3, :3] for view in views]
    selected, seen = [], set()
    groups = [(None, rankings.get(None, []))] if mode == 'real-direct' else [
        (index, rankings.get(index, [])) for _, _, index in coarse]
    for index, ranked in groups:
        added = 0
        for name, score in ranked:
            if name in seen or name not in registered:
                continue
            rotation = registered[name]['rotation']
            if index is None:
                angles = [rotation_angle(rotation, R) for R in rotations]
                virtual_index = int(np.argmin(angles))
                angle = float(angles[virtual_index])
            else:
                virtual_index = index
                angle = rotation_angle(rotation, rotations[index])
                if angle > max_angle:
                    continue
            selected.append((name, virtual_index, float(score), angle))
            seen.add(name)
            added += 1
            if len(selected) >= budget:
                return selected
            if mode == 'gs-real' and added >= per_view:
                break
    return selected
