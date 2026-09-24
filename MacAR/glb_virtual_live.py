#!/usr/bin/env python3
"""Live virtual-view localization for GLB or 3DGS reference maps."""

from __future__ import annotations

import math
import concurrent.futures
import threading
import time
import json
import hashlib
import os
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from aliked_lightglue_pnp import (
    _extract,
    _image_tensor,
    _load_models,
    _match_pair,
    geometric_filter,
    solve_pose,
)
from glb_virtual_view_pnp import (
    VirtualTransfer,
    load_manifest,
    make_transfers,
)
from offline_pnp_test import MapData
from pose_quality import assess_pose
from realtime_pnp_server import (
    LivePnPError,
    LiveResult,
    LocalizationDebug,
    TrackingSeed,
    TemporalGlobalPnPLocalizer,
    rc_to_colmap,
)
from view_retrieval import DEFAULT_ROI, ViewRetrievalIndex


class AsyncTemporalGLBLocalizer:
    """Run slow learned relocalization in background and track in the foreground."""

    def __init__(
        self,
        global_localizer,
        *,
        reseed_frames: int,
        min_track_points: int,
        flow_window: int,
        flow_levels: int,
        forward_backward_error: float,
        tracking_iterations: int,
        max_seed_age: float = 5.0,
    ) -> None:
        self.global_localizer = global_localizer
        self.tracker = TemporalGlobalPnPLocalizer(
            global_localizer,
            reseed_frames=reseed_frames,
            min_track_points=min_track_points,
            flow_window=flow_window,
            flow_levels=flow_levels,
            forward_backward_error=forward_backward_error,
            tracking_iterations=tracking_iterations,
        )
        self.map_data = global_localizer.map_data
        self.mpr_renderer = global_localizer.mpr_renderer
        # Learned relocalization runs on a worker.  Render MPR only on the
        # foreground tracking frame so one renderer is never used concurrently.
        self.global_localizer.mpr_renderer = None
        self.tracker.mpr_renderer = self.mpr_renderer
        self.last_debug = global_localizer.last_debug
        self.reseed_frames = reseed_frames
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="glb-relocalize"
        )
        self.future = None
        self.search_started = 0.0
        self.last_search_result = None
        self.last_search_display = None
        self.last_search_diagnostics = {}
        self.max_seed_age = max_seed_age
        self.last_search_age_ms = 0.0
        self.pending_seed = None
        self.pending_reference_name = None
        self.active_reference_name = None
        self.coarse_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='glb-coarse-view')
        self.coarse_future = None
        self.coarse_submitted = 0.0
        self.coarse_reference_name = None
        self.coarse_score = 0
        self.coarse_margin = 0.0
        self.coarse_age_ms = 0.0
        self.coarse_crop_debug = {}

    @property
    def source_label(self) -> str:
        return self.global_localizer.source_label

    def _submit(self, image, camera_matrix, distortion) -> None:
        self.search_started = time.monotonic()
        self.future = self.executor.submit(
            self.global_localizer.localize,
            image.copy(),
            camera_matrix.copy(),
            distortion.copy(),
        )

    def _coarse_search(self, image):
        ranked, _, _, debug = self.global_localizer.retriever.rank_debug(
            image, top_k=2, excluded=self.global_localizer.excluded_retrieval_views)
        if not ranked or (self.global_localizer.retrieval_mode == 'sift' and ranked[0][0] == 0):
            return None, 0, 0.0, debug
        top, _, index = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0
        margin = top / runner_up if runner_up > 0 else 0.0
        return self.global_localizer.views[index]['image_name'], top, margin, debug

    def _update_coarse(self, image):
        now = time.monotonic()
        if self.coarse_future is not None and self.coarse_future.done():
            (self.coarse_reference_name, self.coarse_score,
             self.coarse_margin, self.coarse_crop_debug) = self.coarse_future.result()
            self.coarse_age_ms = (now - self.coarse_submitted) * 1000.
            self.coarse_future = None
        if self.coarse_future is None and now-self.coarse_submitted >= 1.0:
            self.coarse_submitted = now
            self.coarse_future = self.coarse_executor.submit(self._coarse_search, image.copy())

    def _consume_completed(self) -> None:
        if self.future is None or not self.future.done():
            return
        self.last_search_result, self.last_search_display = self.future.result()
        self.last_debug = self.global_localizer.last_debug
        self.last_search_diagnostics = getattr(self.global_localizer, 'last_diagnostics', {})
        self.future = None
        self.last_search_age_ms = (time.monotonic()-self.search_started)*1000
        seed = self.global_localizer.last_tracking_seed
        if seed is not None and self.tracker.seed is not None:
            # Never overwrite a working track with a search-frame seed before
            # validating that seed against the *current* frame.
            self.pending_seed = self.tracker._copy_seed(seed) if seed is not None else None
            self.pending_reference_name = (self.global_localizer.last_reference_name
                                           if seed is not None else None)
            seed = None
        if seed is not None:
            self.tracker.seed = self.tracker._copy_seed(seed)
            self.tracker.frames_since_global = 0
            self.tracker.last_debug = self.global_localizer.last_debug
            self.active_reference_name = self.global_localizer.last_reference_name
            self.pending_seed = None
            self.pending_reference_name = None

    def localize(self, image, camera_matrix, distortion):
        self._update_coarse(image)
        self._consume_completed()
        tracked = None
        if (self.pending_seed is not None and self.tracker.seed is not None and
            self.last_search_age_ms <= self.max_seed_age*1000):
            old_seed = self.tracker.seed
            old_reference = self.active_reference_name
            self.tracker.seed = self.pending_seed
            tracked = self.tracker._track(image, camera_matrix, distortion)
            if tracked is not None:
                self.active_reference_name = self.pending_reference_name
            else:
                self.tracker.seed = old_seed
                self.active_reference_name = old_reference
            self.pending_seed = None
            self.pending_reference_name = None
        if self.tracker.seed is not None:
            if tracked is None:
                tracked = self.tracker._track(image, camera_matrix, distortion)
            if tracked is not None:
                self.pending_seed = None
                self.pending_reference_name = None
                self.tracker.frames_since_global += 1
                self.last_debug = self.tracker.last_debug
                if (
                    self.future is None
                    and self.tracker.frames_since_global >= self.reseed_frames
                ):
                    self._submit(image, camera_matrix, distortion)
                    self.tracker.frames_since_global = 0
                return tracked
            self.tracker.seed = None
            self.active_reference_name = None
        if self.pending_seed is not None:
            self.tracker.seed = self.pending_seed
            self.active_reference_name = self.pending_reference_name
            self.pending_seed = None
            self.pending_reference_name = None
            tracked = self.tracker._track(image, camera_matrix, distortion)
            if tracked is not None:
                self.tracker.frames_since_global = 0
                self.last_debug = self.tracker.last_debug
                return tracked
            self.tracker.seed = None
            self.active_reference_name = None
        if self.future is None:
            self._submit(image, camera_matrix, distortion)
        return self._search_display(image)

    def _search_display(self, image):
        elapsed = max(0.0, time.monotonic() - self.search_started)
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"SEARCH  {self.source_label} ALIKED+LightGlue  {elapsed:.1f} s",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )
        previous = "none"
        if self.last_search_result is not None:
            diagnostics = self.last_search_diagnostics
            quality = diagnostics.get("quality") or {}
            reasons = quality.get("reasons") or []
            if diagnostics.get("ambiguous_hypotheses"):
                reasons = ["ambiguous_pose"]
            if not reasons and diagnostics.get("reason"):
                reasons = [diagnostics["reason"]]
            failure = f" ({','.join(reasons)})" if reasons else ""
            previous = (
                f"{self.last_search_result.status}{failure}, "
                f"{self.last_search_result.inliers}/{self.last_search_result.matches} PnP inliers"
            )
        cv2.putText(
            canvas,
            f"Camera live; previous {self.source_label}: {previous}; search {self.last_search_age_ms:.0f}ms",
            (14, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        result = LiveResult(
            "FAIL",
            "REFERENCE_SEARCH_RUNNING",
            0,
            0,
            0,
            0.0,
            None,
            None,
            None,
            None,
            elapsed * 1000.0,
            "HIDDEN",
        )
        return result, canvas

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.coarse_executor.shutdown(wait=False, cancel_futures=True)


class GLBVirtualLiveLocalizer:
    """Low-frequency learned relocalizer that seeds the existing LK tracker."""

    def __init__(
        self,
        views_path: Path,
        *,
        top_k: int = 5,
        sift_ratio: float = 0.78,
        max_keypoints: int = 2048,
        resize: int = 1024,
        match_threshold: float = 0.01,
        fundamental_threshold: float = 3.0,
        xyz_search_radius: int = 2,
        ransac_threshold: float = 6.0,
        confidence: float = 0.999,
        iterations: int = 10_000,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.2,
        max_reprojection_error: float = 6.0,
        device: str = "auto",
        mpr_renderer=None,
        pipeline_policy: str = "validated",
        diagnostics_path: Optional[Path] = None,
        fast_consensus: bool = False,
        appearance_audit: bool = False,
        max_coarse_rotation_deg: float = 35.0,
        retrieval_mode: str = "sift",
        retrieval_crop: bool = False,
        retrieval_roi=DEFAULT_ROI,
        retrieval_auto_roi: bool = False,
    ) -> None:
        if top_k <= 0 or not 0 < sift_ratio < 1 or max_keypoints <= 0:
            raise LivePnPError("Invalid GLB virtual-view matching parameters")
        self.views_path = Path(views_path)
        try:
            self.manifest, self.views = load_manifest(self.views_path)
        except RuntimeError as exc:
            raise LivePnPError(str(exc)) from exc
        self.top_k = min(top_k, len(self.views))
        if pipeline_policy not in {'validated', 'legacy'}:
            raise LivePnPError('Unknown virtual-view pipeline policy')
        self.pipeline_policy = pipeline_policy
        self.diagnostics_path = diagnostics_path
        self.fast_consensus = fast_consensus
        self.appearance_audit = appearance_audit
        if not math.isfinite(max_coarse_rotation_deg) or not 0 < max_coarse_rotation_deg <= 180:
            raise LivePnPError('Invalid coarse-pose rotation threshold')
        self.max_coarse_rotation_deg = max_coarse_rotation_deg
        self.last_diagnostics = {}
        self.geometry_cache = {}
        self.source_type = str(self.manifest.get("source_type", "glb")).lower()
        self.source_label = "3DGS" if self.source_type == "3dgs" else "GLB"
        self.runtime_transform = str(
            self.manifest.get(
                "runtime_transform",
                "identity" if self.source_type == "3dgs" else "rc_to_colmap",
            )
        ).lower()
        if self.runtime_transform not in {"identity", "rc_to_colmap"}:
            raise LivePnPError(
                f"Unsupported virtual-view runtime transform: {self.runtime_transform}"
            )
        self.sift_ratio = sift_ratio
        self.retrieval_mode = retrieval_mode
        self.retrieval_crop = retrieval_crop or retrieval_auto_roi
        self.retrieval_auto_roi = retrieval_auto_roi
        self.retrieval_roi = tuple(retrieval_roi)
        self.excluded_retrieval_views = set()
        self.resize = resize
        self.fundamental_threshold = fundamental_threshold
        self.xyz_search_radius = xyz_search_radius
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.iterations = iterations
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reprojection_error = max_reprojection_error
        self.mpr_renderer = mpr_renderer
        try:
            self.retriever = ViewRetrievalIndex(
                self.views_path, self.views, mode=retrieval_mode,
                crop=self.retrieval_crop, roi=retrieval_roi, ratio=sift_ratio,
                device=device, auto_roi=retrieval_auto_roi)
        except (RuntimeError, ValueError, OSError) as exc:
            raise LivePnPError(f"Virtual-view retrieval setup failed: {exc}") from exc
        # Retained for old diagnostics and callers that inspect the SIFT index.
        self.reference_descriptors = (self.retriever.reference
                                      if retrieval_mode == "sift" else [])
        try:
            self.torch, self.extractor, self.matcher, self.device = _load_models(
                max_keypoints, match_threshold, device
            )
        except RuntimeError as exc:
            raise LivePnPError(str(exc)) from exc
        self.reference_feature_cache = {}
        self.cache_directory = self.views_path / '.aliked_cache'
        # Content-addressed, no pickle: image bytes + actual extractor weights
        # + preprocessing configuration prevent stale reference descriptors.
        digest = hashlib.sha256(f'v1:{resize}:{max_keypoints}:{self.extractor.conf}'.encode())
        for name, tensor in self.extractor.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        self.extractor_fingerprint = digest.hexdigest()
        self.xyz_cache = {}
        self.lock = threading.Lock()
        self.map_data = self._build_viewer_map()
        self.last_debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )
        self.last_tracking_seed: Optional[TrackingSeed] = None
        self.last_crop_debug = {}
        self.last_reference_name = "—"
        self.last_candidate_log = "none"
        print(
            f"{self.source_label} virtual-view map: {len(self.views)} views, "
            f"device={self.device}, top-k={self.top_k}, "
            f"retrieval={self.retrieval_mode}, crop={self.retrieval_crop}, "
            f"auto_roi={self.retrieval_auto_roi}, "
            f"transform={self.runtime_transform}, "
            f"source={self.manifest.get('ply', self.manifest.get('glb'))}"
        )
        print(f"Virtual pipeline: {self.pipeline_policy}; geometry="
              f"{self.manifest.get('geometry_renderer', 'legacy-cpu-centers')}; "
              f"coarse rotation <= {self.max_coarse_rotation_deg:g} deg; "
              "image-space QC only (CT/physical accuracy not certified)")

    def _build_viewer_map(self) -> MapData:
        first = self.views[0]
        try:
            with np.load(
                self.views_path / str(first["data_name"]), allow_pickle=False
            ) as data:
                xyz = data["xyz"][::4, ::4].reshape((-1, 3))
        except (OSError, ValueError, KeyError) as exc:
            raise LivePnPError(f"Could not load virtual-view XYZ: {exc}") from exc
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) == 0:
            raise LivePnPError("Virtual-view XYZ buffer contains no surface points")
        xyz = self._source_xyz_to_runtime(xyz).astype(np.float64)
        count = len(xyz)
        return MapData(
            np.arange(count, dtype=np.int64),
            xyz,
            np.zeros((count, 128), dtype=np.float32),
            np.zeros(count, dtype=np.int64),
        )

    def _rank_views(self, image: np.ndarray):
        ranked, count, all_ranked, self.last_crop_debug = self.retriever.rank_debug(
            image, top_k=self.top_k, excluded=self.excluded_retrieval_views,
            include_zero=self.pipeline_policy == 'legacy')
        self.last_retrieval = [dict(
            image_name=self.views[index]['image_name'],
            score=score, mode=self.retrieval_mode,
            ratio_matches=int(score) if self.retrieval_mode == 'sift' else None,
            mean_distance=(-tie if math.isfinite(tie) else None)
            if self.retrieval_mode == 'sift' else None)
            for score, tie, index in all_ranked]
        return ranked, count

    def _reference_features(self, view_index: int):
        cached = self.reference_feature_cache.get(view_index)
        if cached is not None:
            return cached
        view = self.views[view_index]
        image_path = self.views_path / str(view['image_name'])
        cache_key = hashlib.sha256(image_path.read_bytes() + self.extractor_fingerprint.encode()).hexdigest()
        cache_path = self.cache_directory / f'{cache_key}.npz'
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    features = {key: self.torch.from_numpy(archive[key].copy()).to(self.device)
                                for key in archive.files}
                if not {'keypoints', 'descriptors', 'image_size'} <= features.keys():
                    raise ValueError('Incomplete reference cache')
                self.reference_feature_cache[view_index] = features
                return features
            except (OSError, ValueError):
                pass  # Re-extract an interrupted/corrupt cache; never trust pickle.
        image = cv2.imread(
            str(self.views_path / str(view["image_name"])), cv2.IMREAD_COLOR
        )
        if image is None:
            raise LivePnPError(f"Virtual RGB missing: {view['image_name']}")
        with self.torch.inference_mode():
            features = _extract(
                self.extractor,
                _image_tensor(self.torch, image, self.device),
                self.resize,
            )
        self.reference_feature_cache[view_index] = features
        try:
            self.cache_directory.mkdir(exist_ok=True)
            temporary = cache_path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as output:
                np.savez_compressed(output, **{key: value.detach().cpu().numpy()
                                              for key, value in features.items()})
            os.replace(temporary, cache_path)
        except OSError:
            pass  # Read-only reference directories still work with RAM caching.
        return features

    def _xyz(self, view_index: int):
        cached = self.xyz_cache.get(view_index)
        if cached is not None:
            return cached
        view = self.views[view_index]
        try:
            with np.load(
                self.views_path / str(view["data_name"]), allow_pickle=False
            ) as data:
                xyz = data["xyz"].astype(np.float32, copy=True)
                if (self.pipeline_policy != 'legacy' and
                    self.manifest.get('geometry_renderer') in
                    {'shared-gaussian-moments', 'ray-consistent-legacy-depth'}):
                    self.geometry_cache[view_index] = {
                        name: data[name].copy() for name in
                        ('depth', 'valid', 'camera_matrix', 'camera_to_world')}
        except (OSError, ValueError, KeyError) as exc:
            raise LivePnPError(
                f"Virtual XYZ could not be read: {view['data_name']}: {exc}"
            ) from exc
        self.xyz_cache[view_index] = xyz
        return xyz

    def validate_pose(self, object_points, image_points, rvec, tvec, indices,
                      image_shape, camera_matrix, distortion):
        return assess_pose(object_points, image_points, rvec, tvec, indices,
            image_shape, camera_matrix, distortion, self.min_inliers,
            self.min_inlier_ratio, self.max_reprojection_error)

    def _pose_quality(self, pose, transfers, image, camera_matrix, distortion):
        return self.validate_pose(np.asarray([t.xyz for t in transfers]),
            np.asarray([t.query_uv for t in transfers]), pose.rvec, pose.tvec,
            pose.inlier_indices, image.shape, camera_matrix, distortion)

    @staticmethod
    def _pose_distance(first, second):
        a, b = cv2.Rodrigues(first.rvec)[0], cv2.Rodrigues(second.rvec)[0]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))
        distance = float(np.linalg.norm(-a.T @ first.tvec + b.T @ second.tvec))
        return angle, distance

    def _coarse_pose_check(self, pose, view):
        """Compare a PnP rotation with the virtual camera that supplied XYZ.

        This is a coarse retrieval consistency check, not metric validation.
        Only identity-coordinate 3DGS references have an unambiguous camera
        rotation in the current runtime frame.
        """
        matrix = view.get('coarse_anchor', view).get('world_to_camera')
        if self.source_type != '3dgs' or self.runtime_transform != 'identity' or matrix is None:
            return None
        reference = np.asarray(matrix, dtype=np.float64)
        if reference.shape != (4, 4) or not np.isfinite(reference).all():
            return None
        actual = cv2.Rodrigues(pose.rvec)[0]
        relative = actual @ reference[:3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1.)/2., -1., 1.))))
        return dict(rotation_deg=angle, threshold_deg=self.max_coarse_rotation_deg,
                    passed=angle <= self.max_coarse_rotation_deg)

    def _write_diagnostics(self):
        if self.diagnostics_path is not None:
            path = Path(self.diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as output:
                output.write(json.dumps(self.last_diagnostics, allow_nan=False) + '\n')

    @staticmethod
    def _to_colmap(transfers):
        converted = []
        for item in transfers:
            point = rc_to_colmap(np.asarray(item.xyz, np.float64))
            converted.append(
                VirtualTransfer(
                    item.query_index,
                    tuple(float(value) for value in point),
                    item.query_uv,
                    item.reference_uv,
                    item.match_score,
                )
            )
        return converted

    def _source_xyz_to_runtime(self, points):
        values = np.asarray(points, dtype=np.float64)
        if self.runtime_transform == "identity":
            return values
        return rc_to_colmap(values)

    def _to_runtime(self, transfers):
        if self.runtime_transform == "identity":
            return list(transfers)
        return self._to_colmap(transfers)

    @staticmethod
    def _merge_transfers(transfer_groups):
        """Combine candidates while keeping one 3D hypothesis per live keypoint.

        ALIKED is run once for the live image, so the same keypoint has exactly
        the same floating-point coordinate in every reference match.  Keeping
        only its strongest LightGlue match prevents one image feature from
        voting for several conflicting 3D positions during PnP.
        """
        candidates = [item for group in transfer_groups for item in group]
        candidates.sort(key=lambda item: item.match_score, reverse=True)
        merged = []
        used_query = set()
        used_xyz = set()
        for item in candidates:
            query_key = tuple(round(value, 3) for value in item.query_uv)
            xyz_key = tuple(round(value, 1) for value in item.xyz)
            if query_key in used_query or xyz_key in used_xyz:
                continue
            used_query.add(query_key)
            used_xyz.add(xyz_key)
            merged.append(
                VirtualTransfer(
                    len(merged),
                    item.xyz,
                    item.query_uv,
                    item.reference_uv,
                    item.match_score,
                )
            )
        return merged

    def localize(self, image, camera_matrix, distortion):
        with self.lock:
            return self._localize_locked(image, camera_matrix, distortion)

    def _failure(
        self,
        image,
        started,
        detail,
        *,
        keypoints=0,
        matches=0,
        sift_ms=0.0,
        matching_ms=0.0,
        pnp_ms=0.0,
        matched_xyz=None,
    ):
        if matched_xyz is None:
            matched_xyz = np.empty((0, 3), dtype=np.float64)
        self.last_debug = LocalizationDebug(
            np.asarray(matched_xyz, np.float64).reshape((-1, 3)),
            frozenset(),
            None,
            False,
        )
        self.last_tracking_seed = None
        self.last_diagnostics.update(status='FAIL', reason=detail)
        self.last_diagnostics.setdefault('timings_ms', {}).update(
            matching=matching_ms, pnp=pnp_ms,
            total=(time.perf_counter()-started)*1000)
        self._write_diagnostics()
        result = LiveResult(
            "FAIL",
            detail,
            keypoints,
            matches,
            0,
            0.0,
            None,
            None,
            None,
            None,
            (time.perf_counter() - started) * 1000.0,
            "HIDDEN",
            sift_ms,
            matching_ms,
            pnp_ms,
            0.0,
        )
        return result, self._decorate(image, result, (), frozenset())

    def _localize_locked(self, image, camera_matrix, distortion):
        started = time.perf_counter()
        self.last_reference_name = '—'
        self.last_candidate_log = 'none'
        self.last_retrieval = []
        self.last_diagnostics = dict(policy=self.pipeline_policy,
            geometry=self.manifest.get('geometry_renderer', 'legacy-cpu-centers'),
            candidates=[], timings_ms={}, physical_accuracy_validated=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ranked, sift_keypoints = self._rank_views(image)
        sift_done = time.perf_counter()
        sift_ms = (sift_done - started) * 1000.0
        self.last_diagnostics['retrieval'] = self.last_retrieval
        self.last_diagnostics['retrieval_mode'] = self.retrieval_mode
        self.last_diagnostics['retrieval_crop'] = self.retrieval_crop
        self.last_diagnostics['retrieval_roi'] = self.retrieval_roi
        self.last_diagnostics['retrieval_detected_bbox'] = self.last_crop_debug.get('bbox')
        detection = self.last_crop_debug.get('detection') or {}
        self.last_diagnostics['retrieval_detection'] = {
            key: value for key, value in detection.items() if key != 'mask'}
        self.last_diagnostics['timings_ms']['retrieval'] = sift_ms
        if not ranked:
            return self._failure(
                image, started,
                f"{self.source_label}_NO_OBJECT_ROI" if self.retrieval_auto_roi
                else f"{self.source_label}_NO_REFERENCE", sift_ms=sift_ms
            )
        with self.torch.inference_mode():
            query_features = _extract(
                self.extractor,
                _image_tensor(self.torch, image, self.device),
                self.resize,
            )
        query_feature_count = int(query_features["keypoints"].shape[1])
        self.last_diagnostics['timings_ms']['query_features'] = (time.perf_counter()-sift_done)*1000
        total_pnp_ms = 0.0
        total_quality_ms = 0.0
        hypotheses = []
        transfer_groups = []
        candidate_log = []
        for retrieval_score, _, view_index in ranked:
            candidate_started = time.perf_counter()
            view = self.views[view_index]
            cache_hit = view_index in self.reference_feature_cache
            reference_features = self._reference_features(view_index)
            features_done = time.perf_counter()
            query_kp, reference_kp, pairs, scores = _match_pair(
                self.torch, self.matcher, query_features, reference_features
            )
            match_done = time.perf_counter()
            if len(pairs):
                query_points = query_kp[pairs[:, 0]]
                reference_points = reference_kp[pairs[:, 1]]
                query_points, reference_points, scores = geometric_filter(
                    query_points,
                    reference_points,
                    scores,
                    self.fundamental_threshold,
                )
            else:
                query_points = np.empty((0, 2), np.float32)
                reference_points = np.empty((0, 2), np.float32)
            geometry_done = time.perf_counter()
            rejected_xyz = {}
            transfers = make_transfers(
                query_points,
                reference_points,
                scores,
                self._xyz(view_index),
                self.xyz_search_radius,
                geometry=self.geometry_cache.get(view_index),
                diagnostics=rejected_xyz,
            )
            transfers = self._to_runtime(transfers)
            transfer_groups.append(transfers)
            transfer_done = time.perf_counter()
            pose = solve_pose(
                transfers,
                camera_matrix,
                distortion,
                self.ransac_threshold,
                self.iterations,
                validate=self.pipeline_policy != 'legacy',
            )
            pose_done = time.perf_counter()
            total_pnp_ms += (pose_done-transfer_done)*1000
            quality = None if pose is None else self._pose_quality(pose, transfers, image, camera_matrix, distortion)
            coarse_pose = None if pose is None else self._coarse_pose_check(pose, view)
            quality_ms = (time.perf_counter()-pose_done)*1000
            total_quality_ms += quality_ms
            appearance = None
            appearance_started = time.perf_counter()
            if self.appearance_audit and pose is not None and quality['passed']:
                from pose_appearance import evaluate_pose_appearance
                reference_rgb = cv2.imread(str(self.views_path/view['image_name']))
                # Pose above is in runtime coordinates; convert the full XYZ
                # buffer using the same transform as the transferred points.
                appearance = evaluate_pose_appearance(image, reference_rgb,
                    self._source_xyz_to_runtime(self._xyz(view_index)),
                    pose.rvec, pose.tvec, camera_matrix, distortion,
                    [transfers[i].reference_uv for i in pose.inlier_indices])
            appearance_ms = (time.perf_counter()-appearance_started)*1000
            self.last_diagnostics['candidates'].append(dict(
                image_name=view['image_name'],
                retrieval_score=retrieval_score,
                sift_matches=retrieval_score if self.retrieval_mode == 'sift' else None,
                local_matches=len(pairs), geometric_inliers=len(query_points),
                xyz_correspondences=len(transfers), xyz_rejections=rejected_xyz,
                quality=quality, coarse_pose=coarse_pose,
                reference_cache_hit=cache_hit,
                appearance_audit=appearance,
                pose=None if pose is None else dict(rvec=pose.rvec.ravel().tolist(),
                                                   tvec=pose.tvec.ravel().tolist()),
                timings_ms=dict(reference_features=(features_done-candidate_started)*1000,
                    local_matching=(match_done-features_done)*1000,
                    geometry_filter=(geometry_done-match_done)*1000,
                    xyz=(transfer_done-geometry_done)*1000,
                    pnp=(pose_done-transfer_done)*1000, quality=quality_ms,
                    appearance_audit=appearance_ms)))
            inliers = 0 if pose is None else len(pose.inlier_indices)
            error_text = "-" if pose is None else f"{pose.mean_error:.1f}"
            retrieval_text = (f"S{retrieval_score:.0f}" if self.retrieval_mode == 'sift'
                              else f"D{retrieval_score:.3f}")
            candidate_log.append(
                f"{view['image_name']} {retrieval_text}/L{len(pairs)}/"
                f"G{len(query_points)}/X{len(transfers)}/I{inliers}/E{error_text}"
            )
            if pose is not None:
                hypotheses.append((inliers, len(transfers), pose, transfers, view))

            # Stop only after >=3 candidates and two independently supported,
            # agreeing local poses. A single confident repeated motif is not
            # sufficient. Legacy policy evaluates the whole shortlist.
            if self.fast_consensus and self.pipeline_policy != 'legacy' and len(candidate_log) >= 3:
                strong = [h for h in hypotheses if h[0] >= 40 and h[0]/max(1,h[1]) >= .5
                          and self._pose_quality(h[2], h[3], image, camera_matrix, distortion)['passed']
                          and (self._coarse_pose_check(h[2], h[4]) or {'passed': True})['passed']]
                if len(strong) >= 2:
                    angle, distance = self._pose_distance(strong[-1][2], strong[-2][2])
                    if angle < 3 and distance < 5:
                        self.last_diagnostics['early_stop'] = 'two_agreeing_supported_poses'
                        break

        # Do not reintroduce global ambiguity by pooling incompatible regions.
        # Only merge candidates whose independent PnP estimates agree. Keep
        # multiple clusters as separate hypotheses for the ambiguity gate.
        groups = []
        if self.pipeline_policy == 'legacy':
            groups = [('MERGED_LEGACY', transfer_groups)]
        else:
            eligible = [h for h in hypotheses if self._pose_quality(
                h[2], h[3], image, camera_matrix, distortion)['passed']
                and (self._coarse_pose_check(h[2], h[4]) or {'passed': True})['passed']]
            seen_clusters = set()
            for anchor in eligible:
                cluster = [h for h in eligible if all(x <= limit for x, limit in
                    zip(self._pose_distance(anchor[2], h[2]), (10., 30.)))]
                key = tuple(sorted(h[4]['image_name'] for h in cluster))
                if len(cluster) >= 2 and key not in seen_clusters:
                    seen_clusters.add(key)
                    groups.append((f"MERGED_{len(cluster)}_{anchor[4]['image_name']}",
                                   [h[3] for h in cluster], anchor[4]))
        for entry in groups:
            group_name, group = entry[:2]
            anchor_view = entry[2] if len(entry) > 2 else None
            merged_transfers = self._merge_transfers(group)
            merged_started = time.perf_counter()
            merged_pose = solve_pose(merged_transfers, camera_matrix, distortion,
                self.ransac_threshold, self.iterations, validate=self.pipeline_policy != 'legacy')
            total_pnp_ms += (time.perf_counter()-merged_started)*1000
            merged_inliers = 0 if merged_pose is None else len(merged_pose.inlier_indices)
            merged_error = '-' if merged_pose is None else f'{merged_pose.mean_error:.1f}'
            candidate_log.append(f'{group_name}/X{len(merged_transfers)}/I{merged_inliers}/E{merged_error}')
            if merged_pose is not None:
                hypotheses.append((merged_inliers, len(merged_transfers), merged_pose,
                                   merged_transfers, {'image_name': group_name,
                                                      'coarse_anchor': anchor_view}
                                   if anchor_view is not None else {'image_name': group_name}))
        matching_done = time.perf_counter()
        matching_ms = (matching_done - sift_done) * 1000.0
        matching_ms -= total_pnp_ms
        self.last_candidate_log = ", ".join(candidate_log)
        print(f"{self.source_label} hypotheses: " + self.last_candidate_log)
        if not hypotheses:
            self.last_diagnostics['status'] = 'FAIL_NO_POSE'
            self.last_diagnostics['timings_ms'].update(pnp=total_pnp_ms, matching=matching_ms)
            best_group = max(transfer_groups, key=len, default=[])
            return self._failure(
                image,
                started,
                f"{self.source_label}_PNP_FAILED",
                keypoints=query_feature_count,
                matches=len(best_group),
                sift_ms=sift_ms,
                matching_ms=matching_ms,
                pnp_ms=total_pnp_ms,
                matched_xyz=[item.xyz for item in best_group],
            )
        hypotheses.sort(
            key=lambda item: (
                (self.pipeline_policy == 'legacy' or
                 self._pose_quality(item[2], item[3], image, camera_matrix, distortion)['passed']),
                (self.pipeline_policy == 'legacy' or
                 (self._coarse_pose_check(item[2], item[4]) or {'passed': True})['passed']),
                item[2].mean_error <= self.max_reprojection_error,
                item[0],
                item[0] / max(1, item[1]),
                -item[2].mean_error,
            ),
            reverse=True,
        )
        inlier_count, correspondence_count, pose, transfers, view = hypotheses[0]
        inlier_ratio = inlier_count / max(1, correspondence_count)
        accepted = (
            inlier_count >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
            and pose.mean_error <= self.max_reprojection_error
        )
        quality = self._pose_quality(pose, transfers, image, camera_matrix, distortion)
        coarse_pose = self._coarse_pose_check(pose, view)
        ambiguity = []
        if self.pipeline_policy != 'legacy':
            accepted = accepted and quality['passed']
            if coarse_pose is not None and not coarse_pose['passed']:
                accepted = False
                quality = dict(quality, passed=False,
                               reasons=list(quality['reasons']) + ['coarse_pose_mismatch'])
            # Comparable support for an incompatible pose is not a trustworthy
            # PASS, even if each hypothesis has a small reprojection residual.
            for other in hypotheses[1:]:
                if other[0] < .7 * inlier_count:
                    continue
                if not self._pose_quality(other[2], other[3], image, camera_matrix, distortion)['passed']:
                    continue
                other_coarse = self._coarse_pose_check(other[2], other[4])
                if other_coarse is not None and not other_coarse['passed']:
                    continue
                angle, distance = self._pose_distance(pose, other[2])
                if angle > 15 or distance > 30:
                    ambiguity.append(dict(reference=other[4]['image_name'],
                                          rotation_deg=angle, translation=distance))
            if ambiguity:
                accepted = False
        self.last_diagnostics.update(selected=view['image_name'], quality=quality,
                                     coarse_pose=coarse_pose,
                                     ambiguous_hypotheses=ambiguity,
                                     status='PASS' if accepted else 'FAIL_QC')
        object_points = np.asarray([item.xyz for item in transfers], np.float64)
        image_points = np.asarray([item.query_uv for item in transfers], np.float64)
        rotation, _ = cv2.Rodrigues(pose.rvec)
        camera_center = (-rotation.T @ pose.tvec).reshape(3)
        self.last_debug = LocalizationDebug(
            object_points.copy(),
            frozenset(int(value) for value in pose.inlier_indices),
            tuple(float(value) for value in camera_center),
            True,
        )
        self.last_reference_name = str(view["image_name"])
        if accepted:
            self.last_tracking_seed = TrackingSeed(
                gray.copy(),
                image_points[pose.inlier_indices].copy(),
                object_points[pose.inlier_indices].copy(),
                pose.rvec.copy(),
                pose.tvec.copy(),
            )
        else:
            self.last_tracking_seed = None
        pnp_done = time.perf_counter()
        pnp_ms = total_pnp_ms + (pnp_done - matching_done) * 1000.0
        self.last_diagnostics['timings_ms'].update(matching=matching_ms, pnp=pnp_ms,
            pnp_solve=total_pnp_ms,
            quality=total_quality_ms + (pnp_done-matching_done)*1000,
            total=(pnp_done-started)*1000)
        self._write_diagnostics()
        canvas = image.copy()
        mpr_status = "HIDDEN"
        mpr_started = time.perf_counter()
        if accepted and self.mpr_renderer is not None:
            canvas, mpr_status = self.mpr_renderer.render(
                canvas,
                camera_matrix,
                distortion,
                pose.rvec,
                pose.tvec,
            )
        mpr_ms = (time.perf_counter() - mpr_started) * 1000.0
        result = LiveResult(
            "PASS" if accepted else "FAIL",
            f"{self.source_label} REF {view['image_name']}" +
            (' AMBIGUOUS' if ambiguity else '') +
            (' COARSE_MISMATCH' if coarse_pose is not None and not coarse_pose['passed'] else ''),
            query_feature_count,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            pose.mean_error,
            tuple(float(value) for value in pose.rvec.ravel()),
            tuple(float(value) for value in pose.tvec.ravel()),
            tuple(float(value) for value in camera_center),
            (time.perf_counter() - started) * 1000.0,
            mpr_status,
            sift_ms,
            matching_ms,
            pnp_ms,
            mpr_ms,
        )
        return result, self._decorate(
            canvas,
            result,
            transfers,
            frozenset(int(value) for value in pose.inlier_indices),
        )

    def _decorate(self, image, result, transfers, inliers):
        canvas = image.copy()
        for index, transfer in enumerate(transfers):
            u, v = transfer.query_uv
            color = (40, 230, 70) if index in inliers else (40, 40, 230)
            cv2.circle(
                canvas, (round(u), round(v)), 3, color, -1, cv2.LINE_AA
            )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        color = (80, 230, 80) if result.status == "PASS" else (70, 70, 240)
        cv2.putText(
            canvas,
            f"{result.status}  {self.source_label} ALIKED+LightGlue  "
            f"matches {result.matches}  "
            f"inliers {result.inliers}  MPR {result.mpr_status}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{result.detail}  error "
            f"{result.mean_reprojection_error or 0.0:.2f}px  "
            f"{result.processing_ms / 1000.0:.1f}s",
            (14, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return canvas
