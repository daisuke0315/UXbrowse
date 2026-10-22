#!/usr/bin/env python3
"""Live virtual-view localization for GLB or 3DGS reference maps."""

from __future__ import annotations

import math
import copy
import concurrent.futures
import threading
import time
import json
import hashlib
import os
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
    sample_xyz_buffer,
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
from view_retrieval import object_bbox
from object_roi import reference_rgb_mask
from concurrent_query_tracking import SearchPointTracks, publish_query_points
from live_search_resources import run_live_search
from guided_matching import spatial_indices, subset_features, project_candidates, pair_gate
from tracking_handoff import TrackingHandoff
from aliked_query import ALIKEDQuery
from search_frame_quality import RecentSearchFrames, image_quality, correct_luminance, PHOTOMETRIC_MODE


def exclude_virtual_query_polygon(pairs, scores, query_keypoints, offset, polygon):
    """Reject live matches on a projected model face in full-image UV space."""
    if polygon is None or not len(pairs):
        return pairs, scores, 0
    shape = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    if len(shape) < 3 or not np.isfinite(shape).all():
        raise ValueError('Invalid excluded face polygon')
    query_uv = query_keypoints[pairs[:, 0]] + np.asarray(offset, np.float32)
    keep = np.asarray([cv2.pointPolygonTest(shape, (float(uv[0]), float(uv[1])), False) < 0
                       for uv in query_uv], bool)
    return pairs[keep], scores[keep], int((~keep).sum())


def crop_live_object(image: np.ndarray, bbox, padding: float = .04):
    """Crop a normalized object bbox at the original matching resolution."""
    if bbox is None:
        return None
    x, y, width, height = np.asarray(bbox, np.float64)
    if (not np.isfinite((x, y, width, height)).all() or width <= 0 or
        height <= 0 or padding < 0):
        return None
    rows, columns = image.shape[:2]
    x0 = max(0, int(math.floor((x-padding*width)*columns)))
    y0 = max(0, int(math.floor((y-padding*height)*rows)))
    x1 = min(columns, int(math.ceil((x+(1.+padding)*width)*columns)))
    y1 = min(rows, int(math.ceil((y+(1.+padding)*height)*rows)))
    if x1-x0 < 32 or y1-y0 < 32:
        return None
    return np.ascontiguousarray(image[y0:y1, x0:x1]), (x0, y0, x1, y1)


OBJECT_GEOMETRY_SIDE = 1024.0


def object_scale_geometric_filter(query_points, reference_points, scores,
                                  threshold, query_shape, reference_shape):
    """Apply a pixel threshold in a common object-crop coordinate system.

    Feature coordinates are left in their original crop spaces for subsequent
    offset restoration. Resizing either source image cannot silently change
    the effective Fundamental-matrix tolerance.
    """
    query_scale = OBJECT_GEOMETRY_SIDE / max(query_shape[:2])
    reference_scale = OBJECT_GEOMETRY_SIDE / max(reference_shape[:2])
    normalized_query, normalized_reference, selected_scores = geometric_filter(
        np.asarray(query_points) * query_scale,
        np.asarray(reference_points) * reference_scale,
        scores, threshold)
    return (normalized_query / query_scale,
            normalized_reference / reference_scale, selected_scores)


def has_spread_triangle(points, min_distance_mm=30., min_height_mm=15.):
    """Existence of three finite XYZs with separated edges and non-thin triangle.

    Height is measured against the longest edge (the smallest altitude).
    Coordinates must be in millimeters. This is a replenishment trigger only,
    not a replacement for PnP's inlier or pose-quality acceptance checks.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return False
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    along = np.einsum('ij,j->i', centered, axes[0])
    perpendicular = centered - np.outer(along, axes[0])
    # A narrow cylinder bounds every triangle's smallest altitude. Reject
    # near-collinear sets early rather than enumerating all their triples.
    if 2. * np.max(np.linalg.norm(perpendicular, axis=1)) < min_height_mm:
        return False
    separation2 = min_distance_mm ** 2
    for i in range(len(points) - 2):
        vectors = points[i+1:] - points[i]
        distances2 = np.einsum('ij,ij->i', vectors, vectors)
        for j in np.flatnonzero(distances2 >= separation2):
            if j >= len(vectors) - 1:
                continue
            remaining = vectors[j+1:]
            other2 = distances2[j+1:]
            third2 = np.sum((remaining - vectors[j]) ** 2, axis=1)
            valid = (other2 >= separation2) & (third2 >= separation2)
            if not valid.any():
                continue
            longest2 = np.maximum(np.maximum(distances2[j], other2[valid]), third2[valid])
            area2 = np.sum(np.cross(vectors[j], remaining[valid]) ** 2, axis=1)
            if np.any(area2 >= min_height_mm ** 2 * longest2):
                return True
    return False


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
        replenish_frames: int = 0,
        replenish_mode: str = 'interval',
        replenish_distance_mm: float = 30.,
        replenish_height_mm: float = 15.,
        replenish_cooldown: float = 1.,
        replenish_image_coverage: float = .02,
        replenish_angle_deg: float = 12.,
        orbit_recovery: bool = False,
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
        self.replenish_frames = replenish_frames
        self.frames_since_replenish = 0
        self.last_replenished = 0
        self.replenish_mode = replenish_mode
        self.replenish_distance_mm = replenish_distance_mm
        self.replenish_height_mm = replenish_height_mm
        self.replenish_cooldown = replenish_cooldown
        self.replenish_image_coverage = replenish_image_coverage
        self.replenish_angle_deg = replenish_angle_deg
        self.last_replenish_attempt = -math.inf
        self.last_coverage_ok = None
        self.search_mode = 'global'
        self.last_search_age_ms = 0.0
        self.pending_seed = None
        self.pending_mode = None
        self.pending_reference_name = None
        self.active_reference_name = None
        self.coarse_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='glb-coarse-view')
        self.coarse_future = None
        self.coarse_submitted = 0.0
        self.coarse_started = 0.0
        self.coarse_reference_name = None
        self.coarse_score = 0
        self.coarse_margin = 0.0
        self.coarse_age_ms = 0.0
        self.coarse_crop_debug = {}
        self.search_frames = deque(maxlen=48)
        self.query_tracks = SearchPointTracks(flow_window=flow_window,
            flow_levels=flow_levels, fb_error=forward_backward_error)
        self.global_localizer.query_tracking_callback = self.query_tracks.publish
        self.pending_seed_stamp = None
        self.replenish_anchor = None
        self.last_replenish_reasons = []
        self.viewer_coarse_enabled = replenish_mode != 'background'
        self.last_track_failure = None
        from background_refresh import BackgroundRefresh
        self.background_refresh = BackgroundRefresh(cooldown=replenish_cooldown)
        self.last_refresh_verified = False
        self.handoff = TrackingHandoff()
        self.search_frame_buffer = (RecentSearchFrames()
            if getattr(global_localizer, 'source_type', '') == '3dgs'
            and getattr(global_localizer, 'runtime_transform', '') == 'identity'
            and getattr(global_localizer, 'pipeline_policy', '') != 'legacy' else None)
        self.last_capture_image = None
        self.search_frame_selection = {}
        from tracking_recovery import ShortTrackRecovery
        self.short_recovery = (ShortTrackRecovery() if self.search_frame_buffer is not None
                               and replenish_mode == 'background' and orbit_recovery else None)
        self.background_refresh.fast_recovery = self.short_recovery is not None
        self.refine_refresh_pose = self.short_recovery is not None
        self.resilient_transport = self.short_recovery is not None
        self.last_refresh_refinement = {}

    def record_capture(self, image, matrix, distortion, stamp, frame_id=None):
        buffer = getattr(self, 'search_frame_buffer', None)
        if buffer is not None:
            buffer.add(image, matrix, distortion, stamp, frame_id)
            self.last_capture_image = image

    def search_progress(self):
        """A non-blocking worker snapshot; not a pose for the current image."""
        active = self.future is not None
        reader = getattr(self.global_localizer, 'progress_snapshot', None)
        progress = reader() if reader is not None else {}
        if active and progress.get('started_at', 0.) < self.search_started:
            progress = dict(stage='queued')
        return dict(progress, active=active,
                    mode=getattr(self, 'search_mode', 'global'),
                    elapsed_ms=max(0., (time.monotonic()-self.search_started)*1000.) if active else 0.,
                    tracking=self.tracker.seed is not None,
                    query_tracking=(self.query_tracks.snapshot()
                        if hasattr(self, 'query_tracks') else {}),
                    lost_reason=getattr(self, 'last_track_failure', None),
                    device=str(getattr(self.global_localizer, 'device', 'unknown')),
                    last_search_ms=self.last_search_age_ms,
                    seed_rejected=getattr(self, 'last_search_diagnostics', {}).get('seed_rejected'),
                    seed_transport=getattr(self, 'last_search_diagnostics', {}).get('seed_transport'),
                    background=(self.background_refresh.snapshot()
                                if getattr(self, 'replenish_mode', '') == 'background' else None),
                    handoff=getattr(getattr(self, 'handoff', None), 'info', {}),
                    viewpoint_refresh=getattr(self, 'last_viewpoint_refresh', {}),
                    image_selection=getattr(self, 'search_frame_selection', {}),
                    recovery=(self.short_recovery.snapshot(time.monotonic())
                              if getattr(self, 'short_recovery', None) is not None else {}),
                    refinement=getattr(self, 'last_refresh_refinement', {}),
                    last_result=self.last_search_result.as_dict() if self.last_search_result else None)

    @property
    def source_label(self) -> str:
        return self.global_localizer.source_label

    def _diagnostic(self, event, **values):
        sink = getattr(self, 'diagnostic_sink', None)
        if sink is not None:
            if hasattr(values.get('result'), 'as_dict'):
                values['result'] = values['result'].as_dict()
            sink(event, search_id=getattr(self, 'search_started', None),
                 search_mode=getattr(self, 'search_mode', None),
                 search_frame_id=getattr(self, 'search_frame_id', None),
                 search_capture_monotonic=getattr(self, 'search_source_stamp', None), **values)

    def _submit(self, image, camera_matrix, distortion, pose_prior=None, recovery_hint=None) -> None:
        if self.future is not None:
            return  # Never build an inference queue behind an existing job.
        now = time.monotonic()
        from background_refresh import VIEW_REFRESH_REASONS
        triggers = list(getattr(self, 'last_replenish_reasons', []))
        new_view = (pose_prior is not None and getattr(self, 'replenish_mode', '') == 'background'
                    and bool(VIEW_REFRESH_REASONS.intersection(triggers)))
        buffer = getattr(self, 'search_frame_buffer', None)
        selected = buffer.select(now, require_pose=pose_prior is not None,
                                 max_age=.1 if new_view else None) if buffer is not None else None
        if buffer is not None and selected is None:
            self.search_frame_selection = dict(waiting=True, reason='no_recent_sharp_object_frame',
                                               max_age_ms=100 if new_view else 300)
            if getattr(self, 'replenish_mode', '') == 'background':
                self.background_refresh.state = 'image_wait'
            return
        self.search_started = time.monotonic()
        self.search_frames.clear()
        if hasattr(self, 'query_tracks'):
            self.query_tracks.begin(self.search_started)
        self.search_source_stamp = selected.stamp if selected is not None else self.search_started
        self.search_frame_image = selected.image.copy() if selected is not None else image.copy()
        self.search_frame_id = selected.frame_id if selected is not None else None
        if selected is not None:
            image, camera_matrix, distortion = selected.image, selected.matrix, selected.distortion
            if pose_prior is not None:
                pose_prior = SimpleNamespace(rvec=selected.pose[0], tvec=selected.pose[1])
            self.search_frame_selection = dict(waiting=False, frame_id=selected.frame_id,
                age_ms=(self.search_started-selected.stamp)*1000,
                roi_known=selected.roi is not None, quality=selected.quality)
            # Include the PRE-submission history when an older source is chosen.
            for frame in buffer.frames:
                if frame.stamp > selected.stamp:
                    self.search_frames.append(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY))
        self.last_replenish_attempt = self.search_started
        self.search_mode = ('nearby' if recovery_hint is not None else
                            'guided' if pose_prior is not None else 'global')
        self.global_localizer.diagnostic_context = dict(search_id=self.search_started,
            search_mode=self.search_mode, search_frame_id=self.search_frame_id,
            search_capture_monotonic=self.search_source_stamp, trigger=triggers)
        self._diagnostic('search_submitted', trigger=getattr(self, 'last_replenish_reasons', []),
                         viewpoint=getattr(self, 'last_viewpoint_refresh', {}),
                         image_selection=getattr(self, 'search_frame_selection', {}),
                         pose_prior=pose_prior is not None)
        if getattr(self, 'replenish_mode', '') == 'background':
            self.background_refresh.submitted(self.search_mode, self.search_started)
        if recovery_hint is not None:
            self.future = self.executor.submit(run_live_search, self.global_localizer, self.global_localizer.localize_nearby,
                image.copy(), camera_matrix.copy(), None if distortion is None else distortion.copy(),
                *(value.copy() for value in recovery_hint))
            self.short_recovery.nearby_used = True
        elif pose_prior is None:
            self.future = self.executor.submit(run_live_search, self.global_localizer, self.global_localizer.localize,
                image.copy(), camera_matrix.copy(), None if distortion is None else distortion.copy())
        else:
            prediction = getattr(getattr(self, 'handoff', None), 'prediction', None)
            if selected is not None and selected is not buffer.frames[-1]:
                prediction = None  # Never combine an old source pose with a newer motion hint.
            extra = (dict(view_prediction=tuple(value.copy() for value in prediction))
                     if prediction is not None and self.replenish_mode == 'background' else {})
            self.future = self.executor.submit(run_live_search, self.global_localizer, self.global_localizer.localize_guided,
                image.copy(), camera_matrix.copy(), None if distortion is None else distortion.copy(),
                pose_prior.rvec.copy(), pose_prior.tvec.copy(), **extra)

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
            completed = self.coarse_future.result()
            # A viewer request predating a full search must not overwrite that
            # search's newer crop/view preview when it eventually completes.
            if self.coarse_started >= self.search_started:
                (self.coarse_reference_name, self.coarse_score,
                 self.coarse_margin, self.coarse_crop_debug) = completed
                self.coarse_age_ms = (now - self.coarse_started) * 1000.
            self.coarse_future = None
        # Localization already produces a retrieval/crop preview. Do not run
        # a second DINO inference for the viewer while that search is active.
        if (self.coarse_future is None and self.future is None and
            getattr(self, 'viewer_coarse_enabled', True) and
            self.tracker.seed is not None and now-self.coarse_submitted >= 1.0):
            self.coarse_submitted = now
            self.coarse_started = now
            self.coarse_future = self.coarse_executor.submit(self._coarse_search, image.copy())

    def _consume_completed(self, image=None, camera_matrix=None, distortion=None) -> None:
        if self.future is None or not self.future.done():
            return
        if getattr(self, 'discard_search_result', False):
            # A running inference cannot be cancelled; never reseed from it
            # after the user requested an independent fresh search.
            self.future = None
            self.discard_search_result = False
            self.search_frames.clear()
            if hasattr(self, 'query_tracks'):
                self.query_tracks.begin(None)
            return
        worker_failed = False
        try:
            self.last_search_result, self.last_search_display = self.future.result()
        except Exception as exc:
            if getattr(self, 'replenish_mode', '') != 'background':
                raise
            worker_failed = True
            self.last_search_result = LiveResult('FAIL', f'BACKGROUND_SEARCH_ERROR: {exc}',
                0, 0, 0, 0., None, None, None, None,
                (time.monotonic()-self.search_started)*1000, 'HIDDEN')
            self.last_search_display = None
        self.last_debug = self.global_localizer.last_debug
        self.last_search_diagnostics = getattr(self.global_localizer, 'last_diagnostics', {})
        completed_mode = getattr(self, 'search_mode', 'global')
        self.future = None
        self.last_replenish_attempt = time.monotonic()
        self.last_search_age_ms = (time.monotonic()-getattr(self, 'search_source_stamp', self.search_started))*1000
        seed = None if worker_failed else self.global_localizer.last_tracking_seed
        tracks = getattr(self, 'query_tracks', None)
        if tracks is not None and not tracks.initialized and image is not None:
            # A fast worker can publish and finish between foreground advance
            # and consumption. Drain its snapshot against the latest image too.
            history = list(getattr(self, 'search_frames', ()))
            tracks.advance(history or [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], time.monotonic())
        continuous = tracks is not None and tracks.initialized
        verified_current = False
        background = getattr(self, 'replenish_mode', '') == 'background'
        if background:
            self.background_refresh.completed(time.monotonic(), (time.monotonic()-self.search_started)*1000)
        if ((background and not continuous and self.last_search_age_ms > self.max_seed_age*1000) or
            getattr(self.last_search_result, 'status', 'PASS') != 'PASS'):
            seed = None
            self.last_search_diagnostics['seed_rejected'] = 'stale_or_failed'
            self.last_search_diagnostics['seed_rejection_detail'] = (
                'search_failed' if getattr(self.last_search_result, 'status', 'PASS') != 'PASS'
                else 'age_exceeded_before_transport')
        if background and completed_mode == 'guided' and self.tracker.seed is None:
            seed = None
            self.last_search_diagnostics['seed_rejected'] = 'guided_prior_lost'
        if completed_mode == 'global':
            retrieval = getattr(self.global_localizer, 'last_retrieval', [])
            if retrieval:
                self.coarse_reference_name = retrieval[0]['image_name']
                self.coarse_score = retrieval[0]['score']
                runner_up = retrieval[1]['score'] if len(retrieval) > 1 else 0.
                self.coarse_margin = self.coarse_score/runner_up if runner_up > 0 else 0.
                self.coarse_crop_debug = getattr(self.global_localizer, 'last_crop_debug', {})
                self.coarse_age_ms = self.last_search_age_ms
                self.coarse_submitted = time.monotonic()
        if seed is not None and continuous:
            seed, detail = tracks.attach(seed)
            self.last_search_diagnostics['seed_transport'] = detail
            if seed is not None:
                seed = self._validate_current_query_seed(seed, image, camera_matrix, distortion, detail)
                verified_current = seed is not None
            if seed is None:
                self.last_search_diagnostics['seed_rejected'] = 'continuous_transport_failed'
                self.last_search_diagnostics['seed_rejection_detail'] = detail['reason']
        elif seed is not None and getattr(self, 'search_frames', None):
            seed = self._advance_search_seed(seed)
            if seed is None:
                self.last_search_diagnostics['seed_rejection_detail'] = 'flow_transport_failed'
            self.search_frames.clear()
            self.last_search_age_ms = (time.monotonic()-getattr(self, 'search_source_stamp', self.search_started))*1000
            if background and self.last_search_age_ms > self.max_seed_age*1000:
                seed = None
                self.last_search_diagnostics['seed_rejected'] = 'stale_after_transport'
        if tracks is not None:
            tracks.begin(None)
        if hasattr(self, 'search_frames'):
            self.search_frames.clear()
        if background and seed is None:
            self.background_refresh.rejected('search_or_transport_rejected')
        self._diagnostic('search_completed', result=self.last_search_result,
            age_ms=self.last_search_age_ms, seed_available=seed is not None,
            diagnostics=self.last_search_diagnostics)
        if seed is not None and self.tracker.seed is not None:
            # Never overwrite a working track with a search-frame seed before
            # validating that seed against the *current* frame.
            self.pending_seed = self.tracker._copy_seed(seed) if seed is not None else None
            self.pending_mode = completed_mode
            self.pending_reference_name = (self.global_localizer.last_reference_name
                                           if seed is not None else None)
            self.pending_seed_stamp = time.monotonic() if verified_current else None
            seed = None
        if seed is not None:
            self.tracker.seed = self.tracker._copy_seed(seed)
            self.tracker.frames_since_global = 0
            self.tracker.last_debug = self.global_localizer.last_debug
            self.active_reference_name = self.global_localizer.last_reference_name
            self.pending_seed = None
            self.pending_mode = None
            self.pending_reference_name = None
            # Establish the baseline only after foreground current-frame QC.
            self.replenish_anchor = None
            if background:
                self.awaiting_global_confirmation = True

    def _validate_current_query_seed(self, seed, image, camera_matrix, distortion, detail):
        """Solve independently on current observations, never display the old pose."""
        if image is None or camera_matrix is None:
            detail['reason'] = 'no_current_image'
            return None
        if not np.array_equal(seed.gray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):
            detail['reason'] = 'latest_frame_not_reached'
            return None
        started = time.perf_counter()
        transfers = [SimpleNamespace(xyz=xyz, query_uv=uv)
                     for xyz, uv in zip(seed.object_points, seed.image_points)]
        pose = solve_pose(transfers, camera_matrix, distortion,
            self.global_localizer.ransac_threshold, self.tracker.tracking_iterations)
        detail['pnp_ms'] = (time.perf_counter()-started)*1000
        if pose is None:
            detail['reason'] = 'current_pnp_failed'
            return None
        quality = self.global_localizer.validate_pose(seed.object_points, seed.image_points,
            pose.rvec, pose.tvec, pose.inlier_indices, image.shape, camera_matrix, distortion)
        detail['quality'] = quality
        if not quality['passed']:
            detail['reason'] = 'current_qc_failed'
            return None
        detail['current_inliers'] = len(pose.inlier_indices)
        detail['reason'] = 'current_pnp_verified'
        return TrackingSeed(seed.gray, seed.image_points[pose.inlier_indices],
            seed.object_points[pose.inlier_indices], pose.rvec, pose.tvec)

    def _pending_seed_age_ms(self):
        stamp = getattr(self, 'pending_seed_stamp', None)
        return ((time.monotonic()-stamp)*1000 if stamp is not None
                else self.last_search_age_ms)

    def _advance_search_seed(self, seed):
        """Transport a search-frame seed in bounded, forward/backward LK hops.

        Pose acceptance still happens on the current frame through PnP/QC.
        The resilient path skips bad intermediate hops, but must reach the
        latest image. At most eight hops bound foreground work.
        """
        frames = list(self.search_frames)
        if len(frames) > 8:
            frames = [frames[i] for i in np.unique(
                np.linspace(0, len(frames)-1, 8).astype(int))]
        points = np.asarray(seed.image_points, np.float32).reshape(-1, 1, 2)
        xyz = np.asarray(seed.object_points, np.float64)
        previous = seed.gray
        lk = dict(winSize=(self.tracker.flow_window, self.tracker.flow_window),
                  maxLevel=self.tracker.flow_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        started = time.perf_counter()
        skipped = 0
        detail = self.last_search_diagnostics['seed_transport'] = dict(
            hops=len(frames), source_points=len(points), retained_points=len(points),
            skipped_hops=0, reason='running')
        for gray in frames:
            if previous.shape != gray.shape or len(points) < 4:
                detail['reason'] = 'shape_mismatch_or_too_few_points'
                return None
            forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **lk)
            if forward is None or status is None:
                if not getattr(self, 'resilient_transport', False):
                    detail['reason'] = 'forward_flow_failed'
                    return None
                skipped += 1
                continue
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, **lk)
            if backward is None or back_status is None:
                if not getattr(self, 'resilient_transport', False):
                    detail['reason'] = 'backward_flow_failed'
                    return None
                skipped += 1
                continue
            uv = forward.reshape(-1, 2)
            error = np.linalg.norm((backward-points).reshape(-1, 2), axis=1)
            valid = ((status.ravel() != 0) & (back_status.ravel() != 0) &
                     np.isfinite(uv).all(axis=1) & np.isfinite(error) &
                     (error <= self.tracker.forward_backward_error) &
                     (uv[:, 0] >= 0) & (uv[:, 0] < gray.shape[1]) &
                     (uv[:, 1] >= 0) & (uv[:, 1] < gray.shape[0]))
            # One blurred/occluded hop must not destroy a useful source set.
            # Retry the next hop from the last good image. The latest image
            # still must be reached, then pass foreground PnP/QC.
            minimum = (max(4, min(20, int(len(points)*.1)))
                       if getattr(self, 'resilient_transport', False) else 4)
            if valid.sum() < minimum:
                if not getattr(self, 'resilient_transport', False):
                    detail.update(reason='too_few_flow_inliers', retained_points=int(valid.sum()))
                    return None
                skipped += 1
                continue
            points, xyz = forward[valid], xyz[valid]
            detail['retained_points'] = len(points)
            previous = gray
        detail.update(skipped_hops=skipped, retained_points=len(points),
                      ms=(time.perf_counter()-started)*1000, reason='transported')
        if len(points) < 4 or (frames and previous is not frames[-1]):
            detail['reason'] = 'latest_frame_not_reached'
            return None
        return TrackingSeed(previous, points.reshape(-1, 2).astype(np.float64),
                            xyz, seed.rvec.copy(), seed.tvec.copy())

    def _replenish_current_seed(self, candidate, image, camera_matrix, distortion):
        """Flow a verified search seed forward and add independently checked points."""
        self.last_refresh_verified = False
        self.last_refresh_refinement = {}
        detail = self.last_replenish_diagnostics = dict(reason='not_checked',
            source_points=len(candidate.image_points), added=0)
        current = self.tracker.seed
        if current is None:
            detail['reason'] = 'no_current_track'
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_points = np.asarray(candidate.image_points, np.float32).reshape((-1, 1, 2))
        if len(source_points) < 4:
            detail['reason'] = 'too_few_source_points'
            return 0
        lk = dict(winSize=(self.tracker.flow_window, self.tracker.flow_window),
                  maxLevel=self.tracker.flow_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            candidate.gray, gray, source_points, None, **lk)
        if forward is None or status is None:
            detail['reason'] = 'forward_flow_failed'
            return 0
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, candidate.gray, forward, None, **lk)
        if backward is None or back_status is None:
            detail['reason'] = 'backward_flow_failed'
            return 0
        points = forward.reshape((-1, 2)).astype(np.float64)
        xyz = np.asarray(candidate.object_points, np.float64)
        fb_error = np.linalg.norm(backward.reshape((-1, 2))-
                                  source_points.reshape((-1, 2)), axis=1)
        flow_valid = ((status.reshape(-1) != 0) & (back_status.reshape(-1) != 0)
            & np.isfinite(points).all(axis=1) & np.isfinite(xyz).all(axis=1)
            & (fb_error <= self.tracker.forward_backward_error)
            & (points[:,0] >= 0) & (points[:,0] < gray.shape[1])
            & (points[:,1] >= 0) & (points[:,1] < gray.shape[0]))
        detail['flow_valid'] = int(flow_valid.sum())
        context_refresh = getattr(self, 'last_search_diagnostics', {}).get('reference_matching') == 'pose_references'
        if getattr(self, 'refine_refresh_pose', False) or context_refresh:
            from replenishment_refinement import refine_pose
            current, self.last_refresh_refinement = refine_pose(current,
                points[flow_valid], xyz[flow_valid], image.shape, camera_matrix,
                distortion, self.global_localizer)
            self.last_refresh_refinement['committed'] = False
            if self.last_refresh_refinement.get('conflict'):
                detail['reason'] = 'fresh_pose_conflict'
                return 0
        projected = cv2.projectPoints(xyz, current.rvec, current.tvec,
                                      camera_matrix, distortion)[0].reshape((-1, 2))
        rotation = cv2.Rodrigues(np.asarray(current.rvec, np.float64))[0]
        depths = (np.einsum('ni,ji->nj', xyz, rotation) + np.asarray(current.tvec).reshape(3))[:, 2]
        height, width = gray.shape
        residual = np.linalg.norm(projected-points, axis=1)
        residual_limit = min(4., self.global_localizer.ransac_threshold)
        valid = ((status.reshape(-1) != 0) & (back_status.reshape(-1) != 0)
            & np.isfinite(points).all(axis=1) & np.isfinite(projected).all(axis=1)
            & np.isfinite(xyz).all(axis=1)
            & (fb_error <= self.tracker.forward_backward_error)
            & (residual <= residual_limit)
            & (depths > 0) & (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height))
        detail.update(projected_valid=int(valid.sum()),
                      residual_limit_px=residual_limit,
            residual_rejected=int((flow_valid & np.isfinite(residual) &
                (residual > residual_limit)).sum()),
            depth_rejected=int((flow_valid & (depths <= 0)).sum()),
            track_points_before=len(current.image_points))
        existing_uv = np.asarray(current.image_points, np.float64)
        existing_xyz = np.asarray(current.object_points, np.float64)
        detail.update(duplicate_existing=0, duplicate_new=0)
        additions_uv, additions_xyz = [], []
        eligible = np.flatnonzero(valid)
        # Give newly appearing regions a chance, rather than spending all 80
        # additions on the lowest-error cluster on the old face.
        from guided_matching import replenishment_indices
        ordered = replenishment_indices(points[eligible], -residual[eligible], existing_uv)
        for index in eligible[ordered]:
            uv, point = points[index], xyz[index]
            if (np.any(np.linalg.norm(existing_uv-uv, axis=1) < 6.) or
                np.any(np.linalg.norm(existing_xyz-point, axis=1) < 4.)):
                detail['duplicate_existing'] += 1
                continue
            if additions_uv and (np.any(np.linalg.norm(np.asarray(additions_uv)-uv, axis=1) < 6.)
                                 or np.any(np.linalg.norm(np.asarray(additions_xyz)-point, axis=1) < 4.)):
                detail['duplicate_new'] += 1
                continue
            additions_uv.append(uv)
            additions_xyz.append(point)
            if len(additions_uv) >= 80:
                break
        if len(additions_uv) < 4:
            detail.update(unique_additions=len(additions_uv),
                reason='too_few_unique_additions' if valid.any() else 'current_projection_or_flow_rejected')
            # A changed view may independently confirm already tracked points
            # without adding new XYZs. Count that verified refresh as the new
            # angle anchor, otherwise the same rotation triggers forever.
            if valid.sum() >= getattr(self.global_localizer, 'min_inliers', 10):
                quality = self.global_localizer.validate_pose(
                    xyz[valid], points[valid], current.rvec, current.tvec,
                    list(range(int(valid.sum()))), image.shape, camera_matrix, distortion)
                if quality['passed']:
                    detail['reason'] = 'existing_points_verified'
                    self.last_refresh_verified = True
                    if current is not self.tracker.seed:
                        self.tracker.seed = TrackingSeed(gray.copy(), existing_uv.copy(),
                            existing_xyz.copy(), current.rvec.copy(), current.tvec.copy())
                    self.last_refresh_refinement['committed'] = self.last_refresh_refinement.get('accepted', False)
                    self.replenish_anchor = SimpleNamespace(
                        rvec=current.rvec.copy(), tvec=current.tvec.copy())
                else:
                    detail.update(reason='fresh_points_qc_rejected', quality=quality)
            return 0
        combined_uv = np.vstack((existing_uv, additions_uv))
        combined_xyz = np.vstack((existing_xyz, additions_xyz))
        retained_additions = len(additions_uv)
        if len(combined_uv) > 2000:
            # Continuous refresh must not grow LK workload without bound.
            projected_all = cv2.projectPoints(combined_xyz, current.rvec, current.tvec,
                                              camera_matrix, distortion)[0].reshape(-1, 2)
            errors = np.linalg.norm(projected_all-combined_uv, axis=1)
            keep = spatial_indices(combined_uv, -errors, 2000)
            retained_additions = int(np.sum(keep >= len(existing_uv)))
            combined_uv, combined_xyz = combined_uv[keep], combined_xyz[keep]
        quality = self.global_localizer.validate_pose(
            combined_xyz, combined_uv, current.rvec, current.tvec,
            list(range(len(combined_uv))), image.shape, camera_matrix, distortion)
        if not quality['passed']:
            detail.update(reason='combined_points_qc_rejected', quality=quality,
                          unique_additions=len(additions_uv))
            return 0
        self.last_refresh_verified = True
        self.tracker.seed = TrackingSeed(gray.copy(), combined_uv, combined_xyz,
                                         current.rvec.copy(), current.tvec.copy())
        self.last_refresh_refinement['committed'] = self.last_refresh_refinement.get('accepted', False)
        detail.update(reason='added', added=retained_additions, unique_additions=len(additions_uv),
                      track_points_after=len(combined_uv), quality=quality)
        from guided_matching import region_counts
        box = (0,0,gray.shape[1],gray.shape[0])
        detail['regions'] = dict(grid=4,bbox=list(box),
            before=region_counts(existing_uv,box), after=region_counts(combined_uv,box),
            proposed_additions=region_counts(additions_uv,box))
        return retained_additions

    def _recover_recent_track(self, image, camera_matrix, distortion):
        recovery = getattr(self, 'short_recovery', None)
        if self.tracker.seed is not None or recovery is None:
            return None
        candidate = recovery.claim(image.shape, camera_matrix, distortion, time.monotonic())
        if candidate is None:
            return None
        self.tracker.seed = self.tracker._copy_seed(candidate)
        started = time.perf_counter()
        tracked = self.tracker._track(image, camera_matrix, distortion)
        self.background_refresh.track_sample((time.perf_counter()-started)*1000)
        if tracked is None:
            self.tracker.seed = None
            return None
        # This pose has freshly passed forward/backward flow, PnP and all QC.
        # Retaining an anchor alone never authorizes displaying its old pose.
        recovery.accepted()
        self.active_reference_name = recovery.reference
        self.last_track_failure = None
        return replace(tracked[0], detail='TRACKING_RECOVERED'), tracked[1]

    def _remember_verified_track(self, camera_matrix, distortion):
        self.last_track_failure = None
        recovery = getattr(self, 'short_recovery', None)
        if recovery is not None:
            recovery.remember(self.tracker.seed, self.active_reference_name,
                              camera_matrix, distortion, time.monotonic())

    def _viewpoint_reasons(self, camera_matrix, distortion, image_shape, prediction=None):
        from viewpoint_refresh import ViewpointRefresh
        monitor = getattr(self, 'viewpoint_refresh', None)
        if monitor is None:
            map_data = getattr(self.global_localizer, 'map_data', None)
            xyz = np.asarray(getattr(map_data, 'xyz', np.empty((0, 3))), np.float64)
            center = getattr(self.global_localizer, 'manifest', {}).get('center')
            if center is None:
                finite = xyz[np.isfinite(xyz).all(axis=1)]
                center = np.median(finite, axis=0) if len(finite) else np.zeros(3)
            self.viewpoint_refresh = monitor = ViewpointRefresh(xyz, center)
        reasons = monitor.reasons(self.tracker.seed, self.replenish_anchor,
            camera_matrix, distortion, image_shape, getattr(self, 'replenish_angle_deg', 12.),
            prediction, early=getattr(self, 'replenish_mode', '') == 'background')
        self.last_viewpoint_refresh = monitor.info.copy()
        return reasons

    def localize(self, image, camera_matrix, distortion):
        background = getattr(self, 'replenish_mode', '') == 'background'
        if getattr(self, 'search_frame_buffer', None) is not None and getattr(self, 'last_capture_image', None) is not image:
            self.record_capture(image, camera_matrix, distortion, time.monotonic())
        if self.future is not None and hasattr(self, 'search_frames'):
            self._remember_search_frame(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            tracks = getattr(self, 'query_tracks', None)
            if tracks is not None:
                tracks.advance(list(self.search_frames), time.monotonic())
                if tracks.initialized:
                    self.search_frames.clear()
        # Recover first, so a completed replenishment can be merged into a
        # freshly verified current track instead of discarded for a short gap.
        tracked = self._recover_recent_track(image, camera_matrix, distortion)
        self._consume_completed(image, camera_matrix, distortion)
        self.last_replenished = 0
        previous_pose = (SimpleNamespace(rvec=self.tracker.seed.rvec.copy(),
                         tvec=self.tracker.seed.tvec.copy())
                         if background and self.tracker.seed is not None else None)
        if (self.pending_seed is not None and self.tracker.seed is not None and
            not background and
            getattr(self, 'pending_mode', 'global') != 'guided' and
            self._pending_seed_age_ms() <= self.max_seed_age*1000):
            old_seed = self.tracker.seed
            old_reference = self.active_reference_name
            self.tracker.seed = self.pending_seed
            tracked = self.tracker._track(image, camera_matrix, distortion)
            if tracked is not None:
                if getattr(self, 'search_frame_buffer', None) is not None:
                    self.search_frame_buffer.mark_pose(self.tracker.seed,
                        getattr(self.global_localizer, 'manifest', {}).get('bounds'))
                self.active_reference_name = self.pending_reference_name
                self.tracker.frames_since_global = 0
                self.replenish_anchor = SimpleNamespace(
                    rvec=self.tracker.seed.rvec.copy(), tvec=self.tracker.seed.tvec.copy()) if hasattr(self.tracker.seed, 'rvec') else None
            else:
                self.tracker.seed = old_seed
                self.active_reference_name = old_reference
            self.pending_seed = None
            self.pending_mode = None
            self.pending_reference_name = None
        if self.tracker.seed is not None:
            if tracked is None:
                track_started = time.perf_counter()
                tracked = self.tracker._track(image, camera_matrix, distortion)
                if background:
                    self.background_refresh.track_sample((time.perf_counter()-track_started)*1000)
            if tracked is not None:
                if background and getattr(self, 'awaiting_global_confirmation', False):
                    self._diagnostic('track_confirmed', result=tracked[0])
                    self.background_refresh.accepted()
                    self.awaiting_global_confirmation = False
                if self.pending_seed is not None and (background or getattr(self, 'pending_mode', 'global') == 'guided'):
                    self.last_refresh_verified = False
                    if self._pending_seed_age_ms() <= self.max_seed_age*1000:
                        before_refresh = self.tracker.seed
                        before_anchor = getattr(self, 'replenish_anchor', None)
                        before_debug = self.tracker.last_debug
                        before_quality = getattr(self.global_localizer, 'last_tracking_quality', {})
                        refresh_started = time.perf_counter()
                        self.last_replenished = self._replenish_current_seed(
                            self.pending_seed, image, camera_matrix, distortion)
                        if getattr(self, 'last_refresh_refinement', {}).get('conflict'):
                            # Independent, strong current matches contradict
                            # the old track. Neither pose may drive the overlay.
                            conflict = self.last_refresh_refinement.copy()
                            self._diagnostic('replenishment', added=0,
                                diagnostics=getattr(self, 'last_replenish_diagnostics', {}),
                                refinement=conflict)
                            self.reset_tracking()
                            self.last_refresh_refinement = conflict
                            self.last_track_failure = 'fresh_pose_conflict'
                            self.record_capture(image, camera_matrix, distortion, time.monotonic())
                            self._submit(image, camera_matrix, distortion)
                            return LiveResult('FAIL', 'FRESH_POSE_CONFLICT', 0, 0, 0, 0.,
                                None, None, None, None,
                                (time.perf_counter()-refresh_started)*1000, 'HIDDEN'), image.copy()
                        if self.last_replenished or getattr(self, 'last_refresh_refinement', {}).get('committed'):
                            # Publish the merged points and current PnP together,
                            # never the pre-merge debug/counts from this frame.
                            refreshed = self.tracker._track(image, camera_matrix, distortion)
                            if refreshed is not None:
                                elapsed = (time.perf_counter()-refresh_started)*1000
                                refined = getattr(self, 'last_refresh_refinement', {}).get('committed')
                                tracked = (replace(refreshed[0], detail='TRACKING_REFINED' if refined else 'TRACKING_REPLENISHED',
                                    processing_ms=tracked[0].processing_ms+elapsed), refreshed[1])
                                self.last_replenish_diagnostics.update(final_current_frame_qc=True,
                                    track_points_after_final_pnp=len(self.tracker.seed.image_points))
                            else:
                                self.tracker.seed, self.tracker.last_debug = before_refresh, before_debug
                                self.replenish_anchor = before_anchor
                                self.global_localizer.last_tracking_quality = before_quality
                                self.last_refresh_refinement['committed'] = False
                                self.last_refresh_refinement['reason'] = 'final_current_frame_qc'
                                self.last_refresh_verified = False
                                attempted = self.last_replenished
                                self.last_replenished = 0
                                self.last_replenish_diagnostics.update(added=0, attempted_added=attempted,
                                    reason='final_current_frame_qc', committed=False,
                                    final_current_frame_qc=False,
                                    track_points_after=len(before_refresh.image_points))
                        self._diagnostic('replenishment', added=self.last_replenished,
                            verified=self.last_refresh_verified,
                            diagnostics=getattr(self, 'last_replenish_diagnostics', {}),
                            refinement=getattr(self, 'last_refresh_refinement', {}),
                            integration_ms=(time.perf_counter()-refresh_started)*1000)
                        if self.last_replenished:
                            self.replenish_anchor = SimpleNamespace(
                                rvec=self.tracker.seed.rvec.copy(), tvec=self.tracker.seed.tvec.copy())
                            print(f'{self.source_label} pose-guided replenishment: '
                                  f'+{self.last_replenished} points, '
                                  f'{len(self.tracker.seed.image_points)} tracked')
                    if background:
                        if self.last_refresh_verified:
                            self.background_refresh.accepted(self.last_replenished)
                            if self.pending_mode == 'global':
                                self.tracker.frames_since_global = 0
                        else:
                            self.background_refresh.rejected('current_frame_qc_rejected')
                self.pending_seed = None
                self.pending_mode = None
                self.pending_reference_name = None
                if getattr(self, 'search_frame_buffer', None) is not None:
                    self.search_frame_buffer.mark_pose(self.tracker.seed,
                        getattr(self.global_localizer, 'manifest', {}).get('bounds'))
                self.tracker.frames_since_global += 1
                self.frames_since_replenish = getattr(self, 'frames_since_replenish', 0) + 1
                self.last_debug = self.tracker.last_debug
                coverage_due = False
                if getattr(self, 'replenish_mode', 'interval') in ('coverage', 'background'):
                    debug = self.tracker.last_debug
                    indices = sorted(debug.inlier_indices) if debug is not None else []
                    xyz = (np.asarray(debug.matched_xyz)[indices] if indices
                           else np.empty((0, 3)))
                    self.last_coverage_ok = has_spread_triangle(
                        xyz, self.replenish_distance_mm, self.replenish_height_mm)
                    reasons = [] if self.last_coverage_ok else ['3d_support']
                    quality = getattr(self.global_localizer, 'last_tracking_quality', {})
                    if quality.get('coverage', 1.) < getattr(self, 'replenish_image_coverage', .02):
                        reasons.append('image_coverage')
                    anchor = getattr(self, 'replenish_anchor', None)
                    if anchor is None:
                        self.replenish_anchor = SimpleNamespace(
                            rvec=self.tracker.seed.rvec.copy(), tvec=self.tracker.seed.tvec.copy())
                    forecast_pose = None
                    if background:
                        if not hasattr(self, 'handoff'):
                            self.handoff = TrackingHandoff()
                        radius = getattr(self.global_localizer, 'manifest', {}).get('radius', 100.)
                        reasons.extend(self.handoff.observe(time.monotonic(), self.tracker.seed,
                            self.background_refresh.search_ms, radius,
                            max(20, getattr(self.global_localizer, 'min_inliers', 10)*2)))
                        if (getattr(self, 'short_recovery', None) is not None and
                            'points_declining' in reasons and
                            self.handoff.info.get('forecast_points', float('inf')) <
                            max(20, getattr(self.global_localizer, 'min_inliers', 10)*2)):
                            reasons.append('support_critical')
                        prediction = self.handoff.prediction
                        if prediction is not None:
                            forecast_pose = SimpleNamespace(rvec=prediction[0], tvec=prediction[1])
                        if len(self.tracker.seed.image_points) < max(20, getattr(self.global_localizer, 'min_inliers', 10)*2):
                            reasons.append('few_points')
                        if ('support_critical' not in reasons and
                            len(self.tracker.seed.image_points) < max(40, getattr(self.global_localizer, 'min_inliers', 10)*4)):
                            reasons.append('support_critical')
                        error = getattr(tracked[0], 'mean_reprojection_error', None)
                        if error is not None and error > min(3., getattr(self.global_localizer, 'max_reprojection_error', 6.)*.6):
                            reasons.append('residual_rising')
                        if previous_pose is not None:
                            angle, distance = GLBVirtualLiveLocalizer._pose_distance(previous_pose, self.tracker.seed)
                            radius = getattr(self.global_localizer, 'manifest', {}).get('radius', 100.)
                            if angle > 20. or distance > max(1., radius)*.25:
                                reasons.append('pose_jump')
                    reasons.extend(self._viewpoint_reasons(camera_matrix, distortion, image.shape,
                                                          forecast_pose))
                    self.last_replenish_reasons = reasons
                    coverage_due = (bool(reasons) and
                        time.monotonic() - self.last_replenish_attempt >=
                        self.replenish_cooldown)
                if background:
                    decision = self.background_refresh.decide(time.monotonic(), busy=self.future is not None,
                        weak_reasons=self.last_replenish_reasons,
                        safety_due=self.tracker.frames_since_global >= self.reseed_frames)
                    if decision:
                        self._submit(image, camera_matrix, distortion,
                            pose_prior=self.tracker.seed if decision == 'guided' else None)
                elif self.future is None:
                    if self.tracker.frames_since_global >= self.reseed_frames:
                        self._submit(image, camera_matrix, distortion)
                        self.tracker.frames_since_global = 0
                        self.frames_since_replenish = 0
                    elif (coverage_due or
                          (getattr(self, 'replenish_mode', 'interval') == 'interval' and
                           getattr(self, 'replenish_frames', 0) > 0 and
                           self.frames_since_replenish >= self.replenish_frames)):
                        self._submit(image, camera_matrix, distortion,
                                     pose_prior=self.tracker.seed)
                        if coverage_due:
                            print(f'{self.source_label} replenish: {",".join(reasons)} '
                                  f'({len(xyz)} points; edges >= '
                                  f'{self.replenish_distance_mm:g} mm, triangle height >= '
                                  f'{self.replenish_height_mm:g} mm)', flush=True)
                        self.frames_since_replenish = 0
                self._remember_verified_track(camera_matrix, distortion)
                self._update_coarse(image)
                return tracked
            self.last_track_failure = getattr(self.tracker, 'last_failure_reason', None)
            self._diagnostic('tracking_lost', reason=self.last_track_failure,
                search_in_progress=self.future is not None,
                quality=getattr(self.global_localizer, 'last_tracking_quality', {}))
            if background and getattr(self, 'awaiting_global_confirmation', False):
                self.background_refresh.rejected('current_frame_qc_rejected')
                self.awaiting_global_confirmation = False
            self.tracker.seed = None
            if background and hasattr(self, 'handoff'):
                self.handoff.clear()
            self.active_reference_name = None
            if getattr(self, 'pending_mode', None) == 'guided':
                self.pending_seed = None
                self.pending_mode = None
                self.pending_reference_name = None
        if self.pending_seed is not None:
            if background and getattr(self, 'pending_mode', None) == 'guided':
                # Lost tracking invalidates the guided search's prior.
                self.pending_seed = None
                self.pending_mode = None
                self.pending_reference_name = None
        if self.pending_seed is not None:
            self.tracker.seed = self.pending_seed
            self.active_reference_name = self.pending_reference_name
            self.pending_seed = None
            self.pending_mode = None
            self.pending_reference_name = None
            tracked = self.tracker._track(image, camera_matrix, distortion)
            if tracked is not None:
                if getattr(self, 'search_frame_buffer', None) is not None:
                    self.search_frame_buffer.mark_pose(self.tracker.seed,
                        getattr(self.global_localizer, 'manifest', {}).get('bounds'))
                self.tracker.frames_since_global = 0
                self.last_debug = self.tracker.last_debug
                self.replenish_anchor = SimpleNamespace(
                    rvec=self.tracker.seed.rvec.copy(), tvec=self.tracker.seed.tvec.copy()) if hasattr(self.tracker.seed, 'rvec') else None
                self._remember_verified_track(camera_matrix, distortion)
                self._update_coarse(image)
                return tracked
            self.tracker.seed = None
            self.active_reference_name = None
        if self.future is None:
            retry = (not background or self.background_refresh.decide(time.monotonic(),
                     busy=False, weak_reasons=['tracking_lost']) is not None)
            if retry:
                recovery = getattr(self, 'short_recovery', None)
                hint = recovery.nearby_hint(time.monotonic()) if recovery is not None else None
                if hint is not None:
                    self._submit(image, camera_matrix, distortion, recovery_hint=hint)
                else:
                    self._submit(image, camera_matrix, distortion)
        self._update_coarse(image)
        return self._search_display(image)

    def _remember_search_frame(self, gray):
        # Keep frames near the inference source even during a cold/long search.
        # Decimate at fixed memory capacity rather than evicting all old frames.
        history = self.search_frames
        if history.maxlen is not None and len(history) >= history.maxlen:
            frames = list(history)
            indices = np.unique(np.linspace(0, len(frames)-1,
                                           max(2, history.maxlen//2)).astype(int))
            history.clear()
            history.extend(frames[index] for index in indices)
        history.append(gray)

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

    def reset_tracking(self) -> None:
        self.tracker.seed = None
        self.pending_seed = None
        self.pending_mode = None
        self.active_reference_name = None
        self.pending_reference_name = None
        self.discard_search_result = self.future is not None
        self.search_frames.clear()
        if hasattr(self, 'query_tracks'):
            self.query_tracks.begin(None)
        self.pending_seed_stamp = None
        self.last_search_result = None
        self.last_search_diagnostics = {}
        self.last_debug = None
        self.replenish_anchor = None
        if hasattr(self, 'viewpoint_refresh'):
            self.viewpoint_refresh.reset()
        self.last_viewpoint_refresh = {}
        if getattr(self, 'search_frame_buffer', None) is not None:
            self.search_frame_buffer.clear()
            self.last_capture_image = None
            self.search_frame_selection = {}
        if hasattr(self, 'handoff'):
            self.handoff.clear()
        if getattr(self, 'short_recovery', None) is not None:
            self.short_recovery.clear()
        self.last_refresh_refinement = {}
        self.coarse_reference_name = None
        if getattr(self, 'replenish_mode', '') == 'background':
            from background_refresh import BackgroundRefresh
            self.background_refresh = BackgroundRefresh(cooldown=self.replenish_cooldown)
            self.background_refresh.fast_recovery = getattr(self, 'short_recovery', None) is not None
            self.awaiting_global_confirmation = False

    def close(self) -> None:
        self.query_tracks.begin(None)
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
        max_keypoints: int = 2000,
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
        guided_optimization: bool = True,
        guided_feature_budget: int = 2000,
        guided_pixel_radius: float = 96.,
        guided_early_descriptors: bool = True,
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
        if guided_feature_budget < 32 or not math.isfinite(guided_pixel_radius) or guided_pixel_radius <= 0:
            raise LivePnPError('Invalid guided feature budget or projection radius')
        self.guided_optimization = guided_optimization
        self.guided_feature_budget = guided_feature_budget
        self.guided_pixel_radius = guided_pixel_radius
        self.guided_early_descriptors = guided_early_descriptors
        self.reference_feature_xyz = {}
        if not math.isfinite(max_coarse_rotation_deg) or not 0 < max_coarse_rotation_deg <= 180:
            raise LivePnPError('Invalid coarse-pose rotation threshold')
        self.max_coarse_rotation_deg = max_coarse_rotation_deg
        self.last_diagnostics = {}
        self.progress_lock = threading.Lock()
        self.progress = {}
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
        self.reference_offsets = {}
        self.reference_crop_shapes = {}
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
        self.last_aliked_crop = None
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

    def _progress(self, stage, **values):
        # Never acquire the inference lock in the browser main/capture thread.
        if not hasattr(self, 'progress_lock'):
            self.progress_lock = threading.Lock()
        with self.progress_lock:
            self.progress = dict(getattr(self, 'progress', {}), stage=stage, **values)
        sink = getattr(self, 'diagnostic_sink', None)
        if sink is not None:
            sink('search_stage', **getattr(self, 'diagnostic_context', {}), stage=stage, **values)

    def progress_snapshot(self):
        with self.progress_lock:
            return dict(self.progress)

    def _pose_guided_views(self, rvec, tvec, top_k=3):
        """Select nearby virtual cameras without appearance-wide retrieval."""
        if self.source_type != '3dgs' or self.runtime_transform != 'identity':
            return []
        rotation = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
        translation = np.asarray(tvec, np.float64).reshape(3)
        center = -rotation.T @ translation
        radius = max(float(self.manifest.get('radius', 1.)), 1.)
        candidates = []
        for index, view in enumerate(self.views):
            matrix = np.asarray(view.get('world_to_camera'), np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                continue
            reference_rotation = matrix[:3, :3]
            relative = rotation @ reference_rotation.T
            angle = float(np.degrees(np.arccos(np.clip(
                (np.trace(relative)-1.)/2., -1., 1.))))
            reference_center = -reference_rotation.T @ matrix[:3, 3]
            distance = float(np.linalg.norm(center-reference_center))
            candidates.append((angle + 20. * distance/radius, angle, distance, index))
        candidates.sort()
        # A far-off pose is not a useful prior. Let the normal retrieval recover.
        return [index for _, angle, _, index in candidates[:top_k]
                if angle <= self.max_coarse_rotation_deg]

    def _handoff_views(self, rvec, tvec, view_prediction):
        current = self._pose_guided_views(rvec, tvec)
        if not current or view_prediction is None:
            return current
        upcoming = self._pose_guided_views(*view_prediction)
        # Always keep the best current view, then introduce adjacent/new faces.
        # Future-only points will still fail CURRENT-image projection/matching.
        return list(dict.fromkeys(current[:1]+upcoming+current))[:3]

    def localize_guided(self, image, camera_matrix, distortion, rvec, tvec, view_prediction=None):
        """Match only pose-near views; reject solutions contradicting the prior."""
        indices = self._handoff_views(rvec, tvec, view_prediction)
        if not indices:
            return self.localize(image, camera_matrix, distortion)
        ranked = [(1., 0., index) for index in indices]
        with self.lock:
            self._suppress_diagnostics = True
            completed = False
            try:
                answer = self._quality_attempts(lambda: self._guided_attempts(
                    image, camera_matrix, distortion, ranked, rvec, tvec,
                    handoff=view_prediction is not None))
                completed = True
                return answer
            finally:
                self._suppress_diagnostics = False
                if completed:
                    self._write_diagnostics()

    def localize_nearby(self, image, camera_matrix, distortion, rvec, tvec):
        """Recent pose selects views only; solve a fresh independent PnP."""
        indices = self._pose_guided_views(rvec, tvec)
        if not indices:
            return self.localize(image, camera_matrix, distortion)
        ranked = [(1., 0., index) for index in indices]
        with self.lock:
            return self._quality_attempts(lambda: self._localize_locked(
                image, camera_matrix, distortion, ranked_override=ranked,
                pose_prior=None, optimize_guided=False))

    def _guided_attempts(self, image, camera_matrix, distortion, ranked, rvec, tvec, handoff=False):
        workspace = {} if getattr(self, 'guided_query_reuse', True) else None
        result, canvas = self._localize_locked(image, camera_matrix, distortion,
            ranked_override=ranked, pose_prior=(rvec, tvec), query_workspace=workspace,
            allow_guided_early_stop=not handoff)
        if (result.status == 'PASS' or not self.guided_optimization or self.pipeline_policy == 'legacy'
            or self.last_diagnostics.get('reason', '').endswith('_IMAGE_QUALITY_HOLD')):
            return result, canvas
        # A stale/noisy pose or proxy geometry must not permanently block
        # replenishment. Retry the same shortlist without the new filters.
        first = self.last_diagnostics.copy()
        first_result = result
        result, canvas = self._localize_locked(image, camera_matrix, distortion,
            ranked_override=ranked, pose_prior=(rvec, tvec), optimize_guided=False,
            query_workspace=workspace, allow_guided_early_stop=not handoff)
        self.last_diagnostics['guided_fallback'] = dict(
            reason=first.get('status'), optimized_timings_ms=first.get('timings_ms'),
            optimized_candidates=first.get('candidates'))
        for key, value in first.get('timings_ms', {}).items():
            if key in self.last_diagnostics['timings_ms']:
                self.last_diagnostics['timings_ms'][key] += value
        result = replace(result, processing_ms=result.processing_ms + first_result.processing_ms,
                         matching_ms=result.matching_ms + first_result.matching_ms,
                         sift_ms=result.sift_ms + first_result.sift_ms,
                         pnp_ms=result.pnp_ms + first_result.pnp_ms)
        return result, canvas

    def _feature_xyz(self, view_index, features):
        cache_index = self._variant_key(view_index)
        cached = self.reference_feature_xyz.get(cache_index)
        if cached is not None:
            return cached
        points = features['keypoints'][0].detach().cpu().numpy()
        points = points + np.asarray(getattr(self, 'reference_offsets', {}).get(
            view_index, (0., 0.)), np.float32)
        xyz = self._xyz(view_index)
        geometry = self.geometry_cache.get(view_index)
        if geometry is None:
            sampled, indices = sample_xyz_buffer(points, xyz, self.xyz_search_radius)
        else:
            from splat_geometry import sample_geometry
            sampled, indices, _ = sample_geometry(points, geometry)
        mapped = np.full((len(points), 3), np.nan)
        if len(indices):
            mapped[np.asarray(indices, int)] = self._source_xyz_to_runtime(np.asarray(sampled))
        self.reference_feature_xyz[cache_index] = mapped
        return mapped

    @staticmethod
    def _feature_arrays(features):
        points = features['keypoints'][0].detach().cpu().numpy()
        scores = features.get('keypoint_scores')
        return points, None if scores is None else scores[0].detach().cpu().numpy()

    def _guided_reference_features(self, view_index, features, pose_prior,
                                   image, matrix, distortion, bbox, query_points):
        xyz = self._feature_xyz(view_index, features)
        predicted, valid = project_candidates(xyz, *pose_prior, matrix,
                                             distortion, image.shape, bbox)
        indices = np.flatnonzero(valid)
        if len(indices) and len(query_points):
            distances, _ = cv2.batchDistance(
                np.ascontiguousarray(predicted[indices], np.float32),
                np.ascontiguousarray(query_points, np.float32), cv2.CV_32F,
                normType=cv2.NORM_L2, K=1)
            indices = indices[distances[:, 0] <= self.guided_pixel_radius]
        points, scores = self._feature_arrays(features)
        selected = spatial_indices(points[indices], None if scores is None else scores[indices],
                                   self.guided_feature_budget)
        indices = indices[selected]
        return subset_features(self.torch, features, indices), predicted[indices], dict(
            raw=len(points), valid_xyz=int(np.isfinite(xyz).all(axis=1).sum()),
            projected_in_roi=int(valid.sum()), selected=len(indices))

    def _guided_strong_pose(self, pose, transfers, quality, coarse_pose, pose_prior):
        if pose is None or not quality or not quality['passed']:
            return False
        if coarse_pose is not None and not coarse_pose['passed']:
            return False
        count = len(pose.inlier_indices)
        if count < max(20, self.min_inliers*2) or count/max(1, len(transfers)) < .5:
            return False
        if not math.isfinite(pose.mean_error) or pose.mean_error > min(3., self.max_reprojection_error):
            return False
        prior = SimpleNamespace(rvec=np.asarray(pose_prior[0]), tvec=np.asarray(pose_prior[1]))
        angle, distance = self._pose_distance(pose, prior)
        xyz = [transfers[i].xyz for i in pose.inlier_indices]
        return angle <= 5. and distance <= 10. and has_spread_triangle(xyz)

    def _variant_key(self, view_index):
        variant = getattr(self, '_feature_variant', 'raw')
        if self._normalized_object_matching():
            return (view_index, variant, self._matching_resize(), 'object-crop-v1')
        return view_index if variant == 'raw' else (view_index, variant)

    def _normalized_object_matching(self):
        return (getattr(self, 'source_type', None) == '3dgs'
                and getattr(self, 'pipeline_policy', None) != 'legacy'
                and getattr(self, 'retrieval_crop', False))

    def _matching_resize(self):
        return int(getattr(self, 'query_resize', self.resize))

    def _reference_features(self, view_index: int):
        cache_index = self._variant_key(view_index)
        cached = self.reference_feature_cache.get(cache_index)
        if cached is not None:
            return cached
        view = self.views[view_index]
        image_path = self.views_path / str(view['image_name'])
        normalized = self._normalized_object_matching()
        matching_resize = self._matching_resize() if normalized else self.resize
        signature = self.extractor_fingerprint.encode()
        if normalized:
            signature += f':object-crop-v1:{matching_resize}'.encode()
        if getattr(self, '_feature_variant', 'raw') != 'raw':
            signature += PHOTOMETRIC_MODE.encode()
        image = None
        if normalized:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise LivePnPError(f"Virtual RGB missing: {view['image_name']}")
            bbox = object_bbox(reference_rgb_mask(image), margin=.08)
            cropped = crop_live_object(image, bbox)
            if cropped is None:
                raise LivePnPError(f"Virtual object crop missing: {view['image_name']}")
            image, (x0, y0, _, _) = cropped
            self.reference_offsets[view_index] = (x0, y0)
            self.reference_crop_shapes[view_index] = image.shape
        cache_key = hashlib.sha256(image_path.read_bytes() + signature).hexdigest()
        cache_path = self.cache_directory / f'{cache_key}.npz'
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    features = {key: self.torch.from_numpy(archive[key].copy()).to(self.device)
                                for key in archive.files}
                if not {'keypoints', 'descriptors', 'image_size'} <= features.keys():
                    raise ValueError('Incomplete reference cache')
                self.reference_feature_cache[cache_index] = features
                return features
            except (OSError, ValueError):
                pass  # Re-extract an interrupted/corrupt cache; never trust pickle.
        if image is None:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise LivePnPError(f"Virtual RGB missing: {view['image_name']}")
        if getattr(self, '_feature_variant', 'raw') != 'raw':
            image = correct_luminance(image)
        with self.torch.inference_mode():
            features = _extract(
                self.extractor,
                _image_tensor(self.torch, image, self.device),
                matching_resize,
            )
        self.reference_feature_cache[cache_index] = features
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
                    {'shared-gaussian-moments', 'ray-consistent-legacy-depth',
                     'ray-legacy-depth'}):
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
            self.min_inlier_ratio, self.max_reprojection_error,
            min_coverage=getattr(self, 'min_coverage', .005),
            max_p95_error=1.5*self.max_reprojection_error)

    def _pose_quality(self, pose, transfers, image, camera_matrix, distortion):
        return self.validate_pose(np.asarray([t.xyz for t in transfers]),
            np.asarray([t.query_uv for t in transfers]), pose.rvec, pose.tvec,
            pose.inlier_indices, image.shape, camera_matrix, distortion)

    @staticmethod
    def _pose_distance(first, second):
        a, b = cv2.Rodrigues(first.rvec)[0], cv2.Rodrigues(second.rvec)[0]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))
        # OpenCV returns (3,1), while LiveResult/JSON use (3,). Mixing them
        # otherwise broadcasts into a 3x3 matrix and invents a pose jump.
        first_t = np.asarray(first.tvec, np.float64).reshape(3)
        second_t = np.asarray(second.tvec, np.float64).reshape(3)
        distance = float(np.linalg.norm(-a.T @ first_t + b.T @ second_t))
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
        if getattr(self, '_suppress_diagnostics', False):
            return
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
            return self._quality_attempts(lambda: self._localize_locked(image, camera_matrix, distortion))

    def compare_excluded_face(self, image, camera_matrix, distortion, polygon,
                              max_mean_error_px=None, min_coverage=None,
                              view_count=None, reference_names=None):
        """Re-localize one frozen image twice with an identical coarse shortlist.

        This deliberately bypasses temporal tracking.  The exclusion affects
        only live-to-reference correspondences, never retrieval or query RGB.
        """
        polygon = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('Invalid excluded face polygon')
        with self.lock:
            # A pending live search may still need these fields when tracking
            # resumes.  Validation results are returned, never left as a seed.
            saved = {name: getattr(self, name, None) for name in (
                'last_tracking_seed', 'last_debug', 'last_diagnostics',
                'last_reference_name', 'last_retrieval', 'last_crop_debug',
                'last_candidate_log', 'last_aliked_crop', '_suppress_diagnostics')}
            previous_progress = self.progress_snapshot()
            original_max_error = getattr(self, 'max_reprojection_error', None)
            original_min_coverage = getattr(self, 'min_coverage', None)
            original_top_k = self.top_k if view_count is not None else None
            try:
                if max_mean_error_px is not None:
                    self.max_reprojection_error = float(max_mean_error_px)
                if min_coverage is not None:
                    self.min_coverage = float(min_coverage)
                if view_count is not None:
                    self.top_k = min(int(view_count), len(self.views))
                self._suppress_diagnostics = True
                if reference_names is None:
                    ranked, _ = self._rank_views(image)
                    ranked = list(ranked)
                    crop_debug = dict(self.last_crop_debug)
                else:
                    by_name = {view['image_name']: index for index, view in enumerate(self.views)}
                    missing = [name for name in reference_names if name not in by_name]
                    if missing or not reference_names:
                        raise ValueError(f'Original PASS reference view unavailable: {missing or reference_names}')
                    ranked = [(1., 0., by_name[name]) for name in dict.fromkeys(reference_names)]
                    if self.retrieval_auto_roi:
                        bbox, detection = self.retriever.detect_query(image)
                        crop_debug = dict(bbox=bbox, detection=detection)
                    else:
                        crop_debug = dict(bbox=self.retrieval_roi if self.retrieval_crop else None)
                baseline, _ = self._quality_attempts(lambda: self._localize_locked(
                    image, camera_matrix, distortion, ranked_override=ranked,
                    query_workspace={'crop_debug': crop_debug}))
                baseline_diagnostics = copy.deepcopy(self.last_diagnostics)
                excluded, _ = self._quality_attempts(lambda: self._localize_locked(
                    image, camera_matrix, distortion, ranked_override=ranked,
                    query_workspace={'crop_debug': crop_debug},
                    excluded_polygon=polygon))
                diagnostics = copy.deepcopy(self.last_diagnostics)
                diagnostics['baseline_diagnostics'] = baseline_diagnostics
                return baseline, excluded, self.last_debug, diagnostics
            finally:
                if max_mean_error_px is not None:
                    if original_max_error is None:
                        del self.max_reprojection_error
                    else:
                        self.max_reprojection_error = original_max_error
                if min_coverage is not None:
                    if original_min_coverage is None:
                        del self.min_coverage
                    else:
                        self.min_coverage = original_min_coverage
                if view_count is not None:
                    self.top_k = original_top_k
                for name, value in saved.items():
                    setattr(self, name, value)
                with self.progress_lock:
                    self.progress = previous_progress

    def _quality_attempts(self, attempt):
        """Raw first; a single mild photometric retry cannot bypass pose QC."""
        self._feature_variant = 'raw'
        suppressed = getattr(self, '_suppress_diagnostics', False)
        self._suppress_diagnostics = True
        completed = False
        try:
            result, canvas = attempt()
            first = self.last_diagnostics.copy()
            quality = first.get('input_quality', {})
            # Do not use contrast enhancement to rescue ambiguous/failed-QC poses.
            retry = (getattr(self, 'source_type', '') == '3dgs'
                and getattr(self, 'pipeline_policy', '') != 'legacy'
                and result.status == 'FAIL' and first.get('status') == 'FAIL'
                and quality.get('needs_correction', False)
                and (str(first.get('reason', '')).endswith('_PNP_FAILED')
                     or str(first.get('reason', '')).endswith('_NO_OBJECT_FEATURES')))
            if retry:
                self._feature_variant = PHOTOMETRIC_MODE
                self._progress('features', photometric_mode=PHOTOMETRIC_MODE,
                               photometric_retry=True)
                second, canvas = attempt()  # New query workspace; never reuse raw descriptors.
                self.last_diagnostics['photometric_retry'] = dict(
                    raw_status=result.status, raw_matches=result.matches,
                    raw_timings_ms=first.get('timings_ms', {}), mode=PHOTOMETRIC_MODE)
                for key, value in first.get('timings_ms', {}).items():
                    timings = self.last_diagnostics.setdefault('timings_ms', {})
                    timings[key] = timings.get(key, 0.) + value
                result = replace(second, processing_ms=second.processing_ms+result.processing_ms,
                    sift_ms=second.sift_ms+result.sift_ms,
                    matching_ms=second.matching_ms+result.matching_ms,
                    pnp_ms=second.pnp_ms+result.pnp_ms, mpr_ms=second.mpr_ms+result.mpr_ms)
            self._progress('done', photometric_retry=retry,
                           total_ms=result.processing_ms)
            completed = True
            return result, canvas
        finally:
            self._feature_variant = 'raw'
            self._suppress_diagnostics = suppressed
            if completed:
                self._write_diagnostics()

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
        self._progress('done', outcome='FAIL', reason=detail)
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

    def _localize_locked(self, image, camera_matrix, distortion,
                         ranked_override=None, pose_prior=None, optimize_guided=None,
                         query_workspace=None, allow_guided_early_stop=True,
                         excluded_polygon=None):
        started = time.perf_counter()
        if not hasattr(self, 'progress_lock'):
            self.progress_lock = threading.Lock()
        with self.progress_lock:
            self.progress = dict(stage='retrieval', started_at=time.monotonic(),
                                 candidates=[], candidate_index=0, candidate_total=0,
                                 photometric_mode=getattr(self, '_feature_variant', 'raw'))
        optimized = (pose_prior is not None and self.pipeline_policy != 'legacy' and
            (self.guided_optimization if optimize_guided is None else optimize_guided))
        self.last_aliked_crop = None
        self.last_reference_name = '—'
        self.last_candidate_log = 'none'
        self.last_retrieval = []
        fixed_views = ranked_override is not None and pose_prior is None
        self.last_diagnostics = dict(policy=self.pipeline_policy,
            search_mode='fixed_views' if fixed_views else
                        'pose_guided' if ranked_override is not None else 'global',
            geometry=self.manifest.get('geometry_renderer', 'legacy-cpu-centers'),
            candidates=[], timings_ms={}, physical_accuracy_validated=False,
            excluded_face_matches=0)
        self.last_diagnostics['guided_optimized'] = optimized
        self.last_diagnostics['face_handoff'] = not allow_guided_early_stop
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if ranked_override is None:
            ranked, sift_keypoints = self._rank_views(image)
        else:
            ranked, sift_keypoints = ranked_override, 0
            self.last_retrieval = [dict(image_name=self.views[index]['image_name'],
                                        score=score, mode='fixed_views' if fixed_views else 'pose_guided')
                                   for score, _, index in ranked]
            if query_workspace is not None and 'crop_debug' in query_workspace:
                self.last_crop_debug = query_workspace['crop_debug']
                self.last_diagnostics['query_roi_reused'] = True
            elif self.retrieval_auto_roi:
                bbox, detection = self.retriever.detect_query(image)
                self.last_crop_debug = dict(bbox=bbox, detection=detection)
            else:
                self.last_crop_debug = dict(
                    bbox=self.retrieval_roi if self.retrieval_crop else None)
            if query_workspace is not None:
                query_workspace['crop_debug'] = self.last_crop_debug
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
        self.last_diagnostics['patch_rerank'] = self.last_crop_debug.get('patch_rerank')
        self._progress('features', roi=self.last_crop_debug.get('bbox'),
                       object_detection=self.last_diagnostics['retrieval_detection'],
                       retrieval_ms=sift_ms,
                       patch_rerank=self.last_crop_debug.get('patch_rerank'),
                       candidates=[dict(index=index, image_name=self.views[index]['image_name'],
                                        score=float(score)) for score, _, index in ranked],
                       candidate_total=len(ranked))
        if not ranked:
            return self._failure(
                image, started,
                f"{self.source_label}_NO_OBJECT_ROI" if self.retrieval_auto_roi
                else f"{self.source_label}_NO_REFERENCE", sift_ms=sift_ms
            )
        aliked_image = image
        aliked_offset = np.zeros(2, np.float32)
        if self.retrieval_crop:
            cropped = crop_live_object(image, self.last_crop_debug.get('bbox'))
            if cropped is None:
                return self._failure(image, started, f"{self.source_label}_NO_OBJECT_ROI",
                                     sift_ms=sift_ms)
            aliked_image, (x0, y0, x1, y1) = cropped
            aliked_offset[:] = (x0, y0)
            self.last_aliked_crop = aliked_image.copy()
            self.last_diagnostics['aliked_bbox_pixels'] = [x0, y0, x1, y1]
            self.last_diagnostics['aliked_input_size'] = [x1-x0, y1-y0]
        if (self.source_type == '3dgs' and self.pipeline_policy != 'legacy'
            and self.retrieval_crop):
            # Validate the detected/projected OBJECT, not a sharp background.
            quality = image_quality(aliked_image, (0., 0., 1., 1.))
            self.last_diagnostics['input_quality'] = quality
            self.last_diagnostics['photometric_mode'] = getattr(self, '_feature_variant', 'raw')
            self._progress('features', input_quality=quality)
            if not quality['usable']:
                return self._failure(image, started, f"{self.source_label}_IMAGE_QUALITY_HOLD",
                                     sift_ms=sift_ms)
            if getattr(self, '_feature_variant', 'raw') != 'raw':
                aliked_image = correct_luminance(aliked_image)
                self.last_aliked_crop = aliked_image.copy()
        normalized_object_matching = self._normalized_object_matching()
        if normalized_object_matching:
            self.last_diagnostics['object_scale_normalization'] = dict(
                enabled=True, geometry_side=OBJECT_GEOMETRY_SIDE,
                feature_resize=self._matching_resize(),
                query_crop_size=[aliked_image.shape[1], aliked_image.shape[0]],
                query_geometry_scale=OBJECT_GEOMETRY_SIDE / max(aliked_image.shape[:2]),
                fundamental_threshold=self.fundamental_threshold,
                pnp_threshold_unchanged=self.ransac_threshold,
                pose_qc_unchanged=self.max_reprojection_error)
        with self.torch.inference_mode():
            prepared = None if query_workspace is None else query_workspace.get('prepared')
            early = (optimized and getattr(self, 'guided_early_descriptors', True) and
                     all(hasattr(self.extractor, name) for name in
                         ('extract_dense_map', 'dkd', 'desc_head', 'preprocess_conf')))
            if prepared is not None or early:
                if prepared is None:
                    prepared = ALIKEDQuery(self.torch, self.extractor,
                        _image_tensor(self.torch, aliked_image, self.device), self._matching_resize())
                    if query_workspace is not None:
                        query_workspace['prepared'] = prepared
                else:
                    self.last_diagnostics['query_dense_reused'] = True
                previous_count = prepared.descriptor_points_computed
                query_features = prepared.features(self.guided_feature_budget if optimized else None)
                self.last_diagnostics['query_descriptor_points_computed'] = (
                    prepared.descriptor_points_computed-previous_count)
                self.last_diagnostics['query_extraction_stages_ms'] = prepared.timings_ms.copy()
            else:
                query_features = None if query_workspace is None else query_workspace.get('full_features')
                if query_features is None:
                    query_features = _extract(self.extractor,
                        _image_tensor(self.torch, aliked_image, self.device), self._matching_resize())
                    if query_workspace is not None:
                        query_workspace['full_features'] = query_features
                else:
                    self.last_diagnostics['query_features_reused'] = True
        query_feature_count = int(query_features['keypoints'].shape[1])
        if query_feature_count == 0:
            return self._failure(image, started, f"{self.source_label}_NO_OBJECT_FEATURES",
                                 sift_ms=sift_ms)
        query_points_all, query_scores = self._feature_arrays(query_features)
        self.last_diagnostics['query_features_raw'] = (prepared.count if prepared is not None
                                                      else query_feature_count)
        if optimized and prepared is None:
            chosen = spatial_indices(query_points_all, query_scores, self.guided_feature_budget)
            query_features = subset_features(self.torch, query_features, chosen)
            query_points_all = query_points_all[chosen]
            query_feature_count = len(chosen)
        self.last_diagnostics['query_features_selected'] = query_feature_count
        publish_query_points(self, image, query_points_all + aliked_offset)
        self.last_diagnostics['timings_ms']['query_features'] = (time.perf_counter()-sift_done)*1000
        self._progress('reference', query_features=query_feature_count,
                       features_ms=self.last_diagnostics['timings_ms']['query_features'])
        if prepared is not None:
            extraction = self.last_diagnostics['query_extraction_stages_ms']
            print(f'{self.source_label} ALIKED {self.device}: '
                  f'{prepared.count}->{query_feature_count} points, '
                  f'{self.last_diagnostics["query_descriptor_points_computed"]} new descriptors, '
                  f'extract {self.last_diagnostics["timings_ms"]["query_features"]:.0f}ms '
                  f'(dense/detect {extraction["dense_detection"]:.0f}ms, '
                  f'descriptors cumulative {extraction["descriptor"]:.0f}ms, '
                  f'reused {self.last_diagnostics.get("query_dense_reused", False)})', flush=True)
        total_pnp_ms = 0.0
        total_quality_ms = 0.0
        hypotheses = []
        transfer_groups = []
        candidate_log = []
        for candidate_number, (retrieval_score, _, view_index) in enumerate(ranked, 1):
            candidate_started = time.perf_counter()
            view = self.views[view_index]
            self._progress('reference', candidate_index=candidate_number,
                           reference_index=view_index, reference_name=view['image_name'],
                           raw_matches=None, geometric_inliers=None, correspondences=None)
            cache_hit = self._variant_key(view_index) in self.reference_feature_cache
            reference_features = self._reference_features(view_index)
            reference_selection = None
            predicted = None
            if optimized:
                reference_features, predicted, reference_selection = self._guided_reference_features(
                    view_index, reference_features, pose_prior, image, camera_matrix, distortion,
                    self.last_diagnostics.get('aliked_bbox_pixels'), query_points_all + aliked_offset)
            features_done = time.perf_counter()
            self._progress('matching')
            if reference_features['keypoints'].shape[1] >= 4:
                query_kp, reference_kp, pairs, scores = _match_pair(
                    self.torch, self.matcher, query_features, reference_features)
            else:
                query_kp = query_points_all
                reference_kp = np.empty((0, 2))
                pairs, scores = np.empty((0, 2), int), np.empty(0)
            raw_match_count = len(pairs)
            if optimized:
                keep = pair_gate(query_kp, pairs, predicted, self.guided_pixel_radius, aliked_offset)
                pairs, scores = pairs[keep], scores[keep]
            pairs, scores, excluded_count = exclude_virtual_query_polygon(
                pairs, scores, query_kp, aliked_offset, excluded_polygon)
            self.last_diagnostics['excluded_face_matches'] += excluded_count
            match_done = time.perf_counter()
            if len(pairs):
                query_points = query_kp[pairs[:, 0]]
                reference_points = reference_kp[pairs[:, 1]]
                if normalized_object_matching:
                    reference_shape = self.reference_crop_shapes[view_index]
                    query_points, reference_points, scores = object_scale_geometric_filter(
                        query_points, reference_points, scores, self.fundamental_threshold,
                        aliked_image.shape, reference_shape)
                    reference_points = reference_points + np.asarray(
                        self.reference_offsets[view_index], np.float32)
                else:
                    query_points, reference_points, scores = geometric_filter(
                        query_points + aliked_offset, reference_points, scores,
                        self.fundamental_threshold)
                if normalized_object_matching:
                    query_points = query_points + aliked_offset
            else:
                query_points = np.empty((0, 2), np.float32)
                reference_points = np.empty((0, 2), np.float32)
            geometry_done = time.perf_counter()
            self._progress('xyz', raw_matches=raw_match_count,
                           geometric_inliers=len(query_points))
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
            self._progress('pnp', correspondences=len(transfers))
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
                reference_crop_size=([self.reference_crop_shapes[view_index][1],
                                      self.reference_crop_shapes[view_index][0]]
                                     if normalized_object_matching else None),
                reference_geometry_scale=(OBJECT_GEOMETRY_SIDE /
                                          max(self.reference_crop_shapes[view_index][:2])
                                          if normalized_object_matching else None),
                reference_selection=reference_selection,
                raw_local_matches=raw_match_count,
                projection_rejected=raw_match_count-len(pairs),
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
            if optimized:
                print(f'{self.source_label} guided {view["image_name"]}: '
                      f'live {self.last_diagnostics["query_features_raw"]}->{query_feature_count}, '
                      f'ref {reference_selection["raw"]}->{reference_selection["selected"]}, '
                      f'projection gate {raw_match_count}->{len(pairs)} matches '
                      f'({self.guided_pixel_radius:g}px), '
                      f'LightGlue {(match_done-features_done)*1000:.0f}ms', flush=True)
            if pose is not None:
                hypotheses.append((inliers, len(transfers), pose, transfers, view))

            # Only a verified existing track supplies this strong prior.
            # Initial/global localization retains all ambiguity hypotheses.
            if optimized and allow_guided_early_stop and self._guided_strong_pose(
                    pose, transfers, quality, coarse_pose, pose_prior):
                self.last_diagnostics['early_stop'] = 'guided_strong_pose_and_3D_coverage'
                break

            # Global search needs >=3 candidates. Handoff already has a checked
            # current-pose prior and may stop after TWO agreeing supported views
            # (not one old face), leaving their transfers available for merging.
            consensus_minimum = 3 if allow_guided_early_stop else 2
            if (self.fast_consensus or not allow_guided_early_stop) and self.pipeline_policy != 'legacy' and len(candidate_log) >= consensus_minimum:
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
            groups = [('MERGED_LEGACY', transfer_groups, None,
                       [h[4]['image_name'] for h in hypotheses])]
        elif self.last_diagnostics.get('early_stop') != 'guided_strong_pose_and_3D_coverage':
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
                                   [h[3] for h in cluster], anchor[4],
                                   [h[4]['image_name'] for h in cluster]))
        for entry in groups:
            group_name, group = entry[:2]
            anchor_view = entry[2] if len(entry) > 2 else None
            source_view_names = entry[3] if len(entry) > 3 else []
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
                                                      'coarse_anchor': anchor_view,
                                                      'source_view_names': source_view_names}
                                   if anchor_view is not None else
                                   {'image_name': group_name,
                                    'source_view_names': source_view_names}))
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
                (self.last_diagnostics.get('early_stop') == 'guided_strong_pose_and_3D_coverage'
                 and self._guided_strong_pose(item[2], item[3],
                     self._pose_quality(item[2], item[3], image, camera_matrix, distortion),
                     self._coarse_pose_check(item[2], item[4]), pose_prior)),
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
        self._progress('qc')
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
        if pose_prior is not None:
            prior = SimpleNamespace(rvec=np.asarray(pose_prior[0]),
                                    tvec=np.asarray(pose_prior[1]))
            prior_angle, prior_distance = self._pose_distance(pose, prior)
            prior_ok = prior_angle <= 15. and prior_distance <= 30.
            self.last_diagnostics['pose_prior'] = dict(
                rotation_deg=prior_angle, translation=prior_distance,
                passed=prior_ok)
            if not prior_ok:
                accepted = False
                quality = dict(quality, passed=False,
                    reasons=list(quality['reasons']) + ['pose_prior_mismatch'])
        self.last_diagnostics.update(selected=view['image_name'],
                                     selected_reference_views=view.get('source_view_names',
                                                                       [view['image_name']]),
                                     quality=quality,
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
            image_points.copy(),
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
        self._progress('done', outcome='PASS' if accepted else 'FAIL',
                       inliers=inlier_count, reason='AMBIGUOUS' if ambiguity else ','.join(quality['reasons']),
                       total_ms=self.last_diagnostics['timings_ms']['total'])
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
