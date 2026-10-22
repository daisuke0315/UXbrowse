#!/usr/bin/env python3
"""Live 3DGS-view retrieval -> registered-photo ALIKED map -> PnP.

The 3DGS reference is appearance-only here. PnP XYZ comes exclusively from
multiview-triangulated ALIKED tracks in ``build_aliked_object_map.py``.
"""

from __future__ import annotations

from macar_paths import DATA_DIR

import json
import copy
import math
import os
import time
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from aliked_lightglue_pnp import Transfer, _extract, _image_tensor, _load_models, _match_pair, solve_pose
from glb_virtual_live import crop_live_object, GLBVirtualLiveLocalizer, OBJECT_GEOMETRY_SIDE
from glb_virtual_view_pnp import load_manifest
from offline_pnp_test import MapData, load_reference_images, quaternion_to_rotation
from realtime_pnp_server import LivePnPError, LiveResult, LocalizationDebug, TrackingSeed
from view_retrieval import ViewRetrievalIndex
from pose_landmark_matching import PoseLandmarkBank, match_projected_landmarks
from spatial_pose_scoring import spatial_pose_score, registered_hypothesis_rank, comparable_support
from guided_matching import region_counts


def rotation_angle(first, second):
    relative = first @ second.T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))


def reference_exit_decision(hypotheses, *, min_inliers, min_ratio, max_error,
                            max_coarse_angle, scene_scale, crop_area):
    """Conservative scheduling gate, NOT a replacement for final pose QC.

    Require two independently matched photos to give strong, agreeing poses.
    Thresholds are heuristics, not a calibrated probability of correctness.
    Camera-centre tolerance uses map units (5% of robust object diagonal).
    """
    strong, checks = [], {}
    for hypothesis in hypotheses:
        pose, transfers, quality, name, angle = hypothesis
        uv = np.asarray([t.query_uv for t in transfers], np.float32)[pose.inlier_indices]
        area = float(cv2.contourArea(cv2.convexHull(uv))) if len(uv) >= 3 else 0.
        required = dict(
            normal_qc=bool(quality['passed']),
            inliers=len(pose.inlier_indices) >= max(20, min_inliers),
            ratio=quality.get('inlier_ratio', 0.) >= max(.5, min_ratio),
            mean=0. <= (quality.get('mean_error') if quality.get('mean_error') is not None else math.inf) <= min(2., max_error),
            p95=0. <= (quality.get('p95_error') if quality.get('p95_error') is not None else math.inf) <= min(3., 1.5*max_error),
            spread=quality.get('spread_3d', 0.) >= .1,
            crop_coverage=crop_area > 0 and area/crop_area >= .1,
            coarse_angle=math.isfinite(angle) and angle <= min(20., max_coarse_angle))
        checks[name] = [key for key, passed in required.items() if not passed]
        if all(required.values()):
            strong.append(hypothesis)
    decision = dict(stop=False, reason='insufficient_strong_support', checks=checks,
                    supporting_references=[], max_center_delta=.05*scene_scale)
    if len(strong) < 2 or not math.isfinite(scene_scale) or scene_scale <= 0:
        return decision
    def difference(a, b):
        ra, rb = cv2.Rodrigues(a.rvec)[0], cv2.Rodrigues(b.rvec)[0]
        ca, cb = -ra.T @ a.tvec.reshape(3), -rb.T @ b.tvec.reshape(3)
        return rotation_angle(ra, rb), float(np.linalg.norm(ca-cb))
    strong.sort(key=registered_hypothesis_rank, reverse=True)
    best = strong[0]
    for other in hypotheses:
        if other is best or not other[2]['passed']:
            continue
        if comparable_support(other, best):
            angle, delta = difference(best[0], other[0])
            if not math.isfinite(delta) or angle > 15. or delta > .05*scene_scale:
                return dict(decision, reason='competing_pose', conflict_reference=other[3])
    for other in strong[1:]:
        angle, delta = difference(best[0], other[0])
        if other[3] != best[3] and angle <= 5. and delta <= .05*scene_scale:
            return dict(decision, stop=True, reason='two_strong_agreeing_photos',
                        supporting_references=[best[3], other[3]],
                        rotation_delta_deg=angle, center_delta=delta)
    return dict(decision, reason='strong_poses_disagree')


def exclude_query_polygon(pairs, scores, query_uv, ref_uv, polygon):
    """Remove 2D matches on one projected face before geometry/PnP."""
    if polygon is None or not len(pairs):
        return pairs, scores, query_uv, ref_uv, 0
    shape = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    if len(shape) < 3 or not np.isfinite(shape).all():
        raise ValueError('Invalid excluded face polygon')
    keep = np.asarray([cv2.pointPolygonTest(shape, tuple(float(v) for v in uv), False) < 0
                       for uv in query_uv], bool)
    return pairs[keep], scores[keep], query_uv[keep], ref_uv[keep], int((~keep).sum())


def make_registered_transfers(pairs, scores, query_uv, ref_uv, point_ids, xyz):
    """Keep full-reference feature indices until assigning triangulated XYZ.

    Untriangulated features provide LightGlue/geometric context, but cannot
    supply a PnP landmark. Never replace their XYZ with a nearby point/depth.
    """
    used_query, used_ids, transfers = set(), set(), []
    rejected = dict(missing_xyz=0, duplicate=0)
    for index in np.argsort(-scores):
        qi, ri = (int(value) for value in pairs[index])
        point_id = int(point_ids[ri])
        if point_id < 0 or not np.isfinite(xyz[ri]).all():
            rejected['missing_xyz'] += 1
            continue
        if qi in used_query or point_id in used_ids:
            rejected['duplicate'] += 1
            continue
        used_query.add(qi)
        used_ids.add(point_id)
        transfers.append(Transfer(qi, ri, point_id,
            tuple(float(v) for v in xyz[ri]),
            tuple(float(v) for v in query_uv[index]),
            tuple(float(v) for v in ref_uv[index]),
            tuple(float(v) for v in ref_uv[index]), 0., float(scores[index])))
    return transfers, rejected


def candidate_references(virtual_views, registered, object_center, top_views,
                         count, max_angle=55., per_view=2):
    """Select RC photos by orientation and object-centred viewing direction.

    Camera distance is deliberately omitted: virtual camera radius is arbitrary
    relative to the RC photographs. All rotations are COLMAP/OpenCV W2C.
    """
    selected = []
    seen = set()
    centre = np.asarray(object_center, np.float64)
    for _, _, view_index in top_views:
        view = virtual_views[view_index]
        matrix = np.asarray(view['world_to_camera'], np.float64)
        virtual_rotation = matrix[:3, :3]
        virtual_center = -virtual_rotation.T @ matrix[:3, 3]
        virtual_direction = virtual_center - centre
        virtual_direction /= max(np.linalg.norm(virtual_direction), 1e-12)
        ranked = []
        for name, item in registered.items():
            angular = rotation_angle(item['rotation'], virtual_rotation)
            direction = item['center'] - centre
            direction /= max(np.linalg.norm(direction), 1e-12)
            view_angle = math.degrees(math.acos(float(np.clip(
                np.dot(direction, virtual_direction), -1., 1.))))
            ranked.append((angular + .35 * view_angle, angular, name))
        ranked.sort()
        added = 0
        for score, angle, name in ranked:
            if name not in seen and angle <= max_angle:
                selected.append((name, view_index, float(score), float(angle)))
                seen.add(name)
                added += 1
                if added >= per_view:
                    break
        if len(selected) >= count * per_view:
            break
    # If several virtual views point to the same registered photo, fill the
    # shortlist from the strongest virtual view, still respecting angle QC.
    if top_views and len(selected) < count * per_view:
        view_index = top_views[0][2]
        matrix = np.asarray(virtual_views[view_index]['world_to_camera'], np.float64)
        ranked = sorted((rotation_angle(item['rotation'], matrix[:3, :3]), name)
                        for name, item in registered.items() if name not in seen)
        for angle, name in ranked:
            if angle > max_angle or len(selected) >= count * per_view:
                break
            selected.append((name, view_index, float(angle), float(angle)))
    return selected


class RegisteredALIKEDLiveLocalizer:
    """One synchronous search. Wrap with AsyncReferenceLocalizer for live UI."""

    source_label = 'RC-ALIKED'

    def __init__(self, map_path: Path, views_path: Path, *, top_k=3,
                 max_keypoints=2048, resize=1024, match_threshold=.01,
                 fundamental_threshold=5., ransac_threshold=6., iterations=10000,
                 min_inliers=10, min_inlier_ratio=.2, max_reprojection_error=6.,
                 max_coarse_rotation_deg=55., device='auto', mpr_renderer=None,
                 retrieval_mode='dino-patch', search_policy='adaptive'):
        if search_policy not in ('adaptive', 'exhaustive'):
            raise ValueError('Unknown registered reference search policy')
        self.search_policy = search_policy
        self.map_path = Path(map_path)
        self.views_path = Path(views_path)
        summary_path = self.map_path / 'summary.json'
        xyz_path = self.map_path / 'object_aliked_map.npz'
        if not summary_path.is_file() or not xyz_path.is_file():
            raise LivePnPError(f'Registered ALIKED map incomplete: {self.map_path}')
        with summary_path.open(encoding='utf-8') as source:
            summary = json.load(source)
        with np.load(xyz_path, allow_pickle=False) as archive:
            xyz = np.asarray(archive['xyz'], np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise LivePnPError('Invalid triangulated ALIKED XYZ')
        self.scene_scale = (float(np.linalg.norm(np.percentile(xyz, 95, axis=0) -
                                                 np.percentile(xyz, 5, axis=0))) if len(xyz) else 0.)
        self.map_data = MapData(np.arange(len(xyz), dtype=np.int64), xyz,
                                np.zeros((len(xyz), 128), np.float32),
                                np.zeros(len(xyz), np.int64))
        images_path = Path(summary['input_files']['images'])
        references = load_reference_images(images_path)
        feature_dir = self.map_path / 'reference_features'
        self.registered = {}
        for name, record in references.items():
            feature_path = feature_dir / (Path(name).stem + '.npz')
            if not feature_path.is_file():
                continue
            rotation = quaternion_to_rotation(np, record.pose)
            translation = np.asarray(record.pose[4:], np.float64)
            self.registered[name] = dict(path=feature_path, rotation=rotation,
                                         center=-rotation.T @ translation)
        if not self.registered:
            raise LivePnPError(f'No registered feature archives: {feature_dir}')
        self.manifest, self.views = load_manifest(self.views_path)
        if (self.manifest.get('source_type') != '3dgs' or
                self.manifest.get('runtime_transform') != 'identity'):
            raise LivePnPError('Registered ALIKED mode requires identity-coordinate 3DGS views')
        source_ply = Path(summary['input_files']['splat'])
        virtual_ply = Path(self.manifest['ply'])
        if (unicodedata.normalize('NFC', str(source_ply.resolve())) !=
                unicodedata.normalize('NFC', str(virtual_ply.resolve()))):
            raise LivePnPError('Registered ALIKED map and virtual views use different splat.ply files')
        os.environ.setdefault('TORCH_HOME', str(DATA_DIR / '.model_cache' / 'torch'))
        self.retriever = ViewRetrievalIndex(self.views_path, self.views,
                                            mode=retrieval_mode, crop=True,
                                            auto_roi=True, device=device)
        # Same appearance/retrieval and temporal contract as the GS route.
        # XYZ still comes ONLY from the registered, triangulated map.
        self.source_type = '3dgs'
        self.runtime_transform = 'identity'
        self.pipeline_policy = 'validated'
        self.retrieval_mode = retrieval_mode
        self.excluded_retrieval_views = set()
        self.lock = threading.RLock()
        self.progress_lock = threading.Lock()
        self.progress = {}
        self.last_retrieval = []
        self.last_tracking_seed = None
        self.last_aliked_crop = None
        self.torch, self.extractor, self.matcher, self.device = _load_models(
            max_keypoints, match_threshold, device)
        self.reference_cache = OrderedDict()
        self.real_appearance = None
        self.landmark_bank = None
        self.top_k = min(int(top_k), len(self.views))
        self.resize = resize
        self.fundamental_threshold = fundamental_threshold
        self.ransac_threshold = ransac_threshold
        self.iterations = iterations
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reprojection_error = max_reprojection_error
        self.max_coarse_rotation_deg = max_coarse_rotation_deg
        self.mpr_renderer = mpr_renderer
        self.last_debug = LocalizationDebug(np.empty((0, 3), np.float64), frozenset(), None, False)
        self.last_crop_debug = {}
        self.last_reference_name = '—'
        self.last_diagnostics = {}
        print(f'Registered ALIKED live: {len(xyz)} landmarks, '
              f'{len(self.registered)} RC views, {len(self.views)} 3DGS coarse views, '
              f'top-{self.top_k} virtual / up to {self.top_k*2} RC, '
              f'references=gs-real, search={self.search_policy}, device={self.device}')

    def _appearance_references(self, image, ranked, workspace):
        from real_appearance_retrieval import RegisteredAppearanceIndex
        from view_retrieval import crop_box
        from patch_view_rerank import square_mask
        self._progress('reference', reference_selection='gs-real')
        if self.real_appearance is None:
            self.real_appearance = RegisteredAppearanceIndex(self.map_path, self.registered,
                self.retriever.encoder, progress=lambda **values: self._progress('reference', **values))
        if 'appearance_query' not in workspace:
            box = self.last_crop_debug['bbox']
            crop = crop_box(image,box)
            vector, patches = self.real_appearance.encoder.encode_tokens(crop)
            mask = (self.last_crop_debug.get('detection') or {}).get('mask')
            if mask is None:
                mask = np.ones(crop.shape[:2],np.uint8)
            else:
                mask = cv2.resize(mask.astype(np.uint8),(image.shape[1],image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                mask = crop_box(mask,box)
            workspace['appearance_query'] = (vector, patches, square_mask(mask))
        return self.real_appearance.select(self.views,self.registered,ranked,
            workspace['appearance_query'],budget=self.top_k*2,max_angle=self.max_coarse_rotation_deg)

    def _reference(self, name):
        cached = self.reference_cache.get(name)
        if cached is not None:
            self.reference_cache.move_to_end(name)
            return cached
        with np.load(self.registered[name]['path'], allow_pickle=False) as archive:
            ids = np.asarray(archive['point_ids'], np.int64)
            xyz = np.asarray(archive['xyz'], np.float64)
            tensor = {key: self.torch.from_numpy(archive[key].copy()).to(self.device)
                      for key in ('keypoints', 'descriptors', 'keypoint_scores', 'image_size')}
            # All saved features are already object-masked. Retain their
            # original indices and context, including features without XYZ.
            output = (tensor, ids, xyz)
        self.reference_cache[name] = output
        if len(self.reference_cache) > 24:
            self.reference_cache.popitem(last=False)
        return output

    def _candidate_diagnostic(self):
        sink = getattr(self, 'diagnostic_sink', None)
        if sink is not None:
            sink('reference_candidate', **getattr(self, 'diagnostic_context', {}),
                 candidate=self.last_diagnostics['candidates'][-1])

    def _fail(self, image, started, detail, keypoints=0, matches=0, retrieval_ms=0.,
              matching_ms=0., pnp_ms=0., xyz=None):
        self.last_diagnostics.update(status='FAIL', reason=detail,
                                     matching_ms=matching_ms)
        self.last_tracking_seed = None
        self.last_debug = LocalizationDebug(
            np.empty((0, 3), np.float64) if xyz is None else np.asarray(xyz, np.float64),
            frozenset(), None, False)
        result = LiveResult('FAIL', detail, keypoints, matches, 0, 0., None,
                            None, None, None, (time.perf_counter()-started)*1000,
                            'HIDDEN', retrieval_ms, matching_ms, pnp_ms, 0.)
        self._progress('done', outcome='FAIL', reason=detail, inliers=0,
                       total_ms=result.processing_ms)
        return result, self._decorate(image, result, [], frozenset())

    @staticmethod
    def _decorate(image, result, transfers, inliers):
        canvas = image.copy()
        for index, item in enumerate(transfers):
            uv = tuple(int(round(v)) for v in item.query_uv)
            cv2.circle(canvas, uv, 4, (50, 230, 70) if index in inliers else (40, 80, 230), -1)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        color = (50, 230, 70) if result.status == 'PASS' else (0, 190, 255)
        cv2.putText(canvas, f'{result.status}  RC photo ALIKED  {result.matches} pairs / '
                    f'{result.inliers} inliers', (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, result.detail[:95], (14, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas

    _progress = GLBVirtualLiveLocalizer._progress
    progress_snapshot = GLBVirtualLiveLocalizer.progress_snapshot
    _pose_guided_views = GLBVirtualLiveLocalizer._pose_guided_views
    _handoff_views = GLBVirtualLiveLocalizer._handoff_views
    validate_pose = GLBVirtualLiveLocalizer.validate_pose

    def localize(self, image, camera_matrix, distortion, *, excluded_polygon=None):
        return self._search(image, camera_matrix, distortion, excluded_polygon=excluded_polygon)

    def localize_validation(self, image, camera_matrix, distortion, polygon,
                            max_mean_error_px, min_coverage, reference_names, baseline_pose):
        """Frozen comparison uses original references and the accepted camera.

        A discrete GS view is a retrieval seed, not the ground-truth camera
        orientation. Do not compare a reduced, often planar support set with a
        newly retrieved GS view. Preserve all live QC settings and seed state.
        """
        with self.lock:
            absent = object()
            saved = {key: getattr(self, key, absent) for key in (
                'last_tracking_seed', 'last_debug', 'last_diagnostics',
                'last_reference_name', 'last_retrieval', 'last_crop_debug',
                'last_aliked_crop', 'max_reprojection_error', 'min_coverage')}
            progress = self.progress_snapshot()
            try:
                self.max_reprojection_error = float(max_mean_error_px)
                self.min_coverage = float(min_coverage)
                if tuple(reference_names or ()) == ('POSE_LANDMARKS',):
                    answer = self._localize_landmarks(image, camera_matrix, distortion,
                                                      *baseline_pose, excluded_polygon=polygon)
                    return answer[0], self.last_debug, copy.deepcopy(self.last_diagnostics)
                workspace = {'validation_pose': baseline_pose}
                if reference_names:
                    missing = set(reference_names) - self.registered.keys()
                    if missing:
                        raise ValueError(f'Original RC references unavailable: {sorted(missing)}')
                    # Only the original actual photo references; no DINO rerank.
                    indices = self._pose_guided_views(*baseline_pose, top_k=1)
                    if not indices:
                        indices = [0]
                    workspace.update(ranked_override=[(1., 0., indices[0])],
                        references=[(name, indices[0], 0., 0.)
                                    for name in dict.fromkeys(reference_names)])
                answer = self._search_attempts(image, camera_matrix, distortion,
                                                polygon, workspace)
                return answer[0], self.last_debug, copy.deepcopy(self.last_diagnostics)
            finally:
                for key, value in saved.items():
                    if value is absent:
                        if hasattr(self, key):
                            delattr(self, key)
                    else:
                        setattr(self, key, value)
                with self.progress_lock:
                    self.progress = progress

    def localize_guided(self, image, camera_matrix, distortion, rvec, tvec,
                        view_prediction=None):
        # The prior belongs to THIS search image, not the browser display's
        # possibly later pose. No DINO search or RC-photo shortlist here.
        with self.lock:
            from background_refresh import VIEW_REFRESH_REASONS
            now = time.monotonic()
            context = getattr(self, 'diagnostic_context', {})
            new_view = bool(VIEW_REFRESH_REASONS.intersection(context.get('trigger', [])))
            if (context.get('search_mode') == 'guided'
                    and (new_view or now-getattr(self, 'last_context_started', -math.inf) >= 4.)):
                self.last_context_started = now
                return self._localize_pose_references(image, camera_matrix, distortion, rvec, tvec)
            return self._localize_landmarks(image, camera_matrix, distortion, rvec, tvec)

    def _localize_pose_references(self, image, camera_matrix, distortion, rvec, tvec):
        """Recover sparse regions using coherent photo context, not DINO search."""
        from guided_matching import project_candidates
        started = time.monotonic()
        timed = time.perf_counter()
        self.last_tracking_seed = None
        self.last_diagnostics = dict(reference_matching='pose_references', candidates=[])
        self._progress('projection', started_at=started, matching_method='pose_references',
                       candidate_index=0, candidate_total=2, retry=False)
        projected, valid = project_candidates(self.map_data.xyz, rvec, tvec,
                                              camera_matrix, distortion, image.shape)
        indices = self._pose_guided_views(rvec, tvec, top_k=1)
        if valid.sum() < 4 or not indices:
            return self._fail(image, timed, 'NO_POSE_LOCAL_LANDMARKS')
        low, high = projected[valid].min(axis=0), projected[valid].max(axis=0)
        height, width = image.shape[:2]
        bbox = [low[0]/width, low[1]/height, (high[0]-low[0])/width, (high[1]-low[1])/height]
        transform = np.eye(4)
        transform[:3,:3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
        transform[:3,3] = np.asarray(tvec).reshape(3)
        references = candidate_references([dict(world_to_camera=transform)], self.registered,
            self.manifest['center'], [(1.,0.,0)], 1, max_angle=self.max_coarse_rotation_deg)
        references = [(name, indices[0], score, angle) for name, _, score, angle in references]
        workspace = dict(retrieval=([(1.,0.,indices[0])], 0, [], dict(bbox=bbox, mode='pose_projection')),
            references=references, reference_selection=dict(mode='pose_geometry', count=len(references)),
            pose_prior=(rvec,tvec), reference_matching='pose_references')
        result, canvas = self._search_attempts(image, camera_matrix, distortion, None, workspace)
        self.last_diagnostics['reference_matching'] = 'pose_references'
        self._progress('done', outcome=result.status, reason=result.detail,
                       inliers=result.inliers, total_ms=result.processing_ms)
        return result, canvas

    def _check_search_pose(self, quality, rvec, tvec, prior):
        rotation = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
        previous = cv2.Rodrigues(np.asarray(prior[0], np.float64))[0]
        angle = rotation_angle(rotation, previous)
        center = -rotation.T @ np.asarray(tvec).reshape(3)
        prior_center = -previous.T @ np.asarray(prior[1]).reshape(3)
        distance = float(np.linalg.norm(center-prior_center))
        maximum = .2*float(np.linalg.norm(np.ptp(self.map_data.xyz, axis=0)))
        if angle > 15. or distance > maximum:
            quality['passed'] = False
            quality['reasons'].append('pose_prior_mismatch')
        quality['pose_comparison'] = dict(reference='search_frame_pose', rotation_deg=angle,
            max_rotation_deg=15., center_distance=distance, max_center_distance=maximum)

    def _localize_landmarks(self, image, camera_matrix, distortion, rvec, tvec,
                            *, excluded_polygon=None):
        started = time.perf_counter()
        self.last_tracking_seed = None
        self.last_reference_name = 'POSE_LANDMARKS'
        self.last_diagnostics = dict(reference_matching='pose_landmarks', candidates=[],
            selected='POSE_LANDMARKS', selected_reference_views=['POSE_LANDMARKS'],
            excluded_face_matches=0, visibility='positive_depth_bounds_observed_view_cone')
        self._progress('projection', started_at=time.monotonic(), candidates=[],
            reference_index=None, reference_name=None, registered_reference_name=None,
            candidate_index=0, candidate_total=0, query_features=None, raw_matches=None,
            geometric_inliers=None, correspondences=None, outcome=None, reason=None,
            retry=False, matching_method='pose_landmarks', bank_loaded=None, bank_total=None)
        if getattr(self, 'landmark_bank', None) is None:
            self._progress('landmark_bank', bank_loaded=0, bank_total=len(self.registered))
            self.landmark_bank = PoseLandmarkBank.from_registered(
                self.map_data.xyz, self.registered,
                lambda n, total: self._progress('landmark_bank', bank_loaded=n, bank_total=total))
        selection_started = time.perf_counter()
        candidates = self.landmark_bank.candidates(rvec, tvec, camera_matrix, distortion,
                                                   image.shape)
        candidate_count = len(np.unique(candidates.point_ids))
        self.last_diagnostics.update(bank_observations=len(self.landmark_bank.ids),
                                     projected_landmarks=candidate_count)
        self._progress('projection', projected_landmarks=candidate_count)
        if candidate_count < 4:
            return self._fail(image, started, 'NO_POSE_LOCAL_LANDMARKS')
        low, high = candidates.projected.min(axis=0), candidates.projected.max(axis=0)
        h, w = image.shape[:2]
        bbox = (low[0]/w, low[1]/h, (high[0]-low[0])/w, (high[1]-low[1])/h)
        cropped = crop_live_object(image, bbox, padding=.12)
        self.last_crop_debug = dict(bbox=list(bbox), mode='pose_projection')
        if cropped is None:
            return self._fail(image, started, 'NO_POSE_LOCAL_ROI')
        crop, (x0, y0, _, _) = cropped
        self.last_aliked_crop = crop
        projection_ms = (time.perf_counter()-selection_started)*1000
        self._progress('features', roi=list(bbox))
        features_started = time.perf_counter()
        with self.torch.inference_mode():
            query = _extract(self.extractor, _image_tensor(self.torch, crop, self.device),
                             getattr(self, 'query_resize', self.resize))
        features_ms = (time.perf_counter()-features_started)*1000
        pixels = query['keypoints'][0].detach().cpu().numpy() + np.asarray([x0, y0])
        from concurrent_query_tracking import publish_query_points
        publish_query_points(self, image, pixels)
        descriptors = query['descriptors'][0].detach().cpu().numpy()
        # 40 px at a 1024-px object crop, not a fixed threshold on 24MP photos.
        radius = 40.*max(crop.shape[:2])/OBJECT_GEOMETRY_SIDE
        self._progress('landmark_matching', query_features=len(pixels), search_radius_px=radius)
        matching_started = time.perf_counter()
        pairs, scores, matching = match_projected_landmarks(pixels, descriptors, candidates,
                                                           radius=radius)
        query_uv = pixels[pairs[:, 0]]
        hints = candidates.projected[pairs[:, 1]]
        pairs, scores, query_uv, hints, excluded = exclude_query_polygon(
            pairs, scores, query_uv, hints, excluded_polygon)
        transfers, rejected = make_registered_transfers(pairs, scores, query_uv, hints,
                                                       candidates.point_ids, candidates.xyz)
        matching_ms = (time.perf_counter()-matching_started)*1000
        self.last_diagnostics.update(query_features_raw=len(pixels), matching=matching,
            search_radius_px=radius, excluded_face_matches=excluded,
            regions=dict(grid=4,bbox=[x0,y0,x0+crop.shape[1],y0+crop.shape[0]],
                query=region_counts(pixels,(x0,y0,x0+crop.shape[1],y0+crop.shape[0])),
                matched=region_counts(query_uv,(x0,y0,x0+crop.shape[1],y0+crop.shape[0]))),
            timings_ms=dict(projection=projection_ms, query_features=features_ms,
                            matching=matching_ms))
        self.last_diagnostics['candidates'] = [dict(name='POSE_LANDMARKS',
            reference_features=len(candidates.point_ids), reference_xyz_features=len(candidates.point_ids),
            unique_landmarks=candidate_count,
            raw=matching['descriptor_pass'], geometric=len(pairs), local_matches=len(pairs),
            transferred=len(transfers), xyz_rejections=rejected)]
        self._progress('pnp', raw_matches=matching['descriptor_pass'],
                       geometric_inliers=len(pairs), correspondences=len(transfers))
        pnp_started = time.perf_counter()
        # Independent RANSAC/LM, without initial rvec/tvec. Only descriptor
        # observations (not predicted pixels) reach the unchanged solver.
        pose = solve_pose(transfers, camera_matrix, distortion,
                          self.ransac_threshold, self.iterations)
        pnp_ms = (time.perf_counter()-pnp_started)*1000
        if pose is None:
            result, canvas = self._fail(image, started, 'NO_POSE_LOCAL_PNP', len(pixels),
                len(transfers), features_ms, matching_ms, pnp_ms)
            self._progress('done', outcome=result.status, reason=result.detail,
                           total_ms=result.processing_ms)
            return result, canvas
        xyz = np.asarray([item.xyz for item in transfers], np.float64)
        uv = np.asarray([item.query_uv for item in transfers], np.float64)
        self._progress('qc')
        quality = self.validate_pose(xyz, uv, pose.rvec, pose.tvec, pose.inlier_indices,
                                     image.shape, camera_matrix, distortion)
        rotation = cv2.Rodrigues(pose.rvec)[0]
        center = -rotation.T @ pose.tvec.reshape(3)
        self._check_search_pose(quality, pose.rvec, pose.tvec, (rvec,tvec))
        accepted = quality['passed']
        self.last_debug = LocalizationDebug(xyz, frozenset(map(int, pose.inlier_indices)),
            tuple(float(v) for v in center), True, uv)
        if accepted:
            self.last_tracking_seed = TrackingSeed(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                uv[pose.inlier_indices].copy(), xyz[pose.inlier_indices].copy(),
                pose.rvec.copy(), pose.tvec.copy())
        self.last_diagnostics.update(quality=quality, status='PASS' if accepted else 'FAIL_QC')
        self.last_diagnostics['regions']['pnp'] = region_counts(uv[pose.inlier_indices],
                                                               self.last_diagnostics['regions']['bbox'])
        self.last_diagnostics['candidates'][0].update(inliers=len(pose.inlier_indices), quality=quality)
        result = LiveResult('PASS' if accepted else 'FAIL',
            f'pose-local landmarks / {candidate_count} candidates / '
            f'{"QC OK" if accepted else ",".join(quality["reasons"])}', len(pixels),
            len(transfers), len(pose.inlier_indices), len(pose.inlier_indices)/max(1, len(transfers)),
            pose.mean_error, tuple(float(v) for v in pose.rvec.ravel()),
            tuple(float(v) for v in pose.tvec.ravel()), tuple(float(v) for v in center),
            (time.perf_counter()-started)*1000, 'HIDDEN', features_ms, matching_ms, pnp_ms)
        self._progress('done', outcome=result.status, reason=result.detail,
                       inliers=result.inliers, total_ms=result.processing_ms)
        print(f'RC pose-landmarks: {candidate_count} projected / {len(pixels)} query / '
              f'{len(transfers)} pairs / {result.inliers} inliers / {result.status} / '
              f'{result.processing_ms:.0f} ms', flush=True)
        return result, self._decorate(image, result, transfers, frozenset(map(int, pose.inlier_indices)))

    def localize_nearby(self, image, camera_matrix, distortion, rvec, tvec):
        indices = self._pose_guided_views(rvec, tvec, self.top_k)
        return self._search(image, camera_matrix, distortion,
                            ranked_override=[(1., 0., i) for i in indices] if indices else None)

    def _search(self, image, camera_matrix, distortion, *, excluded_polygon=None,
                ranked_override=None):
        # Frozen validation and background matching must not mutate one engine
        # concurrently. Progress uses a separate short-lived lock.
        if not hasattr(self, 'lock'):
            self.lock = threading.RLock()
        with self.lock:
            self._progress('retrieval', started_at=time.monotonic(),
                           candidates=[], candidate_index=0, candidate_total=0,
                           query_features=None, geometric_inliers=None, correspondences=None,
                           outcome=None, reason=None, retry=False,
                           reference_search=None,
                           matching_method='lightglue', projected_landmarks=None)
            workspace = {} if ranked_override is None else {'ranked_override': ranked_override}
            answer = self._search_attempts(image, camera_matrix, distortion,
                                            excluded_polygon, workspace)
            result = answer[0]
            self._progress('done', outcome=result.status, reason=result.detail,
                           inliers=result.inliers, total_ms=result.processing_ms)
            return answer

    def _search_attempts(self, image, camera_matrix, distortion, excluded_polygon, workspace):
        result, canvas = self._localize_once(image, camera_matrix, distortion,
            excluded_polygon=excluded_polygon, workspace=workspace)
        if result.status == 'PASS':
            return result, canvas
        # Full-context matching can prefer correct but untriangulated pixels
        # over existing landmarks. Retry the original landmark-only search
        # on the SAME query features, ROI and reference shortlist.
        diagnostics = copy.deepcopy(self.last_diagnostics)
        support_limited = (result.inliers < self.min_inliers or
                           diagnostics.get('reason') == 'NO_PNP_HYPOTHESIS')
        if not support_limited:
            return result, canvas
        if not any((c.get('xyz_rejections') or {}).get('missing_xyz', 0)
                   for c in diagnostics.get('candidates', ())):
            return result, canvas
        full_debug, full_reference = self.last_debug, self.last_reference_name
        full_seed = getattr(self, 'last_tracking_seed', None)
        self._progress('matching', retry=True, candidate_index=0, reference_search=None)
        fallback, fallback_canvas = self._localize_once(image, camera_matrix, distortion,
            excluded_polygon=excluded_polygon, workspace=workspace, xyz_only=True)
        fallback_diagnostics = copy.deepcopy(self.last_diagnostics)
        if fallback.status == 'PASS' or fallback.inliers > result.inliers:
            chosen, chosen_canvas = fallback, fallback_canvas
        else:
            chosen, chosen_canvas = result, canvas
            self.last_diagnostics = copy.deepcopy(diagnostics)
            self.last_debug, self.last_reference_name = full_debug, full_reference
            self.last_tracking_seed = full_seed
        self.last_diagnostics['all_feature_attempt'] = diagnostics
        self.last_diagnostics['all_feature_result'] = dict(
            status=result.status, inliers=result.inliers, matches=result.matches)
        self.last_diagnostics['xyz_only_fallback'] = dict(
            status=fallback.status, inliers=fallback.inliers,
            selected=fallback_diagnostics.get('selected'),
            adopted=chosen is fallback)
        print(f'RC-ALIKED fallback: all-feature {result.status}/{result.inliers} -> '
              f'XYZ-only {fallback.status}/{fallback.inliers}', flush=True)
        return replace(chosen, processing_ms=result.processing_ms+fallback.processing_ms,
            sift_ms=result.sift_ms+fallback.sift_ms,
            matching_ms=result.matching_ms+fallback.matching_ms,
            pnp_ms=result.pnp_ms+fallback.pnp_ms), chosen_canvas

    def _localize_once(self, image, camera_matrix, distortion, *,
                       excluded_polygon=None, workspace=None, xyz_only=False):
        workspace = {} if workspace is None else workspace
        started = time.perf_counter()
        self.last_reference_name = '—'
        self.last_diagnostics = {'candidates': [], 'excluded_face_matches': 0,
                                 'reference_matching': 'xyz_only_fallback' if xyz_only else 'all_object_features'}
        if 'retrieval' not in workspace:
            if 'ranked_override' in workspace:
                bbox, detection = self.retriever.detect_query(image)
                workspace['retrieval'] = (workspace['ranked_override'], 0, [],
                                          dict(bbox=bbox, detection=detection))
            else:
                workspace['retrieval'] = self.retriever.rank_debug(image, top_k=self.top_k,
                                                                  capture_query_tokens=True)
                tokens = workspace['retrieval'][3].pop('_query_tokens',None)
                if tokens is not None:
                    workspace['appearance_query'] = tokens
        ranked, _, _, self.last_crop_debug = workspace['retrieval']
        self.last_diagnostics['object_detection'] = {
            key: value for key, value in (self.last_crop_debug.get('detection') or {}).items()
            if key != 'mask'}
        self.last_retrieval = [dict(image_name=self.views[i]['image_name'], score=float(score),
                                   mode=self.retrieval_mode) for score, _, i in ranked]
        self._progress('features', roi=self.last_crop_debug.get('bbox'),
                       object_detection=self.last_diagnostics['object_detection'],
                       candidates=[dict(index=i, image_name=self.views[i]['image_name'], score=float(score))
                                   for score, _, i in ranked])
        retrieval_ms = (time.perf_counter()-started)*1000
        self.last_diagnostics['virtual_views'] = [self.views[i]['image_name']
                                                  for _, _, i in ranked]
        self.last_diagnostics['retrieval_ms'] = retrieval_ms
        if not ranked:
            return self._fail(image, started, 'NO_OBJECT_ROI', retrieval_ms=retrieval_ms)
        cropped = crop_live_object(image, self.last_crop_debug.get('bbox'))
        if cropped is None:
            return self._fail(image, started, 'NO_OBJECT_ROI', retrieval_ms=retrieval_ms)
        crop, (x0, y0, x1, y1) = cropped
        self.last_aliked_crop = crop
        feature_started = time.perf_counter()
        reused_query = 'query' in workspace
        if 'query' not in workspace:
            with self.torch.inference_mode():
                workspace['query'] = _extract(self.extractor,
                    _image_tensor(self.torch, crop, self.device),
                    getattr(self, 'query_resize', self.resize))
        query = workspace['query']
        keypoint_count = int(query['keypoints'].shape[1])
        self.last_diagnostics['timings_ms'] = dict(retrieval=retrieval_ms,
            query_features=(time.perf_counter()-feature_started)*1000,
            query_reused=reused_query)
        self._progress('reference', query_features=keypoint_count)
        self.last_diagnostics['query_features_raw'] = keypoint_count
        self.last_diagnostics['regions'] = dict(grid=4,bbox=[x0,y0,x1,y1],
            query=region_counts(query['keypoints'][0].detach().cpu().numpy()+[x0,y0],(x0,y0,x1,y1)))
        if keypoint_count < 4:
            return self._fail(image, started, 'NO_OBJECT_FEATURES', keypoint_count,
                              retrieval_ms=retrieval_ms)
        if (getattr(self, 'query_tracking_callback', None) is not None and
                not workspace.get('tracking_published')):
            from concurrent_query_tracking import publish_query_points
            publish_query_points(self, image,
                query['keypoints'][0].detach().cpu().numpy() + np.asarray((x0, y0), np.float32))
            workspace['tracking_published'] = True
        matches_started = time.perf_counter()
        if 'references' not in workspace:
            selection_started = time.perf_counter()
            workspace['references'] = self._appearance_references(image,ranked,workspace)
            workspace['reference_selection'] = dict(mode='gs-real',
                elapsed_ms=(time.perf_counter()-selection_started)*1000)
        self.last_diagnostics['reference_selection'] = workspace.get('reference_selection',
            dict(mode='fixed'))
        references = workspace['references']
        if not references:
            return self._fail(image, started, 'NO_POSE_NEAR_RC_PHOTO', keypoint_count,
                              retrieval_ms=retrieval_ms)
        hypotheses = []
        # Frozen face comparisons always retain the complete original shortlist.
        policy = getattr(self, 'search_policy', 'adaptive')
        validation = excluded_polygon is not None or 'validation_pose' in workspace
        context = 'pose_prior' in workspace
        effective_policy = 'exhaustive' if validation or context else policy
        decision = dict(stop=False, reason='pose_context_full_search' if context else 'validation_full_search' if validation else
                        'exhaustive_policy' if policy == 'exhaustive' else 'insufficient_strong_support')
        checkpoints = []
        self._progress('reference', search_policy=effective_policy)
        pnp_ms = 0.0
        for number, (name, virtual_index, _, coarse_angle) in enumerate(references, 1):
            self._progress('reference', candidate_index=number, candidate_total=len(references),
                           reference_index=virtual_index, reference_name=self.views[virtual_index]['image_name'],
                           registered_reference_name=name,
                           geometric_inliers=0, correspondences=0)
            reference_started = time.perf_counter()
            cached_reference = name in getattr(self, 'reference_cache', {})
            features, ids, xyz = self._reference(name)
            load_ms = (time.perf_counter()-reference_started)*1000
            if xyz_only:
                keep = np.flatnonzero((ids >= 0) & np.isfinite(xyz).all(axis=1))
                indices = self.torch.as_tensor(keep, dtype=self.torch.long, device=self.device)
                features = {key: value.index_select(1, indices)
                    if key in ('keypoints', 'descriptors', 'keypoint_scores') else value
                    for key, value in features.items()}
                ids, xyz = ids[keep], xyz[keep]
            stage = dict(name=name, reference_features=len(ids),
                         reference_xyz_features=int(((ids >= 0) & np.isfinite(xyz).all(axis=1)).sum()),
                         coarse_rotation_deg=coarse_angle,
                         reference_cached=cached_reference, timings_ms=dict(reference_load=load_ms))
            if len(ids) < 4:
                self.last_diagnostics['candidates'].append(dict(stage, raw=0, geometric=0))
                self._candidate_diagnostic()
                continue
            self._progress('matching')
            pair_started = time.perf_counter()
            query_kp, _, pairs, scores = _match_pair(self.torch, self.matcher,
                                                      query, features)
            stage['timings_ms']['lightglue'] = (time.perf_counter()-pair_started)*1000
            raw = len(pairs)
            if raw < 4:
                self.last_diagnostics['candidates'].append(dict(stage, raw=raw, geometric=0))
                self._candidate_diagnostic()
                continue
            query_uv = query_kp[pairs[:, 0]] + np.asarray((x0, y0), np.float32)
            with np.errstate(invalid='ignore'):
                ref_uv = features['keypoints'][0].detach().cpu().numpy()[pairs[:, 1]]
            pairs, scores, query_uv, ref_uv, excluded = exclude_query_polygon(
                pairs, scores, query_uv, ref_uv, excluded_polygon)
            self.last_diagnostics['excluded_face_matches'] += excluded
            after_face = len(pairs)
            if len(pairs) < 4:
                self.last_diagnostics['candidates'].append(dict(
                    stage, raw=raw, excluded=excluded, local_matches=after_face, geometric=0))
                self._candidate_diagnostic()
                continue
            geometry_started = time.perf_counter()
            if len(pairs) >= 8:
                try:
                    # Match geometry in the same 1024px object-crop space as GS,
                    # not mixed full-photo and reference pixel resolutions.
                    ref_size = features['image_size'][0].detach().cpu().numpy()
                    _, mask = cv2.findFundamentalMat(
                        (query_uv-np.asarray((x0, y0))) * OBJECT_GEOMETRY_SIDE/max(crop.shape[:2]),
                        ref_uv.astype(np.float64) * OBJECT_GEOMETRY_SIDE/max(ref_size), cv2.FM_RANSAC,
                        self.fundamental_threshold, .999, 10000)
                except cv2.error:
                    mask = None
                keep = np.zeros(len(pairs), bool) if mask is None else mask.ravel().astype(bool)
            else:
                keep = np.ones(len(pairs), bool)
            pairs, scores, query_uv, ref_uv = (value[keep] for value in
                                               (pairs, scores, query_uv, ref_uv))
            stage['timings_ms']['geometry'] = (time.perf_counter()-geometry_started)*1000
            self._progress('xyz', raw_matches=raw, geometric_inliers=int(keep.sum()))
            xyz_started = time.perf_counter()
            transfers, rejected_xyz = make_registered_transfers(
                pairs, scores, query_uv, ref_uv, ids, xyz)
            stage['timings_ms']['xyz_assignment'] = (time.perf_counter()-xyz_started)*1000
            self.last_diagnostics['candidates'].append(dict(stage, raw=raw,
                excluded=excluded, local_matches=after_face,
                geometric=int(keep.sum()), transferred=len(transfers),
                xyz_rejections=rejected_xyz))
            pnp_started = time.perf_counter()
            self._progress('pnp', correspondences=len(transfers))
            pose = solve_pose(transfers, camera_matrix, distortion,
                              self.ransac_threshold, self.iterations)
            candidate_pnp_ms = (time.perf_counter()-pnp_started)*1000
            stage['timings_ms']['pnp'] = candidate_pnp_ms
            pnp_ms += candidate_pnp_ms
            if pose is not None:
                qc_started = time.perf_counter()
                points = np.asarray([item.xyz for item in transfers], np.float64)
                pixels = np.asarray([item.query_uv for item in transfers], np.float64)
                quality = self.validate_pose(points, pixels, pose.rvec, pose.tvec,
                    pose.inlier_indices, image.shape, camera_matrix, distortion)
                projected = cv2.projectPoints(points, pose.rvec, pose.tvec,
                    camera_matrix, distortion)[0].reshape(-1, 2)
                indices = pose.inlier_indices
                scoring_started = time.perf_counter()
                quality['spatial_score'] = spatial_pose_score(pixels[indices],
                    np.linalg.norm(projected[indices]-pixels[indices], axis=1),
                    (x0, y0, x1, y1), self.ransac_threshold)
                stage['timings_ms']['spatial_scoring'] = (time.perf_counter()-scoring_started)*1000
                virtual_rotation = np.asarray(self.views[virtual_index]['world_to_camera'],
                                              np.float64)[:3, :3]
                comparison_pose = workspace.get('validation_pose')
                comparison_rotation = (virtual_rotation if comparison_pose is None else
                    cv2.Rodrigues(np.asarray(comparison_pose[0], np.float64))[0])
                pose_angle = rotation_angle(cv2.Rodrigues(pose.rvec)[0], comparison_rotation)
                if context:
                    # The continuous search-image pose is the comparison, not
                    # the arbitrary nearest discrete virtual view's orientation.
                    self._check_search_pose(quality, pose.rvec, pose.tvec, workspace['pose_prior'])
                    pose_angle = quality['pose_comparison']['rotation_deg']
                else:
                    if pose_angle > self.max_coarse_rotation_deg:
                        quality['passed'] = False
                        quality['reasons'].append('coarse_pose_mismatch' if comparison_pose is None
                                                 else 'baseline_pose_mismatch')
                    quality['pose_comparison'] = dict(
                        reference='3dgs_view' if comparison_pose is None else 'original_pass',
                        rotation_deg=pose_angle, max_rotation_deg=self.max_coarse_rotation_deg)
                hypotheses.append((pose, transfers, quality, name, pose_angle))
                self.last_diagnostics['candidates'][-1].update(
                    inliers=len(pose.inlier_indices), quality=quality,
                    pose_rotation_deg=pose_angle,
                    pose=dict(rvec=pose.rvec.ravel().tolist(), tvec=pose.tvec.ravel().tolist()))
                stage['timings_ms']['qc'] = (time.perf_counter()-qc_started)*1000
            self._candidate_diagnostic()
            # Finish a coarse-view group before considering early termination.
            group_finished = number == len(references) or references[number][1] != virtual_index
            if effective_policy == 'adaptive' and group_finished and number < len(references):
                decision = reference_exit_decision(hypotheses,
                    min_inliers=self.min_inliers, min_ratio=self.min_inlier_ratio,
                    max_error=self.max_reprojection_error,
                    max_coarse_angle=self.max_coarse_rotation_deg,
                    scene_scale=getattr(self, 'scene_scale', 0.),
                    crop_area=crop.shape[0]*crop.shape[1])
                checkpoints.append(dict(processed=number, **decision))
                if decision['stop']:
                    break
        processed = len(self.last_diagnostics['candidates'])
        summary = dict(policy=effective_policy, configured_policy=policy,
            processed=processed, total=len(references), skipped=len(references)-processed,
            early_exit=bool(decision['stop']), reason=decision['reason'],
            supporting_references=decision.get('supporting_references', []), checkpoints=checkpoints,
            processed_names=[c['name'] for c in self.last_diagnostics['candidates']],
            skipped_names=[r[0] for r in references[processed:]])
        self.last_diagnostics['reference_search'] = summary
        self._progress('qc', reference_search=summary)
        sink = getattr(self, 'diagnostic_sink', None)
        if sink is not None:
            sink('reference_search_decision', **getattr(self, 'diagnostic_context', {}), **summary)
        matching_ms = max(0., (time.perf_counter()-matches_started)*1000-pnp_ms)
        print(f'RC-ALIKED search: {effective_policy} {processed}/{len(references)} '
              f'skipped={summary["skipped"]} reason={summary["reason"]}', flush=True)
        print('RC-ALIKED: virtual=' + ','.join(self.last_diagnostics['virtual_views']) +
              ' | ' + '; '.join(f"{item['name']} raw={item['raw']} "
              f"ref={item['reference_features']} (XYZ {item['reference_xyz_features']}) "
              f"F={item['geometric']} XYZ={item.get('transferred', 0)} "
              f"noXYZ={item.get('xyz_rejections', {}).get('missing_xyz', 0)} "
              f"PnP={item.get('inliers', 0)}"
              for item in self.last_diagnostics['candidates']), flush=True)
        if not hypotheses:
            return self._fail(image, started, 'NO_PNP_HYPOTHESIS', keypoint_count,
                              retrieval_ms=retrieval_ms, matching_ms=matching_ms,
                              pnp_ms=pnp_ms)
        self._progress('qc')
        hypotheses.sort(key=registered_hypothesis_rank, reverse=True)
        pose, transfers, quality, name, pose_angle = hypotheses[0]
        accepted = bool(quality['passed'])
        if accepted:
            for other in hypotheses[1:]:
                other_pose, _, other_quality, other_name, _ = other
                if not comparable_support(other, hypotheses[0]):
                    continue
                angle = rotation_angle(cv2.Rodrigues(pose.rvec)[0],
                                       cv2.Rodrigues(other_pose.rvec)[0])
                first_center = -cv2.Rodrigues(pose.rvec)[0].T @ pose.tvec.reshape(3)
                other_center = -cv2.Rodrigues(other_pose.rvec)[0].T @ other_pose.tvec.reshape(3)
                if angle > 15. or np.linalg.norm(first_center-other_center) > 30.:
                    accepted = False
                    quality['reasons'].append('ambiguous_pose')
                    self.last_diagnostics['ambiguity'] = other_name
                    break
        rotation = cv2.Rodrigues(pose.rvec)[0]
        center = -rotation.T @ pose.tvec.reshape(3)
        points = np.asarray([item.xyz for item in transfers], np.float64)
        pixels = np.asarray([item.query_uv for item in transfers], np.float64)
        self.last_tracking_seed = (TrackingSeed(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            pixels[pose.inlier_indices].copy(), points[pose.inlier_indices].copy(),
            pose.rvec.copy(), pose.tvec.copy()) if accepted else None)
        self.last_debug = LocalizationDebug(points, frozenset(int(i) for i in pose.inlier_indices),
            tuple(float(v) for v in center), True,
            np.asarray([item.query_uv for item in transfers], np.float64))
        self.last_diagnostics['regions']['pnp'] = region_counts(pixels[pose.inlier_indices],(x0,y0,x1,y1))
        self.last_reference_name = name
        self.last_diagnostics.update(selected=name, quality=quality,
            candidate_scoring='spatial_8x8_squared_median_residual',
            selected_reference_views=[name],
            status='PASS' if accepted else 'FAIL_QC', matching_ms=matching_ms)
        selected_candidate = next(item for item in self.last_diagnostics['candidates']
                                  if item['name'] == name)
        selected_index = next(index for ref, index, _, _ in references if ref == name)
        self._progress('qc', reference_index=selected_index,
            reference_name=self.views[selected_index]['image_name'], registered_reference_name=name,
            raw_matches=selected_candidate['raw'], geometric_inliers=selected_candidate['geometric'],
            correspondences=len(transfers))
        canvas = image.copy()
        mpr_status = 'HIDDEN'
        mpr_started = time.perf_counter()
        if accepted and self.mpr_renderer is not None:
            canvas, mpr_status = self.mpr_renderer.render(
                canvas, camera_matrix, distortion, pose.rvec, pose.tvec)
        mpr_ms = (time.perf_counter()-mpr_started)*1000
        result = LiveResult('PASS' if accepted else 'FAIL',
            f'{name} / angle {pose_angle:.1f} deg / '
            f'{"QC OK" if accepted else ",".join(quality["reasons"])} / '
            f'refs {processed}/{len(references)} {"early" if summary["early_exit"] else "full"}',
            keypoint_count, len(transfers), len(pose.inlier_indices),
            len(pose.inlier_indices)/max(1, len(transfers)), pose.mean_error,
            tuple(float(v) for v in pose.rvec.ravel()),
            tuple(float(v) for v in pose.tvec.ravel()),
            tuple(float(v) for v in center), (time.perf_counter()-started)*1000,
            mpr_status, retrieval_ms, matching_ms, pnp_ms, mpr_ms)
        return result, self._decorate(canvas, result, transfers,
                                      frozenset(int(i) for i in pose.inlier_indices))
