#!/usr/bin/env python3
"""Local browser UI. Camera opening/reading stays on the Python main thread.

HTTP workers never touch capture, localizers, or MPR. The main loop consumes
commands and publishes atomic image/pose/layer packets. Not a medical device.
"""
from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import datetime as dt
from dataclasses import replace
import concurrent.futures
from contextlib import nullcontext
import json
import math
import os
import platform
import queue
import secrets
import struct
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import cv2
import numpy as np

from realtime_pnp import (DEFAULT_SPLAT_VIEWS, DEFAULT_REGISTERED_MAP,
                          DEFAULT_VOLUME, DEFAULT_COORDINATES, build_parser,
                          build_localizer, open_camera, resize_for_processing,
                          camera_parameters, read_camera_startup)
from realtime_pnp_server import LiveResult
from ct_overlay_validation import CTValidationSession
from browser_controls import (ControlMailbox, slice_position, render_snapshot, render_layer,
                              project_slice_corners)
from browser_photo import (decode_photo, photo_engine, localize_photo, MAX_PHOTO_BYTES,
                           PHOTO_RETRIEVAL_MAX_WIDTH)
from runtime_diagnostics import RuntimeDiagnostics
from browser_telemetry import validate_browser_batch
from map_preprocessing import (PreprocessingManager, reference_details, reference_preview,
                               recommended_map, audited_map)
from browser_file_picker import choose_local_file
from image_occlusion import occlude_image, compare_image_occlusion

ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(mode='registered', camera=1, top_k=3, views_path=str(DEFAULT_SPLAT_VIEWS),
                map_path=str(DEFAULT_REGISTERED_MAP), volume_path=str(DEFAULT_VOLUME),
                coordinates_path=str(DEFAULT_COORDINATES), calibration_path='', mpr=True,
                quality_profile='precision', max_processing_width=1280, fov_deg=72.,
                coarse_method='dino-patch', refresh_policy='background', input_source='camera', photo_feature_size=1024,
                registered_search_policy='adaptive')


def reference_presets():
    choices = [('original', '元参照（デフォルト）', DEFAULT_SPLAT_VIEWS),
               ('alpha15', 'アルファ1.5（比較）', DATA_DIR/'output_3dgs_reference_views_upper_rgb_20261003_alpha15')]
    return [dict(id=key, label=label, path=str(path), available=(path/'manifest.json').is_file())
            for key,label,path in choices]


def runtime_arguments(settings):
    """Use the current upper reference preset, without any OpenCV windows."""
    values = dict(DEFAULTS, **settings)
    if values['input_source'] not in ('camera','photo') or int(values['photo_feature_size']) not in (1024,1600,2048):
        raise ValueError('Invalid input source / photo feature resolution')
    mode = values['mode']
    if mode not in ('gs', 'registered'):
        raise ValueError('Unknown matching mode')
    if values['coarse_method'] not in ('dino-global', 'dino-patch'):
        raise ValueError('Unknown coarse retrieval method')
    if values['refresh_policy'] not in ('background', 'coverage', 'orbit-experimental'):
        raise ValueError('Unknown refresh policy')
    if values['registered_search_policy'] not in ('adaptive', 'exhaustive'):
        raise ValueError('Unknown registered reference search policy')
    camera, top_k = int(values['camera']), int(values['top_k'])
    if camera < 0 or not 1 <= top_k <= 10:
        raise ValueError('Camera must be >= 0; Top-K must be 1–10')
    profile = values['quality_profile']
    processing_width = int(values['max_processing_width'])
    fov = float(values['fov_deg'])
    if profile not in ('standard', 'precision', 'fast') or processing_width not in (1280, 1920):
        raise ValueError('Invalid quality profile / processing resolution')
    if not math.isfinite(fov) or not 30 <= fov <= 110:
        raise ValueError('Approximate HFOV must be 30–110 degrees')
    views = Path(values['views_path']).expanduser().resolve()
    manifest = json.loads((views / 'manifest.json').read_text())
    if manifest.get('runtime_transform') != 'identity':
        raise ValueError('Browser currently supports identity runtime_transform only')
    ply = Path(manifest['ply']).expanduser()
    if not ply.is_file():
        raise ValueError(f'3DGS PLY not found: {ply}')
    options = ['--camera', str(camera), '--capture-backend', 'opencv',
               '--width', '0', '--height', '0', '--max-processing-width', str(processing_width),
               '--fov-deg', str(fov),
               '--process-fps', '30', '--splat-views', str(views),
               '--glb-top-k', str(top_k), '--min-inliers', '10',
               '--min-inlier-ratio', '.20', '--mpr-size', '192',
               '--retrieval-mode', values['coarse_method'], '--retrieval-auto-roi',
               '--glb-max-keypoints', '2000', '--glb-resize', '1024',
               '--glb-match-threshold', '.01', '--fundamental-threshold', '3',
               '--tracking-reseed-frames', '300', '--tracking-min-points', '8',
               '--tracking-replenish-mode',
               'background' if values['refresh_policy']=='orbit-experimental' else values['refresh_policy'],
               '--tracking-replenish-frames', '60', '--tracking-replenish-distance-mm', '30',
               '--tracking-replenish-height-mm', '15', '--tracking-replenish-cooldown', '1',
               '--tracking-replenish-image-coverage', '.02', '--tracking-replenish-angle', '12',
               '--virtual-max-seed-age', '10']
    if values['refresh_policy'] in ('background', 'orbit-experimental'):
        options += ['--tracking-reseed-frames', '1800']
    if values['refresh_policy'] == 'orbit-experimental':
        options += ['--tracking-orbit-recovery']
    if mode == 'gs':
        options += ['--matching-mode', 'splat-track', '--virtual-policy', 'validated',
                    '--virtual-max-coarse-rotation', '35', '--ransac-threshold', '8',
                    '--iterations', '10000', '--virtual-guided-features', '2000',
                    '--virtual-guided-radius', '96']
    else:
        map_path = Path(values['map_path']).expanduser()
        if not (map_path / 'object_aliked_map.npz').is_file():
            raise ValueError(f'Registered ALIKED map not found: {map_path}')
        options += ['--matching-mode', 'registered-aliked', '--registered-map', str(map_path),
                    '--registered-search-policy', values['registered_search_policy'],
                    '--virtual-max-coarse-rotation', '55', '--ransac-threshold', '8']
    if values['mpr']:
        for key in ('volume_path', 'coordinates_path'):
            if not Path(values[key]).expanduser().is_file():
                raise ValueError(f'File not found: {values[key]}')
        options += ['--volume', str(Path(values['volume_path']).expanduser()),
                    '--coordinates', str(Path(values['coordinates_path']).expanduser())]
        if mode == 'registered':
            # Keep the registered-map/CT bounds precheck in build_localizer.
            # BrowserApplication owns live rendering and frozen measurement;
            # this CLI validation flag must not gate browser live overlays.
            options += ['--ct-validation']
    else:
        options += ['--no-mpr']
    if values['calibration_path'].strip():
        options += ['--calibration', str(Path(values['calibration_path']).expanduser())]
    if profile == 'precision':
        options += ['--ransac-threshold', '4', '--glb-max-reprojection-error', '3',
                    '--tracking-reseed-frames', '90']
    elif profile == 'fast':
        # Keep the user's 2000-point budget in every profile; only resize here.
        options += ['--glb-resize', '768']
    return build_parser().parse_args(options), manifest, ply.resolve()


def pose_camera(result, matrix, width, height):
    if result is None or result.status != 'PASS' or result.rvec is None or result.tvec is None:
        return None
    rvec, tvec = np.asarray(result.rvec, np.float64).reshape(3), np.asarray(result.tvec, np.float64).reshape(3)
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all() or not np.isfinite(matrix).all():
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    center = -rotation.T @ tvec
    return dict(eye=center.tolist(), target=(center + rotation.T[:, 2]).tolist(),
                up=(-rotation.T[:, 1]).tolist(), K=np.asarray(matrix).tolist(),
                width=width, height=height)


def encode_packet(metadata, jpeg, layer=b'', mask=b''):
    header = json.dumps(dict(metadata, jpeg_bytes=len(jpeg), layer_bytes=len(layer), mask_bytes=len(mask)),
                        ensure_ascii=False, allow_nan=False).encode()
    return struct.pack('<I', len(header)) + header + jpeg + layer + mask


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    return value


def correspondence_debug(debug, result, matrix, distortion, bounds=None):
    """Display actual observed UV and predicted UV, in rectified image pixels.

    This is a reprojection check, NOT proof of correct landmark identity/TRE.
    Caller must supply the immutable debug snapshot from the result's frame.
    """
    if debug is None or result is None or matrix is None:
        return None
    uv = getattr(debug, 'image_points', None)
    if uv is None:
        return None
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    xyz = np.asarray(debug.matched_xyz, np.float64).reshape(-1, 3)
    if len(uv) != len(xyz) or not len(uv):
        return None
    observed = cv2.undistortPoints(uv.reshape(-1,1,2),matrix,distortion,P=matrix).reshape(-1,2)
    projected = None
    box = None
    if debug.pnp_solved and result.rvec is not None and result.tvec is not None:
        rvec, tvec = np.asarray(result.rvec, np.float64), np.asarray(result.tvec, np.float64)
        if np.isfinite(rvec).all() and np.isfinite(tvec).all():
            projected = cv2.projectPoints(xyz,rvec,tvec,matrix,np.zeros_like(distortion))[0].reshape(-1,2)
            if bounds is not None:
                lo,hi=np.asarray(bounds,np.float64)
                corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                camera_points=(cv2.Rodrigues(rvec)[0]@corners.T).T+tvec.reshape(3)
                if np.all(camera_points[:,2]>0):
                    box=cv2.projectPoints(corners,rvec,tvec,matrix,np.zeros_like(distortion))[0].reshape(-1,2).tolist()
    mask=[i in debug.inlier_indices for i in range(len(uv))]
    indices=np.flatnonzero(mask)
    coverage=0.
    if len(indices)>=3:
        coverage=float(cv2.contourArea(cv2.convexHull(observed[indices].astype(np.float32))))
    return finite_json(dict(observed=observed, xyz=xyz, projected=projected, inliers=mask,
        accepted=result.status=='PASS', box=box, hull_area_pixels=coverage,
        legend='observed UV versus reprojected XYZ; not absolute registration error'))


def validation_result_summary(result):
    """Comparable matching/QC numbers, not a CT registration accuracy estimate."""
    return finite_json(dict(status=result.status, keypoints=result.keypoints,
        matches=result.matches, inliers=result.inliers, inlier_ratio=result.inlier_ratio,
        mean_reprojection_error=result.mean_reprojection_error,
        processing_ms=result.processing_ms))


def validation_qc_diagnosis(result, diagnostics, engine, max_error_override=None,
                            min_coverage_override=None):
    """Expose the actual QC gates and stage counts, without accepting a failed pose."""
    diagnostics = diagnostics or {}
    quality = diagnostics.get('quality') or {}
    limit_inliers = int(getattr(engine, 'min_inliers', 10))
    limit_ratio = float(getattr(engine, 'min_inlier_ratio', .20))
    limit_error = float(max_error_override if max_error_override is not None else
                        getattr(engine, 'max_reprojection_error', 3.))
    limit_coverage = float(min_coverage_override if min_coverage_override is not None else
                           getattr(engine, 'min_coverage', .005))
    reasons = list(quality.get('reasons') or []) if result.status != 'PASS' else []
    if result.status != 'PASS':
        stop_reason = diagnostics.get('reason') or result.detail or ''
        if any(stop_reason.endswith(code) for code in
               ('NO_OBJECT_ROI', 'NO_OBJECT_FEATURES', 'NO_POSE_NEAR_RC_PHOTO')):
            reasons.insert(0, stop_reason)
        if result.inliers < limit_inliers and 'few_inliers' not in reasons:
            reasons.append('few_inliers')
        if result.inlier_ratio < limit_ratio and 'low_inlier_ratio' not in reasons:
            reasons.append('low_inlier_ratio')
        if (result.mean_reprojection_error is not None
            and result.mean_reprojection_error > limit_error
            and 'reprojection_error' not in reasons):
            reasons.append('reprojection_error')
        if diagnostics.get('ambiguous_hypotheses') and 'ambiguous_pose' not in reasons:
            reasons.append('ambiguous_pose')
        if diagnostics.get('ambiguity') and 'ambiguous_pose' not in reasons:
            reasons.append('ambiguous_pose')
        if not reasons:
            reasons = [diagnostics.get('reason') or result.detail or 'unknown_qc']
    candidates = diagnostics.get('candidates') or []
    selected = diagnostics.get('selected')
    candidate = next((item for item in candidates if item.get('image_name', item.get('name')) == selected), None)
    if candidate is None and candidates:
        candidate = max(candidates, key=lambda item: item.get('xyz_correspondences', item.get('transferred', 0)))
    candidate = candidate or {}
    stages = dict(reference=selected or candidate.get('image_name', candidate.get('name')),
                  reference_features=candidate.get('reference_features'),
                  reference_xyz_features=candidate.get('reference_xyz_features'),
                  raw_matches=candidate.get('raw_local_matches', candidate.get('raw')),
                  geometric_inliers=candidate.get('geometric_inliers', candidate.get('geometric')),
                  xyz_correspondences=candidate.get('xyz_correspondences', candidate.get('transferred')),
                  missing_xyz=(candidate.get('xyz_rejections') or {}).get('missing_xyz'))
    candidate_stages = [dict(
        reference=item.get('image_name', item.get('name')),
        reference_features=item.get('reference_features'),
        reference_xyz_features=item.get('reference_xyz_features'),
        raw_matches=item.get('raw_local_matches', item.get('raw')),
        after_face_filter=item.get('local_matches'),
        geometric_inliers=item.get('geometric_inliers', item.get('geometric')),
        xyz_correspondences=item.get('xyz_correspondences', item.get('transferred')),
        xyz_rejections=item.get('xyz_rejections'),
        pnp_inliers=item.get('inliers')) for item in candidates]
    return finite_json(dict(reasons=reasons, stages=stages,
        object_detection=diagnostics.get('object_detection', diagnostics.get('retrieval_detection')),
        candidate_stages=candidate_stages,
        reference_matching=diagnostics.get('reference_matching'),
        all_feature_result=diagnostics.get('all_feature_result'),
        xyz_only_fallback=diagnostics.get('xyz_only_fallback'),
        coverage=quality.get('coverage'), spread_3d=quality.get('spread_3d'),
        p95_error=quality.get('p95_error'),
        pose_comparison=quality.get('pose_comparison'),
        limits=dict(min_inliers=limit_inliers, min_ratio=limit_ratio,
                    max_mean_px=limit_error,
                    max_p95_px=quality.get('max_p95_error'),
                    min_coverage=limit_coverage, min_spread_3d=.01)))


def localize_with_validation_limits(engine, image, matrix, distortion, polygon,
                                    max_mean_error_px, min_coverage):
    """Temporarily change frozen-image QC limits; preserve live settings."""
    with getattr(engine, 'lock', nullcontext()):
        return _localize_with_validation_limits(engine, image, matrix, distortion,
                                                polygon, max_mean_error_px, min_coverage)


def _localize_with_validation_limits(engine, image, matrix, distortion, polygon,
                                     max_mean_error_px, min_coverage):
    original_error = getattr(engine, 'max_reprojection_error', None)
    original_coverage = getattr(engine, 'min_coverage', None)
    try:
        engine.max_reprojection_error = float(max_mean_error_px)
        engine.min_coverage = float(min_coverage)
        return engine.localize(image, matrix, distortion, excluded_polygon=polygon)
    finally:
        if original_error is None:
            del engine.max_reprojection_error
        else:
            engine.max_reprojection_error = original_error
        if original_coverage is None:
            del engine.min_coverage
        else:
            engine.min_coverage = original_coverage


def projected_model_faces(bounds, result, matrix, distortion):
    """Visible faces of the model box, in the supplied image's pixel space."""
    if bounds is None or result is None or result.rvec is None or matrix is None:
        return []
    lo, hi = np.asarray(bounds, np.float64)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
        return []
    rotation = cv2.Rodrigues(np.asarray(result.rvec, np.float64))[0]
    translation = np.asarray(result.tvec, np.float64).reshape(3)
    eye = -rotation.T @ translation
    faces = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side, bound in (('min', lo[axis]), ('max', hi[axis])):
            center = (lo + hi) / 2
            center[axis] = bound
            normal = np.zeros(3)
            normal[axis] = -1. if side == 'min' else 1.
            if float(np.dot(normal, eye - center)) <= 0:
                continue
            corners = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = center.copy()
                point[other[0]] = (lo[other[0]], hi[other[0]])[u]
                point[other[1]] = (lo[other[1]], hi[other[1]])[v]
                corners.append(point)
            corners = np.asarray(corners, np.float64)
            camera = (rotation @ corners.T).T + translation
            if np.any(camera[:, 2] <= 1e-6):
                continue
            uv = cv2.projectPoints(corners, np.asarray(result.rvec, np.float64),
                translation, matrix, distortion)[0].reshape(-1, 2)
            if not np.isfinite(uv).all():
                continue
            faces.append(dict(id=f'{"xyz"[axis]}_{side}',
                              label=f'{"XYZ"[axis]}{side}', polygon=uv.tolist(),
                              depth=float(np.mean(camera[:, 2]))))
    return faces


class BrowserApplication:
    def __init__(self, demo=False, log_directory=None):
        self.preprocessor = PreprocessingManager()
        self.demo = demo
        self.log_directory = Path(log_directory) if log_directory is not None else DATA_DIR / 'output_runtime_logs'
        self.diagnostics = None
        self.token = secrets.token_urlsafe(32)
        self.commands = ControlMailbox(maxsize=128)
        self.condition = threading.Condition()
        self.state = dict(status='idle', detail='設定を確認して「開始」を押してください。', demo=demo)
        self.sequence = 0
        self.packet = None
        self.shutdown = threading.Event()
        self.capture = self.localizer = self.renderer = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.layer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='mpr-layer')
        self.layer_future = None
        self.layer_source = None
        self.slice_request_id = 0
        self.slice_interacting = False
        self.display_plane_revision = -1
        self.render_barrier = 0
        self.layer_discard = False
        self.future = None
        self.future_source = None
        self.discard_future = False
        self.session = None
        self.last_valid = None
        self.current = None
        self.current_debug = None
        self.last_valid_debug = None
        self.last_published = None
        self.slice_points = []
        self.slice_percent = 50.
        self.camera_linked_slice = True
        self.camera_slice_distance = None
        self.opacity = 82.
        self.slice_mask_enabled = True
        self.generation = 0
        self.source_id = 0
        self.ply = None
        self.manifest = None
        self.args = None
        self.settings = dict(DEFAULTS)
        self.rectify_maps = None
        self.last_process = 0.
        self.last_packet = 0.
        self.last_render_key = None
        self.cached_layer = b''
        self.cached_layer_rect = None
        self.live_layer = None
        self.rate_started = time.monotonic()
        self.rate_counts = dict(tracking=0, mpr=0, packets=0)
        self.rates = dict(tracking=0., mpr=0., packets=0.)
        self.control_revision = 0
        self.localize_ms = 0.
        self.browser_batches = {}
        self.search_preview = None
        self.search_source_id = None
        self.search_generation = 0
        self.photo_upload = None
        self.photo_image = self.photo_matcher = None
        self.photo_done = False
        self.photo_encoded = self.photo_encoding_key = None
        self.photo_image_revision = 0
        self.photo_rectified = None
        self.validation_baseline = None
        self.validation_baseline_source = None
        self.validation_comparison = None
        self.validation_future = None
        self.validation_source = None
        self.validation_abort = False
        self.validation_baseline_preview = self.validation_latest_preview = None
        self.validation_attempt_id = 0
        self.validation_mask_face = self.validation_polygon = None
        self.validation_kind = 'face'
        self.validation_qc_limit = None
        self.validation_min_coverage = None
        self.validation_view_count = None
        self.validation_reference_policy = None
        self.last_valid_reference_views = ()
        self.validation_reference_views = ()

    @property
    def is_photo(self):
        return self.settings.get('input_source') == 'photo'

    def current_gs_reference_views(self):
        """Snapshot the view(s) that supplied the accepted GS pose, not a new retrieval."""
        if getattr(self.args, 'matching_mode', None) == 'registered-aliked':
            engine = self.photo_matcher if self.is_photo else getattr(self.localizer, 'localizer', None)
            diagnostic = (getattr(engine, 'last_diagnostics', {}) if self.is_photo else
                          getattr(self.localizer, 'last_search_diagnostics', {})) or {}
            active = (diagnostic.get('selected') if self.is_photo else
                      getattr(self.localizer, 'active_reference_name', None))
            if active and (active == 'POSE_LANDMARKS' or active in getattr(engine, 'registered', {})):
                return (active,)
            return ()
        if getattr(self.args, 'matching_mode', None) != 'splat-track':
            return ()
        valid = {view['image_name'] for view in (self.manifest or {}).get('views', ())}
        if self.is_photo:
            diagnostic = getattr(self.photo_matcher, 'last_diagnostics', {}) or {}
            active = diagnostic.get('selected')
        else:
            diagnostic = getattr(self.localizer, 'last_search_diagnostics', {}) or {}
            active = getattr(self.localizer, 'active_reference_name', None)
        names = diagnostic.get('selected_reference_views') or ()
        if active and diagnostic.get('selected') == active and names and all(name in valid for name in names):
            return tuple(dict.fromkeys(names))
        if active in valid:
            return (active,)
        return ()

    def update(self, **values):
        logger = getattr(self, 'diagnostics', None)
        if logger is not None and ('status' in values or values.get('error')):
            logger.emit('application_state', **{key: values[key] for key in
                ('status', 'detail', 'error') if key in values})
        with self.condition:
            if values.get('error'):
                self.state['error_revision'] = self.state.get('error_revision', 0) + 1
                print(f'ViewSync: {values["error"]}', flush=True)
            self.state.update(values)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(self.state, generation=self.generation, sequence=self.sequence)

    def record_browser_batch(self, values):
        values = validate_browser_batch(values)
        with self.condition:
            logger = getattr(self, 'diagnostics', None)
            if (logger is None or logger.closed or values['generation'] != self.generation
                    or values['session_id'] != logger.session_id):
                raise ValueError('Inactive or stale telemetry session')
            key = (logger.session_id, values['client_id'])
            previous = self.browser_batches.get(key, 0)
            if values['batch_id'] <= previous:
                raise ValueError('Duplicate or reordered browser batch')
            if key not in self.browser_batches and len(self.browser_batches) >= 8:
                raise ValueError('Too many telemetry clients')
            self.browser_batches[key] = values['batch_id']
            logger.emit('browser_batch', **{k:v for k,v in values.items() if k != 'session_id'},
                timing_scope='python_read_to_browser_paint_opportunity_estimate',
                physical_end_to_end_measured=False)

    def start(self, settings):
        if self.capture is not None or self.args is not None:
            raise ValueError('停止してから登録データを変更してください。')
        args, manifest, ply = runtime_arguments(settings)
        self.rate_started = time.monotonic()
        self.rate_counts = dict.fromkeys(self.rate_counts, 0)
        self.rates = dict.fromkeys(self.rates, 0.)
        self.settings = {key: settings.get(key, value) for key, value in DEFAULTS.items()}
        self.update(status='loading', detail='写真と参照モデルを読み込み中…' if self.is_photo else 'カメラと参照モデルを読み込み中…', error=None)
        self.args, self.manifest, self.ply = args, manifest, ply
        if self.is_photo:
            if self.photo_upload is None:
                raise ValueError('入力する写真を選択してください。')
            self.photo_image,self.photo_name = self.photo_upload
            self.photo_done=False
            self.photo_encoded=self.photo_encoding_key=self.photo_rectified=None
            self.photo_image_revision=0
        self.camera_slice_distance = None
        if self.camera_linked_slice:
            self.slice_percent = 50.
        self.generation += 1
        self.browser_batches = {}
        try:
            self.diagnostics = RuntimeDiagnostics(self.log_directory)
            self.diagnostics.emit('configuration', settings=self.settings, demo=self.demo,
                                  generation=self.generation, opencv_version=cv2.__version__,
                                  runtime_environment=dict(python=platform.python_version(),
                                      os=platform.platform(),machine=platform.machine(),
                                      cpu_count=os.cpu_count(),opencv_threads=cv2.getNumThreads(),
                                      monotonic_resolution_ms=time.get_clock_info('monotonic').resolution*1000),
                                  capture_timestamp_scope='opencv_read_completed')
            print(f'ViewSync diagnostics: {self.diagnostics.path}', flush=True)
        except OSError as exc:
            self.diagnostics = None
            print(f'ViewSync diagnostics unavailable: {exc}', flush=True)
        if not self.demo or self.is_photo:
            # Focus behavior: the same direct AVFoundation open/read on main thread.
            if self.is_photo:
                frame=self.photo_image
            else:
                self.capture = open_camera(args.camera, 0, 0, capture_backend='opencv', ffmpeg_path=args.ffmpeg)
                ok, frame = read_camera_startup(self.capture)
                if not ok:
                    raise ValueError('カメラからフレームを取得できませんでした。')
            self.publish(frame, None, None, None, time.monotonic(), self.source_id,
                         'モデル読込中', force=True)
            self.localizer = build_localizer(args, temporal_registered=True)
            if self.diagnostics is not None:
                backend = getattr(self.localizer, 'global_localizer', None) or getattr(self.localizer, 'localizer', self.localizer)
                self.diagnostics.emit('backend_environment', device=str(getattr(backend,'device','unknown')),
                    torch_version=getattr(getattr(backend,'torch',None),'__version__',None),
                    numpy_version=np.__version__)
            if self.diagnostics is not None:
                for owner in (self.localizer, getattr(self.localizer, 'global_localizer', None),
                              getattr(self.localizer, 'localizer', None)):
                    if owner is not None:
                        owner.diagnostic_sink = self.diagnostics.emit
            # The browser needs no extra DINO inference just for the old viewer.
            if hasattr(self.localizer, 'viewer_coarse_enabled'):
                self.localizer.viewer_coarse_enabled = False
            self.renderer = self.localizer.mpr_renderer
            # Avoid rendering MPR twice or from worker threads.
            for owner in (self.localizer, getattr(self.localizer, 'tracker', None),
                          getattr(self.localizer, 'localizer', None),
                          getattr(self.localizer, 'global_localizer', None)):
                if owner is not None:
                    owner.mpr_renderer = None
            if self.is_photo:
                self.photo_matcher=photo_engine(self.localizer,self.settings['photo_feature_size'])
                if self.diagnostics is not None:
                    self.photo_matcher.diagnostic_sink = self.diagnostics.emit
                if self.renderer:self.renderer.texture_size=512
        else:
            if not args.no_mpr:
                from realtime_pnp_server import MPRRenderer
                self.renderer = MPRRenderer(args.volume, args.coordinates, texture_size=args.mpr_size)
            view = manifest['views'][0]
            self.demo_frame = cv2.imread(str(args.splat_views / view['image_name']))
            if self.demo_frame is None:
                raise ValueError('Demo reference image not found')
        self.rectify_maps = None
        self.last_valid = self.current = self.session = None
        self.validation_baseline = self.validation_comparison = None
        self.validation_baseline_source = None
        self.validation_abort = False
        self.validation_source = None
        self.validation_baseline_preview = self.validation_latest_preview = None
        self.validation_attempt_id = 0
        self.validation_mask_face = self.validation_polygon = None
        self.validation_qc_limit = None
        self.validation_min_coverage = None
        self.validation_view_count = None
        self.validation_reference_policy = None
        self.last_valid_reference_views = ()
        self.validation_reference_views = ()
        self.current_debug = self.last_valid_debug = None
        self.last_published = None
        self.search_preview = None
        self.search_source_id = None
        self.slice_points = []
        self.cached_layer = b''
        self.last_render_key = None
        self.last_process = 0.
        if self.renderer:
            self.renderer.set_slice_percent(self.slice_percent)
            self.renderer.set_opacity_percent(self.opacity)
        self.update(status='running', detail=(f'写真を照合中：{self.photo_name}' if self.is_photo else 'ライブ取得中' if not self.demo else 'DEMO：登録画像・既知姿勢（ライブ推定ではありません）'),
                    mode=self.settings['mode'], manifest=manifest, ply_bytes=ply.stat().st_size,
                    frozen=False, measurement=None, validation_comparison=None,
                    validation_reprocess_status='idle')

    def stop(self):
        self.update(status='stopping', detail='停止処理中…')
        # Return the native camera before potentially long inference waits.
        capture, self.capture = self.capture, None
        if capture is not None:
            capture.release()
        del capture
        if self.future is not None:
            self.future.cancel()
            concurrent.futures.wait([self.future])
        if self.validation_future is not None:
            self.validation_future.cancel()
            concurrent.futures.wait([self.validation_future])
        self.validation_future = self.validation_source = None
        self.validation_abort = False
        self.validation_baseline = self.validation_comparison = None
        self.validation_baseline_source = None
        self.validation_baseline_preview = self.validation_latest_preview = None
        self.validation_attempt_id = 0
        self.validation_mask_face = self.validation_polygon = None
        self.validation_qc_limit = None
        self.validation_min_coverage = None
        self.validation_view_count = None
        self.validation_reference_policy = None
        self.last_valid_reference_views = ()
        self.validation_reference_views = ()
        if self.layer_future is not None:
            concurrent.futures.wait([self.layer_future])
        self.layer_future = self.layer_source = None
        self.live_layer = None
        if self.localizer is not None:
            self.localizer.close()
            workers = [getattr(self.localizer, name, None) for name in
                       ('future', 'coarse_future', 'search_future', 'guided_future')]
            concurrent.futures.wait([job for job in workers if isinstance(job, concurrent.futures.Future)])
        self.future = self.future_source = None
        self.capture = self.localizer = self.renderer = self.args = None
        self.photo_image=self.photo_matcher=self.photo_rectified=None
        self.photo_encoded=self.photo_encoding_key=None
        self.photo_image_revision=0
        self.cached_layer_rect=None
        self.session = self.last_valid = self.current = None
        self.current_debug = self.last_valid_debug = self.last_published = None
        self.manifest = self.ply = None
        self.rectify_maps = None
        self.slice_points = []
        self.generation += 1
        self.search_preview = None
        with self.condition:
            self.packet = None
        self.search_source_id = None
        self.update(status='idle', detail='停止しました。設定を変更して再開できます。', frozen=False,
                    measurement=None, validation_comparison=None, validation_reprocess_status='idle')
        if getattr(self, 'diagnostics', None) is not None:
            self.diagnostics.close()
            self.diagnostics = None

    def freeze_validation_source(self, source, debug):
        image, result, matrix, distortion, stamp, source_id, label = source
        if self.renderer is None:
            raise ValueError('CT/MPRを有効にして開始してください。')
        # Reprocessing a single deliberately captured frame may take over five
        # seconds. Preserve its real capture timestamp instead of pretending
        # the completed result came from the latest camera frame.
        if self.session is None:
            self.live_mpr_size = self.renderer.texture_size
        self.renderer.texture_size = 512
        image = self.rectify(image, matrix, distortion)
        if debug is not None and debug.image_points is not None:
            corrected_uv = cv2.undistortPoints(np.asarray(debug.image_points,np.float64).reshape(-1,1,2),
                                              matrix, distortion, P=matrix).reshape(-1,2)
            debug = replace(debug, image_points=corrected_uv)
        distortion = np.zeros_like(distortion)
        self.session = CTValidationSession(self.renderer, image, matrix, distortion, result, label)
        self.frozen_source = (image, result, matrix, distortion, stamp, source_id, label)
        self.frozen_debug = debug
        self.slice_points = []
        if self.validation_baseline is None:
            self.validation_baseline = validation_result_summary(result)
            self.validation_baseline_source = source
            self.validation_reference_views = self.last_valid_reference_views
            preview, _ = resize_for_processing(image, 640)
            self.validation_baseline_preview = cv2.imencode('.jpg', preview)[1].tobytes()
        self.update(frozen=True, measurement=None,
                    validation_comparison=self.validation_comparison,
                    validation_reprocess_status='idle')

    def finish_validation_reprocess(self):
        if self.validation_future is None or not self.validation_future.done():
            return
        future, source = self.validation_future, self.validation_source
        self.validation_future = self.validation_source = None
        if self.validation_abort or self.session is None:
            self.validation_abort = False
            return
        self.validation_attempt_id += 1
        occlusion = self.validation_kind == 'occlusion'
        title = '画像遮蔽' if occlusion else '面除外'
        painted, _, _ = occlude_image(source[0], self.validation_polygon) if occlusion else (
            source[0].copy(), None, None)
        if not occlusion:
            cv2.fillConvexPoly(painted, np.rint(self.validation_polygon).astype(np.int32), (75, 75, 75))
        preview, _ = resize_for_processing(self.rectify(painted, source[2], source[3]), 320)
        self.validation_latest_preview = cv2.imencode('.jpg', preview)[1].tobytes()
        gs_mode = self.settings.get('mode') == 'gs'
        registered_comparison = not gs_mode and hasattr(
            self.photo_matcher if self.is_photo else getattr(self.localizer, 'localizer', None),
            'localize_validation')
        try:
            if gs_mode or occlusion:
                baseline_result, result, debug, diagnostics = future.result()
                self.validation_baseline = validation_result_summary(baseline_result)
            elif registered_comparison:
                result, debug, diagnostics = future.result()
            else:
                result, _ = future.result()
        except Exception as exc:
            self.validation_comparison = dict(baseline=self.validation_baseline,
                latest=dict(status='ERROR', detail=str(exc),
                            diagnosis=dict(reasons=['processing_error'], detail=str(exc))),
                attempt=self.validation_attempt_id, face=self.validation_mask_face,
                kind=self.validation_kind, applied=False, max_mean_error_px=self.validation_qc_limit,
                min_coverage=self.validation_min_coverage,
                view_count=self.validation_view_count,
                reference_policy=self.validation_reference_policy,
                reference_views=([] if occlusion and self.validation_reference_policy != 'original'
                                 else list(self.validation_reference_views)))
            self.update(detail=f'{title}の再照合エラー：{exc}',
                        validation_comparison=self.validation_comparison,
                        validation_reprocess_status='done')
            return
        engine = self.photo_matcher if self.is_photo else (
            self.localizer.global_localizer if gs_mode else self.localizer.localizer)
        excluded_debug = debug if gs_mode or registered_comparison or occlusion else getattr(engine, 'last_debug', None)
        if excluded_debug is not None and getattr(excluded_debug, 'image_points', None) is not None:
            marked, scale = resize_for_processing(
                self.rectify(painted, source[2], source[3]), 640)
            image_points = np.asarray(excluded_debug.image_points, np.float64).reshape(-1, 2)
            if np.any(np.abs(source[3]) > 1e-12):
                image_points = cv2.undistortPoints(image_points.reshape(-1, 1, 2),
                    source[2], source[3], P=source[2]).reshape(-1, 2)
            inlier_indices = excluded_debug.inlier_indices
            for index, point in enumerate(image_points):
                if np.isfinite(point).all():
                    xy = tuple(int(value) for value in np.rint(point * scale))
                    cv2.circle(marked, xy, 4, (45, 220, 70) if index in inlier_indices
                               else (0, 165, 255), -1, cv2.LINE_AA)
            self.validation_latest_preview = cv2.imencode('.jpg', marked)[1].tobytes()
        result_diagnostics = diagnostics if gs_mode or registered_comparison or occlusion else (getattr(engine, 'last_diagnostics', {}) or {})
        if gs_mode or occlusion:
            self.validation_baseline['diagnosis'] = validation_qc_diagnosis(
                baseline_result, diagnostics.get('baseline_diagnostics'), engine,
                self.validation_qc_limit, self.validation_min_coverage)
        elif 'diagnosis' not in self.validation_baseline:
            self.validation_baseline['diagnosis'] = validation_qc_diagnosis(
                source[1], None, engine)
        latest = validation_result_summary(result)
        latest['diagnosis'] = validation_qc_diagnosis(
            result, result_diagnostics, engine, self.validation_qc_limit,
            self.validation_min_coverage)
        latest['excluded_face_matches'] = result_diagnostics.get('excluded_face_matches', 0)
        for summary, details in ((self.validation_baseline, result_diagnostics.get('baseline_diagnostics', {})),
                                 (latest, result_diagnostics)):
            summary['retrieval'] = details.get('retrieval', details.get('virtual_views', []))
        applied = result.status == 'PASS' and (not (gs_mode or occlusion) or baseline_result.status == 'PASS')
        self.validation_comparison = dict(baseline=self.validation_baseline,
            latest=latest, attempt=self.validation_attempt_id, face=self.validation_mask_face,
            kind=self.validation_kind, applied=applied, max_mean_error_px=self.validation_qc_limit,
            occluded_fraction=result_diagnostics.get('occluded_fraction'),
            polygon=self.validation_polygon.tolist() if occlusion else None,
            min_coverage=self.validation_min_coverage,
            view_count=self.validation_view_count,
            reference_policy=self.validation_reference_policy,
            reference_views=([] if occlusion and self.validation_reference_policy != 'original'
                             else list(self.validation_reference_views)),
            default_max_mean_error_px=float(self.args.glb_max_reprojection_error)
            if hasattr(self.args, 'glb_max_reprojection_error') else None)
        if self.diagnostics is not None:
            self.diagnostics.emit('validation_comparison',
                                  comparison=self.validation_comparison,
                                  source_id=source[5], calibration=source[6])
        if applied:
            observed = (source[0], result, *source[2:])
            self.current = self.last_valid = observed
            self.current_debug = self.last_valid_debug = debug if gs_mode or registered_comparison or occlusion else engine.last_debug
            self.freeze_validation_source(observed, self.last_valid_debug)
            self.control_revision += 1
            self.render_barrier = self.control_revision
            self.last_render_key = None
            self.layer_discard = True
            note = '除外後の姿勢を表示中。測定点は選び直してください。'
        else:
            note = ('元の固定姿勢を維持。同一画像の全点再探索もPASSしなかったため比較不能。'
                    if (gs_mode or occlusion) and baseline_result.status != 'PASS' else
                    '元の固定姿勢を維持。除外後の姿勢はCTへ重畳していません。')
        self.update(detail=f'{title}：{result.status}、{result.inliers} inliers。{note}',
                    validation_comparison=self.validation_comparison,
                    validation_reprocess_status='done')

    def controls(self, data):
        logger = getattr(self, 'diagnostics', None)
        if logger is not None:
            logger.emit('control', action=data.get('action'),
                values={k: v for k, v in data.items() if k not in ('action', 'token')})
        if 'generation' in data and int(data['generation']) != self.generation:
            raise ValueError('セッションが更新されました。新しい画面で操作してください。')
        action = data.get('action')
        if action in ('slice', 'opacity'):
            value = float(data['value'])
            if not math.isfinite(value):
                raise ValueError('Slider value must be finite')
            if action == 'slice':
                self.slice_percent = float(np.clip(value, -100, 150))
                self.slice_request_id = int(data.get('request_id', 0))
                self.slice_interacting = bool(data.get('interacting', False))
                if self.renderer:
                    self.renderer.set_slice_percent(self.slice_percent)
                    if self.camera_linked_slice and self.last_published is not None:
                        position = self.last_published[1].get('slice_position')
                        if position:
                            self.camera_slice_distance = (position['camera_depth']-position['offset']
                                                          + position['span']*self.slice_percent/100.)
                    if self.session:
                        self.renderer.texture_size = 192 if self.slice_interacting else 512
            else:
                self.opacity = float(np.clip(value, 0, 100))
                return  # Display-only opacity never invalidates CT geometry.
        elif action == 'slice_mask':
            if not isinstance(data.get('value'), bool):
                raise ValueError('Slice mask value must be boolean')
            self.slice_mask_enabled = data['value']
        elif action == 'camera_linked_slice':
            self.camera_linked_slice = bool(data['value'])
            self.camera_slice_distance = None
            if self.last_published is not None:
                self.slice_percent = self.last_published[1]['slice_percent']
                position = self.last_published[1].get('slice_position')
                if self.camera_linked_slice and position:
                    self.camera_slice_distance = position['camera_depth']
        elif action == 'freeze':
            if self.last_valid is None or (not self.is_photo and time.monotonic() - self.last_valid[4] > 5.):
                raise ValueError('5秒以内のPASS画像がありません。姿勢確定後に固定してください。')
            self.validation_qc_limit = None
            self.validation_min_coverage = None
            self.validation_view_count = None
            self.validation_reference_policy = None
            self.freeze_validation_source(self.last_valid, self.last_valid_debug)
        elif action == 'resume':
            if self.renderer and self.session:
                self.renderer.texture_size = self.live_mpr_size
            self.session = None
            self.slice_points = []
            self.validation_abort = True
            self.validation_baseline = self.validation_comparison = None
            self.validation_baseline_source = None
            self.validation_baseline_preview = self.validation_latest_preview = None
            self.validation_attempt_id = 0
            self.validation_mask_face = self.validation_polygon = None
            self.validation_qc_limit = None
            self.validation_min_coverage = None
            self.validation_view_count = None
            self.validation_reference_policy = None
            self.validation_reference_views = ()
            if not self.is_photo:
                self.last_valid = None
                self.current = None
            self.discard_future = self.future is not None
            if not self.is_photo and self.localizer is not None:
                reset = getattr(self.localizer, 'reset_tracking', None)
                if reset is not None:
                    reset()
            self.update(frozen=False, measurement=None, validation_comparison=None,
                        validation_reprocess_status='idle')
        elif action == 'validation_occlusion':
            if (self.session is None or self.demo or self.args is None
                    or self.args.matching_mode not in ('registered-aliked', 'splat-track')):
                raise ValueError('3DGSまたは登録実写のCT検証で固定してから実行してください。')
            if self.validation_future is not None:
                raise ValueError('前の再照合が完了するまでお待ちください。')
            source = self.validation_baseline_source
            if source is None:
                raise ValueError('比較する基準画像がありません。もう一度固定してください。')
            reference_policy = data.get('reference_policy', 'fresh_global')
            if reference_policy not in ('fresh_global', 'original'):
                raise ValueError('遮蔽照合の参照指定が不正です。')
            fixed = reference_policy == 'original'
            if fixed and (not self.validation_reference_views or
                          'POSE_LANDMARKS' in self.validation_reference_views):
                raise ValueError('元PASS時の参照view／登録写真を特定できません。全体再照合でPASS後に固定し直してください。')
            # UI coordinates are rectified image pixels. Both trials use that
            # image and zero distortion, not an approximate inverse polygon.
            rgb = self.rectify(source[0], source[2], source[3])
            polygon = np.asarray(data.get('polygon', []), np.float32)
            occlude_image(rgb, polygon)  # Validate before queuing work.
            engine = self.photo_matcher if self.is_photo else (
                self.localizer.global_localizer if self.settings['mode'] == 'gs' else self.localizer.localizer)
            limit = float(data.get('max_mean_error_px', getattr(engine, 'max_reprojection_error', 3.)))
            coverage = float(data.get('min_coverage_percent', .5))
            if not math.isfinite(limit) or not .5 <= limit <= 12.:
                raise ValueError('検証用の平均再投影誤差上限は0.5〜12 pxにしてください。')
            if not math.isfinite(coverage) or not 0 <= coverage <= 2.:
                raise ValueError('検証用の画像分布下限は0〜2%にしてください。')
            self.validation_kind = 'occlusion'
            self.validation_source = (rgb, source[1], source[2], np.zeros_like(source[3]), *source[4:])
            self.validation_mask_face = None
            self.validation_polygon = polygon
            self.validation_qc_limit, self.validation_min_coverage = limit, coverage/100.
            self.validation_view_count = getattr(engine, 'top_k', None)
            self.validation_reference_policy = reference_policy
            self.validation_abort = False
            self.validation_future = self.executor.submit(compare_image_occlusion,
                engine, rgb.copy(), source[2].copy(), np.zeros_like(source[3]), polygon.copy(),
                limit, coverage/100., lambda detail: self.update(detail=detail),
                reference_names=self.validation_reference_views if fixed else None,
                baseline_pose=(np.asarray(source[1].rvec, np.float64),
                               np.asarray(source[1].tvec, np.float64)) if fixed else None)
            self.update(detail=('元PASS参照を固定して遮蔽前後の局所照合・PnPを再実行中…' if fixed else
                                '遮蔽前後を対象物検出・粗検索から再ローカライズ中…'),
                        validation_reprocess_status='running')
        elif action == 'validation_reprocess':
            if (self.session is None or self.demo or self.args is None
                or self.args.matching_mode not in ('registered-aliked', 'splat-track')):
                raise ValueError('3DGSまたは登録実写のCT検証で固定してから実行してください。')
            if self.validation_future is not None:
                raise ValueError('前の再照合が完了するまでお待ちください。')
            source = self.validation_baseline_source
            if source is None:
                raise ValueError('比較する基準画像がありません。もう一度固定してください。')
            faces = projected_model_faces(self.manifest.get('bounds') if self.manifest else None,
                                          source[1], source[2], source[3])
            face = next((item for item in faces if item['id'] == data.get('face')), None)
            if face is None:
                raise ValueError('現在の視点で見えるモデルの面を選択してください。')
            polygon = np.asarray(face['polygon'], np.float32)
            gs_mode = self.args.matching_mode == 'splat-track'
            reference_policy = data.get('reference_policy', 'original') if gs_mode else None
            if gs_mode and reference_policy not in ('original', 'rerank'):
                raise ValueError('参照viewの指定が不正です。')
            if gs_mode and reference_policy == 'original' and not self.validation_reference_views:
                raise ValueError('元PASS時の参照viewを特定できません。再度PASSさせて固定するか、診断用のTop-K再検索を選んでください。')
            engine = self.photo_matcher if self.is_photo else (
                self.localizer.global_localizer if gs_mode else self.localizer.localizer)
            max_mean_error_px = float(data.get('max_mean_error_px',
                getattr(engine, 'max_reprojection_error', 3.)))
            if not math.isfinite(max_mean_error_px) or not .5 <= max_mean_error_px <= 12.:
                raise ValueError('検証用の平均再投影誤差上限は0.5〜12 pxにしてください。')
            min_coverage_percent = float(data.get('min_coverage_percent', .5))
            if not math.isfinite(min_coverage_percent) or not 0 <= min_coverage_percent <= 2.:
                raise ValueError('検証用の画像分布下限は0〜2%にしてください。')
            min_coverage = min_coverage_percent / 100.
            view_count = (int(data.get('view_count', getattr(engine, 'top_k', 3)))
                          if gs_mode and reference_policy == 'rerank' else None)
            if view_count is not None and not 1 <= view_count <= 10:
                raise ValueError('検証用の候補view数は1〜10にしてください。')
            self.validation_source = source
            self.validation_kind = 'face'
            self.validation_baseline = validation_result_summary(source[1])
            self.validation_mask_face = face['id']
            self.validation_polygon = polygon
            self.validation_qc_limit = max_mean_error_px
            self.validation_min_coverage = min_coverage
            self.validation_view_count = view_count
            self.validation_reference_policy = reference_policy
            if gs_mode:
                self.validation_future = self.executor.submit(
                    engine.compare_excluded_face, source[0].copy(), source[2].copy(),
                    source[3].copy(), polygon, max_mean_error_px, min_coverage,
                    view_count, self.validation_reference_views if reference_policy == 'original' else None)
            elif hasattr(engine, 'localize_validation'):
                self.validation_reference_policy = 'original'
                self.validation_future = self.executor.submit(
                    engine.localize_validation, source[0].copy(), source[2].copy(),
                    source[3].copy(), polygon, max_mean_error_px, min_coverage,
                    self.validation_reference_views,
                    (np.asarray(source[1].rvec, np.float64), np.asarray(source[1].tvec, np.float64)))
            else:
                self.validation_future = self.executor.submit(
                    localize_with_validation_limits, engine, source[0].copy(),
                    source[2].copy(), source[3].copy(), polygon, max_mean_error_px,
                    min_coverage)
            self.validation_abort = False
            self.update(detail=f'同じ固定画像から{face["label"]}面の対応候補を除いて再照合中… '
                               f'（検証用QC上限 {max_mean_error_px:g} px、画像分布下限 {min_coverage_percent:g}%、参照view {", ".join(self.validation_reference_views) if reference_policy == "original" else f"再検索 Top-{view_count}" if gs_mode else "登録実写"}）',
                        validation_reprocess_status='running')
        elif action == 'clear':
            self.slice_points = []
            if self.session:
                self.session.live_uv = self.session.ct_xyz = self.session.ct_voxel = None
            self.update(measurement=None)
        elif action == 'point':
            if self.session is None:
                raise ValueError('「固定して測定」を押してからクリックしてください。')
            # Reject clicks on an image/plane that has already changed.
            if (int(data.get('plane_revision', -1)) != self.control_revision
                or self.display_plane_revision != self.control_revision):
                raise ValueError('表示が更新されました。もう一度クリックしてください。')
            x, y = float(data['x']), float(data['y'])
            h, w = self.session.frame.shape[:2]
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h):
                raise ValueError('画像内をクリックしてください。')
            kind = data.get('kind')
            if kind == 'live':
                self.session.live_uv = np.array([x, y])
            else:
                selected = self.renderer.display_to_ct_point(x, y)
                if selected is None:
                    raise ValueError('表示中のCT断面内をクリックしてください。')
                if kind == 'ct':
                    self.session.ct_xyz, self.session.ct_voxel = selected
                elif kind == 'distance':
                    if len(self.slice_points) == 2:
                        self.slice_points = []
                    if self.slice_points:
                        normal = cv2.Rodrigues(self.session.rvec)[0].T[:, 2]
                        if abs(float(np.dot(selected[0] - self.slice_points[0], normal))) > .01:
                            raise ValueError('2点の距離は同じCT断面で測ってください。点を消して選び直せます。')
                    self.slice_points.append(selected[0].copy())
                else:
                    raise ValueError('Unknown point kind')
        elif action == 'return':
            if self.session:
                value = self.session.return_to_marker_slice()
                if value is not None:
                    self.slice_percent = value
                    if self.camera_linked_slice:
                        self.camera_slice_distance = slice_position(self.renderer,
                            self.session.rvec, self.session.tvec, value)['camera_depth']
        elif action == 'save':
            if not self.session or self.session.measurement() is None:
                raise ValueError('ライブ点とCT点の両方を指定してください。')
            path = DATA_DIR / 'output_browser_validation' / 'display_consistency_ct_coordinates.csv'
            self.session.save(path, calibration_label=self.session.intrinsics_label)
            self.update(detail=f'保存しました：{path}')
        elif action == 'relocalize':
            if self.is_photo:
                if self.future is not None:
                    raise ValueError('写真の照合が終わってから再実行してください。')
                if self.renderer and self.session:self.renderer.texture_size=self.live_mpr_size
                self.session=None
                self.photo_done=False
                self.current=self.last_valid=None
                self.current_debug=self.last_valid_debug=None
                self.slice_points=[]
                self.update(detail='写真を全体から再照合します。',frozen=False)
                self.control_revision+=1
                self.render_barrier=self.control_revision
                return
            if self.session:
                raise ValueError('固定を解除してから再探索してください。')
            if self.localizer is None:
                raise ValueError('ライブを開始してから再探索してください。')
            reset = getattr(self.localizer, 'reset_tracking', None)
            if reset is not None:
                reset()
            else:
                self.discard_future = self.future is not None
            self.current = self.last_valid = None
            self.current_debug = self.last_valid_debug = None
            self.last_process = 0.
            self.update(detail='追跡点を破棄し、新しい画像から全体を再探索します。')
        elif action == 'diagnostic':
            if getattr(self, 'last_published', None) is None:
                raise ValueError('診断保存できる画像がありません。')
            image, metadata = self.last_published
            folder=DATA_DIR/'output_browser_diagnostics'/dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder.mkdir(parents=True, exist_ok=False)
            if not cv2.imwrite(str(folder/'frame.png'),image):
                raise ValueError('Diagnostic image could not be saved')
            record=dict(metadata, settings=self.settings,
                        note='rectified processing image; reprojection residual is not clinical TRE')
            (folder/'diagnostic.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            self.update(detail=f'画像・姿勢・対応点を保存しました：{folder}')
        else:
            raise ValueError('Unknown action')
        self.control_revision += 1
        if action != 'slice':
            self.render_barrier = self.control_revision

    def measurement(self):
        if not self.session:
            return None
        session = self.session
        points = []
        if session.live_uv is not None:
            points.append(dict(uv=session.live_uv.tolist(), label='LIVE', color='#ffe066'))
        xyz = ([] if session.ct_xyz is None else [session.ct_xyz]) + self.slice_points
        for i, point in enumerate(xyz):
            uv = cv2.projectPoints(np.asarray(point).reshape(1, 3), session.rvec,
                                   session.tvec, session.matrix, session.distortion)[0].reshape(2)
            points.append(dict(uv=uv.tolist(), label='CT' if i == 0 and session.ct_xyz is not None else str(i+1), color='#fb82ff'))
        measurement = session.measurement()
        return finite_json(dict(points=points, consistency=measurement, slice_offset=session.slice_offset(), frozen_at=session.frozen_at,
                           distance=(float(np.linalg.norm(self.slice_points[1] - self.slice_points[0]))
                                     if len(self.slice_points) == 2 else None),
                           unit=self.renderer.unit_label))

    def rectify(self, image, matrix, distortion):
        if self.is_photo and image is self.photo_image and self.photo_rectified is not None:
            return self.photo_rectified
        if np.any(np.abs(distortion) > 1e-12):
            if self.rectify_maps is None:
                self.rectify_maps = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix,
                                                              (image.shape[1], image.shape[0]), cv2.CV_32FC1)
            rectified=cv2.remap(image, *self.rectify_maps, cv2.INTER_LINEAR)
            if self.is_photo and image is self.photo_image:self.photo_rectified=rectified
            return rectified
        return image

    def publish(self, image, result, matrix, distortion, stamp, source_id, label, force=False):
        now = time.monotonic()
        if not force and now - self.last_packet < 1 / 30:
            return
        if self.session is not None and self.last_render_key == (source_id, self.control_revision, self.generation):
            return
        ready_layer = None
        live = not self.is_photo and self.session is None and not force
        if not live:
            self.live_layer = None
        rendered_revision = self.control_revision
        rendered_percent = self.slice_percent
        rendered_request_id = self.slice_request_id
        debug = (getattr(self, 'frozen_debug', None) if self.session else self.current_debug)
        visible = (pose_camera(result, matrix, image.shape[1], image.shape[0]) is not None
                   and self.renderer is not None)
        if visible:
            position = slice_position(self.renderer, result.rvec, result.tvec, self.slice_percent)
            if self.camera_linked_slice:
                if self.camera_slice_distance is None:
                    self.camera_slice_distance = position['camera_depth']
                near = position['camera_depth']-position['offset']
                self.slice_percent = 100.*(self.camera_slice_distance-near)/max(position['span'],1e-9)
            with self.renderer._control_lock:
                self.renderer.camera_slice_distance = (self.camera_slice_distance
                                                      if self.camera_linked_slice else None)
                self.renderer.slice_fraction = self.slice_percent/100.
            rendered_percent = self.slice_percent
        if not force and self.layer_future is not None:
            if not visible:
                self.layer_discard = True
            if not self.layer_future.done():
                if visible and not live:
                    return  # Capture/controls/tracking continue; no inference queue.
            else:
                completed = self.layer_future.result()
                self.rate_counts['mpr'] += 1
                old_source, old_debug, old_key, old_percent, old_request_id = self.layer_source
                self.layer_future = self.layer_source = None
                if (visible and not self.layer_discard and old_key[0] == self.generation
                    and old_key[1] >= self.render_barrier
                    and bool(getattr(completed[3], '_native_layer', False)) == live
                    and not (self.is_photo and old_key[1] < self.control_revision)
                    and (self.is_photo or self.session is not None or now-old_source[4] <= 1.)):
                    if live:
                        self.live_layer = (completed, old_source, old_key, old_percent, old_request_id)
                    else:
                        image, result, matrix, distortion, stamp, source_id, label = old_source
                        debug = old_debug
                        ready_layer = completed
                        rendered_revision = old_key[1]
                        rendered_percent = old_percent
                        rendered_request_id = old_request_id
        source = (image, result, matrix, distortion, stamp, source_id, label)
        pose = None
        points = correspondence_debug(debug,result,matrix,distortion,
                                     self.manifest.get('bounds') if self.manifest else None)
        if matrix is not None:
            image = self.rectify(image, matrix, distortion)
            distortion = np.zeros_like(distortion)
            pose = pose_camera(result, matrix, image.shape[1], image.shape[0])
        layer, mpr_status = b'', 'HIDDEN'
        live_snapshot = None
        mpr_quad = mpr_source_id = mpr_age_ms = layer_id = None
        if live:
            if not visible:
                self.live_layer = None
            elif self.renderer is not None:
                mpr_status = 'PENDING'
                if self.live_layer is not None:
                    cached, old_source, key, percent, request_id = self.live_layer
                    if (key[0] == self.generation and key[1] >= self.render_barrier
                            and now-old_source[4] <= 1.):
                        png, status, self.mpr_ms, snapshot = cached
                        mpr_quad = project_slice_corners(
                            getattr(snapshot, '_last_layer_corners', None), result, matrix, image.shape)
                        if status != 'VISIBLE' or mpr_quad is not None:
                            layer = png if mpr_quad is not None else b''
                            mpr_status = status
                            live_snapshot = snapshot
                            rendered_revision, rendered_percent = key[1], percent
                            rendered_request_id = request_id
                            mpr_source_id, mpr_age_ms = old_source[5], (now-old_source[4])*1000
                            layer_id = f'{key[0]}:{key[1]}:{mpr_source_id}:{request_id}'
                needs_slice = (self.live_layer is None or self.live_layer[1][5] != source_id
                               or self.live_layer[2][1] != self.control_revision)
                if self.layer_future is None and needs_slice:
                    job = render_snapshot(self.renderer, image, matrix, distortion, result,
                                          native_layer=True)
                    self.layer_source = (source, debug, (self.generation, self.control_revision),
                                         self.slice_percent, self.slice_request_id)
                    self.layer_discard = False
                    self.layer_future = self.layer_executor.submit(render_layer, *job)
        elif pose is not None and self.renderer is not None:
            render_key = (source_id, rendered_revision, self.generation)
            if self.last_render_key != render_key:
                if ready_layer is None:
                    job = render_snapshot(self.renderer, image, matrix, distortion, result,
                                          crop_layer=self.is_photo)
                    if not force:
                        self.layer_source = (source, debug, (self.generation, self.control_revision),
                                             self.slice_percent, self.slice_request_id)
                        self.layer_discard = False
                        self.layer_future = self.layer_executor.submit(render_layer, *job)
                        return
                    ready_layer = render_layer(*job)
                self.cached_layer, self.cached_mpr_status, self.mpr_ms, snapshot = ready_layer
                self.cached_layer_rect = getattr(snapshot, '_last_layer_rect', None)
                with self.renderer._control_lock:
                    for name in ('_last_inverse_homography', '_last_inverse_view_homography', '_last_validation_plane', '_last_render_plane'):
                        setattr(self.renderer, name, getattr(snapshot, name))
                self.display_plane_revision = rendered_revision
                self.last_render_key = render_key
            layer, mpr_status = self.cached_layer, self.cached_mpr_status
        render_plane = (getattr(live_snapshot, '_last_render_plane', None) if live else
                        getattr(self.renderer, '_last_render_plane', None)
                        if pose is not None and self.renderer is not None else None)
        stats = self.capture.stats() if self.capture else {}
        metadata = finite_json(dict(source_id=source_id, captured_at=stamp, sent_at=now,
                    generation=self.generation, width=image.shape[1], height=image.shape[0],
                    pose=pose, pose_source_id=source_id if pose else None, frozen=self.session is not None,
                    demo=self.demo and not self.is_photo, input_source='photo' if self.is_photo else 'camera',
                    photo_name=getattr(self,'photo_name','') if self.is_photo else '',
                    photo_feature_size=self.settings['photo_feature_size'] if self.is_photo else None,
                    photo_retrieval_max_width=PHOTO_RETRIEVAL_MAX_WIDTH if self.is_photo else None,
                    calibration=label, status=result.status if result else 'SEARCH',
                    model_center=self.manifest.get('center') if self.manifest else None,
                    model_bounds=self.manifest.get('bounds') if self.manifest else None,
                    mpr_plane=(dict(center=render_plane[0], normal=render_plane[1]) if render_plane is not None else None),
                    world_up=[0,0,-1] if self.manifest and self.manifest.get('up_axis')=='negative-z' else [0,0,1],
                    result=result.as_dict() if result else None, capture_fps=stats.get('capture_fps', 0),
                    source_age_ms=(now-stamp)*1000, localize_ms=self.localize_ms,
                    layer_mode='projective' if live else 'image', layer_id=layer_id,
                    mpr_quad=mpr_quad, mpr_source_id=mpr_source_id, mpr_age_ms=mpr_age_ms,
                    mpr_ms=getattr(self, 'mpr_ms', 0) if layer else 0, mpr_status=mpr_status,
                    slice_percent=rendered_percent, opacity=self.opacity,
                    camera_linked_slice=self.camera_linked_slice,
                    slice_mask_enabled=self.slice_mask_enabled,
                    slice_mask_available=bool(self.renderer is not None and self.manifest and self.manifest.get('bounds')),
                    slice_request_id=rendered_request_id,
                    slice_position=(slice_position(self.renderer, result.rvec, result.tvec, rendered_percent)
                                    if pose is not None and self.renderer is not None else None),
                    layer_opacity_mode='client',
                    plane_revision=rendered_revision, measurement=self.measurement(),
                    validation_comparison=self.validation_comparison,
                    validation_default_max_error_px=(self.args.glb_max_reprojection_error
                        if self.args is not None and hasattr(self.args, 'glb_max_reprojection_error')
                        else None), validation_default_min_coverage_percent=.5,
                    validation_default_view_count=(self.args.glb_top_k
                        if self.args is not None and hasattr(self.args, 'glb_top_k') else 3),
                    validation_reference_views=list(self.validation_reference_views)))
        metadata['correspondences'] = points
        metadata['display_K'] = matrix.tolist() if matrix is not None else None
        metadata['display_distortion'] = np.asarray(distortion).ravel().tolist() if distortion is not None else None
        metadata['validation_faces'] = (projected_model_faces(
            self.manifest.get('bounds') if self.manifest else None,
            self.validation_baseline_source[1], matrix, distortion)
            if self.session is not None and getattr(self.args, 'matching_mode', None)
            in ('registered-aliked', 'splat-track') else [])
        metadata['quality_profile'] = self.settings['quality_profile']
        if points:
            points['image_coverage'] = points['hull_area_pixels'] / (image.shape[0]*image.shape[1])
        metadata['mpr_loaded'] = self.renderer is not None
        metadata['mpr_sampling_size'] = self.renderer.texture_size if self.renderer else None
        metadata['layer_rect'] = self.cached_layer_rect if self.is_photo and layer else None
        reader = getattr(self.localizer, 'search_progress', None)
        metadata['search'] = finite_json(reader()) if reader is not None and not self.session else dict(active=False)
        metadata['search'].update(source_id=self.search_source_id,
                                  preview_id=self.search_generation, generation=self.generation)
        if self.is_photo:
            progress=getattr(self.photo_matcher,'progress_snapshot',lambda:{})() or {}
            metadata['search']=finite_json(dict(progress,active=self.future is not None,mode='photo',
                stage=progress.get('stage','registered'),elapsed_ms=(now-getattr(self,'photo_started',now))*1000,
                view_expansion=getattr(self.photo_matcher,'photo_view_expansion',None),
                source_id=self.search_source_id,preview_id=self.search_generation,generation=self.generation))
        metadata['inspection_camera'] = None
        if matrix is not None and self.manifest and self.manifest.get('views'):
            metadata['inspection_camera'] = dict(eye=self.manifest['views'][0]['eye'],
                target=self.manifest['center'], up=metadata['world_up'], K=matrix.tolist(),
                width=image.shape[1], height=image.shape[0])
        if self.is_photo:
            key=(self.generation,id(image))
            if key!=self.photo_encoding_key:
                self.photo_encoded=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,1])[1].tobytes()
                self.photo_encoding_key=key
                self.photo_image_revision += 1
            # The immutable full-resolution photograph is served separately
            # once per browser client/revision, not in every slice packet.
            metadata['photo_image_revision'] = self.photo_image_revision
            jpeg=b''
            metadata['image_mime']='image/png'
        else:
            jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        metadata['sent_at'] = time.monotonic()
        metadata['source_age_ms'] = (metadata['sent_at'] - stamp) * 1000
        logger = getattr(self, 'diagnostics', None)
        metadata['diagnostic_log'] = logger.snapshot() if logger is not None else None
        metadata['capture_monotonic'] = stamp
        self.rate_counts['packets'] += 1
        interval = time.monotonic()-self.rate_started
        if interval >= 1.:
            self.rates = {key: count/interval for key, count in self.rate_counts.items()}
            self.rate_counts = dict.fromkeys(self.rate_counts, 0)
            self.rate_started = time.monotonic()
        metadata['pipeline_fps'] = dict(self.rates)
        with self.condition:
            # Preserve cadence phase instead of resetting after PNG/JPEG work.
            # Otherwise small capture jitter can halve an intended 30Hz gate.
            period = 1/30
            self.last_packet = now - ((now-self.last_packet) % period)
            self.sequence += 1
            metadata['sequence'] = self.sequence
            self.last_published = (image, metadata)
            self.packet = encode_packet(metadata, jpeg, layer)
            self.condition.notify_all()
        if logger is not None:
            logger.sample(metadata)

    def tick(self):
        if self.args is None:
            return
        if self.is_photo:
            self.tick_photo()
            return
        if self.demo:
            time.sleep(.015)
            frame = self.demo_frame
        else:
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError('カメラ映像の取得に失敗しました。接続を確認してください。')
        stamp = time.monotonic()
        now = stamp
        self.source_id += 1
        image, _ = resize_for_processing(frame, self.args.max_processing_width)
        matrix, distortion, label = camera_parameters(self.args, (image.shape[1], image.shape[0]))
        self.finish_validation_reprocess()
        if self.demo:
            view = self.manifest['views'][0]
            matrix = np.asarray(view['camera_matrix'], np.float64)
            transform = np.asarray(view['world_to_camera'], np.float64)
            result = LiveResult('PASS', 'DEMO known reference pose (not estimated)', 0, 0, 0, 0, 0,
                                tuple(cv2.Rodrigues(transform[:3, :3])[0].ravel()),
                                tuple(transform[:3, 3]), None, 0, 'HIDDEN')
            self.current = (image, result, matrix, distortion, stamp, self.source_id, 'DEMO')
        elif self.session is None:
            recorder = getattr(self.localizer, 'record_capture', None)
            if recorder is not None:
                recorder(image, matrix, distortion, stamp, self.source_id)
            if now - self.last_process >= 1/self.args.process_fps:
                period = 1/self.args.process_fps
                self.last_process = now - ((now-self.last_process) % period)
                started = time.monotonic()
                previous_search = getattr(self.localizer, 'search_started', None)
                result, _ = self.localizer.localize(image, matrix, distortion)
                self.rate_counts['tracking'] += 1
                if getattr(self.localizer, 'search_started', None) != previous_search:
                    self.search_source_id = getattr(self.localizer, 'search_frame_id', None) or self.source_id
                    self.search_generation += 1
                    source_image = getattr(self.localizer, 'search_frame_image', image)
                    preview, _ = resize_for_processing(source_image, 320)
                    self.search_preview = cv2.imencode('.jpg', preview)[1].tobytes()
                self.localize_ms = (time.monotonic()-started)*1000
                self.current = (image, result, matrix, distortion, stamp, self.source_id, label)
                self.current_debug = self.localizer.last_debug if result.rvec is not None else None
        if self.current and self.current[1] and self.current[1].status == 'PASS' and self.session is None:
            self.last_valid = self.current
            self.last_valid_debug = self.current_debug
            self.last_valid_reference_views = self.current_gs_reference_views()
        if self.session:
            self.publish(*self.frozen_source)
        elif self.current:
            self.publish(*self.current)

    def tick_photo(self):
        """One global matching job per photo; no LK or repeated video frames."""
        time.sleep(.015)
        self.finish_validation_reprocess()
        if not self.photo_done and self.future is None:
            image=self.photo_image
            matrix,distortion,label=camera_parameters(self.args,(image.shape[1],image.shape[0]))
            self.photo_started=time.monotonic();self.source_id+=1
            self.future_source=(image,matrix,distortion,self.photo_started,self.source_id,label)
            self.current=(image,None,matrix,distortion,self.photo_started,self.source_id,label)
            self.search_source_id=self.source_id;self.search_generation+=1
            self.photo_matcher.diagnostic_context = dict(search_id=self.photo_started,
                search_mode='photo', search_frame_id=self.source_id,
                search_capture_monotonic=self.photo_started)
            preview,_=resize_for_processing(image,320)
            self.search_preview=cv2.imencode('.jpg',preview)[1].tobytes()
            self.future=self.executor.submit(localize_photo,self.photo_matcher,image,
                                             matrix.copy(),distortion.copy())
        if self.future is not None and self.future.done():
            result,_=self.future.result();self.future=None;self.photo_done=True
            self.current=(self.future_source[0],result,*self.future_source[1:])
            self.current_debug=getattr(self.photo_matcher,'last_debug',None)
            self.localize_ms=(time.monotonic()-self.photo_started)*1000.
            expansion=getattr(self.photo_matcher,'photo_view_expansion',None)
            expanded_note=(f' · 候補view {expansion["initial_top_k"]}→{expansion["expanded_top_k"]}'
                           if expansion else '')
            self.update(detail=f'写真：{result.status} · {result.inliers} inliers'
                               f'{expanded_note} · {self.photo_name}')
            if result.status=='PASS':
                self.last_valid=self.current;self.last_valid_debug=self.current_debug
                self.last_valid_reference_views=self.current_gs_reference_views()
                if self.renderer:self.controls(dict(action='freeze'))
        if self.session:self.publish(*self.frozen_source)
        elif self.current:
            if self.photo_done and self.last_published is not None:
                previous=self.last_published[1]
                if previous.get('result')==finite_json(self.current[1].as_dict()) and previous['plane_revision']==self.control_revision:return
            if time.monotonic()-self.last_packet>=.5 or self.photo_done:self.publish(*self.current)

    def run(self):
        try:
            while not self.shutdown.is_set():
                try:
                    while True:
                        command, values = self.commands.get_nowait()
                        if command == 'start':
                            self.start(values)
                        elif command == 'stop':
                            self.stop()
                        else:
                            self.controls(values)
                except queue.Empty:
                    pass
                except Exception as exc:
                    self.update(detail=str(exc), error=str(exc))
                    if self.state['status'] == 'loading':
                        self.stop()
                        self.update(status='error', detail=str(exc))
                try:
                    if self.args is None:
                        time.sleep(.01)
                    else:
                        self.tick()
                except Exception as exc:
                    self.stop()
                    self.update(status='error', detail=str(exc), error=str(exc))
        finally:
            self.stop()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.layer_executor.shutdown(wait=True, cancel_futures=True)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, body, mime='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            # No wildcard CORS, no arbitrary filesystem-serving endpoint.
            host = self.headers.get('Host', '')
            if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                return self.respond(403, b'{}')
            route = urlparse(self.path)
            if route.path == '/api/clock':
                received = time.monotonic()*1000
                return self.respond(200, json.dumps(dict(server_receive_ms=received,
                    server_send_ms=time.monotonic()*1000)).encode())
            if route.path == '/api/config':
                return self.respond(200, json.dumps(dict(defaults=app.settings, token=app.token,
                    demo=app.demo, reference_presets=reference_presets())).encode())
            if route.path.startswith('/api/preprocess/'):
                try:
                    query = parse_qs(route.query)
                    if route.path == '/api/preprocess/status':
                        body = app.preprocessor.status()
                    elif route.path == '/api/preprocess/map':
                        body = app.preprocessor.inspect(query['source'][0], audit=query.get('audit', ['0'])[0]=='1')
                    elif route.path == '/api/preprocess/reference':
                        body = reference_details(query['source'][0], query['image'][0])
                    elif route.path == '/api/preprocess/image':
                        return self.respond(200, reference_preview(query['source'][0], query['image'][0]), 'image/jpeg')
                    elif route.path == '/api/preprocess/cloud':
                        # Sample for display only. Saved map/IDs are never reduced.
                        source = Path(query['source'][0]).expanduser().resolve()
                        with np.load(source/'object_aliked_map.npz', allow_pickle=False) as archive:
                            points = archive['xyz']
                            points = points[::max(1, math.ceil(len(points)/15000))]
                            body = dict(xyz=points.tolist())
                    else:
                        return self.respond(404, b'{}')
                    return self.respond(200, json.dumps(finite_json(body), ensure_ascii=False).encode())
                except (ValueError, OSError, KeyError, IndexError) as exc:
                    return self.respond(400, json.dumps(dict(error=str(exc)), ensure_ascii=False).encode())
            if route.path == '/api/state':
                return self.respond(200, json.dumps(finite_json(app.snapshot()), ensure_ascii=False).encode())
            if route.path == '/api/search-image':
                query = parse_qs(route.query)
                if query.get('generation', [''])[0] != str(app.generation) or query.get('preview_id', [''])[0] != str(app.search_generation):
                    return self.respond(404, b'{}')
                preview = app.search_preview
                return self.respond(200, preview, 'image/jpeg') if preview else self.respond(404, b'{}')
            if route.path == '/api/validation-image':
                query = parse_qs(route.query)
                if (query.get('generation', [''])[0] != str(app.generation)
                    or query.get('attempt', [''])[0] != str(app.validation_attempt_id)
                    or app.validation_attempt_id < 1):
                    return self.respond(404, b'{}')
                kind = query.get('kind', [''])[0]
                preview = (app.validation_baseline_preview if kind == 'baseline' else
                           app.validation_latest_preview if kind == 'latest' else None)
                return self.respond(200, preview, 'image/jpeg') if preview else self.respond(404, b'{}')
            if route.path == '/api/reference-image':
                try:
                    query = parse_qs(route.query)
                    if query.get('generation', [''])[0] != str(app.generation):
                        raise ValueError('Old generation')
                    index = int(query['index'][0])
                    args, manifest = app.args, app.manifest
                    if args is None or manifest is None or not 0 <= index < len(manifest['views']):
                        raise ValueError('Invalid view index')
                    path = (args.splat_views / manifest['views'][index]['image_name']).resolve()
                    path.relative_to(args.splat_views.resolve())
                    return self.respond(200, path.read_bytes(), 'image/png')
                except (KeyError, ValueError, OSError, IndexError):
                    return self.respond(404, b'{}')
            if route.path == '/api/frame':
                try:
                    after = int(parse_qs(route.query).get('after', ['0'])[0])
                except ValueError:
                    return self.respond(400, b'{}')
                with app.condition:
                    app.condition.wait_for(lambda: (app.packet is not None and app.sequence > after) or app.shutdown.is_set(), timeout=1.)
                    packet = app.packet if app.sequence > after else None
                return self.respond(200 if packet else 204, packet or b'', 'application/octet-stream')
            if route.path == '/api/photo-image':
                query = parse_qs(route.query)
                if (query.get('generation', [''])[0] != str(app.generation)
                    or query.get('revision', [''])[0] != str(app.photo_image_revision)
                    or not app.is_photo or app.photo_encoded is None):
                    return self.respond(404, b'{}')
                return self.respond(200, app.photo_encoded, 'image/png')
            if route.path == '/asset.ply':
                if app.ply is None or parse_qs(route.query).get('generation', [''])[0] != str(app.generation):
                    return self.respond(404, b'{}')
                path = app.ply
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    with path.open('rb') as stream:
                        while True:
                            chunk = stream.read(1024*1024)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            names = {'/': 'index.html', '/app.js': 'app.js', '/gs-view.js': 'gs-view.js', '/styles.css': 'styles.css', '/projection.js': 'projection.js', '/feature-overlay.js': 'feature-overlay.js', '/localization-progress.js': 'localization-progress.js', '/slice-controls.js':'slice-controls.js', '/foreground-mask.js':'foreground-mask.js', '/anatomy-demo.js':'anatomy-demo.js', '/measurement-workspace.js':'measurement-workspace.js', '/validation-diagnostics.js':'validation-diagnostics.js'}
            names['/mpr-plane.js'] = 'mpr-plane.js'
            names.update({'/preprocess': 'preprocess.html', '/preprocess.js': 'preprocess.js',
                          '/preprocess.css': 'preprocess.css', '/preprocess-progress.js':'preprocess-progress.js'})
            names['/reference-selection.js'] = 'reference-selection.js'
            names['/browser-benchmark.js'] = 'browser-benchmark.js'
            if route.path == '/playcanvas.mjs':
                path = ROOT / '.tools/gs-browser/node_modules/playcanvas/build/playcanvas.mjs'
            elif route.path in names:
                path = ROOT / 'browser_ui' / names[route.path]
            else:
                return self.respond(404, b'{}')
            mime = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css'}[path.suffix]
            self.respond(200, path.read_bytes(), mime)

        def do_POST(self):
            expected = f'http://{self.headers.get("Host", "")}'
            if self.headers.get('Origin') not in (None, expected) or not secrets.compare_digest(self.headers.get('X-ViewSync-Token', ''), app.token):
                return self.respond(403, b'{"error":"Invalid origin/token"}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if self.path=='/api/photo':
                    if not 0<length<=MAX_PHOTO_BYTES:
                        raise ValueError('写真は64 MB以下のJPEG・PNGを選んでください。')
                    if app.snapshot()['status'] not in ('idle','error'):
                        raise ValueError('停止してから写真を変更してください。')
                    image=decode_photo(self.rfile.read(length))
                    name=Path(unquote(self.headers.get('X-Photo-Name','photo')).replace('\\','/')).name[:300]
                    with app.condition:
                        if app.state['status'] not in ('idle','error'):
                            raise ValueError('停止してから写真を変更してください。')
                        app.photo_upload=(image,name)
                    return self.respond(200,json.dumps(dict(name=name,width=image.shape[1],height=image.shape[0])).encode())
                if self.path == '/api/browser-telemetry':
                    if not 0 < length <= 65536:
                        raise ValueError('Browser telemetry must be <=64 KiB')
                    values = json.loads(self.rfile.read(length))
                    if not isinstance(values,dict):raise ValueError('Expected telemetry object')
                    app.record_browser_batch(values)
                    return self.respond(202, b'{"accepted":true}')
                if not 0 < length <= 16384:
                    raise ValueError('Invalid body size')
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict):
                    raise ValueError('Expected object')
                if self.path in ('/api/select-file', '/api/recommended-map'):
                    if app.snapshot()['status'] not in ('idle', 'error'):
                        raise ValueError('停止してから登録データを変更してください。')
                    if self.path == '/api/select-file':
                        result = choose_local_file(values.get('field'))
                    else:
                        path = recommended_map(DATA_DIR/'output_aliked_object_map_enriched_pilot')
                        path, report = audited_map(path)
                        result = dict(path=str(path), cancelled=False, name=path.name,
                                      xyz_ratio=report['xyz_ratio'], xyz_observations=report['xyz_observations'])
                    return self.respond(200, json.dumps(result, ensure_ascii=False).encode())
                if self.path.startswith('/api/preprocess/'):
                    action = self.path.rsplit('/', 1)[-1]
                    with app.preprocessor.lock:
                        if action in ('start', 'activate', 'resume') and app.snapshot()['status'] not in ('idle','error'):
                            raise ValueError('ライブ／写真照合を停止してから前処理・地図切替を行ってください。')
                        if action == 'start':
                            result = app.preprocessor.start(values)
                        elif action == 'cancel':
                            app.preprocessor.cancel(values['job_id'])
                            result = dict(accepted=True)
                        elif action == 'pause':
                            result = app.preprocessor.pause(values['job_id'])
                        elif action == 'resume':
                            result = app.preprocessor.resume(values['job_id'])
                        elif action == 'activate':
                            result = app.preprocessor.activate(values['job_id'])
                            app.settings['map_path'] = result['map_path']
                        else:
                            return self.respond(404, b'{}')
                    return self.respond(200, json.dumps(result, ensure_ascii=False).encode())
                command = {'/api/start': 'start', '/api/stop': 'stop', '/api/control': 'control'}.get(self.path)
                if command is None:
                    return self.respond(404, b'{}')
                if command == 'start':
                    with app.preprocessor.lock:
                        if app.preprocessor.busy():
                            raise ValueError('前処理が実行中です。一時停止・完了・中止後にライブ照合を開始してください。')
                        if app.snapshot()['status'] not in ('idle', 'error'):
                            raise ValueError('Session is already active')
                        app.update(status='loading', detail='開始要求を受け付けました。', error=None)
                elif command == 'stop':
                    app.update(status='stopping', detail='停止要求を受け付けました。')
                app.commands.put_nowait((command, values))
                self.respond(202, b'{"accepted":true}')
            except (ValueError, OSError, KeyError, queue.Full) as exc:
                self.respond(400, json.dumps(dict(error=str(exc))).encode())
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Known reference image/pose; NOT live localization')
    parser.add_argument('--log-directory', type=Path, default=DATA_DIR / 'output_runtime_logs',
                        help='Per-session JSONL diagnostics (no camera images)')
    args = parser.parse_args(argv)
    app = BrowserApplication(args.demo, log_directory=args.log_directory)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'ViewSync Browser: {url}', flush=True)
    print('Camera remains in this process/main thread. Stop with Ctrl+C or VS Code Stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown.set()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
