"""Frozen-image information-loss experiment with fresh or fixed references."""
import copy
import math

import cv2
import numpy as np


def occlude_image(image, polygon, margin=0):
    points = np.asarray(polygon, np.float32)
    h, w = image.shape[:2]
    if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 64
            or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points[:, 0] >= w)
            or np.any(points[:, 1] >= h)
            or abs(cv2.contourArea(points)) < 4):
        raise ValueError('遮蔽領域は画像内の3〜64点で、面積のある多角形を指定してください。')
    # Intersecting edges have no unambiguous interior for this experiment.
    def cross(a, b, c):
        v, u = b-a, c-a
        return float(v[0]*u[1]-v[1]*u[0])
    for i in range(len(points)):
        a, b = points[i], points[(i+1) % len(points)]
        for j in range(i+1, len(points)):
            if j == i+1 or (i == 0 and j == len(points)-1):
                continue
            c, d = points[j], points[(j+1) % len(points)]
            if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                if np.all(np.maximum(np.minimum(a,b), np.minimum(c,d)) <=
                          np.minimum(np.maximum(a,b), np.maximum(c,d))):
                    raise ValueError('遮蔽領域の辺が交差しています。点を選び直してください。')
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    masked = image.copy()
    masked[mask != 0] = 75
    fraction = float(np.count_nonzero(mask)/mask.size)
    radius = int(math.ceil(margin))
    if radius:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (radius*2+1, radius*2+1)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    guard = max(contours, key=cv2.contourArea).astype(np.float32)
    return masked, guard, fraction


def compare_image_occlusion(engine, image, matrix, distortion, polygon,
                            max_error, min_coverage, phase=None, *,
                            reference_names=None, baseline_pose=None):
    # A small guard also rejects features created by the artificial gray edge.
    margin = 8 * max(image.shape[:2])/max(1, getattr(engine, 'query_resize', 1600))
    masked, guard, fraction = occlude_image(image, polygon, margin)
    fixed = reference_names is not None
    names = tuple(dict.fromkeys(reference_names or ()))
    ranked = None
    if fixed:
        available = (engine.registered if hasattr(engine, 'registered') else
                     {view['image_name']: index for index, view in enumerate(engine.views)})
        if not names or any(name not in available for name in names):
            raise ValueError('元PASS時の参照view／登録写真を利用できません。全体再照合でPASS後に固定し直してください。')
        if hasattr(engine, 'registered'):
            if baseline_pose is None:
                raise ValueError('固定参照の比較に使う元PASS姿勢がありません。固定し直してください。')
        else:
            ranked = [(1., 0., available[name]) for name in names]
    absent = object()
    with engine.lock:
        fields = ('last_tracking_seed', 'last_debug', 'last_diagnostics',
                  'last_reference_name', 'last_retrieval', 'last_crop_debug',
                  'last_candidate_log', 'last_aliked_crop', '_suppress_diagnostics',
                  '_feature_variant', 'query_tracking_callback',
                  'diagnostic_sink', 'max_reprojection_error', 'min_coverage', 'search_policy')
        saved = {key: getattr(engine, key, absent) for key in fields}
        with engine.progress_lock:
            progress = copy.deepcopy(engine.progress)
        try:
            engine.max_reprojection_error = float(max_error)
            engine.min_coverage = float(min_coverage)
            engine._suppress_diagnostics = True
            engine.query_tracking_callback = None
            engine.diagnostic_sink = None
            if hasattr(engine, 'registered'):
                engine.search_policy = 'exhaustive'  # Identical budget in both trials.
            def search(rgb, excluded):
                if fixed and hasattr(engine, 'registered'):
                    # Registered validation uses an RLock and returns its trial
                    # diagnostics without leaving a tracking seed behind.
                    return engine.localize_validation(rgb, matrix, distortion, excluded,
                        max_error, min_coverage, names, baseline_pose)
                if hasattr(engine, 'registered'):
                    # A new workspace forces ROI/retrieval/features to be rebuilt.
                    result = engine._search_attempts(rgb, matrix, distortion, excluded, {})[0]
                else:
                    options = (dict(ranked_override=ranked, query_workspace={}) if fixed else {})
                    result = engine._quality_attempts(lambda: engine._localize_locked(
                        rgb, matrix, distortion, excluded_polygon=excluded, **options))[0]
                return result, engine.last_debug, copy.deepcopy(engine.last_diagnostics)
            procedure = '元PASS参照を固定して局所照合・PnP' if fixed else '対象物検出と粗検索から照合'
            if phase:
                phase(f'1/2：遮蔽なし・{procedure}中…')
            baseline, _, baseline_diagnostics = search(image, None)
            if phase:
                phase(f'2/2：遮蔽あり・{procedure}中…')
            result, debug, diagnostics = search(masked, guard)
            diagnostics.update(baseline_diagnostics=baseline_diagnostics,
                               occluded_fraction=fraction,
                               search_mode='fixed_references' if fixed else 'fresh_global',
                               reference_views=list(names))
            return baseline, result, debug, diagnostics
        finally:
            for key, value in saved.items():
                if value is absent:
                    if hasattr(engine, key):
                        delattr(engine, key)
                else:
                    setattr(engine, key, value)
            with engine.progress_lock:
                engine.progress = progress
