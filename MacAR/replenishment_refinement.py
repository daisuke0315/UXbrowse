"""Select live poses on one current-frame pool, never pre-gate by the old pose."""
from types import SimpleNamespace
from itertools import product

import cv2
import numpy as np

from aliked_lightglue_pnp import solve_pose
from guided_matching import spatial_indices
from realtime_pnp_server import TrackingSeed
from spatial_pose_scoring import spatial_pose_score, registered_hypothesis_rank


def _unique(uv, xyz, order, *, both=False):
    """One vote per subpixel location or exact fixed-map landmark, not vicinity."""
    selected, pixel_bins, seen_worlds = [], {}, set()
    pixels = np.floor(uv/.5).astype(np.int64).tolist()
    worlds = list(map(tuple, xyz))
    offsets = tuple(product((-1, 0, 1), repeat=2))
    for i in order:
        world = worlds[i]
        if not both and world in seen_worlds:
            continue
        x, y = pixels[i]
        nearby = [j for dx, dy in offsets for j in pixel_bins.get((x+dx, y+dy), ())]
        if nearby:
            near_uv = np.sum((uv[nearby]-uv[i])**2, axis=1) < .5**2
            if both:
                near_uv &= np.array([worlds[j] == world for j in nearby])
            if np.any(near_uv):
                continue
        selected.append(int(i))
        seen_worlds.add(world)
        pixel_bins.setdefault((x, y), []).append(int(i))
    return np.asarray(selected, int)


def _pool(current, fresh, shape):
    # Fresh measurements replace consistent duplicates. Contradictory UV/XYZ
    # assignments remain alternatives, never extra votes for a single pose.
    new_ids = spatial_indices(fresh.image_points, None, 2000)
    old_ids = spatial_indices(current.image_points, None, 2000)
    uv = np.vstack((fresh.image_points[new_ids], current.image_points[old_ids]))
    xyz = np.vstack((fresh.object_points[new_ids], current.object_points[old_ids]))
    fresh_flags = np.arange(len(uv)) < len(new_ids)
    h, w = shape[:2]
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1) &
             (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    uv, xyz, fresh_flags = uv[valid], xyz[valid], fresh_flags[valid]
    keep = _unique(uv, xyz, np.arange(len(uv)), both=True)
    duplicates = len(uv)-len(keep)
    uv, xyz, fresh_flags = uv[keep], xyz[keep], fresh_flags[keep]
    old_worlds = set(map(tuple, current.object_points))
    novel = np.array([tuple(point) not in old_worlds for point in xyz], bool)
    low = uv.min(axis=0) if len(uv) else np.zeros(2)
    high = np.maximum(uv.max(axis=0), low+1.) if len(uv) else np.ones(2)
    return SimpleNamespace(image_points=uv, object_points=xyz, fresh=fresh_flags,
        novel=novel, roi=(*low, *high), duplicates=duplicates, hypotheses=[])


def _evaluate(pool, rvec, tvec, shape, matrix, distortion, localizer, name):
    rvec, tvec = np.asarray(rvec, float).reshape(3), np.asarray(tvec, float).reshape(3)
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    xyz, uv = pool.object_points, pool.image_points
    if len(uv) < 4:
        return None
    projected = cv2.projectPoints(xyz, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected-uv, axis=1)
    depths = (np.einsum('ni,ji->nj', xyz, cv2.Rodrigues(rvec)[0]) + tvec)[:, 2]
    ids = np.flatnonzero(np.isfinite(errors) & (depths > 0) &
                         (errors <= localizer.ransac_threshold))
    ids = _unique(uv, xyz, ids[np.argsort(errors[ids], kind='stable')])
    quality = localizer.validate_pose(xyz, uv, rvec, tvec, ids, shape, matrix, distortion)
    quality['spatial_score'] = spatial_pose_score(uv[ids], errors[ids], pool.roi,
                                                  localizer.ransac_threshold)
    return dict(name=name, rvec=rvec, tvec=tvec, indices=ids, quality=quality)


def _rank(candidate):
    return registered_hypothesis_rank((None, None, candidate['quality']))


def _summary(candidate, pool):
    ids = candidate['indices']
    return dict(name=candidate['name'], evaluated_points=len(pool.image_points),
        inliers=len(ids), fresh_inliers=int(pool.fresh[ids].sum()), quality=candidate['quality'],
        pose=dict(rvec=candidate['rvec'].tolist(), tvec=candidate['tvec'].tolist()))


def select_pose(current, fresh, shape, matrix, distortion, localizer, iterations=150):
    """Return winning unique inliers, JSON-safe diagnostics, and a transient pool."""
    pool = _pool(current, fresh, shape)
    detail = dict(accepted=False, committed=False, reason='insufficient_common_support',
        old_points=len(current.image_points), source_points=len(fresh.image_points),
        common_points=len(pool.image_points), duplicates=pool.duplicates, added=0, hypotheses=[])
    if pool.fresh.sum() < 4:
        return None, detail, pool
    old = _evaluate(pool, current.rvec, current.tvec, shape, matrix, distortion, localizer, 'old')
    if old is not None:
        pool.hypotheses.append(old)
    for name, indices in [('fresh', np.flatnonzero(pool.fresh)),
                          ('joint', spatial_indices(pool.image_points, None, 2000))]:
        transfers = [SimpleNamespace(xyz=pool.object_points[i], query_uv=pool.image_points[i])
                     for i in indices]
        pose = solve_pose(transfers, matrix, distortion, localizer.ransac_threshold, iterations)
        if pose is not None:
            evaluated = _evaluate(pool, pose.rvec, pose.tvec, shape, matrix, distortion, localizer, name)
            if evaluated is not None:
                pool.hypotheses.append(evaluated)
    detail['hypotheses'] = [_summary(h, pool) for h in pool.hypotheses]
    if not pool.hypotheses:
        return None, detail, pool
    best = max(pool.hypotheses, key=_rank)
    detail['selected'] = best['name']
    if not best['quality']['passed']:
        detail.update(reason='common_pose_qc', quality=best['quality'])
        return None, detail, pool
    ids = best['indices']
    if pool.fresh[ids].sum() < 4:
        detail['reason'] = 'no_fresh_support'
        return None, detail, pool
    ids = ids[spatial_indices(pool.image_points[ids], None, 2000)]
    seed = TrackingSeed(fresh.gray.copy(), pool.image_points[ids].copy(),
        pool.object_points[ids].copy(), best['rvec'].copy(), best['tvec'].copy())
    detail.update(accepted=True, reason='common_support_selected', quality=best['quality'],
        method='common_pool',
        new_points=len(ids), fresh_inliers=int(pool.fresh[ids].sum()),
        retained_old=int((~pool.novel[ids]).sum()), added=int(pool.novel[ids].sum()),
        reinitialized=old is None or not old['quality']['passed'])
    return seed, detail, pool


def final_pose_check(seed, pool, shape, matrix, distortion, localizer):
    """Do not publish a final LK/PnP result that lost QC or spatial support."""
    quality = localizer.validate_pose(seed.object_points, seed.image_points, seed.rvec, seed.tvec,
        np.arange(len(seed.image_points)), shape, matrix, distortion)
    detail = dict(accepted=False, reason='final_current_qc', quality=quality)
    if not quality['passed']:
        return False, detail
    if len(_unique(seed.image_points, seed.object_points,
                   np.arange(len(seed.image_points)))) != len(seed.image_points):
        detail['reason'] = 'final_duplicate_support'
        return False, detail
    final = _evaluate(pool, seed.rvec, seed.tvec, shape, matrix, distortion, localizer, 'final')
    if final is None or not final['quality']['passed']:
        detail['reason'] = 'final_common_qc'
        return False, detail
    projected = cv2.projectPoints(seed.object_points, seed.rvec, seed.tvec,
                                 matrix, distortion)[0].reshape(-1, 2)
    score = spatial_pose_score(seed.image_points,
        np.linalg.norm(projected-seed.image_points, axis=1), pool.roi, localizer.ransac_threshold)
    baseline = next((h for h in pool.hypotheses if h['name'] == 'old'), None)
    detail.update(spatial_score=score, common_quality=final['quality'])
    if (baseline is not None and baseline['quality']['passed'] and
            score['score']+1e-6 < baseline['quality']['spatial_score']['score']):
        detail['reason'] = 'final_support_worse_than_old'
        return False, detail
    novelty = {tuple(point): bool(novel) for point, novel in zip(pool.object_points, pool.novel)}
    added = sum(novelty[tuple(point)] for point in seed.object_points)
    detail.update(accepted=True, reason='final_common_support_verified', added=added,
                  retained_old=len(seed.object_points)-added)
    return True, detail
