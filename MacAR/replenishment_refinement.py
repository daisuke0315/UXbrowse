"""Use fresh correspondences to correct drift only with old/new agreement."""
from types import SimpleNamespace

import cv2
import numpy as np

from guided_matching import spatial_indices


def refine_pose(current, points, xyz, image_shape, matrix, distortion, localizer):
    info = dict(accepted=False, reason='insufficient_fresh_support')
    minimum = max(20, localizer.min_inliers*2)
    if len(points) < minimum:
        return current, info
    # Balance spatial support and bound PnP cost. Feature extraction remains 2000.
    selected = spatial_indices(points, None, min(500,len(points)))
    points, xyz = np.asarray(points)[selected], np.asarray(xyz)[selected]
    ok, rvec, tvec, indices = cv2.solvePnPRansac(xyz, points, matrix, distortion,
        rvec=np.asarray(current.rvec).copy(), tvec=np.asarray(current.tvec).copy(),
        useExtrinsicGuess=True, iterationsCount=200,
        reprojectionError=localizer.ransac_threshold, confidence=.999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or indices is None or len(indices) < minimum:
        return current, info
    indices = indices.ravel()
    quality = localizer.validate_pose(xyz, points, rvec, tvec, indices,
                                      image_shape, matrix, distortion)
    if not quality['passed']:
        return current, dict(info, reason='fresh_pose_qc')
    a, b = cv2.Rodrigues(current.rvec)[0], cv2.Rodrigues(rvec)[0]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
    displacement = float(np.linalg.norm(-a.T@np.asarray(current.tvec).reshape(3)
                                        + b.T@np.asarray(tvec).reshape(3)))
    info.update(rotation_deg=angle, center_delta=displacement,
                fresh_inliers=len(indices), fresh_ratio=len(indices)/len(points))
    strong = (len(indices) >= 40 and len(indices)/len(points) >= .5
              and quality['coverage'] >= .02 and quality['mean_error'] <= 3.
              and quality['p95_error'] <= 5.)
    # A weak old set cannot indefinitely veto a much stronger independently
    # matched/PnP-validated current set. This is a new initialization, not a
    # blend or extrapolation of the old pose. Source matching already passed
    # the reference ambiguity/coarse checks before reaching this function.
    old_count = len(current.image_points)
    if (old_count < 80 and len(indices) >= max(60,old_count*2)
        and len(indices)/len(points) >= .5 and quality['coverage'] >= .02
        and quality['mean_error'] <= 3. and quality['p95_error'] <= 5.
        and angle <= 20. and displacement <= 100.):
        keep = spatial_indices(points[indices],None,min(200,len(indices)))
        selected = indices[keep]
        return SimpleNamespace(rvec=rvec,tvec=tvec,image_points=points[selected],
            object_points=xyz[selected]),dict(info,accepted=True,reason='strong_fresh_reseed',
                                            replaced_points=old_count,new_points=len(selected))
    if angle > 12. or displacement > 25.:
        return current, dict(info, reason='large_pose_change', conflict=bool(strong))
    old_xyz, old_uv = np.asarray(current.object_points), np.asarray(current.image_points)
    def errors(world, uv, rotation, translation):
        projected = cv2.projectPoints(world, rotation, translation, matrix, distortion)[0].reshape(-1,2)
        return np.linalg.norm(projected-uv,axis=1)
    old_at_old = errors(old_xyz,old_uv,current.rvec,current.tvec)
    old_at_new = errors(old_xyz,old_uv,rvec,tvec)
    fresh_at_old = errors(xyz[indices],points[indices],current.rvec,current.tvec)
    fresh_at_new = errors(xyz[indices],points[indices],rvec,tvec)
    threshold = localizer.ransac_threshold
    retained = np.flatnonzero(old_at_new <= threshold)
    info['old_support_fraction'] = float(len(retained)/max(1,len(old_xyz)))
    if len(retained) < localizer.min_inliers or len(retained) < .6*len(old_xyz):
        return current, dict(info, reason='old_points_disagree', conflict=bool(strong))
    # Each observation set gets equal weight; a large old cluster cannot win
    # simply by outnumbering newly visible regions. Robust loss caps outliers.
    cost = lambda first,second: float(np.mean(np.minimum(first,threshold)**2)
                                      +np.mean(np.minimum(second,threshold)**2))
    before, after = cost(old_at_old,fresh_at_old), cost(old_at_new,fresh_at_new)
    info.update(cost_before=before,cost_after=after)
    if after >= before*.95 or np.median(fresh_at_old)-np.median(fresh_at_new) < .25:
        return current, dict(info, reason='no_joint_improvement')
    quality = localizer.validate_pose(old_xyz,old_uv,rvec,tvec,retained,
                                      image_shape,matrix,distortion)
    if not quality['passed']:
        return current, dict(info, reason='retained_track_qc')
    # The caller still validates the complete merged set before committing.
    return SimpleNamespace(rvec=rvec,tvec=tvec,image_points=old_uv[retained],
                           object_points=old_xyz[retained]), dict(info,accepted=True,reason='joint_improvement')
