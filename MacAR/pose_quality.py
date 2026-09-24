"""Image-space pose checks. Passing these does not certify CT registration/TRE."""
import cv2
import numpy as np


def assess_pose(object_points, image_points, rvec, tvec, indices, image_shape,
                matrix, distortion, min_inliers, min_ratio, max_error,
                min_coverage=0.005):
    indices = np.asarray(indices, int)
    reasons = []
    if len(indices) < min_inliers:
        reasons.append('few_inliers')
    ratio = len(indices) / max(1, len(image_points))
    if ratio < min_ratio:
        reasons.append('low_inlier_ratio')
    coverage, spread, mean, p95 = 0.0, 0.0, None, None
    if len(indices) >= 4:
        xyz = np.asarray(object_points)[indices]
        uv = np.asarray(image_points)[indices]
        projected = cv2.projectPoints(xyz, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected-uv, axis=1)
        mean, p95 = float(np.mean(errors)), float(np.percentile(errors, 95))
        if not np.isfinite(errors).all() or mean > max_error:
            reasons.append('reprojection_error')
        depth = (np.einsum('ni,ji->nj', xyz, cv2.Rodrigues(rvec)[0]) + np.asarray(tvec).reshape(3))[:, 2]
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            reasons.append('behind_camera')
        h, w = image_shape[:2]
        coverage = float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32))) / (w*h))
        if coverage < min_coverage:
            reasons.append('clustered_image_support')
        singular = np.linalg.svd(xyz-xyz.mean(axis=0), compute_uv=False)
        spread = float(singular[1] / max(singular[0], 1e-12))
        if spread < 0.01:
            reasons.append('collinear_3d_support')
    return dict(passed=not reasons, reasons=reasons, inliers=int(len(indices)),
                inlier_ratio=float(ratio), coverage=coverage, spread_3d=spread,
                mean_error=mean, p95_error=p95)
