"""Explicit LPS/mm and world-to-camera geometry; no image or UI dependencies."""
import numpy as np


def similarity(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('GCP coordinates must be matching Nx3 arrays')
    if len(source) < 3 or not np.isfinite([source, target]).all():
        raise ValueError('At least three finite GCPs are required')
    x, y = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError('Collinear/coincident GCPs cannot constrain Sim(3)')
    u, values, vt = np.linalg.svd(y.T @ x / len(x))
    if np.linalg.det(u @ vt) < 0 and np.linalg.matrix_rank(x) == 3:
        raise ValueError('GCP correspondence implies a reflection')
    sign = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    scale = np.sum(values * np.diag(sign)) / np.mean(np.sum(x*x, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Sim(3) scale must be positive')
    return float(scale), rotation, target.mean(0) - scale * rotation @ source.mean(0)


def transform_pose(camera_rotation, camera_translation, scale, rotation, translation):
    r = np.asarray(camera_rotation) @ np.asarray(rotation).T
    return r, scale * np.asarray(camera_translation) - r @ np.asarray(translation)


def triangulate(poses, normalized_uv, min_angle_deg=.5):
    poses, uv = np.asarray(poses, float), np.asarray(normalized_uv, float)
    if len(poses) < 2 or poses.shape[1:] != (3, 4) or uv.shape != (len(poses), 2):
        raise ValueError('Two or more camera observations required')
    if not np.isfinite(poses).all() or not np.isfinite(uv).all():
        raise ValueError('Non-finite triangulation input')
    rows = []
    for p, (u, v) in zip(poses, uv):
        rows.extend((u*p[2]-p[0], v*p[2]-p[1]))
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-12:
        raise ValueError('Triangulation at infinity')
    xyz = vt[-1, :3] / vt[-1, 3]
    rays = []
    for pose in poses:
        if (pose[:, :3] @ xyz + pose[:, 3])[2] <= 0:
            raise ValueError('Point is behind a camera')
        center = -pose[:, :3].T @ pose[:, 3]
        ray = xyz - center
        rays.append(ray / np.linalg.norm(ray))
    angles = np.degrees(np.arccos(np.clip(np.asarray(rays) @ np.asarray(rays).T, -1, 1)))
    if np.max(angles) < min_angle_deg:
        raise ValueError('Insufficient triangulation parallax')
    return xyz
