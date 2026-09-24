"""Geometry contract for Gaussian appearance references (OpenCV pixel centers).

Alpha-composited depth is an estimate, not ground-truth surface geometry.
Low occupancy, high depth variance and discontinuities are rejected explicitly.
"""
import numpy as np


def moments_to_geometry(moments, camera_matrix, camera_to_world,
                        min_alpha=0.9, max_depth_std=3.0):
    occupancy = np.asarray(moments[..., 3], np.float64)
    denominator = np.maximum(occupancy, 1e-12)
    depth = np.asarray(moments[..., 0], np.float64) / denominator * 1000.0
    variance = np.maximum(0, np.asarray(moments[..., 1], np.float64) /
                          denominator * 1e6 - depth * depth)
    std = np.sqrt(variance)
    valid = (np.isfinite(moments).all(axis=-1) & (occupancy >= min_alpha) &
             (depth > 0) & (std <= max_depth_std))
    yy, xx = np.indices(depth.shape)
    xyz = unproject(np.stack((xx, yy), axis=-1), depth, camera_matrix, camera_to_world)
    xyz[~valid] = np.nan
    return dict(depth=depth.astype(np.float32), xyz=xyz.astype(np.float32),
                alpha=occupancy.astype(np.float32), depth_std=std.astype(np.float32),
                valid=valid)


def unproject(uv, depth, camera_matrix, camera_to_world):
    homogeneous = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), axis=-1)
    rays = np.einsum('...i,ji->...j', homogeneous, np.linalg.inv(camera_matrix))
    camera = rays * np.asarray(depth)[..., None]
    return np.einsum('...i,ji->...j', camera, camera_to_world[:3, :3]) + camera_to_world[:3, 3]


def sample_geometry(reference_points, geometry, max_depth_jump=5.0):
    """Subpixel depth, same-pixel ray. Never borrow XYZ from an adjacent surface."""
    points = np.asarray(reference_points, np.float64).reshape(-1, 2)
    depth, valid = geometry['depth'], geometry['valid']
    h, w = depth.shape
    values, indices = [], []
    rejected = dict(outside=0, invalid_support=0, depth_edge=0)
    for index, (u, v) in enumerate(points):
        if not np.isfinite((u, v)).all() or u < 0 or v < 0 or u >= w-1 or v >= h-1:
            rejected['outside'] += 1
            continue
        x, y = int(np.floor(u)), int(np.floor(v))
        if not valid[y:y+2, x:x+2].all():
            rejected['invalid_support'] += 1
            continue
        patch = depth[y:y+2, x:x+2].astype(np.float64)
        if not np.isfinite(patch).all() or np.ptp(patch) > max_depth_jump:
            rejected['depth_edge'] += 1
            continue
        dx, dy = u-x, v-y
        z = ((1-dy) * ((1-dx)*patch[0, 0]+dx*patch[0, 1]) +
             dy * ((1-dx)*patch[1, 0]+dx*patch[1, 1]))
        xyz = unproject(np.asarray((u, v)), z, geometry['camera_matrix'],
                        geometry['camera_to_world'])
        values.append(tuple(float(x) for x in xyz))
        indices.append(index)
    return values, indices, rejected
