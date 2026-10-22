"""Bounded descriptor matching inside fixed-camera epipolar constraints.

Uses observed full-image pixels, not projected/nearest XYZ as measurements.
Subsequent independent multiview triangulation and QC remain mandatory.
"""
from dataclasses import dataclass
import hashlib
import json

import cv2
import numpy as np

from offline_pnp_test import quaternion_to_rotation


@dataclass
class PoseFeatures:
    rays: np.ndarray
    descriptors: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    focal: float


def prepare_features(uv, descriptors, record, matrix, distortion):
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    descriptors = np.asarray(descriptors, np.float32)
    if descriptors.ndim != 2 or len(descriptors) != len(uv):
        raise ValueError('Descriptor/pixel shape mismatch')
    rays = (cv2.undistortPoints(uv.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
            if len(uv) else np.empty((0, 2)))
    rays = np.column_stack([rays, np.ones(len(rays))])
    descriptors = descriptors/np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
    return PoseFeatures(rays, descriptors, quaternion_to_rotation(np, record.pose),
                        np.asarray(record.pose[4:], np.float64),
                        float(np.mean([matrix[0, 0], matrix[1, 1]])))


def pair_cache_key(method, camera_hash, first_hash, second_hash, epipolar_px=2.5,
                   ratio=.85, max_distance=.9):
    if method == 'lightglue':
        # Same computation identity as existing LightGlue pair results.
        value = 'aliked-lg-0.01-v1:'+camera_hash+first_hash+second_hash
    elif method == 'known-pose':
        value = json.dumps(['aliked-known-pose-v1', camera_hash, first_hash, second_hash,
                            epipolar_px, ratio, max_distance])
    else:
        raise ValueError('Unknown preprocessing matcher')
    return hashlib.sha256(value.encode()).hexdigest()


def match_features(first, second, *, epipolar_px=2.5, ratio=.85,
                   max_distance=.9, block_size=128):
    if epipolar_px <= 0 or not 0 < ratio < 1 or max_distance <= 0 or block_size < 1:
        raise ValueError('Invalid matching limits')
    n, m = len(first.rays), len(second.rays)
    info = dict(total_pairs=n*m, candidate_pairs=0, descriptor_pass=0, mutual_pass=0)
    empty = np.empty((0, 2), np.int64), np.empty(0, np.float64)
    if not n or not m:
        return *empty, info
    relative = second.rotation @ first.rotation.T
    translation = second.translation-relative @ first.translation
    if not np.isfinite(translation).all() or np.linalg.norm(translation) < 1e-9:
        return *empty, dict(info, reason='degenerate_baseline')
    x, y, z = translation
    essential = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]]) @ relative
    essential /= np.linalg.norm(essential)
    left_lines = np.einsum('ni,ji->nj', first.rays, essential)
    right_lines = np.einsum('ni,ij->nj', second.rays, essential)
    limit = (epipolar_px/((first.focal+second.focal)/2.))**2
    valid_first = (np.isfinite(first.rays).all(axis=1) & (np.abs(first.rays[:, :2]) < 1e4).all(axis=1)
                   & np.isfinite(first.descriptors).all(axis=1)
                   & (np.linalg.norm(first.descriptors, axis=1) > 1e-8))
    valid_second = (np.isfinite(second.rays).all(axis=1) & (np.abs(second.rays[:, :2]) < 1e4).all(axis=1)
                    & np.isfinite(second.descriptors).all(axis=1)
                    & (np.linalg.norm(second.descriptors, axis=1) > 1e-8))
    best = np.full(n, np.inf)
    runner_up = np.full(n, np.inf)
    nearest = np.full(n, -1, np.int64)
    reverse = np.full(m, -1, np.int64)
    reverse_cost = np.full(m, np.inf)
    for start in range(0, n, block_size):
        stop = min(n, start+block_size)
        lines = left_lines[start:stop]
        residual = (lines[:, 0, None]*second.rays[None, :, 0]
                    + lines[:, 1, None]*second.rays[None, :, 1]+lines[:, 2, None])
        denominator = ((lines[:, 0]**2+lines[:, 1]**2)[:, None]
                       +(right_lines[:, 0]**2+right_lines[:, 1]**2)[None, :])
        eligible = ((residual**2 <= limit*np.maximum(denominator, 1e-18))
                    & valid_first[start:stop, None] & valid_second[None, :]
                    & (denominator > 1e-18))
        q, r = np.nonzero(eligible)
        info['candidate_pairs'] += len(q)
        costs = np.full((stop-start, m), np.inf, np.float32)
        # Bound descriptor gather arrays too, even in nearly degenerate views.
        for offset in range(0, len(q), 8192):
            qi, ri = q[offset:offset+8192], r[offset:offset+8192]
            similarities = np.einsum('ij,ij->i', first.descriptors[start+qi], second.descriptors[ri])
            costs[qi, ri] = np.maximum(0., 2.-2.*similarities)
        nearest[start:stop] = np.argmin(costs, axis=1)
        if m >= 2:
            smallest = np.partition(costs, 1, axis=1)[:, :2]
            best[start:stop], runner_up[start:stop] = smallest[:, 0], smallest[:, 1]
        reverse_local = np.argmin(costs, axis=0)
        reverse_local_cost = costs[reverse_local, np.arange(m)]
        better = reverse_local_cost < reverse_cost
        reverse[better] = start+reverse_local[better]
        reverse_cost[better] = reverse_local_cost[better]
    passed = (np.isfinite(runner_up) & (best <= max_distance**2)
              & (best < ratio**2*runner_up))
    info['descriptor_pass'] = int(passed.sum())
    q = np.flatnonzero(passed)
    q = q[reverse[nearest[q]] == q]
    pairs = np.column_stack([q, nearest[q]])
    info['mutual_pass'] = len(pairs)
    return pairs, 1.-best[q]/max_distance**2, info
