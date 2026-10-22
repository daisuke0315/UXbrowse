"""Pose-local descriptor search over immutable, triangulated ALIKED landmarks.

Projected pixels are search hints, NEVER observations supplied to PnP. A viewing
cone is only a conservative visibility candidate filter, not a surface/depth
occlusion test. Unknown/occluded points must survive descriptor matching and QC.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from guided_matching import project_candidates, spatial_indices


def normalized_descriptors(values):
    values = np.asarray(values, np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


@dataclass
class LandmarkCandidates:
    point_ids: np.ndarray
    xyz: np.ndarray
    projected: np.ndarray
    descriptors: np.ndarray
    directions: np.ndarray


class PoseLandmarkBank:
    """Keep view-dependent descriptors, but identify geometry by landmark ID."""

    def __init__(self, xyz, point_ids, descriptors, directions, scores, rotations=None):
        self.xyz = np.asarray(xyz, np.float64)
        ids = np.asarray(point_ids, np.int64)
        descriptors = np.asarray(descriptors, np.float32)
        directions = np.asarray(directions, np.float32)
        scores = np.asarray(scores, np.float32)
        rotations = None if rotations is None else np.asarray(rotations, np.float32)
        valid = ((ids >= 0) & (ids < len(self.xyz)) &
                 np.isfinite(descriptors).all(axis=1) &
                 (np.linalg.norm(descriptors, axis=1) > 1e-8) &
                 np.isfinite(directions).all(axis=1) & np.isfinite(scores))
        ids, descriptors, directions, scores = (
            a[valid] for a in (ids, descriptors, directions, scores))
        if rotations is not None:
            rotations = rotations[valid]
        valid_xyz = np.isfinite(self.xyz[ids]).all(axis=1)
        ids, descriptors, directions, scores = (
            a[valid_xyz] for a in (ids, descriptors, directions, scores))
        if rotations is not None:
            rotations = rotations[valid_xyz]
        order = np.argsort(ids, kind='stable')
        self.ids = ids[order]
        self.descriptors = normalized_descriptors(descriptors[order])
        directions = directions[order]
        self.directions = directions / np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        self.scores = scores[order]
        self.rotations = None if rotations is None else rotations[order]
        self.observed_ids, self.starts = np.unique(self.ids, return_index=True)

    @classmethod
    def from_registered(cls, xyz, registered, progress=None):
        """Read existing XYZ-backed observations once; no new reconstruction."""
        ids, descriptors, directions, scores, rotations = [], [], [], [], []
        for number, (name, record) in enumerate(registered.items(), 1):
            if progress:
                progress(number, len(registered))
            with np.load(record['path'], allow_pickle=False) as archive:
                point_ids = np.asarray(archive['point_ids'], np.int64)
                stored_xyz = np.asarray(archive['xyz'], np.float64)
                keep = np.flatnonzero((point_ids >= 0) & (point_ids < len(xyz)) &
                                      np.isfinite(stored_xyz).all(axis=1))
                if not len(keep):
                    continue
                selected_ids = point_ids[keep]
                # IDs, not an approximate depth or neighbouring pixel, bind XYZ.
                consistent = np.all(np.isclose(stored_xyz[keep], xyz[selected_ids],
                                               rtol=1e-5, atol=1e-4), axis=1)
                if not consistent.all():
                    raise ValueError(f'Landmark XYZ inconsistent with map: {name}')
                ids.append(selected_ids)
                descriptors.append(archive['descriptors'][0, keep].astype(np.float32))
                directions.append(record['center'] - xyz[selected_ids])
                scores.append(archive['keypoint_scores'][0, keep].astype(np.float32))
                rotations.append(np.broadcast_to(record['rotation'], (len(keep), 3, 3)))
        if not ids:
            raise ValueError('No XYZ-backed ALIKED descriptors in registered map')
        return cls(xyz, np.concatenate(ids), np.concatenate(descriptors),
                   np.concatenate(directions), np.concatenate(scores), np.concatenate(rotations))

    def candidates(self, rvec, tvec, matrix, distortion, shape, *,
                   budget=6000, variants=2, max_view_angle=55.):
        projected, valid = project_candidates(self.xyz, rvec, tvec, matrix,
                                              distortion, shape)
        rotation = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
        center = -rotation.T @ np.asarray(tvec, np.float64).reshape(3)
        directions = center - self.xyz
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        similarity = np.einsum('ij,ij->i', self.directions, directions[self.ids])
        eligible = valid[self.ids] & (similarity >= math.cos(math.radians(max_view_angle)))
        preference = similarity.copy()
        if self.rotations is not None:
            # ALIKED appearance depends on roll as well as the ray direction.
            # Ranking observations uses pose metadata, not photo retrieval.
            orientation = np.clip((np.einsum('nij,ij->n', self.rotations, rotation)-1.)/2., -1., 1.)
            preference += .5*orientation
        observations = np.flatnonzero(eligible)
        # Best observed viewing directions per landmark; no discrete photo/view
        # selection and no descriptor averaging across opposite faces.
        rank = np.lexsort((-self.scores[observations], -preference[observations],
                           self.ids[observations]))
        observations = observations[rank]
        selected_ids, starts, counts = np.unique(self.ids[observations],
                                               return_index=True, return_counts=True)
        chosen_points = spatial_indices(projected[selected_ids],
                                        similarity[observations[starts]], budget)
        chosen = []
        for i in chosen_points:
            chosen.extend(observations[starts[i]:starts[i]+min(variants, counts[i])])
        chosen = np.asarray(chosen, np.int64)
        point_ids = self.ids[chosen]
        return LandmarkCandidates(point_ids, self.xyz[point_ids], projected[point_ids],
                                  self.descriptors[chosen], similarity[chosen])


def match_projected_landmarks(query_uv, query_descriptors, candidates, *,
                              radius, ratio=.85, max_distance=.9, neighbours=64):
    """Local mutual NN + ratio across distinct IDs, not descriptor variants.

    Radius is in original image pixels. Returned indices point at ACTUAL query
    keypoints and bank observations; callers must never substitute predictions.
    Bounds keep memory/work proportional to query features * neighbours.
    """
    if radius <= 0 or not 0 < ratio < 1 or neighbours < 2:
        raise ValueError('Invalid local descriptor search limits')
    pixels = np.asarray(query_uv, np.float64)
    descriptors = normalized_descriptors(query_descriptors)
    if not len(pixels) or not len(candidates.point_ids):
        return np.empty((0, 2), np.int64), np.empty(0, np.float32), dict(
            neighbourhood_pairs=0, descriptor_pass=0, mutual_pass=0)
    cells = {}
    for i, uv in enumerate(candidates.projected):
        key = tuple(np.floor(uv / radius).astype(int))
        cells.setdefault(key, []).append(i)
    query_indices, reference_indices = [], []
    for qi, uv in enumerate(pixels):
        if not np.isfinite(uv).all() or not np.isfinite(descriptors[qi]).all():
            continue
        x, y = np.floor(uv / radius).astype(int)
        pool = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for i in cells.get((x+dx, y+dy), ())]
        if not pool:
            continue
        pool = np.asarray(pool, np.int64)
        distances = np.linalg.norm(candidates.projected[pool] - uv, axis=1)
        order = np.argsort(distances, kind='stable')
        pool = pool[order[distances[order] <= radius]][:neighbours*2]
        query_indices.extend([qi]*len(pool))
        reference_indices.extend(pool)
    q = np.asarray(query_indices, np.int64)
    r = np.asarray(reference_indices, np.int64)
    if not len(q):
        return np.empty((0, 2), np.int64), np.empty(0, np.float32), dict(
            neighbourhood_pairs=0, descriptor_pass=0, mutual_pass=0)
    distances = np.sqrt(np.maximum(0., 2. - 2.*np.einsum(
        'ij,ij->i', descriptors[q], candidates.descriptors[r])))
    # Per-query, distinct-landmark minima. Two descriptor variants of the same
    # physical point must not be each other's Lowe runner-up.
    order = np.lexsort((distances, candidates.point_ids[r], q))
    identities = np.column_stack((q[order], candidates.point_ids[r[order]]))
    first = np.r_[True, np.any(identities[1:] != identities[:-1], axis=1)]
    unique = order[first]
    ranked = unique[np.lexsort((distances[unique], q[unique]))]
    _, starts, counts = np.unique(q[ranked], return_index=True, return_counts=True)
    proposals = []
    for start, count in zip(starts, counts):
        if count < 2:  # No ambiguity evidence: do not invent a passing ratio.
            continue
        best, second = ranked[start:start+2]
        if distances[best] <= max_distance and distances[best] < ratio*distances[second]:
            proposals.append(int(best))
    # Mutual best QUERY per landmark among all local competitors, including
    # those rejected by their forward ratio test.
    reverse = unique[np.lexsort((q[unique], distances[unique], candidates.point_ids[r[unique]]))]
    point_ids, starts = np.unique(candidates.point_ids[r[reverse]], return_index=True)
    best_queries = dict(zip(point_ids.tolist(), q[reverse[starts]].tolist()))
    accepted = [i for i in proposals if best_queries[int(candidates.point_ids[r[i]])] == q[i]]
    pairs = np.column_stack((q[accepted], r[accepted])).astype(np.int64)
    scores = (1.-distances[accepted]**2/2.).astype(np.float32)
    return pairs, scores, dict(neighbourhood_pairs=len(q), descriptor_pass=len(proposals),
                              mutual_pass=len(accepted))
