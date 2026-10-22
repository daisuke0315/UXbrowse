"""Reuse fixed-camera rays and batch hypothesis projections, without looser QC.

OpenCV still handles distortion (including rational/thin-prism/tilted models).
The caller retains hypothesis order, support thresholds and landmark numbering.
"""
import math

import cv2
import numpy as np

from offline_pnp_test import quaternion_to_rotation


class TriangulationContext:
    """Job-local cache; never reuse rays across different maps/calibrations."""

    def __init__(self, full_uv, records, matrices, distortions):
        self.full_uv = full_uv
        self.records = records
        self.matrices = matrices
        self.distortions = distortions
        self.images = {}

    def image(self, name):
        if name not in self.images:
            record = self.records[name]
            rotation = quaternion_to_rotation(np, record.pose)
            translation = np.asarray(record.pose[4:], np.float64)
            matrix = self.matrices[record.camera_id]
            distortion = self.distortions[record.camera_id]
            pixels = np.asarray(self.full_uv[name], np.float64)
            rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2),
                                      matrix, distortion).reshape(-1, 2)
            self.images[name] = (rotation, translation, -rotation.T @ translation,
                                np.column_stack((rotation, translation)),
                                cv2.Rodrigues(rotation)[0], matrix, distortion,
                                pixels, rays)
        return self.images[name]

    def prepare(self, nodes):
        return PreparedTrack(nodes, self)


class PreparedTrack:
    def __init__(self, nodes, context):
        self.nodes = nodes
        self.cameras = [context.image(name) for name, _ in nodes]
        self.centers = np.asarray([camera[2] for camera in self.cameras])
        self.pixels = np.asarray([camera[7][index]
                                 for camera, (_, index) in zip(self.cameras, nodes)])
        self.rows = np.asarray([
            (camera[8][index, 0]*camera[3][2]-camera[3][0],
             camera[8][index, 1]*camera[3][2]-camera[3][1])
            for camera, (_, index) in zip(self.cameras, nodes)], np.float64)

    def errors(self, points, indices=None):
        """All hypotheses in ONE projectPoints call per observing camera."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        indices = np.arange(len(self.nodes)) if indices is None else indices
        errors = np.empty((len(points), len(indices)), np.float64)
        for column, index in enumerate(indices):
            rotation, translation, _, _, rvec, matrix, distortion, _, _ = self.cameras[index]
            uv = cv2.projectPoints(points, rvec, translation, matrix,
                                   distortion)[0].reshape(-1, 2)
            errors[:, column] = np.linalg.norm(uv-self.pixels[index], axis=1)
            depth = np.einsum('ni,i->n', points, rotation[2]) + translation[2]
            errors[depth <= 0, column] = np.inf
        return errors

    def pair_hypotheses(self, pairs, bounds, min_angle, max_error, margin=10.):
        """Same two-view DLT/QC, evaluated in batches with original pair order."""
        pairs = np.asarray(pairs, np.intp).reshape(-1, 2)
        rows = self.rows[pairs].reshape(-1, 4, 4)
        _, _, vh = np.linalg.svd(rows, full_matrices=False)
        homogeneous = vh[:, -1]
        finite = np.abs(homogeneous[:, 3]) >= 1e-10
        xyz = np.zeros((len(pairs), 3), np.float64)
        xyz[finite] = homogeneous[finite, :3] / homogeneous[finite, 3, None]
        valid = finite & np.isfinite(xyz).all(axis=1)
        valid &= (xyz >= bounds[0]-margin).all(axis=1)
        valid &= (xyz <= bounds[1]+margin).all(axis=1)
        rays = xyz[:, None, :] - self.centers[pairs]
        rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-12)
        angles = np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',
                            rays[:, 0], rays[:, 1]), -1, 1)))
        valid &= angles >= min_angle
        errors = np.full((len(pairs), len(self.nodes)), np.inf)
        if valid.any():
            errors[valid] = self.errors(xyz[valid])
        # A seed must pass the same positive-depth and reprojection checks in
        # both seed cameras. Other observations determine consensus afterwards.
        seed_errors = errors[np.arange(len(pairs))[:, None], pairs]
        valid &= np.isfinite(seed_errors).all(axis=1)
        valid &= (seed_errors <= max_error).all(axis=1)
        return xyz, errors, valid

    def triangulate(self, indices, bounds, min_angle, max_error, margin=10.):
        indices = np.asarray(indices, np.intp)
        _, _, vh = np.linalg.svd(self.rows[indices].reshape(-1, 4), full_matrices=False)
        if abs(vh[-1, 3]) < 1e-10:
            return None
        xyz = vh[-1, :3] / vh[-1, 3]
        if (not np.isfinite(xyz).all() or np.any(xyz < bounds[0]-margin)
                or np.any(xyz > bounds[1]+margin)):
            return None
        rays = xyz-self.centers[indices]
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        first, second = np.triu_indices(len(indices), 1)
        widest = np.min(np.einsum('ni,ni->n', rays[first], rays[second]))
        angle = math.degrees(math.acos(np.clip(widest, -1, 1)))
        if angle < min_angle:
            return None
        errors = self.errors(xyz, indices)[0]
        if not np.isfinite(errors).all() or np.max(errors) > max_error:
            return None
        return xyz, errors.tolist(), angle
