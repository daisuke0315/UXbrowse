"""Cheap object-relative direction and newly projected map-region triggers.

Projection is a conservative screen/depth check, not an occlusion estimator.
Fixed world regions prevent camera roll from manufacturing new image cells.
"""
import cv2
import numpy as np

from guided_matching import project_candidates


def camera_center(pose):
    return -cv2.Rodrigues(np.asarray(pose.rvec, np.float64))[0].T @ np.asarray(pose.tvec).reshape(3)


def direction_angle(first, second, object_center):
    a, b = camera_center(first)-object_center, camera_center(second)-object_center
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


class ViewpointRefresh:
    def __init__(self, xyz, object_center):
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        self.object_center = np.asarray(object_center, np.float64).reshape(3)
        self.points = np.empty((0, 3), np.float64)
        if len(xyz):
            low, high = np.percentile(xyz, (2, 98), axis=0)
            normalized = (xyz-low)/np.maximum(high-low, 1e-9)
            cells = np.minimum(np.maximum(np.floor(normalized*8), 0), 7).astype(int)
            ids = cells[:, 0]+8*cells[:, 1]+64*cells[:, 2]
            distance = np.linalg.norm(normalized-(cells+.5)/8., axis=1)
            order = np.lexsort((distance, ids))
            _, starts = np.unique(ids[order], return_index=True)
            self.points = xyz[order[starts]].copy()
        self.reset()

    def reset(self):
        self.anchor = None
        self.camera_signature = None
        self.baseline = np.zeros(len(self.points), bool)
        self.info = {}

    def _visible(self, pose, matrix, distortion, shape, margin):
        h, w = shape[:2]
        return project_candidates(self.points, pose.rvec, pose.tvec, matrix, distortion,
            shape, bbox=(margin*w, margin*h, (1.-margin)*w, (1.-margin)*h))[1]

    def reasons(self, current, anchor, matrix, distortion, shape, angle_threshold, prediction=None,
                *, early=False):
        signature = (tuple(shape[:2]), np.asarray(matrix).tobytes(),
                     None if distortion is None else np.asarray(distortion).tobytes())
        if anchor is not self.anchor or signature != self.camera_signature:
            self.anchor = anchor
            self.camera_signature = signature
            # An outer baseline and inner current viewport suppress edge jitter.
            self.baseline = self._visible(anchor, matrix, distortion, shape, -.02)
        visible = self._visible(current, matrix, distortion, shape, .02)
        new = int((visible & ~self.baseline).sum())
        fraction = new/max(1, int(visible.sum()))
        angle = direction_angle(anchor, current, self.object_center)
        forecast = None if prediction is None else direction_angle(anchor, prediction, self.object_center)
        self.info = dict(direction_delta_deg=angle, forecast_direction_delta_deg=forecast,
                         represented_regions=len(self.points), visible_regions=int(visible.sum()),
                         new_regions=new, new_region_fraction=fraction)
        reasons = []
        if angle is not None and angle >= angle_threshold:
            reasons.append('view_direction_changed')
        elif early and angle is not None and angle >= min(4., angle_threshold):
            reasons.append('view_direction_emerging')
        if new >= 4 and fraction >= .1:
            reasons.append('new_regions_visible')
        if angle is not None and forecast is not None and angle >= 3. and forecast >= 8.:
            reasons.append('view_direction_approaching')
        return reasons
