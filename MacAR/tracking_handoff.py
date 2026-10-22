"""Bounded motion hints for reference selection, never for display or PnP priors."""
from collections import deque

import cv2
import numpy as np


class TrackingHandoff:
    def __init__(self):
        self.samples = deque(maxlen=24)
        self.info = {}
        self.prediction = None

    def clear(self):
        self.samples.clear()
        self.info = {}
        self.prediction = None

    def observe(self, now, seed, search_ms, radius, minimum_points):
        rotation = cv2.Rodrigues(np.asarray(seed.rvec, np.float64).reshape(3))[0]
        center = -rotation.T @ np.asarray(seed.tvec, np.float64).reshape(3)
        count = len(seed.image_points)
        self.samples.append((now, rotation, center, count))
        while len(self.samples) > 1 and now-self.samples[0][0] > .6:
            self.samples.popleft()
        self.prediction = None
        self.info = dict(points=count, rotation_speed_deg_s=0., lookahead_deg=0.,
                         forecast_points=float(count), horizon_s=0.)
        reasons = []
        if len(self.samples) < 3:
            return reasons
        before, old_rotation, old_center, old_count = self.samples[0]
        span = now-before
        if span < .15:
            return reasons
        # Point loss is measured BEFORE it hits the PnP minimum. Additions reset
        # the baseline naturally; a large point budget is not itself support.
        loss_rate = max(0., (old_count-count)/span)
        completion_horizon = float(np.clip(search_ms/1000., .5, 2.5))
        forecast = max(0., count-loss_rate*completion_horizon)
        if old_count-count >= max(4, old_count*.2) or (
                loss_rate > 0 and forecast < minimum_points*2 and old_count-count >= 3):
            reasons.append('points_declining')
        delta = cv2.Rodrigues(rotation @ old_rotation.T)[0].reshape(3)/span
        speed = float(np.linalg.norm(delta))
        # Long searches do not justify arbitrarily distant/invisible views.
        horizon = float(np.clip(search_ms/1000., .15, .8))
        step = delta*horizon
        angle = float(np.linalg.norm(step))
        step *= min(1., np.deg2rad(12.)/max(angle, 1e-12))
        displacement = (center-old_center)/span*horizon
        displacement *= min(1., max(1., radius)*.15/max(np.linalg.norm(displacement), 1e-12))
        predicted_rotation = cv2.Rodrigues(step)[0] @ rotation
        predicted_center = center+displacement
        if (np.degrees(np.linalg.norm(step)) >= 2. or
                np.linalg.norm(displacement) >= max(1., radius)*.01):
            self.prediction = (cv2.Rodrigues(predicted_rotation)[0].reshape(3),
                               -predicted_rotation @ predicted_center)
        self.info.update(rotation_speed_deg_s=float(np.degrees(speed)),
                         lookahead_deg=float(np.degrees(np.linalg.norm(step))),
                         forecast_points=forecast, horizon_s=horizon)
        return reasons
