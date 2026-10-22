"""Retain a recently verified observation set across a short tracking gap."""
import numpy as np


class ShortTrackRecovery:
    def __init__(self, horizon=.8, attempts=6):
        self.horizon = horizon
        self.max_attempts = attempts
        self.recovered = 0
        self.clear()

    def clear(self):
        self.seed = None
        self.reference = None
        self.matrix = self.distortion = None
        self.stamp = -float('inf')
        self.last_attempt = -float('inf')
        self.attempts = 0
        self.state = 'empty'
        self.nearby_used = False

    def remember(self, seed, reference, matrix, distortion, now):
        # TrackingSeed is immutable; subsequent tracking constructs new arrays.
        self.seed, self.reference, self.stamp = seed, reference, now
        self.matrix = np.asarray(matrix).copy()
        self.distortion = None if distortion is None else np.asarray(distortion).copy()
        self.attempts = 0
        self.state = 'tracking'
        self.nearby_used = False

    def nearby_hint(self, now):
        if self.seed is None or self.nearby_used or not 0 <= now-self.stamp <= 2.:
            return None
        return self.seed.rvec.copy(), self.seed.tvec.copy()

    def claim(self, image_shape, matrix, distortion, now):
        if self.seed is None:
            return None
        if not 0 <= now-self.stamp <= self.horizon or self.attempts >= self.max_attempts:
            self.state = 'expired'
            return None
        same_distortion = ((distortion is None and self.distortion is None) or
            (distortion is not None and self.distortion is not None and
             np.shape(distortion) == self.distortion.shape and
             np.allclose(distortion, self.distortion)))
        if (self.seed.gray.shape != image_shape[:2] or
            np.shape(matrix) != self.matrix.shape or not np.allclose(matrix, self.matrix)
            or not same_distortion):
            self.clear()
            return None
        if now-self.last_attempt < .06:
            return None
        self.attempts += 1
        self.last_attempt = now
        self.state = 'recovering'
        return self.seed

    def accepted(self):
        self.recovered += 1
        self.state = 'recovered'

    def snapshot(self, now):
        return dict(state=self.state, attempts=self.attempts, recovered=self.recovered,
                    anchor_age_ms=(now-self.stamp)*1000 if self.seed is not None else None)
