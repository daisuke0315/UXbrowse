"""Track exact query pixels while their 3D correspondence is still unknown."""
import threading

import cv2
import numpy as np

from realtime_pnp_server import TrackingSeed


def publish_query_points(engine, image, pixels):
    callback = getattr(engine, 'query_tracking_callback', None)
    if callback is not None:
        callback(getattr(engine, 'diagnostic_context', {}).get('search_id'),
                 cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pixels)


class SearchPointTracks:
    def __init__(self, *, flow_window, flow_levels, fb_error):
        self.lock = threading.Lock()
        self.lk = dict(winSize=(flow_window, flow_window), maxLevel=flow_levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        self.fb_error = fb_error
        self.begin(None)

    def begin(self, search_id):
        # Only the foreground changes active tracks. The worker publishes a
        # copied source snapshot under this short lock, never does LK here.
        with self.lock:
            self.search_id = search_id
            self.pending = None
            self.published = False
        self.initialized = False
        self.gray = None
        self.source_gray = None
        self.source = self.points = np.empty((0, 2), np.float32)
        self.indices = np.empty(0, np.int64)
        self.frames = 0
        self.last_stamp = None
        self.reason = 'waiting_features'

    def publish(self, search_id, gray, pixels):
        with self.lock:
            if search_id is None or search_id != self.search_id or self.published:
                return
            self.pending = (np.asarray(gray).copy(), np.asarray(pixels, np.float32).reshape(-1, 2).copy())
            self.published = True

    def advance(self, frames, stamp):
        with self.lock:
            pending, self.pending = self.pending, None
        if pending is not None:
            self.gray, self.source = pending
            self.source_gray = self.gray
            self.points = self.source.copy()
            self.indices = np.arange(len(self.source))
            self.initialized = True
            self.reason = 'tracking'
        if not self.initialized:
            return
        for gray in frames:
            if self.reason != 'tracking':
                break
            if gray.shape != self.gray.shape:
                self.reason = 'shape_changed'
                break
            if len(self.points) < 4:
                self.reason = 'too_few_points'
                break
            p = self.points.reshape(-1, 1, 2)
            forward, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, p, None, **self.lk)
            if forward is None or status is None:
                self.reason = 'forward_flow_failed'
                break
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, forward, None, **self.lk)
            if backward is None or back_status is None:
                self.reason = 'backward_flow_failed'
                break
            uv = forward.reshape(-1, 2)
            error = np.linalg.norm((backward-p).reshape(-1, 2), axis=1)
            valid = ((status.ravel() != 0) & (back_status.ravel() != 0) &
                     np.isfinite(uv).all(axis=1) & np.isfinite(error) & (error <= self.fb_error) &
                     (uv[:,0] >= 0) & (uv[:,0] < gray.shape[1]) &
                     (uv[:,1] >= 0) & (uv[:,1] < gray.shape[0]))
            self.points, self.indices = uv[valid], self.indices[valid]
            self.gray = gray
            self.frames += 1
            if len(self.points) < 4:
                self.reason = 'too_few_points'
        self.last_stamp = stamp

    def attach(self, seed):
        info = dict(method='continuous_query_lk', frames=self.frames,
                    source_points=len(self.source), retained_points=0, reason=self.reason)
        if not self.initialized or self.reason != 'tracking' or seed.gray.shape != self.gray.shape:
            return None, info
        if not np.array_equal(seed.gray, self.source_gray):
            info['reason'] = 'source_image_mismatch'
            return None, info
        # Query pixels are copied from the exact ALIKED set, including crop
        # offset. Never assign XYZ to a nearby, different feature or interpolate.
        lookup = {tuple(p): i for i, p in enumerate(self.source)}
        alive = {int(i): p for i, p in zip(self.indices, self.points)}
        uv, xyz = [], []
        for pixel, point in zip(np.asarray(seed.image_points, np.float32), seed.object_points):
            index = lookup.get(tuple(pixel))
            if index in alive:
                uv.append(alive[index]); xyz.append(point)
        info['retained_points'] = len(uv)
        if len(uv) < 4:
            info['reason'] = 'too_few_matched_tracks'
            return None, info
        info['reason'] = 'transported'
        return TrackingSeed(self.gray, np.asarray(uv, np.float64), np.asarray(xyz, np.float64),
                            seed.rvec.copy(), seed.tvec.copy()), info

    def snapshot(self):
        return dict(active=self.initialized, reason=self.reason, frames=self.frames,
                    source_points=len(self.source), retained_points=len(self.points))
