"""Bounded UI commands and isolated, immutable-input MPR rendering jobs."""
from collections import deque
import copy
import queue
import threading
import time

import cv2
import numpy as np

from realtime_pnp_server import projected_box_depth_interval


class ControlMailbox:
    """Coalesce adjacent absolute slider updates, never cross a button command."""
    def __init__(self, maxsize=128):
        self.items = deque()
        self.lock = threading.Lock()
        self.maxsize = maxsize

    def put_nowait(self, item):
        command, data = item
        with self.lock:
            if (command == 'control' and data.get('action') in ('slice', 'opacity')
                and self.items and self.items[-1][0] == command
                and self.items[-1][1].get('action') == data['action']
                and self.items[-1][1].get('generation') == data.get('generation')):
                self.items[-1] = item
                return
            if len(self.items) >= self.maxsize:
                raise queue.Full
            self.items.append(item)

    def get_nowait(self):
        with self.lock:
            if not self.items:
                raise queue.Empty
            return self.items.popleft()

    def empty(self):
        with self.lock:
            return not self.items


def slice_position(renderer, rvec, tvec, percent):
    rotation = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    eye = -rotation.T @ np.asarray(tvec).reshape(3)
    near, far = projected_box_depth_interval(eye, rotation.T[:, 2], renderer.volume_corners)
    span = float(far-near)
    return dict(span=span, offset=span*percent/100., camera_depth=float(near+span*percent/100.),
                unit=renderer.unit_label, outside=not 0 <= percent <= 100)


def render_snapshot(renderer, image, matrix, distortion, result, *, crop_layer=False,
                    native_layer=False):
    # The volume is read-only and can be shared. Controls and picking mappings
    # must not be modified by the worker on the live renderer.
    snapshot = copy.copy(renderer)
    snapshot._control_lock = threading.Lock()
    snapshot._measurement_points = []
    snapshot._zoom_center = None
    snapshot._crop_layer = crop_layer
    snapshot._native_layer = native_layer
    snapshot.set_opacity_percent(100.)
    return snapshot, image, matrix.copy(), distortion.copy(), result


def render_layer(snapshot, image, matrix, distortion, result):
    started = time.monotonic()
    layer, status = snapshot.render(image, matrix, distortion,
        np.asarray(result.rvec).reshape(3, 1), np.asarray(result.tvec).reshape(3, 1),
        layer_only=True, crop_layer=getattr(snapshot, '_crop_layer', False),
        native_layer=getattr(snapshot, '_native_layer', False))
    encoded = cv2.imencode('.png', layer, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
    return encoded, status, (time.monotonic()-started)*1000., snapshot


def project_slice_corners(corners, result, matrix, image_shape):
    """Current-camera homogeneous clip vertices; GPU interpolates UV in 3D."""
    if corners is None or result is None or result.status != 'PASS':
        return None
    rotation = cv2.Rodrigues(np.asarray(result.rvec, np.float64))[0]
    camera = np.asarray(corners) @ rotation.T + np.asarray(result.tvec)
    if camera.shape != (4, 3) or not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1e-6):
        return None
    pixels = camera @ np.asarray(matrix).T
    z = camera[:, 2]
    height, width = image_shape[:2]
    return np.column_stack((2*(pixels[:, 0]+.5*z)/width-z,
                            z-2*(pixels[:, 1]+.5*z)/height, np.zeros(4), z)).tolist()
