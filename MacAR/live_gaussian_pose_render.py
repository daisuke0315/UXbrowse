"""Optional asynchronous true-3DGS appearance rendering at a live PnP pose.

This is display-only. It never supplies matches, XYZ, or a pose to PnP/MPR.
The persistent Chrome/PlayCanvas process loads the PLY once; the camera loop
only submits the newest verified pose and reads the latest completed image.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT = (Path(__file__).resolve().parent / 'tools' /
          'gs_browser_renderer' / 'render_live_pose.mjs')


def find_node():
    requested = os.environ.get('VIEWSYNC_NODE')
    if requested:
        return requested if Path(requested).is_file() else None
    installed = shutil.which('node')
    if installed:
        return installed
    bundled = (Path.home() / '.cache' / 'codex-runtimes' /
               'codex-primary-runtime' / 'dependencies' / 'node' / 'bin' / 'node')
    return str(bundled) if bundled.is_file() else None


def pose_camera_request(result, camera_matrix, frame_shape, *, request_id: int):
    """Convert OpenCV world-to-camera pose into a PlayCanvas look-at camera."""
    if result.status != 'PASS' or result.rvec is None or result.tvec is None:
        return None
    rotation = cv2.Rodrigues(np.asarray(result.rvec, np.float64).reshape(3))[0]
    translation = np.asarray(result.tvec, np.float64).reshape(3)
    matrix = np.asarray(camera_matrix, np.float64)
    frame_height = int(frame_shape[0])
    if frame_height <= 0 or matrix.shape != (3, 3) or matrix[1, 1] <= 0:
        return None
    eye = -rotation.T @ translation
    forward = rotation.T @ np.asarray((0., 0., 1.))
    up = rotation.T @ np.asarray((0., -1., 0.))
    fov = math.degrees(2. * math.atan(frame_height / (2. * matrix[1, 1])))
    if not (np.isfinite(eye).all() and np.isfinite(forward).all()
            and np.isfinite(up).all() and math.isfinite(fov)):
        return None
    return {
        'id': int(request_id),
        'eye': eye.tolist(),
        'target': (eye + forward).tolist(),
        'up': up.tolist(),
        'fov': fov,
    }


class LiveGaussianPoseRenderer:
    def __init__(self, views_path: Path, frame_shape, *, width=640, max_hz=5.0):
        views_path = Path(views_path)
        with (views_path / 'manifest.json').open(encoding='utf-8') as source:
            manifest = json.load(source)
        if manifest.get('source_type') != '3dgs' or manifest.get('runtime_transform') != 'identity':
            raise ValueError('Live Gaussian pose rendering needs identity-coordinate 3DGS views')
        self.ply = Path(manifest['ply'])
        if not self.ply.is_file():
            raise FileNotFoundError(f'3DGS PLY not found: {self.ply}')
        self.node = find_node()
        if not self.node or not SCRIPT.is_file():
            raise FileNotFoundError('Node.js or the live 3DGS renderer script is missing')
        self.width = int(width)
        self.height = max(64, round(self.width * frame_shape[0] / frame_shape[1]))
        self.max_hz = float(max_hz)
        if self.width < 64 or not 0.1 <= self.max_hz <= 10:
            raise ValueError('Invalid 3DGS preview size or refresh rate')
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending = None
        self._last_queued = -math.inf
        self._next_id = 1
        self._image = None
        self._pose_time = None
        self._render_ms = 0
        self._status = 'LOADING 3DGS'
        self._process = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='gaussian-pose-qc')
        self._thread.start()

    def submit(self, result, camera_matrix, frame_shape):
        now = time.monotonic()
        if now - self._last_queued < 1. / self.max_hz:
            return
        with self._lock:
            request = pose_camera_request(
                result, camera_matrix, frame_shape, request_id=self._next_id)
            if request is None:
                return
            self._next_id += 1
            self._last_queued = now
            self._pending = (request, now)
        self._wake.set()

    def snapshot(self):
        with self._lock:
            image = None if self._image is None else self._image.copy()
            pose_time = self._pose_time
            render_ms = self._render_ms
            status = self._status
        age_ms = None if pose_time is None else max(0., (time.monotonic()-pose_time)*1000.)
        return image, age_ms, render_ms, status

    def _drain_stderr(self, stream):
        for line in stream:
            if line.strip():
                print('3DGS renderer:', line.rstrip(), flush=True)

    def _run(self):
        process = None
        try:
            process = subprocess.Popen(
                [self.node, str(SCRIPT), str(self.ply),
                 str(self.width), str(self.height)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            self._process = process
            threading.Thread(target=self._drain_stderr, args=(process.stderr,),
                             daemon=True).start()
            line = process.stdout.readline()
            ready = json.loads(line) if line else {}
            if not ready.get('ready'):
                raise RuntimeError(ready.get('error') or 'Renderer exited before ready')
            with self._lock:
                self._status = 'READY'
            print(f'3DGS pose preview ready ({self.width}x{self.height}, '
                  f'{self.max_hz:g} Hz max)', flush=True)
            while not self._stop.is_set():
                self._wake.wait(.25)
                self._wake.clear()
                with self._lock:
                    sample = self._pending
                    self._pending = None
                if sample is None:
                    continue
                request, pose_time = sample
                process.stdin.write(json.dumps(request) + '\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('3DGS renderer stopped during a frame')
                reply = json.loads(line)
                if reply.get('error'):
                    with self._lock:
                        self._status = 'ERROR: ' + reply['error'].splitlines()[0][:100]
                    continue
                if reply.get('id') != request['id']:
                    raise RuntimeError('3DGS renderer response ID mismatch')
                encoded = base64.b64decode(reply['png'], validate=True)
                image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError('3DGS renderer returned an invalid PNG')
                with self._lock:
                    self._image = image
                    self._pose_time = pose_time
                    self._render_ms = float(reply.get('render_ms', 0))
                    self._status = 'READY'
        except Exception as exc:
            with self._lock:
                self._status = f'UNAVAILABLE: {exc}'
            print(f'3DGS pose preview unavailable: {exc}', flush=True)
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write('{"command":"close"}\n')
                    process.stdin.flush()
                    process.stdin.close()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=3)
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
