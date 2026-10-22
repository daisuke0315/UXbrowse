"""Bounded, asynchronous JSONL diagnostics; never save camera pixels/descriptors."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time


def safe_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_value(v) for v in value]
    if hasattr(value, 'shape'):
        # Accidental image/descriptor/point-array input must not enlarge logs.
        if getattr(value, 'size', 0) > 32:
            return dict(omitted_array_shape=list(value.shape))
        return safe_value(value.tolist())
    return str(value)


class RuntimeDiagnostics:
    def __init__(self, directory, *, capacity=2048):
        self.started = time.monotonic()
        self.wall_started = dt.datetime.now().astimezone()
        self.session_id = self.wall_started.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(3)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (self.session_id + '.jsonl')
        self.file = self.path.open('x', encoding='utf-8')
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.closed = False
        self.error = None
        self.sequence = self.dropped = 0
        self.last_sample = -math.inf
        self.last_status = None
        self.worker = threading.Thread(target=self._write, name='diagnostic-jsonl', daemon=True)
        self.worker.start()
        self.emit('session_open', schema_version=2,
                  wall_started=self.wall_started.isoformat(), monotonic_started=self.started,
                  sample_interval_ms=500, images_saved=False,
                  browser_telemetry_schema=1, sensor_exposure_timestamp_available=False,
                  physical_presentation_timestamp_available=False)

    def emit(self, event, **values):
        # Encode a detached small record before enqueueing: worker state can
        # change immediately after this call. No disk IO on capture/inference.
        now = time.monotonic()
        wall = dt.datetime.now().astimezone()
        with self.lock:
            if self.closed or self.error:
                return
            self.sequence += 1
            record = safe_value(values)
            record.update(session_id=self.session_id, record=self.sequence, event=event,
                          thread=threading.current_thread().name,
                          wall_time=wall.isoformat(timespec='milliseconds'),
                          elapsed_ms=(now-self.started)*1000, monotonic=now,
                          dropped_records=self.dropped)
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
            try:
                self.queue.put_nowait(encoded)
            except queue.Full:
                self.dropped += 1

    def sample(self, metadata):
        now = time.monotonic()
        status = (metadata.get('status'), metadata.get('frozen'))
        if now-self.last_sample < .5 and status == self.last_status:
            return
        self.last_sample, self.last_status = now, status
        fields = ('sequence', 'source_id', 'pose_source_id', 'status', 'frozen',
                  'input_source', 'width', 'height', 'result', 'calibration',
                  'source_age_ms', 'localize_ms', 'mpr_ms', 'mpr_status', 'capture_fps',
                  'quality_profile', 'search', 'capture_monotonic', 'pipeline_fps',
                  'mpr_source_id', 'mpr_age_ms', 'layer_mode')
        points = metadata.get('correspondences') or {}
        self.emit('frame_sample', **{k: metadata.get(k) for k in fields},
                  image_coverage=points.get('image_coverage'))

    def snapshot(self):
        return dict(session_id=self.session_id, file_name=self.path.name,
                    path=str(self.path), wall_time=dt.datetime.now().astimezone().isoformat(timespec='milliseconds'),
                    elapsed_ms=(time.monotonic()-self.started)*1000,
                    dropped_records=self.dropped, error=self.error)

    def _write(self):
        last_flush = time.monotonic()
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    line = self.queue.get(timeout=.1)
                except queue.Empty:
                    line = None
                if line is not None:
                    self.file.write(line)
                if time.monotonic()-last_flush >= 1.:
                    self.file.flush()
                    last_flush = time.monotonic()
            self.file.flush()
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            self.file.close()

    def close(self):
        if self.closed:
            return
        self.emit('session_close')
        with self.lock:
            self.closed = True
        self.stop.set()
        self.worker.join(timeout=2.)
        if self.worker.is_alive():
            self.error = 'Log flush timed out'
