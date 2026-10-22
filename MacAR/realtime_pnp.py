#!/usr/bin/env python3
"""Run object localization and MPR overlay from a Mac/Continuity camera."""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import concurrent.futures
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from realtime_pnp_server import (
    DEFAULT_COORDINATES,
    DEFAULT_FEATURES,
    DEFAULT_VOLUME,
    LivePnPError,
    LivePnPLocalizer,
    LiveResult,
    LocalizationDebug,
    MPRRenderer,
    TemporalGlobalPnPLocalizer,
)
from splat_pose_viewer import SplatPoseViewer
from map_preprocessing import selected_map


DEFAULT_IMAGES = DATA_DIR / "colmap_image" / "Images.txt"
DEFAULT_GLB_VIEWS = DATA_DIR / "output_glb_reference_views_72"
DEFAULT_SPLAT_VIEWS = DATA_DIR / "output_3dgs_reference_views_72_upper_negz_v2"
DEFAULT_REGISTERED_MAP = selected_map(DATA_DIR / "output_aliked_object_map_enriched_pilot")
PREVIEW_WINDOW = "ViewSync Continuity Camera PnP + MPR"


class MPRTrackbarControls:
    """OpenCV controls for slice depth, opacity, and image measurement."""

    SLICE = "MPR Slice x0.1%"
    OPACITY = "MPR Opacity %"
    ZOOM = "Measure Zoom +100%"
    SLICE_MIN = -1000
    SLICE_MAX = 1500
    SLICE_RANGE = SLICE_MAX - SLICE_MIN
    ZOOM_MAX = 1200

    def __init__(self, window_name: str, renderer: MPRRenderer) -> None:
        self.window_name = window_name
        self.renderer = renderer
        self._last_measurement_click_time = -math.inf
        self._last_measurement_click_position: Optional[tuple[int, int]] = None
        cv2.createTrackbar(
            self.SLICE,
            window_name,
            round(renderer.slice_fraction * 1000.0) - self.SLICE_MIN,
            self.SLICE_RANGE,
            self._slice_changed,
        )
        cv2.createTrackbar(
            self.OPACITY,
            window_name,
            round(renderer.opacity * 100),
            100,
            self._opacity_changed,
        )
        cv2.createTrackbar(
            self.ZOOM,
            window_name,
            round(renderer.zoom * 100) - 100,
            self.ZOOM_MAX - 100,
            self._zoom_changed,
        )
        cv2.setMouseCallback(window_name, self._mouse_event)

    def _slice_changed(self, value: int) -> None:
        logical_value = value + self.SLICE_MIN
        self.renderer.set_slice_percent(logical_value / 10.0)

    def _opacity_changed(self, value: int) -> None:
        self.renderer.set_opacity_percent(value)

    def _zoom_changed(self, value: int) -> None:
        self.renderer.set_zoom_percent(value + 100)

    def _mouse_event(self, event: int, x: int, y: int, flags: int, param) -> None:
        del param
        if event == cv2.EVENT_MOUSEWHEEL:
            self.renderer.center_zoom_on_display_point(x, y)
            raw_delta = (flags >> 16) & 0xFFFF
            direction = raw_delta - 0x10000 if raw_delta & 0x8000 else raw_delta
            if direction == 0:
                direction = 1 if flags > 0 else -1
            current = self.renderer.zoom
            requested = current * (1.25 if direction > 0 else 0.8)
            percent = int(round(np.clip(requested * 100.0, 100, self.ZOOM_MAX)))
            cv2.setTrackbarPos(self.ZOOM, self.window_name, percent - 100)
            return
        if event == cv2.EVENT_LBUTTONUP:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.renderer.center_zoom_on_display_point(x, y)
                print("MPR zoom centered at selected image point")
                return
            now = time.monotonic()
            previous = self._last_measurement_click_position
            if (
                previous is not None
                and now - self._last_measurement_click_time < 0.35
                and math.hypot(x - previous[0], y - previous[1]) <= 8.0
            ):
                return
            self._last_measurement_click_time = now
            self._last_measurement_click_position = (x, y)
            distance = self.renderer.add_measurement_image_point(x, y)
            if distance is not None:
                print(f"MPR measurement: {distance:.2f} {self.renderer.unit_label}")
            elif self.renderer.measurement_point_count() == 1:
                print("MPR first measurement point selected")
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.renderer.center_zoom_on_display_point(x, y)
            print("MPR zoom centered at selected image point")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.renderer.clear_measurement()
            print("MPR measurement cleared")


class AsyncReferenceLocalizer:
    """Keep camera/AF display responsive while reference search runs."""

    def __init__(self, localizer, result_hold_seconds: float = 0.75,
                 show_live_during_hold: bool = False) -> None:
        self.localizer = localizer
        self.map_data = localizer.map_data
        self.mpr_renderer = localizer.mpr_renderer
        self.last_debug = localizer.last_debug
        self.result_hold_seconds = result_hold_seconds
        self.show_live_during_hold = show_live_during_hold
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reference-pnp"
        )
        self.future = None
        self.last_result = None
        self.last_display = None
        self.last_sample_frame = None
        self.search_frame = None
        self.last_pass_result = None
        self.last_pass_frame = None
        self.last_pass_time = 0.
        self.discard_future = False
        self.show_result_until = 0.0
        self.search_started = 0.0

    def localize(self, image, camera_matrix, distortion):
        now = time.monotonic()
        if self.future is not None and self.future.done():
            if not self.discard_future:
                self.last_result, self.last_display = self.future.result()
                self.last_sample_frame = self.search_frame
                self.last_debug = self.localizer.last_debug
                if self.last_result.status == 'PASS':
                    self.last_pass_result = self.last_result
                    self.last_pass_frame = self.last_sample_frame
                    self.last_pass_time = now
            self.future = None
            self.discard_future = False
            if self.last_result is not None:
                self.show_result_until = now + self.result_hold_seconds
        if self.future is None:
            self.search_started = now
            self.search_frame = image.copy()
            self.future = self.executor.submit(
                self.localizer.localize,
                self.search_frame.copy(),
                camera_matrix.copy(),
                distortion.copy(),
            )
        if self.last_result is not None and now < self.show_result_until:
            if self.show_live_during_hold:
                canvas = image.copy()
                height, width = canvas.shape[:2]
                thumbnail_width = max(160, min(320, width // 4))
                if self.last_display is not None:
                    sample = cv2.resize(
                        self.last_display,
                        (thumbnail_width, max(1, round(
                            self.last_display.shape[0] * thumbnail_width /
                            self.last_display.shape[1]))),
                        interpolation=cv2.INTER_AREA,
                    )
                    sample_height = min(sample.shape[0], height // 3)
                    sample = sample[:sample_height]
                    left = width - thumbnail_width - 10
                    top = height - sample_height - 12
                    canvas[top:top+sample_height, left:left+thumbnail_width] = sample
                    cv2.putText(canvas, 'LAST PnP SAMPLE', (left, max(18, top-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255),
                                1, cv2.LINE_AA)
                cv2.rectangle(canvas, (0, 0), (width, 68), (20, 20, 20), -1)
                cv2.putText(canvas,
                    f"LIVE camera | previous PnP: {self.last_result.status} "
                    f"{self.last_result.inliers} inliers",
                    (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (50, 230, 70) if self.last_result.status == 'PASS'
                    else (0, 190, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, self.last_result.detail[:95],
                            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (230, 230, 230), 1, cv2.LINE_AA)
                return self.last_result, canvas
            return self.last_result, self.last_display.copy()
        elapsed = max(0.0, now - self.search_started)
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"SEARCH  {getattr(self.localizer, 'source_label', 'reference')}  {elapsed:.1f} s",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 190, 255), 2, cv2.LINE_AA,
        )
        previous = "none"
        if self.last_result is not None:
            previous = (
                f"{self.last_result.status}, {self.last_result.inliers} inliers, "
                f"{self.last_result.detail}"
            )
        cv2.putText(
            canvas,
            f"Camera remains live; previous result: {previous}",
            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (240, 240, 240), 1, cv2.LINE_AA,
        )
        result = LiveResult(
            "FAIL", "REFERENCE_SEARCH_RUNNING", 0, 0, 0, 0.0,
            None, None, None, None, elapsed * 1000.0, "HIDDEN",
        )
        return result, canvas

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def clear_previous_result(self) -> None:
        """Ignore pre-freeze searches when returning to live localization."""
        self.last_result = None
        self.last_display = None
        self.last_sample_frame = None
        self.last_pass_result = None
        self.last_pass_frame = None
        self.last_pass_time = 0.
        self.show_result_until = 0.
        self.discard_future = self.future is not None


class FFmpegCapture:
    """MJPEG pipe from AVFoundation; avoids OpenCV's unstable device indexing."""

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        *,
        ffmpeg_path: Optional[str] = None,
        framerate: float = 30.0,
    ) -> None:
        executable = ffmpeg_path or shutil.which("ffmpeg")
        if not executable:
            raise LivePnPError("FFmpeg was not found. Install it or use --capture-backend opencv.")
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "avfoundation",
            "-framerate",
            f"{framerate:g}",
        ]
        if width > 0 and height > 0:
            command.extend(("-video_size", f"{width}x{height}"))
        command.extend(
            (
                "-i",
                f"{index}:none",
                "-an",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "mjpeg",
                "-q:v",
                "4",
                "-f",
                "image2pipe",
                "pipe:1",
            )
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.buffer = bytearray()
        self.error_lines: list[str] = []
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.latest_sequence = 0
        self.delivered_sequence = 0
        self.skipped_frames = 0
        self.capture_fps = 0.0
        self.stopping = False
        self.stream_ended = False
        self.last_frame_time = 0.0
        self.error_thread = threading.Thread(
            target=self._drain_errors, name="ffmpeg-camera-errors", daemon=True
        )
        self.error_thread.start()
        self.frame_thread = threading.Thread(
            target=self._capture_latest_frames,
            name="ffmpeg-latest-frame",
            daemon=True,
        )
        self.frame_thread.start()

    def _drain_errors(self) -> None:
        if self.process.stderr is None:
            return
        for raw in iter(self.process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self.error_lines.append(line)
                del self.error_lines[:-20]

    def _read_one(self):
        if self.process.stdout is None:
            return False, None
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    jpeg = bytes(self.buffer[start : end + 2])
                    del self.buffer[: end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
            chunk = self.process.stdout.read(65_536)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)
            if len(self.buffer) > 32_000_000:
                latest = self.buffer.rfind(b"\xff\xd8")
                self.buffer = self.buffer[latest:] if latest >= 0 else bytearray()

    def _capture_latest_frames(self) -> None:
        previous_time = 0.0
        while not self.stopping:
            ok, frame = self._read_one()
            if not ok or frame is None:
                break
            timestamp = time.monotonic()
            with self.condition:
                if previous_time > 0:
                    instantaneous = 1.0 / max(timestamp - previous_time, 1e-9)
                    self.capture_fps = (
                        instantaneous
                        if self.capture_fps == 0
                        else self.capture_fps * 0.9 + instantaneous * 0.1
                    )
                previous_time = timestamp
                self.latest_frame = frame
                self.latest_frame_time = timestamp
                self.latest_sequence += 1
                self.condition.notify_all()
        with self.condition:
            self.stream_ended = True
            self.condition.notify_all()

    def read(self):
        deadline = time.monotonic() + 8.0
        with self.condition:
            while (
                self.latest_sequence <= self.delivered_sequence
                and not self.stream_ended
                and not self.stopping
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self.condition.wait(timeout=remaining)
            if self.latest_sequence <= self.delivered_sequence or self.latest_frame is None:
                return False, None
            self.skipped_frames += max(
                0, self.latest_sequence - self.delivered_sequence - 1
            )
            self.delivered_sequence = self.latest_sequence
            self.last_frame_time = self.latest_frame_time
            return True, self.latest_frame

    def stats(self):
        with self.condition:
            return {
                "capture_fps": self.capture_fps,
                "skipped_frames": self.skipped_frames,
                "frame_time": self.last_frame_time,
                "sequence": self.delivered_sequence,
            }

    def error_detail(self) -> str:
        return " | ".join(self.error_lines[-4:])

    def release(self) -> None:
        self.stopping = True
        with self.condition:
            self.condition.notify_all()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()


class OpenCVDirectCapture:
    """Read AVFoundation synchronously, matching the minimal camera test."""

    def __init__(self, index: int, width: int, height: int) -> None:
        backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        for attempt in range(3):
            self.capture = cv2.VideoCapture(index, backend)
            if self.capture.isOpened():
                break
            self.capture.release()
            if attempt < 2:
                time.sleep(0.2)
        else:
            raise LivePnPError(
                f"Camera {index} could not be opened. Check macOS Camera permission and --camera."
            )
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture_fps = 0.0
        self.last_frame_time = 0.0
        self.previous_frame_time = 0.0
        self.sequence = 0
        self.read_failures = 0

    def read(self):
        ok, frame = self.capture.read()
        timestamp = time.monotonic()
        if not ok or frame is None:
            self.read_failures += 1
            return False, None
        if self.previous_frame_time > 0:
            instantaneous = 1.0 / max(timestamp - self.previous_frame_time, 1e-9)
            self.capture_fps = (
                instantaneous
                if self.capture_fps == 0
                else self.capture_fps * 0.9 + instantaneous * 0.1
            )
        self.previous_frame_time = timestamp
        self.last_frame_time = timestamp
        self.sequence += 1
        return True, frame

    def stats(self):
        return {
            "capture_fps": self.capture_fps,
            "skipped_frames": 0,
            "frame_time": self.last_frame_time,
            "sequence": self.sequence,
        }

    def error_detail(self) -> str:
        return f"OpenCV frame read failures: {self.read_failures}"

    def release(self) -> None:
        self.capture.release()


def read_camera_startup(capture):
    """Allow transient empty AVFoundation frames on (re)connect, not forever."""
    for attempt in range(20):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame
        if attempt < 19:
            time.sleep(0.1)
    return False, None


class OpenCVLatestFrameCapture:
    """Continuously drain AVFoundation and expose only its newest frame."""

    def __init__(self, index: int, width: int, height: int) -> None:
        backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise LivePnPError(
                f"Camera {index} could not be opened. Check macOS Camera permission and --camera."
            )
        # Zero leaves AVFoundation at its native setting, matching the simple
        # cv2.VideoCapture(..., CAP_AVFOUNDATION) camera test.
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.latest_sequence = 0
        self.delivered_sequence = 0
        self.skipped_frames = 0
        self.capture_fps = 0.0
        self.last_frame_time = 0.0
        self.stopping = False
        self.stream_ended = False
        self.read_failures = 0
        self.frame_thread = threading.Thread(
            target=self._capture_latest_frames,
            name="opencv-latest-frame",
            daemon=True,
        )
        self.frame_thread.start()

    def _capture_latest_frames(self) -> None:
        previous_time = 0.0
        consecutive_failures = 0
        while not self.stopping:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                self.read_failures += 1
                if consecutive_failures >= 30:
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            timestamp = time.monotonic()
            with self.condition:
                if previous_time > 0:
                    instantaneous = 1.0 / max(timestamp - previous_time, 1e-9)
                    self.capture_fps = (
                        instantaneous
                        if self.capture_fps == 0
                        else self.capture_fps * 0.9 + instantaneous * 0.1
                    )
                previous_time = timestamp
                self.latest_frame = frame
                self.latest_frame_time = timestamp
                self.latest_sequence += 1
                self.condition.notify_all()
        with self.condition:
            self.stream_ended = True
            self.condition.notify_all()

    def read(self):
        deadline = time.monotonic() + 8.0
        with self.condition:
            while (
                self.latest_sequence <= self.delivered_sequence
                and not self.stream_ended
                and not self.stopping
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self.condition.wait(timeout=remaining)
            if self.latest_sequence <= self.delivered_sequence or self.latest_frame is None:
                return False, None
            self.skipped_frames += max(
                0, self.latest_sequence - self.delivered_sequence - 1
            )
            self.delivered_sequence = self.latest_sequence
            self.last_frame_time = self.latest_frame_time
            return True, self.latest_frame

    def stats(self):
        with self.condition:
            return {
                "capture_fps": self.capture_fps,
                "skipped_frames": self.skipped_frames,
                "frame_time": self.last_frame_time,
                "sequence": self.delivered_sequence,
            }

    def error_detail(self) -> str:
        return f"OpenCV frame read failures: {self.read_failures}"

    def release(self) -> None:
        self.stopping = True
        with self.condition:
            self.condition.notify_all()
        if self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1.0)
        self.capture.release()
        if self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2.0)


def open_camera(
    index: int,
    width: int,
    height: int,
    *,
    capture_backend: str = "ffmpeg",
    ffmpeg_path: Optional[str] = None,
):
    if capture_backend == "ffmpeg":
        return FFmpegCapture(index, width, height, ffmpeg_path=ffmpeg_path)
    if capture_backend == "opencv-latest":
        return OpenCVLatestFrameCapture(index, width, height)
    return OpenCVDirectCapture(index, width, height)


def ffmpeg_camera_devices(ffmpeg_path: Optional[str] = None):
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise LivePnPError("FFmpeg was not found")
    completed = subprocess.run(
        (
            executable,
            "-hide_banner",
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    devices = []
    in_video_section = False
    for line in completed.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match and not match.group(2).startswith("Capture screen"):
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def list_cameras(max_index: int, *, capture_backend: str, ffmpeg_path=None) -> int:
    if capture_backend == "ffmpeg":
        print("Camera devices (FFmpeg / AVFoundation):")
        devices = ffmpeg_camera_devices(ffmpeg_path)
        for index, name in devices:
            print(f"  {index}: {name}")
        if not devices:
            print("No AVFoundation camera devices were found.")
            return 2
        print("Use the displayed number with --camera. FFmpeg keeps this mapping stable.")
        return 0

    print("Camera index scan (OpenCV / AVFoundation):")
    found = 0
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index, backend)
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        capture.release()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"  {index}: available ({width}x{height})")
            found += 1
        else:
            print(f"  {index}: unavailable")
    if found == 0:
        print("No camera frames were found. Allow camera access for Terminal/Python first.")
        return 2
    print("OpenCV exposes indices, not AVFoundation device names; try each available index.")
    return 0


def load_calibration(path: Path):
    if not path.is_file():
        raise LivePnPError(f"Calibration file not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            matrix = data["camera_matrix"].astype(np.float64, copy=True)
            distortion = data["distortion"].astype(np.float64, copy=True).reshape(-1)
            image_size = tuple(int(value) for value in data["image_size"].tolist())
    except (OSError, KeyError, ValueError) as exc:
        raise LivePnPError(f"Invalid camera calibration file: {path}: {exc}") from exc
    if matrix.shape != (3, 3) or len(image_size) != 2 or distortion.size < 4:
        raise LivePnPError(f"Invalid camera calibration arrays: {path}")
    return matrix, distortion, image_size


def scaled_calibration(matrix, source_size, target_size):
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    if abs(scale_x - scale_y) > 0.01:
        raise LivePnPError(
            "Camera frame aspect ratio differs from calibration. Use the same resolution/crop."
        )
    result = matrix.copy()
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result


def approximate_camera_matrix(width: int, height: int, horizontal_fov_deg: float):
    if not 10 < horizontal_fov_deg < 170:
        raise LivePnPError("--fov-deg must be between 10 and 170")
    focal = width * 0.5 / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    return np.asarray(
        ((focal, 0, width * 0.5), (0, focal, height * 0.5), (0, 0, 1)),
        dtype=np.float64,
    )


def camera_parameters(args, frame_size):
    width, height = frame_size
    if args.calibration is not None:
        matrix, distortion, source_size = load_calibration(args.calibration)
        return (
            scaled_calibration(matrix, source_size, frame_size),
            distortion,
            f"CALIBRATED {args.calibration.name}",
        )
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        matrix = np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
        return matrix, np.asarray(args.distortion, dtype=np.float64), "MANUAL K"
    return (
        approximate_camera_matrix(width, height, args.fov_deg),
        np.zeros(5, dtype=np.float64),
        f"APPROX K / HFOV {args.fov_deg:.1f}deg",
    )


class PointCloud3DViewer:
    """Small dependency-free 3D QC view rendered into an OpenCV window."""

    BOX_EDGES = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )

    def __init__(self, localizer: LivePnPLocalizer, width: int = 900, height: int = 760):
        point_ids = localizer.map_data.point3d_ids
        _, unique_indices = np.unique(point_ids, return_index=True)
        self.points = localizer.map_data.xyz[np.sort(unique_indices)].astype(
            np.float64, copy=False
        )
        if localizer.mpr_renderer is not None:
            self.minimum = localizer.mpr_renderer.box_minimum.copy()
            self.maximum = localizer.mpr_renderer.box_maximum.copy()
        else:
            self.minimum = self.points.min(axis=0)
            self.maximum = self.points.max(axis=0)
        self.center = (self.minimum + self.maximum) * 0.5
        self.radius = max(float(np.max(self.maximum - self.minimum)) * 0.62, 1.0)
        self.width = width
        self.height = height
        self.yaw = math.radians(-42.0)
        self.pitch = math.radians(24.0)

    def _basis(self):
        forward = np.asarray(
            (
                math.cos(self.pitch) * math.cos(self.yaw),
                math.cos(self.pitch) * math.sin(self.yaw),
                math.sin(self.pitch),
            ),
            dtype=np.float64,
        )
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-9:
            right = np.asarray((1.0, 0.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)
        return right, up, forward

    def _project(self, points):
        values = np.asarray(points, dtype=np.float64).reshape((-1, 3)) - self.center
        right, up, forward = self._basis()
        scale = min(self.width, self.height) * 0.42 / self.radius
        x = values @ right * scale + self.width * 0.5
        y = -(values @ up) * scale + self.height * 0.53
        depth = values @ forward
        return np.column_stack((x, y)), depth

    def _box_corners(self):
        return np.asarray(
            [
                (x, y, z)
                for x in (self.minimum[0], self.maximum[0])
                for y in (self.minimum[1], self.maximum[1])
                for z in (self.minimum[2], self.maximum[2])
            ],
            dtype=np.float64,
        )

    def render(self, debug: LocalizationDebug, status: str):
        canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        projected, depth = self._project(self.points)
        pixels = np.rint(projected).astype(np.int32)
        valid = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < self.width)
            & (pixels[:, 1] >= 72)
            & (pixels[:, 1] < self.height)
        )
        normalized_depth = np.clip(
            (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9), 0, 1
        )
        brightness = np.rint(65 + normalized_depth * 85).astype(np.uint8)
        selected = pixels[valid]
        selected_brightness = brightness[valid]
        canvas[selected[:, 1], selected[:, 0]] = np.column_stack(
            (selected_brightness, selected_brightness, selected_brightness)
        )

        box_pixels, _ = self._project(self._box_corners())
        box_pixels = np.rint(box_pixels).astype(np.int32)
        for first, second in self.BOX_EDGES:
            cv2.line(
                canvas,
                tuple(box_pixels[first]),
                tuple(box_pixels[second]),
                (0, 210, 255),
                1,
                cv2.LINE_AA,
            )

        matched_pixels, _ = self._project(debug.matched_xyz)
        matched_pixels = np.rint(matched_pixels).astype(np.int32)
        for index, pixel in enumerate(matched_pixels):
            if not (0 <= pixel[0] < self.width and 72 <= pixel[1] < self.height):
                continue
            if debug.pnp_solved:
                color = (40, 235, 70) if index in debug.inlier_indices else (50, 70, 235)
            else:
                color = (0, 180, 255)
            cv2.circle(canvas, tuple(pixel), 5, (5, 5, 5), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pixel), 3, color, -1, cv2.LINE_AA)

        if debug.camera_center_model is not None:
            camera_pixel, _ = self._project(
                np.asarray(debug.camera_center_model, dtype=np.float64)
            )
            camera_pixel = np.rint(camera_pixel[0]).astype(np.int32)
            center_pixel, _ = self._project(self.center)
            center_pixel = np.rint(center_pixel[0]).astype(np.int32)
            if 0 <= camera_pixel[0] < self.width and 72 <= camera_pixel[1] < self.height:
                cv2.line(canvas, tuple(camera_pixel), tuple(center_pixel), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(camera_pixel), 7, (255, 255, 255), 2, cv2.LINE_AA)

        axis_length = self.radius * 0.34
        origin_pixel, _ = self._project(self.center)
        origin_pixel = np.rint(origin_pixel[0]).astype(np.int32)
        for axis, color, label in (
            (np.asarray((axis_length, 0, 0)), (50, 70, 255), "X"),
            (np.asarray((0, axis_length, 0)), (50, 230, 70), "Y"),
            (np.asarray((0, 0, axis_length)), (255, 100, 40), "Z"),
        ):
            endpoint, _ = self._project(self.center + axis)
            endpoint = np.rint(endpoint[0]).astype(np.int32)
            cv2.arrowedLine(canvas, tuple(origin_pixel), tuple(endpoint), color, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(canvas, label, tuple(endpoint + (5, -5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (self.width, 72), (24, 24, 24), -1)
        inliers = len(debug.inlier_indices)
        cv2.putText(
            canvas,
            f"LIVE 3D MATCH COVERAGE | {status} | candidates {len(debug.matched_xyz)} | inliers {inliers}",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (70, 230, 80) if status == "PASS" else (60, 80, 240), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "gray: feature map  green: inlier  red: outlier  orange: PnP unsolved  yellow: object box",
            (14, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "A/D: rotate  W/S: tilt  R: reset",
            (self.width - 275, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return canvas

    def handle_key(self, key: int) -> bool:
        step = math.radians(6.0)
        if key == ord("a"):
            self.yaw -= step
        elif key == ord("d"):
            self.yaw += step
        elif key == ord("w"):
            self.pitch = min(self.pitch + step, math.radians(84))
        elif key == ord("s"):
            self.pitch = max(self.pitch - step, math.radians(-84))
        elif key == ord("r"):
            self.yaw = math.radians(-42.0)
            self.pitch = math.radians(24.0)
        else:
            return False
        return True


def temporal_view_localizer(engine, args):
    """Shared bounded background search / current-frame LK+PnP for both maps."""
    from glb_virtual_live import AsyncTemporalGLBLocalizer
    return AsyncTemporalGLBLocalizer(engine,
        reseed_frames=args.tracking_reseed_frames,
        min_track_points=args.tracking_min_points,
        flow_window=args.tracking_flow_window,
        flow_levels=args.tracking_flow_levels,
        forward_backward_error=args.tracking_fb_error,
        tracking_iterations=args.tracking_iterations,
        max_seed_age=args.virtual_max_seed_age,
        replenish_frames=args.tracking_replenish_frames,
        replenish_mode=args.tracking_replenish_mode,
        replenish_distance_mm=args.tracking_replenish_distance_mm,
        replenish_height_mm=args.tracking_replenish_height_mm,
        replenish_cooldown=args.tracking_replenish_cooldown,
        replenish_image_coverage=args.tracking_replenish_image_coverage,
        replenish_angle_deg=args.tracking_replenish_angle,
        orbit_recovery=args.tracking_orbit_recovery)


def build_localizer(args, *, temporal_registered=False):
    if args.ct_validation and args.matching_mode != "registered-aliked":
        raise LivePnPError("--ct-validation requires --matching-mode registered-aliked")
    if args.ct_validation and args.no_mpr:
        raise LivePnPError("--ct-validation cannot be combined with --no-mpr")
    if args.matching_mode == "registered-aliked" and not args.no_mpr and not args.ct_validation:
        raise LivePnPError(
            "Registered ALIKED MPR is available only in frozen CT validation. "
            "Add --ct-validation or use --no-mpr."
        )
    renderer = None
    if not args.no_mpr:
        renderer = MPRRenderer(
            args.volume,
            args.coordinates,
            texture_size=args.mpr_size,
            window_level=args.window_level,
            window_width=args.window_width,
            opacity=args.mpr_opacity,
        )
    if args.matching_mode == "registered-aliked":
        from registered_aliked_live import RegisteredALIKEDLiveLocalizer

        registered_localizer = RegisteredALIKEDLiveLocalizer(
            args.registered_map, args.splat_views,
            top_k=args.glb_top_k,
            max_keypoints=args.glb_max_keypoints,
            resize=args.glb_resize,
            match_threshold=args.glb_match_threshold,
            fundamental_threshold=args.fundamental_threshold,
            ransac_threshold=args.ransac_threshold,
            iterations=args.iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            max_reprojection_error=args.glb_max_reprojection_error,
            max_coarse_rotation_deg=args.virtual_max_coarse_rotation,
            device=args.glb_device,
            mpr_renderer=None,
            retrieval_mode=args.retrieval_mode if args.retrieval_mode != 'sift' else 'dino-patch',
            search_policy=getattr(args, 'registered_search_policy', 'adaptive'),
        )
        if args.ct_validation:
            points = np.asarray(registered_localizer.map_data.xyz, np.float64)
            inside = np.all((points >= renderer.volume_box_minimum - 10.) &
                            (points <= renderer.volume_box_maximum + 10.), axis=1)
            if renderer.unit_label != 'mm' or np.mean(inside) < .5:
                raise LivePnPError(
                    'CT volume/model bounds do not overlap the registered ALIKED map '
                    'in millimeters. Check --volume and --coordinates.'
                )
            print(f'CT validation bounds precheck: {np.mean(inside):.1%} '
                  'of ALIKED XYZ inside CT bounds (+10 mm). '
                  'This does not prove CT registration accuracy.')
        if temporal_registered:
            wrapper = temporal_view_localizer(registered_localizer, args)
            # Compatibility for frozen registered-map validation helpers.
            wrapper.localizer = registered_localizer
            wrapper.mpr_renderer = renderer
            return wrapper
        wrapper = AsyncReferenceLocalizer(
            registered_localizer,
            result_hold_seconds=args.reference_result_hold,
            show_live_during_hold=True,
        )
        wrapper.mpr_renderer = renderer
        return wrapper
    if args.matching_mode in {"glb-track", "splat-track"}:
        from glb_virtual_live import AsyncTemporalGLBLocalizer, GLBVirtualLiveLocalizer

        glb_localizer = GLBVirtualLiveLocalizer(
            args.splat_views if args.matching_mode == "splat-track" else args.glb_views,
            top_k=args.glb_top_k,
            sift_ratio=args.glb_sift_ratio,
            max_keypoints=args.glb_max_keypoints,
            resize=args.glb_resize,
            match_threshold=args.glb_match_threshold,
            fundamental_threshold=args.fundamental_threshold,
            xyz_search_radius=args.glb_xyz_search_radius,
            ransac_threshold=args.ransac_threshold,
            confidence=args.confidence,
            iterations=args.iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            max_reprojection_error=args.glb_max_reprojection_error,
            device=args.glb_device,
            mpr_renderer=renderer,
            pipeline_policy=args.virtual_policy,
            diagnostics_path=args.virtual_diagnostics,
            fast_consensus=args.virtual_fast_consensus,
            appearance_audit=args.virtual_appearance_audit,
            max_coarse_rotation_deg=args.virtual_max_coarse_rotation,
            retrieval_mode=args.retrieval_mode,
            retrieval_crop=args.retrieval_crop,
            retrieval_roi=args.retrieval_roi,
            retrieval_auto_roi=args.retrieval_auto_roi,
            guided_optimization=not args.virtual_guided_baseline,
            guided_feature_budget=args.virtual_guided_features,
            guided_pixel_radius=args.virtual_guided_radius,
            guided_early_descriptors=not args.virtual_guided_full_descriptors,
        )
        return AsyncTemporalGLBLocalizer(
            glb_localizer,
            reseed_frames=args.tracking_reseed_frames,
            min_track_points=args.tracking_min_points,
            flow_window=args.tracking_flow_window,
            flow_levels=args.tracking_flow_levels,
            forward_backward_error=args.tracking_fb_error,
            tracking_iterations=args.tracking_iterations,
            max_seed_age=args.virtual_max_seed_age,
            replenish_frames=(args.tracking_replenish_frames
                              if args.matching_mode == "splat-track" else 0),
            replenish_mode=(args.tracking_replenish_mode
                            if args.matching_mode == "splat-track" else 'interval'),
            replenish_distance_mm=args.tracking_replenish_distance_mm,
            replenish_height_mm=args.tracking_replenish_height_mm,
            replenish_cooldown=args.tracking_replenish_cooldown,
            replenish_image_coverage=args.tracking_replenish_image_coverage,
            replenish_angle_deg=args.tracking_replenish_angle,
            orbit_recovery=args.tracking_orbit_recovery,
        )
    if args.matching_mode == "reference":
        from reference_guided_live import ReferenceGuidedLiveLocalizer

        reference_localizer = ReferenceGuidedLiveLocalizer(
            args.features,
            args.images,
            shortlist=args.reference_shortlist,
            retrieval_neighbors=args.reference_neighbors,
            retrieval_distance=args.reference_distance,
            ratio_threshold=args.reference_ratio,
            fundamental_threshold=args.fundamental_threshold,
            guided_hypotheses=args.guided_hypotheses,
            guided_radii=args.guided_radii,
            guided_ratio=args.guided_ratio,
            guided_distance=args.guided_distance,
            guided_max_candidates=args.guided_max_candidates,
            matching_workers=args.reference_workers,
            ransac_threshold=args.ransac_threshold,
            iterations=args.iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            mpr_renderer=renderer,
        )
        return AsyncReferenceLocalizer(
            reference_localizer,
            result_hold_seconds=args.reference_result_hold,
        )
    global_localizer = LivePnPLocalizer(
        args.features,
        ratio_threshold=args.ratio,
        knn=args.knn,
        ransac_threshold=args.ransac_threshold,
        confidence=args.confidence,
        iterations=args.iterations,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        mpr_renderer=renderer,
    )
    if args.matching_mode == "global-track":
        return TemporalGlobalPnPLocalizer(
            global_localizer,
            reseed_frames=args.tracking_reseed_frames,
            min_track_points=args.tracking_min_points,
            flow_window=args.tracking_flow_window,
            flow_levels=args.tracking_flow_levels,
            forward_backward_error=args.tracking_fb_error,
            tracking_iterations=args.tracking_iterations,
        )
    return global_localizer


def add_banner(frame, text: str, color) -> None:
    cv2.rectangle(frame, (0, frame.shape[0] - 34), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, frame.shape[0] - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def resize_for_processing(frame, maximum_width: int):
    height, width = frame.shape[:2]
    if maximum_width <= 0 or width <= maximum_width:
        return frame, 1.0
    scale = maximum_width / width
    resized = cv2.resize(
        frame,
        (maximum_width, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def center_focus_score(frame) -> float:
    """Return a relative sharpness score for the center two-thirds of a frame."""
    height, width = frame.shape[:2]
    region = frame[height // 6 : height * 5 // 6, width // 6 : width * 5 // 6]
    if region.shape[1] > 640:
        scale = 640 / region.shape[1]
        region = cv2.resize(
            region,
            (640, max(1, round(region.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def add_performance_hud(
    frame,
    result,
    *,
    capture_fps: float,
    process_fps: float,
    skipped_frames: int,
    queue_age_ms: float,
    localize_ms: float,
    viewer_ms: float,
    pipeline_ms: float,
    focus_score: float,
    input_size,
    work_size,
    args,
    replenished: int = 0,
) -> None:
    cv2.rectangle(frame, (0, 68), (frame.shape[1], 171), (24, 24, 24), -1)
    measured_latency = queue_age_ms + pipeline_ms
    debug_3d = "OFF" if args.no_3d else f"ON {args.debug_3d_fps:g} Hz"
    if args.matching_mode == "reference":
        matcher_text = (
            f"reference shortlist {args.reference_shortlist} | "
            f"guided hypotheses {args.guided_hypotheses} | "
            f"ratio<{args.reference_ratio:g} | RANSAC {args.ransac_threshold:g}px"
        )
    elif args.matching_mode == "registered-aliked":
        matcher_text = (
            f"3DGS DINO auto-ROI -> RC ALIKED+LightGlue top-{args.glb_top_k} "
            f"-> triangulated XYZ -> PnP | RANSAC {args.ransac_threshold:g}px"
        )
    elif args.matching_mode in {"glb-track", "splat-track"}:
        source_label = "3DGS" if args.matching_mode == "splat-track" else "GLB"
        replenish_text = (f" + pose-guided/{args.tracking_replenish_frames}f"
                          if args.matching_mode == "splat-track" and args.tracking_replenish_frames
                          else "")
        if args.matching_mode == "splat-track" and args.tracking_replenish_mode in ('coverage', 'background'):
            replenish_text = (f" + dynamic XYZ {args.tracking_replenish_distance_mm:g}mm"
                              f"/height {args.tracking_replenish_height_mm:g}mm")
        if args.matching_mode == "splat-track" and not args.virtual_guided_baseline:
            replenish_text += (f" | guided <= {args.virtual_guided_features} features"
                               f"/{args.virtual_guided_radius:g}px")
            if not args.virtual_guided_full_descriptors:
                replenish_text += ' | selective descriptors + retry cache'
        matcher_text = (
            f"{source_label} {args.retrieval_mode}"
            f"{'+autoROI' if args.retrieval_auto_roi else '+crop' if args.retrieval_crop else ''} "
            f"-> ALIKED+LightGlue top-{args.glb_top_k} + LK tracking | "
            f"reseed {args.tracking_reseed_frames} frames{replenish_text} | "
            f"RANSAC {args.ransac_threshold:g}px"
        )
    elif args.matching_mode == "global-track":
        matcher_text = (
            f"global + LK tracking | reseed {args.tracking_reseed_frames} frames | "
            f"FB<{args.tracking_fb_error:g}px | RANSAC {args.ransac_threshold:g}px"
        )
    else:
        matcher_text = (
            f"ratio<{args.ratio:g} | knn {args.knn} | "
            f"RANSAC {args.ransac_threshold:g}px"
        )
    retrieval_label = ('DINO+RC' if args.matching_mode == 'registered-aliked' else
                       args.retrieval_mode if args.matching_mode in {'glb-track', 'splat-track'}
                       else 'SIFT')
    mpr_state = ('OFF' if args.no_mpr else
                 'READY (PASS then V)' if args.ct_validation else 'ON')
    lines = (
        (
            f"capture {capture_fps:4.1f} fps | process {process_fps:4.1f} fps "
            f"(cap {args.process_fps:g}) | latest-frame drops {skipped_frames} | "
            f"queue age {queue_age_ms:.0f} ms | app latency {measured_latency:.0f} ms"
        ),
        (
            f"{retrieval_label} {result.sift_ms:.0f} ms | match {result.matching_ms:.0f} ms | "
            f"PnP {result.pnp_ms:.0f} ms | MPR {result.mpr_ms:.0f} ms | "
            f"localize {localize_ms:.0f} ms | debug3D {viewer_ms:.0f} ms | "
            f"keypoints {result.keypoints}"
            f"{' | guided +' + str(replenished) if replenished else ''}"
        ),
        (
            f"pipeline {pipeline_ms:.0f} ms | input {input_size[0]}x{input_size[1]} "
            f"-> work {work_size[0]}x{work_size[1]} | debug3D {debug_3d} | "
            f"MPR {mpr_state} | center focus {focus_score:.1f} "
            f"(PnP gate >={args.min_focus_score:g})"
        ),
        (
            f"{matcher_text} | iter {args.iterations} | PASS: inliers>={args.min_inliers} "
            f"and ratio>={args.min_inlier_ratio:.0%}"
        ),
    )
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, 89 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )


def draw_aliked_object_window(live_frame, live_bbox, matched_crop):
    """Compare the live detected ROI with the exact last ALIKED input."""
    from glb_virtual_live import crop_live_object

    panel = np.full((410, 800, 3), 26, np.uint8)
    cv2.putText(panel, "LIVE ROI (latest detection)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(panel, "LAST ALIKED INPUT", (410, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (225, 225, 225), 1, cv2.LINE_AA)
    live = crop_live_object(live_frame, live_bbox)
    for column, content in ((0, None if live is None else live[0]),
                            (400, matched_crop)):
        if content is None:
            cv2.putText(panel, "No valid object ROI yet", (column+10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 180, 255), 1, cv2.LINE_AA)
            continue
        height, width = content.shape[:2]
        scale = min(380./width, 355./height)
        preview = cv2.resize(content, (max(1, round(width*scale)),
                                       max(1, round(height*scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        top = 42 + (355-preview.shape[0])//2
        left = column + 10 + (380-preview.shape[1])//2
        panel[top:top+preview.shape[0], left:left+preview.shape[1]] = preview
    return panel


def add_focus_hold(frame, focus_score: float, threshold: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 68), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"FOCUS HOLD  center focus {focus_score:.1f} < {threshold:g}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 190, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "PnP/MPR paused; waiting for a sharp latest frame",
        (14, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def run_camera(args) -> int:
    capture = open_camera(
        args.camera,
        args.width,
        args.height,
        capture_backend=args.capture_backend,
        ffmpeg_path=args.ffmpeg,
    )
    if isinstance(capture, FFmpegCapture):
        print("Autofocus: device-managed (FFmpeg does not expose focus control)")
    elif isinstance(capture, OpenCVDirectCapture):
        print(
            "Autofocus: device-managed; direct OpenCV/AVFoundation read "
            "without camera property overrides"
        )
    else:
        print(
            "Autofocus: device-managed; OpenCV/AVFoundation is continuously "
            "drained on a latest-frame thread"
        )
    print('Loading localization map and models; camera preview opens after initialization.',
          flush=True)
    try:
        localizer = build_localizer(args)
    except Exception:
        capture.release()
        raise
    ok, frame = capture.read()
    if not ok or frame is None:
        detail = capture.error_detail()
        capture.release()
        suffix = f": {detail}" if detail else ""
        raise LivePnPError(f"Camera {args.camera} opened but returned no frame{suffix}")
    input_height, input_width = frame.shape[:2]
    work_frame, resize_scale = resize_for_processing(frame, args.max_processing_width)
    work_height, work_width = work_frame.shape[:2]
    matrix, distortion, calibration_label = camera_parameters(
        args, (work_width, work_height)
    )
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
    mpr_controls = None
    if localizer.mpr_renderer is not None:
        mpr_controls = MPRTrackbarControls(PREVIEW_WINDOW, localizer.mpr_renderer)
    approximate = args.calibration is None and args.intrinsics is None
    gs_pose_renderer = None
    if args.no_3d:
        viewer = None
    elif args.matching_mode == "splat-track":
        viewer = SplatPoseViewer(localizer, show_pose_render=args.show_gs_pose_render)
        if args.show_gs_pose_render:
            try:
                from live_gaussian_pose_render import LiveGaussianPoseRenderer
                gs_pose_renderer = LiveGaussianPoseRenderer(
                    args.splat_views, work_frame.shape,
                    max_hz=args.gs_pose_render_hz)
            except (OSError, ValueError) as exc:
                print(f'3DGS pose preview unavailable: {exc}', flush=True)
    else:
        viewer = PointCloud3DViewer(localizer)

    def render_viewer(frame, result):
        if isinstance(viewer, SplatPoseViewer):
            live_for_comparison = work_frame if args.show_gs_pose_render else frame
            gs_image, gs_age_ms, gs_render_ms, gs_status = (
                gs_pose_renderer.snapshot() if gs_pose_renderer is not None
                else (None, None, 0., 'UNAVAILABLE'))
            return viewer.render(
                live_for_comparison, result,
                search_result=localizer.last_search_result,
                diagnostics=localizer.last_search_diagnostics,
                candidate_age_ms=localizer.last_search_age_ms,
                active_reference_name=localizer.active_reference_name,
                coarse_reference_name=localizer.coarse_reference_name,
                coarse_margin=localizer.coarse_margin,
                coarse_mode=localizer.global_localizer.retrieval_mode,
                gs_image=gs_image,
                gs_age_ms=gs_age_ms,
                gs_render_ms=gs_render_ms,
                gs_status=gs_status,
            )
        return viewer.render(localizer.last_debug, result.status)
    last_3d_display = None
    last_result = None
    validation_session = None
    last_queue_age_ms = 0.0
    last_localize_ms = 0.0
    last_viewer_ms = 0.0
    last_pipeline_ms = 0.0
    last_focus_score = 0.0
    smoothed_process_fps = 0.0
    previous_completion_time = 0.0
    last_3d_rendered = -math.inf
    last_search_log = -math.inf
    focus_hold = False
    print(
        f"Camera {args.camera}: input {input_width}x{input_height}, "
        f"processing {work_width}x{work_height} (scale {resize_scale:.3f})"
    )
    print(
        "K: "
        f"fx={matrix[0, 0]:.3f} fy={matrix[1, 1]:.3f} "
        f"cx={matrix[0, 2]:.3f} cy={matrix[1, 2]:.3f} ({calibration_label})"
    )
    if approximate:
        print("WARNING: approximate intrinsics; calibrate before evaluating MPR alignment accuracy.")
    if args.ct_validation:
        print('CT validation is ready: wait for PASS, then press V in the camera window '
              'to freeze the frame and show MPR.', flush=True)
    print("Press Q or Esc in the preview window to stop.")

    period = 1.0 / args.process_fps
    last_processed = -math.inf
    last_display = work_frame
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LivePnPError("Camera frame capture failed")
            if args.raw_preview:
                # Show the exact frame returned by OpenCVLatestFrameCapture
                # before resize, focus gating, SIFT, PnP, or any drawing.
                cv2.imshow("RAW latest frame (unprocessed)", frame)
                raw_key = cv2.waitKey(1) & 0xFF
                if raw_key in (ord("q"), 27):
                    break
            now = time.monotonic()
            if validation_session is not None:
                if now - last_processed >= period:
                    last_display = validation_session.render()
                    last_processed = now
                cv2.imshow(PREVIEW_WINDOW, last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    validation_session = None
                    clear_result = getattr(localizer, 'clear_previous_result', None)
                    if clear_result is not None:
                        clear_result()
                    last_result = None
                    if mpr_controls is not None:
                        cv2.setMouseCallback(PREVIEW_WINDOW, mpr_controls._mouse_event)
                    print('CT validation released; returning to live localization.', flush=True)
                elif key == ord('n'):
                    percent = validation_session.return_to_marker_slice()
                    if percent is not None:
                        cv2.setTrackbarPos(MPRTrackbarControls.SLICE, PREVIEW_WINDOW,
                            int(round(percent * 10)) - MPRTrackbarControls.SLICE_MIN)
                elif key == ord('s'):
                    try:
                        measured = validation_session.save(
                            args.validation_output, calibration_label=calibration_label)
                    except (ValueError, OSError) as exc:
                        print(f'CT validation not saved: {exc}', flush=True)
                    else:
                        print(f'CT validation saved: {args.validation_output} | '
                              f'{measured["pixel_error"]:.2f}px | '
                              f'{measured["lateral_error"]:.2f} {measured["unit"]}',
                              flush=True)
                continue
            if now - last_processed >= period:
                pipeline_started = now
                work_frame, _ = resize_for_processing(frame, args.max_processing_width)
                last_focus_score = center_focus_score(work_frame)
                capture_stats = capture.stats()
                last_queue_age_ms = max(
                    0.0, (now - float(capture_stats["frame_time"])) * 1000.0
                )
                focus_hold = (
                    args.min_focus_score > 0
                    and last_focus_score < args.min_focus_score
                )
                if focus_hold:
                    last_result = None
                    last_display = work_frame.copy()
                    last_3d_display = (
                        render_viewer(last_display, None)
                        if isinstance(viewer, SplatPoseViewer) else None
                    )
                    last_localize_ms = 0.0
                    last_viewer_ms = 0.0
                    completion_time = time.monotonic()
                    last_pipeline_ms = (completion_time - pipeline_started) * 1000.0
                    print(
                        f"FOCUS_HOLD focus={last_focus_score:.1f} "
                        f"threshold={args.min_focus_score:g} "
                        f"queue={last_queue_age_ms:.0f}ms"
                    )
                else:
                    localize_started = time.monotonic()
                    result, last_display = localizer.localize(
                        work_frame, matrix, distortion
                    )
                    if gs_pose_renderer is not None:
                        gs_pose_renderer.submit(result, matrix, work_frame.shape)
                    last_localize_ms = (time.monotonic() - localize_started) * 1000.0
                    last_result = result
                    last_viewer_ms = 0.0
                    if (
                        viewer is not None
                        and (isinstance(viewer, SplatPoseViewer)
                             or result.detail != "REFERENCE_SEARCH_RUNNING")
                        and pipeline_started - last_3d_rendered
                        >= 1.0 / args.debug_3d_fps
                    ):
                        viewer_started = time.monotonic()
                        last_3d_display = render_viewer(last_display, result)
                        last_viewer_ms = (
                            time.monotonic() - viewer_started
                        ) * 1000.0
                        last_3d_rendered = pipeline_started
                    completion_time = time.monotonic()
                    last_pipeline_ms = (
                        completion_time - pipeline_started
                    ) * 1000.0
                    error = (
                        "—"
                        if result.mean_reprojection_error is None
                        else f"{result.mean_reprojection_error:.2f}px"
                    )
                    should_log = result.detail != "REFERENCE_SEARCH_RUNNING"
                    if not should_log and pipeline_started - last_search_log >= 1.0:
                        should_log = True
                        last_search_log = pipeline_started
                    if should_log:
                        print(
                            f"{result.status} matches={result.matches} inliers={result.inliers} "
                            f"error={error} MPR={result.mpr_status} "
                            f"total={result.processing_ms:.0f}ms "
                            f"[{'DINO+RC' if args.matching_mode == 'registered-aliked' else args.retrieval_mode if args.matching_mode in {'glb-track', 'splat-track'} else 'SIFT'} "
                            f"{result.sift_ms:.0f} match {result.matching_ms:.0f} "
                            f"PnP {result.pnp_ms:.0f} MPR {result.mpr_ms:.0f}] "
                            f"localize={last_localize_ms:.0f}ms "
                            f"debug3D={last_viewer_ms:.0f}ms "
                            f"pipeline={last_pipeline_ms:.0f}ms "
                            f"queue={last_queue_age_ms:.0f}ms "
                            f"keypoints={result.keypoints} focus={last_focus_score:.1f}"
                        )
                if previous_completion_time > 0:
                    instantaneous_fps = 1.0 / max(
                        completion_time - previous_completion_time, 1e-9
                    )
                    smoothed_process_fps = (
                        instantaneous_fps
                        if smoothed_process_fps == 0
                        else smoothed_process_fps * 0.8 + instantaneous_fps * 0.2
                    )
                previous_completion_time = completion_time
                last_processed = now
            banner_color = (0, 190, 255) if approximate else (70, 230, 70)
            add_banner(last_display, calibration_label, banner_color)
            if focus_hold:
                add_focus_hold(last_display, last_focus_score, args.min_focus_score)
            elif last_result is not None:
                capture_stats = capture.stats()
                add_performance_hud(
                    last_display,
                    last_result,
                    capture_fps=float(capture_stats["capture_fps"]),
                    process_fps=smoothed_process_fps,
                    skipped_frames=int(capture_stats["skipped_frames"]),
                    queue_age_ms=last_queue_age_ms,
                    localize_ms=last_localize_ms,
                    viewer_ms=last_viewer_ms,
                    pipeline_ms=last_pipeline_ms,
                    focus_score=last_focus_score,
                    input_size=(input_width, input_height),
                    work_size=(work_width, work_height),
                    args=args,
                    replenished=getattr(localizer, "last_replenished", 0),
                )
            if localizer.mpr_renderer is not None:
                localizer.mpr_renderer.set_measurement_enabled(
                    not focus_hold
                    and last_result is not None
                    and last_result.mpr_status == "VISIBLE"
                )
            cv2.imshow(PREVIEW_WINDOW, last_display)
            if args.show_retrieval_crops and args.matching_mode in {'glb-track', 'splat-track', 'registered-aliked'}:
                from view_retrieval import draw_crop_comparison
                crop_debug = (localizer.localizer.last_crop_debug
                              if args.matching_mode == 'registered-aliked'
                              else localizer.coarse_crop_debug)
                if crop_debug:
                    cv2.imshow('ViewSync Embedding Crops',
                               draw_crop_comparison(crop_debug, mode=args.retrieval_mode))
            if (args.matching_mode == 'splat-track' and
                (args.retrieval_auto_roi or args.retrieval_crop)):
                aliked_crop = localizer.global_localizer.last_aliked_crop
                live_bbox = (localizer.coarse_crop_debug or {}).get('bbox')
                cv2.imshow('ViewSync ALIKED Object ROI',
                           draw_aliked_object_window(work_frame, live_bbox,
                                                     aliked_crop))
            if viewer is not None and last_3d_display is not None:
                if isinstance(viewer, SplatPoseViewer):
                    viewer.update_live(last_3d_display,
                                       work_frame if args.show_gs_pose_render
                                       else last_display)
                    if gs_pose_renderer is not None:
                        gs_image, gs_age_ms, gs_render_ms, gs_status = (
                            gs_pose_renderer.snapshot())
                        viewer.update_pose_render(
                            last_3d_display, gs_image, gs_age_ms,
                            gs_render_ms, gs_status,
                            verified=(last_result is not None and
                                      last_result.status == 'PASS'))
                    viewer_window = viewer.WINDOW
                else:
                    viewer_window = "ViewSync Live 3D Match Coverage"
                cv2.imshow(viewer_window, last_3d_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.ct_validation and key == ord('v'):
                pass_result = getattr(localizer, 'last_pass_result', None)
                sample = getattr(localizer, 'last_pass_frame', None)
                pass_age = time.monotonic() - getattr(localizer, 'last_pass_time', 0.)
                if (pass_result is None or sample is None or pass_age > 10. or
                        pass_result.rvec is None or pass_result.tvec is None):
                    print('CT validation: wait for a PASS from the last 10 seconds.',
                          flush=True)
                else:
                    from ct_overlay_validation import CTValidationSession
                    validation_session = CTValidationSession(
                        localizer.mpr_renderer, sample, matrix, distortion,
                        pass_result, intrinsics_label=calibration_label)
                    localizer.mpr_renderer.clear_measurement()
                    localizer.mpr_renderer.set_measurement_enabled(False)
                    cv2.setMouseCallback(PREVIEW_WINDOW, validation_session.mouse_event)
                    print('CT validation frozen. Click LIVE marker; Shift+click CT marker. '
                          'Use slice/zoom sliders, N=same CT slice, S=save, R=resume.',
                          flush=True)
                continue
            if viewer is not None and viewer.handle_key(key) and last_result is not None:
                last_3d_display = render_viewer(last_display, last_result)
    finally:
        capture.release()
        if gs_pose_renderer is not None:
            gs_pose_renderer.close()
        close_localizer = getattr(localizer, "close", None)
        if close_localizer is not None:
            close_localizer()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an iPhone Continuity Camera directly from OpenCV for live PnP + MPR."
    )
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-camera-index", type=int, default=5)
    parser.add_argument(
        "--previous-success-settings",
        action="store_true",
        help=(
            "reproduce the earlier successful FFmpeg/AVFoundation live setup "
            "with native resolution and the original matching thresholds"
        ),
    )
    parser.add_argument(
        "--capture-backend",
        choices=("opencv", "opencv-latest", "ffmpeg"),
        default="opencv",
        help="opencv directly matches the minimal AVFoundation camera test",
    )
    parser.add_argument("--ffmpeg", help="path to FFmpeg executable")
    parser.add_argument("--width", type=int, default=0, help="requested width; 0 uses device default")
    parser.add_argument("--height", type=int, default=0, help="requested height; 0 uses device default")
    parser.add_argument("--process-fps", type=float, default=5.0)
    raw_preview = parser.add_mutually_exclusive_group()
    raw_preview.add_argument(
        "--raw-preview",
        dest="raw_preview",
        action="store_true",
        help="show the unprocessed latest-frame diagnostic window",
    )
    raw_preview.add_argument(
        "--no-raw-preview",
        dest="raw_preview",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(raw_preview=False)
    parser.add_argument(
        "--min-focus-score",
        type=float,
        default=0.0,
        help="pause PnP below this center sharpness score; 0 disables the gate",
    )
    parser.add_argument(
        "--max-processing-width",
        type=int,
        default=0,
        help="resize before SIFT while preserving aspect ratio; 0 disables",
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--matching-mode",
        choices=("global", "global-track", "reference", "glb-track", "splat-track", "registered-aliked"),
        default="global-track",
        help="2D-3D correspondence mode",
    )
    parser.add_argument("--tracking-reseed-frames", type=int, default=10)
    parser.add_argument("--tracking-replenish-frames", type=int, default=30,
                        help="3DGS replenishment cadence in interval mode; 0 disables interval")
    parser.add_argument("--tracking-replenish-mode", choices=('coverage', 'interval', 'background'),
                        default='coverage', help="3DGS replenishment trigger")
    parser.add_argument('--tracking-orbit-recovery', action='store_true',
                        help='Experimental GS background orbit recovery; accuracy is not validated')
    parser.add_argument("--tracking-replenish-distance-mm", type=float, default=30.,
                        help="minimum pairwise separation of three tracked PnP inliers")
    parser.add_argument("--tracking-replenish-height-mm", type=float, default=15.,
                        help="minimum triangle altitude against its longest edge")
    parser.add_argument("--tracking-replenish-cooldown", type=float, default=1.,
                        help="minimum seconds between replenishment searches")
    parser.add_argument('--tracking-replenish-image-coverage', type=float, default=.02,
                        help='Replenish when the inlier convex hull covers less than this image fraction')
    parser.add_argument('--tracking-replenish-angle', type=float, default=12.,
                        help='Replenish after this object-to-camera direction change from the last verified refresh')
    parser.add_argument("--tracking-min-points", type=int, default=10)
    parser.add_argument("--tracking-flow-window", type=int, default=31)
    parser.add_argument("--tracking-flow-levels", type=int, default=3)
    parser.add_argument("--tracking-fb-error", type=float, default=1.5)
    parser.add_argument("--tracking-iterations", type=int, default=150)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--glb-views", type=Path, default=DEFAULT_GLB_VIEWS)
    parser.add_argument("--splat-views", type=Path, default=DEFAULT_SPLAT_VIEWS)
    parser.add_argument("--registered-map", type=Path, default=DEFAULT_REGISTERED_MAP,
                        help="Directory built by build_aliked_object_map.py")
    parser.add_argument('--registered-search-policy', choices=('adaptive', 'exhaustive'),
                        default='adaptive', help='Stop registered-photo search only after two strong '
                        'agreeing poses, or process the complete shortlist for comparison')
    parser.add_argument("--glb-top-k", type=int, default=5)
    parser.add_argument("--glb-sift-ratio", type=float, default=0.78)
    parser.add_argument("--retrieval-mode", choices=("sift", "dino-global", "dino-spatial", "dino-patch"),
                        default="sift", help="Virtual-view coarse retrieval; SIFT remains the default")
    parser.add_argument("--retrieval-crop", action="store_true",
                        help="Crop reference retrieval views and live ALIKED input to object ROIs")
    parser.add_argument("--retrieval-auto-roi", action="store_true",
                        help="Detect the live 3DGS object via reference-supervised DINO patches")
    parser.add_argument("--show-retrieval-crops", action="store_true",
                        help="Show exact live/reference images fed to the coarse embedding")
    parser.add_argument("--retrieval-roi", nargs=4, type=float,
                        metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        default=(0.10, 0.10, 0.80, 0.80),
                        help="Normalized fixed live-image ROI for retrieval and ALIKED")
    parser.add_argument("--glb-max-keypoints", type=int, default=2000)
    parser.add_argument("--glb-resize", type=int, default=1024)
    parser.add_argument("--glb-match-threshold", type=float, default=0.01)
    parser.add_argument("--glb-xyz-search-radius", type=int, default=2)
    parser.add_argument("--glb-max-reprojection-error", type=float, default=6.0)
    parser.add_argument('--virtual-policy', choices=('validated', 'legacy'), default='validated')
    parser.add_argument('--virtual-diagnostics', type=Path,
                        help='Append per-search retrieval/local/XYZ/PnP/QC timings to JSONL')
    parser.add_argument('--virtual-max-seed-age', type=float, default=5.0)
    parser.add_argument('--virtual-max-coarse-rotation', type=float, default=35.0,
                        help='Max PnP rotation difference from supplying 3DGS view (degrees)')
    parser.add_argument('--virtual-fast-consensus', action='store_true',
                        help='Experimental early exit after agreeing candidate poses (off by default)')
    parser.add_argument('--virtual-guided-baseline', action='store_true',
                        help='Disable pose-guided point filtering and sequential early exit')
    parser.add_argument('--virtual-guided-features', type=int, default=2000,
                        help='Spatially distributed feature budget during tracked replenishment only')
    parser.add_argument('--virtual-guided-radius', type=float, default=96.,
                        help='Pose-prior matching gate in full-image pixels')
    parser.add_argument('--virtual-guided-full-descriptors', action='store_true',
                        help='Comparison: compute all descriptors before guided selection')
    parser.add_argument('--virtual-appearance-audit', action='store_true',
                        help='Log broad photometric evidence excluding PnP features; diagnostic only')
    parser.add_argument(
        "--glb-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--reference-shortlist", type=int, default=20)
    parser.add_argument("--reference-neighbors", type=int, default=8)
    parser.add_argument("--reference-workers", type=int, default=4)
    parser.add_argument("--reference-distance", type=float, default=300.0)
    parser.add_argument("--reference-ratio", type=float, default=0.82)
    parser.add_argument("--reference-result-hold", type=float, default=1.5)
    parser.add_argument("--fundamental-threshold", type=float, default=5.0)
    parser.add_argument("--guided-hypotheses", type=int, default=4)
    parser.add_argument(
        "--guided-radii", nargs="+", type=float, default=(160.0, 80.0, 40.0)
    )
    parser.add_argument("--guided-ratio", type=float, default=0.82)
    parser.add_argument("--guided-distance", type=float, default=280.0)
    parser.add_argument("--guided-max-candidates", type=int, default=80)
    parser.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--distortion", nargs=5, type=float, default=(0, 0, 0, 0, 0), metavar=("K1", "K2", "P1", "P2", "K3"))
    parser.add_argument("--fov-deg", type=float, default=72.0, help="Approximate horizontal FOV when no calibration is supplied")
    parser.add_argument("--no-mpr", action="store_true")
    parser.add_argument("--ct-validation", action="store_true",
                        help="Frozen PASS frame; compare live and CT marker positions (research only)")
    parser.add_argument("--validation-output", type=Path,
                        default=DATA_DIR / "output_ct_validation" / "measurements.csv")
    parser.add_argument("--no-3d", action="store_true", help="hide live 3D match coverage window")
    parser.add_argument("--show-gs-pose-render", action="store_true",
                        help="render the true 3DGS appearance at each verified PnP pose in a separate process")
    parser.add_argument("--gs-pose-render-hz", type=float, default=5.0,
                        help="maximum asynchronous true-3DGS pose-preview rate")
    parser.add_argument(
        "--debug-3d-fps",
        type=float,
        default=2.0,
        help="3D debug window refresh rate; localization still uses --process-fps",
    )
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    parser.add_argument("--mpr-size", type=int, default=192)
    parser.add_argument("--window-level", type=float, default=0.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--mpr-opacity", type=float, default=0.82)
    return parser


def apply_runtime_preset(args):
    """Apply named, reproducible settings after argument parsing."""
    if not args.previous_success_settings:
        return args
    args.capture_backend = "opencv"
    args.width = 0
    args.height = 0
    args.process_fps = 5.0
    args.max_processing_width = 0
    args.min_focus_score = 0.0
    args.raw_preview = False
    args.matching_mode = "global"
    args.fov_deg = 65.0
    args.ratio = 0.75
    args.knn = 16
    args.ransac_threshold = 4.0
    args.confidence = 0.999
    args.iterations = 3000
    args.min_inliers = 20
    args.min_inlier_ratio = 0.2
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = apply_runtime_preset(build_parser().parse_args(argv))
    try:
        if args.previous_success_settings:
            print(
                "Preset previous-success: direct OpenCV/AVFoundation native "
                "resolution, full-resolution SIFT, 5 fps, no focus gate, "
                "ratio 0.75, RANSAC 4 px"
            )
        if args.list_cameras:
            return list_cameras(
                args.max_camera_index,
                capture_backend=args.capture_backend,
                ffmpeg_path=args.ffmpeg,
            )
        if (
            args.process_fps <= 0
            or args.width < 0
            or args.height < 0
            or args.max_processing_width < 0
            or args.debug_3d_fps <= 0
            or not 0.1 <= args.gs_pose_render_hz <= 10
            or (args.show_gs_pose_render and args.matching_mode != 'splat-track')
            or args.min_focus_score < 0
            or args.reference_shortlist <= 0
            or args.reference_neighbors < 2
            or args.reference_workers <= 0
            or args.guided_hypotheses <= 0
            or not args.guided_radii
            or min(args.guided_radii) <= 0
            or not 0 < args.reference_ratio < 1
            or not 0 < args.guided_ratio < 1
            or args.reference_result_hold < 0
            or args.glb_top_k <= 0
            or not 0 < args.glb_sift_ratio < 1
            or (args.retrieval_auto_roi and args.retrieval_mode == 'sift')
            or (args.retrieval_crop and (args.retrieval_roi[0] < 0 or args.retrieval_roi[1] < 0
                or args.retrieval_roi[2] <= 0 or args.retrieval_roi[3] <= 0
                or args.retrieval_roi[0] + args.retrieval_roi[2] > 1
                or args.retrieval_roi[1] + args.retrieval_roi[3] > 1))
            or args.glb_max_keypoints <= 0
            or args.virtual_guided_features < 32
            or not math.isfinite(args.virtual_guided_radius)
            or args.virtual_guided_radius <= 0
            or args.glb_resize <= 0
            or args.glb_match_threshold < 0
            or args.glb_xyz_search_radius < 0
            or args.glb_max_reprojection_error <= 0
            or not math.isfinite(args.virtual_max_seed_age)
            or args.virtual_max_seed_age <= 0
            or not math.isfinite(args.virtual_max_coarse_rotation)
            or not 0 < args.virtual_max_coarse_rotation <= 180
            or args.tracking_reseed_frames <= 0
            or args.tracking_replenish_frames < 0
            or not math.isfinite(args.tracking_replenish_distance_mm)
            or args.tracking_replenish_distance_mm <= 0
            or not math.isfinite(args.tracking_replenish_height_mm)
            or args.tracking_replenish_height_mm <= 0
            or not math.isfinite(args.tracking_replenish_cooldown)
            or args.tracking_replenish_cooldown <= 0
            or not math.isfinite(args.tracking_replenish_image_coverage)
            or not 0 <= args.tracking_replenish_image_coverage <= 1
            or not math.isfinite(args.tracking_replenish_angle)
            or not 0 < args.tracking_replenish_angle <= 180
            or args.tracking_min_points < 4
            or args.tracking_flow_window < 5
            or args.tracking_flow_window % 2 == 0
            or args.tracking_flow_levels < 0
            or args.tracking_fb_error <= 0
            or args.tracking_iterations <= 0
        ):
            raise LivePnPError("Invalid camera resolution or --process-fps")
        return run_camera(args)
    except (LivePnPError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
