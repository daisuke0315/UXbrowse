#!/usr/bin/env python3
"""Run object localization and MPR overlay from a Mac/Continuity camera."""

from __future__ import annotations

import argparse
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
    LocalizationDebug,
    MPRRenderer,
)


SCRIPT_DIR = Path(__file__).resolve().parent


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
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise LivePnPError(
            f"Camera {index} could not be opened. Check macOS Camera permission and --camera."
        )
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


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


def build_localizer(args) -> LivePnPLocalizer:
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
    return LivePnPLocalizer(
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
    input_size,
    work_size,
    args,
) -> None:
    cv2.rectangle(frame, (0, 68), (frame.shape[1], 171), (24, 24, 24), -1)
    measured_latency = queue_age_ms + pipeline_ms
    debug_3d = "OFF" if args.no_3d else f"ON {args.debug_3d_fps:g} Hz"
    lines = (
        (
            f"capture {capture_fps:4.1f} fps | process {process_fps:4.1f} fps "
            f"(cap {args.process_fps:g}) | latest-frame drops {skipped_frames} | "
            f"queue age {queue_age_ms:.0f} ms | app latency {measured_latency:.0f} ms"
        ),
        (
            f"SIFT {result.sift_ms:.0f} ms | match {result.matching_ms:.0f} ms | "
            f"PnP {result.pnp_ms:.0f} ms | MPR {result.mpr_ms:.0f} ms | "
            f"localize {localize_ms:.0f} ms | debug3D {viewer_ms:.0f} ms"
        ),
        (
            f"pipeline {pipeline_ms:.0f} ms | input {input_size[0]}x{input_size[1]} "
            f"-> work {work_size[0]}x{work_size[1]} | debug3D {debug_3d} | "
            f"MPR {'OFF' if args.no_mpr else 'ON'}"
        ),
        (
            f"ratio<{args.ratio:g} | knn {args.knn} | RANSAC {args.ransac_threshold:g}px "
            f"iter {args.iterations} | PASS: inliers>={args.min_inliers} "
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


def run_camera(args) -> int:
    capture = open_camera(
        args.camera,
        args.width,
        args.height,
        capture_backend=args.capture_backend,
        ffmpeg_path=args.ffmpeg,
    )
    localizer = build_localizer(args)
    ok, frame = capture.read()
    if not ok or frame is None:
        detail = capture.error_detail() if isinstance(capture, FFmpegCapture) else ""
        capture.release()
        suffix = f": {detail}" if detail else ""
        raise LivePnPError(f"Camera {args.camera} opened but returned no frame{suffix}")
    input_height, input_width = frame.shape[:2]
    work_frame, resize_scale = resize_for_processing(frame, args.max_processing_width)
    work_height, work_width = work_frame.shape[:2]
    matrix, distortion, calibration_label = camera_parameters(
        args, (work_width, work_height)
    )
    approximate = args.calibration is None and args.intrinsics is None
    viewer = None if args.no_3d else PointCloud3DViewer(localizer)
    last_3d_display = None
    last_result = None
    last_queue_age_ms = 0.0
    last_localize_ms = 0.0
    last_viewer_ms = 0.0
    last_pipeline_ms = 0.0
    smoothed_process_fps = 0.0
    previous_completion_time = 0.0
    last_3d_rendered = -math.inf
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
    print("Press Q or Esc in the preview window to stop.")

    period = 1.0 / args.process_fps
    last_processed = -math.inf
    last_display = work_frame
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LivePnPError("Camera frame capture failed")
            now = time.monotonic()
            if now - last_processed >= period:
                pipeline_started = now
                work_frame, _ = resize_for_processing(frame, args.max_processing_width)
                capture_stats = (
                    capture.stats()
                    if isinstance(capture, FFmpegCapture)
                    else {
                        "capture_fps": 0.0,
                        "skipped_frames": 0,
                        "frame_time": now,
                    }
                )
                last_queue_age_ms = max(
                    0.0, (now - float(capture_stats["frame_time"])) * 1000.0
                )
                localize_started = time.monotonic()
                result, last_display = localizer.localize(
                    work_frame, matrix, distortion
                )
                last_localize_ms = (time.monotonic() - localize_started) * 1000.0
                last_result = result
                last_viewer_ms = 0.0
                if (
                    viewer is not None
                    and pipeline_started - last_3d_rendered >= 1.0 / args.debug_3d_fps
                ):
                    viewer_started = time.monotonic()
                    last_3d_display = viewer.render(localizer.last_debug, result.status)
                    last_viewer_ms = (time.monotonic() - viewer_started) * 1000.0
                    last_3d_rendered = pipeline_started
                completion_time = time.monotonic()
                last_pipeline_ms = (completion_time - pipeline_started) * 1000.0
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
                error = (
                    "—"
                    if result.mean_reprojection_error is None
                    else f"{result.mean_reprojection_error:.2f}px"
                )
                print(
                    f"{result.status} matches={result.matches} inliers={result.inliers} "
                    f"error={error} MPR={result.mpr_status} total={result.processing_ms:.0f}ms "
                    f"[SIFT {result.sift_ms:.0f} match {result.matching_ms:.0f} "
                    f"PnP {result.pnp_ms:.0f} MPR {result.mpr_ms:.0f}] "
                    f"localize={last_localize_ms:.0f}ms debug3D={last_viewer_ms:.0f}ms "
                    f"pipeline={last_pipeline_ms:.0f}ms queue={last_queue_age_ms:.0f}ms"
                )
            banner_color = (0, 190, 255) if approximate else (70, 230, 70)
            add_banner(last_display, calibration_label, banner_color)
            if last_result is not None:
                capture_stats = (
                    capture.stats()
                    if isinstance(capture, FFmpegCapture)
                    else {"capture_fps": 0.0, "skipped_frames": 0}
                )
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
                    input_size=(input_width, input_height),
                    work_size=(work_width, work_height),
                    args=args,
                )
            cv2.imshow("ViewSync Continuity Camera PnP + MPR", last_display)
            if viewer is not None and last_3d_display is not None:
                cv2.imshow("ViewSync Live 3D Match Coverage", last_3d_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if viewer is not None and viewer.handle_key(key) and last_result is not None:
                last_3d_display = viewer.render(localizer.last_debug, last_result.status)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an iPhone Continuity Camera directly from OpenCV for live PnP + MPR."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-camera-index", type=int, default=5)
    parser.add_argument(
        "--capture-backend",
        choices=("ffmpeg", "opencv"),
        default="ffmpeg",
        help="FFmpeg is the stable default for Continuity Camera",
    )
    parser.add_argument("--ffmpeg", help="path to FFmpeg executable")
    parser.add_argument("--width", type=int, default=0, help="requested width; 0 uses device default")
    parser.add_argument("--height", type=int, default=0, help="requested height; 0 uses device default")
    parser.add_argument("--process-fps", type=float, default=10.0)
    parser.add_argument(
        "--max-processing-width",
        type=int,
        default=1280,
        help="resize before SIFT while preserving aspect ratio; 0 disables",
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--distortion", nargs=5, type=float, default=(0, 0, 0, 0, 0), metavar=("K1", "K2", "P1", "P2", "K3"))
    parser.add_argument("--fov-deg", type=float, default=65.0, help="Approximate horizontal FOV when no calibration is supplied")
    parser.add_argument("--no-mpr", action="store_true")
    parser.add_argument("--no-3d", action="store_true", help="hide live 3D match coverage window")
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
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    parser.add_argument("--mpr-size", type=int, default=192)
    parser.add_argument("--window-level", type=float, default=0.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--mpr-opacity", type=float, default=0.82)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
        ):
            raise LivePnPError("Invalid camera resolution or --process-fps")
        return run_camera(args)
    except (LivePnPError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
