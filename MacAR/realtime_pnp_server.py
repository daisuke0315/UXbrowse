#!/usr/bin/env python3
"""Receive iPhone camera frames and perform object PnP + MPR overlay on the Mac."""

from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from offline_pnp_test import MapData, load_feature_map, select_unique_point_matches


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = SCRIPT_DIR / "output_colmap_image" / "object_features.npz"
DEFAULT_VOLUME = SCRIPT_DIR.parent / "data" / "260911" / "volume_preprocessed.model-bounds.mha"
DEFAULT_COORDINATES = SCRIPT_DIR.parent / "data" / "260911" / "model-coordinate.json"


class LivePnPError(RuntimeError):
    """Expected configuration, input, or localization error."""


@dataclass(frozen=True)
class LiveResult:
    status: str
    detail: str
    keypoints: int
    matches: int
    inliers: int
    inlier_ratio: float
    mean_reprojection_error: Optional[float]
    rvec: Optional[tuple[float, float, float]]
    tvec: Optional[tuple[float, float, float]]
    camera_center_model: Optional[tuple[float, float, float]]
    processing_ms: float
    mpr_status: str
    sift_ms: float = 0.0
    matching_ms: float = 0.0
    pnp_ms: float = 0.0
    mpr_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "detail": self.detail,
            "keypoints": self.keypoints,
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "mean_reprojection_error": self.mean_reprojection_error,
            "rvec": self.rvec,
            "tvec": self.tvec,
            "camera_center_model": self.camera_center_model,
            "processing_ms": self.processing_ms,
            "mpr_status": self.mpr_status,
            "sift_ms": self.sift_ms,
            "matching_ms": self.matching_ms,
            "pnp_ms": self.pnp_ms,
            "mpr_ms": self.mpr_ms,
        }


@dataclass(frozen=True)
class LocalizationDebug:
    matched_xyz: object
    inlier_indices: frozenset[int]
    camera_center_model: Optional[tuple[float, float, float]]
    pnp_solved: bool


@dataclass(frozen=True)
class TrackingSeed:
    """Accepted 2D-3D correspondences carried into the next video frame."""

    gray: object
    image_points: object
    object_points: object
    rvec: object
    tvec: object


@dataclass(frozen=True)
class MHAData:
    voxels: object
    dimensions: tuple[int, int, int]
    spacing: object
    origin: object
    inverse_direction: object


def _header_fields_and_offset(path: Path) -> tuple[dict[str, str], int]:
    marker = b"ElementDataFile = LOCAL"
    with path.open("rb") as source:
        header = bytearray()
        while True:
            line = source.readline()
            if not line:
                raise LivePnPError("MHA header has no LOCAL voxel marker")
            header.extend(line)
            if line.strip() == marker:
                break
        offset = source.tell()
    try:
        text = header.decode("ascii")
    except UnicodeDecodeError as exc:
        raise LivePnPError("MHA header is not ASCII") from exc
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    return fields, offset


def load_mha(path: Path) -> MHAData:
    if not path.is_file():
        raise LivePnPError(f"MHA not found: {path}")
    fields, offset = _header_fields_and_offset(path)
    if fields.get("elementtype", "").upper() != "MET_FLOAT":
        raise LivePnPError("Only MET_FLOAT MHA is supported")
    if fields.get("compresseddata", "false").lower() == "true":
        raise LivePnPError("Compressed MHA is not supported")

    def values(name: str, count: int, cast):
        try:
            parsed = [cast(value) for value in fields[name].split()]
        except (KeyError, ValueError) as exc:
            raise LivePnPError(f"Invalid MHA field: {name}") from exc
        if len(parsed) != count:
            raise LivePnPError(f"Invalid MHA field length: {name}")
        return parsed

    width, height, depth = values("dimsize", 3, int)
    spacing = np.asarray(values("elementspacing", 3, float), dtype=np.float64)
    origin = np.asarray(
        values("offset" if "offset" in fields else "origin", 3, float),
        dtype=np.float64,
    )
    matrix = np.asarray(values("transformmatrix", 9, float), dtype=np.float64).reshape(3, 3)
    count = width * height * depth
    voxels = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=offset,
        shape=(depth, height, width),
        order="C",
    )
    if voxels.size != count:
        raise LivePnPError("MHA voxel data is shorter than DimSize")
    return MHAData(
        voxels,
        (width, height, depth),
        spacing,
        origin,
        np.linalg.inv(matrix),
    )


def rc_to_colmap(points):
    """RealityCapture/GLB (x,y,z) -> exported COLMAP (x,-z,y)."""
    array = np.asarray(points, dtype=np.float64)
    return np.stack((array[..., 0], -array[..., 2], array[..., 1]), axis=-1)


def colmap_to_rc(points):
    """Exported COLMAP (x,y,z) -> RealityCapture/GLB (x,z,-y)."""
    array = np.asarray(points, dtype=np.float64)
    return np.stack((array[..., 0], array[..., 2], -array[..., 1]), axis=-1)


def ray_box_interval(origin, direction, minimum, maximum):
    near = -math.inf
    far = math.inf
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-9:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return None
            continue
        first = (minimum[axis] - origin[axis]) / direction[axis]
        second = (maximum[axis] - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if near > far:
            return None
    if far < max(near, 0):
        return None
    return float(near), float(far)


def projected_box_depth_interval(origin, direction, corners):
    """Return near/far plane depths without requiring the optical axis to hit the box."""
    points = np.asarray(corners, dtype=np.float64)
    depths = (points - np.asarray(origin, dtype=np.float64)) @ np.asarray(
        direction, dtype=np.float64
    )
    return float(np.min(depths)), float(np.max(depths))


class MPRRenderer:
    def __init__(
        self,
        volume_path: Path,
        coordinate_path: Path,
        *,
        texture_size: int = 192,
        window_level: float = 0.0,
        window_width: float = 400.0,
        opacity: float = 0.82,
    ) -> None:
        if texture_size < 32 or window_width <= 0 or not 0 <= opacity <= 1:
            raise LivePnPError("Invalid MPR rendering parameter")
        self.volume = load_mha(volume_path)
        try:
            metadata = json.loads(coordinate_path.read_text(encoding="utf-8"))
            source = metadata["sourceCoordinateSystem"]
            rc_minimum = np.asarray(source["boundsMinimum"], dtype=np.float64)
            rc_maximum = np.asarray(source["boundsMaximum"], dtype=np.float64)
            source_unit = str(source.get("unit", "model unit"))
        except (OSError, UnicodeError, KeyError, ValueError, TypeError) as exc:
            raise LivePnPError(f"Cannot load coordinate metadata: {coordinate_path}: {exc}") from exc
        colmap_corners = rc_to_colmap(np.asarray([
            (x, y, z)
            for x in (rc_minimum[0], rc_maximum[0])
            for y in (rc_minimum[1], rc_maximum[1])
            for z in (rc_minimum[2], rc_maximum[2])
        ]))
        self.box_minimum = colmap_corners.min(axis=0)
        self.box_maximum = colmap_corners.max(axis=0)
        self.texture_size = texture_size
        self.window_level = window_level
        self.window_width = window_width
        self.opacity = opacity
        self.slice_fraction = 0.5
        self.zoom = 1.0
        self.unit_label = "mm" if source_unit.lower() == "millimeter" else source_unit
        self._control_lock = threading.Lock()
        self._measurement_enabled = False
        self._measurement_points: list[object] = []
        self._last_inverse_homography = None
        self._last_inverse_view_homography = None
        self._zoom_center = None

        direction = np.linalg.inv(self.volume.inverse_direction)
        width, height, depth = self.volume.dimensions
        voxel_corners = np.asarray(
            [
                (x, y, z)
                for x in (0.0, width - 1.0)
                for y in (0.0, height - 1.0)
                for z in (0.0, depth - 1.0)
            ],
            dtype=np.float64,
        )
        oriented = voxel_corners * self.volume.spacing
        volume_rc_corners = self.volume.origin + oriented @ direction.T
        volume_colmap_corners = rc_to_colmap(volume_rc_corners)
        self.volume_corners = volume_colmap_corners
        self.volume_center = volume_colmap_corners.mean(axis=0)
        self.volume_box_minimum = volume_colmap_corners.min(axis=0)
        self.volume_box_maximum = volume_colmap_corners.max(axis=0)
        # A square whose side is the volume diagonal covers every camera-facing
        # cross-section, including when the object is near an image edge.
        self.field_of_view = float(
            np.linalg.norm(self.volume_box_maximum - self.volume_box_minimum) * 1.02
        )

    def set_slice_percent(self, percent: float) -> None:
        with self._control_lock:
            self.slice_fraction = float(np.clip(percent, -100.0, 150.0)) / 100.0

    def set_opacity_percent(self, percent: float) -> None:
        with self._control_lock:
            self.opacity = float(np.clip(percent, 0.0, 100.0)) / 100.0

    def set_zoom_percent(self, percent: float) -> None:
        with self._control_lock:
            self.zoom = float(np.clip(percent, 100.0, 1200.0)) / 100.0

    def center_zoom_on_display_point(self, x: float, y: float) -> None:
        """Center the next zoomed view on a point in the current displayed image."""
        with self._control_lock:
            inverse = self._last_inverse_view_homography
            if inverse is None:
                self._zoom_center = np.asarray((x, y), dtype=np.float64)
                return
            inverse = inverse.copy()
        display_point = np.asarray([[[x, y]]], dtype=np.float32)
        image_point = cv2.perspectiveTransform(display_point, inverse).reshape(2)
        if np.isfinite(image_point).all():
            with self._control_lock:
                self._zoom_center = image_point.astype(np.float64)

    def set_measurement_enabled(self, enabled: bool) -> None:
        with self._control_lock:
            self._measurement_enabled = bool(enabled)

    def clear_measurement(self) -> None:
        with self._control_lock:
            self._measurement_points.clear()

    def measurement_point_count(self) -> int:
        with self._control_lock:
            return len(self._measurement_points)

    def add_measurement_image_point(self, x: float, y: float) -> Optional[float]:
        """Add an image click and return the completed planar distance."""
        with self._control_lock:
            if not self._measurement_enabled or self._last_inverse_homography is None:
                return None
            inverse = self._last_inverse_homography.copy()
            inverse_view = self._last_inverse_view_homography
            if inverse_view is not None:
                inverse_view = inverse_view.copy()
        display_point = np.asarray([[[x, y]]], dtype=np.float32)
        source_point = cv2.perspectiveTransform(display_point, inverse).reshape(2)
        maximum = float(self.texture_size - 1)
        if (
            not np.isfinite(source_point).all()
            or source_point[0] < 0
            or source_point[0] > maximum
            or source_point[1] < 0
            or source_point[1] > maximum
        ):
            return None
        with self._control_lock:
            if len(self._measurement_points) >= 2:
                self._measurement_points.clear()
            self._measurement_points.append(source_point.astype(np.float64))
            if len(self._measurement_points) < 2:
                if inverse_view is not None:
                    image_point = cv2.perspectiveTransform(
                        display_point, inverse_view
                    ).reshape(2)
                    if np.isfinite(image_point).all():
                        self._zoom_center = image_point.astype(np.float64)
                return None
            pixel_distance = float(
                np.linalg.norm(
                    np.asarray(self._measurement_points[1])
                    - np.asarray(self._measurement_points[0])
                )
            )
        return pixel_distance * self.field_of_view / max(self.texture_size - 1, 1)

    def _sample_plane(self, center_colmap, u_colmap, v_colmap, opacity):
        size = self.texture_size
        half = (size - 1) * 0.5
        coordinates = np.arange(size, dtype=np.float64)
        pixel_spacing = self.field_of_view / max(size - 1, 1)
        offset_u = (coordinates - half) * pixel_spacing
        offset_v = (half - coordinates) * pixel_spacing
        grid_u, grid_v = np.meshgrid(offset_u, offset_v)
        center_rc = colmap_to_rc(center_colmap)
        u_rc = colmap_to_rc(u_colmap)
        v_rc = colmap_to_rc(v_colmap)
        world = (
            center_rc[None, None, :]
            + grid_u[..., None] * u_rc[None, None, :]
            + grid_v[..., None] * v_rc[None, None, :]
        )
        relative = world - self.volume.origin
        oriented = relative @ self.volume.inverse_direction.T
        voxel = oriented / self.volume.spacing
        width, height, depth = self.volume.dimensions
        valid = (
            (voxel[..., 0] >= 0)
            & (voxel[..., 1] >= 0)
            & (voxel[..., 2] >= 0)
            & (voxel[..., 0] <= width - 1)
            & (voxel[..., 1] <= height - 1)
            & (voxel[..., 2] <= depth - 1)
        )
        clipped = voxel.copy()
        clipped[..., 0] = np.clip(clipped[..., 0], 0, width - 1)
        clipped[..., 1] = np.clip(clipped[..., 1], 0, height - 1)
        clipped[..., 2] = np.clip(clipped[..., 2], 0, depth - 1)
        lower = np.floor(clipped).astype(np.int32)
        upper = lower.copy()
        upper[..., 0] = np.minimum(lower[..., 0] + 1, width - 1)
        upper[..., 1] = np.minimum(lower[..., 1] + 1, height - 1)
        upper[..., 2] = np.minimum(lower[..., 2] + 1, depth - 1)
        fraction = clipped - lower
        x0, y0, z0 = lower[..., 0], lower[..., 1], lower[..., 2]
        x1, y1, z1 = upper[..., 0], upper[..., 1], upper[..., 2]
        xd, yd, zd = fraction[..., 0], fraction[..., 1], fraction[..., 2]
        data = self.volume.voxels
        c00 = data[z0, y0, x0] * (1 - xd) + data[z0, y0, x1] * xd
        c10 = data[z0, y1, x0] * (1 - xd) + data[z0, y1, x1] * xd
        c01 = data[z1, y0, x0] * (1 - xd) + data[z1, y0, x1] * xd
        c11 = data[z1, y1, x0] * (1 - xd) + data[z1, y1, x1] * xd
        sample = (c00 * (1 - yd) + c10 * yd) * (1 - zd) + (c01 * (1 - yd) + c11 * yd) * zd
        minimum = self.window_level - self.window_width * 0.5
        gray = np.clip((sample - minimum) / self.window_width, 0, 1)
        gray = np.rint(gray * 255).astype(np.uint8)
        alpha = np.where(sample <= -900, 0, np.where(sample < -500, 100, 238)).astype(np.float32)
        alpha *= opacity / 255.0
        alpha[~valid] = 0
        return gray, alpha

    @staticmethod
    def _apply_display_zoom(image, zoom: float, center):
        """Zoom the rendered camera view and return its original-to-display transform."""
        height, width = image.shape[:2]
        identity = np.eye(3, dtype=np.float64)
        if zoom <= 1.0 + 1e-9:
            return image, identity
        if center is None:
            center = np.asarray(((width - 1) * 0.5, (height - 1) * 0.5))
        else:
            center = np.asarray(center, dtype=np.float64)
        half_width = (width - 1) * 0.5 / zoom
        half_height = (height - 1) * 0.5 / zoom
        center_x = float(np.clip(center[0], half_width, (width - 1) - half_width))
        center_y = float(np.clip(center[1], half_height, (height - 1) - half_height))
        target_x = (width - 1) * 0.5
        target_y = (height - 1) * 0.5
        transform = np.asarray(
            (
                (zoom, 0.0, target_x - zoom * center_x),
                (0.0, zoom, target_y - zoom * center_y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        zoomed = cv2.warpPerspective(
            image, transform, (width, height), flags=cv2.INTER_LINEAR
        )
        return zoomed, transform

    def render(self, image, camera_matrix, distortion, rvec, tvec):
        with self._control_lock:
            slice_fraction = self.slice_fraction
            opacity = self.opacity
            zoom = self.zoom
            zoom_center = (
                None if self._zoom_center is None else self._zoom_center.copy()
            )
            measurement_points = [
                np.asarray(point, dtype=np.float64).copy()
                for point in self._measurement_points
            ]
            self._last_inverse_homography = None
        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = (-rotation.T @ tvec).reshape(3)
        forward = rotation.T @ np.asarray((0.0, 0.0, 1.0))
        forward /= np.linalg.norm(forward)
        near, far = projected_box_depth_interval(
            camera_center, forward, self.volume_corners
        )
        slice_depth = (far - near) * slice_fraction
        target_depth = near + slice_depth
        center_depth = float(np.dot(self.volume_center - camera_center, forward))
        center = self.volume_center + forward * (target_depth - center_depth)
        right = rotation.T @ np.asarray((1.0, 0.0, 0.0))
        up = rotation.T @ np.asarray((0.0, -1.0, 0.0))
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        gray, alpha = self._sample_plane(center, right, up, opacity)
        if float(alpha.max()) <= 0:
            return image, "VOLUME_SAMPLE_EMPTY"

        half = self.field_of_view * 0.5
        corners = np.asarray(
            (
                center - right * half + up * half,
                center + right * half + up * half,
                center + right * half - up * half,
                center - right * half - up * half,
            ),
            dtype=np.float64,
        )
        camera_z = (rotation @ corners.T + tvec).T[:, 2]
        if np.any(camera_z <= 1e-6):
            return image, "PLANE_BEHIND_CAMERA"
        projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, distortion)
        destination = projected.reshape(4, 2).astype(np.float32)
        size = self.texture_size
        source = np.asarray(((0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)), dtype=np.float32)
        homography = cv2.getPerspectiveTransform(source, destination)
        output_size = (image.shape[1], image.shape[0])
        warped_gray = cv2.warpPerspective(gray, homography, output_size, flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpPerspective(alpha, homography, output_size, flags=cv2.INTER_LINEAR)
        overlay = np.repeat(warped_gray[..., None], 3, axis=2).astype(np.float32)
        amount = np.clip(warped_alpha[..., None], 0, 1)
        blended = image.astype(np.float32) * (1 - amount) + overlay * amount
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended, view_homography = self._apply_display_zoom(
            blended, zoom, zoom_center
        )
        display_homography = view_homography @ homography
        with self._control_lock:
            self._last_inverse_view_homography = np.linalg.inv(view_homography)
            self._last_inverse_homography = np.linalg.inv(display_homography)
        self._draw_measurement(blended, display_homography, measurement_points)
        label = (
            f"MPR slice {slice_fraction * 100:6.1f}% "
            f"({slice_depth:.1f} {self.unit_label} from near) | "
            f"opacity {opacity * 100:.0f}% | zoom {zoom:.1f}x"
        )
        instruction = (
            "Measure: left-click 2 points (1st centers zoom) | Shift+click/wheel: center | "
            "clear: right-click"
        )
        label_width = min(blended.shape[1] - 1, 940)
        label_top = max(0, blended.shape[0] - 96)
        cv2.rectangle(
            blended,
            (0, label_top),
            (label_width, max(label_top, blended.shape[0] - 35)),
            (18, 18, 18),
            -1,
        )
        cv2.putText(
            blended,
            label,
            (12, max(18, blended.shape[0] - 68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            instruction,
            (12, max(18, blended.shape[0] - 44)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return blended, "VISIBLE"

    def _draw_measurement(self, image, homography, source_points) -> None:
        if not source_points:
            return
        projected = cv2.perspectiveTransform(
            np.asarray(source_points, dtype=np.float32).reshape((1, -1, 2)),
            homography,
        ).reshape((-1, 2))
        first = tuple(int(round(value)) for value in projected[0])
        cv2.circle(image, first, 6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(image, first, 6, (255, 220, 0), 2, cv2.LINE_AA)
        if len(projected) < 2:
            cv2.putText(
                image, "select second point", (first[0] + 10, first[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 2, cv2.LINE_AA,
            )
            return
        second = tuple(int(round(value)) for value in projected[1])
        cv2.line(image, first, second, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(image, first, second, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.circle(image, second, 6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(image, second, 6, (255, 220, 0), 2, cv2.LINE_AA)
        source_distance = float(
            np.linalg.norm(
                np.asarray(source_points[1], dtype=np.float64)
                - np.asarray(source_points[0], dtype=np.float64)
            )
        )
        distance = source_distance * self.field_of_view / max(self.texture_size - 1, 1)
        midpoint = (projected[0] + projected[1]) * 0.5
        text_origin = (int(round(midpoint[0] + 8)), int(round(midpoint[1] - 8)))
        text = f"{distance:.1f} {self.unit_label}"
        cv2.putText(
            image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (255, 220, 0), 2, cv2.LINE_AA,
        )


class LivePnPLocalizer:
    def __init__(
        self,
        feature_path: Path,
        *,
        ratio_threshold: float = 0.75,
        knn: int = 16,
        ransac_threshold: float = 4.0,
        confidence: float = 0.999,
        iterations: int = 3000,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.2,
        mpr_renderer: Optional[MPRRenderer] = None,
    ) -> None:
        if not 0 < ratio_threshold < 1 or knn < 2:
            raise LivePnPError("Invalid descriptor matching parameters")
        self.map_data: MapData = load_feature_map(np, feature_path)
        self.ratio_threshold = ratio_threshold
        self.knn = min(knn, len(self.map_data.descriptors))
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.iterations = iterations
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.mpr_renderer = mpr_renderer
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
        self.matcher.add([self.map_data.descriptors])
        self.matcher.train()
        self.lock = threading.Lock()
        self.last_debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )
        self.last_tracking_seed: Optional[TrackingSeed] = None

    def localize(self, image, camera_matrix, distortion):
        started = time.perf_counter()
        with self.lock:
            return self._localize_locked(image, camera_matrix, distortion, started)

    def _failure(
        self,
        image,
        started,
        detail,
        keypoints=0,
        matches=0,
        matched_xyz=None,
        sift_ms=0.0,
        matching_ms=0.0,
        pnp_ms=0.0,
    ):
        if matched_xyz is None:
            matched_xyz = np.empty((0, 3), dtype=np.float64)
        self.last_debug = LocalizationDebug(
            np.asarray(matched_xyz, dtype=np.float64).reshape((-1, 3)),
            frozenset(),
            None,
            False,
        )
        self.last_tracking_seed = None
        result = LiveResult(
            "FAIL", detail, keypoints, matches, 0, 0.0, None, None, None, None,
            (time.perf_counter() - started) * 1000.0, "HIDDEN",
            sift_ms, matching_ms, pnp_ms, 0.0,
        )
        return result, self._decorate(image, result, [], None)

    def _localize_locked(self, image, camera_matrix, distortion, started):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        sift_done = time.perf_counter()
        sift_ms = (sift_done - started) * 1000.0
        if descriptors is None or not keypoints:
            return self._failure(
                image, started, "NO_SIFT_FEATURES", sift_ms=sift_ms
            )
        neighbors = self.matcher.knnMatch(
            descriptors.astype(np.float32, copy=False), k=self.knn
        )
        matches = select_unique_point_matches(
            neighbors, self.map_data.point3d_ids, self.ratio_threshold
        )
        matching_done = time.perf_counter()
        matching_ms = (matching_done - sift_done) * 1000.0
        object_points = np.asarray(
            [self.map_data.xyz[match.map_index] for match in matches], dtype=np.float64
        ).reshape((-1, 3))
        if len(matches) < 4:
            return self._failure(
                image,
                started,
                "INSUFFICIENT_2D3D_MATCHES",
                len(keypoints),
                len(matches),
                object_points,
                sift_ms,
                matching_ms,
            )
        image_points = np.asarray(
            [keypoints[match.query_index].pt for match in matches], dtype=np.float64
        )
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            iterationsCount=self.iterations,
            reprojectionError=self.ransac_threshold,
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        pnp_ransac_done = time.perf_counter()
        pnp_ms = (pnp_ransac_done - matching_done) * 1000.0
        if not success or inliers is None or len(inliers) < 4:
            return self._failure(
                image,
                started,
                "PNP_RANSAC_FAILED",
                len(keypoints),
                len(matches),
                object_points,
                sift_ms,
                matching_ms,
                pnp_ms,
            )
        inlier_indices = [int(value) for value in inliers.ravel()]
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        inlier_ratio = len(inlier_indices) / len(matches)
        accepted = (
            len(inlier_indices) >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
        )
        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = (-rotation.T @ tvec).reshape(3)
        self.last_debug = LocalizationDebug(
            object_points.copy(),
            frozenset(inlier_indices),
            tuple(float(value) for value in camera_center),
            True,
        )
        if accepted:
            self.last_tracking_seed = TrackingSeed(
                gray.copy(),
                image_points[inlier_indices].copy(),
                object_points[inlier_indices].copy(),
                rvec.copy(),
                tvec.copy(),
            )
        else:
            self.last_tracking_seed = None
        pnp_ms = (time.perf_counter() - matching_done) * 1000.0
        canvas = image.copy()
        mpr_status = "HIDDEN"
        mpr_started = time.perf_counter()
        if accepted and self.mpr_renderer is not None:
            canvas, mpr_status = self.mpr_renderer.render(
                canvas, camera_matrix, distortion, rvec, tvec
            )
        mpr_ms = (time.perf_counter() - mpr_started) * 1000.0
        result = LiveResult(
            "PASS" if accepted else "FAIL",
            "OK" if accepted else "PNP_BELOW_ACCEPTANCE_THRESHOLD",
            len(keypoints),
            len(matches),
            len(inlier_indices),
            inlier_ratio,
            float(np.mean(errors[inlier_indices])),
            tuple(float(value) for value in rvec.ravel()),
            tuple(float(value) for value in tvec.ravel()),
            tuple(float(value) for value in camera_center),
            (time.perf_counter() - started) * 1000.0,
            mpr_status,
            sift_ms,
            matching_ms,
            pnp_ms,
            mpr_ms,
        )
        return result, self._decorate(canvas, result, matches, set(inlier_indices), keypoints)

    @staticmethod
    def _decorate(image, result, matches, inliers, keypoints=None):
        canvas = image.copy()
        if keypoints is not None and inliers is not None:
            for index, match in enumerate(matches):
                u, v = keypoints[match.query_index].pt
                color = (40, 230, 70) if index in inliers else (40, 40, 230)
                cv2.circle(canvas, (round(u), round(v)), 2, color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        color = (80, 230, 80) if result.status == "PASS" else (70, 70, 240)
        cv2.putText(
            canvas,
            f"{result.status}  matches {result.matches}  inliers {result.inliers}  MPR {result.mpr_status}",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{result.detail}  {result.processing_ms:.0f} ms",
            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
        return canvas


class TemporalGlobalPnPLocalizer:
    """Track accepted global-map inliers and periodically replenish them.

    Global SIFT matching remains the relocalizer.  Between global searches the
    accepted 2D observations retain their 3D identities and are propagated with
    pyramidal Lucas-Kanade optical flow.  A forward/backward check rejects drift,
    and the previous pose seeds a small, fast PnP RANSAC.
    """

    def __init__(
        self,
        global_localizer: LivePnPLocalizer,
        *,
        reseed_frames: int = 10,
        min_track_points: int = 10,
        flow_window: int = 31,
        flow_levels: int = 3,
        forward_backward_error: float = 1.5,
        tracking_iterations: int = 150,
    ) -> None:
        if reseed_frames <= 0 or min_track_points < 4:
            raise LivePnPError("Invalid temporal tracking frame thresholds")
        if flow_window < 5 or flow_window % 2 == 0 or flow_levels < 0:
            raise LivePnPError("Invalid optical-flow window or pyramid level")
        if forward_backward_error <= 0 or tracking_iterations <= 0:
            raise LivePnPError("Invalid optical-flow or tracking PnP threshold")
        self.global_localizer = global_localizer
        self.map_data = global_localizer.map_data
        self.mpr_renderer = global_localizer.mpr_renderer
        self.reseed_frames = reseed_frames
        self.min_track_points = min_track_points
        self.flow_window = flow_window
        self.flow_levels = flow_levels
        self.forward_backward_error = forward_backward_error
        self.tracking_iterations = tracking_iterations
        self.seed: Optional[TrackingSeed] = None
        self.frames_since_global = reseed_frames
        self.last_debug = global_localizer.last_debug

    def localize(self, image, camera_matrix, distortion):
        if self.seed is not None:
            if self.frames_since_global < self.reseed_frames:
                tracked = self._track(image, camera_matrix, distortion)
                if tracked is not None:
                    self.frames_since_global += 1
                    return tracked
            else:
                # Replenishment is allowed to fail without throwing away a
                # still-valid temporal track.  This prevents a single weak
                # global frame from causing periodic MPR flicker.
                previous_seed = self._copy_seed(self.seed)
                global_result = self._global(image, camera_matrix, distortion)
                if global_result[0].status == "PASS":
                    return global_result
                self.seed = previous_seed
                tracked = self._track(image, camera_matrix, distortion)
                if tracked is not None:
                    self.frames_since_global = 0
                    return tracked
        return self._global(image, camera_matrix, distortion)

    def _global(self, image, camera_matrix, distortion):
        result, display = self.global_localizer.localize(
            image, camera_matrix, distortion
        )
        self.last_debug = self.global_localizer.last_debug
        self.frames_since_global = 0
        seed = self.global_localizer.last_tracking_seed
        self.seed = self._copy_seed(seed) if seed is not None else None
        return result, display

    @staticmethod
    def _copy_seed(seed: TrackingSeed) -> TrackingSeed:
        return TrackingSeed(
            np.asarray(seed.gray).copy(),
            np.asarray(seed.image_points, dtype=np.float64).copy(),
            np.asarray(seed.object_points, dtype=np.float64).copy(),
            np.asarray(seed.rvec, dtype=np.float64).copy(),
            np.asarray(seed.tvec, dtype=np.float64).copy(),
        )

    def _track(self, image, camera_matrix, distortion):
        started = time.perf_counter()
        seed = self.seed
        if seed is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        previous = np.asarray(seed.image_points, dtype=np.float32).reshape((-1, 1, 2))
        if len(previous) < self.min_track_points:
            return None
        lk = dict(
            winSize=(self.flow_window, self.flow_window),
            maxLevel=self.flow_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            np.asarray(seed.gray), gray, previous, None, **lk
        )
        if forward is None or forward_status is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, np.asarray(seed.gray), forward, None, **lk
        )
        if backward is None or backward_status is None:
            return None
        forward_flat = forward.reshape((-1, 2))
        backward_flat = backward.reshape((-1, 2))
        previous_flat = previous.reshape((-1, 2))
        fb_error = np.linalg.norm(backward_flat - previous_flat, axis=1)
        height, width = gray.shape[:2]
        valid = (
            (forward_status.reshape(-1) != 0)
            & (backward_status.reshape(-1) != 0)
            & np.isfinite(forward_flat).all(axis=1)
            & (fb_error <= self.forward_backward_error)
            & (forward_flat[:, 0] >= 0)
            & (forward_flat[:, 0] < width)
            & (forward_flat[:, 1] >= 0)
            & (forward_flat[:, 1] < height)
        )
        image_points = forward_flat[valid].astype(np.float64, copy=False)
        object_points = np.asarray(seed.object_points, dtype=np.float64)[valid]
        flow_done = time.perf_counter()
        flow_ms = (flow_done - started) * 1000.0
        if len(image_points) < self.min_track_points:
            return None
        rvec_guess = np.asarray(seed.rvec, dtype=np.float64).copy()
        tvec_guess = np.asarray(seed.tvec, dtype=np.float64).copy()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec=rvec_guess,
            tvec=tvec_guess,
            useExtrinsicGuess=True,
            iterationsCount=self.tracking_iterations,
            reprojectionError=self.global_localizer.ransac_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < 4:
            return None
        inlier_indices = [int(value) for value in inliers.ravel()]
        guarded = (getattr(self.global_localizer, 'pipeline_policy', 'legacy') == 'validated')
        initial_rvec, initial_tvec = rvec.copy(), tvec.copy()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        errors = np.linalg.norm(projected.reshape((-1, 2)) - image_points, axis=1)
        if guarded:
            initial_projection = cv2.projectPoints(object_points, initial_rvec,
                initial_tvec, camera_matrix, distortion)[0].reshape(-1, 2)
            initial_errors = np.linalg.norm(initial_projection-image_points, axis=1)
            if (not np.isfinite(errors).all() or
                np.mean(errors[inlier_indices]**2) > np.mean(initial_errors[inlier_indices]**2)):
                rvec, tvec, errors = initial_rvec, initial_tvec, initial_errors
            rotation = cv2.Rodrigues(rvec)[0]
            depths = (np.einsum('ni,ji->nj', object_points, rotation) + tvec.reshape(3))[:, 2]
            inlier_indices = np.flatnonzero((errors <= self.global_localizer.ransac_threshold)
                & np.isfinite(errors) & (depths > 0)).tolist()
        inlier_ratio = len(inlier_indices) / len(image_points)
        accepted = (
            len(inlier_indices) >= self.global_localizer.min_inliers
            and inlier_ratio >= self.global_localizer.min_inlier_ratio
        )
        if not accepted:
            return None
        if guarded:
            quality = self.global_localizer.validate_pose(object_points, image_points,
                rvec, tvec, inlier_indices, image.shape, camera_matrix, distortion)
            self.global_localizer.last_tracking_quality = quality
            if not quality['passed']:
                return None
        rotation, _ = cv2.Rodrigues(rvec)
        camera_center = (-rotation.T @ tvec).reshape(3)
        self.last_debug = LocalizationDebug(
            object_points.copy(),
            frozenset(inlier_indices),
            tuple(float(value) for value in camera_center),
            True,
        )
        self.seed = TrackingSeed(
            gray.copy(),
            image_points[inlier_indices].copy(),
            object_points[inlier_indices].copy(),
            rvec.copy(),
            tvec.copy(),
        )
        pnp_done = time.perf_counter()
        canvas = image.copy()
        mpr_status = "HIDDEN"
        mpr_started = time.perf_counter()
        if self.mpr_renderer is not None:
            canvas, mpr_status = self.mpr_renderer.render(
                canvas, camera_matrix, distortion, rvec, tvec
            )
        mpr_ms = (time.perf_counter() - mpr_started) * 1000.0
        pnp_ms = (pnp_done - flow_done) * 1000.0
        result = LiveResult(
            "PASS",
            "TRACKING",
            len(image_points),
            len(image_points),
            len(inlier_indices),
            inlier_ratio,
            float(np.mean(errors[inlier_indices])),
            tuple(float(value) for value in rvec.ravel()),
            tuple(float(value) for value in tvec.ravel()),
            tuple(float(value) for value in camera_center),
            (time.perf_counter() - started) * 1000.0,
            mpr_status,
            0.0,
            flow_ms,
            pnp_ms,
            mpr_ms,
        )
        return result, self._decorate_tracking(
            canvas, result, image_points, set(inlier_indices)
        )

    @staticmethod
    def _decorate_tracking(image, result, image_points, inliers):
        canvas = image.copy()
        for index, (u, v) in enumerate(image_points):
            color = (40, 230, 70) if index in inliers else (40, 40, 230)
            cv2.circle(canvas, (round(float(u)), round(float(v))), 3, color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"PASS  optical-flow tracking  points {result.matches}  inliers {result.inliers}  MPR {result.mpr_status}",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (80, 230, 80), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"TRACKING  {result.processing_ms:.0f} ms  periodic global reseed",
            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (240, 240, 240), 1, cv2.LINE_AA,
        )
        return canvas


class LiveState:
    def __init__(self, localizer: LivePnPLocalizer) -> None:
        self.localizer = localizer
        self.display_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self.last_result: Optional[dict[str, object]] = None

    def publish_display(self, frame) -> None:
        try:
            self.display_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            self.display_queue.put_nowait(frame)


def _parse_float_header(handler: BaseHTTPRequestHandler, name: str) -> float:
    raw = handler.headers.get(name)
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError as exc:
        raise LivePnPError(f"Invalid {name} header") from exc
    if not math.isfinite(value):
        raise LivePnPError(f"Missing or invalid {name} header")
    return value


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ViewSyncLivePnP/1.0"

        def _json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._json(200, {"status": "ready", "last_result": state.last_result})
            else:
                self._json(404, {"status": "error", "detail": "NOT_FOUND"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/frame":
                self._json(404, {"status": "error", "detail": "NOT_FOUND"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 12_000_000:
                    raise LivePnPError("Invalid JPEG body length")
                encoded = np.frombuffer(self.rfile.read(length), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise LivePnPError("JPEG decode failed")
                fx = _parse_float_header(self, "X-Fx")
                fy = _parse_float_header(self, "X-Fy")
                cx = _parse_float_header(self, "X-Cx")
                cy = _parse_float_header(self, "X-Cy")
                camera_matrix = np.asarray(
                    ((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64
                )
                distortion = np.zeros(5, dtype=np.float64)
                result, display = state.localizer.localize(
                    image, camera_matrix, distortion
                )
                payload = result.as_dict()
                state.last_result = payload
                state.publish_display(display)
                self._json(200, payload)
            except (LivePnPError, ValueError, cv2.error) as exc:
                self._json(400, {"status": "ERROR", "detail": str(exc)})
            except Exception as exc:  # keep the capture session alive, but report the frame error
                self._json(500, {"status": "ERROR", "detail": f"SERVER_ERROR: {exc}"})

        def log_message(self, format: str, *args) -> None:
            if self.command == "POST":
                result = state.last_result or {}
                print(
                    f"frame: {result.get('status', '?')} matches={result.get('matches', 0)} "
                    f"inliers={result.get('inliers', 0)} {result.get('processing_ms', 0):.0f}ms"
                )
            else:
                super().log_message(format, *args)

    return Handler


def local_ip_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


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


def run_test_image(args, localizer: LivePnPLocalizer) -> int:
    if args.intrinsics is None:
        raise LivePnPError("--test-image requires --intrinsics FX FY CX CY")
    image = cv2.imread(str(args.test_image), cv2.IMREAD_COLOR)
    if image is None:
        raise LivePnPError(f"Cannot read test image: {args.test_image}")
    fx, fy, cx, cy = args.intrinsics
    matrix = np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
    result, display = localizer.localize(image, matrix, np.zeros(5, dtype=np.float64))
    output = args.test_output
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), display):
        raise LivePnPError(f"Cannot write test output: {output}")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    print(f"Overlay: {output}")
    return 0 if result.status == "PASS" else 2


def serve(args, localizer: LivePnPLocalizer) -> int:
    state = LiveState(localizer)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    worker = threading.Thread(target=server.serve_forever, name="http-server", daemon=True)
    worker.start()
    address = local_ip_address() if args.host in ("0.0.0.0", "::") else args.host
    print("Live PnP server is ready")
    print(f"iPhone server URL: http://{address}:{args.port}")
    print(f"Health check: http://{address}:{args.port}/health")
    print("Press Q in the preview window or Ctrl-C to stop.")
    try:
        while True:
            try:
                frame = state.display_queue.get(timeout=0.05)
                if not args.headless:
                    cv2.imshow("ViewSync Live PnP + MPR", frame)
            except queue.Empty:
                pass
            if not args.headless and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive iPhone JPEG frames and run SIFT/RANSAC-PnP/MPR on the Mac."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-mpr", action="store_true")
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
    parser.add_argument("--test-image", type=Path)
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument(
        "--test-output",
        type=Path,
        default=SCRIPT_DIR / "output_colmap_image" / "live_pnp_test.jpg",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        localizer = build_localizer(args)
        if args.test_image is not None:
            return run_test_image(args, localizer)
        return serve(args, localizer)
    except (LivePnPError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
