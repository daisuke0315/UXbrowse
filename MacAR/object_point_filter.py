#!/usr/bin/env python3
"""Extract object-only points from a COLMAP text Points3D.txt file.

The filtering core intentionally uses only the Python standard library. NumPy and
OpenCV are imported lazily when a QC preview or interactive viewer is requested.
"""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import hashlib
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple


DEFAULT_INPUT = DATA_DIR / "colmap" / "Points3D.txt"
DEFAULT_OUTPUT_DIR = DATA_DIR / "output"
RC_TO_COLMAP = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


class FilterError(RuntimeError):
    """An expected input or filtering error suitable for showing to the user."""


@dataclass(frozen=True)
class PointRecord:
    point_id: int
    xyz: Tuple[float, float, float]
    rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class FocusBox:
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    rotation: Tuple[Tuple[float, float, float], ...]  # local -> world
    rotation_euler: Optional[Tuple[float, float, float]]
    angle_unit: str = "degrees"
    input_coordinate_system: str = "colmap"
    input_center: Optional[Tuple[float, float, float]] = None
    input_rotation_matrix: Optional[Tuple[Tuple[float, float, float], ...]] = None

    def contains(self, xyz: Sequence[float], epsilon: float = 1e-9) -> bool:
        # For an orthonormal rotation matrix, inverse(R) == transpose(R).
        delta = tuple(xyz[index] - self.center[index] for index in range(3))
        local = tuple(
            sum(self.rotation[row][column] * delta[row] for row in range(3))
            for column in range(3)
        )
        return all(
            abs(local[index]) <= self.size[index] * 0.5 + epsilon
            for index in range(3)
        )


@dataclass(frozen=True)
class FilterResult:
    input_points: int
    selected_points: int
    rejected_points: int
    filtered_path: Path
    selected_ids_path: Path
    summary_path: Path
    selected_sample: Sequence[Tuple[float, float, float, int, int, int]]
    rejected_sample: Sequence[Tuple[float, float, float, int, int, int]]


class ReservoirSampler:
    """Fixed-size deterministic reservoir for large point clouds."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = max(0, capacity)
        self.items: List[Tuple[float, float, float, int, int, int]] = []
        self.seen = 0
        self.random = random.Random(seed)

    def add(self, record: PointRecord) -> None:
        self.seen += 1
        item = (*record.xyz, *record.rgb)
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self.random.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = item


def _finite_float(value: str, label: str, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise FilterError(f"Line {line_number}: {label} is not a number: {value!r}") from exc
    if not math.isfinite(result):
        raise FilterError(f"Line {line_number}: {label} must be finite: {value!r}")
    return result


def parse_point_line(line: str, line_number: int) -> PointRecord:
    fields = line.split()
    if len(fields) < 8:
        raise FilterError(
            f"Line {line_number}: expected at least 8 COLMAP fields, got {len(fields)}"
        )
    try:
        point_id = int(fields[0])
    except ValueError as exc:
        raise FilterError(
            f"Line {line_number}: POINT3D_ID is not an integer: {fields[0]!r}"
        ) from exc

    xyz = tuple(
        _finite_float(fields[index], "XYZ"[index - 1], line_number)
        for index in range(1, 4)
    )
    try:
        rgb = tuple(int(fields[index]) for index in range(4, 7))
    except ValueError as exc:
        raise FilterError(f"Line {line_number}: RGB must contain integers") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise FilterError(f"Line {line_number}: RGB values must be in the range 0..255")

    _finite_float(fields[7], "ERROR", line_number)
    track = fields[8:]
    if len(track) % 2 != 0:
        raise FilterError(
            f"Line {line_number}: TRACK must contain IMAGE_ID/POINT2D_IDX pairs"
        )
    try:
        [int(value) for value in track]
    except ValueError as exc:
        raise FilterError(f"Line {line_number}: TRACK values must be integers") from exc

    return PointRecord(point_id, xyz, rgb)  # type: ignore[arg-type]


def euler_rotation_matrix(
    angles: Sequence[float], *, degrees: bool = True
) -> Tuple[Tuple[float, float, float], ...]:
    """Return Rz @ Ry @ Rx, an active local-to-world rotation matrix."""
    rx, ry, rz = angles
    if degrees:
        rx, ry, rz = (math.radians(value) for value in (rx, ry, rz))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def multiply_matrices(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def transform_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> Tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def validate_rotation_matrix(
    values: Sequence[float], tolerance: float = 1e-6
) -> Tuple[Tuple[float, float, float], ...]:
    if len(values) != 9 or any(not math.isfinite(value) for value in values):
        raise FilterError("Rotation matrix must contain 9 finite numbers")
    matrix = tuple(tuple(values[row * 3 : row * 3 + 3]) for row in range(3))
    for row_a in range(3):
        for row_b in range(3):
            dot = sum(matrix[row_a][i] * matrix[row_b][i] for i in range(3))
            expected = 1.0 if row_a == row_b else 0.0
            if abs(dot - expected) > tolerance:
                raise FilterError("Rotation matrix must be orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise FilterError("Rotation matrix determinant must be +1")
    return matrix


def make_focus_box(
    center: Sequence[float],
    size: Sequence[float],
    rotation_euler: Optional[Sequence[float]] = None,
    rotation_matrix: Optional[Sequence[float]] = None,
    *,
    degrees: bool = True,
    input_coordinate_system: str = "colmap",
) -> FocusBox:
    if len(center) != 3 or len(size) != 3:
        raise FilterError("Center and size must each contain exactly 3 numbers")
    if any(not math.isfinite(value) for value in (*center, *size)):
        raise FilterError("Center and size must contain finite numbers")
    if any(value <= 0 for value in size):
        raise FilterError("Focus size values Sx, Sy and Sz must all be greater than 0")
    if rotation_euler is not None and rotation_matrix is not None:
        raise FilterError("Specify either Euler rotation or a rotation matrix, not both")
    if input_coordinate_system not in ("rc", "colmap"):
        raise FilterError("Input coordinate system must be 'rc' or 'colmap'")

    if rotation_matrix is not None:
        input_matrix = validate_rotation_matrix(rotation_matrix)
        euler = None
    else:
        euler_values = tuple(rotation_euler or (0.0, 0.0, 0.0))
        if len(euler_values) != 3 or any(not math.isfinite(value) for value in euler_values):
            raise FilterError("Euler rotation must contain 3 finite numbers")
        input_matrix = euler_rotation_matrix(euler_values, degrees=degrees)
        euler = euler_values

    input_center = tuple(center)
    if input_coordinate_system == "rc":
        resolved_center = transform_vector(RC_TO_COLMAP, input_center)
        resolved_rotation = multiply_matrices(RC_TO_COLMAP, input_matrix)
    else:
        resolved_center = input_center
        resolved_rotation = input_matrix
    return FocusBox(
        resolved_center,
        tuple(size),
        resolved_rotation,
        euler,
        "degrees" if degrees else "radians",
        input_coordinate_system,
        input_center,  # type: ignore[arg-type]
        input_matrix,
    )  # type: ignore[arg-type]


def _temporary_file(output_dir: Path, suffix: str) -> Tuple[TextIO, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output_dir, suffix=suffix, delete=False
    )
    return handle, Path(handle.name)


def _verify_outputs(
    filtered_path: Path,
    selected_ids_path: Path,
    box: FocusBox,
    expected_count: int,
    expected_digest: str,
) -> None:
    count = 0
    digest = hashlib.sha256()
    with (
        filtered_path.open("r", encoding="utf-8", newline="") as source,
        selected_ids_path.open("r", encoding="utf-8") as ids_source,
    ):
        ids_iterator = (line.strip() for line in ids_source if line.strip())
        for line_number, line in enumerate(source, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            record = parse_point_line(line, line_number)
            if not box.contains(record.xyz):
                raise FilterError("QC2 failed: an output point is outside the focus region")
            digest.update(line.encode("utf-8"))
            try:
                selected_id = int(next(ids_iterator))
            except StopIteration as exc:
                raise FilterError("QC failed: selected_point_ids.txt has too few IDs") from exc
            except ValueError as exc:
                raise FilterError(
                    "QC failed: selected_point_ids.txt contains a non-integer"
                ) from exc
            if selected_id != record.point_id:
                raise FilterError("QC failed: selected point IDs or ordering do not match")
            count += 1
        try:
            next(ids_iterator)
        except StopIteration:
            pass
        else:
            raise FilterError("QC failed: selected_point_ids.txt has too many IDs")
    if count != expected_count:
        raise FilterError("QC failed: selected point count does not match")
    if digest.hexdigest() != expected_digest:
        raise FilterError("QC3 failed: one or more selected point records changed")


def _summary_text(
    input_count: int,
    selected_count: int,
    box: FocusBox,
    preview_path: Optional[Path],
) -> str:
    rejected_count = input_count - selected_count
    ratio = selected_count / input_count if input_count else 0.0
    input_center = box.input_center or box.center
    lines = [
        f"Input points: {input_count}",
        f"Selected points: {selected_count}",
        f"Rejected points: {rejected_count}",
        f"Selected ratio: {ratio:.6%}",
        "",
        f"Input coordinate system: {box.input_coordinate_system}",
        "RC to COLMAP mapping: (X, Y, Z) -> (X, -Z, Y)" if box.input_coordinate_system == "rc"
        else "RC to COLMAP mapping: not applied",
        "",
        "Focus center:",
        f"Cx: {input_center[0]:.17g}",
        f"Cy: {input_center[1]:.17g}",
        f"Cz: {input_center[2]:.17g}",
        "",
        "Focus size:",
        f"Sx: {box.size[0]:.17g}",
        f"Sy: {box.size[1]:.17g}",
        f"Sz: {box.size[2]:.17g}",
        "",
        "Focus rotation:",
    ]
    if box.rotation_euler is not None:
        lines.extend(
            [
                f"Rx: {box.rotation_euler[0]:.17g}",
                f"Ry: {box.rotation_euler[1]:.17g}",
                f"Rz: {box.rotation_euler[2]:.17g}",
                f"Angle unit: {box.angle_unit}",
                "Convention: R = Rz @ Ry @ Rx (local to world)",
            ]
        )
    else:
        lines.extend(
            [
                "Rx: N/A (rotation matrix supplied)",
                "Ry: N/A (rotation matrix supplied)",
                "Rz: N/A (rotation matrix supplied)",
            ]
        )
    lines.extend(
        [
            "Input rotation matrix (row-major, local to input coordinates):",
            *[
                " ".join(f"{value:.17g}" for value in row)
                for row in (box.input_rotation_matrix or box.rotation)
            ],
            "",
            "Resolved COLMAP focus center:",
            f"Cx: {box.center[0]:.17g}",
            f"Cy: {box.center[1]:.17g}",
            f"Cz: {box.center[2]:.17g}",
            "Resolved rotation matrix (row-major, local to COLMAP world):",
            *[" ".join(f"{value:.17g}" for value in row) for row in box.rotation],
            "",
            "QC:",
            "QC1 count consistency: PASS",
            "QC2 all selected points inside focus region: PASS",
            "QC3 selected COLMAP records unchanged: PASS",
            (
                f"QC4 visual inspection: REQUIRED ({preview_path})"
                if preview_path is not None
                else "QC4 visual inspection: NOT RUN (use --preview or --visualize)"
            ),
            "",
            "Point-record coordinate transform: none (focus box conversion only)",
        ]
    )
    return "\n".join(lines) + "\n"


def filter_points(
    input_path: Path,
    output_dir: Path,
    box: FocusBox,
    *,
    sample_capacity: int = 0,
    preview_path: Optional[Path] = None,
) -> FilterResult:
    if not input_path.is_file():
        raise FilterError(f"Points3D.txt not found: {input_path}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FilterError(f"Cannot create output directory: {output_dir}: {exc}") from exc

    body_handle, body_temp = _temporary_file(output_dir, ".points-body.tmp")
    ids_handle, ids_temp_path = _temporary_file(output_dir, ".ids.tmp")
    ids_temp: Optional[Path] = ids_temp_path
    selected_sampler = ReservoirSampler(sample_capacity, 23063)
    rejected_sampler = ReservoirSampler(sample_capacity, 57287)
    input_count = selected_count = 0
    selected_digest = hashlib.sha256()
    final_temp: Optional[Path] = None
    summary_temp: Optional[Path] = None

    try:
        try:
            source = input_path.open("r", encoding="utf-8", newline="")
        except (OSError, UnicodeError) as exc:
            raise FilterError(f"Cannot read Points3D.txt: {input_path}: {exc}") from exc
        with source, body_handle, ids_handle:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                record = parse_point_line(line, line_number)
                input_count += 1
                if box.contains(record.xyz):
                    body_handle.write(line)
                    ids_handle.write(f"{record.point_id}\n")
                    selected_digest.update(line.encode("utf-8"))
                    selected_count += 1
                    selected_sampler.add(record)
                else:
                    rejected_sampler.add(record)
        if input_count == 0:
            raise FilterError("Points3D.txt contains no 3D point records")
        if selected_count == 0:
            raise FilterError("No points found inside focus region")
        rejected_count = input_count - selected_count
        if input_count != selected_count + rejected_count:
            raise FilterError("QC1 failed: input count != selected + rejected")

        filtered_path = output_dir / "filtered_Points3D.txt"
        selected_ids_path = output_dir / "selected_point_ids.txt"
        summary_path = output_dir / "filter_summary.txt"

        final_handle, final_temp = _temporary_file(output_dir, ".filtered.tmp")
        with final_handle, body_temp.open("r", encoding="utf-8", newline="") as body:
            final_handle.write("# 3D point list with one line of data per point:\n")
            final_handle.write(
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            )
            final_handle.write(f"# Number of points: {selected_count}\n")
            shutil.copyfileobj(body, final_handle)

        # Verify temporary products before replacing any existing successful output.
        assert final_temp is not None and ids_temp is not None
        _verify_outputs(
            final_temp,
            ids_temp,
            box,
            selected_count,
            selected_digest.hexdigest(),
        )
        summary_handle, summary_temp = _temporary_file(output_dir, ".summary.tmp")
        with summary_handle:
            summary_handle.write(
                _summary_text(input_count, selected_count, box, preview_path)
            )

        os.replace(final_temp, filtered_path)
        final_temp = None
        os.replace(ids_temp, selected_ids_path)
        ids_temp = None
        os.replace(summary_temp, summary_path)
        summary_temp = None
        return FilterResult(
            input_count,
            selected_count,
            rejected_count,
            filtered_path,
            selected_ids_path,
            summary_path,
            selected_sampler.items,
            rejected_sampler.items,
        )
    except UnicodeError as exc:
        raise FilterError(f"Points3D.txt is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise FilterError(f"File operation failed: {exc}") from exc
    finally:
        if not body_handle.closed:
            body_handle.close()
        if not ids_handle.closed:
            ids_handle.close()
        for path in (body_temp, ids_temp, final_temp, summary_temp):
            if path and path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


def analyze_points(input_path: Path) -> Tuple[int, Tuple[float, ...], Tuple[float, ...]]:
    if not input_path.is_file():
        raise FilterError(f"Points3D.txt not found: {input_path}")
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    count = 0
    try:
        with input_path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                record = parse_point_line(line, line_number)
                count += 1
                for axis in range(3):
                    minima[axis] = min(minima[axis], record.xyz[axis])
                    maxima[axis] = max(maxima[axis], record.xyz[axis])
    except (OSError, UnicodeError) as exc:
        raise FilterError(f"Cannot read Points3D.txt: {input_path}: {exc}") from exc
    if count == 0:
        raise FilterError("Points3D.txt contains no 3D point records")
    return count, tuple(minima), tuple(maxima)


def _box_corners(box: FocusBox) -> List[Tuple[float, float, float]]:
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                local = (sx * box.size[0], sy * box.size[1], sz * box.size[2])
                world = tuple(
                    box.center[row]
                    + sum(box.rotation[row][column] * local[column] for column in range(3))
                    for row in range(3)
                )
                corners.append(world)  # type: ignore[arg-type]
    return corners


def show_point_cloud(
    selected: Sequence[Tuple[float, float, float, int, int, int]],
    rejected: Sequence[Tuple[float, float, float, int, int, int]],
    box: FocusBox,
    preview_path: Path,
    *,
    interactive: bool,
) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise FilterError(
            "Visualization requires OpenCV and NumPy. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    selected_array = np.asarray(selected, dtype=np.float64)
    rejected_array = np.asarray(rejected, dtype=np.float64)
    corners = np.asarray(_box_corners(box), dtype=np.float64)
    target = np.asarray(box.center, dtype=np.float64)
    focus_radius = max(float(np.linalg.norm(corners - target, axis=1).max()), 1e-9)
    all_xyz = [array[:, :3] for array in (selected_array, rejected_array) if len(array)]
    all_radius = focus_radius
    if all_xyz:
        all_radius = max(
            focus_radius,
            max(float(np.linalg.norm(array - target, axis=1).max()) for array in all_xyz),
        )

    state = {
        "yaw": math.radians(-35.0),
        "pitch": math.radians(25.0),
        "zoom": 1.0,
        "show_rejected": True,
        "fit_radius": focus_radius * 1.35,
        "dragging": False,
        "mouse": (0, 0),
        "dirty": True,
    }
    width, height = 1280, 800
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]

    def view_rotation() -> "np.ndarray":
        yaw, pitch = state["yaw"], state["pitch"]
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
        rx = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float64)
        return rx @ rz

    def project(xyz: "np.ndarray", rotation: "np.ndarray") -> "np.ndarray":
        if not len(xyz):
            return np.empty((0, 3), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            viewed = (xyz - target) @ rotation.T
        scale = min(width, height) * 0.42 * state["zoom"] / state["fit_radius"]
        viewed[:, 0] = width * 0.5 + viewed[:, 0] * scale
        viewed[:, 1] = height * 0.5 - viewed[:, 1] * scale
        return viewed

    def draw_points(
        image: "np.ndarray",
        sample: "np.ndarray",
        rotation: "np.ndarray",
        selected_points: bool,
    ) -> None:
        if not len(sample):
            return
        projected = project(sample[:, :3], rotation)
        visible = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0) & (projected[:, 0] <= width - 1)
            & (projected[:, 1] >= 0) & (projected[:, 1] <= height - 1)
        )
        projected = projected[visible]
        colors = sample[visible, 3:6]
        if not len(projected):
            return
        order = np.argsort(projected[:, 2])
        xy = np.rint(projected[order, :2]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        if selected_points:
            bgr = colors[order, ::-1]
            bgr = np.clip(np.maximum(bgr, 75), 0, 255).astype(np.uint8)
            image[xy[:, 1], xy[:, 0]] = bgr
            # A second pixel makes selected points easier to distinguish.
            x2 = np.clip(xy[:, 0] + 1, 0, width - 1)
            image[xy[:, 1], x2] = bgr
        else:
            image[xy[:, 1], xy[:, 0]] = (72, 72, 72)

    def render() -> "np.ndarray":
        image = np.full((height, width, 3), 18, dtype=np.uint8)
        rotation = view_rotation()
        if state["show_rejected"]:
            draw_points(image, rejected_array, rotation, False)
        draw_points(image, selected_array, rotation, True)
        projected_corners = project(corners.copy(), rotation)
        for first, second in edges:
            p1 = tuple(np.rint(projected_corners[first, :2]).astype(int))
            p2 = tuple(np.rint(projected_corners[second, :2]).astype(int))
            cv2.line(image, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)

        origin = project(target.reshape(1, 3).copy(), rotation)[0]
        axis_length = min(box.size) * 0.4
        axis_colors = ((0, 0, 255), (0, 255, 0), (255, 80, 30))
        for axis in range(3):
            endpoint = target.copy()
            endpoint[axis] += axis_length
            projected_end = project(endpoint.reshape(1, 3), rotation)[0]
            cv2.arrowedLine(
                image,
                tuple(np.rint(origin[:2]).astype(int)),
                tuple(np.rint(projected_end[:2]).astype(int)),
                axis_colors[axis],
                2,
                cv2.LINE_AA,
                tipLength=0.15,
            )
            cv2.putText(
                image,
                "XYZ"[axis],
                tuple(np.rint(projected_end[:2] + (5, -5)).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                axis_colors[axis],
                1,
                cv2.LINE_AA,
            )

        cv2.putText(image, "Selected points (source RGB)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(image, "Rejected sample (gray) | Focus box (yellow)", (20, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(image, f"Selected: {len(selected):,} sampled | Rejected: {len(rejected):,} sampled",
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        if interactive:
            cv2.putText(image, "Drag: orbit  Wheel: zoom  A: background  F/G: fit focus/all  S: save  Q: quit",
                        (20, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (175, 175, 175), 1, cv2.LINE_AA)
        return image

    image = render()
    if not cv2.imwrite(str(preview_path), image):
        raise FilterError(f"OpenCV could not write preview image: {preview_path}")
    if not interactive:
        return

    window_name = "Object-only Sparse Point Cloud QC"

    def mouse_callback(event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["mouse"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            old_x, old_y = state["mouse"]
            state["yaw"] += (x - old_x) * 0.008
            state["pitch"] += (y - old_y) * 0.008
            state["pitch"] = max(-math.pi * 0.49, min(math.pi * 0.49, state["pitch"]))
            state["mouse"] = (x, y)
            state["dirty"] = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            state["zoom"] *= 1.12 if flags > 0 else 1 / 1.12
            state["zoom"] = max(0.05, min(50.0, state["zoom"]))
            state["dirty"] = True

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            if state["dirty"]:
                image = render()
                state["dirty"] = False
            cv2.imshow(window_name, image)
            key = cv2.waitKeyEx(16) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                state["show_rejected"] = not state["show_rejected"]
                state["dirty"] = True
            elif key == ord("f"):
                state["fit_radius"] = focus_radius * 1.35
                state["zoom"] = 1.0
                state["dirty"] = True
            elif key == ord("g"):
                state["fit_radius"] = all_radius * 1.08
                state["zoom"] = 1.0
                state["dirty"] = True
            elif key == ord("r"):
                state.update(yaw=math.radians(-35.0), pitch=math.radians(25.0), zoom=1.0)
                state["dirty"] = True
            elif key == ord("s"):
                cv2.imwrite(str(preview_path), image)
    except cv2.error as exc:
        raise FilterError(
            f"OpenCV GUI could not be opened; the PNG preview was still saved: {preview_path}: {exc}"
        ) from exc
    finally:
        cv2.destroyAllWindows()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract points inside an AABB/OBB from COLMAP Points3D.txt."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Validate the input and print XYZ bounds")
    analyze.add_argument("--input", type=Path, default=DEFAULT_INPUT)

    filtering = subparsers.add_parser("filter", help="Filter points and write QC outputs")
    filtering.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    filtering.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    filtering.add_argument(
        "--input-coordinates",
        choices=("rc", "colmap"),
        default="rc",
        help="Coordinate system of center/size/rotation values (default: rc)",
    )
    filtering.add_argument("--center", nargs=3, type=float, metavar=("CX", "CY", "CZ"), required=True)
    filtering.add_argument("--size", nargs=3, type=float, metavar=("SX", "SY", "SZ"), required=True)
    rotation_group = filtering.add_mutually_exclusive_group()
    rotation_group.add_argument(
        "--rotation", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
        help="Euler rotation; R = Rz @ Ry @ Rx (default: 0 0 0)",
    )
    rotation_group.add_argument(
        "--rotation-matrix", nargs=9, type=float,
        metavar=tuple(f"R{i}{j}" for i in range(3) for j in range(3)),
        help="Row-major orthonormal local-to-world rotation matrix",
    )
    filtering.add_argument("--radians", action="store_true", help="Interpret Euler angles as radians")
    filtering.add_argument(
        "--preview", type=Path, nargs="?", const=Path("point_cloud_preview.png"),
        help="Save an OpenCV QC preview; optional path (default: output directory)",
    )
    filtering.add_argument(
        "--visualize", action="store_true", help="Open the interactive OpenCV 3D QC viewer",
    )
    filtering.add_argument(
        "--max-preview-points", type=int, default=100_000,
        help="Reservoir size for each of selected/rejected preview points (default: 100000)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            count, minima, maxima = analyze_points(args.input)
            center = tuple((minima[index] + maxima[index]) * 0.5 for index in range(3))
            size = tuple(maxima[index] - minima[index] for index in range(3))
            print(f"Input: {args.input}")
            print(f"Points: {count}")
            print("Minimum XYZ: " + " ".join(f"{value:.17g}" for value in minima))
            print("Maximum XYZ: " + " ".join(f"{value:.17g}" for value in maxima))
            print("Bounds center: " + " ".join(f"{value:.17g}" for value in center))
            print("Bounds size: " + " ".join(f"{value:.17g}" for value in size))
            return 0

        if args.max_preview_points <= 0:
            raise FilterError("--max-preview-points must be greater than 0")
        box = make_focus_box(
            args.center,
            args.size,
            args.rotation,
            args.rotation_matrix,
            degrees=not args.radians,
            input_coordinate_system=args.input_coordinates,
        )
        wants_preview = args.preview is not None or args.visualize
        preview_path: Optional[Path] = None
        if wants_preview:
            requested = args.preview or Path("point_cloud_preview.png")
            preview_path = requested if requested.is_absolute() else args.output_dir / requested
        result = filter_points(
            args.input,
            args.output_dir,
            box,
            sample_capacity=args.max_preview_points if wants_preview else 0,
            preview_path=preview_path,
        )
        if wants_preview and preview_path is not None:
            show_point_cloud(
                result.selected_sample,
                result.rejected_sample,
                box,
                preview_path,
                interactive=args.visualize,
            )
        print(f"Input points: {result.input_points}")
        print(f"Selected points: {result.selected_points}")
        print(f"Rejected points: {result.rejected_points}")
        print(f"Input coordinates: {args.input_coordinates}")
        print("Resolved COLMAP center: " + " ".join(f"{value:.9g}" for value in box.center))
        print(f"Output directory: {args.output_dir}")
        if preview_path is not None:
            print(f"QC preview: {preview_path}")
        return 0
    except FilterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
