#!/usr/bin/env python3
"""Build an object-only SIFT 3D feature map from Step 2 observations."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from extract_object_observations import Camera, ObservationError, load_cameras


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_OBSERVATIONS = DEFAULT_OUTPUT_DIR / "object_observations.csv"
DEFAULT_CAMERAS = SCRIPT_DIR / "colmap" / "Cameras.txt"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "image"

REQUIRED_COLUMNS = (
    "image_id",
    "image_name",
    "camera_id",
    "u",
    "v",
    "point3d_id",
    "X",
    "Y",
    "Z",
)

METADATA_COLUMNS = (
    "point3d_id",
    "image_id",
    "image_name",
    "X",
    "Y",
    "Z",
    "rc_u",
    "rc_v",
    "sift_u",
    "sift_v",
    "pixel_distance",
    "sift_scale",
    "sift_angle",
    "sift_response",
    "point_observation_count",
)

FAILURE_COLUMNS = (
    "image_id",
    "image_name",
    "camera_id",
    "observation_count",
    "error_code",
    "detail",
)

UNMATCHED_COLUMNS = (
    "point3d_id",
    "image_id",
    "image_name",
    "rc_u",
    "rc_v",
    "error_code",
)


class FeatureMapError(RuntimeError):
    """Expected input, validation, or output error."""


@dataclass(frozen=True)
class Observation:
    image_id: int
    image_name: str
    camera_id: int
    u: float
    v: float
    point3d_id: int
    xyz: Tuple[float, float, float]
    xyz_text: Tuple[str, str, str]
    u_text: str
    v_text: str


@dataclass(frozen=True)
class ObservationScan:
    total_observations: int
    total_images: int
    xyz_by_point: Dict[int, Tuple[float, float, float]]
    point_observation_counts: Counter[int]


@dataclass(frozen=True)
class FeatureMatch:
    observation_index: int
    keypoint_index: int
    distance: float


@dataclass(frozen=True)
class ImageFeatureCount:
    image_id: int
    image_name: str
    observation_count: int
    descriptor_count: int


@dataclass
class ImageQCData:
    image_id: int
    image_name: str
    source_path: Path
    observation_count: int
    matches: List[Tuple[float, float, float, float, float]]
    unmatched_uv: List[Tuple[float, float]]


@dataclass(frozen=True)
class FeatureMapResult:
    total_images: int
    processed_images: int
    failed_images: int
    total_observations: int
    observations_with_descriptor: int
    unique_points: int
    points_with_descriptor: int
    features_path: Path
    metadata_path: Path
    summary_path: Path
    failures_path: Path
    unmatched_path: Path
    spatial_qc_path: Optional[Path]
    overlay_paths: Sequence[Path]


def _finite_float(text: str, label: str, row_number: int) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise FeatureMapError(
            f"object_observations.csv row {row_number}: invalid {label}: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise FeatureMapError(
            f"object_observations.csv row {row_number}: {label} must be finite"
        )
    return value


def _parse_observation(row: Dict[str, str], row_number: int) -> Observation:
    try:
        image_id = int(row["image_id"])
        camera_id = int(row["camera_id"])
        point3d_id = int(row["point3d_id"])
        raw_image_name = row["image_name"]
        image_name = raw_image_name.strip() if raw_image_name is not None else ""
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise FeatureMapError(
            f"object_observations.csv row {row_number}: invalid ID or image name"
        ) from exc
    if image_id <= 0 or camera_id <= 0 or point3d_id < 0 or not image_name:
        raise FeatureMapError(
            f"object_observations.csv row {row_number}: invalid ID or image name"
        )
    u_text, v_text = row["u"], row["v"]
    xyz_text = (row["X"], row["Y"], row["Z"])
    return Observation(
        image_id,
        image_name,
        camera_id,
        _finite_float(u_text, "u", row_number),
        _finite_float(v_text, "v", row_number),
        point3d_id,
        tuple(_finite_float(value, axis, row_number) for value, axis in zip(xyz_text, "XYZ")),  # type: ignore[arg-type]
        xyz_text,
        u_text,
        v_text,
    )


def _open_observation_reader(path: Path):
    if not path.is_file():
        raise FeatureMapError(f"object_observations.csv not found: {path}")
    try:
        source = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        raise FeatureMapError(f"Cannot read object_observations.csv: {path}: {exc}") from exc
    reader = csv.DictReader(source)
    missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
    if missing:
        source.close()
        raise FeatureMapError(
            "object_observations.csv is missing columns: " + ", ".join(missing)
        )
    return source, reader


def scan_observations(path: Path) -> ObservationScan:
    source, reader = _open_observation_reader(path)
    total = 0
    images: Dict[int, Tuple[str, int]] = {}
    xyz_by_point: Dict[int, Tuple[float, float, float]] = {}
    point_counts: Counter[int] = Counter()
    try:
        with source:
            for row_number, row in enumerate(reader, 2):
                observation = _parse_observation(row, row_number)
                image_identity = (observation.image_name, observation.camera_id)
                previous_image = images.setdefault(observation.image_id, image_identity)
                if previous_image != image_identity:
                    raise FeatureMapError(
                        f"IMAGE_ID {observation.image_id} has inconsistent name or CAMERA_ID"
                    )
                previous_xyz = xyz_by_point.setdefault(
                    observation.point3d_id, observation.xyz
                )
                if previous_xyz != observation.xyz:
                    raise FeatureMapError(
                        f"POINT3D_ID {observation.point3d_id} has inconsistent XYZ"
                    )
                point_counts[observation.point3d_id] += 1
                total += 1
    except (OSError, UnicodeError) as exc:
        raise FeatureMapError(f"Cannot scan object_observations.csv: {exc}") from exc
    if total == 0:
        raise FeatureMapError("object_observations.csv contains no observations")
    return ObservationScan(total, len(images), xyz_by_point, point_counts)


def iter_observation_groups(path: Path) -> Iterator[List[Observation]]:
    source, reader = _open_observation_reader(path)
    current_image_id: Optional[int] = None
    current: List[Observation] = []
    completed: Set[int] = set()
    try:
        with source:
            for row_number, row in enumerate(reader, 2):
                observation = _parse_observation(row, row_number)
                if current_image_id is None:
                    current_image_id = observation.image_id
                if observation.image_id != current_image_id:
                    completed.add(current_image_id)
                    yield current
                    if observation.image_id in completed:
                        raise FeatureMapError(
                            "object_observations.csv must group rows by IMAGE_ID; "
                            f"IMAGE_ID {observation.image_id} appears again"
                        )
                    current = []
                    current_image_id = observation.image_id
                if current and (
                    observation.image_name != current[0].image_name
                    or observation.camera_id != current[0].camera_id
                ):
                    raise FeatureMapError(
                        f"IMAGE_ID {observation.image_id} has inconsistent metadata"
                    )
                current.append(observation)
            if current:
                yield current
    except (OSError, UnicodeError) as exc:
        raise FeatureMapError(f"Cannot stream object_observations.csv: {exc}") from exc


def resolve_source_image(
    image_name: str,
    image_dir: Path,
    image_template: Optional[str],
    image_index_offset: int,
) -> Optional[Path]:
    exact = image_dir / image_name
    if exact.is_file():
        return exact
    if image_template is None:
        return None
    stem = Path(image_name).stem
    try:
        index = int(stem) + image_index_offset
        relative = image_template.format(index=index, stem=stem, name=image_name)
    except (ValueError, KeyError, IndexError) as exc:
        raise FeatureMapError(
            f"Cannot apply --image-template to {image_name!r}: {exc}"
        ) from exc
    candidate = image_dir / relative
    return candidate if candidate.is_file() else None


def match_observations_to_keypoints(
    observations: Sequence[Observation],
    keypoints: Sequence[object],
    max_distance: float,
) -> Tuple[List[FeatureMatch], Dict[int, str]]:
    """Find each observation's nearest keypoint, then enforce keypoint uniqueness."""
    if max_distance <= 0 or not math.isfinite(max_distance):
        raise FeatureMapError("--max-distance must be a positive finite number")
    cell_size = max_distance
    grid: Dict[Tuple[int, int], List[int]] = {}
    for keypoint_index, keypoint in enumerate(keypoints):
        u, v = keypoint.pt  # type: ignore[attr-defined]
        cell = (math.floor(u / cell_size), math.floor(v / cell_size))
        grid.setdefault(cell, []).append(keypoint_index)

    candidates: List[FeatureMatch] = []
    unmatched: Dict[int, str] = {}
    maximum_squared = max_distance * max_distance
    for observation_index, observation in enumerate(observations):
        center = (
            math.floor(observation.u / cell_size),
            math.floor(observation.v / cell_size),
        )
        best: Optional[Tuple[float, int]] = None
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for keypoint_index in grid.get(
                    (center[0] + offset_x, center[1] + offset_y), []
                ):
                    keypoint = keypoints[keypoint_index]
                    sift_u, sift_v = keypoint.pt  # type: ignore[attr-defined]
                    distance_squared = (
                        (observation.u - sift_u) ** 2
                        + (observation.v - sift_v) ** 2
                    )
                    candidate = (distance_squared, keypoint_index)
                    if distance_squared <= maximum_squared and (
                        best is None or candidate < best
                    ):
                        best = candidate
        if best is None:
            unmatched[observation_index] = "NO_NEARBY_FEATURE"
        else:
            candidates.append(
                FeatureMatch(observation_index, best[1], math.sqrt(best[0]))
            )

    # When multiple observations choose one keypoint, the closest pair wins.
    winners: Dict[int, FeatureMatch] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (item.distance, item.observation_index, item.keypoint_index),
    ):
        if candidate.keypoint_index not in winners:
            winners[candidate.keypoint_index] = candidate
        else:
            unmatched[candidate.observation_index] = "SIFT_KEYPOINT_CONFLICT"
    matches = sorted(winners.values(), key=lambda item: item.observation_index)
    return matches, unmatched


def _temporary_path(output_dir: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(dir=output_dir, suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    return path


def _choose_qc_images(
    counts: Sequence[ImageFeatureCount],
) -> List[Tuple[str, ImageFeatureCount]]:
    usable = sorted(
        (item for item in counts if item.descriptor_count > 0),
        key=lambda item: (item.descriptor_count, item.image_id),
    )
    if not usable:
        return []
    candidates = (
        ("low", usable[0]),
        ("medium", usable[len(usable) // 2]),
        ("high", usable[-1]),
    )
    result: List[Tuple[str, ImageFeatureCount]] = []
    used: Set[int] = set()
    for category, item in candidates:
        if item.image_id not in used:
            result.append((category, item))
            used.add(item.image_id)
    return result


def _generate_match_overlays(
    cv2,
    qc_data: Dict[int, ImageQCData],
    counts: Sequence[ImageFeatureCount],
    qc_dir: Path,
) -> List[Path]:
    targets = _choose_qc_images(counts)
    if not targets:
        return []
    qc_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    read_flags = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    for category, count in targets:
        data = qc_data[count.image_id]
        image = cv2.imread(str(data.source_path), read_flags)
        if image is None:
            continue
        for rc_u, rc_v in data.unmatched_uv:
            cv2.circle(
                image, (round(rc_u), round(rc_v)), 3, (40, 40, 230), 1, cv2.LINE_AA
            )
        for rc_u, rc_v, sift_u, sift_v, _distance in data.matches:
            rc_point = (round(rc_u), round(rc_v))
            sift_point = (round(sift_u), round(sift_v))
            cv2.line(image, rc_point, sift_point, (0, 220, 80), 1, cv2.LINE_AA)
            cv2.circle(image, rc_point, 3, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(
                image,
                sift_point,
                (255, 80, 230),
                cv2.MARKER_TILTED_CROSS,
                7,
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(image, (0, 0), (image.shape[1], 54), (25, 25, 25), -1)
        label = (
            f"{category.upper()} | {data.image_name} | "
            f"matched {len(data.matches)}/{data.observation_count}"
        )
        cv2.putText(
            image,
            label,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        safe_stem = "".join(
            character if character.isalnum() else "_"
            for character in Path(data.image_name).stem
        )
        output_path = qc_dir / f"{category}_image_{data.image_id}_{safe_stem}.jpg"
        if cv2.imwrite(str(output_path), image):
            paths.append(output_path)
    return paths


def _generate_spatial_qc(
    cv2,
    np,
    xyz_by_point: Dict[int, Tuple[float, float, float]],
    points_with_descriptor: Set[int],
    output_path: Path,
) -> Path:
    point_ids = list(xyz_by_point)
    xyz = np.asarray([xyz_by_point[point_id] for point_id in point_ids], dtype=np.float64)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    yaw, pitch = math.radians(-35), math.radians(24)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    rx = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float64)
    # OpenCV may leave floating-point status flags set after a long SIFT run on
    # some macOS builds. Ignore those stale flags here, then explicitly verify
    # the projection result instead of emitting misleading NumPy warnings.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        viewed = (xyz - center) @ (rx @ rz).T
    if not np.isfinite(viewed).all():
        raise FeatureMapError("Spatial QC projection contains non-finite values")
    width, height = 1280, 820
    extent = np.ptp(viewed[:, :2], axis=0)
    scale = min((width - 100) / max(extent[0], 1e-9), (height - 150) / max(extent[1], 1e-9))
    projected = np.empty((len(viewed), 2), dtype=np.int32)
    projected[:, 0] = np.rint(width * 0.5 + viewed[:, 0] * scale).astype(np.int32)
    projected[:, 1] = np.rint(height * 0.54 - viewed[:, 1] * scale).astype(np.int32)
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    matched_mask = np.asarray(
        [point_id in points_with_descriptor for point_id in point_ids], dtype=bool
    )
    for u, v in projected[~matched_mask]:
        if 0 <= u < width and 0 <= v < height:
            image[v, u] = (65, 65, 65)
    for u, v in projected[matched_mask]:
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(image, (int(u), int(v)), 1, (50, 230, 80), -1)
    cv2.putText(
        image,
        "3D feature distribution: descriptor points (green), no descriptor (gray)",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{len(points_with_descriptor):,} / {len(point_ids):,} unique 3D points",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise FeatureMapError(f"OpenCV could not write spatial QC: {output_path}")
    return output_path


def _summary_text(
    scan: ObservationScan,
    processed_images: int,
    failed_images: int,
    descriptor_point_counts: Counter[int],
    descriptor_count: int,
    distances: Sequence[float],
    image_errors: Counter[str],
    observation_errors: Counter[str],
    max_distance: float,
) -> str:
    per_point = list(descriptor_point_counts.values())
    acquisition_rate = descriptor_count / scan.total_observations
    mean_per_point = statistics.fmean(per_point) if per_point else 0.0
    median_per_point = statistics.median(per_point) if per_point else 0.0
    max_per_point = max(per_point, default=0)
    mean_distance = statistics.fmean(distances) if distances else 0.0
    max_pixel_distance = max(distances, default=0.0)
    points_with = len(descriptor_point_counts)
    lines = [
        f"Total images: {scan.total_images}",
        f"Processed images: {processed_images}",
        f"Failed images: {failed_images}",
        "",
        f"Object observations: {scan.total_observations}",
        f"Observations with SIFT descriptor: {descriptor_count}",
        f"Observations without descriptor: {scan.total_observations - descriptor_count}",
        f"Descriptor acquisition rate: {acquisition_rate:.6%}",
        "",
        f"Unique POINT3D_IDs: {len(scan.xyz_by_point)}",
        f"POINT3D_IDs with >=1 descriptor: {points_with}",
        f"POINT3D_IDs without descriptor: {len(scan.xyz_by_point) - points_with}",
        "",
        f"Mean descriptors / 3D point: {mean_per_point:.6f}",
        f"Median descriptors / 3D point: {median_per_point:.6f}",
        f"Max descriptors / 3D point: {max_per_point}",
        "",
        f"Mean RC-SIFT pixel distance: {mean_distance:.6f}",
        f"Max RC-SIFT pixel distance: {max_pixel_distance:.6f}",
        f"Configured maximum pixel distance: {max_distance:.6f}",
        "",
        "Image errors:",
    ]
    lines.extend(
        f"{code}: {count}" for code, count in sorted(image_errors.items())
    )
    if not image_errors:
        lines.append("None")
    lines.extend(["", "Observation errors:"])
    lines.extend(
        f"{code}: {count}" for code, count in sorted(observation_errors.items())
    )
    if not observation_errors:
        lines.append("None")
    return "\n".join(lines) + "\n"


def _verify_outputs(
    np,
    features_path: Path,
    metadata_path: Path,
    scan: ObservationScan,
    expected_count: int,
    max_distance: float,
) -> None:
    expected_shapes = {
        "point3d_ids": (expected_count,),
        "xyz": (expected_count, 3),
        "descriptors": (expected_count, 128),
        "image_ids": (expected_count,),
        "rc_uv": (expected_count, 2),
        "sift_uv": (expected_count, 2),
        "pixel_distances": (expected_count,),
        "scales": (expected_count,),
        "angles": (expected_count,),
        "responses": (expected_count,),
        "point_observation_counts": (expected_count,),
    }
    try:
        with np.load(features_path, allow_pickle=False) as feature_map:
            if set(feature_map.files) != set(expected_shapes):
                raise FeatureMapError("QC failed: object_features.npz array names differ")
            for name, shape in expected_shapes.items():
                if feature_map[name].shape != shape:
                    raise FeatureMapError(
                        f"QC failed: {name} has shape {feature_map[name].shape}, expected {shape}"
                    )
            if expected_count:
                if not np.isfinite(feature_map["descriptors"]).all():
                    raise FeatureMapError("QC failed: descriptors contain non-finite values")
                if not np.isfinite(feature_map["pixel_distances"]).all():
                    raise FeatureMapError("QC failed: pixel distances contain non-finite values")
                if np.any(feature_map["pixel_distances"] > max_distance + 1e-5):
                    raise FeatureMapError("QC failed: a match exceeds --max-distance")
                for point_id, xyz in zip(
                    feature_map["point3d_ids"].tolist(), feature_map["xyz"]
                ):
                    if point_id not in scan.xyz_by_point or tuple(xyz) != scan.xyz_by_point[point_id]:
                        raise FeatureMapError(
                            f"QC failed: POINT3D_ID {point_id} has invalid XYZ"
                        )
    except (OSError, ValueError) as exc:
        raise FeatureMapError(f"Cannot verify object_features.npz: {exc}") from exc

    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != METADATA_COLUMNS:
                raise FeatureMapError("QC failed: feature metadata columns differ")
            metadata_count = sum(1 for _ in reader)
    except (OSError, UnicodeError) as exc:
        raise FeatureMapError(f"Cannot verify feature metadata: {exc}") from exc
    if metadata_count != expected_count:
        raise FeatureMapError(
            f"QC failed: NPZ has {expected_count} records but metadata has {metadata_count}"
        )


def build_feature_map(
    observations_path: Path,
    cameras_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    max_distance: float = 3.0,
    sift_nfeatures: int = 0,
    image_template: Optional[str] = None,
    image_index_offset: int = 0,
    create_visualizations: bool = True,
    report_progress: bool = False,
) -> FeatureMapResult:
    if max_distance <= 0 or not math.isfinite(max_distance):
        raise FeatureMapError("--max-distance must be a positive finite number")
    if sift_nfeatures < 0:
        raise FeatureMapError("--sift-nfeatures must be zero or greater")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise FeatureMapError(
            "Step 3 requires OpenCV and NumPy. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    if not hasattr(cv2, "SIFT_create"):
        raise FeatureMapError("This OpenCV build does not provide SIFT_create()")

    try:
        cameras = load_cameras(cameras_path)
    except ObservationError as exc:
        raise FeatureMapError(str(exc)) from exc
    scan = scan_observations(observations_path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FeatureMapError(f"Cannot create output directory {output_dir}: {exc}") from exc

    final_features = output_dir / "object_features.npz"
    final_metadata = output_dir / "object_features_metadata.csv"
    final_summary = output_dir / "feature_summary.txt"
    final_failures = output_dir / "feature_failures.csv"
    final_unmatched = output_dir / "feature_unmatched_observations.csv"
    temporary_paths = {
        "features": _temporary_path(output_dir, ".features.tmp"),
        "metadata": _temporary_path(output_dir, ".metadata.tmp"),
        "summary": _temporary_path(output_dir, ".summary.tmp"),
        "failures": _temporary_path(output_dir, ".failures.tmp"),
        "unmatched": _temporary_path(output_dir, ".unmatched.tmp"),
    }

    point3d_ids: List[int] = []
    xyz_values: List[Tuple[float, float, float]] = []
    descriptor_values: List[object] = []
    image_ids: List[int] = []
    rc_uv: List[Tuple[float, float]] = []
    sift_uv: List[Tuple[float, float]] = []
    pixel_distances: List[float] = []
    scales: List[float] = []
    angles: List[float] = []
    responses: List[float] = []
    observation_counts: List[int] = []
    descriptor_point_counts: Counter[int] = Counter()
    image_errors: Counter[str] = Counter()
    observation_errors: Counter[str] = Counter()
    feature_counts: List[ImageFeatureCount] = []
    qc_data: Dict[int, ImageQCData] = {}
    processed_images = 0
    failed_images = 0

    sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    grayscale_flags = cv2.IMREAD_GRAYSCALE | getattr(
        cv2, "IMREAD_IGNORE_ORIENTATION", 0
    )

    try:
        with (
            temporary_paths["metadata"].open("w", encoding="utf-8", newline="") as metadata_file,
            temporary_paths["failures"].open("w", encoding="utf-8", newline="") as failures_file,
            temporary_paths["unmatched"].open("w", encoding="utf-8", newline="") as unmatched_file,
        ):
            metadata_writer = csv.writer(metadata_file, lineterminator="\n")
            failures_writer = csv.writer(failures_file, lineterminator="\n")
            unmatched_writer = csv.writer(unmatched_file, lineterminator="\n")
            metadata_writer.writerow(METADATA_COLUMNS)
            failures_writer.writerow(FAILURE_COLUMNS)
            unmatched_writer.writerow(UNMATCHED_COLUMNS)

            def fail_image(
                group: Sequence[Observation], code: str, detail: str
            ) -> None:
                nonlocal failed_images
                first = group[0]
                failed_images += 1
                image_errors[code] += 1
                failures_writer.writerow(
                    (
                        first.image_id,
                        first.image_name,
                        first.camera_id,
                        len(group),
                        code,
                        detail,
                    )
                )
                feature_counts.append(
                    ImageFeatureCount(
                        first.image_id, first.image_name, len(group), 0
                    )
                )

            for group_index, group in enumerate(
                iter_observation_groups(observations_path), 1
            ):
                first = group[0]
                camera = cameras.get(first.camera_id)
                if camera is None:
                    fail_image(group, "CAMERA_NOT_FOUND", f"CAMERA_ID {first.camera_id}")
                    continue
                source_path = resolve_source_image(
                    first.image_name,
                    image_dir,
                    image_template,
                    image_index_offset,
                )
                if source_path is None:
                    fail_image(
                        group,
                        "IMAGE_NOT_FOUND",
                        f"No exact source image found under {image_dir}",
                    )
                    continue
                image = cv2.imread(str(source_path), grayscale_flags)
                if image is None:
                    fail_image(group, "IMAGE_READ_FAILED", str(source_path))
                    continue
                actual_height, actual_width = image.shape[:2]
                if (actual_width, actual_height) != (camera.width, camera.height):
                    fail_image(
                        group,
                        "IMAGE_SIZE_MISMATCH",
                        (
                            f"COLMAP={camera.width}x{camera.height}, "
                            f"file={actual_width}x{actual_height}, path={source_path}"
                        ),
                    )
                    continue

                valid_group: List[Observation] = []
                image_unmatched: List[Tuple[float, float]] = []
                for observation in group:
                    if not (
                        0 <= observation.u < camera.width
                        and 0 <= observation.v < camera.height
                    ):
                        code = "RC_COORDINATE_OUT_OF_BOUNDS"
                        observation_errors[code] += 1
                        unmatched_writer.writerow(
                            (
                                observation.point3d_id,
                                observation.image_id,
                                observation.image_name,
                                observation.u_text,
                                observation.v_text,
                                code,
                            )
                        )
                        image_unmatched.append((observation.u, observation.v))
                    else:
                        valid_group.append(observation)

                keypoints, descriptors = sift.detectAndCompute(image, None)
                if not keypoints:
                    fail_image(group, "NO_SIFT_FEATURES", "SIFT detected zero keypoints")
                    continue
                if descriptors is None or descriptors.shape != (len(keypoints), 128):
                    shape = None if descriptors is None else descriptors.shape
                    fail_image(
                        group,
                        "DESCRIPTOR_FAILED",
                        f"Expected ({len(keypoints)}, 128), got {shape}",
                    )
                    continue

                matches, unmatched = match_observations_to_keypoints(
                    valid_group, keypoints, max_distance
                )
                processed_images += 1
                image_matches: List[Tuple[float, float, float, float, float]] = []
                for match in matches:
                    observation = valid_group[match.observation_index]
                    keypoint = keypoints[match.keypoint_index]
                    sift_u, sift_v = keypoint.pt
                    point3d_ids.append(observation.point3d_id)
                    xyz_values.append(observation.xyz)
                    descriptor_values.append(descriptors[match.keypoint_index].copy())
                    image_ids.append(observation.image_id)
                    rc_uv.append((observation.u, observation.v))
                    sift_uv.append((sift_u, sift_v))
                    pixel_distances.append(match.distance)
                    scales.append(keypoint.size)
                    angles.append(keypoint.angle)
                    responses.append(keypoint.response)
                    point_count = scan.point_observation_counts[observation.point3d_id]
                    observation_counts.append(point_count)
                    descriptor_point_counts[observation.point3d_id] += 1
                    metadata_writer.writerow(
                        (
                            observation.point3d_id,
                            observation.image_id,
                            observation.image_name,
                            *observation.xyz_text,
                            observation.u_text,
                            observation.v_text,
                            f"{sift_u:.9g}",
                            f"{sift_v:.9g}",
                            f"{match.distance:.9g}",
                            f"{keypoint.size:.9g}",
                            f"{keypoint.angle:.9g}",
                            f"{keypoint.response:.9g}",
                            point_count,
                        )
                    )
                    image_matches.append(
                        (observation.u, observation.v, sift_u, sift_v, match.distance)
                    )
                for valid_index, code in unmatched.items():
                    observation = valid_group[valid_index]
                    observation_errors[code] += 1
                    unmatched_writer.writerow(
                        (
                            observation.point3d_id,
                            observation.image_id,
                            observation.image_name,
                            observation.u_text,
                            observation.v_text,
                            code,
                        )
                    )
                    image_unmatched.append((observation.u, observation.v))
                feature_counts.append(
                    ImageFeatureCount(
                        first.image_id, first.image_name, len(group), len(matches)
                    )
                )
                if create_visualizations:
                    qc_data[first.image_id] = ImageQCData(
                        first.image_id,
                        first.image_name,
                        source_path,
                        len(group),
                        image_matches,
                        image_unmatched,
                    )
                if report_progress and (
                    group_index == 1
                    or group_index % 25 == 0
                    or group_index == scan.total_images
                ):
                    print(
                        f"Progress: {group_index}/{scan.total_images} images, "
                        f"{len(point3d_ids)} descriptors",
                        file=sys.stderr,
                        flush=True,
                    )

        descriptor_array = (
            np.asarray(descriptor_values, dtype=np.float32).reshape((-1, 128))
            if descriptor_values
            else np.empty((0, 128), dtype=np.float32)
        )
        with temporary_paths["features"].open("wb") as feature_file:
            np.savez_compressed(
                feature_file,
                point3d_ids=np.asarray(point3d_ids, dtype=np.int64),
                xyz=np.asarray(xyz_values, dtype=np.float64).reshape((-1, 3)),
                descriptors=descriptor_array,
                image_ids=np.asarray(image_ids, dtype=np.int64),
                rc_uv=np.asarray(rc_uv, dtype=np.float64).reshape((-1, 2)),
                sift_uv=np.asarray(sift_uv, dtype=np.float32).reshape((-1, 2)),
                pixel_distances=np.asarray(pixel_distances, dtype=np.float32),
                scales=np.asarray(scales, dtype=np.float32),
                angles=np.asarray(angles, dtype=np.float32),
                responses=np.asarray(responses, dtype=np.float32),
                point_observation_counts=np.asarray(observation_counts, dtype=np.int32),
            )

        with temporary_paths["summary"].open("w", encoding="utf-8") as summary_file:
            summary_file.write(
                _summary_text(
                    scan,
                    processed_images,
                    failed_images,
                    descriptor_point_counts,
                    len(point3d_ids),
                    pixel_distances,
                    image_errors,
                    observation_errors,
                    max_distance,
                )
            )

        _verify_outputs(
            np,
            temporary_paths["features"],
            temporary_paths["metadata"],
            scan,
            len(point3d_ids),
            max_distance,
        )

        replacements = (
            (temporary_paths["features"], final_features),
            (temporary_paths["metadata"], final_metadata),
            (temporary_paths["summary"], final_summary),
            (temporary_paths["failures"], final_failures),
            (temporary_paths["unmatched"], final_unmatched),
        )
        for temporary, final in replacements:
            os.replace(temporary, final)

        spatial_qc_path: Optional[Path] = None
        overlay_paths: Sequence[Path] = []
        if create_visualizations:
            overlay_paths = _generate_match_overlays(
                cv2, qc_data, feature_counts, output_dir / "feature_qc"
            )
            spatial_qc_path = _generate_spatial_qc(
                cv2,
                np,
                scan.xyz_by_point,
                set(descriptor_point_counts),
                output_dir / "feature_spatial_distribution.png",
            )

        return FeatureMapResult(
            scan.total_images,
            processed_images,
            failed_images,
            scan.total_observations,
            len(point3d_ids),
            len(scan.xyz_by_point),
            len(descriptor_point_counts),
            final_features,
            final_metadata,
            final_summary,
            final_failures,
            final_unmatched,
            spatial_qc_path,
            overlay_paths,
        )
    except (OSError, UnicodeError) as exc:
        raise FeatureMapError(f"File operation failed: {exc}") from exc
    finally:
        for path in temporary_paths.values():
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an object-only SIFT 3D feature map from Step 2 output."
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum RC-to-SIFT pixel distance (default: 3.0)",
    )
    parser.add_argument(
        "--sift-nfeatures",
        type=int,
        default=0,
        help="Maximum retained SIFT features per image; 0 means OpenCV default/unlimited",
    )
    parser.add_argument(
        "--image-template",
        help=(
            "Explicit renamed-image pattern, e.g. "
            "'output_image2_{index:04d}.jpg'; supports {index}, {stem}, {name}"
        ),
    )
    parser.add_argument(
        "--image-index-offset",
        type=int,
        default=0,
        help="Offset added to numeric image stem for --image-template",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip 2D match overlays and 3D spatial distribution PNG",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = build_feature_map(
            args.observations,
            args.cameras,
            args.image_dir,
            args.output_dir,
            max_distance=args.max_distance,
            sift_nfeatures=args.sift_nfeatures,
            image_template=args.image_template,
            image_index_offset=args.image_index_offset,
            create_visualizations=not args.no_visualization,
            report_progress=True,
        )
        print(f"Total images: {result.total_images}")
        print(f"Processed images: {result.processed_images}")
        print(f"Failed images: {result.failed_images}")
        print(f"Object observations: {result.total_observations}")
        print(
            "Observations with SIFT descriptor: "
            f"{result.observations_with_descriptor}"
        )
        print(f"3D points with descriptor: {result.points_with_descriptor}")
        print(f"Output: {result.features_path}")
        print(f"Metadata: {result.metadata_path}")
        print(f"Summary: {result.summary_path}")
        if result.observations_with_descriptor == 0:
            print(
                "Error: No descriptors were generated; check feature_failures.csv",
                file=sys.stderr,
            )
            return 2
        return 0
    except FeatureMapError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
