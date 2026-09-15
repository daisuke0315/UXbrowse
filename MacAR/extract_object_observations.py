#!/usr/bin/env python3
"""Build an object-only COLMAP 2D-3D observation table."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, TextIO, Tuple

from object_point_filter import parse_point_line


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = SCRIPT_DIR / "colmap" / "Images.txt"
DEFAULT_CAMERAS = SCRIPT_DIR / "colmap" / "Cameras.txt"
DEFAULT_STEP1_DIR = SCRIPT_DIR / "output"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "image"

CSV_FIELDS = (
    "image_id",
    "image_name",
    "camera_id",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "u",
    "v",
    "point3d_id",
    "X",
    "Y",
    "Z",
)


class ObservationError(RuntimeError):
    """Expected input, consistency, or output error."""


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int


@dataclass(frozen=True)
class PointXYZ:
    xyz: Tuple[float, float, float]
    xyz_text: Tuple[str, str, str]


@dataclass(frozen=True)
class ImageHeader:
    image_id: int
    camera_id: int
    name: str
    pose: Tuple[float, float, float, float, float, float, float]
    pose_text: Tuple[str, str, str, str, str, str, str]


@dataclass(frozen=True)
class ImageCount:
    image_id: int
    image_name: str
    camera_id: int
    observation_count: int


@dataclass(frozen=True)
class OverlayStatus:
    category: str
    image_id: int
    image_name: str
    observation_count: int
    status: str
    output_path: Optional[Path]


@dataclass(frozen=True)
class ObservationResult:
    selected_points: int
    total_images: int
    images_with_observations: int
    total_observations: int
    points_with_observations: int
    points_without_observations: int
    out_of_bounds_observations: int
    minimum_per_image: int
    maximum_per_image: int
    mean_per_image: float
    observations_path: Path
    out_of_bounds_path: Path
    counts_path: Path
    summary_path: Path
    overlays: Sequence[OverlayStatus]


def _finite_float(text: str, label: str, line_number: int) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ObservationError(
            f"Line {line_number}: {label} is not a number: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise ObservationError(f"Line {line_number}: {label} must be finite")
    return value


def load_cameras(path: Path) -> Dict[int, Camera]:
    if not path.is_file():
        raise ObservationError(f"Cameras.txt not found: {path}")
    cameras: Dict[int, Camera] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Some RC export bundles prepend this companion-file note to
                # Cameras.txt without a COLMAP comment marker. It is metadata,
                # not a camera record; all other malformed lines remain errors.
                if stripped.startswith(
                    "The files Cameras.txt, Images.txt and Points3D.txt are in the same folder."
                ):
                    continue
                fields = stripped.split()
                if len(fields) < 5:
                    raise ObservationError(
                        f"Cameras.txt line {line_number}: expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS"
                    )
                try:
                    camera_id = int(fields[0])
                    width = int(fields[2])
                    height = int(fields[3])
                except ValueError as exc:
                    raise ObservationError(
                        f"Cameras.txt line {line_number}: invalid ID or image dimensions"
                    ) from exc
                if camera_id <= 0 or width <= 0 or height <= 0:
                    raise ObservationError(
                        f"Cameras.txt line {line_number}: ID, WIDTH and HEIGHT must be positive"
                    )
                if camera_id in cameras:
                    raise ObservationError(f"Duplicate CAMERA_ID: {camera_id}")
                for parameter in fields[4:]:
                    _finite_float(parameter, "camera parameter", line_number)
                cameras[camera_id] = Camera(camera_id, fields[1], width, height)
    except (OSError, UnicodeError) as exc:
        raise ObservationError(f"Cannot read Cameras.txt: {path}: {exc}") from exc
    if not cameras:
        raise ObservationError("Cameras.txt contains no camera records")
    return cameras


def load_selected_ids(path: Path) -> Set[int]:
    if not path.is_file():
        raise ObservationError(f"selected_point_ids.txt not found: {path}")
    selected: Set[int] = set()
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    point_id = int(stripped)
                except ValueError as exc:
                    raise ObservationError(
                        f"selected_point_ids.txt line {line_number}: invalid POINT3D_ID"
                    ) from exc
                if point_id < 0:
                    raise ObservationError(
                        f"selected_point_ids.txt line {line_number}: POINT3D_ID must be non-negative"
                    )
                if point_id in selected:
                    raise ObservationError(f"Duplicate selected POINT3D_ID: {point_id}")
                selected.add(point_id)
    except (OSError, UnicodeError) as exc:
        raise ObservationError(f"Cannot read selected_point_ids.txt: {path}: {exc}") from exc
    if not selected:
        raise ObservationError("selected_point_ids.txt contains no IDs")
    return selected


def load_filtered_points(path: Path, selected_ids: Set[int]) -> Dict[int, PointXYZ]:
    if not path.is_file():
        raise ObservationError(f"filtered_Points3D.txt not found: {path}")
    points: Dict[int, PointXYZ] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    record = parse_point_line(line, line_number)
                except RuntimeError as exc:
                    raise ObservationError(f"filtered_Points3D.txt: {exc}") from exc
                if record.point_id in points:
                    raise ObservationError(
                        f"Duplicate POINT3D_ID in filtered_Points3D.txt: {record.point_id}"
                    )
                fields = line.split()
                points[record.point_id] = PointXYZ(record.xyz, tuple(fields[1:4]))  # type: ignore[arg-type]
    except (OSError, UnicodeError) as exc:
        raise ObservationError(f"Cannot read filtered_Points3D.txt: {path}: {exc}") from exc
    point_ids = set(points)
    missing = selected_ids - point_ids
    unexpected = point_ids - selected_ids
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"{len(missing)} selected IDs are missing from filtered_Points3D.txt")
        if unexpected:
            details.append(f"{len(unexpected)} filtered points are absent from selected IDs")
        raise ObservationError("Step 1 outputs are inconsistent: " + "; ".join(details))
    return points


def parse_image_header(line: str, line_number: int) -> ImageHeader:
    fields = line.split(maxsplit=9)
    if len(fields) != 10:
        raise ObservationError(
            f"Images.txt line {line_number}: expected 10 image header fields"
        )
    try:
        image_id = int(fields[0])
        camera_id = int(fields[8])
    except ValueError as exc:
        raise ObservationError(
            f"Images.txt line {line_number}: IMAGE_ID and CAMERA_ID must be integers"
        ) from exc
    if image_id <= 0 or camera_id <= 0 or not fields[9].strip():
        raise ObservationError(
            f"Images.txt line {line_number}: invalid image ID, camera ID, or image name"
        )
    pose_text = tuple(fields[1:8])
    pose = tuple(
        _finite_float(value, "camera pose", line_number) for value in pose_text
    )
    quaternion_norm = math.sqrt(sum(value * value for value in pose[:4]))
    if quaternion_norm <= 1e-12:
        raise ObservationError(f"Images.txt line {line_number}: zero camera quaternion")
    return ImageHeader(
        image_id,
        camera_id,
        fields[9].strip(),
        pose,  # type: ignore[arg-type]
        pose_text,  # type: ignore[arg-type]
    )


def iter_images(path: Path) -> Iterator[Tuple[ImageHeader, List[str], int]]:
    if not path.is_file():
        raise ObservationError(f"Images.txt not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            line_number = 0
            while True:
                header_line = source.readline()
                if header_line == "":
                    break
                line_number += 1
                stripped = header_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                header = parse_image_header(header_line, line_number)
                points_line = source.readline()
                if points_line == "":
                    raise ObservationError(
                        f"Images.txt ended after image header at line {line_number}"
                    )
                line_number += 1
                if points_line.lstrip().startswith("#"):
                    raise ObservationError(
                        f"Images.txt line {line_number}: expected POINTS2D data, found comment"
                    )
                fields = points_line.split()
                if len(fields) % 3 != 0:
                    raise ObservationError(
                        f"Images.txt line {line_number}: POINTS2D must contain (X Y POINT3D_ID) triples"
                    )
                yield header, fields, line_number
    except (OSError, UnicodeError) as exc:
        raise ObservationError(f"Cannot read Images.txt: {path}: {exc}") from exc


def iter_points2d(fields: Sequence[str], line_number: int) -> Iterator[Tuple[str, str, float, float, int]]:
    for index in range(0, len(fields), 3):
        u_text, v_text, point_id_text = fields[index : index + 3]
        u = _finite_float(u_text, "u", line_number)
        v = _finite_float(v_text, "v", line_number)
        try:
            point_id = int(point_id_text)
        except ValueError as exc:
            raise ObservationError(
                f"Images.txt line {line_number}: POINT3D_ID must be an integer"
            ) from exc
        yield u_text, v_text, u, v, point_id


def _temporary_csv(output_dir: Path, suffix: str) -> Tuple[TextIO, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_dir,
        suffix=suffix,
        delete=False,
    )
    return handle, Path(handle.name)


def choose_qc_images(counts: Sequence[ImageCount]) -> List[Tuple[str, ImageCount]]:
    positive = sorted(
        (item for item in counts if item.observation_count > 0),
        key=lambda item: (item.observation_count, item.image_id),
    )
    if not positive:
        return []
    candidates = [
        ("low", positive[0]),
        ("medium", positive[len(positive) // 2]),
        ("high", positive[-1]),
    ]
    result: List[Tuple[str, ImageCount]] = []
    used: Set[int] = set()
    for category, item in candidates:
        if item.image_id not in used:
            result.append((category, item))
            used.add(item.image_id)
    return result


def collect_qc_observations(
    images_path: Path,
    target_ids: Set[int],
    selected_ids: Set[int],
) -> Dict[int, List[Tuple[float, float, int]]]:
    observations = {image_id: [] for image_id in target_ids}
    for header, fields, line_number in iter_images(images_path):
        if header.image_id not in target_ids:
            continue
        for _u_text, _v_text, u, v, point_id in iter_points2d(fields, line_number):
            if point_id >= 0 and point_id in selected_ids:
                observations[header.image_id].append((u, v, point_id))
    return observations


def resolve_image_path(
    image_name: str,
    image_dir: Path,
    image_template: Optional[str],
    image_index_offset: int,
) -> Optional[Path]:
    exact = image_dir / image_name
    if exact.is_file():
        return exact
    stem = Path(image_name).stem
    parent = image_dir / Path(image_name).parent
    for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = parent / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    if image_template is not None:
        try:
            index = int(stem) + image_index_offset
            relative = image_template.format(index=index, stem=stem, name=image_name)
        except (ValueError, KeyError, IndexError) as exc:
            raise ObservationError(f"Cannot apply --image-template to {image_name!r}: {exc}") from exc
        candidate = image_dir / relative
        if candidate.is_file():
            return candidate
    return None


def generate_qc_overlays(
    images_path: Path,
    selected_ids: Set[int],
    cameras: Dict[int, Camera],
    counts: Sequence[ImageCount],
    headers: Dict[int, ImageHeader],
    image_dir: Path,
    qc_dir: Path,
    *,
    image_template: Optional[str],
    image_index_offset: int,
    allow_rescaled_images: bool,
) -> List[OverlayStatus]:
    targets = choose_qc_images(counts)
    if not targets:
        return []
    observations = collect_qc_observations(
        images_path, {item.image_id for _, item in targets}, selected_ids
    )
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        return [
            OverlayStatus(category, item.image_id, item.image_name, item.observation_count,
                          "OpenCV is not installed", None)
            for category, item in targets
        ]

    qc_dir.mkdir(parents=True, exist_ok=True)
    results: List[OverlayStatus] = []
    for category, item in targets:
        image_path = resolve_image_path(
            item.image_name, image_dir, image_template, image_index_offset
        )
        if image_path is None:
            results.append(
                OverlayStatus(
                    category,
                    item.image_id,
                    item.image_name,
                    item.observation_count,
                    f"source image not found under {image_dir}",
                    None,
                )
            )
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            results.append(
                OverlayStatus(category, item.image_id, item.image_name, item.observation_count,
                              f"OpenCV could not read {image_path}", None)
            )
            continue
        camera = cameras[item.camera_id]
        actual_height, actual_width = image.shape[:2]
        dimensions_match = (actual_width, actual_height) == (camera.width, camera.height)
        if not dimensions_match and not allow_rescaled_images:
            results.append(
                OverlayStatus(
                    category,
                    item.image_id,
                    item.image_name,
                    item.observation_count,
                    (
                        f"dimension mismatch: COLMAP {camera.width}x{camera.height}, "
                        f"file {actual_width}x{actual_height}; use exact exported images"
                    ),
                    None,
                )
            )
            continue

        scale_x = actual_width / camera.width
        scale_y = actual_height / camera.height
        for u, v, _point_id in observations[item.image_id]:
            position = (int(round(u * scale_x)), int(round(v * scale_y)))
            cv2.circle(image, position, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, position, 2, (0, 255, 80), -1, cv2.LINE_AA)

        approximate = not dimensions_match
        banner_color = (0, 0, 190) if approximate else (30, 30, 30)
        cv2.rectangle(image, (0, 0), (actual_width, 58), banner_color, -1)
        label = (
            f"{category.upper()} | image {item.image_id} | {item.image_name} | "
            f"{item.observation_count} object observations"
        )
        cv2.putText(image, label, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if approximate:
            cv2.putText(
                image,
                "APPROXIMATE: coordinates linearly scaled to a different-size image",
                (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        safe_stem = "".join(character if character.isalnum() else "_" for character in Path(item.image_name).stem)
        output_path = qc_dir / f"{category}_image_{item.image_id}_{safe_stem}.jpg"
        if not cv2.imwrite(str(output_path), image):
            results.append(
                OverlayStatus(category, item.image_id, item.image_name, item.observation_count,
                              f"OpenCV could not write {output_path}", None)
            )
            continue
        status = "PASS: exact image dimensions" if dimensions_match else "APPROXIMATE: rescaled image"
        results.append(
            OverlayStatus(category, item.image_id, item.image_name, item.observation_count,
                          status, output_path)
        )
    return results


def verify_observation_csv(
    path: Path,
    selected_ids: Set[int],
    points: Dict[int, PointXYZ],
    cameras: Dict[int, Camera],
    headers: Dict[int, ImageHeader],
    expected_count: int,
    expected_out_of_bounds: int,
) -> None:
    count = 0
    out_of_bounds = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ObservationError("QC failed: object_observations.csv columns are invalid")
            for row_number, row in enumerate(reader, 2):
                try:
                    image_id = int(row["image_id"])
                    camera_id = int(row["camera_id"])
                    point_id = int(row["point3d_id"])
                    u, v = float(row["u"]), float(row["v"])
                    xyz = tuple(float(row[axis]) for axis in ("X", "Y", "Z"))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ObservationError(f"QC failed at CSV row {row_number}: invalid value") from exc
                if point_id not in selected_ids or point_id not in points:
                    raise ObservationError(f"QC1 failed at CSV row {row_number}: invalid POINT3D_ID")
                if xyz != points[point_id].xyz:
                    raise ObservationError(f"QC2 failed at CSV row {row_number}: XYZ mismatch")
                if image_id not in headers or camera_id != headers[image_id].camera_id:
                    raise ObservationError(f"QC failed at CSV row {row_number}: image/camera mismatch")
                camera = cameras[camera_id]
                if not (0 <= u < camera.width and 0 <= v < camera.height):
                    out_of_bounds += 1
                count += 1
    except (OSError, UnicodeError) as exc:
        raise ObservationError(f"Cannot verify object_observations.csv: {exc}") from exc
    if count != expected_count:
        raise ObservationError(
            f"QC failed: expected {expected_count} observations, found {count}"
        )
    if out_of_bounds != expected_out_of_bounds:
        raise ObservationError(
            f"QC failed: expected {expected_out_of_bounds} out-of-bounds observations, "
            f"found {out_of_bounds}"
        )


def _summary_text(
    selected_count: int,
    counts: Sequence[ImageCount],
    observed_ids: Set[int],
    total_observations: int,
    out_of_bounds_observations: int,
    out_of_bounds_images: int,
    overlays: Sequence[OverlayStatus],
) -> str:
    per_image = [item.observation_count for item in counts]
    images_with = sum(value > 0 for value in per_image)
    minimum = min(per_image) if per_image else 0
    maximum = max(per_image) if per_image else 0
    mean = statistics.fmean(per_image) if per_image else 0.0
    lines = [
        f"Selected 3D points: {selected_count}",
        f"Total images: {len(counts)}",
        f"Images containing selected points: {images_with}",
        f"Total object observations: {total_observations}",
        "",
        f"Minimum observations / image: {minimum}",
        f"Maximum observations / image: {maximum}",
        f"Mean observations / image: {mean:.6f}",
        "",
        f"3D points with observations: {len(observed_ids)}",
        f"3D points without observations: {selected_count - len(observed_ids)}",
        "",
        "QC:",
        "QC1 all observation IDs are selected: PASS",
        "QC2 POINT3D_ID to XYZ consistency: PASS",
        (
            "QC3 all selected (u,v) inside camera dimensions: PASS"
            if out_of_bounds_observations == 0
            else (
                "QC3 all selected (u,v) inside camera dimensions: "
                f"FAIL ({out_of_bounds_observations} observations in "
                f"{out_of_bounds_images} images are outside bounds)"
            )
        ),
        "QC4 per-image observation counts recorded: PASS",
        "",
        "Visualization QC:",
    ]
    if overlays:
        for overlay in overlays:
            location = f" -> {overlay.output_path}" if overlay.output_path else ""
            lines.append(
                f"{overlay.category}: image_id={overlay.image_id}, "
                f"image_name={overlay.image_name}, observations={overlay.observation_count}, "
                f"status={overlay.status}{location}"
            )
    else:
        lines.append("Not generated")
    lines.extend(["", "Per-image observation counts (image_name, observation_count):"])
    lines.extend(f"{item.image_name}, {item.observation_count}" for item in counts)
    return "\n".join(lines) + "\n"


def extract_observations(
    images_path: Path,
    cameras_path: Path,
    selected_ids_path: Path,
    filtered_points_path: Path,
    output_dir: Path,
    *,
    create_visualizations: bool = True,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    image_template: Optional[str] = None,
    image_index_offset: int = 0,
    allow_rescaled_images: bool = False,
) -> ObservationResult:
    cameras = load_cameras(cameras_path)
    selected_ids = load_selected_ids(selected_ids_path)
    points = load_filtered_points(filtered_points_path, selected_ids)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ObservationError(f"Cannot create output directory {output_dir}: {exc}") from exc

    csv_handle, csv_temp = _temporary_csv(output_dir, ".observations.tmp")
    invalid_handle, invalid_temp = _temporary_csv(output_dir, ".out-of-bounds.tmp")
    counts_handle, counts_temp = _temporary_csv(output_dir, ".counts.tmp")
    summary_handle: Optional[TextIO] = None
    summary_temp: Optional[Path] = None
    observations_path = output_dir / "object_observations.csv"
    out_of_bounds_path = output_dir / "out_of_bounds_observations.csv"
    counts_path = output_dir / "image_observation_counts.csv"
    summary_path = output_dir / "observation_summary.txt"
    image_counts: List[ImageCount] = []
    headers: Dict[int, ImageHeader] = {}
    observed_ids: Set[int] = set()
    out_of_bounds_image_ids: Set[int] = set()
    total_observations = 0
    out_of_bounds_observations = 0

    try:
        with csv_handle, invalid_handle:
            writer = csv.writer(csv_handle, lineterminator="\n")
            invalid_writer = csv.writer(invalid_handle, lineterminator="\n")
            writer.writerow(CSV_FIELDS)
            invalid_writer.writerow(CSV_FIELDS)
            for header, fields, line_number in iter_images(images_path):
                if header.image_id in headers:
                    raise ObservationError(f"Duplicate IMAGE_ID in Images.txt: {header.image_id}")
                if header.camera_id not in cameras:
                    raise ObservationError(
                        f"IMAGE_ID {header.image_id} references missing CAMERA_ID {header.camera_id}"
                    )
                headers[header.image_id] = header
                camera = cameras[header.camera_id]
                observation_count = 0
                for u_text, v_text, u, v, point_id in iter_points2d(fields, line_number):
                    if point_id < 0 or point_id not in selected_ids:
                        continue
                    point = points[point_id]
                    row = (
                        str(header.image_id),
                        header.name,
                        str(header.camera_id),
                        *header.pose_text,
                        u_text,
                        v_text,
                        str(point_id),
                        *point.xyz_text,
                    )
                    writer.writerow(row)
                    if not (0 <= u < camera.width and 0 <= v < camera.height):
                        invalid_writer.writerow(row)
                        out_of_bounds_observations += 1
                        out_of_bounds_image_ids.add(header.image_id)
                    observation_count += 1
                    total_observations += 1
                    observed_ids.add(point_id)
                image_counts.append(
                    ImageCount(header.image_id, header.name, header.camera_id, observation_count)
                )
        if not headers:
            raise ObservationError("Images.txt contains no image records")
        if total_observations == 0:
            raise ObservationError("No object observations found in Images.txt")

        verify_observation_csv(
            csv_temp,
            selected_ids,
            points,
            cameras,
            headers,
            total_observations,
            out_of_bounds_observations,
        )

        with counts_handle:
            writer = csv.writer(counts_handle, lineterminator="\n")
            writer.writerow(("image_id", "image_name", "camera_id", "observation_count"))
            for item in image_counts:
                writer.writerow(
                    (item.image_id, item.image_name, item.camera_id, item.observation_count)
                )

        overlays: Sequence[OverlayStatus] = []
        if create_visualizations:
            overlays = generate_qc_overlays(
                images_path,
                selected_ids,
                cameras,
                image_counts,
                headers,
                image_dir,
                output_dir / "observation_qc",
                image_template=image_template,
                image_index_offset=image_index_offset,
                allow_rescaled_images=allow_rescaled_images,
            )

        summary_handle, summary_temp = _temporary_csv(output_dir, ".observation-summary.tmp")
        with summary_handle:
            summary_handle.write(
                _summary_text(
                    len(selected_ids),
                    image_counts,
                    observed_ids,
                    total_observations,
                    out_of_bounds_observations,
                    len(out_of_bounds_image_ids),
                    overlays,
                )
            )

        os.replace(csv_temp, observations_path)
        csv_temp = Path()
        os.replace(invalid_temp, out_of_bounds_path)
        invalid_temp = Path()
        os.replace(counts_temp, counts_path)
        counts_temp = Path()
        os.replace(summary_temp, summary_path)
        summary_temp = None

        per_image = [item.observation_count for item in image_counts]
        return ObservationResult(
            len(selected_ids),
            len(image_counts),
            sum(value > 0 for value in per_image),
            total_observations,
            len(observed_ids),
            len(selected_ids) - len(observed_ids),
            out_of_bounds_observations,
            min(per_image),
            max(per_image),
            statistics.fmean(per_image),
            observations_path,
            out_of_bounds_path,
            counts_path,
            summary_path,
            overlays,
        )
    except OSError as exc:
        raise ObservationError(f"File operation failed: {exc}") from exc
    finally:
        if not csv_handle.closed:
            csv_handle.close()
        if not invalid_handle.closed:
            invalid_handle.close()
        if not counts_handle.closed:
            counts_handle.close()
        if summary_handle is not None and not summary_handle.closed:
            summary_handle.close()
        for path in (csv_temp, invalid_temp, counts_temp, summary_temp):
            if path and path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract object-only 2D-3D observations from COLMAP text files."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--selected-ids",
        type=Path,
        default=DEFAULT_STEP1_DIR / "selected_point_ids.txt",
    )
    parser.add_argument(
        "--filtered-points",
        type=Path,
        default=DEFAULT_STEP1_DIR / "filtered_Points3D.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STEP1_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip the three OpenCV overlay images",
    )
    parser.add_argument(
        "--image-template",
        help=(
            "Optional renamed-image pattern, e.g. "
            "'output_image2_{index:04d}.jpg'; supports {index}, {stem}, {name}"
        ),
    )
    parser.add_argument(
        "--image-index-offset",
        type=int,
        default=0,
        help="Offset added to a numeric COLMAP image stem for --image-template",
    )
    parser.add_argument(
        "--allow-rescaled-images",
        action="store_true",
        help="Allow approximate QC overlays when image dimensions differ from Cameras.txt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = extract_observations(
            args.images,
            args.cameras,
            args.selected_ids,
            args.filtered_points,
            args.output_dir,
            create_visualizations=not args.no_visualization,
            image_dir=args.image_dir,
            image_template=args.image_template,
            image_index_offset=args.image_index_offset,
            allow_rescaled_images=args.allow_rescaled_images,
        )
        print(f"Selected 3D points: {result.selected_points}")
        print(f"Total images: {result.total_images}")
        print(f"Images containing selected points: {result.images_with_observations}")
        print(f"Total object observations: {result.total_observations}")
        print(f"Out-of-bounds observations: {result.out_of_bounds_observations}")
        print(f"3D points without observations: {result.points_without_observations}")
        print(f"Output: {result.observations_path}")
        print(f"Out-of-bounds report: {result.out_of_bounds_path}")
        for overlay in result.overlays:
            print(
                f"QC {overlay.category}: {overlay.image_name}: {overlay.status}"
                + (f" -> {overlay.output_path}" if overlay.output_path else "")
            )
        return 0
    except ObservationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
