#!/usr/bin/env python3
"""Offline SIFT matching and RANSAC-PnP validation for an object feature map."""

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
from typing import Dict, List, Optional, Sequence, Tuple

from extract_object_observations import (
    Camera,
    ObservationError,
    load_cameras,
    parse_image_header,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "colmap_image"
DEFAULT_FEATURE_DIR = SCRIPT_DIR / "output_colmap_image"
DEFAULT_OUTPUT_DIR = DEFAULT_FEATURE_DIR / "pnp_test"

RESULT_COLUMNS = (
    "query_name",
    "status",
    "detail",
    "camera_id",
    "query_keypoints",
    "ratio_matches",
    "pnp_inliers",
    "inlier_ratio",
    "mean_inlier_reprojection_error",
    "max_inlier_reprojection_error",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x",
    "tvec_y",
    "tvec_z",
    "camera_center_x",
    "camera_center_y",
    "camera_center_z",
    "rotation_error_deg",
    "camera_center_error",
)

MATCH_COLUMNS = (
    "query_name",
    "query_keypoint_index",
    "u",
    "v",
    "point3d_id",
    "X",
    "Y",
    "Z",
    "descriptor_distance",
    "ratio",
    "reprojection_error",
    "is_inlier",
)


class PnPTestError(RuntimeError):
    """Expected input, feature-map, or localization error."""


@dataclass(frozen=True)
class ReferenceImage:
    image_id: int
    camera_id: int
    name: str
    pose: Tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class MapData:
    point3d_ids: object
    xyz: object
    descriptors: object
    image_ids: object


@dataclass(frozen=True)
class CandidateMatch:
    query_index: int
    map_index: int
    point3d_id: int
    descriptor_distance: float
    ratio: float


@dataclass(frozen=True)
class EvaluatedMatch:
    candidate: CandidateMatch
    u: float
    v: float
    point3d_id: int
    xyz: Tuple[float, float, float]
    reprojection_error: float
    is_inlier: bool


@dataclass(frozen=True)
class LocalizationResult:
    query_name: str
    status: str
    detail: str
    camera_id: Optional[int]
    query_keypoints: int
    ratio_matches: int
    pnp_inliers: int
    inlier_ratio: float
    mean_inlier_reprojection_error: Optional[float]
    max_inlier_reprojection_error: Optional[float]
    rvec: Optional[Tuple[float, float, float]]
    tvec: Optional[Tuple[float, float, float]]
    camera_center: Optional[Tuple[float, float, float]]
    rotation_error_deg: Optional[float]
    camera_center_error: Optional[float]
    overlay_path: Optional[Path]


def load_reference_images(path: Path) -> Dict[str, ReferenceImage]:
    if not path.is_file():
        raise PnPTestError(f"Images.txt not found: {path}")
    references: Dict[str, ReferenceImage] = {}
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
                try:
                    header = parse_image_header(header_line, line_number)
                except ObservationError as exc:
                    raise PnPTestError(str(exc)) from exc
                points_line = source.readline()
                if points_line == "":
                    raise PnPTestError(
                        f"Images.txt ended after header at line {line_number}"
                    )
                line_number += 1
                if header.name in references:
                    raise PnPTestError(f"Duplicate image name in Images.txt: {header.name}")
                references[header.name] = ReferenceImage(
                    header.image_id,
                    header.camera_id,
                    header.name,
                    header.pose,
                )
    except (OSError, UnicodeError) as exc:
        raise PnPTestError(f"Cannot read Images.txt: {path}: {exc}") from exc
    if not references:
        raise PnPTestError("Images.txt contains no image records")
    return references


def load_feature_map(np, path: Path) -> MapData:
    if not path.is_file():
        raise PnPTestError(f"object_features.npz not found: {path}")
    required = {"point3d_ids", "xyz", "descriptors", "image_ids"}
    try:
        with np.load(path, allow_pickle=False) as feature_map:
            missing = required - set(feature_map.files)
            if missing:
                raise PnPTestError(
                    "object_features.npz is missing arrays: " + ", ".join(sorted(missing))
                )
            point3d_ids = feature_map["point3d_ids"].astype(np.int64, copy=True)
            xyz = feature_map["xyz"].astype(np.float64, copy=True)
            descriptors = feature_map["descriptors"].astype(np.float32, copy=True)
            image_ids = feature_map["image_ids"].astype(np.int64, copy=True)
    except (OSError, ValueError) as exc:
        raise PnPTestError(f"Cannot load object_features.npz: {exc}") from exc
    count = len(point3d_ids)
    if count == 0:
        raise PnPTestError("object_features.npz contains no descriptors")
    if xyz.shape != (count, 3) or descriptors.shape != (count, 128):
        raise PnPTestError("object_features.npz has invalid XYZ or descriptor shape")
    if image_ids.shape != (count,):
        raise PnPTestError("object_features.npz has invalid image_ids shape")
    if not np.isfinite(xyz).all() or not np.isfinite(descriptors).all():
        raise PnPTestError("object_features.npz contains non-finite values")
    xyz_by_id: Dict[int, Tuple[float, float, float]] = {}
    for point_id, point_xyz in zip(point3d_ids.tolist(), xyz):
        value = tuple(float(component) for component in point_xyz)
        previous = xyz_by_id.setdefault(point_id, value)
        if previous != value:
            raise PnPTestError(f"POINT3D_ID {point_id} has inconsistent XYZ in map")
    return MapData(point3d_ids, xyz, descriptors, image_ids)


def camera_matrix_and_distortion(np, camera: Camera, camera_line: Sequence[str]):
    params = [float(value) for value in camera_line]
    model = camera.model.upper()
    if model == "SIMPLE_PINHOLE" and len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = np.zeros(5, dtype=np.float64)
    elif model == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        distortion = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL" and len(params) == 4:
        fx = fy = params[0]
        cx, cy, k1 = params[1:4]
        distortion = np.asarray((k1, 0, 0, 0, 0), dtype=np.float64)
    elif model == "RADIAL" and len(params) == 5:
        fx = fy = params[0]
        cx, cy, k1, k2 = params[1:5]
        distortion = np.asarray((k1, k2, 0, 0, 0), dtype=np.float64)
    elif model == "OPENCV" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = np.asarray((k1, k2, p1, p2, 0), dtype=np.float64)
    else:
        raise PnPTestError(
            f"Unsupported camera model or parameter count: {camera.model} ({len(params)})"
        )
    matrix = np.asarray(
        ((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64
    )
    return matrix, distortion


def load_camera_parameters(path: Path) -> Tuple[Dict[int, Camera], Dict[int, Tuple[str, ...]]]:
    try:
        cameras = load_cameras(path)
    except ObservationError as exc:
        raise PnPTestError(str(exc)) from exc
    parameter_text: Dict[int, Tuple[str, ...]] = {}
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("The files Cameras.txt"):
                    continue
                fields = stripped.split()
                parameter_text[int(fields[0])] = tuple(fields[4:])
    except (OSError, UnicodeError, ValueError) as exc:
        raise PnPTestError(f"Cannot read camera parameters: {exc}") from exc
    return cameras, parameter_text


def quaternion_to_rotation(np, pose: Sequence[float]):
    qw, qx, qy, qz = pose[:4]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise PnPTestError("Ground-truth quaternion has zero norm")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        ),
        dtype=np.float64,
    )


def pose_errors(cv2, np, rvec, tvec, reference: ReferenceImage):
    estimated_rotation, _ = cv2.Rodrigues(rvec)
    ground_truth_rotation = quaternion_to_rotation(np, reference.pose)
    ground_truth_translation = np.asarray(reference.pose[4:7], dtype=np.float64).reshape(3, 1)
    delta = estimated_rotation @ ground_truth_rotation.T
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))
    rotation_error = math.degrees(math.acos(cosine))
    estimated_center = -estimated_rotation.T @ tvec
    ground_truth_center = -ground_truth_rotation.T @ ground_truth_translation
    center_error = float(np.linalg.norm(estimated_center - ground_truth_center))
    return rotation_error, center_error, tuple(float(value) for value in estimated_center.ravel())


def select_unique_point_matches(
    raw_neighbors: Sequence[Sequence[object]],
    map_point_ids,
    ratio_threshold: float,
) -> List[CandidateMatch]:
    if not 0 < ratio_threshold < 1:
        raise PnPTestError("--ratio must be between 0 and 1")
    candidates: List[CandidateMatch] = []
    for query_index, neighbors in enumerate(raw_neighbors):
        unique: List[object] = []
        seen_points = set()
        for neighbor in neighbors:
            point_id = int(map_point_ids[neighbor.trainIdx])
            if point_id in seen_points:
                continue
            unique.append(neighbor)
            seen_points.add(point_id)
            if len(unique) == 2:
                break
        if len(unique) < 2:
            continue
        first, second = unique
        if first.distance < ratio_threshold * second.distance:
            candidates.append(
                CandidateMatch(
                    query_index,
                    int(first.trainIdx),
                    int(map_point_ids[first.trainIdx]),
                    float(first.distance),
                    float(first.distance / max(second.distance, 1e-12)),
                )
            )

    # PnP receives at most one 2D observation for each 3D point.
    result: List[CandidateMatch] = []
    used_points = set()
    for candidate in sorted(
        candidates, key=lambda item: (item.descriptor_distance, item.query_index)
    ):
        if candidate.point3d_id not in used_points:
            result.append(candidate)
            used_points.add(candidate.point3d_id)
    return sorted(result, key=lambda item: item.query_index)


def _empty_result(name: str, status: str, detail: str, camera_id=None, keypoints=0, matches=0):
    return LocalizationResult(
        name,
        status,
        detail,
        camera_id,
        keypoints,
        matches,
        0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _draw_overlay(
    cv2,
    np,
    image,
    query_name: str,
    keypoints,
    matches: Sequence[CandidateMatch],
    inlier_indices: set,
    output_path: Path,
    status: str,
) -> Path:
    canvas = image.copy()
    for index, match in enumerate(matches):
        u, v = keypoints[match.query_index].pt
        is_inlier = index in inlier_indices
        color = (40, 220, 70) if is_inlier else (40, 40, 230)
        cv2.circle(canvas, (round(u), round(v)), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (round(u), round(v)), 2, color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (25, 25, 25), -1)
    label = (
        f"{status} | {query_name} | matches {len(matches)} | "
        f"PnP inliers {len(inlier_indices)}"
    )
    cv2.putText(
        canvas,
        label,
        (16, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise PnPTestError(f"Could not write QC overlay: {output_path}")
    return output_path


def localize_query(
    cv2,
    np,
    query_path: Path,
    map_data: MapData,
    reference: Optional[ReferenceImage],
    camera: Optional[Camera],
    camera_params: Optional[Sequence[str]],
    *,
    manual_intrinsics: Optional[Sequence[float]],
    manual_distortion: Sequence[float],
    exclude_self: bool,
    ratio_threshold: float,
    knn: int,
    ransac_threshold: float,
    confidence: float,
    iterations: int,
    min_inliers: int,
    min_inlier_ratio: float,
    overlay_dir: Path,
) -> Tuple[LocalizationResult, List[EvaluatedMatch]]:
    read_flags = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    image = cv2.imread(str(query_path), read_flags)
    if image is None:
        return _empty_result(query_path.name, "ERROR", "IMAGE_READ_FAILED"), []
    height, width = image.shape[:2]
    camera_id = reference.camera_id if reference is not None else None
    if reference is not None:
        if camera is None or camera_params is None:
            return _empty_result(query_path.name, "ERROR", "CAMERA_NOT_FOUND", camera_id), []
        if (width, height) != (camera.width, camera.height):
            return _empty_result(
                query_path.name,
                "ERROR",
                f"IMAGE_SIZE_MISMATCH: expected {camera.width}x{camera.height}, got {width}x{height}",
                camera_id,
            ), []
        camera_matrix, distortion = camera_matrix_and_distortion(np, camera, camera_params)
    elif manual_intrinsics is not None:
        fx, fy, cx, cy = manual_intrinsics
        camera_matrix = np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
        distortion = np.asarray(manual_distortion, dtype=np.float64)
    else:
        return _empty_result(
            query_path.name,
            "ERROR",
            "CAMERA_PARAMETERS_UNAVAILABLE: provide --intrinsics",
        ), []

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, query_descriptors = cv2.SIFT_create().detectAndCompute(grayscale, None)
    if query_descriptors is None or not keypoints:
        return _empty_result(query_path.name, "ERROR", "NO_SIFT_FEATURES", camera_id), []

    mask = np.ones(len(map_data.point3d_ids), dtype=bool)
    if exclude_self and reference is not None:
        mask &= map_data.image_ids != reference.image_id
    train_descriptors = map_data.descriptors[mask]
    train_point_ids = map_data.point3d_ids[mask]
    train_xyz = map_data.xyz[mask]
    if len(train_descriptors) < 2:
        return _empty_result(
            query_path.name, "ERROR", "FEATURE_MAP_TOO_SMALL", camera_id, len(keypoints)
        ), []

    actual_knn = min(max(2, knn), len(train_descriptors))
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=96)
    )
    raw_neighbors = matcher.knnMatch(
        query_descriptors.astype(np.float32, copy=False),
        train_descriptors,
        k=actual_knn,
    )
    matches = select_unique_point_matches(
        raw_neighbors, train_point_ids, ratio_threshold
    )
    if len(matches) < 4:
        return _empty_result(
            query_path.name,
            "FAIL",
            "INSUFFICIENT_2D3D_MATCHES",
            camera_id,
            len(keypoints),
            len(matches),
        ), []

    object_points = np.asarray(
        [train_xyz[match.map_index] for match in matches], dtype=np.float64
    )
    image_points = np.asarray(
        [keypoints[match.query_index].pt for match in matches], dtype=np.float64
    )
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        iterationsCount=iterations,
        reprojectionError=ransac_threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 4:
        return _empty_result(
            query_path.name,
            "FAIL",
            "PNP_RANSAC_FAILED",
            camera_id,
            len(keypoints),
            len(matches),
        ), []
    inlier_list = [int(value) for value in inliers.ravel()]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_list],
            image_points[inlier_list],
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
    reprojection_errors = np.linalg.norm(
        projected.reshape((-1, 2)) - image_points, axis=1
    )
    inlier_set = set(inlier_list)
    inlier_errors = reprojection_errors[inlier_list]
    inlier_ratio = len(inlier_list) / len(matches)
    status = (
        "PASS"
        if len(inlier_list) >= min_inliers and inlier_ratio >= min_inlier_ratio
        else "FAIL"
    )
    detail = "OK" if status == "PASS" else "PNP_BELOW_ACCEPTANCE_THRESHOLD"
    rotation_error = center_error = None
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_center_array = -rotation_matrix.T @ tvec
    camera_center = tuple(float(value) for value in camera_center_array.ravel())
    if reference is not None:
        rotation_error, center_error, camera_center = pose_errors(
            cv2, np, rvec, tvec, reference
        )
    safe_stem = "".join(
        character if character.isalnum() else "_" for character in query_path.stem
    )
    overlay_path = _draw_overlay(
        cv2,
        np,
        image,
        query_path.name,
        keypoints,
        matches,
        inlier_set,
        overlay_dir / f"{safe_stem}_pnp.jpg",
        status,
    )
    result = LocalizationResult(
        query_path.name,
        status,
        detail,
        camera_id,
        len(keypoints),
        len(matches),
        len(inlier_list),
        inlier_ratio,
        float(np.mean(inlier_errors)),
        float(np.max(inlier_errors)),
        tuple(float(value) for value in rvec.ravel()),
        tuple(float(value) for value in tvec.ravel()),
        camera_center,
        rotation_error,
        center_error,
        overlay_path,
    )
    match_rows = []
    for index, match in enumerate(matches):
        u, v = keypoints[match.query_index].pt
        point_xyz = train_xyz[match.map_index]
        match_rows.append(
            EvaluatedMatch(
                match,
                float(u),
                float(v),
                int(train_point_ids[match.map_index]),
                tuple(float(value) for value in point_xyz),
                float(reprojection_errors[index]),
                index in inlier_set,
            )
        )
    return result, match_rows


def _result_row(result: LocalizationResult) -> List[object]:
    def values(value, count):
        return list(value) if value is not None else [""] * count

    return [
        result.query_name,
        result.status,
        result.detail,
        "" if result.camera_id is None else result.camera_id,
        result.query_keypoints,
        result.ratio_matches,
        result.pnp_inliers,
        f"{result.inlier_ratio:.9g}",
        "" if result.mean_inlier_reprojection_error is None else f"{result.mean_inlier_reprojection_error:.9g}",
        "" if result.max_inlier_reprojection_error is None else f"{result.max_inlier_reprojection_error:.9g}",
        *values(result.rvec, 3),
        *values(result.tvec, 3),
        *values(result.camera_center, 3),
        "" if result.rotation_error_deg is None else f"{result.rotation_error_deg:.9g}",
        "" if result.camera_center_error is None else f"{result.camera_center_error:.9g}",
    ]


def run_offline_pnp_test(
    features_path: Path,
    images_path: Path,
    cameras_path: Path,
    query_dir: Path,
    query_images: Sequence[str],
    output_dir: Path,
    *,
    manual_intrinsics: Optional[Sequence[float]] = None,
    manual_distortion: Sequence[float] = (0, 0, 0, 0, 0),
    exclude_self: bool = True,
    ratio_threshold: float = 0.75,
    knn: int = 16,
    ransac_threshold: float = 4.0,
    confidence: float = 0.999,
    iterations: int = 10000,
    min_inliers: int = 20,
    min_inlier_ratio: float = 0.2,
) -> List[LocalizationResult]:
    if not query_images:
        raise PnPTestError("At least one --query-image is required")
    if not 0 < ratio_threshold < 1:
        raise PnPTestError("--ratio must be between 0 and 1")
    if knn < 2 or ransac_threshold <= 0 or iterations <= 0 or min_inliers < 4:
        raise PnPTestError("Invalid matching or RANSAC parameter")
    if not 0 < confidence < 1 or not 0 <= min_inlier_ratio <= 1:
        raise PnPTestError("Invalid confidence or minimum inlier ratio")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise PnPTestError("Offline PnP requires OpenCV and NumPy") from exc

    map_data = load_feature_map(np, features_path)
    references = load_reference_images(images_path)
    cameras, camera_parameters = load_camera_parameters(cameras_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    def temporary_path(suffix: str) -> Path:
        descriptor, name = tempfile.mkstemp(dir=output_dir, suffix=suffix)
        os.close(descriptor)
        return Path(name)

    temp_results = temporary_path(".results.tmp")
    temp_matches = temporary_path(".matches.tmp")
    temp_summary = temporary_path(".summary.tmp")
    results_path = output_dir / "localization_results.csv"
    matches_path = output_dir / "localization_matches.csv"
    summary_path = output_dir / "localization_summary.txt"
    results: List[LocalizationResult] = []
    try:
        with (
            temp_results.open("w", encoding="utf-8", newline="") as result_file,
            temp_matches.open("w", encoding="utf-8", newline="") as match_file,
        ):
            result_writer = csv.writer(result_file, lineterminator="\n")
            match_writer = csv.writer(match_file, lineterminator="\n")
            result_writer.writerow(RESULT_COLUMNS)
            match_writer.writerow(MATCH_COLUMNS)
            for query_name in query_images:
                query_path = Path(query_name)
                if not query_path.is_absolute():
                    query_path = query_dir / query_path
                reference = references.get(query_path.name)
                camera = cameras.get(reference.camera_id) if reference else None
                params = camera_parameters.get(reference.camera_id) if reference else None
                if not query_path.is_file():
                    result = _empty_result(query_path.name, "ERROR", "IMAGE_NOT_FOUND")
                    match_rows = []
                else:
                    result, match_rows = localize_query(
                        cv2,
                        np,
                        query_path,
                        map_data,
                        reference,
                        camera,
                        params,
                        manual_intrinsics=manual_intrinsics,
                        manual_distortion=manual_distortion,
                        exclude_self=exclude_self,
                        ratio_threshold=ratio_threshold,
                        knn=knn,
                        ransac_threshold=ransac_threshold,
                        confidence=confidence,
                        iterations=iterations,
                        min_inliers=min_inliers,
                        min_inlier_ratio=min_inlier_ratio,
                        overlay_dir=output_dir / "qc",
                    )
                results.append(result)
                result_writer.writerow(_result_row(result))
                if match_rows:
                    for evaluated in match_rows:
                        match = evaluated.candidate
                        match_writer.writerow(
                            (
                                result.query_name,
                                match.query_index,
                                f"{evaluated.u:.9g}",
                                f"{evaluated.v:.9g}",
                                evaluated.point3d_id,
                                *[f"{value:.17g}" for value in evaluated.xyz],
                                f"{match.descriptor_distance:.9g}",
                                f"{match.ratio:.9g}",
                                f"{evaluated.reprojection_error:.9g}",
                                int(evaluated.is_inlier),
                            )
                        )

        passed = [result for result in results if result.status == "PASS"]
        pose_results = [result for result in results if result.rotation_error_deg is not None]
        lines = [
            f"Query images: {len(results)}",
            f"PASS: {len(passed)}",
            f"FAIL or ERROR: {len(results) - len(passed)}",
            f"Self-image descriptors excluded: {exclude_self}",
            f"Lowe ratio threshold: {ratio_threshold}",
            f"RANSAC reprojection threshold: {ransac_threshold}",
            f"Minimum inliers: {min_inliers}",
            f"Minimum inlier ratio: {min_inlier_ratio}",
            "",
        ]
        if pose_results:
            lines.extend(
                (
                    f"Mean rotation error (deg): {statistics.fmean(result.rotation_error_deg for result in pose_results):.6f}",
                    f"Max rotation error (deg): {max(result.rotation_error_deg for result in pose_results):.6f}",
                    f"Mean camera-center error (model units): {statistics.fmean(result.camera_center_error for result in pose_results):.6f}",
                    f"Max camera-center error (model units): {max(result.camera_center_error for result in pose_results):.6f}",
                    "",
                )
            )
        lines.append("Per-query results:")
        for result in results:
            lines.append(
                f"{result.query_name}: {result.status}, matches={result.ratio_matches}, "
                f"inliers={result.pnp_inliers}, inlier_ratio={result.inlier_ratio:.3f}, "
                f"rotation_error_deg={result.rotation_error_deg}, "
                f"camera_center_error={result.camera_center_error}, detail={result.detail}"
            )
        temp_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temp_results, results_path)
        os.replace(temp_matches, matches_path)
        os.replace(temp_summary, summary_path)
        return results
    except (OSError, UnicodeError) as exc:
        raise PnPTestError(f"Output operation failed: {exc}") from exc
    finally:
        for path in (temp_results, temp_matches, temp_summary):
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline SIFT matching and RANSAC-PnP test for an object feature map."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR / "object_features.npz")
    parser.add_argument("--images", type=Path, default=DEFAULT_DATA_DIR / "Images.txt")
    parser.add_argument("--cameras", type=Path, default=DEFAULT_DATA_DIR / "Cameras.txt")
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--query-image", action="append", required=True, dest="query_images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--distortion", nargs=5, type=float, default=(0, 0, 0, 0, 0), metavar=("K1", "K2", "P1", "P2", "K3"))
    parser.add_argument("--keep-self-descriptors", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        results = run_offline_pnp_test(
            args.features,
            args.images,
            args.cameras,
            args.query_dir,
            args.query_images,
            args.output_dir,
            manual_intrinsics=args.intrinsics,
            manual_distortion=args.distortion,
            exclude_self=not args.keep_self_descriptors,
            ratio_threshold=args.ratio,
            knn=args.knn,
            ransac_threshold=args.ransac_threshold,
            confidence=args.confidence,
            iterations=args.iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
        )
        for result in results:
            print(
                f"{result.query_name}: {result.status}, matches={result.ratio_matches}, "
                f"inliers={result.pnp_inliers}, rotation_error={result.rotation_error_deg}, "
                f"center_error={result.camera_center_error}"
            )
        return 0 if all(result.status == "PASS" for result in results) else 2
    except PnPTestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
