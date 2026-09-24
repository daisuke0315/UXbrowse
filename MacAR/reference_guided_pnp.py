#!/usr/bin/env python3
"""Static reference-guided 2D-3D matching and PnP proof of concept."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from offline_pnp_test import load_reference_images, quaternion_to_rotation
from realtime_pnp import approximate_camera_matrix
from realtime_pnp_server import DEFAULT_FEATURES


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY = Path("/Users/daisuketanii/Downloads/IMG_2532.jpeg")
DEFAULT_IMAGES = SCRIPT_DIR / "colmap_image" / "Images.txt"
DEFAULT_REFERENCE_DIR = SCRIPT_DIR / "colmap_image"
DEFAULT_OUTPUT = SCRIPT_DIR / "output_colmap_image" / "reference_guided_pnp"


class ReferenceGuidedError(RuntimeError):
    """Expected input or localization error."""


@dataclass(frozen=True)
class ReferenceScore:
    image_id: int
    image_name: str
    votes: int
    score: float


@dataclass(frozen=True)
class TransferredMatch:
    query_index: int
    map_index: int
    point3d_id: int
    descriptor_distance: float
    ratio: float


@dataclass
class ReferenceResult:
    rank: int
    reference: ReferenceScore
    raw_matches: int
    ratio_matches: int
    mutual_matches: int
    geometric_inliers: int
    transferred: list[TransferredMatch]
    pnp_inliers: int = 0
    pnp_inlier_ratio: float = 0.0
    mean_error: Optional[float] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    guided_correspondences: int = 0
    guided_inliers: int = 0
    guided_inlier_ratio: float = 0.0
    guided_mean_error: Optional[float] = None
    guided_transferred: list[TransferredMatch] = field(default_factory=list)
    guided_rvec: Optional[np.ndarray] = None
    guided_tvec: Optional[np.ndarray] = None
    guided_inlier_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PoseResult:
    correspondences: int
    inliers: int
    inlier_ratio: float
    mean_error: Optional[float]
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    inlier_indices: tuple[int, ...] = ()


class ReferenceRetriever:
    def __init__(self, descriptors: np.ndarray, image_ids: np.ndarray) -> None:
        self.descriptors = descriptors
        self.image_ids = image_ids
        cv2.setRNGSeed(0)
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=96)
        )

    def retrieve(
        self,
        query_descriptors: np.ndarray,
        references_by_id,
        top_k: int,
        neighbors: int = 8,
        maximum_distance: float = 300.0,
    ) -> list[ReferenceScore]:
        actual_k = min(neighbors, len(self.descriptors))
        raw = self.matcher.knnMatch(
            query_descriptors.astype(np.float32, copy=False),
            self.descriptors,
            k=actual_k,
        )
        votes: Counter[int] = Counter()
        scores: defaultdict[int, float] = defaultdict(float)
        for matches in raw:
            seen_images = set()
            for match in matches:
                if match.distance > maximum_distance:
                    continue
                image_id = int(self.image_ids[match.trainIdx])
                if image_id in seen_images or image_id not in references_by_id:
                    continue
                seen_images.add(image_id)
                votes[image_id] += 1
                scores[image_id] += max(
                    0.0, 1.0 - float(match.distance) / maximum_distance
                )
        ranked = sorted(
            votes,
            key=lambda image_id: (votes[image_id], scores[image_id]),
            reverse=True,
        )[:top_k]
        return [
            ReferenceScore(
                image_id,
                references_by_id[image_id].name,
                votes[image_id],
                scores[image_id],
            )
            for image_id in ranked
        ]


class ReferenceMatcher:
    def __init__(
        self,
        point3d_ids: np.ndarray,
        descriptors: np.ndarray,
        image_ids: np.ndarray,
        reference_uv: np.ndarray,
        *,
        ratio_threshold: float,
        fundamental_threshold: float,
    ) -> None:
        self.point3d_ids = point3d_ids
        self.descriptors = descriptors
        self.reference_uv = reference_uv
        self.ratio_threshold = ratio_threshold
        self.fundamental_threshold = fundamental_threshold
        self.indices_by_image = {
            int(image_id): np.flatnonzero(image_ids == image_id)
            for image_id in np.unique(image_ids)
        }

    def match(
        self,
        query_keypoints,
        query_descriptors: np.ndarray,
        image_id: int,
    ) -> tuple[int, int, int, list[TransferredMatch]]:
        indices = self.indices_by_image.get(image_id)
        if indices is None or len(indices) < 2:
            return 0, 0, 0, []
        train = self.descriptors[indices]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_neighbors = matcher.knnMatch(query_descriptors, train, k=2)
        reverse = matcher.match(train, query_descriptors)
        reverse_best = {match.queryIdx: match.trainIdx for match in reverse}
        ratio_candidates = []
        for neighbors in raw_neighbors:
            if len(neighbors) < 2:
                continue
            first, second = neighbors[:2]
            if first.distance >= self.ratio_threshold * second.distance:
                continue
            ratio_candidates.append((first, second))
        mutual = [
            pair
            for pair in ratio_candidates
            if reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx
        ]
        unique = []
        used_query = set()
        used_points = set()
        for first, second in sorted(mutual, key=lambda pair: pair[0].distance):
            map_index = int(indices[first.trainIdx])
            point_id = int(self.point3d_ids[map_index])
            if first.queryIdx in used_query or point_id in used_points:
                continue
            used_query.add(first.queryIdx)
            used_points.add(point_id)
            unique.append((first, second, map_index, point_id))
        if len(unique) < 8:
            return len(raw_neighbors), len(ratio_candidates), len(mutual), []
        query_points = np.asarray(
            [query_keypoints[item[0].queryIdx].pt for item in unique],
            dtype=np.float64,
        )
        reference_points = np.asarray(
            [self.reference_uv[item[2]] for item in unique], dtype=np.float64
        )
        _, mask = cv2.findFundamentalMat(
            query_points,
            reference_points,
            cv2.FM_RANSAC,
            self.fundamental_threshold,
            0.999,
            10_000,
        )
        if mask is None:
            return len(raw_neighbors), len(ratio_candidates), len(mutual), []
        transferred = []
        for keep, (first, second, map_index, point_id) in zip(
            mask.ravel().astype(bool), unique
        ):
            if not keep:
                continue
            transferred.append(
                TransferredMatch(
                    int(first.queryIdx),
                    map_index,
                    point_id,
                    float(first.distance),
                    float(first.distance / max(second.distance, 1e-12)),
                )
            )
        return (
            len(raw_neighbors),
            len(ratio_candidates),
            len(mutual),
            transferred,
        )


class PoseRefiner:
    def __init__(
        self,
        point3d_ids: np.ndarray,
        xyz: np.ndarray,
        camera_matrix: np.ndarray,
        *,
        ransac_threshold: float,
        iterations: int,
        distortion: Optional[np.ndarray] = None,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.ransac_threshold = ransac_threshold
        self.iterations = iterations
        self.xyz_by_id = {}
        for point_id, point_xyz in zip(point3d_ids.tolist(), xyz):
            self.xyz_by_id.setdefault(int(point_id), point_xyz.astype(np.float64))

    def solve(self, keypoints, matches: Sequence[TransferredMatch]) -> PoseResult:
        if len(matches) < 4:
            return PoseResult(len(matches), 0, 0.0, None, None, None)
        object_points = np.asarray(
            [self.xyz_by_id[match.point3d_id] for match in matches],
            dtype=np.float64,
        )
        image_points = np.asarray(
            [keypoints[match.query_index].pt for match in matches],
            dtype=np.float64,
        )
        cv2.setRNGSeed(0)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            iterationsCount=self.iterations,
            reprojectionError=self.ransac_threshold,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < 4:
            return PoseResult(len(matches), 0, 0.0, None, None, None)
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return PoseResult(len(matches), 0, 0.0, None, None, None)
        inlier_indices = inliers.ravel().astype(int)
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                self.camera_matrix,
                self.distortion,
                rvec,
                tvec,
            )
            if np.isfinite(refined_rvec).all() and np.isfinite(refined_tvec).all():
                rvec, tvec = refined_rvec, refined_tvec
        except cv2.error:
            pass
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.distortion
        )
        errors = np.linalg.norm(projected.reshape((-1, 2)) - image_points, axis=1)
        count = len(inlier_indices)
        return PoseResult(
            len(matches),
            count,
            count / len(matches),
            float(np.mean(errors[inlier_indices])),
            rvec,
            tvec,
            tuple(int(value) for value in inlier_indices),
        )


class GuidedMatcher:
    """Expand a provisional pose using descriptor matches near projected 3D points."""

    def __init__(
        self,
        point3d_ids: np.ndarray,
        xyz: np.ndarray,
        descriptors: np.ndarray,
    ) -> None:
        self.descriptors = descriptors.astype(np.float32, copy=False)
        self.descriptor_point_ids = point3d_ids.astype(np.int64, copy=False)
        first_by_id = {}
        descriptor_indices: defaultdict[int, list[int]] = defaultdict(list)
        for index, point_id_value in enumerate(point3d_ids.tolist()):
            point_id = int(point_id_value)
            first_by_id.setdefault(point_id, index)
            descriptor_indices[point_id].append(index)
        self.point_ids = np.asarray(sorted(first_by_id), dtype=np.int64)
        self.xyz = np.asarray(
            [xyz[first_by_id[int(point_id)]] for point_id in self.point_ids],
            dtype=np.float64,
        )
        self.descriptor_indices = descriptor_indices
        self.first_map_index = {
            int(point_id): first_by_id[int(point_id)] for point_id in self.point_ids
        }
        self.point_index_by_id = {
            int(point_id): index for index, point_id in enumerate(self.point_ids)
        }

    def match_from_neighbors(
        self,
        keypoints,
        descriptor_neighbors,
        pose: PoseResult,
        camera_matrix: np.ndarray,
        image_size: tuple[int, int],
        *,
        distortion: Optional[np.ndarray] = None,
        radius: float,
        ratio_threshold: float,
        maximum_distance: float,
        maximum_spatial_candidates: int,
    ) -> list[TransferredMatch]:
        """Filter one precomputed global descriptor search by pose projection."""
        if pose.rvec is None or pose.tvec is None:
            return []
        if not np.isfinite(pose.rvec).all() or not np.isfinite(pose.tvec).all():
            return []
        if np.max(np.abs(pose.rvec)) > 1e6 or np.max(np.abs(pose.tvec)) > 1e12:
            return []
        rotation, _ = cv2.Rodrigues(pose.rvec)
        if not np.isfinite(rotation).all():
            return []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            camera_xyz = (rotation @ self.xyz.T + pose.tvec).T
        if not np.isfinite(camera_xyz).all():
            return []
        projected, _ = cv2.projectPoints(
            self.xyz, pose.rvec, pose.tvec, camera_matrix, distortion
        )
        projected = projected.reshape((-1, 2))
        width, height = image_size
        valid = (
            np.isfinite(projected).all(axis=1)
            & (camera_xyz[:, 2] > 1e-6)
            & (projected[:, 0] >= -radius)
            & (projected[:, 0] < width + radius)
            & (projected[:, 1] >= -radius)
            & (projected[:, 1] < height + radius)
        )
        radius_squared = radius * radius
        candidates = []
        for query_index, neighbors in enumerate(descriptor_neighbors):
            query_uv = np.asarray(keypoints[query_index].pt, dtype=np.float64)
            distinct = []
            seen_points = set()
            for match in neighbors[:maximum_spatial_candidates]:
                if match.distance > maximum_distance:
                    break
                point_id = int(self.descriptor_point_ids[match.trainIdx])
                if point_id in seen_points:
                    continue
                point_index = self.point_index_by_id[point_id]
                if not valid[point_index]:
                    continue
                delta = projected[point_index] - query_uv
                if float(delta @ delta) > radius_squared:
                    continue
                seen_points.add(point_id)
                distinct.append((match, point_id))
                if len(distinct) == 2:
                    break
            if len(distinct) < 2:
                continue
            first, second = distinct
            ratio = first[0].distance / max(second[0].distance, 1e-12)
            if ratio >= ratio_threshold:
                continue
            candidates.append(
                TransferredMatch(
                    query_index,
                    int(first[0].trainIdx),
                    int(first[1]),
                    float(first[0].distance),
                    float(ratio),
                )
            )
        result = []
        used_query = set()
        used_points = set()
        for match in sorted(
            candidates,
            key=lambda item: (item.ratio, item.descriptor_distance),
        ):
            if match.query_index in used_query or match.point3d_id in used_points:
                continue
            used_query.add(match.query_index)
            used_points.add(match.point3d_id)
            result.append(match)
        return sorted(result, key=lambda item: item.query_index)

    def match(
        self,
        keypoints,
        query_descriptors: np.ndarray,
        pose: PoseResult,
        camera_matrix: np.ndarray,
        image_size: tuple[int, int],
        *,
        distortion: Optional[np.ndarray] = None,
        radius: float,
        ratio_threshold: float,
        maximum_distance: float,
        maximum_spatial_candidates: int,
    ) -> list[TransferredMatch]:
        if pose.rvec is None or pose.tvec is None:
            return []
        if not np.isfinite(pose.rvec).all() or not np.isfinite(pose.tvec).all():
            return []
        if np.max(np.abs(pose.rvec)) > 1e6 or np.max(np.abs(pose.tvec)) > 1e12:
            return []
        rotation, _ = cv2.Rodrigues(pose.rvec)
        if not np.isfinite(rotation).all():
            return []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            camera_xyz = (rotation @ self.xyz.T + pose.tvec).T
        if not np.isfinite(camera_xyz).all():
            return []
        projected, _ = cv2.projectPoints(
            self.xyz, pose.rvec, pose.tvec, camera_matrix, distortion
        )
        projected = projected.reshape((-1, 2))
        width, height = image_size
        valid = (
            (camera_xyz[:, 2] > 1e-6)
            & (projected[:, 0] >= -radius)
            & (projected[:, 0] < width + radius)
            & (projected[:, 1] >= -radius)
            & (projected[:, 1] < height + radius)
        )
        cell_size = max(radius, 1.0)
        grid: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for point_index in np.flatnonzero(valid):
            u, v = projected[point_index]
            grid[(math.floor(u / cell_size), math.floor(v / cell_size))].append(
                int(point_index)
            )
        candidates = []
        radius_squared = radius * radius
        for query_index, keypoint in enumerate(keypoints):
            u, v = keypoint.pt
            cell = (math.floor(u / cell_size), math.floor(v / cell_size))
            nearby = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for point_index in grid.get((cell[0] + dx, cell[1] + dy), ()):
                        delta = projected[point_index] - np.asarray((u, v))
                        spatial_squared = float(delta @ delta)
                        if spatial_squared <= radius_squared:
                            nearby.append((spatial_squared, point_index))
            if len(nearby) < 2:
                continue
            nearby.sort(key=lambda item: item[0])
            distinct = []
            query_descriptor = query_descriptors[query_index]
            for spatial_squared, point_index in nearby[:maximum_spatial_candidates]:
                point_id = int(self.point_ids[point_index])
                indices = self.descriptor_indices[point_id]
                distances = np.linalg.norm(
                    self.descriptors[indices] - query_descriptor,
                    axis=1,
                )
                best_local = int(np.argmin(distances))
                distinct.append(
                    (
                        float(distances[best_local]),
                        spatial_squared,
                        point_id,
                        indices[best_local],
                    )
                )
            distinct.sort(key=lambda item: (item[0], item[1]))
            if len(distinct) < 2:
                continue
            first, second = distinct[:2]
            if first[0] > maximum_distance:
                continue
            ratio = first[0] / max(second[0], 1e-12)
            if ratio >= ratio_threshold:
                continue
            candidates.append(
                TransferredMatch(
                    query_index,
                    int(first[3]),
                    int(first[2]),
                    float(first[0]),
                    float(ratio),
                )
            )
        result = []
        used_query = set()
        used_points = set()
        for match in sorted(
            candidates,
            key=lambda item: (item.ratio, item.descriptor_distance),
        ):
            if match.query_index in used_query or match.point3d_id in used_points:
                continue
            used_query.add(match.query_index)
            used_points.add(match.point3d_id)
            result.append(match)
        return sorted(result, key=lambda item: item.query_index)


def load_map(path: Path):
    if not path.is_file():
        raise ReferenceGuidedError(f"Feature Map not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "point3d_ids",
            "xyz",
            "descriptors",
            "image_ids",
            "sift_uv",
        }
        missing = required - set(data.files)
        if missing:
            raise ReferenceGuidedError(
                "Feature Map arrays missing: " + ", ".join(sorted(missing))
            )
        return tuple(data[name].copy() for name in (
            "point3d_ids", "xyz", "descriptors", "image_ids", "sift_uv"
        ))


def load_query(path: Path, maximum_width: int):
    image = cv2.imread(
        str(path), cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    )
    if image is None:
        raise ReferenceGuidedError(f"Could not read query image: {path}")
    if maximum_width > 0 and image.shape[1] > maximum_width:
        scale = maximum_width / image.shape[1]
        image = cv2.resize(
            image,
            (maximum_width, round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        raise ReferenceGuidedError("No SIFT features found in query image")
    return image, keypoints, descriptors.astype(np.float32)


def merge_matches(
    results: Sequence[ReferenceResult], *, use_guided: bool = False
) -> tuple[list[TransferredMatch], int]:
    support: Counter[tuple[int, int]] = Counter()
    best = {}
    alternatives_by_query: defaultdict[int, set[int]] = defaultdict(set)
    for result in results:
        matches = (
            result.guided_transferred
            if use_guided and result.guided_transferred
            else result.transferred
        )
        for match in matches:
            key = (match.query_index, match.point3d_id)
            support[key] += 1
            alternatives_by_query[match.query_index].add(match.point3d_id)
            previous = best.get(key)
            if previous is None or match.descriptor_distance < previous.descriptor_distance:
                best[key] = match
    ordered = sorted(
        best.values(),
        key=lambda match: (
            -support[(match.query_index, match.point3d_id)],
            match.ratio,
            match.descriptor_distance,
        ),
    )
    merged = []
    used_query = set()
    used_points = set()
    for match in ordered:
        if match.query_index in used_query or match.point3d_id in used_points:
            continue
        used_query.add(match.query_index)
        used_points.add(match.point3d_id)
        merged.append(match)
    conflicts = sum(len(values) - 1 for values in alternatives_by_query.values())
    return sorted(merged, key=lambda match: match.query_index), conflicts


def deduplicate_matches(
    match_groups: Sequence[Sequence[TransferredMatch]],
) -> list[TransferredMatch]:
    """Keep one 3D point per live feature and one live feature per 3D point."""
    candidates = [match for group in match_groups for match in group]
    ordered = sorted(
        candidates,
        key=lambda match: (match.ratio, match.descriptor_distance),
    )
    selected = []
    used_query = set()
    used_points = set()
    for match in ordered:
        if match.query_index in used_query or match.point3d_id in used_points:
            continue
        used_query.add(match.query_index)
        used_points.add(match.point3d_id)
        selected.append(match)
    return sorted(selected, key=lambda match: match.query_index)


def reference_pose(reference) -> PoseResult:
    rotation = quaternion_to_rotation(np, reference.pose)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.asarray(reference.pose[4:7], dtype=np.float64).reshape(3, 1)
    return PoseResult(0, 0, 0.0, None, rvec, tvec)


def guided_refine(
    guided_matcher: GuidedMatcher,
    pose_refiner: PoseRefiner,
    keypoints,
    descriptors: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    seed_matches: Sequence[TransferredMatch],
    seed_poses: Sequence[PoseResult],
    *,
    distortion: Optional[np.ndarray] = None,
    descriptor_neighbors=None,
    radii: Sequence[float],
    ratio_threshold: float,
    maximum_distance: float,
    maximum_spatial_candidates: int,
) -> tuple[list[TransferredMatch], PoseResult]:
    best_matches = list(seed_matches)
    best_pose = pose_refiner.solve(keypoints, best_matches)
    for seed_pose in seed_poses:
        if seed_pose.rvec is None or seed_pose.tvec is None:
            continue
        current_pose = seed_pose
        current_matches = list(seed_matches)
        for radius in radii:
            if descriptor_neighbors is None:
                guided = guided_matcher.match(
                    keypoints,
                    descriptors,
                    current_pose,
                    camera_matrix,
                    image_size,
                    distortion=distortion,
                    radius=radius,
                    ratio_threshold=ratio_threshold,
                    maximum_distance=maximum_distance,
                    maximum_spatial_candidates=maximum_spatial_candidates,
                )
            else:
                guided = guided_matcher.match_from_neighbors(
                    keypoints,
                    descriptor_neighbors,
                    current_pose,
                    camera_matrix,
                    image_size,
                    distortion=distortion,
                    radius=radius,
                    ratio_threshold=ratio_threshold,
                    maximum_distance=maximum_distance,
                    maximum_spatial_candidates=maximum_spatial_candidates,
                )
            combined = deduplicate_matches((guided, seed_matches))
            solved = pose_refiner.solve(keypoints, combined)
            if solved.inliers < 4:
                break
            current_pose = solved
            current_matches = combined
            if (solved.inliers, -float(solved.mean_error or math.inf)) > (
                best_pose.inliers,
                -float(best_pose.mean_error or math.inf),
            ):
                best_matches = current_matches
                best_pose = solved
    return best_matches, best_pose


def camera_center(pose: PoseResult):
    if pose.rvec is None or pose.tvec is None:
        return None
    rotation, _ = cv2.Rodrigues(pose.rvec)
    return tuple(float(value) for value in (-rotation.T @ pose.tvec).ravel())


def pose_from_guided_result(result: ReferenceResult) -> PoseResult:
    return PoseResult(
        result.guided_correspondences,
        result.guided_inliers,
        result.guided_inlier_ratio,
        result.guided_mean_error,
        result.guided_rvec,
        result.guided_tvec,
        result.guided_inlier_indices,
    )


def write_best_result(
    output_dir: Path,
    query_image: np.ndarray,
    query_keypoints,
    result: ReferenceResult,
    pose: PoseResult,
    xyz_by_id: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    reprojection_threshold: float,
) -> None:
    if pose.rvec is None or pose.tvec is None or not result.guided_transferred:
        return
    matches = result.guided_transferred
    object_points = np.asarray(
        [xyz_by_id[match.point3d_id] for match in matches], dtype=np.float64
    )
    image_points = np.asarray(
        [query_keypoints[match.query_index].pt for match in matches], dtype=np.float64
    )
    projected, _ = cv2.projectPoints(
        object_points, pose.rvec, pose.tvec, camera_matrix, None
    )
    projected = projected.reshape((-1, 2))
    errors = np.linalg.norm(projected - image_points, axis=1)
    inlier_mask = np.zeros(len(matches), dtype=bool)
    if pose.inlier_indices:
        inlier_mask[np.asarray(pose.inlier_indices, dtype=int)] = True
    else:
        inlier_mask = errors <= reprojection_threshold
    canvas = query_image.copy()
    for point, is_inlier in zip(image_points, inlier_mask):
        color = (40, 220, 60) if is_inlier else (40, 70, 230)
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 6, color, 2, cv2.LINE_AA)
    label = (
        f"Best reference: {result.reference.image_name}  "
        f"PnP {pose.inliers}/{len(matches)}"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 850), 48), (0, 0, 0), -1)
    cv2.putText(
        canvas, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "best_guided_overlay.jpg"), canvas)
    with (output_dir / "best_guided_matches.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            (
                "query_keypoint_index", "u", "v", "point3d_id", "X", "Y", "Z",
                "descriptor_distance", "ratio", "reprojection_error", "is_inlier",
            )
        )
        for match, point_2d, point_3d, error, is_inlier in zip(
            matches, image_points, object_points, errors, inlier_mask
        ):
            writer.writerow(
                (
                    match.query_index, *point_2d.tolist(), match.point3d_id,
                    *point_3d.tolist(), match.descriptor_distance, match.ratio,
                    float(error), int(is_inlier),
                )
            )
    np.savez_compressed(
        output_dir / "best_pose.npz",
        rvec=pose.rvec,
        tvec=pose.tvec,
        camera_matrix=camera_matrix,
        image_name=np.asarray(result.reference.image_name),
        image_id=np.asarray(result.reference.image_id, dtype=np.int64),
    )


def write_match_debug(
    output_path: Path,
    query_image,
    query_keypoints,
    reference_path: Path,
    reference_uv: np.ndarray,
    result: ReferenceResult,
) -> None:
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_image is None or not result.transferred:
        return
    reference_keypoints = [
        cv2.KeyPoint(float(point[0]), float(point[1]), 3)
        for point in reference_uv
    ]
    matches = [
        cv2.DMatch(match.query_index, match.map_index, match.descriptor_distance)
        for match in result.transferred
    ]
    canvas = cv2.drawMatches(
        query_image,
        query_keypoints,
        reference_image,
        reference_keypoints,
        matches,
        None,
        matchColor=(40, 220, 70),
        singlePointColor=(80, 80, 220),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if canvas.shape[1] > 2600:
        scale = 2600 / canvas.shape[1]
        canvas = cv2.resize(
            canvas,
            (2600, round(canvas.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def run(args) -> int:
    point_ids, xyz, map_descriptors, image_ids, reference_uv = load_map(
        args.features
    )
    query_image, query_keypoints, query_descriptors = load_query(
        args.query, args.max_width
    )
    references = load_reference_images(args.images)
    references_by_id = {reference.image_id: reference for reference in references.values()}
    retriever = ReferenceRetriever(map_descriptors, image_ids)
    ranked = retriever.retrieve(
        query_descriptors,
        references_by_id,
        max(args.top_k_values),
        neighbors=args.retrieval_neighbors,
        maximum_distance=args.retrieval_distance,
    )
    if not ranked:
        raise ReferenceGuidedError("Reference retrieval returned no images")
    matcher = ReferenceMatcher(
        point_ids,
        map_descriptors,
        image_ids,
        reference_uv,
        ratio_threshold=args.ratio,
        fundamental_threshold=args.fundamental_threshold,
    )
    height, width = query_image.shape[:2]
    matrix = approximate_camera_matrix(width, height, args.fov_deg)
    pose_refiner = PoseRefiner(
        point_ids,
        xyz,
        matrix,
        ransac_threshold=args.ransac_threshold,
        iterations=args.iterations,
    )
    guided_matcher = GuidedMatcher(point_ids, xyz, map_descriptors)
    results = []
    for rank, reference in enumerate(ranked, start=1):
        raw_count, ratio_count, mutual_count, transferred = matcher.match(
            query_keypoints, query_descriptors, reference.image_id
        )
        pose = pose_refiner.solve(query_keypoints, transferred)
        result = ReferenceResult(
            rank,
            reference,
            raw_count,
            ratio_count,
            mutual_count,
            len(transferred),
            transferred,
            pose.inliers,
            pose.inlier_ratio,
            pose.mean_error,
            pose.rvec,
            pose.tvec,
        )
        results.append(result)
        print(
            f"#{rank:02d} {reference.image_name}: votes={reference.votes} "
            f"ratio={ratio_count} mutual={mutual_count} "
            f"F-inliers={len(transferred)} PnP={pose.inliers}"
        )
    guided_candidates = sorted(
        results,
        key=lambda result: (
            result.pnp_inliers,
            result.geometric_inliers,
            result.reference.votes,
        ),
        reverse=True,
    )[: min(args.guided_hypotheses, len(results))]
    print(f"Guided expansion: {len(guided_candidates)} reference hypotheses")
    for result in guided_candidates:
        reference_record = references_by_id[result.reference.image_id]
        seed_poses = [reference_pose(reference_record)]
        provisional = PoseResult(
            len(result.transferred),
            result.pnp_inliers,
            result.pnp_inlier_ratio,
            result.mean_error,
            result.rvec,
            result.tvec,
        )
        if provisional.rvec is not None and provisional.tvec is not None:
            seed_poses.insert(0, provisional)
        guided_matches, guided_pose = guided_refine(
            guided_matcher,
            pose_refiner,
            query_keypoints,
            query_descriptors,
            matrix,
            (width, height),
            result.transferred,
            seed_poses,
            radii=args.guided_radii,
            ratio_threshold=args.guided_ratio,
            maximum_distance=args.guided_distance,
            maximum_spatial_candidates=args.guided_max_candidates,
        )
        result.guided_correspondences = len(guided_matches)
        result.guided_inliers = guided_pose.inliers
        result.guided_inlier_ratio = guided_pose.inlier_ratio
        result.guided_mean_error = guided_pose.mean_error
        result.guided_transferred = guided_matches
        result.guided_rvec = guided_pose.rvec
        result.guided_tvec = guided_pose.tvec
        result.guided_inlier_indices = guided_pose.inlier_indices
        print(
            f"  #{result.rank:02d} {result.reference.image_name}: "
            f"guided={len(guided_matches)} PnP={guided_pose.inliers} "
            f"ratio={guided_pose.inlier_ratio:.3f} error={guided_pose.mean_error}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    best_result = max(
        results,
        key=lambda result: (
            result.guided_inliers,
            result.guided_inlier_ratio,
            -float(result.guided_mean_error or math.inf),
        ),
    )
    best_pose = pose_from_guided_result(best_result)
    best_passed = (
        best_pose.inliers >= args.min_inliers
        and best_pose.inlier_ratio >= args.min_inlier_ratio
    )
    print(
        f"BEST HYPOTHESIS: #{best_result.rank:02d} "
        f"{best_result.reference.image_name} PnP-inliers={best_pose.inliers} "
        f"ratio={best_pose.inlier_ratio:.3f} error={best_pose.mean_error} "
        f"status={'PASS' if best_passed else 'FAIL'}"
    )
    write_best_result(
        args.output,
        query_image,
        query_keypoints,
        best_result,
        best_pose,
        pose_refiner.xyz_by_id,
        matrix,
        args.ransac_threshold,
    )
    for result in results[: min(5, len(results))]:
        write_match_debug(
            args.output / "qc" / f"rank_{result.rank:02d}_{Path(result.reference.image_name).stem}.jpg",
            query_image,
            query_keypoints,
            args.reference_dir / result.reference.image_name,
            reference_uv,
            result,
        )
    aggregate_rows = []
    for top_k in args.top_k_values:
        selected = results[: min(top_k, len(results))]
        for mode, use_guided in (("reference", False), ("guided", True)):
            merged, conflicts = merge_matches(selected, use_guided=use_guided)
            pose = pose_refiner.solve(query_keypoints, merged)
            center = camera_center(pose)
            passed = (
                pose.inliers >= args.min_inliers
                and pose.inlier_ratio >= args.min_inlier_ratio
            )
            aggregate_rows.append(
                (mode, top_k, len(merged), conflicts, pose, passed, center)
            )
            print(
                f"{mode.upper()} TOP-{top_k}: transferred={len(merged)} "
                f"conflicts={conflicts} PnP-inliers={pose.inliers} "
                f"ratio={pose.inlier_ratio:.3f} error={pose.mean_error} "
                f"status={'PASS' if passed else 'FAIL'}"
            )
    with (args.output / "reference_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(
            (
                "rank", "image_id", "image_name", "retrieval_votes", "retrieval_score",
                "raw_matches", "ratio_matches", "mutual_matches", "geometric_inliers",
                "pnp_inliers", "pnp_inlier_ratio", "mean_reprojection_error",
                "guided_correspondences", "guided_inliers", "guided_inlier_ratio",
                "guided_mean_reprojection_error",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result.rank, result.reference.image_id, result.reference.image_name,
                    result.reference.votes, result.reference.score, result.raw_matches,
                    result.ratio_matches, result.mutual_matches, result.geometric_inliers,
                    result.pnp_inliers, result.pnp_inlier_ratio,
                    "" if result.mean_error is None else result.mean_error,
                    result.guided_correspondences, result.guided_inliers,
                    result.guided_inlier_ratio,
                    "" if result.guided_mean_error is None else result.guided_mean_error,
                )
            )
    lines = [
        f"Query: {args.query}",
        f"Query size: {width}x{height}",
        f"Query SIFT keypoints: {len(query_keypoints)}",
        f"Retrieved references: {len(results)}",
        f"Lowe ratio: {args.ratio}",
        f"Fundamental RANSAC threshold: {args.fundamental_threshold}",
        f"PnP RANSAC threshold: {args.ransac_threshold}",
        f"Guided hypotheses: {args.guided_hypotheses}",
        f"Guided radii: {', '.join(str(value) for value in args.guided_radii)}",
        f"Guided Lowe ratio: {args.guided_ratio}",
        "",
        "Best pose hypothesis:",
        f"Reference rank: {best_result.rank}",
        f"Reference image: {best_result.reference.image_name}",
        f"Correspondences: {best_pose.correspondences}",
        f"PnP inliers: {best_pose.inliers}",
        f"PnP inlier ratio: {best_pose.inlier_ratio:.6f}",
        f"Mean inlier reprojection error: {best_pose.mean_error}",
        f"Camera center: {camera_center(best_pose)}",
        f"Status: {'PASS' if best_passed else 'FAIL'}",
        "",
        "Aggregate results:",
    ]
    for mode, top_k, transferred, conflicts, pose, passed, center in aggregate_rows:
        lines.append(
            f"{mode} Top-{top_k}: transferred={transferred}, conflicts={conflicts}, "
            f"inliers={pose.inliers}, inlier_ratio={pose.inlier_ratio:.6f}, "
            f"mean_error={pose.mean_error}, camera_center={center}, "
            f"status={'PASS' if passed else 'FAIL'}"
        )
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k-values", nargs="+", type=int, default=(1, 5, 10, 20))
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--retrieval-neighbors", type=int, default=8)
    parser.add_argument("--retrieval-distance", type=float, default=300.0)
    parser.add_argument("--ratio", type=float, default=0.80)
    parser.add_argument("--fundamental-threshold", type=float, default=5.0)
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--ransac-threshold", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--guided-hypotheses", type=int, default=8)
    parser.add_argument(
        "--guided-radii", nargs="+", type=float, default=(160.0, 80.0, 40.0)
    )
    parser.add_argument("--guided-ratio", type=float, default=0.82)
    parser.add_argument("--guided-distance", type=float, default=280.0)
    parser.add_argument("--guided-max-candidates", type=int, default=80)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.top_k_values or min(args.top_k_values) <= 0:
            raise ReferenceGuidedError("Top-K values must be positive")
        if args.guided_hypotheses <= 0:
            raise ReferenceGuidedError("--guided-hypotheses must be positive")
        if not args.guided_radii or min(args.guided_radii) <= 0:
            raise ReferenceGuidedError("--guided-radii values must be positive")
        if not 0 < args.guided_ratio < 1:
            raise ReferenceGuidedError("--guided-ratio must be between 0 and 1")
        return run(args)
    except (ReferenceGuidedError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
