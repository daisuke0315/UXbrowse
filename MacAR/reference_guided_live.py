#!/usr/bin/env python3
"""Reference-guided live 2D-3D correspondence generation for realtime_pnp."""

from __future__ import annotations

import math
import concurrent.futures
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from offline_pnp_test import MapData, load_reference_images
from realtime_pnp_server import LivePnPError, LiveResult, LocalizationDebug
from reference_guided_pnp import (
    GuidedMatcher,
    PoseRefiner,
    PoseResult,
    ReferenceResult,
    ReferenceMatcher,
    ReferenceRetriever,
    ReferenceScore,
    TransferredMatch,
    guided_refine,
    load_map,
    reference_pose,
)


class FastReferenceMatcher:
    """Retrieve and geometrically verify references from one global FLANN pass."""

    def __init__(
        self,
        point3d_ids: np.ndarray,
        descriptors: np.ndarray,
        image_ids: np.ndarray,
        reference_uv: np.ndarray,
        references_by_id: dict,
    ) -> None:
        self.point3d_ids = point3d_ids
        self.descriptors = descriptors.astype(np.float32, copy=False)
        self.image_ids = image_ids
        self.reference_uv = reference_uv
        self.references_by_id = references_by_id
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=96)
        )
        self.matcher.add([self.descriptors])
        self.matcher.train()

    def match(
        self,
        keypoints,
        query_descriptors: np.ndarray,
        *,
        shortlist: int,
        neighbors: int,
        maximum_distance: float,
        ratio_threshold: float,
        fundamental_threshold: float,
    ) -> list[ReferenceResult]:
        actual_k = min(neighbors, len(self.descriptors))
        raw = self.matcher.knnMatch(
            query_descriptors.astype(np.float32, copy=False), k=actual_k
        )
        votes: Counter[int] = Counter()
        scores: defaultdict[int, float] = defaultdict(float)
        candidates: defaultdict[int, list[tuple[int, list]]] = defaultdict(list)
        for query_index, neighbors_for_query in enumerate(raw):
            by_image: defaultdict[int, list] = defaultdict(list)
            for match in neighbors_for_query:
                if match.distance > maximum_distance:
                    continue
                image_id = int(self.image_ids[match.trainIdx])
                if image_id not in self.references_by_id:
                    continue
                by_image[image_id].append(match)
            for image_id, image_matches in by_image.items():
                votes[image_id] += 1
                scores[image_id] += max(
                    0.0,
                    1.0 - float(image_matches[0].distance) / maximum_distance,
                )
                candidates[image_id].append((query_index, image_matches))
        ranked_ids = sorted(
            votes,
            key=lambda image_id: (votes[image_id], scores[image_id]),
            reverse=True,
        )[:shortlist]
        results = []
        for rank, image_id in enumerate(ranked_ids, start=1):
            ratio_candidates = []
            for query_index, image_matches in candidates[image_id]:
                distinct = []
                seen_points = set()
                for match in image_matches:
                    point_id = int(self.point3d_ids[match.trainIdx])
                    if point_id in seen_points:
                        continue
                    seen_points.add(point_id)
                    distinct.append((match, point_id))
                    if len(distinct) == 2:
                        break
                if len(distinct) < 2:
                    continue
                first, second = distinct
                if first[0].distance >= ratio_threshold * second[0].distance:
                    continue
                ratio_candidates.append(
                    TransferredMatch(
                        query_index,
                        int(first[0].trainIdx),
                        int(first[1]),
                        float(first[0].distance),
                        float(first[0].distance / max(second[0].distance, 1e-12)),
                    )
                )
            unique = []
            used_query = set()
            used_points = set()
            for match in sorted(
                ratio_candidates,
                key=lambda item: (item.ratio, item.descriptor_distance),
            ):
                if match.query_index in used_query or match.point3d_id in used_points:
                    continue
                used_query.add(match.query_index)
                used_points.add(match.point3d_id)
                unique.append(match)
            transferred = []
            if len(unique) >= 8:
                query_points = np.asarray(
                    [keypoints[match.query_index].pt for match in unique],
                    dtype=np.float64,
                )
                reference_points = np.asarray(
                    [self.reference_uv[match.map_index] for match in unique],
                    dtype=np.float64,
                )
                _, mask = cv2.findFundamentalMat(
                    query_points,
                    reference_points,
                    cv2.FM_RANSAC,
                    fundamental_threshold,
                    0.999,
                    5000,
                )
                if mask is not None:
                    transferred = [
                        match
                        for keep, match in zip(mask.ravel().astype(bool), unique)
                        if keep
                    ]
            reference = self.references_by_id[image_id]
            results.append(
                ReferenceResult(
                    rank,
                    ReferenceScore(
                        image_id,
                        reference.name,
                        votes[image_id],
                        scores[image_id],
                    ),
                    len(candidates[image_id]),
                    len(ratio_candidates),
                    len(unique),
                    len(transferred),
                    transferred,
                )
            )
        return results


class ReferenceGuidedLiveLocalizer:
    def __init__(
        self,
        feature_path: Path,
        images_path: Path,
        *,
        shortlist: int = 20,
        retrieval_neighbors: int = 8,
        retrieval_distance: float = 300.0,
        ratio_threshold: float = 0.82,
        fundamental_threshold: float = 5.0,
        guided_hypotheses: int = 4,
        guided_radii: Sequence[float] = (160.0, 80.0, 40.0),
        guided_ratio: float = 0.82,
        guided_distance: float = 280.0,
        guided_max_candidates: int = 80,
        ransac_threshold: float = 8.0,
        iterations: int = 10_000,
        min_inliers: int = 20,
        min_inlier_ratio: float = 0.2,
        matching_workers: int = 4,
        mpr_renderer=None,
    ) -> None:
        point_ids, xyz, descriptors, image_ids, reference_uv = load_map(feature_path)
        references = load_reference_images(images_path)
        references_by_id = {
            reference.image_id: reference for reference in references.values()
        }
        self.map_data = MapData(point_ids, xyz, descriptors, image_ids)
        self.references_by_id = references_by_id
        self.retriever = ReferenceRetriever(descriptors, image_ids)
        self.reference_matcher = ReferenceMatcher(
            point_ids,
            descriptors,
            image_ids,
            reference_uv,
            ratio_threshold=ratio_threshold,
            fundamental_threshold=fundamental_threshold,
        )
        self.guided_matcher = GuidedMatcher(point_ids, xyz, descriptors)
        cv2.setRNGSeed(0)
        self.guided_descriptor_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=96)
        )
        self.guided_descriptor_matcher.add(
            [descriptors.astype(np.float32, copy=False)]
        )
        self.guided_descriptor_matcher.train()
        self.shortlist = shortlist
        self.retrieval_neighbors = retrieval_neighbors
        self.retrieval_distance = retrieval_distance
        self.ratio_threshold = ratio_threshold
        self.fundamental_threshold = fundamental_threshold
        self.guided_hypotheses = guided_hypotheses
        self.guided_radii = tuple(guided_radii)
        self.guided_ratio = guided_ratio
        self.guided_distance = guided_distance
        self.guided_max_candidates = guided_max_candidates
        self.ransac_threshold = ransac_threshold
        self.iterations = iterations
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.matching_workers = matching_workers
        self.mpr_renderer = mpr_renderer
        self.sift = cv2.SIFT_create()
        self.lock = threading.Lock()
        self.last_debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )
        self.last_reference_name = "—"
        self.last_candidate_summary = ""

    def localize(self, image, camera_matrix, distortion):
        with self.lock:
            return self._localize_locked(image, camera_matrix, distortion)

    def _failure(
        self,
        image,
        started: float,
        detail: str,
        *,
        keypoints: int = 0,
        matches: int = 0,
        sift_ms: float = 0.0,
        matching_ms: float = 0.0,
        pnp_ms: float = 0.0,
    ):
        self.last_debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )
        result = LiveResult(
            "FAIL", detail, keypoints, matches, 0, 0.0, None, None, None, None,
            (time.perf_counter() - started) * 1000.0, "HIDDEN",
            sift_ms, matching_ms, pnp_ms, 0.0,
        )
        return result, self._decorate(image, result, (), frozenset(), None)

    def _localize_locked(self, image, camera_matrix, distortion):
        started = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        sift_done = time.perf_counter()
        sift_ms = (sift_done - started) * 1000.0
        if descriptors is None or not keypoints:
            return self._failure(
                image, started, "NO_SIFT_FEATURES", sift_ms=sift_ms
            )
        ranked = self.retriever.retrieve(
            descriptors,
            self.references_by_id,
            self.shortlist,
            neighbors=self.retrieval_neighbors,
            maximum_distance=self.retrieval_distance,
        )
        if not ranked:
            return self._failure(
                image, started, "NO_REFERENCE_CANDIDATES",
                keypoints=len(keypoints), sift_ms=sift_ms,
            )

        def match_reference(item):
            rank, reference = item
            raw_count, ratio_count, mutual_count, transferred = (
                self.reference_matcher.match(
                    keypoints, descriptors, reference.image_id
                )
            )
            return ReferenceResult(
                rank,
                reference,
                raw_count,
                ratio_count,
                mutual_count,
                len(transferred),
                transferred,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.matching_workers
        ) as executor:
            results = list(executor.map(match_reference, enumerate(ranked, start=1)))
        initial_matching_done = time.perf_counter()
        matching_ms = (initial_matching_done - sift_done) * 1000.0
        if not results:
            return self._failure(
                image,
                started,
                "NO_REFERENCE_CANDIDATES",
                keypoints=len(keypoints),
                sift_ms=sift_ms,
                matching_ms=matching_ms,
            )
        pose_refiner = PoseRefiner(
            self.map_data.point3d_ids,
            self.map_data.xyz,
            camera_matrix,
            ransac_threshold=self.ransac_threshold,
            iterations=self.iterations,
            distortion=distortion,
        )
        guided_neighbors = self.guided_descriptor_matcher.knnMatch(
            descriptors.astype(np.float32, copy=False),
            k=min(self.guided_max_candidates, len(self.map_data.descriptors)),
        )
        for result in results:
            pose = pose_refiner.solve(keypoints, result.transferred)
            result.pnp_inliers = pose.inliers
            result.pnp_inlier_ratio = pose.inlier_ratio
            result.mean_error = pose.mean_error
            result.rvec = pose.rvec
            result.tvec = pose.tvec
        candidates = sorted(
            results,
            key=lambda result: (
                result.pnp_inliers,
                result.geometric_inliers,
                result.reference.votes,
            ),
            reverse=True,
        )[: min(self.guided_hypotheses, len(results))]
        radius_scale = image.shape[1] / 1920.0
        radii = tuple(max(8.0, radius * radius_scale) for radius in self.guided_radii)
        def refine_candidate(result):
            reference_record = self.references_by_id[result.reference.image_id]
            seeds = [reference_pose(reference_record)]
            provisional = PoseResult(
                len(result.transferred),
                result.pnp_inliers,
                result.pnp_inlier_ratio,
                result.mean_error,
                result.rvec,
                result.tvec,
            )
            if provisional.rvec is not None and provisional.tvec is not None:
                seeds.insert(0, provisional)
            matches, pose = guided_refine(
                self.guided_matcher,
                pose_refiner,
                keypoints,
                descriptors,
                camera_matrix,
                (image.shape[1], image.shape[0]),
                result.transferred,
                seeds,
                distortion=distortion,
                descriptor_neighbors=guided_neighbors,
                radii=radii,
                ratio_threshold=self.guided_ratio,
                maximum_distance=self.guided_distance,
                maximum_spatial_candidates=self.guided_max_candidates,
            )
            result.guided_transferred = matches
            result.guided_correspondences = len(matches)
            result.guided_inliers = pose.inliers
            result.guided_inlier_ratio = pose.inlier_ratio
            result.guided_mean_error = pose.mean_error
            result.guided_rvec = pose.rvec
            result.guided_tvec = pose.tvec
            result.guided_inlier_indices = pose.inlier_indices
            return result

        # Keep RANSAC pose hypotheses sequential: OpenCV's RNG is process-global,
        # so concurrent solvePnPRansac calls make the selected pose unstable.
        candidates = [refine_candidate(result) for result in candidates]
        self.last_candidate_summary = ", ".join(
            f"{result.reference.image_name}:{result.guided_inliers}/"
            f"{result.guided_correspondences}"
            for result in candidates
        )
        print(f"Reference hypotheses: {self.last_candidate_summary}")
        best = max(
            candidates,
            key=lambda result: (
                result.guided_inliers,
                result.guided_inlier_ratio,
                -float(result.guided_mean_error or math.inf),
            ),
        )
        matches = best.guided_transferred or best.transferred
        pose = PoseResult(
            len(matches),
            best.guided_inliers,
            best.guided_inlier_ratio,
            best.guided_mean_error,
            best.guided_rvec,
            best.guided_tvec,
            best.guided_inlier_indices,
        )
        if pose.rvec is None or pose.tvec is None or pose.inliers < 4:
            return self._failure(
                image,
                started,
                f"REFERENCE_PNP_FAILED_{best.reference.image_name}",
                keypoints=len(keypoints),
                matches=len(matches),
                sift_ms=sift_ms,
                matching_ms=matching_ms,
                pnp_ms=(time.perf_counter() - initial_matching_done) * 1000.0,
            )
        accepted = (
            pose.inliers >= self.min_inliers
            and pose.inlier_ratio >= self.min_inlier_ratio
        )
        object_points = np.asarray(
            [pose_refiner.xyz_by_id[match.point3d_id] for match in matches],
            dtype=np.float64,
        )
        rotation, _ = cv2.Rodrigues(pose.rvec)
        camera_center = (-rotation.T @ pose.tvec).reshape(3)
        self.last_debug = LocalizationDebug(
            object_points.copy(),
            frozenset(pose.inlier_indices),
            tuple(float(value) for value in camera_center),
            True,
        )
        self.last_reference_name = best.reference.image_name
        canvas = image.copy()
        mpr_status = "HIDDEN"
        mpr_started = time.perf_counter()
        if accepted and self.mpr_renderer is not None:
            canvas, mpr_status = self.mpr_renderer.render(
                canvas, camera_matrix, distortion, pose.rvec, pose.tvec
            )
        mpr_ms = (time.perf_counter() - mpr_started) * 1000.0
        pnp_ms = (mpr_started - initial_matching_done) * 1000.0
        result = LiveResult(
            "PASS" if accepted else "FAIL",
            f"REF {best.reference.image_name}",
            len(keypoints),
            len(matches),
            pose.inliers,
            pose.inlier_ratio,
            pose.mean_error,
            tuple(float(value) for value in pose.rvec.ravel()),
            tuple(float(value) for value in pose.tvec.ravel()),
            tuple(float(value) for value in camera_center),
            (time.perf_counter() - started) * 1000.0,
            mpr_status,
            sift_ms,
            matching_ms,
            pnp_ms,
            mpr_ms,
        )
        return result, self._decorate(
            canvas,
            result,
            matches,
            frozenset(pose.inlier_indices),
            keypoints,
        )

    @staticmethod
    def _decorate(image, result, matches, inliers, keypoints):
        canvas = image.copy()
        if keypoints is not None:
            for index, match in enumerate(matches):
                u, v = keypoints[match.query_index].pt
                color = (40, 230, 70) if index in inliers else (40, 40, 230)
                cv2.circle(canvas, (round(u), round(v)), 3, color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
        color = (80, 230, 80) if result.status == "PASS" else (70, 70, 240)
        cv2.putText(
            canvas,
            f"{result.status}  reference-guided  matches {result.matches}  "
            f"inliers {result.inliers}  MPR {result.mpr_status}",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{result.detail}  {result.processing_ms / 1000.0:.1f} s",
            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (240, 240, 240), 1, cv2.LINE_AA,
        )
        return canvas
