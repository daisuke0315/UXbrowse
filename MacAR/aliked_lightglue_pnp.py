#!/usr/bin/env python3
"""ALIKED + LightGlue reference-guided 2D-3D matching and PnP PoC.

The current PoC uses registered RC/COLMAP images as reference views.  A
LightGlue match on the reference side is associated with the nearest existing
COLMAP object observation, which transfers its POINT3D_ID and XYZ to the query
keypoint.  The final virtual-view implementation can replace that association
with a per-pixel GLB XYZ buffer without changing the PnP stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from offline_pnp_test import (
    PnPTestError,
    camera_matrix_and_distortion,
    load_camera_parameters,
    load_reference_images,
    pose_errors,
)
from realtime_pnp import approximate_camera_matrix
from reference_guided_pnp import ReferenceRetriever, load_map


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    "/Users/daisuketanii/Library/CloudStorage/GoogleDrive-"
    "mp.daisuketanii@gmail.com/マイドライブ/010_splat/data/260917"
)
DEFAULT_COLMAP = DEFAULT_DATA_ROOT / "colmap"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "output_feature_map"


class LearnedMatchingError(RuntimeError):
    """Expected data, dependency, or localization error."""


@dataclass(frozen=True)
class Observation:
    point3d_id: int
    uv: tuple[float, float]
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Transfer:
    query_index: int
    reference_index: int
    point3d_id: int
    xyz: tuple[float, float, float]
    query_uv: tuple[float, float]
    reference_uv: tuple[float, float]
    observation_uv: tuple[float, float]
    observation_distance: float
    match_score: float


@dataclass
class ReferenceEvaluation:
    rank: int
    image_id: int
    image_name: str
    retrieval_votes: int
    retrieval_score: float
    query_keypoints: int = 0
    reference_keypoints: int = 0
    lightglue_matches: int = 0
    geometric_inliers: int = 0
    transferred_points: int = 0
    pnp_inliers: int = 0
    pnp_inlier_ratio: float = 0.0
    mean_reprojection_error: Optional[float] = None
    elapsed_ms: float = 0.0
    status: str = "FAIL"
    detail: str = ""


@dataclass(frozen=True)
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray
    inlier_indices: np.ndarray
    mean_error: float


def load_observations(path: Path) -> Dict[str, list[Observation]]:
    if not path.is_file():
        raise LearnedMatchingError(f"object_observations.csv not found: {path}")
    required = {"image_name", "u", "v", "point3d_id", "X", "Y", "Z"}
    grouped: Dict[str, list[Observation]] = defaultdict(list)
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise LearnedMatchingError(
                    "object_observations.csv is missing columns: "
                    + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, 2):
                try:
                    image_name = row["image_name"].strip()
                    observation = Observation(
                        int(row["point3d_id"]),
                        (float(row["u"]), float(row["v"])),
                        (float(row["X"]), float(row["Y"]), float(row["Z"])),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise LearnedMatchingError(
                        f"Invalid observation at CSV row {row_number}"
                    ) from exc
                values = (*observation.uv, *observation.xyz)
                if not image_name or not all(math.isfinite(value) for value in values):
                    raise LearnedMatchingError(
                        f"Invalid observation at CSV row {row_number}"
                    )
                grouped[image_name].append(observation)
    except (OSError, UnicodeError) as exc:
        raise LearnedMatchingError(f"Could not read observations: {exc}") from exc
    if not grouped:
        raise LearnedMatchingError("object_observations.csv contains no rows")
    return dict(grouped)


def associate_reference_observations(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    match_scores: np.ndarray,
    observations: Sequence[Observation],
    maximum_distance: float,
) -> list[Transfer]:
    """Associate learned reference keypoints with sparse COLMAP observations."""
    if maximum_distance <= 0:
        raise ValueError("maximum_distance must be positive")
    if not observations or len(reference_points) == 0:
        return []
    obs_uv = np.asarray([observation.uv for observation in observations], np.float64)
    delta = reference_points[:, None, :].astype(np.float64) - obs_uv[None, :, :]
    distance_squared = np.einsum("nmi,nmi->nm", delta, delta)
    nearest = np.argmin(distance_squared, axis=1)
    transfers = []
    for query_index, reference_index, observation_index, distance2, score in zip(
        range(len(query_points)),
        range(len(reference_points)),
        nearest.tolist(),
        distance_squared[np.arange(len(nearest)), nearest].tolist(),
        match_scores.tolist(),
    ):
        distance = math.sqrt(float(distance2))
        if distance > maximum_distance:
            continue
        observation = observations[int(observation_index)]
        transfers.append(
            Transfer(
                int(query_index),
                int(reference_index),
                observation.point3d_id,
                observation.xyz,
                tuple(float(value) for value in query_points[query_index]),
                tuple(float(value) for value in reference_points[reference_index]),
                observation.uv,
                distance,
                float(score),
            )
        )
    return deduplicate_transfers(transfers)


def deduplicate_transfers(transfers: Iterable[Transfer]) -> list[Transfer]:
    """Keep a one-to-one query-keypoint/POINT3D_ID assignment."""
    ordered = sorted(
        transfers,
        key=lambda item: (-item.match_score, item.observation_distance),
    )
    result = []
    used_query = set()
    used_points = set()
    for transfer in ordered:
        if transfer.query_index in used_query or transfer.point3d_id in used_points:
            continue
        used_query.add(transfer.query_index)
        used_points.add(transfer.point3d_id)
        result.append(transfer)
    return result


def geometric_filter(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(query_points) < 8 or threshold <= 0:
        return query_points, reference_points, scores
    _, mask = cv2.findFundamentalMat(
        query_points.astype(np.float64),
        reference_points.astype(np.float64),
        cv2.FM_RANSAC,
        threshold,
        0.999,
        10_000,
    )
    if mask is None:
        return query_points[:0], reference_points[:0], scores[:0]
    keep = mask.ravel().astype(bool)
    return query_points[keep], reference_points[keep], scores[keep]


def solve_pose(
    transfers: Sequence[Transfer],
    camera_matrix: np.ndarray,
    distortion: Optional[np.ndarray],
    threshold: float,
    iterations: int,
    *,
    validate: bool = True,
) -> Optional[Pose]:
    if len(transfers) < 4:
        return None
    object_points = np.asarray([item.xyz for item in transfers], np.float64)
    image_points = np.asarray([item.query_uv for item in transfers], np.float64)
    cv2.setRNGSeed(0)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        iterationsCount=iterations,
        reprojectionError=threshold,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None
    indices = inliers.ravel().astype(int)
    initial_rvec, initial_tvec = rvec.copy(), tvec.copy()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[indices],
            image_points[indices],
            camera_matrix,
            distortion,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    if validate:
        # LM can diverge for nearly planar or mismatched support. Keep the
        # RANSAC estimate unless refinement improves its original support.
        initial_projection, _ = cv2.projectPoints(object_points[indices],
            initial_rvec, initial_tvec, camera_matrix, distortion)
        refined_projection, _ = cv2.projectPoints(object_points[indices],
            rvec, tvec, camera_matrix, distortion)
        initial_error = np.linalg.norm(initial_projection.reshape(-1, 2)-image_points[indices], axis=1)
        refined_error = np.linalg.norm(refined_projection.reshape(-1, 2)-image_points[indices], axis=1)
        if not np.isfinite(refined_error).all() or np.mean(refined_error**2) > np.mean(initial_error**2):
            rvec, tvec = initial_rvec, initial_tvec
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    errors = np.linalg.norm(projected.reshape((-1, 2)) - image_points, axis=1)
    if validate:
        rotation = cv2.Rodrigues(rvec)[0]
        depth = (np.einsum('ni,ji->nj', object_points, rotation) + tvec.reshape(3))[:, 2]
        indices = np.flatnonzero(np.isfinite(errors) & (errors <= threshold) & (depth > 0))
        if len(indices) < 4:
            return None
    return Pose(rvec, tvec, indices, float(np.mean(errors[indices])))


def _load_models(max_keypoints: int, match_threshold: float, device_name: str):
    os.environ.setdefault("TORCH_HOME", str(SCRIPT_DIR / ".model_cache" / "torch"))
    try:
        import torch
        from lightglue import ALIKED, LightGlue
    except ImportError as exc:
        raise LearnedMatchingError(
            "ALIKED/LightGlue dependencies are missing. Run: "
            "python3 -m pip install -r requirements-learned-matching.txt"
        ) from exc
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name not in {"cpu", "mps", "cuda"}:
        raise LearnedMatchingError(f"Unsupported device: {device_name}")
    device = torch.device(device_name)
    try:
        extractor = ALIKED(
            max_num_keypoints=max_keypoints,
            detection_threshold=0,
        ).eval().to(device)
        matcher = LightGlue(
            features="aliked",
            filter_threshold=match_threshold,
        ).eval().to(device)
    except Exception as exc:
        raise LearnedMatchingError(
            "Could not initialize ALIKED/LightGlue. The first run needs network "
            "access to download official weights."
        ) from exc
    return torch, extractor, matcher, device


def _extract(extractor, tensor, resize: int):
    preprocess = {} if resize <= 0 else {"resize": resize}
    return extractor.extract(tensor, **preprocess)


def _image_tensor(torch, bgr: np.ndarray, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    data = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(data).float().div_(255.0).to(device)


def _match_pair(torch, matcher, query_features, reference_features):
    from lightglue.utils import rbd

    with torch.inference_mode():
        result = matcher({"image0": query_features, "image1": reference_features})
    query = rbd(query_features)
    reference = rbd(reference_features)
    result = rbd(result)
    pairs = result["matches"].detach().cpu().numpy().astype(np.int64)
    scores = result["scores"].detach().cpu().numpy().astype(np.float64)
    query_keypoints = query["keypoints"].detach().cpu().numpy()
    reference_keypoints = reference["keypoints"].detach().cpu().numpy()
    return (
        query_keypoints,
        reference_keypoints,
        pairs,
        scores,
    )


def _choose_references(args, references_by_name, feature_arrays, query_bgr):
    references_by_id = {item.image_id: item for item in references_by_name.values()}
    if args.reference_name:
        selected = []
        for rank, name in enumerate(args.reference_name, 1):
            reference = references_by_name.get(name)
            if reference is None:
                raise LearnedMatchingError(f"Reference is absent from Images.txt: {name}")
            selected.append((rank, reference, 0, 0.0))
        return selected

    _, _, descriptors, image_ids, _ = feature_arrays
    gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    _, query_descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if query_descriptors is None:
        raise LearnedMatchingError("No SIFT descriptors found for reference retrieval")
    scores = ReferenceRetriever(descriptors, image_ids).retrieve(
        query_descriptors.astype(np.float32),
        references_by_id,
        args.top_k + 1,
        neighbors=args.retrieval_neighbors,
        maximum_distance=args.retrieval_distance,
    )
    query_name = args.query.name
    scores = [score for score in scores if score.image_name != query_name][: args.top_k]
    if not scores:
        raise LearnedMatchingError("Reference retrieval returned no candidate images")
    return [
        (rank, references_by_id[score.image_id], score.votes, score.score)
        for rank, score in enumerate(scores, 1)
    ]


def _camera_for_query(args, query_bgr, references_by_name):
    height, width = query_bgr.shape[:2]
    if args.fov_deg <= 0 or args.fov_deg >= 180:
        raise LearnedMatchingError("--fov-deg must be between 0 and 180")
    query_reference = references_by_name.get(args.query.name)
    camera_id = args.camera_id or (
        query_reference.camera_id if query_reference is not None else None
    )
    if camera_id is None:
        return approximate_camera_matrix(width, height, args.fov_deg), None, None
    cameras, parameters = load_camera_parameters(args.cameras)
    if camera_id not in cameras:
        raise LearnedMatchingError(f"CAMERA_ID {camera_id} not found")
    camera = cameras[camera_id]
    matrix, distortion = camera_matrix_and_distortion(
        np, camera, parameters[camera_id]
    )
    if (width, height) != (camera.width, camera.height):
        matrix = matrix.copy()
        matrix[0, :] *= width / camera.width
        matrix[1, :] *= height / camera.height
    return matrix, distortion, query_reference


def _draw_debug(
    query: np.ndarray,
    reference: np.ndarray,
    transfers: Sequence[Transfer],
    pose: Optional[Pose],
    output: Path,
) -> None:
    target_height = min(900, max(query.shape[0], reference.shape[0]))

    def resize(image):
        scale = target_height / image.shape[0]
        return cv2.resize(image, (round(image.shape[1] * scale), target_height)), scale

    left, left_scale = resize(query)
    right, right_scale = resize(reference)
    canvas = np.hstack((left, right))
    inliers = set() if pose is None else set(int(i) for i in pose.inlier_indices)
    for index, transfer in enumerate(transfers[:300]):
        q = tuple(round(value * left_scale) for value in transfer.query_uv)
        r = (
            round(transfer.reference_uv[0] * right_scale) + left.shape[1],
            round(transfer.reference_uv[1] * right_scale),
        )
        color = (0, 255, 0) if index in inliers else (0, 180, 255)
        cv2.circle(canvas, q, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, r, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, q, r, color, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise LearnedMatchingError(f"Could not write debug image: {output}")


def _write_summary(
    output_dir: Path,
    evaluations: Sequence[ReferenceEvaluation],
    best: Optional[tuple[ReferenceEvaluation, Pose]],
    ground_truth_errors,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reference_results.csv"
    fields = list(asdict(evaluations[0]).keys()) if evaluations else []
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(asdict(evaluation))
    payload = {
        "matching": "ALIKED + LightGlue",
        "reference_source": "RC/COLMAP registered images (PoC)",
        "evaluations": [asdict(item) for item in evaluations],
        "best_reference": None if best is None else best[0].image_name,
        "best_pose": None,
        "ground_truth_error": ground_truth_errors,
    }
    if best is not None:
        evaluation, pose = best
        rotation, _ = cv2.Rodrigues(pose.rvec)
        center = -rotation.T @ pose.tvec
        payload["best_pose"] = {
            "rvec": pose.rvec.ravel().tolist(),
            "tvec": pose.tvec.ravel().tolist(),
            "camera_center": center.ravel().tolist(),
            "inliers": len(pose.inlier_indices),
            "inlier_ratio": evaluation.pnp_inlier_ratio,
            "mean_reprojection_error": pose.mean_error,
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)


def run(args) -> int:
    for path, label in (
        (args.query, "query image"),
        (args.features, "Feature Map"),
        (args.images, "Images.txt"),
        (args.cameras, "Cameras.txt"),
    ):
        if not path.is_file():
            raise LearnedMatchingError(f"{label} not found: {path}")
    query_bgr = cv2.imread(str(args.query), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise LearnedMatchingError(f"Could not read query image: {args.query}")
    references_by_name = load_reference_images(args.images)
    observations = load_observations(args.observations)
    feature_arrays = load_map(args.features)
    selected = _choose_references(
        args, references_by_name, feature_arrays, query_bgr
    )
    matrix, distortion, query_reference = _camera_for_query(
        args, query_bgr, references_by_name
    )
    torch, extractor, matcher, device = _load_models(
        args.max_keypoints, args.match_threshold, args.device
    )
    print(f"Device: {device}")
    print(f"Query: {args.query} ({query_bgr.shape[1]}x{query_bgr.shape[0]})")
    print("Candidates: " + ", ".join(item[1].name for item in selected))
    with torch.inference_mode():
        query_features = _extract(
            extractor, _image_tensor(torch, query_bgr, device), args.resize
        )

    evaluations: list[ReferenceEvaluation] = []
    poses: list[Optional[Pose]] = []
    debug_inputs = []
    for rank, reference_info, votes, retrieval_score in selected:
        started = time.perf_counter()
        evaluation = ReferenceEvaluation(
            rank,
            reference_info.image_id,
            reference_info.name,
            votes,
            retrieval_score,
        )
        reference_path = args.reference_dir / reference_info.name
        reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference_bgr is None:
            evaluation.detail = f"Reference image not found: {reference_path}"
            evaluation.elapsed_ms = (time.perf_counter() - started) * 1000
            evaluations.append(evaluation)
            poses.append(None)
            debug_inputs.append(None)
            continue
        reference_observations = observations.get(reference_info.name, ())
        if not reference_observations:
            evaluation.detail = "No object observations for reference"
            evaluation.elapsed_ms = (time.perf_counter() - started) * 1000
            evaluations.append(evaluation)
            poses.append(None)
            debug_inputs.append(None)
            continue
        with torch.inference_mode():
            reference_features = _extract(
                extractor,
                _image_tensor(torch, reference_bgr, device),
                args.resize,
            )
            query_kp, reference_kp, pairs, match_scores = _match_pair(
                torch, matcher, query_features, reference_features
            )
        evaluation.query_keypoints = len(query_kp)
        evaluation.reference_keypoints = len(reference_kp)
        evaluation.lightglue_matches = len(pairs)
        if len(pairs):
            matched_query = query_kp[pairs[:, 0]]
            matched_reference = reference_kp[pairs[:, 1]]
            matched_query, matched_reference, match_scores = geometric_filter(
                matched_query,
                matched_reference,
                match_scores,
                args.fundamental_threshold,
            )
        else:
            matched_query = np.empty((0, 2), np.float32)
            matched_reference = np.empty((0, 2), np.float32)
        evaluation.geometric_inliers = len(matched_query)
        transfers = associate_reference_observations(
            matched_query,
            matched_reference,
            match_scores,
            reference_observations,
            args.observation_radius,
        )
        evaluation.transferred_points = len(transfers)
        pose = solve_pose(
            transfers,
            matrix,
            distortion,
            args.ransac_threshold,
            args.iterations,
        )
        if pose is not None:
            evaluation.pnp_inliers = len(pose.inlier_indices)
            evaluation.pnp_inlier_ratio = evaluation.pnp_inliers / max(
                1, len(transfers)
            )
            evaluation.mean_reprojection_error = pose.mean_error
            passed = (
                evaluation.pnp_inliers >= args.min_inliers
                and evaluation.pnp_inlier_ratio >= args.min_inlier_ratio
                and pose.mean_error <= args.max_reprojection_error
            )
            evaluation.status = "PASS" if passed else "REJECT"
            evaluation.detail = "quality gate passed" if passed else "quality gate failed"
        else:
            evaluation.detail = "PnP could not estimate a pose"
        evaluation.elapsed_ms = (time.perf_counter() - started) * 1000
        evaluations.append(evaluation)
        poses.append(pose)
        debug_inputs.append((reference_bgr, transfers))
        error_text = (
            "-"
            if evaluation.mean_reprojection_error is None
            else f"{evaluation.mean_reprojection_error:.2f}px"
        )
        print(
            f"[{rank}] {evaluation.image_name}: LG={evaluation.lightglue_matches}, "
            f"F={evaluation.geometric_inliers}, 3D={evaluation.transferred_points}, "
            f"PnP={evaluation.pnp_inliers} ({evaluation.pnp_inlier_ratio:.1%}), "
            f"err={error_text}, {evaluation.status}, {evaluation.elapsed_ms:.0f}ms"
        )

    valid = [
        (evaluation, pose, debug)
        for evaluation, pose, debug in zip(evaluations, poses, debug_inputs)
        if pose is not None
    ]
    valid.sort(
        key=lambda item: (
            item[0].status == "PASS",
            item[0].pnp_inliers,
            item[0].pnp_inlier_ratio,
            -(item[0].mean_reprojection_error or float("inf")),
        ),
        reverse=True,
    )
    best = None
    gt_errors = None
    if valid:
        evaluation, pose, debug = valid[0]
        best = (evaluation, pose)
        reference_bgr, transfers = debug
        _draw_debug(
            query_bgr,
            reference_bgr,
            transfers,
            pose,
            args.output / "best_matches.jpg",
        )
        if query_reference is not None:
            rotation_error, center_error, _ = pose_errors(
                cv2, np, pose.rvec, pose.tvec, query_reference
            )
            gt_errors = {
                "rotation_deg": rotation_error,
                "camera_center": center_error,
            }
            print(
                f"Ground truth error: rotation={rotation_error:.3f}deg, "
                f"camera center={center_error:.3f} model units"
            )
        print(f"Best: {evaluation.image_name} / {evaluation.status}")
    else:
        print("No reference produced a PnP pose")
    _write_summary(args.output, evaluations, best, gt_errors)
    print(f"Results: {args.output}")
    return 0 if best is not None and best[0].status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_OUTPUT_ROOT / "object_features.npz"
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "object_observations.csv",
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_COLMAP / "Images.txt")
    parser.add_argument("--cameras", type=Path, default=DEFAULT_COLMAP / "Cameras.txt")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "aliked_lightglue_pnp"
    )
    parser.add_argument("--reference-name", action="append")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-neighbors", type=int, default=8)
    parser.add_argument("--retrieval-distance", type=float, default=300.0)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--match-threshold", type=float, default=0.1)
    parser.add_argument("--fundamental-threshold", type=float, default=3.0)
    parser.add_argument("--observation-radius", type=float, default=6.0)
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--ransac-threshold", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.20)
    parser.add_argument("--max-reprojection-error", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.top_k <= 0 or args.max_keypoints <= 0:
            raise LearnedMatchingError("--top-k and --max-keypoints must be positive")
        if args.resize == 0 or args.resize < -1:
            raise LearnedMatchingError("--resize must be -1 or a positive pixel size")
        return run(args)
    except (LearnedMatchingError, PnPTestError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
