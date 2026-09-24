#!/usr/bin/env python3
"""Localize one image against GLB-rendered XYZ reference views with ALIKED/LightGlue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from aliked_lightglue_pnp import (
    LearnedMatchingError,
    _extract,
    _image_tensor,
    _load_models,
    _match_pair,
    geometric_filter,
    solve_pose,
)
from realtime_pnp import approximate_camera_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIEWS = SCRIPT_DIR / "output_glb_reference_views"
DEFAULT_OUTPUT = SCRIPT_DIR / "output_glb_virtual_pnp"


class VirtualViewError(RuntimeError):
    """Expected virtual-view input or localization error."""


@dataclass(frozen=True)
class VirtualTransfer:
    query_index: int
    xyz: tuple[float, float, float]
    query_uv: tuple[float, float]
    reference_uv: tuple[float, float]
    match_score: float


@dataclass
class ViewEvaluation:
    rank: int
    image_name: str
    sift_ratio_matches: int
    query_keypoints: int = 0
    reference_keypoints: int = 0
    lightglue_matches: int = 0
    geometric_inliers: int = 0
    xyz_correspondences: int = 0
    pnp_inliers: int = 0
    pnp_inlier_ratio: float = 0.0
    mean_reprojection_error: Optional[float] = None
    elapsed_ms: float = 0.0
    status: str = "FAIL"
    detail: str = ""


def load_manifest(directory: Path):
    path = directory / "manifest.json"
    if not path.is_file():
        raise VirtualViewError(f"Virtual-view manifest not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VirtualViewError(f"Could not read manifest: {exc}") from exc
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise VirtualViewError("manifest.json contains no views")
    return manifest, views


def rank_views_sift(
    query_bgr: np.ndarray,
    directory: Path,
    views: Sequence[dict],
    top_k: int,
    ratio: float,
):
    sift = cv2.SIFT_create(nfeatures=4000)
    query_keypoints, query_descriptors = sift.detectAndCompute(
        cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY), None
    )
    if query_descriptors is None:
        raise VirtualViewError("No SIFT features found in query image")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ranked = []
    for view in views:
        path = directory / str(view["image_name"])
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or len(descriptors) < 2:
            count = 0
            mean_distance = float("inf")
        else:
            neighbors = matcher.knnMatch(query_descriptors, descriptors, k=2)
            accepted = [
                pair[0]
                for pair in neighbors
                if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance
            ]
            count = len(accepted)
            mean_distance = (
                float(np.mean([match.distance for match in accepted]))
                if accepted
                else float("inf")
            )
        ranked.append((count, -mean_distance, view))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(item[2], item[0]) for item in ranked[:top_k]], len(query_keypoints)


def sample_xyz_buffer(
    reference_points: np.ndarray,
    xyz: np.ndarray,
    search_radius: int,
) -> tuple[list[VirtualTransfer], list[int]]:
    """Return valid XYZ samples and their original match indices."""
    height, width = xyz.shape[:2]
    result = []
    source_indices = []
    for index, point in enumerate(reference_points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        value = xyz[y, x]
        if not np.isfinite(value).all() and search_radius > 0:
            best = None
            for radius in range(1, search_radius + 1):
                y0, y1 = max(0, y - radius), min(height, y + radius + 1)
                x0, x1 = max(0, x - radius), min(width, x + radius + 1)
                patch = xyz[y0:y1, x0:x1]
                valid = np.argwhere(np.isfinite(patch).all(axis=2))
                if len(valid):
                    distances = np.sum(
                        (valid + np.asarray((y0 - y, x0 - x))) ** 2,
                        axis=1,
                    )
                    py, px = valid[int(np.argmin(distances))]
                    best = patch[py, px]
                    break
            if best is not None:
                value = best
        if not np.isfinite(value).all():
            continue
        source_indices.append(index)
        result.append(tuple(float(component) for component in value))
    return result, source_indices


def make_transfers(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    scores: np.ndarray,
    xyz: np.ndarray,
    search_radius: int,
    *,
    geometry=None,
    diagnostics=None,
) -> list[VirtualTransfer]:
    if geometry is None:
        sampled, indices = sample_xyz_buffer(reference_points, xyz, search_radius)
    else:
        from splat_geometry import sample_geometry
        sampled, indices, rejected = sample_geometry(reference_points, geometry)
        if diagnostics is not None:
            diagnostics.update(rejected)
    candidates = [
        VirtualTransfer(
            index,
            sampled_xyz,
            tuple(float(value) for value in query_points[source_index]),
            tuple(float(value) for value in reference_points[source_index]),
            float(scores[source_index]),
        )
        for index, (sampled_xyz, source_index) in enumerate(zip(sampled, indices))
    ]
    # LightGlue is one-to-one. Quantized XYZ deduplication additionally removes
    # several neighboring pixels that describe effectively the same surface point.
    result = []
    used_xyz = set()
    for candidate in sorted(candidates, key=lambda item: item.match_score, reverse=True):
        key = tuple(round(value, 1) for value in candidate.xyz)
        if key in used_xyz:
            continue
        used_xyz.add(key)
        result.append(candidate)
    return result


def draw_debug(
    query: np.ndarray,
    reference: np.ndarray,
    transfers: Sequence[VirtualTransfer],
    inlier_indices: Sequence[int],
    output: Path,
) -> None:
    height = min(900, max(query.shape[0], reference.shape[0]))

    def resized(image):
        scale = height / image.shape[0]
        result = cv2.resize(image, (round(image.shape[1] * scale), height))
        return result, scale

    left, left_scale = resized(query)
    right, right_scale = resized(reference)
    canvas = np.hstack((left, right))
    inliers = {int(value) for value in inlier_indices}
    for index, transfer in enumerate(transfers[:300]):
        q = tuple(round(value * left_scale) for value in transfer.query_uv)
        r = (
            round(transfer.reference_uv[0] * right_scale) + left.shape[1],
            round(transfer.reference_uv[1] * right_scale),
        )
        color = (0, 255, 0) if index in inliers else (0, 165, 255)
        cv2.circle(canvas, q, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, r, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, q, r, color, 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise VirtualViewError(f"Could not write debug image: {output}")


def run(args) -> int:
    if not args.query.is_file():
        raise VirtualViewError(f"Query image not found: {args.query}")
    query = cv2.imread(str(args.query), cv2.IMREAD_COLOR)
    if query is None:
        raise VirtualViewError(f"Could not read query image: {args.query}")
    manifest, views = load_manifest(args.views)
    if args.reference_name:
        by_name = {str(view["image_name"]): view for view in views}
        selected = []
        for name in args.reference_name:
            if name not in by_name:
                raise VirtualViewError(f"Reference not in manifest: {name}")
            selected.append((by_name[name], 0))
        sift_keypoints = 0
    else:
        selected, sift_keypoints = rank_views_sift(
            query, args.views, views, args.top_k, args.sift_ratio
        )
    if not selected:
        raise VirtualViewError("No usable virtual reference views")
    print(f"GLB: {manifest.get('glb')}")
    print(f"Query: {args.query} ({query.shape[1]}x{query.shape[0]})")
    print(f"SIFT retrieval keypoints: {sift_keypoints}")
    print(
        "Candidates: "
        + ", ".join(f"{view['image_name']}({score})" for view, score in selected)
    )
    torch, extractor, matcher, device = _load_models(
        args.max_keypoints, args.match_threshold, args.device
    )
    print(f"Device: {device}")
    with torch.inference_mode():
        query_features = _extract(
            extractor, _image_tensor(torch, query, device), args.resize
        )
    camera_matrix = approximate_camera_matrix(
        query.shape[1], query.shape[0], args.query_fov_deg
    )
    evaluations = []
    hypotheses = []
    for rank, (view, sift_matches) in enumerate(selected, 1):
        started = time.perf_counter()
        evaluation = ViewEvaluation(rank, str(view["image_name"]), sift_matches)
        reference = cv2.imread(
            str(args.views / str(view["image_name"])), cv2.IMREAD_COLOR
        )
        if reference is None:
            evaluation.detail = "Reference RGB could not be read"
            evaluations.append(evaluation)
            continue
        try:
            with np.load(args.views / str(view["data_name"]), allow_pickle=False) as data:
                xyz = data["xyz"].astype(np.float32, copy=True)
                geometry = None
                if manifest.get('geometry_renderer') in {'shared-gaussian-moments', 'ray-consistent-legacy-depth'}:
                    geometry = {key: data[key].copy() for key in
                        ('depth', 'valid', 'camera_matrix', 'camera_to_world')}
        except (OSError, ValueError, KeyError) as exc:
            evaluation.detail = f"XYZ buffer error: {exc}"
            evaluations.append(evaluation)
            continue
        with torch.inference_mode():
            reference_features = _extract(
                extractor,
                _image_tensor(torch, reference, device),
                args.resize,
            )
            query_kp, reference_kp, pairs, scores = _match_pair(
                torch, matcher, query_features, reference_features
            )
        evaluation.query_keypoints = len(query_kp)
        evaluation.reference_keypoints = len(reference_kp)
        evaluation.lightglue_matches = len(pairs)
        if len(pairs):
            query_points = query_kp[pairs[:, 0]]
            reference_points = reference_kp[pairs[:, 1]]
            query_points, reference_points, scores = geometric_filter(
                query_points,
                reference_points,
                scores,
                args.fundamental_threshold,
            )
        else:
            query_points = np.empty((0, 2), np.float32)
            reference_points = np.empty((0, 2), np.float32)
        evaluation.geometric_inliers = len(query_points)
        transfers = make_transfers(
            query_points,
            reference_points,
            scores,
            xyz,
            args.xyz_search_radius,
            geometry=geometry,
        )
        evaluation.xyz_correspondences = len(transfers)
        pose = solve_pose(
            transfers,
            camera_matrix,
            None,
            args.ransac_threshold,
            args.iterations,
        )
        if pose is not None:
            evaluation.pnp_inliers = len(pose.inlier_indices)
            evaluation.pnp_inlier_ratio = evaluation.pnp_inliers / max(1, len(transfers))
            evaluation.mean_reprojection_error = pose.mean_error
            passed = (
                evaluation.pnp_inliers >= args.min_inliers
                and evaluation.pnp_inlier_ratio >= args.min_inlier_ratio
                and pose.mean_error <= args.max_reprojection_error
            )
            evaluation.status = "PASS" if passed else "REJECT"
            evaluation.detail = "quality gate passed" if passed else "quality gate failed"
            hypotheses.append((evaluation, pose, reference, transfers, view))
        else:
            evaluation.detail = "PnP could not estimate a pose"
        evaluation.elapsed_ms = (time.perf_counter() - started) * 1000
        evaluations.append(evaluation)
        error = (
            "-"
            if evaluation.mean_reprojection_error is None
            else f"{evaluation.mean_reprojection_error:.2f}px"
        )
        print(
            f"[{rank}] {evaluation.image_name}: SIFT={sift_matches}, "
            f"LG={evaluation.lightglue_matches}, F={evaluation.geometric_inliers}, "
            f"XYZ={evaluation.xyz_correspondences}, PnP={evaluation.pnp_inliers} "
            f"({evaluation.pnp_inlier_ratio:.1%}), err={error}, "
            f"{evaluation.status}, {evaluation.elapsed_ms:.0f}ms"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "view_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        fields = list(asdict(evaluations[0]).keys()) if evaluations else []
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(asdict(evaluation))
    hypotheses.sort(
        key=lambda item: (
            item[0].status == "PASS",
            item[0].pnp_inliers,
            item[0].pnp_inlier_ratio,
            -(item[0].mean_reprojection_error or float("inf")),
        ),
        reverse=True,
    )
    summary = {
        "query": str(args.query.resolve()),
        "glb": manifest.get("glb"),
        "coordinate_system": manifest.get("coordinate_system"),
        "matching": "SIFT retrieval + ALIKED + LightGlue + Fundamental RANSAC",
        "best": None,
        "evaluations": [asdict(item) for item in evaluations],
    }
    exit_code = 2
    if hypotheses:
        evaluation, pose, reference, transfers, view = hypotheses[0]
        rotation, _ = cv2.Rodrigues(pose.rvec)
        camera_center = -rotation.T @ pose.tvec
        summary["best"] = {
            "reference": evaluation.image_name,
            "status": evaluation.status,
            "rvec": pose.rvec.ravel().tolist(),
            "tvec": pose.tvec.ravel().tolist(),
            "camera_center_glb": camera_center.ravel().tolist(),
            "pnp_inliers": evaluation.pnp_inliers,
            "pnp_inlier_ratio": evaluation.pnp_inlier_ratio,
            "mean_reprojection_error": evaluation.mean_reprojection_error,
            "reference_azimuth_deg": view.get("azimuth_deg"),
            "reference_elevation_deg": view.get("elevation_deg"),
        }
        draw_debug(
            query,
            reference,
            transfers,
            pose.inlier_indices,
            args.output / "best_matches.jpg",
        )
        print(
            "Best pose: "
            f"reference={evaluation.image_name}, center(GLB)="
            f"{np.round(camera_center.ravel(), 3).tolist()}, {evaluation.status}"
        )
        exit_code = 0 if evaluation.status == "PASS" else 2
    else:
        print("No virtual reference produced a PnP pose")
    with (args.output / "summary.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
    print(f"Results: {args.output}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--views", type=Path, default=DEFAULT_VIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-name", action="append")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sift-ratio", type=float, default=0.78)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--match-threshold", type=float, default=0.1)
    parser.add_argument("--fundamental-threshold", type=float, default=3.0)
    parser.add_argument("--xyz-search-radius", type=int, default=2)
    parser.add_argument("--query-fov-deg", type=float, default=65.0)
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
            raise VirtualViewError("--top-k and --max-keypoints must be positive")
        if not 0 < args.sift_ratio < 1:
            raise VirtualViewError("--sift-ratio must be between 0 and 1")
        return run(args)
    except (VirtualViewError, LearnedMatchingError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
