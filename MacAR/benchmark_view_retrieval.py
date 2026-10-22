#!/usr/bin/env python3
"""Compare coarse retrieval on identical synthetic holdouts (not clinical accuracy)."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from glb_virtual_view_pnp import load_manifest
from view_retrieval import MODES, ViewRetrievalIndex


def rotation_error(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def run(views_path, query_path, indices, mode, crop, roi, device, auto_roi=False):
    _, views = load_manifest(views_path)
    _, query_views = load_manifest(query_path)
    started = time.perf_counter()
    retriever = ViewRetrievalIndex(views_path, views, mode=mode,
                                   crop=crop, roi=roi, device=device,
                                   auto_roi=auto_roi)
    index_ms = (time.perf_counter() - started) * 1000
    rotations = [np.asarray(v["world_to_camera"])[:3, :3] for v in views]
    rows = []
    for query_index in indices:
        query_view = query_views[query_index]
        query_rotation = np.asarray(query_view["world_to_camera"])[:3, :3]
        image = cv2.imread(str(query_path / query_view["image_name"]))
        if image is None:
            raise RuntimeError(f"Could not read query {query_view['image_name']}")
        # Different folders may still contain hard-linked copies of the same
        # reference images. Exclude identical camera rotations in both cases;
        # an overhead camera can have many different in-plane roll angles.
        excluded = {i for i, view in enumerate(views)
                    if rotation_error(rotations[i], query_rotation) < 0.1}
        started = time.perf_counter()
        ranked, _, all_ranked, debug = retriever.rank_debug(image, top_k=3, excluded=excluded)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not ranked:
            rows.append(dict(query=query_view["image_name"], index=query_index,
                             error="no candidates", retrieval_ms=elapsed_ms,
                             detected_bbox=retriever.last_query_bbox))
            continue
        angles = [rotation_error(rotations[view_index], query_rotation)
                  for _, _, view_index in all_ranked[:10]]
        fine_angles = [rotation_error(rotations[view_index], query_rotation)
                       for _, _, view_index in ranked]
        rows.append(dict(query=query_view["image_name"], index=query_index,
                         excluded=sorted(excluded),
                         top1=views[ranked[0][2]]["image_name"],
                         top1_score=float(ranked[0][0]),
                         detected_bbox=retriever.last_query_bbox,
                         top1_rotation_deg=angles[0],
                         top3_min_rotation_deg=min(fine_angles),
                         fine_candidates=[views[index]['image_name'] for _,_,index in ranked],
                         patch_rerank={key:value for key,value in debug.get('patch_rerank',{}).items()
                                       if key != 'rows'},
                         top5_min_rotation_deg=min(angles[:5]),
                         top10_min_rotation_deg=min(angles),
                         retrieval_ms=elapsed_ms))
    valid = [row for row in rows if "top1_rotation_deg" in row]
    durations = [row["retrieval_ms"] for row in rows]
    summary = dict(mode=mode, crop=crop, auto_roi=auto_roi,
                   queries=len(rows), scored=len(valid),
                   index_ms=index_ms,
                   top1_mean_rotation_deg=float(np.mean([r["top1_rotation_deg"] for r in valid]))
                   if valid else None,
                   top3_near_rate=float(np.mean([r['top3_min_rotation_deg'] <= 30.0001
                                                  for r in valid])) if valid else None,
                   top5_near_rate=float(np.mean([r["top5_min_rotation_deg"] <= 30.0001
                                                  for r in valid])) if valid else None,
                   top10_near_rate=float(np.mean([r["top10_min_rotation_deg"] <= 30.0001
                                                   for r in valid])) if valid else None,
                   retrieval_p50_ms=float(np.percentile(durations, 50)) if durations else None,
                   retrieval_p95_ms=float(np.percentile(durations, 95)) if durations else None)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--query-views", type=Path,
                        help="Independent pose-known queries; defaults to same-manifest holdout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=int, nargs="*", default=None,
                        help="view indices; omit to test all views")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--retrieval-roi", type=float, nargs=4,
                        default=(.10, .10, .80, .80))
    parser.add_argument("--include-spatial", action="store_true",
                        help="Also run the prior patch-spatial ablation")
    parser.add_argument("--include-patch", action="store_true",
                        help="Also test global Top-10 plus masked spatial reranking")
    parser.add_argument('--modes', nargs='+', choices=MODES,
                        help='Run only these object-ROI retrieval modes (DINO uses auto ROI)')
    args = parser.parse_args()
    _, views = load_manifest(args.views)
    query_path = args.query_views or args.views
    _, query_views = load_manifest(query_path)
    indices = args.holdout if args.holdout is not None else list(range(len(query_views)))
    if any(i < 0 or i >= len(query_views) for i in indices):
        parser.error("Holdout index is outside the view manifest")
    results = []
    conditions = [("sift", False, False), ("sift", True, False),
                  ("dino-global", False, False),
                  ("dino-global", True, False),
                  ("dino-global", True, True)]
    if args.include_spatial:
        conditions.extend((("dino-spatial", False, False),
                           ("dino-spatial", True, False),
                           ("dino-spatial", True, True)))
    if args.include_patch:
        conditions.append(('dino-patch', True, True))
    if args.modes:
        conditions = [(mode, True, mode != 'sift') for mode in args.modes]
    for mode, crop, auto_roi in conditions:
        summary, rows = run(args.views, query_path, indices, mode, crop, args.retrieval_roi,
                            args.device, auto_roi=auto_roi)
        results.append(dict(summary=summary, rows=rows))
        print(json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as target:
        json.dump(dict(views=str(args.views), query_views=str(query_path),
                       holdout=indices, roi=args.retrieval_roi,
                       note="Same-renderer synthetic queries; independent query map recommended",
                       results=results), target, indent=2)


if __name__ == "__main__":
    main()
