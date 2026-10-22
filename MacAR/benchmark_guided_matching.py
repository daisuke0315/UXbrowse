"""Paired static baseline/optimized replenishment benchmark; not CT accuracy.

Synthetic cases supply an oracle camera pose and use saved reference RGB.
Real-image cases use the current global PASS as the prior (not ground truth).
"""
import argparse
from macar_paths import DATA_DIR
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import statistics

import cv2
import numpy as np

from glb_virtual_live import GLBVirtualLiveLocalizer
from realtime_pnp import approximate_camera_matrix, DEFAULT_SPLAT_VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=Path, default=DEFAULT_SPLAT_VIEWS)
    parser.add_argument('--query', type=Path)
    parser.add_argument('--synthetic-views', type=int, nargs='*', default=[])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='cpu')
    parser.add_argument('--compare-current', action='store_true',
                        help='Compare previous guided extraction vs selective descriptors; same matching/QC')
    parser.add_argument('--projection-radius', type=float, default=96.,
                        help='Small positive values exercise the dense fallback path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.repeat < 1 or args.projection_radius <= 0 or
        not np.isfinite(args.projection_radius) or (not args.query and not args.synthetic_views)):
        parser.error('Use a new output path, positive repeat count, and at least one case')
    os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))
    cv2.setNumThreads(4)
    localizer = GLBVirtualLiveLocalizer(args.views, top_k=3, device=args.device,
        retrieval_mode='dino-global' if args.query else 'sift',
        retrieval_auto_roi=bool(args.query), ransac_threshold=8.,
        guided_pixel_radius=args.projection_radius)
    cases, initial = [], None
    distortion = np.zeros(5)
    if args.query:
        image = cv2.imread(str(args.query))
        if image is None:
            raise ValueError(f'Cannot read {args.query}')
        scale = min(1., 1280/image.shape[1])
        image = cv2.resize(image, None, fx=scale, fy=scale)
        matrix = approximate_camera_matrix(image.shape[1], image.shape[0], 65)
        cv2.setRNGSeed(0)
        result, _ = localizer.localize(image, matrix, distortion)
        initial = asdict(result)
        if result.status == 'PASS':
            cases.append((str(args.query), image, matrix,
                          np.asarray(result.rvec), np.asarray(result.tvec), True))
        else:
            print('Real image initial global localization failed; no trusted prior available', flush=True)
    for index in args.synthetic_views:
        view = localizer.views[index]
        image = cv2.imread(str(args.views/view['image_name']))
        matrix = np.asarray(view['camera_matrix'])
        pose = np.asarray(view['world_to_camera'])
        cases.append((view['image_name'], image, matrix,
                      cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3], False))
    rows = []
    for name, image, matrix, rvec, tvec, real in cases:
        localizer.retrieval_auto_roi = real
        localizer.retrieval_crop = real
        # Warm both modes so first-use XYZ/descriptor loading is not hidden in
        # one side of the comparison. Cold-start costs are not measured here.
        for optimized in [False, True]:
            localizer.guided_optimization = optimized or args.compare_current
            localizer.guided_early_descriptors = optimized
            localizer.guided_query_reuse = optimized
            cv2.setRNGSeed(0)
            localizer.localize_guided(image, matrix, distortion, rvec, tvec)
        for repeat in range(args.repeat):
            # Reverse order every other repetition to reduce warm/order bias.
            for optimized in ([False, True] if repeat % 2 == 0 else [True, False]):
                localizer.guided_optimization = optimized or args.compare_current
                localizer.guided_early_descriptors = optimized
                localizer.guided_query_reuse = optimized
                cv2.setRNGSeed(0)
                result, _ = localizer.localize_guided(image, matrix, distortion, rvec, tvec)
                stages = localizer.last_diagnostics
                rows.append(dict(case=name, real_image=real, repeat=repeat,
                    optimized=optimized, result=asdict(result), stages=stages))
                print(name, 'optimized' if optimized else 'baseline', result.status,
                      f'{result.processing_ms:.0f} ms', result.inliers, 'inliers',
                      len(stages['candidates']), 'views',
                      'fallback' if stages.get('guided_fallback') else '', flush=True)
    summary = []
    for name, *_ in cases:
        pair = {}
        for optimized in [False, True]:
            group = [r for r in rows if r['case'] == name and r['optimized'] == optimized]
            pair['optimized' if optimized else 'baseline'] = dict(
                median_ms=statistics.median(r['result']['processing_ms'] for r in group),
                median_inliers=statistics.median(r['result']['inliers'] for r in group),
                median_error_px=statistics.median(r['result']['mean_reprojection_error']
                    for r in group if r['result']['mean_reprojection_error'] is not None)
                    if any(r['result']['mean_reprojection_error'] is not None for r in group) else None,
                pass_count=sum(r['result']['status'] == 'PASS' for r in group),
                fallback_count=sum(bool(r['stages'].get('guided_fallback')) for r in group))
        summary.append(dict(case=name, **pair))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(dict(device=str(localizer.device), initial_real_image=initial,
            source_hashes={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ('glb_virtual_live.py', 'guided_matching.py', 'aliked_lightglue_pnp.py',
                             'aliked_query.py', 'pose_quality.py')},
            note='Static repeated frames; synthetic reference poses are oracle priors. '
                 'No live motion, GPU contention, or CT registration accuracy evaluated.',
            parameters=dict(feature_budget=localizer.guided_feature_budget,
                            projection_radius=localizer.guided_pixel_radius,
                            comparison='previous_guided_vs_selective_descriptors' if args.compare_current
                            else 'dense_baseline_vs_guided'),
            summary=summary, rows=rows), output, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
