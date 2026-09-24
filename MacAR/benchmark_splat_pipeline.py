#!/usr/bin/env python3
"""Read-only localization benchmark. Outputs are diagnostics, NOT clinical accuracy."""
import argparse
import json
import time
import hashlib
import sys
import statistics
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from glb_virtual_live import GLBVirtualLiveLocalizer
from glb_virtual_view_pnp import load_manifest
from realtime_pnp import approximate_camera_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--query', type=Path)
    parser.add_argument('--query-views', type=Path,
                        help='Independent synthetic query-view manifest with known poses')
    parser.add_argument('--query-indices', nargs='*', type=int,
                        help='Subset of --query-views; default is every view')
    parser.add_argument('--holdout', nargs='*', type=int, default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--retrieval-mode', choices=('sift', 'dino-global', 'dino-spatial'),
                        default='sift')
    parser.add_argument('--retrieval-crop', action='store_true')
    parser.add_argument('--retrieval-auto-roi', action='store_true')
    parser.add_argument('--retrieval-roi', nargs=4, type=float,
                        default=(.10, .10, .80, .80), metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Precompute all reference features; no camera or localization')
    parser.add_argument('--fast-consensus', action='store_true')
    parser.add_argument('--appearance-audit', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new run directory')
    args.output.mkdir(parents=True)
    source_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('glb_virtual_live.py', 'aliked_lightglue_pnp.py',
                                 'glb_virtual_view_pnp.py', 'splat_geometry.py', 'pose_quality.py',
                                 'view_retrieval.py', 'object_roi.py',
                                 'generate_3dgs_reference_views.py')}
    cv2.setNumThreads(4)
    params = dict(top_k=args.top_k, device=args.device, ransac_threshold=8.0,
                  retrieval_mode=args.retrieval_mode, retrieval_crop=args.retrieval_crop,
                  retrieval_roi=args.retrieval_roi,
                  retrieval_auto_roi=args.retrieval_auto_roi)
    if args.legacy:
        params['pipeline_policy'] = 'legacy'
    if args.fast_consensus:
        params['fast_consensus'] = True
    if args.appearance_audit:
        params['appearance_audit'] = True
    start = time.perf_counter()
    localizer = GLBVirtualLiveLocalizer(args.views, **params)
    startup_ms = (time.perf_counter() - start) * 1000
    if args.prepare_only:
        for index in range(len(localizer.views)):
            localizer._reference_features(index)
            print(f'Cached {index+1}/{len(localizer.views)}', flush=True)
        return
    cases = []
    if args.query:
        cases.append((args.query, None, None))
    for index in args.holdout:
        cases.append((args.views / localizer.views[index]['image_name'], index,
                      localizer.views[index]))
    if args.query_views:
        query_manifest, query_views = load_manifest(args.query_views)
        if (query_manifest.get('ply') != localizer.manifest.get('ply')
            or query_manifest.get('runtime_transform') != localizer.runtime_transform):
            parser.error('Query and reference views must share the same 3DGS source and coordinates')
        selected = args.query_indices if args.query_indices is not None else range(len(query_views))
        for index in selected:
            if index < 0 or index >= len(query_views):
                parser.error('Query-view index outside manifest')
            cases.append((args.query_views / query_views[index]['image_name'], None,
                          query_views[index]))
    results = []
    for path, excluded, truth_view in cases:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read {path}')
        if truth_view is None:
            scale = min(1.0, 1280 / image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale)
            matrix = approximate_camera_matrix(image.shape[1], image.shape[0], 65)
        else:
            matrix = np.asarray(truth_view['camera_matrix'])
        excluded_indices = set()
        if excluded is not None:
            query_view = localizer.views[excluded]
            # The 72-view grid contains two distances for every azimuth/elevation.
            # Leaving the other distance in the index makes the angle test trivial.
            excluded_indices = {i for i, view in enumerate(localizer.views)
                                if view.get('azimuth_deg') == query_view.get('azimuth_deg')
                                and view.get('elevation_deg') == query_view.get('elevation_deg')}
            localizer.excluded_retrieval_views.update(excluded_indices)
        for repeat in range(args.repeat):
            cv2.setRNGSeed(0)
            result, display = localizer.localize(image, matrix, np.zeros(5))
            row = dict(query=str(path), excluded_view=excluded,
                       synthetic_truth=None if truth_view is None else truth_view['image_name'],
                       excluded_angle_twins=sorted(excluded_indices), repeat=repeat,
                       result=asdict(result), candidates=localizer.last_candidate_log,
                       stages=getattr(localizer, 'last_diagnostics', {}))
            if truth_view is not None and row['stages'].get('retrieval'):
                truth_rotation = np.asarray(truth_view['world_to_camera'])[:3, :3]
                by_name = {v['image_name']: v for v in localizer.views}
                angles = []
                for entry in row['stages']['retrieval']:
                    rotation = np.asarray(by_name[entry['image_name']]['world_to_camera'])[:3, :3]
                    angles.append(float(np.degrees(np.arccos(np.clip((np.trace(rotation @ truth_rotation.T)-1)/2, -1, 1)))))
                row['retrieval_known_pose'] = dict(nearest_rotation_rank=int(np.argmin(angles))+1,
                    top1_rotation_deg=angles[0],
                    top5_min_rotation_deg=min(angles[:5]),
                    top10_min_rotation_deg=min(angles[:10]),
                    top5_within_30deg=min(angles[:5]) <= 30.0001,
                    top10_within_30deg=min(angles[:10]) <= 30.0001,
                    top_k_min_rotation_deg=min(angles[:args.top_k]),
                    note='Synthetic view orientation only; not real-image retrieval ground truth')
            if truth_view is not None and result.rvec is not None:
                truth = np.asarray(truth_view['world_to_camera'])
                rotation = cv2.Rodrigues(np.asarray(result.rvec))[0]
                delta = rotation @ truth[:3, :3].T
                angle = np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
                center = -rotation.T @ np.asarray(result.tvec)
                gt_center = -truth[:3, :3].T @ truth[:3, 3]
                row['synthetic_pose_error'] = dict(rotation_deg=float(angle),
                    camera_center_distance=float(np.linalg.norm(center-gt_center)))
            results.append(row)
            cv2.imwrite(str(args.output / f'{path.stem}_{repeat}.jpg'), display)
            with (args.output / 'results.json').open('w') as target:
                json.dump(dict(startup_ms=startup_ms, parameters=vars(args), results=results,
                               source_hashes=source_hashes, python=sys.version, opencv=cv2.__version__),
                          target, indent=2, default=str)
            print(json.dumps({key: value for key, value in row.items() if key != 'stages'}, default=str), flush=True)
        if excluded is not None:
            localizer.excluded_retrieval_views.difference_update(excluded_indices)
    holdouts = [row for row in results if row['synthetic_truth'] is not None]
    scored_holdouts = [row for row in holdouts if 'retrieval_known_pose' in row]
    summary = dict(retrieval_mode=args.retrieval_mode, retrieval_crop=args.retrieval_crop,
                   retrieval_auto_roi=args.retrieval_auto_roi,
                   cases=len(results), synthetic_holdouts=len(holdouts),
                   scored_holdouts=len(scored_holdouts),
                   pass_count=sum(row['result']['status'] == 'PASS' for row in results),
                   ambiguous_count=sum(row['result']['status'] == 'AMBIGUOUS' or
                                       'AMBIGUOUS' in str(row['result'].get('detail', ''))
                                       for row in results),
                   note='Synthetic holdout is same-renderer evaluation, not real-camera ground truth')
    for output_name, field in (('processing_p50_ms', 'processing_ms'),
                               ('retrieval_p50_ms', 'sift_ms'),
                               ('local_matching_p50_ms', 'matching_ms'),
                               ('pnp_p50_ms', 'pnp_ms')):
        values = [float(row['result'][field]) for row in results
                  if row['result'].get(field) is not None]
        summary[output_name] = statistics.median(values) if values else None
    synthetic_passes = [row for row in holdouts
                        if row['result']['status'] == 'PASS' and 'synthetic_pose_error' in row]
    summary['synthetic_passes_with_gt_error_over_10deg_or_30units'] = sum(
        row['synthetic_pose_error']['rotation_deg'] > 10 or
        row['synthetic_pose_error']['camera_center_distance'] > 30
        for row in synthetic_passes)
    summary['synthetic_pass_rotation_error_median_deg'] = (
        statistics.median(row['synthetic_pose_error']['rotation_deg'] for row in synthetic_passes)
        if synthetic_passes else None)
    if scored_holdouts:
        summary.update(top1_mean_rotation_deg=statistics.mean(
                           row['retrieval_known_pose']['top1_rotation_deg'] for row in scored_holdouts),
                       top5_near_rate=statistics.mean(
                           row['retrieval_known_pose']['top5_within_30deg'] for row in scored_holdouts),
                       top10_near_rate=statistics.mean(
                           row['retrieval_known_pose']['top10_within_30deg'] for row in scored_holdouts),
                       synthetic_pass_rate=statistics.mean(
                           row['result']['status'] == 'PASS' for row in holdouts))
    with (args.output / 'summary.json').open('w') as target:
        json.dump(summary, target, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
