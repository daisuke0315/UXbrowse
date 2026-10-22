#!/usr/bin/env python3
"""Read-only comparison of old/new geometry on saved real-photo tracks.

Optional deterministic observation corruption exercises robust consensus.
Timing excludes input loading, model matching and the full preprocessing job.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from build_aliked_object_map import triangulate_track
from enrich_registered_aliked_map import (projected_errors, robust_triangulation,
                                         observation_errors)
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                              camera_matrix_and_distortion)
from prepared_triangulation import TriangulationContext


def legacy_robust(nodes, uv, records, matrices, distortions, bounds,
                  min_angle=2., max_error=3., min_views=3, max_hypotheses=32):
    """Frozen pre-optimization algorithm, for equivalence tests/benchmarks."""
    if len(nodes) < min_views:
        return None
    pairs = list(combinations(nodes, 2))
    if len(pairs) > max_hypotheses:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, max_hypotheses).astype(int)]
    best = None
    for pair in pairs:
        seed, _ = triangulate_track(pair, uv, records, matrices, distortions,
                                    bounds, min_angle, max_error, 10.)
        if seed is None:
            continue
        errors = projected_errors(nodes, seed[0], uv, records, matrices, distortions)
        support = [node for node, error in zip(nodes, errors) if error <= max_error]
        if len(support) < min_views:
            continue
        refined, _ = triangulate_track(support, uv, records, matrices, distortions,
                                       bounds, min_angle, max_error, 10.)
        if refined is None:
            continue
        key = (len(support), -max(refined[1]))
        if best is None or key > best[0]:
            best = key, support, refined
            if len(support) == len(nodes):
                break
    return None if best is None else (best[1], *best[2])


def compare_results(expected, actual):
    if (expected is None) != (actual is None):
        raise AssertionError('Acceptance differs')
    if expected is None:
        return 0.
    if expected[0] != actual[0]:
        raise AssertionError('Consensus observations differ')
    np.testing.assert_allclose(expected[1], actual[1], atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(expected[2], actual[2], atol=1e-7, rtol=1e-9)
    np.testing.assert_allclose(expected[3], actual[3], atol=1e-7, rtol=1e-9)
    return float(np.max(np.abs(expected[1]-actual[1])))


def benchmark(path, samples, outlier_fraction):
    path = Path(path)
    summary = json.loads((path/'summary.json').read_text())
    with np.load(path/'object_aliked_map.npz', allow_pickle=False) as archive:
        ids = np.flatnonzero(archive['track_length'] >= 3)
        xyz = archive['xyz'].copy()
    selected = set(ids[np.linspace(0, len(ids)-1, min(samples, len(ids))).astype(int)].tolist())
    tracks = defaultdict(list)
    uv = defaultdict(dict)
    with (path/'object_aliked_observations.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            pid = int(row['point_id'])
            if pid not in selected:
                continue
            name, index = row['image_name'], int(row['keypoint_index'])
            tracks[pid].append((name, index))
            uv[name][index] = (float(row['u']), float(row['v']))
    full_uv = {}
    for name, values in uv.items():
        pixels = np.zeros((max(values)+1, 2), np.float64)
        for index, pixel in values.items():
            pixels[index] = pixel
        full_uv[name] = pixels
    rng = np.random.default_rng(7)
    corrupted = 0
    for nodes in tracks.values():
        for name, index in nodes:
            if rng.random() < outlier_fraction:
                full_uv[name][index] += rng.normal(0, 60, 2)
                corrupted += 1
    records = load_reference_images(Path(summary['input_files']['images']))
    cameras, parameters = load_camera_parameters(Path(summary['input_files']['cameras']))
    matrices, distortions = {}, {}
    for cid, camera in cameras.items():
        matrices[cid], distortions[cid] = camera_matrix_and_distortion(np, camera, parameters[cid])
    bounds = np.stack((xyz.min(axis=0), xyz.max(axis=0)))
    ordered = [tracks[pid] for pid in sorted(tracks)]
    start = time.perf_counter()
    expected = [legacy_robust(nodes, full_uv, records, matrices, distortions, bounds) for nodes in ordered]
    legacy_seconds = time.perf_counter()-start
    start = time.perf_counter()
    context = TriangulationContext(full_uv, records, matrices, distortions)
    actual = [robust_triangulation(nodes, full_uv, records, matrices, distortions, bounds,
                                   geometry_context=context) for nodes in ordered]
    optimized_seconds = time.perf_counter()-start
    max_delta = max(compare_results(left, right) for left, right in zip(expected, actual))
    # Separately measure the old/new export reprojection checks on one camera.
    name = next(iter(full_uv))
    points = np.tile(xyz[:min(2000, len(xyz))], (3, 1))
    pixels = {name: np.zeros((len(points), 2), np.float64)}
    start = time.perf_counter()
    individual = np.array([projected_errors([(name, i)], point, pixels, records,
                           matrices, distortions)[0] for i, point in enumerate(points)])
    export_old = time.perf_counter()-start
    start = time.perf_counter()
    batched = observation_errors(name, np.arange(len(points)), points, pixels,
                                 records, matrices, distortions)
    export_new = time.perf_counter()-start
    np.testing.assert_allclose(individual, batched, atol=1e-8, rtol=1e-9)
    return dict(map=str(path), tracks=len(ordered), observations=sum(map(len, ordered)),
        artificial_outlier_observations=corrupted,
        accepted=sum(result is not None for result in actual), identical_acceptance_and_support=True,
        max_xyz_difference=max_delta, legacy_seconds=legacy_seconds,
        optimized_seconds=optimized_seconds, speedup=legacy_seconds/optimized_seconds,
        export_projection=dict(points=len(points), legacy_seconds=export_old,
                               optimized_seconds=export_new, speedup=export_old/export_new),
        note='Sampled saved real-photo tracks, not whole-job timing. Inputs are read-only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--outlier-fraction', type=float, default=0.)
    args = parser.parse_args()
    if args.samples < 1 or not 0 <= args.outlier_fraction <= 1:
        parser.error('samples >= 1 and outlier-fraction in [0,1] required')
    print(json.dumps(benchmark(args.map, args.samples, args.outlier_fraction), indent=2))
