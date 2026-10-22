#!/usr/bin/env python3
"""Extend an existing real-photo ALIKED map using fixed COLMAP cameras.

Reuse object-masked feature archives. Preserve existing landmark IDs/XYZ;
extend their observations with independent multi-image support, and create
new landmarks only by robust multi-view triangulation. No splat depth lookup.
"""

from macar_paths import DATA_DIR
import argparse
import csv
import json
import hashlib
import math
import time
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from aliked_lightglue_pnp import _load_models, _match_pair
from build_aliked_object_map import (MapBuildError, TrackUnion, epipolar_inliers,
                                    _feature_file)
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                              camera_matrix_and_distortion, quaternion_to_rotation)
from prepared_triangulation import TriangulationContext
from known_pose_matching import prepare_features, match_features, pair_cache_key


@lru_cache(maxsize=2048)
def _projection_pose(pose):
    rotation = quaternion_to_rotation(np, pose)
    return rotation, cv2.Rodrigues(rotation)[0], np.asarray(pose[4:], np.float64)


def projected_errors(nodes, xyz, full_uv, records, matrices, distortions):
    errors = []
    for name, index in nodes:
        rec = records[name]
        rotation, rvec, translation = _projection_pose(tuple(rec.pose))
        if (rotation @ xyz + translation)[2] <= 0:
            errors.append(float('inf'))
            continue
        uv = cv2.projectPoints(np.asarray(xyz).reshape(1, 3),
            rvec, translation, matrices[rec.camera_id],
            distortions[rec.camera_id])[0].reshape(2)
        errors.append(float(np.linalg.norm(uv-full_uv[name][index])))
    return np.asarray(errors)


def observation_errors(name, indices, points, full_uv, records, matrices, distortions):
    """Batch export verification, retaining OpenCV's original XYZ precision."""
    if not len(indices):
        return np.empty(0, np.float64)
    record = records[name]
    rotation, rvec, translation = _projection_pose(tuple(record.pose))
    points = np.asarray(points).reshape(-1, 3)
    pixels = cv2.projectPoints(points, rvec, translation, matrices[record.camera_id],
                               distortions[record.camera_id])[0].reshape(-1, 2)
    errors = np.linalg.norm(pixels-full_uv[name][indices], axis=1)
    depth = np.einsum('ni,i->n', points, rotation[2]) + translation[2]
    errors[depth <= 0] = np.inf
    return errors


def anchor_pair_errors(matches, first, second, ids, xyz, full_uv, records, matrices, distortions):
    """Batch the SAME fixed-XYZ reprojection check, not a looser QC.

    Avoid two Python/OpenCV projections per anchor when processing all views.
    Non-anchor matches get infinity and are handled by the caller's branches.
    """
    result = np.full(len(matches), np.inf)
    left, right = ids[first][matches[:, 0]], ids[second][matches[:, 1]]
    anchors = np.flatnonzero((left >= 0) ^ (right >= 0))
    if not len(anchors):
        return result
    pid = np.where(left[anchors] >= 0, left[anchors], right[anchors])
    points = xyz[pid]
    errors = []
    for column, name in enumerate((first, second)):
        rec = records[name]
        rotation, rvec, translation = _projection_pose(tuple(rec.pose))
        projected = cv2.projectPoints(points, rvec, translation, matrices[rec.camera_id],
                                      distortions[rec.camera_id])[0].reshape(-1, 2)
        value = np.linalg.norm(projected - full_uv[name][matches[anchors, column]], axis=1)
        depth = np.einsum('ni,ji->nj', points, rotation)[:, 2] + translation[2]
        value[depth <= 0] = np.inf
        errors.append(value)
    result[anchors] = np.maximum(*errors)
    return result


def consensus_assignments(proposals, ids, min_support=2):
    """No nearest-pixel assignment: require independent matched source images.

    Reject competing landmark IDs for one feature, or several target features
    claiming the same landmark within one image. Existing observations win.
    """
    candidates = {}
    rejected = Counter()
    for node, votes in proposals.items():
        name, index = node
        if ids[name][index] >= 0:
            continue
        qualified = [pid for pid, sources in votes.items() if len(sources) >= min_support]
        if len(qualified) != 1:
            rejected['anchor_conflict' if len(qualified) > 1 else 'anchor_low_support'] += 1
            continue
        candidates[node] = qualified[0]
    claims = defaultdict(list)
    for node, pid in candidates.items():
        claims[node[0], pid].append(node)
    accepted = {}
    for (name, pid), nodes in claims.items():
        if len(nodes) != 1 or np.any(ids[name] == pid):
            rejected['anchor_duplicate_observation'] += len(nodes)
            continue
        accepted[nodes[0]] = pid
    return accepted, rejected


def robust_triangulation(nodes, full_uv, records, matrices, distortions, bounds,
                         min_angle=2., max_error=3., min_views=3, max_hypotheses=32,
                         geometry_context=None):
    """Find a multi-view consensus; a single bad observation need not kill a track."""
    if len(nodes) < min_views:
        return None
    pairs = list(combinations(range(len(nodes)), 2))
    if len(pairs) > max_hypotheses:
        indices = np.linspace(0, len(pairs)-1, max_hypotheses).astype(int)
        pairs = [pairs[i] for i in indices]
    if not pairs:
        return None
    context = geometry_context or TriangulationContext(full_uv, records, matrices, distortions)
    prepared = context.prepare(nodes)
    _, errors_by_pair, valid = prepared.pair_hypotheses(pairs, bounds, min_angle, max_error)
    best = None
    for errors, seed_valid in zip(errors_by_pair, valid):
        if not seed_valid:
            continue
        support_indices = np.flatnonzero(errors <= max_error)
        if len(support_indices) < min_views:
            continue
        support = [nodes[index] for index in support_indices]
        refined = prepared.triangulate(support_indices, bounds, min_angle, max_error)
        if refined is None:
            continue
        key = (len(support), -max(refined[1]))
        if best is None or key > best[0]:
            best = key, support, refined
            if len(support) == len(nodes):
                break
    return None if best is None else (best[1], *best[2])


def enrichment_pairs(records, center, targets, max_neighbors=12):
    centers, directions = {}, {}
    for name, rec in records.items():
        centers[name] = -quaternion_to_rotation(np, rec.pose).T @ np.asarray(rec.pose[4:])
        direction = centers[name]-center
        directions[name] = direction/max(np.linalg.norm(direction), 1e-12)
    pairs = set()
    for first in sorted(targets):
        neighbors = []
        for second in records:
            if first == second:
                continue
            angle = math.degrees(math.acos(float(np.clip(np.dot(directions[first], directions[second]), -1, 1))))
            distance1, distance2 = np.linalg.norm(centers[first]-center), np.linalg.norm(centers[second]-center)
            ratio = max(distance1, distance2)/max(min(distance1, distance2), 1e-12)
            if 2. <= angle <= 45. and ratio <= 1.8:
                neighbors.append((abs(angle-12.)+5.*abs(math.log(ratio)), second))
        for _, second in sorted(neighbors)[:max_neighbors]:
            pairs.add(tuple(sorted((first, second))))
    return sorted(pairs)


def enrich(args):
    started = time.perf_counter()
    matching_method = getattr(args, 'matching_method', 'known-pose')
    if matching_method not in ('known-pose', 'lightglue'):
        raise MapBuildError('Unknown preprocessing matcher')
    def progress(stage, **values):
        event = dict(stage=stage, elapsed_seconds=round(time.perf_counter()-started, 2), **values)
        callback = getattr(args, 'progress', None)
        if callback:
            callback(event)
        elif getattr(args, 'json_progress', False):
            print('@@PREPROCESS ' + json.dumps(event), flush=True)
    progress('load', detail='Loading registered features and fixed RC cameras')
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise MapBuildError('Output must be separate from the original map')
    if output.exists() and any(output.iterdir()):
        raise MapBuildError(f'Output directory is not empty: {output}')
    if args.max_neighbors < 1 or args.min_anchor_support < 2 or args.max_error <= 0:
        raise MapBuildError('neighbors >= 1, independent anchor support >= 2, error > 0 required')
    summary = json.loads((source/'summary.json').read_text())
    with np.load(source/'object_aliked_map.npz', allow_pickle=False) as archive:
        old_map = {key: archive[key].copy() for key in archive.files}
    xyz = np.asarray(old_map['xyz'], np.float64)
    records = load_reference_images(Path(summary['input_files']['images']))
    cameras, parameters = load_camera_parameters(Path(summary['input_files']['cameras']))
    matrices, distortions = {}, {}
    for camera_id, camera in cameras.items():
        matrices[camera_id], distortions[camera_id] = camera_matrix_and_distortion(np, camera, parameters[camera_id])
    full_uv, ids, feature_hashes = {}, {}, {}
    for name in records:
        path = _feature_file(source/'reference_features', name)
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                full_uv[name] = np.asarray(archive['full_uv'], np.float64)
                ids[name] = np.asarray(archive['point_ids'], np.int64).copy()
                digest = hashlib.sha256()
                for key in ('keypoints', 'descriptors', 'image_size', 'full_uv'):
                    value = np.ascontiguousarray(archive[key])
                    digest.update(str((key, value.shape, str(value.dtype))).encode())
                    digest.update(value.tobytes())
                feature_hashes[name] = digest.hexdigest()
            if np.any(ids[name] >= len(xyz)):
                raise MapBuildError(f'Landmark ID outside map: {name}')
    records = {name: record for name, record in records.items() if name in ids}
    targets = set(args.target_images or records)
    if targets-set(records):
        raise MapBuildError(f'Unknown reference images: {sorted(targets-set(records))}')
    # Bounds constrain XYZ, never determine it. Use the existing triangulated
    # object's extent, with the same 10-unit margin as the original builder.
    bounds = np.stack((xyz.min(axis=0), xyz.max(axis=0)))
    pairs = enrichment_pairs(records, np.mean(bounds, axis=0), targets, args.max_neighbors)
    cache_dir = getattr(args, 'cache_dir', None)
    if cache_dir:
        cache_dir = Path(cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
    camera_hash = hashlib.sha256(Path(summary['input_files']['images']).read_bytes() +
                                 Path(summary['input_files']['cameras']).read_bytes()).hexdigest()
    progress('models', total_pairs=len(pairs), target_images=len(targets), matching_method=matching_method)
    device = 'cpu'
    if matching_method == 'lightglue':
        torch, extractor, matcher, device = _load_models(2000, .01, args.device)
        del extractor
    @lru_cache(maxsize=12)
    def features(name):
        with np.load(_feature_file(source/'reference_features', name), allow_pickle=False) as archive:
            return {key: torch.from_numpy(archive[key].copy()).to(device)
                for key in ('keypoints','descriptors','keypoint_scores','image_size')}
    @lru_cache(maxsize=12)
    def pose_features(name):
        record = records[name]
        with np.load(_feature_file(source/'reference_features', name), allow_pickle=False) as archive:
            return prepare_features(full_uv[name], archive['descriptors'][0], record,
                                    matrices[record.camera_id], distortions[record.camera_id])
    proposals = defaultdict(lambda: defaultdict(set))
    edges, counts, timings = [], Counter(), Counter()
    print(f'Enrich: {len(targets)} targets, {len(pairs)} pairs, {matching_method}/{device}; original map retained', flush=True)
    for number, (first, second) in enumerate(pairs, 1):
        pair_started = time.perf_counter()
        cache_key = pair_cache_key(matching_method, camera_hash, feature_hashes[first], feature_hashes[second])
        cache_path = cache_dir / (cache_key + '.npz') if cache_dir else None
        cached = False
        if cache_path and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    matches, scores = archive['matches'].copy(), archive['scores'].copy()
                if (matches.ndim != 2 or matches.shape[1] != 2 or len(matches) != len(scores)
                    or not np.isfinite(scores).all() or np.any(matches < 0)
                    or np.any(matches[:, 0] >= len(ids[first])) or np.any(matches[:, 1] >= len(ids[second]))):
                    raise ValueError('Invalid pair cache')
                cached = True
            except (OSError, ValueError, KeyError):
                cached = False
        if not cached:
            if matching_method == 'known-pose':
                matches, scores, detail = match_features(pose_features(first), pose_features(second))
                counts['candidate_pairs'] += detail['candidate_pairs']
                counts['all_feature_pairs'] += detail['total_pairs']
            else:
                _, _, matches, scores = _match_pair(torch, matcher, features(first), features(second))
            if cache_path:
                np.savez_compressed(cache_path, matches=matches, scores=scores)
        counts['cached_pairs' if cached else 'computed_pairs'] += 1
        timings['matching_seconds'] += time.perf_counter()-pair_started
        counts['raw_matches'] += len(matches)
        geometry_started = time.perf_counter()
        keep = epipolar_inliers(full_uv[first][matches[:,0]], full_uv[second][matches[:,1]],
            records[first], records[second], matrices, distortions, 2.5)
        counts['epipolar_matches'] += int(keep.sum())
        kept_matches = matches[keep]
        anchor_errors = anchor_pair_errors(kept_matches, first, second, ids, xyz,
                                          full_uv, records, matrices, distortions)
        for (i, j), score, error in zip(kept_matches, scores[keep], anchor_errors):
            left, right = (first,int(i)), (second,int(j))
            left_id, right_id = int(ids[first][i]), int(ids[second][j])
            if left_id < 0 and right_id < 0:
                edges.append((float(score),left,right))
                continue
            if left_id >= 0 and right_id >= 0:
                counts['seed_id_agree' if left_id == right_id else 'seed_id_conflict'] += 1
                continue
            node, seed, pid = (left,right,right_id) if left_id < 0 else (right,left,left_id)
            if np.isfinite(error) and error <= args.max_error:
                proposals[node][pid].add(seed[0])
            else:
                counts['anchor_reprojection_rejected'] += 1
        timings['geometry_assignment_seconds'] += time.perf_counter()-geometry_started
        if number % 10 == 0 or number == len(pairs):
            print(f'Pairs {number}/{len(pairs)} | geometric {counts["epipolar_matches"]} | {time.perf_counter()-started:.1f}s', flush=True)
            progress('matching', completed=number, total=len(pairs), first=first, second=second,
                     counts=dict(counts), timings=dict(timings), detail='Fixed-pose 2D matching and verified XYZ transfer')
    features.cache_clear()
    pose_features.cache_clear()
    if matching_method == 'lightglue':
        del matcher
    before_ids = {name: value.copy() for name,value in ids.items()}
    accepted, rejected = consensus_assignments(proposals, ids, args.min_anchor_support)
    counts.update(rejected)
    for (name,index), pid in accepted.items():
        ids[name][index] = pid
    counts['extended_observations'] = len(accepted)
    progress('triangulation', detail='Robust multi-view consensus; at least 3 views')
    union = TrackUnion()
    for _, first, second in sorted(edges, reverse=True):
        if ids[first[0]][first[1]] < 0 and ids[second[0]][second[1]] < 0:
            union.join(first,second)
    new_xyz, new_errors = [], []
    components = union.components()
    geometry_context = TriangulationContext(full_uv, records, matrices, distortions)
    triangulation_started = time.perf_counter()
    for number, nodes in enumerate(components, 1):
        if number % 200 == 0:
            timings['triangulation_seconds'] = time.perf_counter()-triangulation_started
            progress('triangulation', completed=number, total=len(components), new_landmarks=len(new_xyz),
                     counts=dict(counts), timings=dict(timings), geometry_engine='prepared-batched-v1')
        if len(nodes) < 3:
            counts['new_short_tracks'] += 1
            continue
        result = robust_triangulation(nodes,full_uv,records,matrices,distortions,bounds,
                                      max_error=args.max_error, geometry_context=geometry_context)
        if result is None:
            counts['new_geometry_rejected'] += 1
            continue
        support, point, errors, angle = result
        pid = len(xyz)+len(new_xyz)
        new_xyz.append(point)
        new_errors.append(max(errors))
        counts['robust_outliers_removed'] += len(nodes)-len(support)
        for name,index in support:
            ids[name][index] = pid
    timings['triangulation_seconds'] = time.perf_counter()-triangulation_started
    all_xyz = np.vstack((xyz,np.asarray(new_xyz).reshape(-1,3))).astype(np.float32)
    lengths = np.zeros(len(all_xyz), np.int32)
    maximum_errors = np.r_[old_map['max_reprojection_error'],new_errors].astype(np.float32)
    output.mkdir(parents=True,exist_ok=True)
    feature_dir = output/'reference_features'
    feature_dir.mkdir()
    per_image = []
    export_started = time.perf_counter()
    progress('export', total=len(records))
    # Streaming CSV/archives: do not retain the complete reference tensor set.
    with (output/'object_aliked_observations.csv').open('w',newline='',encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(('point_id','image_name','image_id','keypoint_index','u','v','X','Y','Z','reprojection_error_px'))
        for name,rec in records.items():
            with np.load(_feature_file(source/'reference_features',name),allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            valid = ids[name] >= 0
            values = np.full((len(ids[name]),3),np.nan,np.float32)
            values[valid] = all_xyz[ids[name][valid]]
            arrays.update(point_ids=ids[name],xyz=values)
            np.savez_compressed(_feature_file(feature_dir,name),**arrays)
            indices = np.flatnonzero(valid)
            errors = observation_errors(name, indices, all_xyz[ids[name][indices]],
                                        full_uv, records, matrices, distortions)
            for index, error in zip(indices, errors):
                pid = int(ids[name][index])
                lengths[pid] += 1
                maximum_errors[pid] = max(maximum_errors[pid],error)
                writer.writerow((pid,name,rec.image_id,index,*full_uv[name][index],*all_xyz[pid],error))
            per_image.append(dict(image=name,features=len(ids[name]),
                before_xyz=int((before_ids[name]>=0).sum()),after_xyz=int(valid.sum())))
            if len(per_image) % 25 == 0:
                timings['export_seconds'] = time.perf_counter()-export_started
                progress('export', completed=len(per_image), total=len(records), timings=dict(timings))
    timings['export_seconds'] = time.perf_counter()-export_started
    np.savez_compressed(output/'object_aliked_map.npz',xyz=all_xyz,track_length=lengths,
                        max_reprojection_error=maximum_errors)
    counts['new_landmarks'] = len(new_xyz)
    counts['added_observations'] = int(lengths.sum())-int(sum((value>=0).sum() for value in before_ids.values()))
    report = dict(source_map=str(source),targets=sorted(targets),max_neighbors=args.max_neighbors,
        matching_method=matching_method,
        pair_candidates=len(pairs),
        timings=dict(timings), geometry_engine='prepared-batched-v1',
        min_anchor_support=args.min_anchor_support,max_reprojection_px=args.max_error,
        counts=dict(counts),per_image=per_image,processing_seconds=time.perf_counter()-started,
        notes='Existing landmark IDs/XYZ retained; added XYZ from fixed-pose multiview triangulation; no GS depth')
    (output/'enrichment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary.update(triangulated_tracks=len(all_xyz),triangulated_observations=int(lengths.sum()),
        median_max_reprojection_px=float(np.median(maximum_errors)),
        p95_max_reprojection_px=float(np.percentile(maximum_errors,95)),
        enrichment=dict(source_map=str(source),report='enrichment_report.json',counts=dict(counts),
                        matching_method=matching_method))
    summary['image_features'] = dict(summary['image_features'],triangulated_tracks=len(all_xyz),
                                     triangulated_observations=int(lengths.sum()))
    # Write the usable-map summary last. An interrupted export cannot look like
    # a completed map to the live loader.
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=counts,seconds=report['processing_seconds'],output=str(output)),indent=2),flush=True)
    progress('complete', output=str(output), counts=dict(counts))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DATA_DIR/'output_aliked_object_map')
    parser.add_argument('--output',type=Path,default=DATA_DIR/'output_aliked_object_map_enriched')
    parser.add_argument('--target-images',nargs='+',help='Optional reference names for a pilot; default all')
    parser.add_argument('--max-neighbors',type=int,default=12)
    parser.add_argument('--min-anchor-support',type=int,default=2)
    parser.add_argument('--max-error',type=float,default=3.)
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--matching-method',choices=('known-pose','lightglue'),default='known-pose')
    parser.add_argument('--cache-dir', type=Path, help='Content-keyed pair matching cache for repeated enrichment')
    parser.add_argument('--json-progress', action='store_true')
    args = parser.parse_args()
    try:
        enrich(args)
    except (MapBuildError,OSError,ValueError,cv2.error) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__ == '__main__':
    main()
