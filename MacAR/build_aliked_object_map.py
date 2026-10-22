#!/usr/bin/env python3
"""Triangulate object-only ALIKED tracks from registered RC/COLMAP photos.

An object-only 3DGS PLY is projected by each registered camera to create a
2D object mask. XYZ is *not* sampled from the PLY: it comes from matched
ALIKED pixels and fixed RC camera poses.
"""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import csv
import json
import math
import time
from functools import lru_cache
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from aliked_lightglue_pnp import _extract, _image_tensor, _load_models, _match_pair
from known_pose_matching import prepare_features, match_features
from offline_pnp_test import (camera_matrix_and_distortion,
                              load_camera_parameters, load_reference_images,
                              quaternion_to_rotation)
from generate_3dgs_reference_views import read_gaussian_ply, gaussian_arrays


DEFAULT_OUTPUT = DATA_DIR / 'output_aliked_object_map'


class MapBuildError(RuntimeError):
    pass


def load_scene_transform(path):
    """Convert trained splat coordinates back into COLMAP world coordinates."""
    if path is None:
        return np.eye(4, dtype=np.float64)
    with Path(path).open('r', encoding='utf-8') as source:
        transform = json.load(source)
    matrix = np.asarray(transform['world_from_train']['matrix_4x4'], np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-10)):
        raise MapBuildError(f'Invalid world_from_train transform: {path}')
    return matrix


def load_splat_points(path, minimum_opacity, maximum_count, world_from_train):
    records, properties = read_gaussian_ply(Path(path))
    xyz, _, _, _, _ = gaussian_arrays(records, properties, minimum_opacity)
    if len(xyz) < 100:
        raise MapBuildError('Too few finite, opaque 3DGS Gaussians')
    xyz = np.einsum('ni,ji->nj', xyz, world_from_train[:3, :3]) + world_from_train[:3, 3]
    bounds = np.stack((xyz.min(axis=0), xyz.max(axis=0)))
    stride = max(1, int(math.ceil(len(xyz) / maximum_count)))
    return xyz[::stride], bounds, len(xyz)


def object_mask(shape, points, record, matrix, distortion,
                downscale=8, margin_fraction=.035, inset_fraction=.03):
    """Project object-only 3DGS points to an RC photo; fill its silhouette."""
    height, width = shape[:2]
    rotation = quaternion_to_rotation(np, record.pose)
    translation = np.asarray(record.pose[4:], np.float64)
    # Avoid an Accelerate/BLAS warning seen with large PLY-derived arrays.
    camera = np.einsum('ni,ji->nj', np.asarray(points, np.float64), rotation) + translation
    visible = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1.)
    if visible.sum() < 100:
        return None, None
    pixels = cv2.projectPoints(np.asarray(points)[visible],
                               cv2.Rodrigues(rotation)[0], translation,
                               matrix, distortion)[0].reshape(-1, 2)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < height)]
    if len(pixels) < 100:
        return None, None
    small_width = int(math.ceil(width/downscale))
    small_height = int(math.ceil(height/downscale))
    small = np.zeros((small_height, small_width), np.uint8)
    px = np.minimum((pixels[:, 0]/downscale).astype(np.int32), small_width-1)
    py = np.minimum((pixels[:, 1]/downscale).astype(np.int32), small_height-1)
    small[py, px] = 255
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    small = cv2.dilate(small, np.ones((3, 3), np.uint8))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(small, 8)
    if components <= 1:
        return None, None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    selected = ((labels == largest).astype(np.uint8)*255)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    selected.fill(0)
    cv2.drawContours(selected, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    ys, xs = np.nonzero(selected)
    object_span = max(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    inset = object_span * inset_fraction
    if inset > 0:
        selected = (cv2.distanceTransform(selected, cv2.DIST_L2, 3) > inset).astype(np.uint8)*255
    mask = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
    y_values, x_values = np.nonzero(mask)
    if len(x_values) < 400:
        return None, None
    x0, x1 = int(x_values.min()), int(x_values.max())+1
    y0, y1 = int(y_values.min()), int(y_values.max())+1
    radius = max(4, int(round(max(x1-x0, y1-y0)*margin_fraction)))
    x0, y0 = max(0, x0-radius), max(0, y0-radius)
    x1, y1 = min(width, x1+radius), min(height, y1+radius)
    if (x1-x0)*(y1-y0) > .85*width*height:
        return None, None
    # Pad the *crop* for ALIKED context, not the accepted feature mask.
    return mask[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def select_pairs(records, object_center, max_neighbors, min_view_angle_deg):
    """Choose pose-near RC views with enough baseline for triangulation."""
    centers = {}
    directions = {}
    for name, record in records.items():
        rotation = quaternion_to_rotation(np, record.pose)
        translation = np.asarray(record.pose[4:], np.float64)
        center = -rotation.T @ translation
        direction = np.asarray(object_center, np.float64) - center
        direction /= max(np.linalg.norm(direction), 1e-12)
        centers[name], directions[name] = center, direction
    scored = []
    names = sorted(records)
    for first, second in combinations(names, 2):
        angle = math.degrees(math.acos(np.clip(np.dot(directions[first], directions[second]), -1, 1)))
        if angle < min_view_angle_deg or angle > 45:
            continue
        distances = [np.linalg.norm(centers[name]-object_center) for name in (first, second)]
        distance_ratio = max(distances)/max(min(distances), 1e-12)
        if distance_ratio > 1.8:
            continue
        # Prefer ~12 degree baselines: enough parallax, still descriptor-friendly.
        score = -abs(angle-12.) - 5.*abs(math.log(distance_ratio))
        scored.append((score, angle, first, second))
    scored.sort(reverse=True)
    degree = Counter()
    selected = []
    for _, angle, first, second in scored:
        if degree[first] >= max_neighbors or degree[second] >= max_neighbors:
            continue
        selected.append((first, second, angle))
        degree[first] += 1
        degree[second] += 1
    return selected


def subset_features(features, mask, offset):
    """Filter ALIKED features to the object mask, retaining crop coordinates."""
    keypoints = features['keypoints'][0].detach().cpu().numpy()
    x = np.rint(keypoints[:, 0]).astype(np.int64)
    y = np.rint(keypoints[:, 1]).astype(np.int64)
    height, width = mask.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep = np.flatnonzero(inside & (mask[np.clip(y, 0, height-1),
                                      np.clip(x, 0, width-1)] != 0))
    device = features['keypoints'].device
    import torch
    indices = torch.as_tensor(keep, dtype=torch.long, device=device)
    filtered = {key: (value.index_select(1, indices) if value.ndim >= 2 and
                     value.shape[1] == len(keypoints) else value)
                for key, value in features.items()}
    full_uv = filtered['keypoints'][0].detach().cpu().numpy().astype(np.float64)
    full_uv += np.asarray(offset, np.float64)
    return filtered, full_uv


def epipolar_inliers(first_uv, second_uv, first, second, matrices, distortion,
                     threshold_px):
    if len(first_uv) == 0:
        return np.zeros(0, bool)
    k1, k2 = matrices[first.camera_id], matrices[second.camera_id]
    d1, d2 = distortion[first.camera_id], distortion[second.camera_id]
    left = cv2.undistortPoints(np.asarray(first_uv, np.float64).reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    right = cv2.undistortPoints(np.asarray(second_uv, np.float64).reshape(-1, 1, 2), k2, d2).reshape(-1, 2)
    r1, r2 = quaternion_to_rotation(np, first.pose), quaternion_to_rotation(np, second.pose)
    t1, t2 = np.asarray(first.pose[4:], np.float64), np.asarray(second.pose[4:], np.float64)
    rotation = r2 @ r1.T
    translation = t2 - rotation @ t1
    tx, ty, tz = translation
    skew = np.asarray(((0, -tz, ty), (tz, 0, -tx), (-ty, tx, 0)), np.float64)
    essential = skew @ rotation
    valid = (np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) &
             (np.abs(left) < 1e4).all(axis=1) & (np.abs(right) < 1e4).all(axis=1))
    result = np.zeros(len(left), bool)
    if not np.isfinite(essential).all() or not valid.any():
        return result
    x1 = np.column_stack((left[valid], np.ones(int(valid.sum()))))
    x2 = np.column_stack((right[valid], np.ones(int(valid.sum()))))
    # Explicit contractions avoid the macOS Accelerate matmul warnings observed
    # with these small matrices; invalid rays were rejected above.
    ex1 = np.einsum('ni,ji->nj', x1, essential)
    etx2 = np.einsum('ni,ij->nj', x2, essential)
    numerator = np.sum(x2*ex1, axis=1)**2
    denominator = ex1[:, 0]**2 + ex1[:, 1]**2 + etx2[:, 0]**2 + etx2[:, 1]**2
    sampson = numerator / np.maximum(denominator, 1e-18)
    focal = np.mean((k1[0, 0], k1[1, 1], k2[0, 0], k2[1, 1]))
    result[valid] = np.isfinite(sampson) & (np.sqrt(sampson)*focal <= threshold_px)
    return result


class TrackUnion:
    """Union matches while enforcing at most one keypoint per RC image."""

    def __init__(self):
        self.parent = {}
        self.images = {}

    def root(self, node):
        self.parent.setdefault(node, node)
        self.images.setdefault(node, {node[0]})
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, first, second):
        left, right = self.root(first), self.root(second)
        if left == right or self.images[left] & self.images[right]:
            return False
        if len(self.images[left]) < len(self.images[right]):
            left, right = right, left
        self.parent[right] = left
        self.images[left].update(self.images[right])
        del self.images[right]
        return True

    def components(self):
        grouped = defaultdict(list)
        for node in self.parent:
            grouped[self.root(node)].append(node)
        return list(grouped.values())


def triangulate_track(nodes, full_uv, records, matrices, distortions, bounds,
                      min_angle_deg, max_error_px, margin):
    """Multi-view DLT in normalized camera rays, with image-space QC."""
    rows = []
    centers = []
    cameras = []
    for name, index in nodes:
        record = records[name]
        rotation = quaternion_to_rotation(np, record.pose)
        translation = np.asarray(record.pose[4:], np.float64)
        uv = np.asarray(full_uv[name][index], np.float64).reshape(1, 1, 2)
        ray = cv2.undistortPoints(uv, matrices[record.camera_id],
                                 distortions[record.camera_id]).reshape(2)
        projection = np.column_stack((rotation, translation))
        rows.extend((ray[0]*projection[2]-projection[0],
                     ray[1]*projection[2]-projection[1]))
        centers.append(-rotation.T @ translation)
        cameras.append((rotation, translation, record, uv.reshape(2)))
    _, _, vh = np.linalg.svd(np.asarray(rows, np.float64), full_matrices=False)
    if abs(vh[-1, 3]) < 1e-10:
        return None, 'infinite'
    xyz = vh[-1, :3] / vh[-1, 3]
    if not np.isfinite(xyz).all():
        return None, 'nonfinite'
    if np.any(xyz < bounds[0]-margin) or np.any(xyz > bounds[1]+margin):
        return None, 'outside_object'
    rays = np.asarray([xyz-center for center in centers])
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    widest = min(float(np.dot(a, b)) for a, b in combinations(rays, 2))
    angle = math.degrees(math.acos(np.clip(widest, -1, 1)))
    if angle < min_angle_deg:
        return None, 'low_parallax'
    errors = []
    for rotation, translation, record, uv in cameras:
        if (rotation @ xyz + translation)[2] <= 0:
            return None, 'behind_camera'
        projected = cv2.projectPoints(xyz.reshape(1, 3), cv2.Rodrigues(rotation)[0],
                                      translation, matrices[record.camera_id],
                                      distortions[record.camera_id])[0].reshape(2)
        errors.append(float(np.linalg.norm(projected-uv)))
    if max(errors) > max_error_px:
        return None, 'reprojection_error'
    return (xyz, errors, angle), None


def _feature_file(directory, name):
    return directory / (Path(name).stem + '.npz')


def save_qc(crop, mask, points, target):
    """Show the exact accepted object mask and surviving ALIKED positions."""
    canvas = crop.copy()
    green = np.zeros_like(canvas)
    green[:] = (50, 180, 50)
    canvas[mask != 0] = cv2.addWeighted(canvas, .72, green, .28, 0)[mask != 0]
    for u, v in np.asarray(points)[::max(1, len(points)//500)]:
        cv2.circle(canvas, (round(float(u)), round(float(v))), 3,
                   (30, 60, 240), -1, cv2.LINE_AA)
    if canvas.shape[1] > 1400:
        scale = 1400 / canvas.shape[1]
        canvas = cv2.resize(canvas, (1400, round(canvas.shape[0]*scale)))
    if not cv2.imwrite(str(target), canvas):
        raise MapBuildError(f'Could not save mask QC: {target}')


def build(args):
    started = time.perf_counter()
    timings = {}
    if args.max_images < 0 or args.max_mask_gaussians < 100 or args.max_keypoints < 1:
        raise MapBuildError('max-images must be >= 0, max-mask-gaussians >= 100, max-keypoints >= 1')
    if args.mask_downscale < 1 or args.max_neighbors < 1 or args.min_track_views < 2:
        raise MapBuildError('mask-downscale and max-neighbors must be >= 1; min-track-views >= 2')
    if not (0 <= args.minimum_opacity < 1 and 0 <= args.mask_inset < .3 and
            0 <= args.mask_margin < .3):
        raise MapBuildError('minimum-opacity must be in [0,1); mask-inset/mask-margin in [0,0.3)')
    if min(args.min_pair_angle, args.epipolar_px, args.min_triangulation_angle,
           args.max_reprojection_px) <= 0:
        raise MapBuildError('Angular and pixel thresholds must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise MapBuildError(f'Output directory is not empty: {args.output}')
    refs = load_reference_images(args.images)
    cameras, parameter_text = load_camera_parameters(args.cameras)
    world_from_train = load_scene_transform(args.scene_transform)
    mask_points, bounds, gaussian_count = load_splat_points(
        args.splat, args.minimum_opacity, args.max_mask_gaussians, world_from_train)
    available = {name: record for name, record in refs.items()
                 if (args.image_dir/name).is_file()}
    if args.max_images:
        available = dict(list(available.items())[:args.max_images])
    if len(available) < 3:
        raise MapBuildError('Need at least 3 object-observed registered images')
    matrices, distortions = {}, {}
    for camera_id, camera in cameras.items():
        matrices[camera_id], distortions[camera_id] = camera_matrix_and_distortion(
            np, camera, parameter_text[camera_id])
    object_center = np.mean(bounds, axis=0)
    pairs = select_pairs(available, object_center, args.max_neighbors,
                         args.min_pair_angle)
    if not pairs:
        raise MapBuildError('No overlapping RC image pairs with sufficient parallax')
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output/'reference_features'
    feature_dir.mkdir(exist_ok=True)
    qc_dir = args.output/'mask_qc'
    qc_dir.mkdir(exist_ok=True)
    torch, extractor, matcher, device = _load_models(args.max_keypoints,
        args.match_threshold, args.device, matcher_required=args.matching_method == 'lightglue')
    full_uv = {}
    counts = Counter()
    selected_names = set(name for pair in pairs for name in pair[:2])
    ordered_names = list(available)
    qc_names = {ordered_names[0], ordered_names[len(ordered_names)//2], ordered_names[-1]}
    phase_started = time.perf_counter()
    for position, name in enumerate(available, 1):
        if name not in selected_names:
            continue
        image = cv2.imread(str(args.image_dir/name), cv2.IMREAD_COLOR)
        if image is None:
            raise MapBuildError(f'Cannot read RC image: {name}')
        camera = cameras[available[name].camera_id]
        if image.shape[:2] != (camera.height, camera.width):
            raise MapBuildError(f'Image/Cameras.txt size mismatch: {name}')
        mask, box = object_mask(image.shape, mask_points, available[name],
                                matrices[available[name].camera_id],
                                distortions[available[name].camera_id],
                                args.mask_downscale, args.mask_margin, args.mask_inset)
        if mask is None:
            counts['no_object_mask'] += 1
            continue
        x0, y0, x1, y1 = box
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        with torch.inference_mode():
            raw = _extract(extractor, _image_tensor(torch, crop, device), args.resize)
        features, points = subset_features(raw, mask, (x0, y0))
        if len(points) == 0:
            counts['empty_features'] += 1
            continue
        arrays = {key: value.detach().cpu().numpy() for key, value in features.items()}
        arrays.update(full_uv=points.astype(np.float32), crop_box=np.asarray(box, np.int32),
                      object_mask=mask,
                      image_id=np.asarray(available[name].image_id, np.int64),
                      camera_id=np.asarray(available[name].camera_id, np.int64))
        np.savez_compressed(_feature_file(feature_dir, name), **arrays)
        full_uv[name] = points
        if name in qc_names:
            save_qc(crop, mask, points - np.asarray((x0, y0)),
                    qc_dir/(Path(name).stem + '_object_aliked.jpg'))
        counts['images'] += 1
        counts['features'] += len(points)
        if counts['images'] % 25 == 0:
            print(f"ALIKED: {counts['images']} images, {counts['features']} masked features", flush=True)
    timings['feature_preparation_seconds'] = time.perf_counter()-phase_started
    if len(full_uv) < 3:
        raise MapBuildError('Too few images with masked ALIKED features')
    del extractor

    # Keep feature caches bounded when processing the complete photo set.
    @lru_cache(maxsize=8)
    def tensor_features(name):
        with np.load(_feature_file(feature_dir, name), allow_pickle=False) as archive:
            return {key: torch.from_numpy(archive[key].copy()).to(device)
                    for key in ('keypoints', 'descriptors', 'keypoint_scores', 'image_size')}

    @lru_cache(maxsize=12)
    def pose_features(name):
        record = available[name]
        with np.load(_feature_file(feature_dir, name), allow_pickle=False) as archive:
            return prepare_features(full_uv[name], archive['descriptors'][0], record,
                                    matrices[record.camera_id], distortions[record.camera_id])

    edges = []
    phase_started = time.perf_counter()
    for pair_number, (first, second, _) in enumerate(pairs, 1):
        if first not in full_uv or second not in full_uv:
            continue
        if args.matching_method == 'known-pose':
            matches, scores, detail = match_features(pose_features(first), pose_features(second),
                                                     epipolar_px=args.epipolar_px)
            counts['candidate_pairs'] += detail['candidate_pairs']
            counts['all_feature_pairs'] += detail['total_pairs']
        else:
            _, _, matches, scores = _match_pair(torch, matcher,
                                                tensor_features(first), tensor_features(second))
        counts['pairs'] += 1
        counts['raw_matches'] += len(matches)
        if len(matches):
            keep = epipolar_inliers(full_uv[first][matches[:, 0]],
                                    full_uv[second][matches[:, 1]], available[first],
                                    available[second], matrices, distortions, args.epipolar_px)
            for (left_index, right_index), score in zip(matches[keep], scores[keep]):
                edges.append((float(score), (first, int(left_index)),
                              (second, int(right_index))))
            counts['epipolar_matches'] += int(keep.sum())
        if pair_number % 50 == 0:
            print(f"{args.matching_method}: {pair_number}/{len(pairs)} pairs, "
                  f"{counts['epipolar_matches']} pose-consistent matches", flush=True)
    timings['matching_seconds'] = time.perf_counter()-phase_started
    phase_started = time.perf_counter()
    union = TrackUnion()
    for _, first, second in sorted(edges, reverse=True):
        if union.join(first, second):
            counts['accepted_edges'] += 1
        else:
            counts['conflicting_edges'] += 1
    tracks = []
    rejection = Counter()
    for nodes in union.components():
        if len(nodes) < args.min_track_views:
            rejection['short_track'] += 1
            continue
        result, reason = triangulate_track(nodes, full_uv, available, matrices,
                                           distortions, bounds, args.min_triangulation_angle,
                                           args.max_reprojection_px, args.bounds_margin)
        if result is None:
            rejection[reason] += 1
            continue
        xyz, errors, angle = result
        tracks.append((nodes, xyz, errors, angle))
    counts['triangulated_tracks'] = len(tracks)
    counts['triangulated_observations'] = sum(len(nodes) for nodes, _, _, _ in tracks)
    if not tracks:
        raise MapBuildError('No ALIKED tracks passed triangulation QC')

    timings['triangulation_seconds'] = time.perf_counter()-phase_started
    phase_started = time.perf_counter()
    # Per-image files are directly usable by a later RC-reference live matcher.
    assignments = defaultdict(dict)
    point_xyz, point_lengths, track_errors = [], [], []
    for point_id, (nodes, xyz, errors, angle) in enumerate(tracks):
        point_xyz.append(xyz)
        point_lengths.append(len(nodes))
        track_errors.append(max(errors))
        for (name, keypoint), error in zip(nodes, errors):
            assignments[name][keypoint] = (point_id, error)
    rows = []
    for name in full_uv:
        path = _feature_file(feature_dir, name)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        ids = np.full(len(full_uv[name]), -1, np.int64)
        xyz_values = np.full((len(ids), 3), np.nan, np.float32)
        for keypoint, (point_id, error) in assignments[name].items():
            ids[keypoint] = point_id
            xyz_values[keypoint] = point_xyz[point_id]
            u, v = full_uv[name][keypoint]
            rows.append((point_id, name, available[name].image_id, keypoint,
                         u, v, *point_xyz[point_id], error))
        arrays.update(point_ids=ids, xyz=xyz_values)
        np.savez_compressed(path, **arrays)
        if name in qc_names:
            image = cv2.imread(str(args.image_dir/name), cv2.IMREAD_COLOR)
            x0, y0, x1, y1 = arrays['crop_box']
            accepted = full_uv[name][ids >= 0] - np.asarray((x0, y0))
            save_qc(image[y0:y1, x0:x1], arrays['object_mask'], accepted,
                    qc_dir/(Path(name).stem + '_triangulated.jpg'))
    np.savez_compressed(args.output/'object_aliked_map.npz',
                        xyz=np.asarray(point_xyz, np.float32),
                        track_length=np.asarray(point_lengths, np.int32),
                        max_reprojection_error=np.asarray(track_errors, np.float32))
    with (args.output/'object_aliked_observations.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(('point_id', 'image_name', 'image_id', 'keypoint_index',
                         'u', 'v', 'X', 'Y', 'Z', 'reprojection_error_px'))
        writer.writerows(rows)
    timings['export_seconds'] = time.perf_counter()-phase_started
    summary = dict(input_images=len(available), image_features=dict(counts), matching_method=args.matching_method,
                   timings=timings, processing_seconds=time.perf_counter()-started,
                   image_dir=str(args.image_dir.resolve()),
                   pair_candidates=len(pairs), rejected_tracks=dict(rejection),
                   triangulated_tracks=len(tracks), triangulated_observations=len(rows),
                   median_max_reprojection_px=float(np.median(track_errors)),
                   p95_max_reprojection_px=float(np.percentile(track_errors, 95)),
                   coordinate_system='RC/COLMAP world coordinates; source units',
                   geometry_source='ALIKED multiview triangulation with fixed RC camera poses',
                   masking='projected object-only 3DGS silhouette (PLY geometry only)',
                   mask_inset_fraction=args.mask_inset,
                   splat_gaussians=gaussian_count,
                   input_files=dict(images=str(args.images), cameras=str(args.cameras),
                                    splat=str(args.splat),
                                    scene_transform=(str(args.scene_transform)
                                                     if args.scene_transform else None)))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path,
                   help='Folder with splat.ply, colmap/images and colmap/sparse/0')
    p.add_argument('--image-dir', type=Path,
                   help='Override the registered image directory')
    p.add_argument('--images', type=Path,
                   help='Override COLMAP Images.txt')
    p.add_argument('--cameras', type=Path,
                   help='Override COLMAP Cameras.txt')
    p.add_argument('--splat', type=Path,
                   help='Object-only 3DGS PLY; scene transform maps its train frame to COLMAP')
    p.add_argument('--scene-transform', type=Path,
                   help='Optional 3DGS scene_transform.json; auto-detected in dataset-root')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--max-images', type=int, default=0, help='0=all; useful for a pilot run')
    p.add_argument('--minimum-opacity', type=float, default=.005)
    p.add_argument('--max-mask-gaussians', type=int, default=80000)
    p.add_argument('--mask-downscale', type=int, default=8)
    p.add_argument('--mask-margin', type=float, default=.035)
    p.add_argument('--mask-inset', type=float, default=.03,
                   help='Shrink the projected object mask by this fraction of its longest side')
    p.add_argument('--max-keypoints', type=int, default=2048)
    p.add_argument('--resize', type=int, default=1024)
    p.add_argument('--match-threshold', type=float, default=.01)
    p.add_argument('--matching-method', choices=('known-pose', 'lightglue'), default='known-pose',
                   help='Fixed-camera epipolar descriptor matching; LightGlue is the comparison mode')
    p.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    p.add_argument('--max-neighbors', type=int, default=5)
    p.add_argument('--min-pair-angle', type=float, default=2.)
    p.add_argument('--epipolar-px', type=float, default=2.5)
    p.add_argument('--min-track-views', type=int, default=3)
    p.add_argument('--min-triangulation-angle', type=float, default=2.)
    p.add_argument('--max-reprojection-px', type=float, default=3.)
    p.add_argument('--bounds-margin', type=float, default=10.,
                   help='Allowed distance beyond the supplied object PLY bounds (source units)')
    return p


def main():
    cli = parser()
    args = cli.parse_args()
    if args.dataset_root is not None:
        args.image_dir = args.image_dir or args.dataset_root/'colmap/images'
        args.images = args.images or args.dataset_root/'colmap/sparse/0/Images.txt'
        args.cameras = args.cameras or args.dataset_root/'colmap/sparse/0/Cameras.txt'
        args.splat = args.splat or args.dataset_root/'splat.ply'
        if args.scene_transform is None:
            transforms = list((args.dataset_root/'colmap/outputs').glob('*/scene_transform.json'))
            if len(transforms) > 1:
                cli.error('Multiple scene transforms found; choose one with --scene-transform')
            if transforms:
                args.scene_transform = transforms[0]
    if any(path is None for path in (args.image_dir, args.images, args.cameras, args.splat)):
        cli.error('Specify --dataset-root, or all of --image-dir, --images, --cameras, --splat')
    try:
        build(args)
    except (MapBuildError, OSError, ValueError, cv2.error) as exc:
        cli.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
