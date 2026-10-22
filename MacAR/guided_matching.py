"""Small, testable filters for pose-guided ALIKED/LightGlue replenishment.

These filters never initialize a pose or certify CT alignment. Coordinates
remain in the original full-image pixels even when ALIKED runs on a crop.
"""
import heapq
import cv2
import numpy as np


def region_counts(points, bbox, grid=4):
    """Row-major occupancy only; no pixels/descriptors or point IDs in logs."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    x0,y0,x1,y1 = bbox
    keep = (np.isfinite(points).all(axis=1) & (points[:,0] >= x0) &
            (points[:,0] < x1) & (points[:,1] >= y0) & (points[:,1] < y1))
    scaled = (points[keep]-[x0,y0])/np.maximum([x1-x0,y1-y0],1.)
    cells = np.minimum((scaled*grid).astype(int),grid-1)
    return np.bincount(cells[:,1]*grid+cells[:,0],minlength=grid*grid).tolist()


def replenishment_indices(points, scores, existing, grid=8):
    """Water-fill sparse cells, accounting for points already being tracked."""
    points = np.asarray(points, np.float64).reshape(-1,2)
    existing = np.asarray(existing, np.float64).reshape(-1,2)
    if not len(points):
        return np.empty(0,np.int64)
    all_points = np.vstack((points,existing))
    low = all_points.min(axis=0)
    span = np.maximum(np.ptp(all_points,axis=0),1.)
    cells = np.minimum(((all_points-low)/span*grid).astype(int),grid-1)
    cell_ids = cells[:,1]*grid+cells[:,0]
    counts = np.bincount(cell_ids[len(points):],minlength=grid*grid)
    ordered = np.argsort(-np.asarray(scores),kind='stable')
    buckets = {int(key):ordered[cell_ids[ordered]==key] for key in np.unique(cell_ids[:len(points)])}
    heap = [(int(counts[key]),key,0) for key in buckets]
    heapq.heapify(heap)
    selected = []
    while heap:
        count,key,offset = heapq.heappop(heap)
        selected.append(int(buckets[key][offset]))
        if offset+1 < len(buckets[key]):
            heapq.heappush(heap,(count+1,key,offset+1))
    return np.asarray(selected,np.int64)


def spatial_indices(points, scores, budget, grid=8):
    """Round-robin spatial cells, highest detector score first within a cell."""
    points = np.asarray(points)
    if not len(points) or len(points) <= budget:
        return np.arange(len(points), dtype=np.int64)
    budget = min(budget, len(points))
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1.)
    cell = np.minimum(((points-low)/span*grid).astype(int), grid-1)
    ids = cell[:, 1]*grid + cell[:, 0]
    scores = np.ones(len(points)) if scores is None else np.asarray(scores).reshape(-1)
    ordered = np.argsort(-scores, kind='stable')
    buckets = [ordered[ids[ordered] == key] for key in range(grid*grid)]
    selected = []
    level = 0
    while len(selected) < budget:
        added = False
        for bucket in buckets:
            if level < len(bucket):
                selected.append(int(bucket[level]))
                added = True
                if len(selected) == budget:
                    break
        if not added:
            break
        level += 1
    return np.asarray(selected, np.int64)


def subset_features(torch, features, indices):
    """Preserve image_size and slice all ALIKED per-keypoint tensors together."""
    count = features['keypoints'].shape[1]
    result = {}
    for key, value in features.items():
        if key != 'image_size' and value.ndim >= 2 and value.shape[:2] == (1, count):
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            result[key] = value.index_select(1, index)
        else:
            result[key] = value
    return result


def project_candidates(xyz, rvec, tvec, matrix, distortion, shape, bbox=None):
    """Positive depth and image/crop bounds only; no unverified occlusion model."""
    xyz = np.asarray(xyz, np.float64)
    projected = np.full((len(xyz), 2), np.nan)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        return projected, np.zeros(len(xyz), bool)
    rotation = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
    depths = np.einsum('ni,ji->nj', xyz[finite], rotation)[:, 2] + np.asarray(tvec).reshape(3)[2]
    projected[finite] = cv2.projectPoints(xyz[finite], np.asarray(rvec, np.float64),
        np.asarray(tvec, np.float64), matrix, distortion)[0].reshape(-1, 2)
    height, width = shape[:2]
    x0, y0, x1, y1 = bbox if bbox is not None else (0, 0, width, height)
    valid = finite.copy()
    valid[finite] &= depths > 0
    valid &= (np.isfinite(projected).all(axis=1) &
        (projected[:, 0] >= x0) & (projected[:, 0] < x1) &
        (projected[:, 1] >= y0) & (projected[:, 1] < y1))
    return projected, valid


def pair_gate(query_points, pairs, predicted, radius, offset):
    """Reject matches contradicting the prior, in full-image pixel units."""
    if not len(pairs):
        return np.zeros(0, bool)
    delta = query_points[pairs[:, 0]] + offset - predicted[pairs[:, 1]]
    return np.isfinite(delta).all(axis=1) & (np.linalg.norm(delta, axis=1) <= radius)
