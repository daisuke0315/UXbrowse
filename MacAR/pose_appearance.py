"""Independent broad appearance evidence for a virtual-view pose hypothesis.

Reprojects GS RGB using its geometry proxy. High-pass correlation reduces
illumination sensitivity. This is QC evidence, not a physical accuracy metric.
"""
import cv2
import numpy as np


def evaluate_pose_appearance(query, reference, xyz, rvec, tvec, matrix, distortion,
                             exclude_reference_uv=(), stride=2, tile_size=32):
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY).astype(np.float32)
    points = xyz[::stride, ::stride].astype(np.float64)
    reference_gray = reference_gray[::stride, ::stride]
    valid = np.isfinite(points).all(axis=2)
    flat = np.nan_to_num(points.reshape(-1, 3))
    rotation = cv2.Rodrigues(rvec)[0]
    depth = (np.einsum('ni,ji->nj', flat, rotation)+np.asarray(tvec).reshape(3))[:, 2]
    projected = cv2.projectPoints(flat, rvec, tvec, matrix, distortion)[0].reshape(*valid.shape, 2)
    valid &= (depth.reshape(valid.shape) > 0)
    h, w = query.shape[:2]
    valid &= (projected[..., 0] >= 2) & (projected[..., 0] < w-2)
    valid &= (projected[..., 1] >= 2) & (projected[..., 1] < h-2)
    # Do not score the features that already supported PnP.
    mask = valid.astype(np.uint8)
    for uv in exclude_reference_uv:
        if np.isfinite(uv).all():
            cv2.circle(mask, (round(uv[0]/stride), round(uv[1]/stride)),
                       max(1, round(8/stride)), 0, -1)
    sampled = cv2.remap(query_gray, projected[..., 0].astype(np.float32),
                       projected[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    a = reference_gray-cv2.GaussianBlur(reference_gray, (0, 0), 2.)
    b = sampled-cv2.GaussianBlur(sampled, (0, 0), 2.)
    # Avoid interpolation and high-pass artifacts near geometry holes/edges.
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    tiles = []
    for y in range(0, a.shape[0], tile_size):
        for x in range(0, a.shape[1], tile_size):
            m = mask[y:y+tile_size, x:x+tile_size]
            if int(m.sum()) < tile_size*tile_size*.35:
                continue
            av = a[y:y+tile_size, x:x+tile_size][m]
            bv = b[y:y+tile_size, x:x+tile_size][m]
            av, bv = av-av.mean(), bv-bv.mean()
            if np.std(av) < 2 or np.std(bv) < 2:
                continue
            correlation = float(np.dot(av, bv) / max(np.linalg.norm(av)*np.linalg.norm(bv), 1e-9))
            tiles.append(dict(x=x*stride, y=y*stride, correlation=correlation, pixels=int(m.sum())))
    scores = [t['correlation'] for t in tiles]
    return dict(tiles=len(tiles), median_correlation=float(np.median(scores)) if scores else None,
                mean_correlation=float(np.mean(scores)) if scores else None,
                positive_fraction=float(np.mean(np.asarray(scores) > .2)) if scores else None,
                tile_scores=tiles, excluded_match_neighborhoods=len(exclude_reference_uv))
