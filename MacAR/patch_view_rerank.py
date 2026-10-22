"""Experimental view reranking, not metric geometry or pose estimation.

Masked, mean-centered DINO patches retain layout instead of just object identity.
Only a global shortlist is reranked. Small image-space similarity RANSAC is a
soft consistency cue, not a hard rigid/planar assumption for a 3D object.
"""
import cv2
import numpy as np


GRID_SIDE = 16
SHORTLIST = 10
WEIGHTS = (.25, .50, .25)  # global, same-location layout, consistent patch matches


def square_mask(mask, side=224):
    height, width = mask.shape
    scale = side/max(height, width)
    resized = cv2.resize(mask.astype(np.float32),
                        (max(1, round(width*scale)), max(1, round(height*scale))),
                        interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side), np.float32)
    top, left = (side-resized.shape[0])//2, (side-resized.shape[1])//2
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    return cv2.resize(canvas, (GRID_SIDE, GRID_SIDE), interpolation=cv2.INTER_AREA) >= .5


def _normalize(rows):
    return rows/np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-8)


def patch_metrics(query, reference, query_mask, reference_mask):
    """Compare dense patches in normalized crop coordinates (never XYZ).

    No arbitrary homography/essential matrix is used to certify these coarse
    correspondences. Reflection, large roll/scale and narrow support are penalized.
    Missing support gives zero evidence, never a fabricated high similarity.
    """
    side = np.asarray(query).shape[0]
    q = np.asarray(query, np.float32).reshape(side**2, -1)
    r = np.asarray(reference, np.float32).reshape(side**2, -1)
    qm, rm = np.asarray(query_mask, bool).ravel(), np.asarray(reference_mask, bool).ravel()
    empty = dict(layout=0., consistency=0., mutual_matches=0, inliers=0,
                 coverage=0., residual=None, pairs=[])
    if min(qm.sum(), rm.sum()) < 8 or q.shape != r.shape:
        return empty
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        return empty
    q = _normalize(q)
    r = _normalize(r)
    qc = _normalize(q-q[qm].mean(axis=0))
    rc = _normalize(r-r[rm].mean(axis=0))
    overlap = qm & rm
    if overlap.sum() >= 8:
        # A soft layout score: neither a true correspondence nor confidence.
        raw = np.sum(q[overlap]*r[overlap], axis=1).mean()
        centered = np.sum(qc[overlap]*rc[overlap], axis=1).mean()
        layout = float(np.clip(.25*raw+.75*(centered+1.)/2., 0., 1.))
        layout *= float(overlap.sum()/max(qm.sum(), rm.sum()))
    else:
        layout = 0.
    qi, ri = np.flatnonzero(qm), np.flatnonzero(rm)
    # Avoid macOS Accelerate float32 matmul warnings on the small CPU matrix.
    similarities = (.3*np.einsum('ic,jc->ij', q[qi], r[ri], dtype=np.float64, optimize=False)+
                    .7*np.einsum('ic,jc->ij', qc[qi], rc[ri], dtype=np.float64, optimize=False))
    best = similarities.argmax(axis=1)
    back = similarities.argmax(axis=0)
    top_two = np.partition(similarities, -2, axis=1)[:, -2:]
    margins = top_two.max(axis=1)-top_two.min(axis=1)
    accepted = ((back[best] == np.arange(len(qi))) &
                (similarities[np.arange(len(qi)), best] > .3) & (margins >= .025))
    pairs = np.column_stack((qi[accepted], ri[best[accepted]]))
    result = dict(empty, layout=layout, mutual_matches=len(pairs))
    if len(pairs) < 4:
        return result
    xy = np.array([(x+.5, y+.5) for y in range(side) for x in range(side)], np.float32)/side
    matrix, inliers = cv2.estimateAffinePartial2D(
        xy[pairs[:, 1]], xy[pairs[:, 0]], method=cv2.RANSAC,
        ransacReprojThreshold=.12, maxIters=256, confidence=.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return result
    scale = float(np.linalg.norm(matrix[:2, 0]))
    roll = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    if not .65 <= scale <= 1.5 or roll > 35 or np.linalg.det(matrix[:, :2]) <= 0:
        return result
    keep = inliers.ravel().astype(bool)
    support = pairs[keep]
    if len(support) < 4:
        return result
    coverage = min(float(cv2.contourArea(cv2.convexHull(xy[support[:, 0]]))),
                   float(cv2.contourArea(cv2.convexHull(xy[support[:, 1]]))))
    prediction = xy[support[:, 1]]@matrix[:, :2].T+matrix[:, 2]
    residual = float(np.linalg.norm(prediction-xy[support[:, 0]], axis=1).mean())
    ratio = len(support)/max(1, min(len(qi), len(ri)))
    consistency = float(np.clip(ratio/.4, 0., 1.)*np.clip(coverage/.20, 0., 1.)*
                        np.exp(-residual/.12))
    return dict(result, consistency=consistency, inliers=len(support),
                coverage=coverage, residual=residual,
                pairs=[dict(query=xy[a].tolist(), reference=xy[b].tolist()) for a,b in support])


def rerank(shortlist, query, query_mask, reference_patches, reference_masks):
    rows = []
    for original_rank, (global_score, _, index) in enumerate(shortlist, 1):
        evidence = patch_metrics(query, reference_patches[index], query_mask, reference_masks[index])
        score = sum(weight*value for weight, value in zip(
            WEIGHTS, (global_score, evidence['layout'], evidence['consistency'])))
        rows.append(dict(index=int(index), global_rank=original_rank,
                         global_score=float(global_score), score=float(score), **evidence))
    rows.sort(key=lambda row:(row['score'], row['global_score']), reverse=True)
    return [(row['score'], row['global_score'], row['index']) for row in rows], rows


def select_diverse(ranked, views, count, minimum_angle=35.):
    """Spend fine-matching slots on different hypotheses, not duplicate poses.

    Known virtual-camera rotations ONLY diversify the shortlist. They never
    substitute for the live pose, enter PnP, or certify the reference's XYZ.
    A fallback fills remaining slots when the shortlist has no other direction.
    """
    count = min(count, len(ranked))
    rotations = {}
    for _, _, index in ranked:
        transform = np.asarray(views[index].get('world_to_camera'), np.float64)
        if transform.shape == (4,4) and np.isfinite(transform).all():
            rotations[index] = transform[:3,:3]
    chosen = []
    for entry in ranked:
        if len(chosen) >= count:
            break
        rotation = rotations.get(entry[2])
        if rotation is None:
            continue
        close = False
        for _, _, previous in chosen:
            relative = rotation@rotations[previous].T
            angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1.)/2.,-1.,1.))))
            if angle < minimum_angle:
                close = True
                break
        if not close:
            chosen.append(entry)
    ids = {entry[2] for entry in chosen}
    chosen.extend(entry for entry in ranked if entry[2] not in ids)
    return chosen[:count]
