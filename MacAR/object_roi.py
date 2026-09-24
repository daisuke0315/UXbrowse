"""Experimental 3DGS-appearance detector for an object crop in live frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def reference_rgb_mask(image, threshold=18):
    """Find the rendered object against the reference's near-uniform backdrop."""
    height, width = image.shape[:2]
    corner = max(4, min(height, width) // 16)
    samples = np.concatenate((image[:corner, :corner].reshape(-1, 3),
                              image[:corner, -corner:].reshape(-1, 3),
                              image[-corner:, :corner].reshape(-1, 3),
                              image[-corner:, -corner:].reshape(-1, 3)))
    background = np.median(samples, axis=0)
    difference = np.max(np.abs(image.astype(np.float32) - background), axis=2)
    raw = (difference > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        raise RuntimeError("No object silhouette in 3DGS RGB reference")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)).astype(bool)


def _unit_rows(values):
    values = np.asarray(values, np.float32)
    lengths = np.linalg.norm(values.astype(np.float64), axis=1).astype(np.float32)
    return values / np.maximum(lengths[:, None], 1e-9)


def _centers(values, count):
    if len(values) <= count:
        return _unit_rows(values)
    cv2.setRNGSeed(17)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, .001)
    _, _, centers = cv2.kmeans(np.ascontiguousarray(values, np.float32), count,
                               None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    return _unit_rows(centers)


def bbox_from_patch_mask(mask, margin=.08):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    rows, columns = mask.shape
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    left, top = max(0., x0 - dx), max(0., y0 - dy)
    right, bottom = min(float(columns), x1 + dx), min(float(rows), y1 + dy)
    return tuple(float(v) for v in
                 (left / columns, top / rows,
                  (right - left) / columns, (bottom - top) / rows))


class GSObjectDetector:
    """Reference-mask-supervised DINO patch objectness; no fixed central ROI.

    This is a conservative experiment, not a semantic segmentation guarantee.
    Low-confidence crops are rejected instead of silently using central crop.
    """

    def __init__(self, encoder, views_path: Path, views, *, max_references=72,
                 foreground_threshold=.55, margin_threshold=.02):
        self.encoder = encoder
        self.foreground_threshold = foreground_threshold
        self.margin_threshold = margin_threshold
        foreground = []
        background = []
        for view in views[:max_references]:
            image = cv2.imread(str(views_path / view["image_name"]))
            if image is None:
                raise RuntimeError(f"Cannot read detector reference: {view['image_name']}")
            valid = reference_rgb_mask(image)
            patches = encoder.patch_grid(image, max_side=224)
            rows, columns = patches.shape[:2]
            fraction = cv2.resize(valid.astype(np.float32), (columns, rows),
                                  interpolation=cv2.INTER_AREA)
            foreground.extend(patches[fraction >= .8])
            background.extend(patches[fraction <= .05])
        if len(foreground) < 32 or len(background) < 8:
            raise RuntimeError("Too few 3DGS foreground/background patches for ROI detector")
        rng = np.random.default_rng(17)
        fg = np.asarray(foreground, np.float32)
        bg = np.asarray(background, np.float32)
        if len(fg) > 4000:
            fg = fg[rng.choice(len(fg), 4000, replace=False)]
        if len(bg) > 2000:
            bg = bg[rng.choice(len(bg), 2000, replace=False)]
        self.foreground = _centers(fg, 48)
        self.background = _centers(bg, 8)

    def detect(self, image):
        patches = self.encoder.patch_grid(image)
        rows, columns = patches.shape[:2]
        flat = patches.reshape(-1, patches.shape[-1]).astype(np.float64)
        foreground = np.max(np.einsum('nc,kc->nk', flat,
                                     self.foreground.astype(np.float64),
                                     optimize=False), axis=1)
        background = np.max(np.einsum('nc,kc->nk', flat,
                                     self.background.astype(np.float64),
                                     optimize=False), axis=1)
        margin = foreground - background
        raw = ((foreground >= self.foreground_threshold)
               & (margin >= self.margin_threshold)).reshape(rows, columns)
        mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
                      if 5 <= stats[i, cv2.CC_STAT_AREA] <= .8 * rows * columns]
        if not candidates:
            return None, dict(reason="no_confident_object", mask=mask,
                              max_similarity=float(np.max(foreground)))
        _, label = max(candidates)
        selected = labels == label
        bbox = bbox_from_patch_mask(selected, margin=.08)
        confidence = float(np.mean(margin.reshape(rows, columns)[selected]))
        return bbox, dict(reason="detected", mask=selected.astype(np.uint8),
                          confidence=confidence,
                          foreground_similarity=float(np.mean(foreground.reshape(rows, columns)[selected])),
                          patch_count=int(selected.sum()))
