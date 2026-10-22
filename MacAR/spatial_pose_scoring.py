"""Region-balanced ranking for registered-photo pose hypotheses.

The grid belongs to the query object ROI, never to an individual hypothesis.
This ranks already solved poses; it does not subsample PnP or change its QC.
"""
import math

import numpy as np


def spatial_pose_score(pixels, errors, roi, threshold):
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    errors = np.asarray(errors, np.float64).reshape(-1)
    bounds = np.asarray(roi, np.float64)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all() or
            np.any(bounds[2:] <= bounds[:2]) or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError('Spatial scoring requires a finite positive ROI and threshold')
    if len(errors) != len(pixels):
        raise ValueError('One reprojection error is required per pixel')
    valid = (np.isfinite(pixels).all(axis=1) & np.isfinite(errors) & (errors >= 0) &
             (pixels >= bounds[:2]).all(axis=1) & (pixels <= bounds[2:]).all(axis=1))
    cells = np.minimum(((pixels[valid]-bounds[:2])/(bounds[2:]-bounds[:2])*8).astype(int), 7)
    ids = cells[:, 1]*8+cells[:, 0]
    residuals = errors[valid]
    medians = np.array([np.median(residuals[ids == cell]) for cell in np.unique(ids)])
    return dict(score=float(np.maximum(0., 1.-(medians/threshold)**2).sum()),
                occupied_cells=int(len(medians)), grid=8,
                balanced_error_px=float(medians.mean()) if len(medians) else None,
                error_threshold_px=float(threshold), roi=[float(v) for v in bounds])


def registered_hypothesis_rank(hypothesis):
    quality = hypothesis[2]
    score = quality['spatial_score']
    error = score['balanced_error_px']
    return (bool(quality['passed']), score['score'], score['occupied_cells'],
            -error if error is not None else -math.inf)


def comparable_support(other, best):
    """Keep the existing 70% gate, expressed in capped spatial votes."""
    return (bool(other[2]['passed']) and
            other[2]['spatial_score']['score'] >= .7*best[2]['spatial_score']['score'])
