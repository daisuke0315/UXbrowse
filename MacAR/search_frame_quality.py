"""Small recent-frame buffer; image quality is NOT a pose confidence score."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

PHOTOMETRIC_MODE = 'luma-gamma085-clahe12-v1'


def image_quality(image, roi=None):
    h, w = image.shape[:2]
    x, y, rw, rh = roi if roi is not None else (.15, .15, .7, .7)
    x0, y0 = max(0, int(x*w)), max(0, int(y*h))
    x1, y1 = min(w, int((x+rw)*w)), min(h, int((y+rh)*h))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return dict(usable=False, sharpness=0., score=0., mean=0., contrast=0.,
                    clipped=1., blurred=True, needs_correction=False)
    scale = min(1., 320/max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)),
                                max(1, round(crop.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Suppress high-frequency sensor noise; do not modify the matching image.
    smooth = cv2.GaussianBlur(gray, (3,3), .6)
    sharpness = float(cv2.Laplacian(smooth, cv2.CV_32F).var())
    mean, deviation = float(gray.mean()), float(gray.std())
    low, high = np.percentile(gray, (10,90))
    contrast = float(high-low)
    clipped = float(np.mean((gray <= 3) | (gray >= 252)))
    blurred = sharpness < 2. and contrast > 30. and mean >= 50.
    usable = not blurred and not (clipped > .97 and contrast < 8.)
    return dict(usable=bool(usable), sharpness=sharpness,
                score=sharpness/(deviation**2+25.) * (1.-min(.8, clipped)),
                mean=mean, contrast=contrast, clipped=clipped, blurred=bool(blurred),
                needs_correction=bool(usable and clipped < .8 and deviation > 6.
                                      and (mean < 65. or 12. <= contrast < 45.)))


def correct_luminance(image):
    """Fixed mild, coordinate-preserving transform for BOTH query and reference."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    table = np.rint((np.arange(256)/255.)**.85*255.).astype(np.uint8)
    lightness = cv2.LUT(lab[:,:,0], table)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8)).apply(lightness)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@dataclass
class SearchFrame:
    image: np.ndarray
    matrix: np.ndarray
    distortion: np.ndarray
    stamp: float
    frame_id: int
    quality: dict
    roi: object = None
    pose: object = None


class RecentSearchFrames:
    horizon = .3

    def __init__(self):
        self.frames = deque(maxlen=10)
        self.counter = 0

    def clear(self):
        self.frames.clear()

    def add(self, image, matrix, distortion, stamp, frame_id=None):
        if self.frames and (stamp <= self.frames[-1].stamp or image.shape != self.frames[-1].image.shape):
            self.clear()
        while self.frames and stamp-self.frames[0].stamp > self.horizon:
            self.frames.popleft()
        self.counter += 1
        frame = SearchFrame(image.copy(), matrix.copy(),
            None if distortion is None else distortion.copy(), stamp,
            self.counter if frame_id is None else frame_id, image_quality(image))
        self.frames.append(frame)
        return frame

    def mark_pose(self, seed, bounds):
        if not self.frames:
            return
        frame = self.frames[-1]
        frame.pose = (seed.rvec.copy(), seed.tvec.copy())
        if bounds is None:
            return
        bounds = np.asarray(bounds)
        corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]], np.float64)
        rotation = cv2.Rodrigues(np.asarray(seed.rvec, np.float64).reshape(3))[0]
        depths = (corners @ rotation.T + np.asarray(seed.tvec).reshape(3))[:,2]
        if not np.all(depths > 0):
            return
        uv = cv2.projectPoints(corners, seed.rvec, seed.tvec, frame.matrix, frame.distortion)[0].reshape(-1,2)
        if not np.isfinite(uv).all():
            return
        h,w = frame.image.shape[:2]
        low, high = np.maximum(uv.min(0), 0), np.minimum(uv.max(0), [w,h])
        if np.any(high-low < 12):
            return
        frame.roi = (low[0]/w, low[1]/h, (high[0]-low[0])/w, (high[1]-low[1])/h)
        frame.quality = image_quality(frame.image, frame.roi)

    def select(self, now, require_pose=False, *, max_age=None):
        age_limit = self.horizon if max_age is None else min(self.horizon, max_age)
        candidates = [f for f in self.frames if 0 <= now-f.stamp <= age_limit
                      and (not require_pose or f.pose is not None)
                      and (max_age is None or f.quality['usable'])]
        if not candidates:
            return None
        # Do not prefer a sharp background when the object ROI is unknown.
        targeted = [f for f in candidates if f.roi is not None and f.quality['usable']]
        if not targeted:
            latest = candidates[-1]
            return latest if latest.roi is None else None
        best = max(f.quality['score'] for f in targeted)
        # Near ties favor freshness rather than unnecessarily old viewpoints.
        return next(f for f in reversed(targeted) if f.quality['score'] >= best*.85)

