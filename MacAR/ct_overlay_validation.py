#!/usr/bin/env python3
"""Frozen-frame CT/live display-consistency measurements (research use only).

One live pixel defines a camera ray, not a unique 3D location. The reported
metric distance is the *lateral* ray-to-CT-point separation at the selected CT
point's depth. It is not target-registration error or an along-ray error.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from realtime_pnp_server import projected_box_depth_interval, colmap_to_rc


class CTValidationSession:
    def __init__(self, renderer, frame, matrix, distortion, result,
                 intrinsics_label=''):
        if result.status != 'PASS' or result.rvec is None or result.tvec is None:
            raise ValueError('Freeze a PASS pose before CT validation')
        self.renderer = renderer
        self.frame = frame.copy()
        self.matrix = np.asarray(matrix, np.float64).copy()
        self.distortion = np.asarray(distortion, np.float64).copy()
        self.rvec = np.asarray(result.rvec, np.float64).reshape(3, 1)
        self.tvec = np.asarray(result.tvec, np.float64).reshape(3, 1)
        self.result = result
        self.intrinsics_label = intrinsics_label
        self.frozen_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self.live_uv = None
        self.ct_xyz = None
        self.ct_voxel = None
        self.last_mpr_status = 'HIDDEN'
        self.last_display = self.frame.copy()

    def mouse_event(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            selected = self.renderer.display_to_ct_point(x, y)
            if selected is None:
                print('CT marker: click inside the visible CT slice', flush=True)
                return
            self.ct_xyz, self.ct_voxel = selected
            print('CT marker voxel:', np.round(self.ct_voxel, 2).tolist(), flush=True)
        else:
            uv = self.renderer.display_to_camera_point(x, y)
            if uv is None:
                return
            height, width = self.frame.shape[:2]
            if not (0 <= uv[0] < width and 0 <= uv[1] < height):
                return
            self.live_uv = uv
            self.renderer.center_zoom_on_display_point(x, y)
            print('Live marker pixel:', np.round(uv, 2).tolist(), flush=True)

    def measurement(self):
        if self.live_uv is None or self.ct_xyz is None:
            return None
        rotation = cv2.Rodrigues(self.rvec)[0]
        camera_point = rotation @ self.ct_xyz + self.tvec.ravel()
        if not np.isfinite(camera_point).all() or camera_point[2] <= 0:
            return None
        undistorted = cv2.undistortPoints(
            self.live_uv.reshape(1, 1, 2), self.matrix, self.distortion
        ).reshape(2)
        ray_at_ct_depth = np.array((undistorted[0] * camera_point[2],
                                    undistorted[1] * camera_point[2],
                                    camera_point[2]), np.float64)
        # Both positions are expressed in the physical coordinate system of the
        # loaded CT volume (the renderer maps COLMAP world back to it via RC).
        live_world = rotation.T @ (ray_at_ct_depth - self.tvec.ravel())
        ct_physical = colmap_to_rc(self.ct_xyz)
        live_physical = colmap_to_rc(live_world)
        delta_physical = live_physical - ct_physical
        lateral_distance = float(np.linalg.norm(delta_physical))
        predicted = cv2.projectPoints(self.ct_xyz.reshape(1, 3), self.rvec,
                                      self.tvec, self.matrix, self.distortion)[0].reshape(2)
        pixel_distance = float(np.linalg.norm(predicted - self.live_uv))
        return dict(projected_uv=predicted, pixel_error=pixel_distance,
                    lateral_error=lateral_distance,
                    ct_physical_xyz=ct_physical,
                    live_at_ct_depth_physical_xyz=live_physical,
                    delta_ct_physical=delta_physical,
                    camera_depth=float(camera_point[2]),
                    coordinate_space='CT volume physical',
                    unit=self.renderer.unit_label)

    @staticmethod
    def _crosshair(canvas, point, color, label):
        x, y = (int(round(value)) for value in point)
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            return
        cv2.line(canvas, (x-13, y), (x+13, y), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(canvas, (x, y-13), (x, y+13), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(canvas, (x-13, y), (x+13, y), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y-13), (x, y+13), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, color, 2, cv2.LINE_AA)

    def slice_offset(self):
        if self.ct_xyz is None:
            return None
        plane = self.renderer.current_validation_plane()
        if plane is None:
            return None
        center, _, _ = plane
        return float(np.dot(self.ct_xyz-center,
                            cv2.Rodrigues(self.rvec)[0].T @ np.array((0., 0., 1.))))

    def return_to_marker_slice(self):
        if self.ct_xyz is None:
            return None
        rotation = cv2.Rodrigues(self.rvec)[0]
        camera_center = -rotation.T @ self.tvec.ravel()
        forward = rotation.T @ np.array((0., 0., 1.))
        near, far = projected_box_depth_interval(
            camera_center, forward, self.renderer.volume_corners)
        if far <= near:
            return None
        depth = float(np.dot(self.ct_xyz-camera_center, forward))
        percent = float(np.clip(100. * (depth-near)/(far-near), -100., 150.))
        self.renderer.set_slice_percent(percent)
        return percent

    def render(self):
        canvas, self.last_mpr_status = self.renderer.render(
            self.frame.copy(), self.matrix, self.distortion, self.rvec, self.tvec)
        if self.live_uv is not None:
            point = self.renderer.camera_to_display_point(*self.live_uv)
            if point is not None:
                self._crosshair(canvas, point, (0, 255, 255), 'LIVE')
        measurement = self.measurement()
        if self.ct_xyz is not None:
            projected = cv2.projectPoints(self.ct_xyz.reshape(1, 3), self.rvec,
                self.tvec, self.matrix, self.distortion)[0].reshape(2)
            point = self.renderer.camera_to_display_point(*projected)
            if point is not None:
                offset = self.slice_offset()
                label = ('CT' if offset is None or abs(offset) < max(self.renderer.volume.spacing)
                         else f'CT off-slice {offset:+.1f} {self.renderer.unit_label}')
                self._crosshair(canvas, point, (255, 0, 255), label)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 113), (20, 20, 20), -1)
        lines = [
            f'FROZEN CT VALIDATION | PnP {self.result.inliers} inliers | '
            f'MPR {self.last_mpr_status} | K {self.intrinsics_label}',
            'Click LIVE marker | Shift+click CT marker | slice/zoom sliders | N return to CT slice | S save | R resume',
        ]
        if measurement is None:
            lines.append('Select the same visible marker in LIVE and CT; depth is not measurable from one camera view')
        else:
            offset = self.slice_offset()
            offset_label = ('unavailable' if offset is None else
                            f'{offset:+.1f} {measurement["unit"]}')
            lines.append(f"Mismatch {measurement['pixel_error']:.2f} px | "
                         f"lateral at CT depth {measurement['lateral_error']:.2f} "
                         f"{measurement['unit']} | slice offset "
                         f"{offset_label}")
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (12, 28+index*32), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (230, 230, 230), 1, cv2.LINE_AA)
        height, width = canvas.shape[:2]
        cv2.rectangle(canvas, (0, max(0, height-100)),
                      (min(width-1, 1050), max(0, height-35)),
                      (18, 18, 18), -1)
        cv2.putText(canvas,
            'Cyan=LIVE pixel; magenta=projected CT point. Slice changes do not move either marker.',
            (12, max(18, height-66)), cv2.FONT_HERSHEY_SIMPLEX,
            .47, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas,
            'Metric value is lateral ray error at CT depth, not along-ray or clinical TRE.',
            (12, max(18, height-43)), cv2.FONT_HERSHEY_SIMPLEX,
            .45, (0, 190, 255), 1, cv2.LINE_AA)
        self.last_display = canvas
        return canvas

    def save(self, output: Path, *, calibration_label: str):
        measurement = self.measurement()
        if measurement is None:
            raise ValueError('Select both LIVE and CT markers before saving')
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ('timestamp_utc', 'ct_volume', 'ct_coordinate_metadata',
                  'pnp_reference', 'live_u', 'live_v', 'ct_i', 'ct_j', 'ct_k',
                  'ct_x', 'ct_y', 'ct_z', 'projected_u', 'projected_v',
                  'pixel_error', 'lateral_error_at_ct_depth',
                  'ct_slice_offset_at_save', 'unit',
                  'pnp_inliers', 'pnp_inlier_ratio', 'pnp_mean_reprojection_px',
                  'camera_fx', 'camera_fy', 'camera_cx', 'camera_cy',
                  'intrinsics_source', 'distortion',
                  'rvec_x', 'rvec_y', 'rvec_z', 'tvec_x', 'tvec_y', 'tvec_z',
                  'ct_physical_x', 'ct_physical_y', 'ct_physical_z',
                  'live_at_ct_depth_physical_x', 'live_at_ct_depth_physical_y', 'live_at_ct_depth_physical_z',
                  'delta_ct_physical_x', 'delta_ct_physical_y', 'delta_ct_physical_z',
                  'camera_depth', 'coordinate_space')
        row = (self.frozen_at, str(getattr(self.renderer, 'volume_path', '')),
               str(getattr(self.renderer, 'coordinate_path', '')),
               self.result.detail, *self.live_uv, *self.ct_voxel, *self.ct_xyz,
               *measurement['projected_uv'], measurement['pixel_error'],
               measurement['lateral_error'], self.slice_offset(),
               measurement['unit'],
               self.result.inliers, self.result.inlier_ratio,
               self.result.mean_reprojection_error,
               self.matrix[0, 0], self.matrix[1, 1],
               self.matrix[0, 2], self.matrix[1, 2], calibration_label,
               ' '.join(str(float(v)) for v in self.distortion.ravel()),
               *self.rvec.ravel(), *self.tvec.ravel(),
               *measurement['ct_physical_xyz'], *measurement['live_at_ct_depth_physical_xyz'],
               *measurement['delta_ct_physical'], measurement['camera_depth'], measurement['coordinate_space'])
        existing = output.is_file()
        if existing:
            with output.open(newline='', encoding='utf-8') as target:
                if next(csv.reader(target), None) != list(fields):
                    raise ValueError('CSV columns differ. Choose a new CSV file for CT-coordinate measurements.')
        with output.open('a', newline='', encoding='utf-8') as target:
            writer = csv.writer(target)
            if not existing:
                writer.writerow(fields)
            writer.writerow(row)
        return measurement
