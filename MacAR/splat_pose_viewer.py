"""Read-only live QC view of 3DGS appearance and camera-pose hypotheses.

An orange camera is only a PnP hypothesis from an older search frame. It must
never be interpreted as the current camera position or a valid CT registration.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class SplatPoseViewer:
    """Show live video, a colored 3DGS point preview, and the chosen virtual RGB."""

    WINDOW = "ViewSync Live | 3DGS Camera Pose"
    BOX_EDGES = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )

    def __init__(self, localizer, *, height: int = 600) -> None:
        source = localizer.global_localizer
        self.to_runtime = source._source_xyz_to_runtime
        self.views_path = Path(source.views_path)
        self.views = list(source.views)
        self.vertical_sign = (-1. if source.manifest.get("up_axis") == "negative-z"
                              else 1.)
        self.view_by_name = {str(view["image_name"]): view for view in self.views}
        self.height = height
        self.live_width = 440
        self.map_width = 500
        self.reference_width = 300
        bounds = np.asarray(source.manifest["bounds"], dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("Invalid 3DGS bounds for pose viewer")
        corners = np.asarray(
            [(x, y, z) for x in bounds[:, 0]
             for y in bounds[:, 1] for z in bounds[:, 2]], dtype=np.float64
        )
        self.corners = source._source_xyz_to_runtime(corners)
        self.minimum = self.corners.min(axis=0)
        self.maximum = self.corners.max(axis=0)
        self.center = (self.minimum + self.maximum) / 2
        self.base_radius = max(float(np.max(self.maximum - self.minimum)) * 0.95, 1.)
        self.points, self.colors = self._load_colored_points(source)
        self.yaw = math.radians(-42.)
        self.pitch = math.radians(24.)
        self.reference_cache = {}
        self.last_pose_state = "UNKNOWN"
        self.last_pose_center = None
        self.last_reference_name = None
        self.last_reference_camera_center = None

    def _load_colored_points(self, source):
        """Sample only a small fixed subset; no Gaussian rendering in the UI loop."""
        if not self.views:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        nearest_elevation = min(self.views, key=lambda v: abs(float(v.get("elevation_deg", 0))))["elevation_deg"]
        nearest_distance = min(float(v.get("distance_multiplier", 1)) for v in self.views)
        selected = [v for v in self.views
                    if float(v.get("elevation_deg", 0)) == float(nearest_elevation)
                    and float(v.get("distance_multiplier", 1)) == nearest_distance]
        if len(selected) < 4:
            selected = self.views[::max(1, len(self.views) // 12)]
        selected = selected[:12]
        point_batches, color_batches = [], []
        for view in selected:
            rgb = cv2.imread(str(self.views_path / str(view["image_name"])), cv2.IMREAD_COLOR)
            if rgb is None:
                continue
            try:
                with np.load(self.views_path / str(view["data_name"]), allow_pickle=False) as data:
                    xyz = data["xyz"][::12, ::12]
            except (OSError, ValueError, KeyError):
                continue
            color = rgb[::12, ::12]
            if xyz.shape[:2] != color.shape[:2]:
                continue
            runtime_xyz = source._source_xyz_to_runtime(xyz.reshape(-1, 3)).reshape(xyz.shape)
            margin = (self.maximum - self.minimum) * .1
            valid = (np.isfinite(runtime_xyz).all(axis=2)
                     & (runtime_xyz >= self.minimum - margin).all(axis=2)
                     & (runtime_xyz <= self.maximum + margin).all(axis=2))
            if not np.any(valid):
                continue
            point_batches.append(runtime_xyz[valid].astype(np.float64))
            color_batches.append(color[valid].astype(np.uint8))
        if not point_batches:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        return np.concatenate(point_batches), np.concatenate(color_batches)

    def _basis(self):
        outward = np.asarray((math.cos(self.pitch) * math.cos(self.yaw),
                              math.cos(self.pitch) * math.sin(self.yaw),
                              self.vertical_sign * math.sin(self.pitch)), dtype=np.float64)
        right = np.cross((0., 0., self.vertical_sign), outward)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = np.cross(outward, right)
        return right, up, outward

    def _project(self, points, radius):
        values = np.asarray(points, np.float64).reshape(-1, 3) - self.center
        right, up, outward = self._basis()
        scale = min(self.map_width, self.height) * .44 / radius
        horizontal = np.einsum('ij,j->i', values, right)
        vertical = np.einsum('ij,j->i', values, up)
        depth = np.einsum('ij,j->i', values, outward)
        pixels = np.column_stack((horizontal * scale + self.map_width * .5,
                                  -vertical * scale + self.height * .52))
        return pixels, depth

    @staticmethod
    def _pose_from_result(result):
        if result is None or result.rvec is None or result.tvec is None:
            return None
        rotation = cv2.Rodrigues(np.asarray(result.rvec, np.float64).reshape(3))[0]
        translation = np.asarray(result.tvec, np.float64).reshape(3)
        center = -rotation.T @ translation
        forward = rotation.T @ np.asarray((0., 0., 1.))
        if not np.isfinite(center).all():
            return None
        return center, forward

    @staticmethod
    def _text(canvas, value, x, y, *, color=(230, 230, 230), size=.48, thickness=1):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thickness, cv2.LINE_AA)

    def _reference_name(self, diagnostics):
        retrieval = diagnostics.get("retrieval") or []
        name = retrieval[0]["image_name"] if retrieval else diagnostics.get("selected")
        if name not in self.view_by_name:
            name = None
        return name

    def _reference_image(self, name):
        if name is None:
            return None
        if name not in self.reference_cache:
            image = cv2.imread(str(self.views_path / name), cv2.IMREAD_COLOR)
            if image is not None:
                self.reference_cache[name] = image
        return self.reference_cache.get(name)

    def _draw_map(self, pose, verified, reference_name, candidate_age_ms):
        panel = np.full((self.height, self.map_width, 3), 22, dtype=np.uint8)
        camera_center = None if pose is None else pose[0]
        reference = self.view_by_name.get(reference_name)
        reference_eye = None
        if reference is not None and "eye" in reference:
            reference_eye = self.to_runtime(np.asarray(reference["eye"], np.float64).reshape(1, 3))[0]
            if not np.isfinite(reference_eye).all():
                reference_eye = None
        self.last_reference_camera_center = (None if reference_eye is None else
                                             tuple(float(v) for v in reference_eye))
        radius = self.base_radius
        if camera_center is not None:
            radius = max(radius, float(np.linalg.norm(camera_center - self.center)) * 1.1)
        if reference_eye is not None:
            radius = max(radius, float(np.linalg.norm(reference_eye - self.center)) * 1.1)
        if len(self.points):
            pixels, depth = self._project(self.points, radius)
            xy = np.rint(pixels).astype(np.int32)
            valid = ((xy[:, 0] >= 0) & (xy[:, 0] < self.map_width) &
                     (xy[:, 1] >= 70) & (xy[:, 1] < self.height - 70))
            xy, depth, color = xy[valid], depth[valid], self.colors[valid]
            order = np.argsort(depth)  # nearest visible sample wins per pixel
            panel[xy[order, 1], xy[order, 0]] = color[order]
        box, _ = self._project(self.corners, radius)
        box = np.rint(box).astype(np.int32)
        for first, second in self.BOX_EDGES:
            cv2.line(panel, tuple(box[first]), tuple(box[second]), (0, 210, 255), 1, cv2.LINE_AA)
        if reference_eye is not None:
            virtual_pixel, _ = self._project(reference_eye, radius)
            virtual_pixel = tuple(np.rint(virtual_pixel[0]).astype(np.int32))
            aim_pixel, _ = self._project(
                reference_eye + (self.center - reference_eye) * .24, radius)
            aim_pixel = tuple(np.rint(aim_pixel[0]).astype(np.int32))
            cv2.arrowedLine(panel, virtual_pixel, aim_pixel,
                            (255, 190, 60), 2, cv2.LINE_AA, tipLength=.3)
            cv2.circle(panel, virtual_pixel, 7, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(panel, virtual_pixel, 5, (255, 190, 60), -1, cv2.LINE_AA)
        if pose is not None:
            center, forward = pose
            marker, _ = self._project(center, radius)
            tip, _ = self._project(center + forward * self.base_radius * .22, radius)
            marker = tuple(np.rint(marker[0]).astype(np.int32))
            tip = tuple(np.rint(tip[0]).astype(np.int32))
            color = (55, 225, 70) if verified else (0, 180, 255)
            cv2.arrowedLine(panel, marker, tip, color, 3, cv2.LINE_AA, tipLength=.3)
            cv2.circle(panel, marker, 9, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(panel, marker, 7, color, -1, cv2.LINE_AA)
            self.last_pose_center = tuple(float(v) for v in center)
        else:
            self.last_pose_center = None
        self.last_pose_state = ("PASS" if verified else
                                "UNVERIFIED" if pose is not None else
                                "COARSE CANDIDATE" if reference_eye is not None else "UNKNOWN")
        cv2.rectangle(panel, (0, 0), (self.map_width, 70), (28, 28, 28), -1)
        self._text(panel, "3DGS + CAMERA", 12, 27, color=(235, 235, 235), size=.65, thickness=2)
        self._text(panel, "-Z UP" if self.vertical_sign < 0 else "+Z UP",
                   self.map_width - 75, 27, color=(180, 180, 180), size=.37)
        status_color = (55, 225, 70) if verified else (0, 180, 255)
        self._text(panel, self.last_pose_state, 12, 54, color=status_color, size=.55, thickness=2)
        cv2.rectangle(panel, (0, self.height - 85), (self.map_width, self.height), (28, 28, 28), -1)
        if camera_center is None:
            self._text(panel, "No camera pose yet", 12, self.height - 59)
        else:
            self._text(panel, "Camera XYZ: " + ", ".join(f"{v:.1f}" for v in camera_center),
                       12, self.height - 59, size=.43)
            if not verified:
                self._text(panel, f"NOT CURRENT - prior search took {candidate_age_ms:.0f} ms",
                           12, self.height - 37, color=(0, 180, 255), size=.38)
        self._text(panel, "Blue: COARSE only   Green: PASS   Orange: stale PnP",
                   12, self.height - 15, size=.37)
        return panel

    def render(self, live_frame, current_result, *, search_result=None,
               diagnostics=None, candidate_age_ms=0., active_reference_name=None,
               coarse_reference_name=None, coarse_margin=0., coarse_mode="sift"):
        diagnostics = diagnostics or {}
        verified = (current_result is not None and current_result.status == "PASS")
        selected_result = current_result if verified else search_result
        pose = self._pose_from_result(selected_result)
        if pose is None:
            verified = False
        name = (active_reference_name if verified and active_reference_name in self.view_by_name
                else coarse_reference_name if coarse_reference_name in self.view_by_name
                else self._reference_name(diagnostics))
        map_panel = self._draw_map(pose, verified, name, candidate_age_ms)

        reference_panel = np.full((self.height, self.reference_width, 3), 28, dtype=np.uint8)
        self.last_reference_name = name
        reference = self._reference_image(name)
        cv2.rectangle(reference_panel, (0, 0), (self.reference_width, 70), (28, 28, 28), -1)
        self._text(reference_panel, "TRACK SOURCE" if verified else "COARSE VIEW ONLY",
                   10, 27, size=.5, thickness=2)
        self._text(reference_panel, name or "none", 10, 53, size=.46)
        if not verified and coarse_reference_name is not None:
            self._text(reference_panel,
                       f"{coarse_mode} top/next {coarse_margin:.2f} - NOT POSE",
                       10, 72, size=.35,
                       color=(0, 180, 255) if coarse_margin < 1.25 else (255, 190, 60))
        if reference is not None:
            h, w = reference.shape[:2]
            scale = min((self.reference_width - 12) / w, (self.height - 145) / h)
            resized = cv2.resize(reference, (round(w * scale), round(h * scale)))
            x = (self.reference_width - resized.shape[1]) // 2
            y = 80 + (self.height - 145 - resized.shape[0]) // 2
            reference_panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.rectangle(reference_panel, (0, self.height - 72),
                      (self.reference_width, self.height), (28, 28, 28), -1)
        quality = (diagnostics.get("quality") or {}) if name == diagnostics.get("selected") else {}
        if quality:
            self._text(reference_panel,
                       f"PnP {quality.get('inliers', 0)} inliers", 10,
                       self.height - 45, size=.42)
            reasons = quality.get("reasons") or []
            self._text(reference_panel, ", ".join(reasons)[:35] or "QC passed",
                       10, self.height - 21, size=.39,
                       color=(0, 180, 255) if reasons else (55, 225, 70))
        else:
            message = ("Tracking PASS; search QC differs" if verified else
                       "Reference candidate only" if
                       diagnostics.get("selected") or diagnostics.get("retrieval") else
                       "Example view; not selected")
            self._text(reference_panel, message, 10,
                       self.height - 38, size=.40)
        canvas = np.hstack((np.empty((self.height, self.live_width, 3), dtype=np.uint8),
                            map_panel, reference_panel))
        return self.update_live(canvas, live_frame)

    def update_live(self, canvas, live_frame):
        """Refresh the camera pane at capture rate without reprojecting 3DGS."""
        panel = canvas[:, :self.live_width]
        panel[:] = 24
        if live_frame is not None:
            source_h, source_w = live_frame.shape[:2]
            scale = min(self.live_width / source_w, (self.height - 60) / source_h)
            resized = cv2.resize(live_frame, (round(source_w * scale), round(source_h * scale)))
            x = (self.live_width - resized.shape[1]) // 2
            y = 50 + (self.height - 60 - resized.shape[0]) // 2
            panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.rectangle(panel, (0, 0), (self.live_width, 42), (28, 28, 28), -1)
        self._text(panel, "LIVE CAMERA", 12, 28, size=.65, thickness=2)
        return canvas

    def handle_key(self, key: int) -> bool:
        step = math.radians(6.)
        if key == ord("a"):
            self.yaw -= step
        elif key == ord("d"):
            self.yaw += step
        elif key == ord("w"):
            self.pitch = min(self.pitch + step, math.radians(84.))
        elif key == ord("s"):
            self.pitch = max(self.pitch - step, math.radians(-84.))
        elif key == ord("r"):
            self.yaw, self.pitch = math.radians(-42.), math.radians(24.)
        else:
            return False
        return True
