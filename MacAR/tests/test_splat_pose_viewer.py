import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splat_pose_viewer import SplatPoseViewer  # noqa: E402


class SplatPoseViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        image = np.full((48, 64, 3), (60, 120, 200), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(directory / "view_0000.png"), image))
        yy, xx = np.indices((48, 64))
        xyz = np.stack((xx, yy, np.ones_like(xx) * 10), axis=-1).astype(np.float32)
        np.savez(directory / "view_0000.npz", xyz=xyz)
        source = SimpleNamespace(
            views_path=directory,
            views=[dict(image_name="view_0000.png", data_name="view_0000.npz",
                        elevation_deg=0, distance_multiplier=1, eye=[20., 20., 100.])],
            manifest=dict(bounds=[[0, 0, 0], [64, 48, 20]]),
            _source_xyz_to_runtime=lambda points: points,
        )
        self.viewer = SplatPoseViewer(SimpleNamespace(global_localizer=source), height=300)
        self.live = np.zeros((120, 160, 3), dtype=np.uint8)

    @staticmethod
    def result(status):
        return SimpleNamespace(status=status, rvec=(0., 0., 0.), tvec=(0., 0., 100.))

    def test_search_shows_colored_3dgs_and_unverified_camera(self):
        diagnostic = dict(selected="view_0000.png",
                          quality=dict(inliers=9, reasons=["few_inliers"]))
        image = self.viewer.render(
            self.live, SimpleNamespace(status="FAIL", rvec=None, tvec=None),
            search_result=self.result("FAIL"), diagnostics=diagnostic,
            candidate_age_ms=5700,
        )
        self.assertEqual(image.shape, (300, 1240, 3))
        self.assertGreater(len(self.viewer.points), 0)
        self.assertEqual(self.viewer.last_pose_state, "UNVERIFIED")
        self.assertEqual(self.viewer.last_pose_center, (0., 0., -100.))
        self.assertEqual(self.viewer.last_reference_name, "view_0000.png")
        self.assertEqual(self.viewer.last_reference_camera_center, (20., 20., 100.))

    def test_only_current_pass_is_green_verified_pose(self):
        self.viewer.render(self.live, self.result("PASS"),
                           search_result=self.result("FAIL"))
        self.assertEqual(self.viewer.last_pose_state, "PASS")
        self.viewer.render(self.live, SimpleNamespace(status="FAIL", rvec=None, tvec=None))
        self.assertEqual(self.viewer.last_pose_state, "UNKNOWN")
        self.assertIsNone(self.viewer.last_pose_center)

    def test_active_track_reference_does_not_follow_unrelated_background_search(self):
        self.assertTrue(cv2.imwrite(str(self.viewer.views_path / 'view_0001.png'),
                                    np.zeros((48, 64, 3), dtype=np.uint8)))
        self.viewer.view_by_name['view_0001.png'] = dict(
            image_name='view_0001.png', eye=[50., 20., 100.])
        self.viewer.render(self.live, self.result('PASS'),
                           diagnostics=dict(selected='view_0000.png',
                                            retrieval=[dict(image_name='view_0000.png')]),
                           active_reference_name='view_0001.png')
        self.assertEqual(self.viewer.last_reference_name, 'view_0001.png')
        self.assertEqual(self.viewer.last_reference_camera_center, (50., 20., 100.))

    def test_live_coarse_reference_overrides_stale_search_reference(self):
        self.assertTrue(cv2.imwrite(str(self.viewer.views_path / 'view_0001.png'),
                                    np.zeros((48, 64, 3), dtype=np.uint8)))
        self.viewer.view_by_name['view_0001.png'] = dict(
            image_name='view_0001.png', eye=[50., 20., 100.])
        self.viewer.render(self.live, SimpleNamespace(status='FAIL', rvec=None, tvec=None),
                           diagnostics=dict(selected='view_0000.png'),
                           coarse_reference_name='view_0001.png', coarse_margin=1.42)
        self.assertEqual(self.viewer.last_reference_name, 'view_0001.png')

    def test_camera_thumbnail_refresh_does_not_reproject_map(self):
        canvas = self.viewer.render(self.live, self.result("PASS"))
        map_before = canvas[:, self.viewer.live_width:].copy()
        self.viewer.update_live(canvas, np.full_like(self.live, 255))
        np.testing.assert_array_equal(canvas[:, self.viewer.live_width:], map_before)
        self.assertTrue(self.viewer.handle_key(ord("a")))
        self.assertFalse(self.viewer.handle_key(ord("x")))

    def test_negative_z_manifest_displays_negative_z_as_up(self):
        self.viewer.vertical_sign = -1.
        above, _ = self.viewer._project([[32., 24., 0.]], self.viewer.base_radius)
        below, _ = self.viewer._project([[32., 24., 20.]], self.viewer.base_radius)
        self.assertLess(above[0, 1], below[0, 1])

    def test_optional_gaussian_pose_panel_is_next_to_live_camera(self):
        source = SimpleNamespace(
            views_path=self.viewer.views_path,
            views=self.viewer.views,
            manifest=dict(bounds=[[0, 0, 0], [64, 48, 20]]),
            _source_xyz_to_runtime=lambda points: points,
        )
        viewer = SplatPoseViewer(SimpleNamespace(global_localizer=source),
                                 height=300, show_pose_render=True)
        image = np.full((120, 160, 3), (20, 90, 210), np.uint8)
        canvas = viewer.render(self.live, self.result('PASS'),
                               gs_image=image, gs_age_ms=100.,
                               gs_render_ms=30., gs_status='READY')
        self.assertEqual(canvas.shape, (300, 1680, 3))
        pose_panel = canvas[90:210, viewer.live_width:viewer.live_width + 440]
        self.assertTrue(np.any(np.all(pose_panel == (20, 90, 210), axis=2)))

    def test_gaussian_pose_refresh_does_not_redraw_map(self):
        source = SimpleNamespace(
            views_path=self.viewer.views_path,
            views=self.viewer.views,
            manifest=dict(bounds=[[0, 0, 0], [64, 48, 20]]),
            _source_xyz_to_runtime=lambda points: points,
        )
        viewer = SplatPoseViewer(SimpleNamespace(global_localizer=source),
                                 height=300, show_pose_render=True)
        first = np.full((120, 160, 3), (20, 90, 210), np.uint8)
        second = np.full((120, 160, 3), (180, 40, 80), np.uint8)
        canvas = viewer.render(self.live, self.result('PASS'),
                               gs_image=first, gs_age_ms=100.,
                               gs_render_ms=30., gs_status='READY')
        map_start = viewer.live_width + viewer.pose_render_width
        map_before = canvas[:, map_start:].copy()
        viewer.update_pose_render(canvas, second, 50., 20., 'READY', verified=True)
        np.testing.assert_array_equal(canvas[:, map_start:], map_before)
        pose_panel = canvas[90:210, viewer.live_width:map_start]
        self.assertTrue(np.any(np.all(pose_panel == (180, 40, 80), axis=2)))


if __name__ == "__main__":
    unittest.main()
