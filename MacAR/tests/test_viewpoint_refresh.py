"""Observe real projected map regions and object-relative camera directions."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewpoint_refresh import ViewpointRefresh


def pose(center=(0., 0., -300.), rotation=(0., 0., 0.)):
    rvec = np.asarray(rotation, np.float64)
    return SimpleNamespace(rvec=rvec, tvec=-cv2.Rodrigues(rvec)[0]@np.asarray(center))


class ViewpointRefreshTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([(x, y, z) for x in np.linspace(-50, 50, 9)
                             for y in np.linspace(-50, 50, 9) for z in (-50, 50)])
        self.K = np.array([[150., 0., 100.], [0., 150., 100.], [0., 0., 1.]])
        self.shape = (200, 200, 3)
        self.anchor = pose()
        self.monitor = ViewpointRefresh(self.xyz, np.zeros(3))

    def reasons(self, current, anchor=None, prediction=None):
        return self.monitor.reasons(current, self.anchor if anchor is None else anchor,
            self.K, None, self.shape, 12., prediction)

    def test_stationary_camera_and_roll_do_not_trigger_refresh(self):
        self.assertEqual(self.reasons(self.anchor), [])
        self.assertEqual(self.reasons(pose(rotation=(0., 0., np.deg2rad(30.)))), [])
        self.assertAlmostEqual(self.monitor.info['direction_delta_deg'], 0.)

    def test_orbit_with_identical_rotation_triggers_direction_change(self):
        angle = np.deg2rad(20.)
        current = pose((300.*np.sin(angle), 0., -300.*np.cos(angle)))
        self.assertIn('view_direction_changed', self.reasons(current))
        self.assertAlmostEqual(self.monitor.info['direction_delta_deg'], 20.)
        np.testing.assert_array_equal(current.rvec, self.anchor.rvec)

    def test_range_change_alone_is_not_a_direction_change(self):
        self.assertNotIn('view_direction_changed', self.reasons(pose((0., 0., -400.))))

    def test_small_motion_stays_below_threshold(self):
        angle = np.deg2rad(2.)
        current = pose((300.*np.sin(angle), 0., -300.*np.cos(angle)))
        self.assertEqual(self.reasons(current), [])

    def test_background_detects_emerging_face_before_twelve_degree_threshold(self):
        angle = np.deg2rad(5.)
        current = pose((300.*np.sin(angle), 0., -300.*np.cos(angle)))
        reasons = self.monitor.reasons(current, self.anchor, self.K, None,
            self.shape, 12., early=True)
        self.assertIn('view_direction_emerging', reasons)
        self.assertNotIn('view_direction_changed', reasons)
        self.assertEqual(self.monitor.reasons(current, current, self.K, None,
            self.shape, 12., early=True), [])
        self.assertEqual(self.monitor.reasons(pose(rotation=(0., 0., .5)), self.anchor,
            self.K, None, self.shape, 12., early=True), [])

    def test_forecast_uses_camera_position_not_rotation(self):
        current = pose((30., 0., -300.))
        rotation_only = pose((30., 0., -300.), (0., .3, 0.))
        self.assertNotIn('view_direction_approaching', self.reasons(current, prediction=rotation_only))
        future = pose((60., 0., -300.))
        self.assertIn('view_direction_approaching', self.reasons(current, prediction=future))
        self.assertLess(self.monitor.info['direction_delta_deg'], 12.)

    def test_new_image_regions_trigger_without_camera_center_motion(self):
        xyz = np.array([(x, y, 0.) for x in np.linspace(-300, 300, 33)
                        for y in np.linspace(-80, 80, 17)])
        self.monitor = ViewpointRefresh(xyz, np.zeros(3))
        self.anchor = pose(rotation=(0., -.4, 0.))
        current = pose()
        reasons = self.reasons(current)
        self.assertIn('new_regions_visible', reasons)
        self.assertNotIn('view_direction_changed', reasons)
        self.assertGreaterEqual(self.monitor.info['new_regions'], 4)
        self.assertGreaterEqual(self.monitor.info['new_region_fraction'], .1)
        # Failure leaves the same anchor; success supplies a new verified anchor.
        self.assertIn('new_regions_visible', self.reasons(current))
        self.assertEqual(self.reasons(current, anchor=current), [])

    def test_failed_attempt_does_not_rebase_direction(self):
        current = pose((100., 0., -300.))
        for _ in range(3):
            self.assertIn('view_direction_changed', self.reasons(current))
        self.assertEqual(self.reasons(current, anchor=current), [])

    def test_border_jitter_does_not_create_new_regions(self):
        xyz = np.array([(x, y, 0.) for x in np.linspace(-150, 150, 33)
                        for y in np.linspace(-80, 80, 17)])
        self.monitor = ViewpointRefresh(xyz, np.zeros(3))
        self.assertEqual(self.reasons(pose(rotation=(0., .001, 0.))), [])

    def test_calibration_or_image_size_change_rebases_region_comparison(self):
        self.reasons(self.anchor)
        changed = self.K.copy(); changed[:2] *= .5
        self.assertEqual(self.monitor.reasons(self.anchor, self.anchor, changed, None,
                                             (100, 100, 3), 12., None), [])

    def test_reset_and_degenerate_camera_direction_are_safe(self):
        self.reasons(self.anchor)
        self.monitor.reset()
        self.assertEqual(self.reasons(self.anchor), [])
        monitor = ViewpointRefresh([], np.zeros(3))
        same = pose((0., 0., 0.))
        self.assertEqual(monitor.reasons(same, same, self.K, None, self.shape, 12., None), [])
        self.assertIsNone(monitor.info['direction_delta_deg'])


if __name__ == '__main__':
    unittest.main()
