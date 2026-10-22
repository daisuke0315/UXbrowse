import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_handoff import TrackingHandoff
from glb_virtual_live import GLBVirtualLiveLocalizer


class HandoffTests(unittest.TestCase):
    def seed(self, angle=0., count=100, center=(0., 0., -500.)):
        rvec = np.array([0., np.deg2rad(angle), 0.])
        rotation = cv2.Rodrigues(rvec)[0]
        return SimpleNamespace(rvec=rvec, tvec=-rotation@np.asarray(center),
                               image_points=np.zeros((count, 2)))

    def observe(self, handoff, now, angle=0., count=100, center=(0., 0., -500.)):
        return handoff.observe(now, self.seed(angle, count, center), 2000., 200., 20)

    def test_static_camera_never_extrapolates_pose(self):
        handoff=TrackingHandoff()
        for now in (0., .1, .2, .3):
            self.assertEqual(self.observe(handoff, now), [])
        self.assertIsNone(handoff.prediction)
        self.assertEqual(handoff.info['rotation_speed_deg_s'], 0.)

    def test_motion_predicts_reference_direction_but_does_not_mutate_pose(self):
        handoff=TrackingHandoff()
        self.observe(handoff, 0.)
        self.observe(handoff, .1, 2.)
        seed=self.seed(4.)
        handoff.observe(.2, seed, 2000., 200., 20)
        angle=np.rad2deg(handoff.prediction[0][1])
        self.assertAlmostEqual(angle, 16.)
        np.testing.assert_allclose(seed.rvec, [0., np.deg2rad(4.), 0.])
        future_center=-cv2.Rodrigues(handoff.prediction[0])[0].T@handoff.prediction[1]
        np.testing.assert_allclose(future_center, [0., 0., -500.], atol=1e-8)

    def test_point_loss_triggers_above_pnp_minimum(self):
        handoff=TrackingHandoff()
        self.observe(handoff, 0., count=100)
        self.observe(handoff, .1, count=92)
        reasons=self.observe(handoff, .2, count=75)
        self.assertIn('points_declining', reasons)
        self.assertGreater(handoff.info['points'], 20)
        self.assertLess(handoff.info['forecast_points'], 20)

    def test_translation_alone_produces_bounded_viewpoint_forecast(self):
        handoff=TrackingHandoff()
        self.observe(handoff,0.,center=(0.,0.,-500.))
        self.observe(handoff,.1,center=(10.,0.,-500.))
        self.observe(handoff,.2,center=(20.,0.,-500.))
        self.assertIsNotNone(handoff.prediction)
        center=-cv2.Rodrigues(handoff.prediction[0])[0].T@handoff.prediction[1]
        np.testing.assert_allclose(center,[50.,0.,-500.])
        self.assertAlmostEqual(handoff.info['lookahead_deg'],0.)

    def test_small_noisy_count_variation_does_not_trigger(self):
        handoff=TrackingHandoff()
        for now,count in ((0.,100),(.1,99),(.2,98),(.3,101)):
            self.assertEqual(self.observe(handoff, now, count=count), [])

    def test_translation_and_rotation_lookahead_are_bounded(self):
        handoff=TrackingHandoff()
        self.observe(handoff, 0.)
        self.observe(handoff, .1, 20., center=(100., 0., -500.))
        self.observe(handoff, .2, 40., center=(200., 0., -500.))
        rotation=cv2.Rodrigues(handoff.prediction[0])[0]
        center=-rotation.T@handoff.prediction[1]
        self.assertAlmostEqual(np.linalg.norm(center-[200.,0.,-500.]), 30.)
        self.assertAlmostEqual(handoff.info['lookahead_deg'], 12.)
        np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_reset_and_long_gap_do_not_reuse_motion(self):
        handoff=TrackingHandoff()
        for now,angle in ((0.,0.),(.1,2.),(.2,4.)):
            self.observe(handoff, now, angle)
        self.assertIsNotNone(handoff.prediction)
        self.observe(handoff, 5., 90.)
        self.assertIsNone(handoff.prediction)
        handoff.clear()
        self.assertFalse(handoff.info)
        self.assertFalse(handoff.samples)

    def test_handoff_keeps_current_view_and_adds_forward_neighbors(self):
        loc=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        loc._pose_guided_views=Mock(side_effect=[[0,1,2], [1,3,0]])
        self.assertEqual(loc._handoff_views(None,None,(None,None)), [0,1,3])
        loc._pose_guided_views=Mock(return_value=[0,1,2])
        self.assertEqual(loc._handoff_views(None,None,None), [0,1,2])
        loc._pose_guided_views.assert_called_once()

    def test_handoff_cannot_early_exit_after_matching_only_old_face(self):
        loc=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        loc.guided_optimization=True
        loc.pipeline_policy='validated'
        loc._localize_locked=Mock(return_value=(SimpleNamespace(status='PASS'),None))
        loc._guided_attempts(None,None,None,[(1,0,0)],np.zeros(3),np.zeros(3),handoff=True)
        self.assertFalse(loc._localize_locked.call_args.kwargs['allow_guided_early_stop'])
        np.testing.assert_array_equal(loc._localize_locked.call_args.kwargs['pose_prior'][0], np.zeros(3))


if __name__=='__main__': unittest.main()
