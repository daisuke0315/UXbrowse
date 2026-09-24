import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from concurrent.futures import Future

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat_geometry import moments_to_geometry, sample_geometry, unproject
from glb_virtual_view_pnp import VirtualTransfer, make_transfers
from aliked_lightglue_pnp import solve_pose
from pose_quality import assess_pose
from glb_virtual_live import AsyncTemporalGLBLocalizer, GLBVirtualLiveLocalizer
from pose_appearance import evaluate_pose_appearance


class SplatGeometryTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[500., 0, 2], [0, 500., 2], [0, 0, 1]])
        self.pose = np.eye(4)

    def test_depth_moments_preserve_pixel_ray(self):
        moments = np.zeros((5, 5, 4), np.float32)
        moments[:] = (.2, .04, 1., 1.)
        geometry = moments_to_geometry(moments, self.k, self.pose)
        points = geometry['xyz'].reshape(-1, 3)
        projected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.k, None)[0].reshape(5, 5, 2)
        yy, xx = np.indices((5, 5))
        np.testing.assert_allclose(projected, np.stack((xx, yy), axis=-1), atol=1e-5)

    def test_transparency_and_mixed_surfaces_are_invalid(self):
        moments = np.array([[[.1, .02, .5, .5], [.2, .05, 1, 1]]], np.float32)
        result = moments_to_geometry(moments, self.k, self.pose)
        self.assertFalse(result['valid'].any())
        self.assertTrue(np.isnan(result['xyz']).all())

    def geometry(self):
        return dict(depth=np.full((5, 5), 200.), valid=np.ones((5, 5), bool),
                    camera_matrix=self.k, camera_to_world=self.pose)

    def test_subpixel_unprojection(self):
        uv = np.array([[1.2, 2.3], [2.75, 1.25]])
        xyz, indices, _ = sample_geometry(uv, self.geometry())
        projection = cv2.projectPoints(np.asarray(xyz), np.zeros(3), np.zeros(3), self.k, None)[0].reshape(-1, 2)
        np.testing.assert_allclose(projection, uv, atol=1e-8)
        self.assertEqual(indices, [0, 1])

    def test_no_nearest_neighbor_borrow_at_hole(self):
        geometry = self.geometry()
        geometry['valid'][2, 2] = False
        xyz, _, rejected = sample_geometry([[1.5, 1.5]], geometry)
        self.assertEqual(xyz, [])
        self.assertEqual(rejected['invalid_support'], 1)

    def test_reject_depth_edge(self):
        geometry = self.geometry()
        geometry['depth'][2, 2] = 400
        xyz, _, rejected = sample_geometry([[1.5, 1.5]], geometry)
        self.assertEqual(xyz, [])
        self.assertEqual(rejected['depth_edge'], 1)

    def test_reject_outside_or_nan(self):
        xyz, _, rejected = sample_geometry([[-1, 0], [4, 4], [np.nan, 2]], self.geometry())
        self.assertEqual(xyz, [])
        self.assertEqual(rejected['outside'], 3)

    def test_rotated_camera_roundtrip(self):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([.3, -.4, .1]))[0]
        pose[:3, 3] = [10, 20, -30]
        uv = np.array([[1.2, 3.4]])
        xyz = unproject(uv, np.array([200.]), self.k, pose)
        inverse = np.linalg.inv(pose)
        projected = cv2.projectPoints(xyz, cv2.Rodrigues(inverse[:3, :3])[0], inverse[:3, 3], self.k, None)[0]
        np.testing.assert_allclose(projected.reshape(-1, 2), uv, atol=1e-8)

    def test_transfer_uses_geometry_not_legacy_xyz(self):
        transfers = make_transfers(np.array([[10, 20]]), np.array([[1.5, 1.5]]),
            np.array([.8]), np.zeros((5, 5, 3)), 2, geometry=self.geometry())
        self.assertAlmostEqual(transfers[0].xyz[2], 200.)


class PoseChecks(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1]])
        rng = np.random.default_rng(31)
        self.xyz = rng.uniform(-80, 80, (100, 3))
        self.rvec, self.tvec = np.array([.2, .1, -.1]), np.array([0., 0., 500.])
        self.uv = cv2.projectPoints(self.xyz, self.rvec, self.tvec, self.k, None)[0].reshape(-1, 2)
        self.transfers = [VirtualTransfer(i, tuple(x), tuple(uv), tuple(uv), 1.)
                          for i, (x, uv) in enumerate(zip(self.xyz, self.uv))]

    def test_known_pose_and_outliers(self):
        transfers = list(self.transfers)
        for i in range(30):
            t = transfers[i]
            transfers[i] = VirtualTransfer(i, t.xyz, (float(i*9), float(i*5)), t.reference_uv, 1.)
        pose = solve_pose(transfers, self.k, None, 2., 10000)
        self.assertIsNotNone(pose)
        self.assertGreaterEqual(len(pose.inlier_indices), 70)
        self.assertLess(pose.mean_error, .01)
        np.testing.assert_allclose(pose.tvec.ravel(), self.tvec, atol=.01)

    def test_divergent_refinement_is_not_used(self):
        with patch('cv2.solvePnPRefineLM', return_value=(np.zeros((3, 1)), np.ones((3, 1))*10000)):
            pose = solve_pose(self.transfers, self.k, None, 2., 1000)
        self.assertIsNotNone(pose)
        self.assertLess(pose.mean_error, .01)

    def quality(self, xyz=None, uv=None, tvec=None):
        return assess_pose(self.xyz if xyz is None else xyz, self.uv if uv is None else uv,
            self.rvec, self.tvec if tvec is None else tvec, np.arange(100),
            (480, 640), self.k, None, 10, .2, 6.)

    def test_good_pose_passes(self):
        self.assertTrue(self.quality()['passed'])

    def test_low_reprojection_is_not_enough_for_clustered_points(self):
        xyz = self.xyz*.001
        uv = cv2.projectPoints(xyz, self.rvec, self.tvec, self.k, None)[0].reshape(-1, 2)
        quality = self.quality(xyz, uv)
        self.assertIn('clustered_image_support', quality['reasons'])

    def test_behind_camera_is_rejected(self):
        tvec = np.array([0., 0., -500.])
        uv = cv2.projectPoints(self.xyz, self.rvec, tvec, self.k, None)[0].reshape(-1, 2)
        quality = self.quality(uv=uv, tvec=tvec)
        self.assertIn('behind_camera', quality['reasons'])

    def test_collinear_support_is_rejected(self):
        xyz = self.xyz.copy()
        xyz[:, 1:] = 0
        uv = cv2.projectPoints(xyz, self.rvec, self.tvec, self.k, None)[0].reshape(-1, 2)
        self.assertIn('collinear_3d_support', self.quality(xyz, uv)['reasons'])

    def test_incompatible_pose_distance(self):
        first = SimpleNamespace(rvec=np.zeros(3), tvec=np.zeros(3))
        second = SimpleNamespace(rvec=np.array([0., 0., np.pi/2]), tvec=np.array([100., 0, 0]))
        angle, distance = GLBVirtualLiveLocalizer._pose_distance(first, second)
        self.assertAlmostEqual(angle, 90.)
        self.assertAlmostEqual(distance, 100.)

    def test_coarse_view_rejects_incompatible_pnp_rotation(self):
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.source_type = '3dgs'
        localizer.runtime_transform = 'identity'
        localizer.max_coarse_rotation_deg = 35.
        view = {'world_to_camera': np.eye(4).tolist()}
        aligned = SimpleNamespace(rvec=np.zeros(3))
        inverted = SimpleNamespace(rvec=np.array([0., 0., np.pi / 2]))
        self.assertTrue(localizer._coarse_pose_check(aligned, view)['passed'])
        self.assertFalse(localizer._coarse_pose_check(inverted, view)['passed'])


class SeedAgeTests(unittest.TestCase):
    def make_async(self):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.future = Future()
        localizer.future.set_result(('PASS', None))
        localizer.global_localizer = SimpleNamespace(last_debug=None, last_tracking_seed='new',
            pipeline_policy='validated', source_label='3DGS', last_reference_name='view_new.png')
        localizer.tracker = SimpleNamespace(seed='existing', _copy_seed=lambda value: value,
                                           frames_since_global=10, last_debug=None)
        localizer.search_started = 10.
        localizer.max_seed_age = 5.
        localizer.reseed_frames = 150
        localizer.pending_seed = None
        localizer.pending_reference_name = None
        localizer.active_reference_name = 'view_old.png'
        return localizer

    def test_stale_seed_does_not_replace_existing_tracking(self):
        localizer = self.make_async()
        with patch('glb_virtual_live.time.monotonic', return_value=20.):
            localizer._consume_completed()
        self.assertEqual(localizer.tracker.seed, 'existing')
        self.assertEqual(localizer.pending_seed, 'new')

    def test_stale_seed_recovers_after_active_track_fails(self):
        localizer = self.make_async()
        calls = []
        def track(*_):
            calls.append(localizer.tracker.seed)
            return None if localizer.tracker.seed == 'existing' else ('PASS', None)
        localizer.tracker._track = track
        localizer._submit = lambda *_: None
        localizer._update_coarse = lambda *_: None
        with patch('glb_virtual_live.time.monotonic', return_value=20.):
            result = localizer.localize(np.zeros((4, 4, 3), np.uint8), np.eye(3), None)
        self.assertEqual(result[0], 'PASS')
        self.assertEqual(calls, ['existing', 'new'])
        self.assertEqual(localizer.active_reference_name, 'view_new.png')

    def test_slow_initial_search_seed_is_offered_for_current_frame_validation(self):
        localizer = self.make_async()
        localizer.tracker.seed = None
        with patch('glb_virtual_live.time.monotonic', return_value=20.):
            localizer._consume_completed()
        self.assertEqual(localizer.tracker.seed, 'new')

    def test_fresh_seed_is_held_until_current_frame_validation(self):
        localizer = self.make_async()
        with patch('glb_virtual_live.time.monotonic', return_value=11.):
            localizer._consume_completed()
        self.assertEqual(localizer.tracker.seed, 'existing')
        self.assertEqual(localizer.pending_seed, 'new')

    def test_failed_fresh_reseed_preserves_working_track(self):
        localizer = self.make_async()
        calls = []
        def track(*_):
            calls.append(localizer.tracker.seed)
            return None if localizer.tracker.seed == 'new' else ('PASS', None)
        localizer.tracker._track = track
        localizer._submit = lambda *_: None
        localizer._update_coarse = lambda *_: None
        with patch('glb_virtual_live.time.monotonic', return_value=11.):
            result = localizer.localize(np.zeros((4, 4, 3), np.uint8), np.eye(3), None)
        self.assertEqual(result[0], 'PASS')
        self.assertEqual(calls, ['new', 'existing'])
        self.assertEqual(localizer.active_reference_name, 'view_old.png')


class AppearanceAuditTests(unittest.TestCase):
    def make_data(self):
        rng = np.random.default_rng(6)
        a = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        matrix = np.array([[300., 0, 128], [0, 300., 128], [0, 0, 1]])
        yy, xx = np.indices(a.shape[:2])
        xyz = unproject(np.stack((xx, yy), axis=-1), np.full((256, 256), 200.), matrix, np.eye(4))
        return a, b, matrix, xyz

    def test_identical_projection_has_high_correlation(self):
        a, _, matrix, xyz = self.make_data()
        result = evaluate_pose_appearance(a, a, xyz, np.zeros(3), np.zeros(3), matrix, None)
        self.assertGreater(result['tiles'], 0)
        self.assertGreater(result['median_correlation'], .99)

    def test_unrelated_texture_does_not_supply_strong_evidence(self):
        a, b, matrix, xyz = self.make_data()
        result = evaluate_pose_appearance(a, b, xyz, np.zeros(3), np.zeros(3), matrix, None)
        self.assertLess(abs(result['median_correlation']), .1)


if __name__ == '__main__':
    unittest.main()
