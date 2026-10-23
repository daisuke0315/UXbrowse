import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from concurrent.futures import Future

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat_geometry import moments_to_geometry, sample_geometry, unproject
from glb_virtual_view_pnp import VirtualTransfer, make_transfers
from aliked_lightglue_pnp import solve_pose
from pose_quality import assess_pose
from glb_virtual_live import (AsyncTemporalGLBLocalizer, GLBVirtualLiveLocalizer,
                             crop_live_object, has_spread_triangle)
from realtime_pnp_server import TrackingSeed, LiveResult
from pose_appearance import evaluate_pose_appearance


class DynamicCoverageTests(unittest.TestCase):
    def test_triangle_separation_and_height(self):
        self.assertTrue(has_spread_triangle([[0, 0, 0], [30, 0, 0], [0, 30, 0]]))
        self.assertFalse(has_spread_triangle([[0, 0, 0], [29, 0, 0], [0, 29, 0]]))
        self.assertFalse(has_spread_triangle([[0, 0, 0], [40, 0, 0], [80, 0, 0]]))
        self.assertFalse(has_spread_triangle([[0, 0, 0], [100, 0, 0], [50, 10, 0]]))
        self.assertTrue(has_spread_triangle([[0, 0, 0], [100, 0, 0], [50, 15, 0]]))
        self.assertFalse(has_spread_triangle([[0, 0, 0], [0, 0, 0], [100, 0, 0]]))
        self.assertFalse(has_spread_triangle([[0, 0, 0], [30, 0, 0], [0, np.nan, 0]]))
        self.assertFalse(has_spread_triangle(np.empty((0, 3))))

    def test_matches_exhaustive_triangle_test(self):
        from itertools import combinations
        rng = np.random.default_rng(42)
        for _ in range(50):
            points = rng.normal(size=(8, 3)) * rng.uniform(5, 60)
            expected = False
            for a, b, c in combinations(points, 3):
                sides = [np.linalg.norm(a-b), np.linalg.norm(a-c), np.linalg.norm(b-c)]
                height = np.linalg.norm(np.cross(b-a, c-a)) / max(sides)
                if min(sides) >= 30 and height >= 15:
                    expected = True
                    break
            self.assertEqual(has_spread_triangle(points), expected)

    def fixture(self, points):
        image = np.zeros((32, 32, 3), np.uint8)
        seed = TrackingSeed(image[:, :, 0], np.zeros((len(points), 2)),
                            np.asarray(points), np.zeros(3), np.zeros(3))
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.global_localizer = SimpleNamespace(source_label='3DGS')
        localizer._update_coarse = lambda *_: None
        localizer._consume_completed = lambda *_: None
        localizer._submit = Mock()
        localizer.future = None
        localizer.pending_seed = None
        debug = SimpleNamespace(matched_xyz=np.asarray(points),
                                inlier_indices=frozenset(range(len(points))))
        localizer.tracker = SimpleNamespace(seed=seed,
            _track=lambda *_: ('PASS', image), frames_since_global=0, last_debug=debug)
        localizer.reseed_frames = 150
        localizer.replenish_frames = 1
        localizer.frames_since_replenish = 0
        localizer.replenish_mode = 'coverage'
        localizer.replenish_distance_mm = 30.
        localizer.replenish_height_mm = 15.
        localizer.replenish_cooldown = 1.
        localizer.last_replenish_attempt = 0.
        return localizer, image

    def test_good_coverage_ignores_interval_trigger(self):
        localizer, image = self.fixture([[0, 0, 0], [50, 0, 0], [0, 50, 0]])
        localizer.localize(image, np.eye(3), None)
        localizer._submit.assert_not_called()
        self.assertTrue(localizer.last_coverage_ok)

    def test_spread_3d_points_still_replenish_when_image_support_is_small(self):
        localizer, image = self.fixture([[0, 0, 0], [50, 0, 0], [0, 50, 0]])
        localizer.global_localizer.last_tracking_quality = {'coverage': .01}
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer.localize(image, np.eye(3), None)
        localizer._submit.assert_called_once()
        self.assertIn('image_coverage', localizer.last_replenish_reasons)

    def test_view_change_replenishes_even_with_good_point_distribution(self):
        localizer, image = self.fixture([[0, 0, 0], [50, 0, 0], [0, 50, 0]])
        localizer.global_localizer.manifest=dict(center=(0.,0.,500.))
        localizer.replenish_anchor = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([200.,0.,0.]))
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer.localize(image, np.eye(3), None)
        localizer._submit.assert_called_once()
        self.assertIn('view_direction_changed', localizer.last_replenish_reasons)

    def test_insufficient_coverage_schedules_guided_not_global(self):
        localizer, image = self.fixture([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer.localize(image, np.eye(3), None)
        self.assertIs(localizer._submit.call_args.kwargs['pose_prior'], localizer.tracker.seed)
        self.assertFalse(localizer.last_coverage_ok)

    def test_cooldown_and_running_search_prevent_repeated_jobs(self):
        localizer, image = self.fixture([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
        with patch('glb_virtual_live.time.monotonic', return_value=.5):
            localizer.localize(image, np.eye(3), None)
        localizer._submit.assert_not_called()
        localizer.future = Future()
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer.localize(image, np.eye(3), None)
        localizer._submit.assert_not_called()

    def test_outliers_do_not_count_as_coverage(self):
        localizer, image = self.fixture([[0, 0, 0], [10, 0, 0], [0, 10, 0],
                                        [100, 0, 0], [0, 100, 0]])
        localizer.tracker.last_debug.inlier_indices = frozenset([0, 1, 2])
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer.localize(image, np.eye(3), None)
        self.assertFalse(localizer.last_coverage_ok)
        localizer._submit.assert_called_once()


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

    def test_small_mean_error_does_not_hide_a_large_tail(self):
        uv = self.uv.copy()
        uv[:10, 0] += 15.
        quality = assess_pose(self.xyz, uv, self.rvec, self.tvec, np.arange(100),
            (480, 640), self.k, None, 10, .2, 6., max_p95_error=9.)
        self.assertLess(quality['mean_error'], 6.)
        self.assertIn('reprojection_tail', quality['reasons'])

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


class PoseGuidedReplenishmentTests(unittest.TestCase):
    def test_object_crop_preserves_original_pixel_coordinates(self):
        image = np.zeros((100, 200, 3), np.uint8)
        cropped, (x0, y0, x1, y1) = crop_live_object(
            image, (.25, .20, .50, .60), padding=0.)
        self.assertEqual((x0, y0, x1, y1), (50, 20, 150, 80))
        self.assertEqual(cropped.shape, (60, 100, 3))
        np.testing.assert_allclose(np.array([10., 15.]) + (x0, y0),
                                   (60., 35.))
        self.assertIsNone(crop_live_object(image, None))

    def test_active_track_schedules_guided_search_without_replacing_seed(self):
        image = np.zeros((32, 32, 3), np.uint8)
        seed = TrackingSeed(np.zeros((32, 32), np.uint8),
            np.zeros((4, 2)), np.zeros((4, 3)), np.zeros(3), np.zeros(3))
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer._update_coarse = lambda *_: None
        localizer._consume_completed = lambda *_: None
        localizer._submit = Mock()
        localizer.future = None
        localizer.pending_seed = None
        localizer.tracker = SimpleNamespace(seed=seed, _track=lambda *_: ('PASS', image),
            frames_since_global=0, last_debug=None)
        localizer.reseed_frames = 150
        localizer.replenish_frames = 1
        localizer.frames_since_replenish = 0
        localizer.last_debug = None
        result = localizer.localize(image, np.eye(3), None)
        self.assertEqual(result[0], 'PASS')
        self.assertIs(localizer.tracker.seed, seed)
        self.assertIs(localizer._submit.call_args.kwargs['pose_prior'], seed)

    def test_completed_guided_search_does_not_replace_working_track(self):
        image = np.zeros((32, 32, 3), np.uint8)
        old = TrackingSeed(np.zeros((32, 32), np.uint8), np.zeros((4, 2)),
                           np.zeros((4, 3)), np.zeros(3), np.zeros(3))
        new = TrackingSeed(np.zeros((32, 32), np.uint8), np.ones((4, 2)),
                           np.ones((4, 3)), np.zeros(3), np.zeros(3))
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.future = Future()
        localizer.future.set_result(('PASS', image))
        localizer.global_localizer = SimpleNamespace(last_debug=None,
            last_diagnostics={}, last_tracking_seed=new, source_label='3DGS',
            last_reference_name='near.png')
        tracking = LiveResult('PASS','TRACKING',4,4,4,1.,0.,(0.,0.,0.),(0.,0.,0.),None,1.,'HIDDEN')
        localizer.tracker = SimpleNamespace(seed=old, _copy_seed=lambda value: value,
            _track=lambda *_: (tracking, image), frames_since_global=0, last_debug=None)
        localizer.search_mode = 'guided'
        localizer.search_started = 10.
        localizer.max_seed_age = 5.
        localizer.reseed_frames = 150
        localizer.replenish_frames = 0
        localizer.frames_since_replenish = 0
        localizer.pending_seed = None
        localizer.pending_reference_name = None
        localizer.active_reference_name = 'old.png'
        localizer._update_coarse = lambda *_: None
        localizer._replenish_current_seed = Mock(return_value=4)
        localizer.last_replenish_diagnostics = {}
        with patch('glb_virtual_live.time.monotonic', return_value=11.):
            result = localizer.localize(image, np.eye(3), None)
        self.assertEqual(result[0].status, 'PASS')
        self.assertIs(localizer.tracker.seed, old)
        localizer._replenish_current_seed.assert_called_once()
        self.assertIsNone(localizer.pending_seed)

    def test_pose_guided_ranking_prefers_near_camera_and_rotation(self):
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.source_type = '3dgs'
        localizer.runtime_transform = 'identity'
        localizer.max_coarse_rotation_deg = 35.
        localizer.manifest = {'radius': 100.}
        near = np.eye(4)
        near[:3, 3] = (0., 0., 100.)
        far = np.eye(4)
        far[:3, 3] = (100., 0., 100.)
        wrong_angle = np.eye(4)
        wrong_angle[:3, :3] = cv2.Rodrigues(
            np.array([0., 0., np.pi/2]))[0]
        localizer.views = [{'world_to_camera': pose.tolist()}
                           for pose in (far, wrong_angle, near)]
        indices = localizer._pose_guided_views(np.zeros(3),
                                                np.array([0., 0., 100.]))
        self.assertEqual(indices[0], 2)
        self.assertNotIn(1, indices)

    def test_replenishment_adds_only_distinct_projected_points(self):
        image = np.zeros((100, 100, 3), np.uint8)
        matrix = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        old_uv = np.array([[20., 20.], [80., 20.], [20., 80.], [80., 80.]])
        new_uv = np.array([[35., 35.], [65., 35.], [35., 65.], [65., 65.]])
        candidate_uv = np.vstack((old_uv[:1], new_uv, [[90., 50.]]))
        old_xyz = np.column_stack((old_uv-50., np.full(4, 100.)))
        candidate_xyz = np.column_stack((candidate_uv-50.,
                                         np.full(len(candidate_uv), 100.)))
        candidate_xyz[-1, 0] = 200.  # Feature at (90, 50) is not this XYZ.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current = TrackingSeed(gray, old_uv, old_xyz, np.zeros(3), np.zeros(3))
        candidate = TrackingSeed(gray, candidate_uv, candidate_xyz,
                                 np.zeros(3), np.zeros(3))
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.tracker = SimpleNamespace(seed=current, flow_window=31,
            flow_levels=3, forward_backward_error=1.5,tracking_iterations=150)
        localizer.global_localizer = SimpleNamespace(ransac_threshold=8.,
            validate_pose=lambda xyz,uv,r,t,ids,shape,k,d:
                assess_pose(xyz,uv,r,t,ids,shape,k,d,4,.2,3.,max_p95_error=4.5))

        def unchanged_flow(_before, _after, points, _unused, **_kwargs):
            return points.copy(), np.ones((len(points), 1), np.uint8), None

        with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK',
                   side_effect=unchanged_flow):
            added = localizer._replenish_current_seed(candidate, image, matrix, None)
        self.assertEqual(added, 4)
        self.assertEqual(len(localizer.tracker.seed.image_points), 8)
        self.assertEqual(set(map(tuple,localizer.tracker.seed.image_points)),
                         set(map(tuple,np.vstack((old_uv,new_uv)))))

    def test_verified_duplicate_points_refresh_angle_anchor_without_new_points(self):
        image = np.zeros((100, 100, 3), np.uint8)
        matrix = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        uv = np.array([[x, y] for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)], float)
        xyz = np.column_stack((uv-50., np.full(len(uv), 100.)))
        seed = TrackingSeed(image[:, :, 0], uv, xyz, np.zeros(3), np.zeros(3))
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.tracker = SimpleNamespace(seed=seed, flow_window=31,
            flow_levels=3, forward_backward_error=1.5,tracking_iterations=150)
        def unchanged(_before, _after, points, _unused, **kwargs):
            return points.copy(), np.ones((len(points), 1), np.uint8), None
        for accepted in (False, True):
            with self.subTest(accepted=accepted):
                anchor = SimpleNamespace(rvec=np.array([0., 0., .3]), tvec=np.zeros(3))
                localizer.replenish_anchor = anchor
                localizer.global_localizer = SimpleNamespace(ransac_threshold=8., min_inliers=10,
                    validate_pose=lambda *args: {'passed': accepted})
                with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK', side_effect=unchanged):
                    self.assertEqual(localizer._replenish_current_seed(seed, image, matrix, None), 0)
                if accepted:
                    np.testing.assert_allclose(localizer.replenish_anchor.rvec, seed.rvec)
                else:
                    self.assertIs(localizer.replenish_anchor, anchor)
                if not accepted:
                    self.assertIs(localizer.tracker.seed, seed)
                else:
                    self.assertEqual(len(localizer.tracker.seed.image_points),len(seed.image_points))


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
