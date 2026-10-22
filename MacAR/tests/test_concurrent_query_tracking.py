import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concurrent_query_tracking import SearchPointTracks
from realtime_pnp_server import TrackingSeed
from glb_virtual_live import AsyncTemporalGLBLocalizer
from background_refresh import BackgroundRefresh


class ConcurrentQueryTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(45)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), np.uint8), (3, 3), 0)
        points = np.array([[x, y] for y in (70, 100, 130) for x in (70, 110, 150)], np.float64)
        tracks = SearchPointTracks(flow_window=21, flow_levels=3, fb_error=1.5)
        tracks.begin(1.)
        tracks.publish(1., gray, points)
        tracks.advance([gray], 0.)
        seed = TrackingSeed(gray, points, np.column_stack((points, np.ones(len(points))*500)),
                            np.zeros(3), np.zeros(3))
        return tracks, gray, points, seed

    def test_twenty_second_search_transports_same_xyz_to_current_pixels(self):
        tracks, gray, points, seed = self.fixture()
        # 300 processed frames at 15 FPS, while LightGlue has not returned XYZ.
        for i in range(1, 301):
            dx = 45.*np.sin(i/300.*np.pi/2)
            dy = 20.*np.sin(i/300.*np.pi/2)
            frame = cv2.warpAffine(gray, np.float32([[1,0,dx],[0,1,dy]]), (320,240))
            tracks.advance([frame], i/15.)
        result, info = tracks.attach(seed)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.image_points, result.object_points[:,:2]+[45,20], atol=.6)
        self.assertGreaterEqual(len(result.image_points), 8)
        self.assertGreaterEqual(info['frames'], 300)
        np.testing.assert_array_equal(seed.image_points, points)

    def test_lost_tracks_are_never_reused_after_occlusion(self):
        tracks, gray, _, seed = self.fixture()
        tracks.advance([np.zeros_like(gray)], 1.)
        tracks.advance([gray], 2.)
        result, info = tracks.attach(seed)
        self.assertIsNone(result)
        self.assertNotEqual(info['reason'], 'transported')

    def test_reset_rejects_late_worker_publication(self):
        tracks, gray, points, seed = self.fixture()
        tracks.begin(2.)
        tracks.publish(1., gray, points)
        tracks.advance([gray], 21.)
        self.assertFalse(tracks.initialized)
        self.assertIsNone(tracks.attach(seed)[0])

    def test_xyz_is_attached_by_source_coordinate_not_array_order(self):
        tracks, gray, points, seed = self.fixture()
        moved = cv2.warpAffine(gray, np.float32([[1,0,5],[0,1,3]]), (320,240))
        tracks.advance([moved], 1.)
        order = np.array([8,2,6,0,7,1,5,3,4])
        subset = TrackingSeed(gray, points[order], seed.object_points[order], seed.rvec, seed.tvec)
        result, _ = tracks.attach(subset)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.image_points, result.object_points[:,:2]+[5,3], atol=.2)
        self.assertTrue(all(any(np.array_equal(p, q) for q in subset.object_points)
                            for p in result.object_points))

    def test_unrelated_or_untracked_points_are_not_given_xyz(self):
        tracks, _, _, seed = self.fixture()
        seed = TrackingSeed(seed.gray, seed.image_points+.5, seed.object_points, seed.rvec, seed.tvec)
        self.assertIsNone(tracks.attach(seed)[0])

    def test_resolution_change_rejects_transport(self):
        tracks, _, _, seed = self.fixture()
        tracks.advance([np.zeros((120,160), np.uint8)], 1.)
        self.assertIsNone(tracks.attach(seed)[0])

    def test_repeat_publication_does_not_restart_old_tracks(self):
        tracks, gray, points, seed = self.fixture()
        moved = cv2.warpAffine(gray, np.float32([[1,0,5],[0,1,3]]), (320,240))
        tracks.advance([moved], 1.)
        tracks.publish(1., gray, points)
        tracks.advance([moved], 2.)
        result, _ = tracks.attach(seed)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.image_points, result.object_points[:,:2]+[5,3], atol=.2)

    def test_matching_pixels_from_other_source_image_are_rejected(self):
        tracks, gray, points, seed = self.fixture()
        seed = TrackingSeed(np.zeros_like(gray), points, seed.object_points, seed.rvec, seed.tvec)
        result, info = tracks.attach(seed)
        self.assertIsNone(result)
        self.assertEqual(info['reason'],'source_image_mismatch')


class CurrentPoseHandoffTests(unittest.TestCase):
    def fixture(self, qc=True, status='PASS'):
        rng = np.random.default_rng(15)
        xyz = rng.uniform([-60,-40,-30], [60,40,30], (40,3))
        matrix = np.array([[500.,0,160],[0,500.,120],[0,0,1]])
        source_uv = cv2.projectPoints(xyz, np.zeros(3), np.array([0.,0.,500.]), matrix, None)[0].reshape(-1,2)
        rv, tv = np.array([0.,.2,0.]), np.array([30.,5.,500.])
        uv = cv2.projectPoints(xyz, rv, tv, matrix, None)[0].reshape(-1,2)
        gray = np.zeros((240,320), np.uint8)
        seed = TrackingSeed(gray, source_uv, xyz, np.zeros(3), np.array([0.,0.,500.]))
        transported = TrackingSeed(gray, uv, xyz, seed.rvec, seed.tvec)
        loc = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        loc.query_tracks = SimpleNamespace(initialized=True,
            attach=Mock(return_value=(transported, dict(method='continuous_query_lk',reason='transported'))),
            begin=Mock())
        loc.global_localizer = SimpleNamespace(last_debug=None, last_diagnostics={},
            last_tracking_seed=seed, last_retrieval=[], last_reference_name='test.png',
            ransac_threshold=4., validate_pose=Mock(return_value=dict(passed=qc,reasons=[])))
        loc.tracker = SimpleNamespace(seed=None, _copy_seed=lambda s:s, tracking_iterations=150)
        loc.future=Future();loc.future.set_result((SimpleNamespace(status=status),None))
        loc.search_started=0.;loc.search_source_stamp=0.;loc.search_mode='global'
        loc.max_seed_age=10.;loc.replenish_mode='background'
        loc.background_refresh=BackgroundRefresh();loc.search_frames=[]
        loc._diagnostic=Mock()
        return loc, cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR), matrix, rv, tv

    def test_old_search_is_accepted_only_after_independent_current_pnp(self):
        loc,image,matrix,rv,tv=self.fixture()
        with patch('glb_virtual_live.time.monotonic',return_value=21.):
            loc._consume_completed(image,matrix,None)
        self.assertIsNotNone(loc.tracker.seed)
        np.testing.assert_allclose(loc.tracker.seed.rvec.ravel(),rv,atol=1e-5)
        np.testing.assert_allclose(loc.tracker.seed.tvec.ravel(),tv,atol=1e-4)
        self.assertGreater(loc.last_search_age_ms,10000)
        self.assertTrue(loc.last_search_diagnostics['seed_transport']['quality']['passed'])

    def test_current_qc_failure_never_initializes_tracking(self):
        loc,image,matrix,_,_=self.fixture(qc=False)
        with patch('glb_virtual_live.time.monotonic',return_value=21.):
            loc._consume_completed(image,matrix,None)
        self.assertIsNone(loc.tracker.seed)

    def test_failed_lightglue_cannot_be_rescued_by_flow(self):
        loc,image,matrix,_,_=self.fixture(status='FAIL')
        with patch('glb_virtual_live.time.monotonic',return_value=21.):
            loc._consume_completed(image,matrix,None)
        self.assertIsNone(loc.tracker.seed)
        loc.query_tracks.attach.assert_not_called()

    def test_transported_seed_must_reach_actual_latest_image(self):
        loc,image,matrix,_,_=self.fixture()
        image[:]=100
        with patch('glb_virtual_live.time.monotonic',return_value=21.):
            loc._consume_completed(image,matrix,None)
        self.assertIsNone(loc.tracker.seed)
        self.assertEqual(loc.last_search_diagnostics['seed_transport']['reason'],'latest_frame_not_reached')

    def test_old_search_does_not_overwrite_healthy_track_before_merge(self):
        loc,image,matrix,_,_=self.fixture()
        old=loc.global_localizer.last_tracking_seed
        loc.tracker.seed=old
        with patch('glb_virtual_live.time.monotonic',return_value=21.):
            loc._consume_completed(image,matrix,None)
            self.assertLess(loc._pending_seed_age_ms(),10000)
        self.assertIs(loc.tracker.seed,old)
        self.assertIsNotNone(loc.pending_seed)

    def test_worker_finishing_between_foreground_update_and_consume_keeps_snapshot(self):
        loc,image,matrix,rv,tv=self.fixture()
        source=loc.global_localizer.last_tracking_seed
        target=loc.query_tracks.attach.return_value[0].image_points
        tracks=SearchPointTracks(flow_window=21,flow_levels=3,fb_error=1.5)
        tracks.begin(1.)
        tracks.advance([source.gray],20.)  # Publication has not arrived yet.
        tracks.publish(1.,source.gray,source.image_points)
        loc.query_tracks=tracks
        delta=(target-source.image_points).astype(np.float32).reshape(-1,1,2)
        count=0
        def flow(_before,_after,points,_unused,**_kwargs):
            nonlocal count
            count+=1
            return points+(delta if count%2 else -delta),np.ones((len(points),1),np.uint8),None
        with patch('glb_virtual_live.time.monotonic',return_value=21.), \
             patch('concurrent_query_tracking.cv2.calcOpticalFlowPyrLK',side_effect=flow):
            loc._consume_completed(image,matrix,None)
        self.assertIsNotNone(loc.tracker.seed)
        np.testing.assert_allclose(loc.tracker.seed.rvec.ravel(),rv,atol=1e-5)
        np.testing.assert_allclose(loc.tracker.seed.tvec.ravel(),tv,atol=1e-4)


if __name__ == '__main__':
    unittest.main()
