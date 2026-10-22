import sys
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glb_virtual_live import AsyncTemporalGLBLocalizer
from realtime_pnp_server import TrackingSeed


class SeedTransportTests(unittest.TestCase):
    def make_localizer(self, frames):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.search_frames = deque(frames)
        localizer.tracker = SimpleNamespace(flow_window=21, flow_levels=3, forward_backward_error=1.5)
        localizer.last_search_diagnostics = {}
        return localizer

    def test_real_lk_tracks_translation_through_intermediate_frames(self):
        rng = np.random.default_rng(17)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (160, 160), np.uint8), (3, 3), 0)
        points = np.array([[x, y] for y in (40, 60, 80) for x in (40, 60, 80)], np.float64)
        xyz = np.column_stack((points, np.ones(len(points))*100))
        seed = TrackingSeed(gray, points, xyz, np.zeros(3), np.zeros(3))
        frames = [cv2.warpAffine(gray, np.float32([[1, 0, 6*i], [0, 1, 3*i]]), (160, 160))
                  for i in range(1, 5)]
        localizer = self.make_localizer(frames)
        result = localizer._advance_search_seed(seed)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.image_points), 4)
        # Forward/backward checks may discard points near the moving border.
        # Retained XYZs must stay paired with their corresponding live pixel.
        np.testing.assert_allclose(result.image_points,
                                   result.object_points[:, :2] + [24, 12], atol=.2)
        np.testing.assert_allclose(seed.image_points, points)

    def test_long_history_is_bounded_and_includes_latest_frame(self):
        points = np.array([[40, 40], [60, 40], [40, 60], [60, 60]], np.float64)
        seed = TrackingSeed(np.zeros((128, 128), np.uint8), points,
                            np.zeros((4, 3)), np.zeros(3), np.zeros(3))
        localizer = self.make_localizer([np.full((128, 128), i, np.uint8) for i in range(1, 21)])
        def flow(before, after, p, _unused, **kwargs):
            delta = float(after[0, 0])-float(before[0, 0])
            return p + delta, np.ones((len(p), 1), np.uint8), None
        with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK', side_effect=flow) as mocked:
            result = localizer._advance_search_seed(seed)
        self.assertEqual(mocked.call_count, 16)
        np.testing.assert_allclose(result.image_points, points+20)
        self.assertEqual(localizer.last_search_diagnostics['seed_transport']['hops'], 8)

    def test_failed_hop_rejects_candidate(self):
        gray = np.zeros((128, 128), np.uint8)
        seed = TrackingSeed(gray, np.ones((4, 2))*50, np.zeros((4, 3)), np.zeros(3), np.zeros(3))
        localizer = self.make_localizer([gray])
        with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK', return_value=(None, None, None)):
            self.assertIsNone(localizer._advance_search_seed(seed))

    def test_transport_skips_short_occlusion_but_must_reach_current_image(self):
        rng=np.random.default_rng(4)
        gray=cv2.GaussianBlur(rng.integers(0,256,(160,160),np.uint8),(3,3),0)
        uv=np.array([[x,y] for x in (40,60,80) for y in (40,60,80)],np.float64)
        seed=TrackingSeed(gray,uv,np.column_stack((uv,np.ones(9))),np.zeros(3),np.zeros(3))
        moved=cv2.warpAffine(gray,np.float32([[1,0,4],[0,1,2]]),(160,160))
        localizer=self.make_localizer([np.zeros_like(gray),moved])
        localizer.resilient_transport=True
        result=localizer._advance_search_seed(seed)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.image_points,result.object_points[:,:2]+[4,2],atol=.1)
        self.assertEqual(localizer.last_search_diagnostics['seed_transport']['skipped_hops'],1)
        localizer.search_frames.append(np.zeros_like(gray))
        self.assertIsNone(localizer._advance_search_seed(seed))

    def test_viewer_does_not_launch_duplicate_inference_during_localization(self):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.coarse_future = None
        localizer.future = Future()
        localizer.coarse_submitted = 0.
        localizer.tracker = SimpleNamespace(seed=object())
        localizer.coarse_executor = SimpleNamespace(submit=Mock())
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer._update_coarse(np.zeros((4, 4, 3), np.uint8))
        localizer.coarse_executor.submit.assert_not_called()

    def test_old_viewer_result_cannot_overwrite_newer_search_preview(self):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.coarse_future = Future()
        localizer.coarse_future.set_result(('old.png', 1., 1., {}))
        localizer.coarse_started = 1.
        localizer.search_started = 2.
        localizer.coarse_submitted = 3.
        localizer.coarse_reference_name = 'new.png'
        localizer.future = None
        localizer.tracker = SimpleNamespace(seed=None)
        with patch('glb_virtual_live.time.monotonic', return_value=4.):
            localizer._update_coarse(np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(localizer.coarse_reference_name, 'new.png')
        self.assertIsNone(localizer.coarse_future)

    def test_long_search_retains_source_near_frames_at_bounded_memory(self):
        localizer = self.make_localizer([])
        localizer.search_frames = deque(maxlen=48)
        for value in range(150):
            localizer._remember_search_frame(np.full((4,4),value,np.uint8))
        self.assertLessEqual(len(localizer.search_frames),48)
        self.assertEqual(localizer.search_frames[0][0,0],0)
        self.assertEqual(localizer.search_frames[-1][0,0],149)
        self.assertEqual(sorted(int(frame[0,0]) for frame in localizer.search_frames),
                         [int(frame[0,0]) for frame in localizer.search_frames])

    def test_browser_can_disable_viewer_only_dino_without_disabling_search(self):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.coarse_future = localizer.future = None
        localizer.viewer_coarse_enabled = False
        localizer.coarse_submitted = 0.
        localizer.tracker = SimpleNamespace(seed=object())
        localizer.coarse_executor = SimpleNamespace(submit=Mock())
        with patch('glb_virtual_live.time.monotonic', return_value=2.):
            localizer._update_coarse(np.zeros((4,4,3),np.uint8))
        localizer.coarse_executor.submit.assert_not_called()

    def test_progress_ignores_last_jobs_candidate_when_new_job_is_queued(self):
        localizer = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.future = Future()
        localizer.search_started = 2.
        localizer.tracker = SimpleNamespace(seed=None)
        localizer.global_localizer = SimpleNamespace(progress_snapshot=lambda:dict(started_at=1.,stage='done',reference_index=42))
        localizer.last_search_age_ms = 123.
        localizer.last_search_result = None
        with patch('glb_virtual_live.time.monotonic',return_value=3.):
            progress=localizer.search_progress()
        self.assertTrue(progress['active'])
        self.assertEqual(progress['stage'],'queued')
        self.assertNotIn('reference_index',progress)
        self.assertEqual(progress['elapsed_ms'],1000.)


if __name__ == '__main__':
    unittest.main()
