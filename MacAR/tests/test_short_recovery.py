import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tracking_recovery import ShortTrackRecovery
from realtime_pnp_server import TrackingSeed, TemporalGlobalPnPLocalizer
from glb_virtual_live import AsyncTemporalGLBLocalizer
from background_refresh import BackgroundRefresh
from pose_quality import assess_pose


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(9)
        gray=cv2.GaussianBlur(rng.integers(0,256,(160,160),np.uint8),(3,3),0)
        uv=np.array([[x,y] for x in range(30,131,20) for y in range(30,131,20)],np.float64)
        K=np.array([[120.,0,80],[0,120.,80],[0,0,1.]])
        xyz=np.column_stack(((uv-80)*2.5,np.full(len(uv),300.)))
        return TrackingSeed(gray,uv,xyz,np.zeros(3),np.zeros(3)),K

    def test_anchor_expires_attempts_are_bounded_and_calibration_must_match(self):
        seed,K=self.fixture();r=ShortTrackRecovery(attempts=2)
        r.remember(seed,'v.png',K,None,1.)
        self.assertIs(r.claim((160,160,3),K,None,1.1),seed)
        self.assertIsNone(r.claim((160,160,3),K,None,1.12))
        self.assertIs(r.claim((160,160,3),K,None,1.2),seed)
        self.assertIsNone(r.claim((160,160,3),K,None,1.3))
        r.remember(seed,'v.png',K,None,2.)
        self.assertIsNone(r.claim((160,160,3),K,None,3.))
        r.remember(seed,'v.png',K,None,4.)
        self.assertIsNone(r.claim((160,160,3),K*2,None,4.1))
        self.assertIsNone(r.seed)

    def test_real_flow_pnp_recovers_after_black_frame_without_publishing_stale_pose(self):
        seed,K=self.fixture()
        model=SimpleNamespace(map_data=None,mpr_renderer=None,last_debug=None,
            min_inliers=10,min_inlier_ratio=.2,ransac_threshold=4.,pipeline_policy='validated')
        model.validate_pose=lambda xyz,uv,r,t,ids,shape,k,d:assess_pose(
            xyz,uv,r,t,ids,shape,k,d,10,.2,3.)
        actor=AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        actor.tracker=TemporalGlobalPnPLocalizer(model)
        actor.background_refresh=BackgroundRefresh()
        actor.short_recovery=ShortTrackRecovery()
        actor.short_recovery.remember(seed,'v.png',K,None,1.)
        with patch('glb_virtual_live.time.monotonic',return_value=1.1):
            result=actor._recover_recent_track(np.zeros((160,160,3),np.uint8),K,None)
        self.assertIsNone(result)
        self.assertIsNone(actor.tracker.seed)
        moved=cv2.warpAffine(seed.gray,np.float32([[1,0,2],[0,1,0]]),(160,160))
        with patch('glb_virtual_live.time.monotonic',return_value=1.2):
            result,_=actor._recover_recent_track(cv2.cvtColor(moved,cv2.COLOR_GRAY2BGR),K,None)
        self.assertEqual(result.status,'PASS')
        self.assertEqual(result.detail,'TRACKING_RECOVERED')
        self.assertGreaterEqual(result.inliers,20)
        self.assertEqual(actor.short_recovery.recovered,1)
        np.testing.assert_allclose(result.tvec,[5.,0.,0.],atol=.4)
        np.testing.assert_array_equal(actor.tracker.seed.gray,moved)

    def test_nearby_hint_is_one_shot_and_never_reuses_old_pose_indefinitely(self):
        seed,K=self.fixture();r=ShortTrackRecovery()
        r.remember(seed,'v.png',K,None,1.)
        self.assertIsNotNone(r.nearby_hint(2.))
        r.nearby_used=True
        self.assertIsNone(r.nearby_hint(2.1))
        r.remember(seed,'v.png',K,None,3.)
        self.assertIsNone(r.nearby_hint(5.1))

    def test_nearby_search_uses_hint_only_for_view_selection(self):
        from threading import Lock
        from unittest.mock import Mock
        from glb_virtual_live import GLBVirtualLiveLocalizer
        worker=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        worker.lock=Lock();worker._pose_guided_views=Mock(return_value=[3,4,5])
        worker._quality_attempts=lambda attempt:attempt()
        worker._localize_locked=Mock(return_value=('result','canvas'))
        worker.localize_nearby(None,None,None,np.zeros(3),np.zeros(3))
        params=worker._localize_locked.call_args.kwargs
        self.assertIsNone(params['pose_prior'])
        self.assertFalse(params['optimize_guided'])
        self.assertEqual(params['ranked_override'],[(1.,0.,3),(1.,0.,4),(1.,0.,5)])


if __name__=='__main__':unittest.main()
