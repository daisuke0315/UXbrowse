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
from search_frame_quality import RecentSearchFrames, image_quality, correct_luminance, PHOTOMETRIC_MODE
from glb_virtual_live import AsyncTemporalGLBLocalizer, GLBVirtualLiveLocalizer
from realtime_pnp_server import LiveResult


def pattern():
    image = np.full((160, 160, 3), 180, np.uint8)
    for x in range(15, 145, 20):
        cv2.rectangle(image, (x, 25), (x+8, 130), (40, 40, 40), -1)
    return image


class FrameQualityTests(unittest.TestCase):
    def test_blur_score_and_roi_ignore_sharp_background(self):
        sharp = pattern()
        blurred = cv2.GaussianBlur(sharp, (0, 0), 8)
        self.assertGreater(image_quality(sharp)['score'], image_quality(blurred)['score']*5)
        scene = np.tile(sharp, (2, 2, 1)); scene[:160,:160] = blurred
        self.assertAlmostEqual(image_quality(scene, (0,0,.5,.5))['sharpness'],
                               image_quality(blurred, (0,0,1,1))['sharpness'])

    def test_clipped_black_white_hold_but_low_texture_is_not_blur(self):
        for value in (0, 255):
            self.assertFalse(image_quality(np.full((80,80,3),value,np.uint8))['usable'])
        self.assertTrue(image_quality(np.full((80,80,3),170,np.uint8))['usable'])

    def test_mild_correction_preserves_pixels_shape_and_does_not_mutate(self):
        image=(pattern()*.25).astype(np.uint8); before=image.copy()
        self.assertTrue(image_quality(image)['needs_correction'])
        corrected=correct_luminance(image)
        np.testing.assert_array_equal(image,before)
        self.assertEqual(corrected.shape,image.shape)
        self.assertEqual(corrected.dtype,image.dtype)
        self.assertGreater(corrected.mean(),image.mean())

    def test_bounded_age_memory_and_nonmonotonic_reset(self):
        buffer=RecentSearchFrames()
        for i in range(40):buffer.add(pattern(),np.eye(3),None,i*.02)
        self.assertLessEqual(len(buffer.frames),10)
        self.assertLessEqual(buffer.frames[-1].stamp-buffer.frames[0].stamp,.3)
        self.assertIsNone(buffer.select(2.))
        buffer.add(pattern(),np.eye(3),None,.1)
        self.assertEqual(len(buffer.frames),1)

    def test_roi_unknown_always_uses_latest_not_sharp_background(self):
        buffer=RecentSearchFrames()
        buffer.add(pattern(),np.eye(3),None,1.)
        newest=buffer.add(cv2.GaussianBlur(pattern(),(0,0),8),np.eye(3),None,1.1)
        self.assertIs(buffer.select(1.2),newest)
        self.assertIsNone(buffer.select(1.2,require_pose=True))

    def test_target_selection_favors_sharp_and_newest_near_ties(self):
        buffer=RecentSearchFrames()
        first=buffer.add(pattern(),np.eye(3),None,1.)
        latest=buffer.add(pattern(),np.eye(3),None,1.1)
        for frame in buffer.frames:frame.roi=(0,0,1,1)
        self.assertIs(buffer.select(1.15),latest)
        latest.quality['score']=first.quality['score']*.5
        self.assertIs(buffer.select(1.15),first)
        for frame in buffer.frames:frame.quality['usable']=False
        self.assertIsNone(buffer.select(1.15))

    def test_new_view_freshness_limit_does_not_reuse_sharp_old_face(self):
        buffer = RecentSearchFrames()
        old = buffer.add(pattern(), np.eye(3), None, 1.)
        new = buffer.add(pattern(), np.eye(3), None, 1.2)
        for frame in (old, new):
            frame.roi = (0,0,1,1)
            frame.pose = (np.zeros(3), np.zeros(3))
        new.quality['score'] = old.quality['score']*.5
        self.assertIs(buffer.select(1.25, require_pose=True), old)
        self.assertIs(buffer.select(1.25, require_pose=True, max_age=.1), new)
        new.quality['usable'] = False
        self.assertIsNone(buffer.select(1.25, require_pose=True, max_age=.1))
        new.roi = None
        self.assertIsNone(buffer.select(1.25, require_pose=True, max_age=.1))

    def test_new_view_submission_uses_fresh_frame_and_carries_event_to_worker(self):
        from background_refresh import BackgroundRefresh
        loc = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        loc.future = None
        loc.search_frames = deque(maxlen=48)
        loc.search_frame_buffer = RecentSearchFrames()
        old = loc.search_frame_buffer.add(pattern(),np.eye(3)*2,None,1.,11)
        new = loc.search_frame_buffer.add(pattern(),np.eye(3)*3,None,1.2,12)
        for frame in (old,new):
            frame.roi=(0,0,1,1)
            frame.pose=(np.zeros(3),np.full(3,frame.frame_id))
        new.quality['score'] = old.quality['score']*.5
        loc.replenish_mode='background'
        loc.background_refresh=BackgroundRefresh()
        loc.last_replenish_reasons=['new_regions_visible']
        loc.global_localizer=SimpleNamespace(localize_guided=Mock())
        loc.executor=SimpleNamespace(submit=Mock(return_value=Future()))
        with patch('glb_virtual_live.time.monotonic',return_value=1.25):
            loc._submit(new.image,new.matrix,None,
                pose_prior=SimpleNamespace(rvec=np.ones(3),tvec=np.ones(3)))
        args=loc.executor.submit.call_args.args
        self.assertEqual(loc.search_frame_id,12)
        np.testing.assert_array_equal(args[7],new.pose[1])
        self.assertEqual(loc.global_localizer.diagnostic_context['trigger'],['new_regions_visible'])

    def test_pose_projection_and_data_are_owned_by_source_frame(self):
        buffer=RecentSearchFrames();image=pattern();matrix=np.array([[80.,0,80],[0,80,80],[0,0,1]])
        frame=buffer.add(image,matrix,None,1.,77)
        seed=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,5.]))
        buffer.mark_pose(seed,[[-1,-1,-1],[1,1,1]])
        self.assertIsNotNone(frame.roi)
        self.assertEqual(frame.frame_id,77)
        self.assertIs(buffer.select(1.1,True),frame)
        seed.tvec[:]=90;image[:]=0;matrix[:]=0
        np.testing.assert_array_equal(frame.pose[1],[0,0,5])
        self.assertGreater(frame.image.mean(),0)
        self.assertEqual(frame.matrix[0,0],80)

    def test_old_source_submit_uses_own_pose_K_and_pre_submit_flow_history(self):
        loc=AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        loc.future=None;loc.search_frames=deque(maxlen=48);loc.replenish_mode='interval'
        loc.search_frame_buffer=RecentSearchFrames()
        first=loc.search_frame_buffer.add(pattern(),np.eye(3)*2,None,1.,11)
        latest=loc.search_frame_buffer.add(pattern(),np.eye(3)*3,None,1.1,12)
        for f in (first,latest):
            f.roi=(0,0,1,1);f.pose=(np.zeros(3),np.full(3,f.frame_id))
        latest.quality['score']=0
        loc.handoff=SimpleNamespace(prediction=(np.ones(3),np.ones(3)))
        loc.global_localizer=SimpleNamespace(localize_guided=Mock())
        loc.executor=SimpleNamespace(submit=Mock(return_value=Future()))
        with patch('glb_virtual_live.time.monotonic',return_value=1.15):
            loc._submit(latest.image,latest.matrix,None,SimpleNamespace(rvec=np.ones(3),tvec=np.ones(3)))
        args=loc.executor.submit.call_args.args
        np.testing.assert_array_equal(args[4],first.matrix)
        np.testing.assert_array_equal(args[7],first.pose[1])
        self.assertIsNone(args[5]);self.assertEqual(loc.search_source_stamp,1.)
        self.assertEqual(loc.search_frame_id,11)
        self.assertEqual(len(loc.search_frames),1)
        np.testing.assert_array_equal(loc.search_frames[-1],cv2.cvtColor(latest.image,cv2.COLOR_BGR2GRAY))
        self.assertNotIn('view_prediction',loc.executor.submit.call_args.kwargs)

    def test_no_usable_object_frame_waits_without_queueing(self):
        loc=AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        loc.future=None;loc.search_frame_buffer=RecentSearchFrames()
        frame=loc.search_frame_buffer.add(pattern(),np.eye(3),None,1.)
        frame.roi=(0,0,1,1);frame.quality['usable']=False
        loc.replenish_mode='background';loc.background_refresh=SimpleNamespace(state='ready')
        loc.executor=SimpleNamespace(submit=Mock())
        with patch('glb_virtual_live.time.monotonic',return_value=1.1):loc._submit(pattern(),np.eye(3),None)
        loc.executor.submit.assert_not_called()
        self.assertEqual(loc.background_refresh.state,'image_wait')


class PhotometricRetryTests(unittest.TestCase):
    def fixture(self, status='FAIL', diagnostic_status='FAIL', reason='3DGS_PNP_FAILED', needs=True):
        loc=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        loc.source_type='3dgs';loc.pipeline_policy='validated'
        loc._progress=Mock();loc._write_diagnostics=Mock()
        result=LiveResult(status,'test',20,20,0,0,None,None,None,None,10,'HIDDEN')
        def attempt():
            loc.last_diagnostics=dict(status=diagnostic_status,reason=reason,
                input_quality=dict(needs_correction=needs),timings_ms=dict(total=10))
            return result,np.zeros((4,4,3),np.uint8)
        return loc,Mock(side_effect=attempt)

    def test_dark_match_failure_retries_once_and_counts_both_attempts(self):
        loc,attempt=self.fixture()
        result,_=loc._quality_attempts(attempt)
        self.assertEqual(attempt.call_count,2)
        self.assertEqual(result.processing_ms,20)
        self.assertEqual(loc.last_diagnostics['timings_ms']['total'],20)
        self.assertEqual(loc.last_diagnostics['photometric_retry']['mode'],PHOTOMETRIC_MODE)
        self.assertEqual(loc._feature_variant,'raw')

    def test_success_ambiguous_QC_wrong_view_and_focus_hold_never_retry(self):
        for kwargs in (dict(status='PASS'),dict(diagnostic_status='FAIL_QC'),
                       dict(reason='AMBIGUOUS'),dict(reason='3DGS_IMAGE_QUALITY_HOLD'),
                       dict(needs=False)):
            loc,attempt=self.fixture(**kwargs);loc._quality_attempts(attempt)
            self.assertEqual(attempt.call_count,1)

    def test_variant_resets_after_exception_and_preserves_raw_cache_keys(self):
        loc,attempt=self.fixture()
        self.assertEqual(loc._variant_key(3),3)
        def failure():
            if loc._feature_variant!='raw':raise ValueError('retry')
            return attempt()
        with self.assertRaises(ValueError):loc._quality_attempts(failure)
        self.assertEqual(loc._variant_key(3),3)
        loc._feature_variant=PHOTOMETRIC_MODE
        self.assertEqual(loc._variant_key(3),(3,PHOTOMETRIC_MODE))

    def test_corrected_XYZ_cache_cannot_reuse_raw_keypoint_coordinates(self):
        loc,_=self.fixture();loc._feature_variant=PHOTOMETRIC_MODE
        loc.reference_feature_xyz={3:np.zeros((1,3)),(3,PHOTOMETRIC_MODE):np.ones((2,3))}
        np.testing.assert_array_equal(loc._feature_xyz(3,None),np.ones((2,3)))

    def test_corrected_XYZ_samples_its_own_keypoints(self):
        import torch
        loc,_=self.fixture();loc._feature_variant=PHOTOMETRIC_MODE
        loc.reference_feature_xyz={3:np.zeros((1,3))};loc.geometry_cache={}
        loc._xyz=Mock(return_value=np.ones((4,4,3)));loc.xyz_search_radius=0
        loc._source_xyz_to_runtime=lambda values:values
        features={'keypoints':torch.tensor([[[2.,2.],[1.,1.]]])}
        with patch('glb_virtual_live.sample_xyz_buffer',return_value=(np.array([[2.,2.,9.],[1.,1.,8.]]),[0,1])) as sampler:
            mapped=loc._feature_xyz(3,features)
        np.testing.assert_array_equal(sampler.call_args.args[0],[[2,2],[1,1]])
        np.testing.assert_array_equal(mapped,[[2,2,9],[1,1,8]])
        np.testing.assert_array_equal(loc.reference_feature_xyz[3],np.zeros((1,3)))

    def test_one_final_diagnostic_and_distinct_retry_variants(self):
        loc,attempt=self.fixture();variants=[]
        def run():variants.append(loc._feature_variant);return attempt()
        loc._quality_attempts(run)
        self.assertEqual(variants,['raw',PHOTOMETRIC_MODE])
        loc._write_diagnostics.assert_called_once()
        self.assertFalse(loc._suppress_diagnostics)


if __name__=='__main__':unittest.main()
