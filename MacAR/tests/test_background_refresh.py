import sys
import unittest
from pathlib import Path
from concurrent.futures import Future
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from background_refresh import BackgroundRefresh
from glb_virtual_live import AsyncTemporalGLBLocalizer
from realtime_pnp_server import TrackingSeed, LiveResult
from browser_live import runtime_arguments


class RefreshPolicyTests(unittest.TestCase):
    def test_healthy_unchanged_view_refreshes_after_bounded_idle_without_queue(self):
        policy = BackgroundRefresh()
        policy.completed(0.,400.)
        self.assertIsNone(policy.decide(.9, busy=False, weak_reasons=[]))
        self.assertEqual(policy.decide(1., busy=False, weak_reasons=[]),'guided')
        self.assertEqual(policy.reason, 'continuous_refresh')
        self.assertIsNone(policy.decide(1., busy=True, weak_reasons=[]))

    def test_busy_worker_never_queues_another_job(self):
        policy = BackgroundRefresh(failures=10)
        self.assertIsNone(policy.decide(100., busy=True, weak_reasons=['few_points'], safety_due=True))
        self.assertEqual(policy.state, 'running')

    def test_occupancy_cooldown_scales_with_measured_search_duration(self):
        policy = BackgroundRefresh()
        policy.completed(10., 2000.)
        self.assertIsNone(policy.decide(13., busy=False, weak_reasons=[]))
        self.assertEqual(policy.decide(14., busy=False, weak_reasons=[]), 'guided')
        # ~2 s running / (2 + 3.71 s idle) is the 35% occupancy target.
        self.assertEqual(policy.state, 'ready')

    def test_slow_foreground_suspends_healthy_refresh_but_allows_recovery(self):
        policy = BackgroundRefresh()
        policy.track_sample(300.)
        self.assertIsNone(policy.decide(10., busy=False, weak_reasons=[]))
        self.assertEqual(policy.state, 'foreground_busy')
        self.assertEqual(policy.decide(10., busy=False, weak_reasons=['few_points']), 'guided')

    def test_failed_near_search_escalates_only_if_tracking_is_weak(self):
        policy = BackgroundRefresh(failures=3)
        self.assertEqual(policy.decide(20., busy=False, weak_reasons=[]),'guided')
        self.assertEqual(policy.decide(20., busy=False, weak_reasons=['few_points']), 'global')
        self.assertEqual(policy.reason, 'near_search_failed')
        policy.submitted('global', 20.)
        self.assertEqual(policy.decide(25., busy=False, weak_reasons=['few_points']), 'guided')

    def test_safety_review_and_failure_backoff(self):
        policy = BackgroundRefresh()
        self.assertEqual(policy.decide(0., busy=False, weak_reasons=[], safety_due=True), 'global')
        policy.completed(1., 100.)
        policy.rejected('qc')
        self.assertIsNone(policy.decide(2., busy=False, weak_reasons=[]))
        self.assertEqual(policy.decide(3., busy=False, weak_reasons=[]), 'guided')
        policy.accepted(8)
        self.assertEqual((policy.failures, policy.added_total, policy.last_added), (0, 8, 8))

    def test_lost_tracking_recovery_does_not_wait_long_healthy_budget(self):
        policy = BackgroundRefresh(failures=4, fast_recovery=True)
        policy.completed(0., 4000.)
        self.assertIsNone(policy.decide(.2, busy=False, weak_reasons=['tracking_lost']))
        self.assertIsNotNone(policy.decide(.3, busy=False, weak_reasons=['tracking_lost']))
        policy.fast_recovery=False
        self.assertIsNone(policy.decide(1., busy=False, weak_reasons=['tracking_lost']))

    def test_predicted_support_expiry_starts_early_but_never_queues(self):
        policy=BackgroundRefresh(failures=2)
        policy.completed(0.,2000.)
        for reason in ('support_critical',):
            self.assertIsNone(policy.decide(.1,busy=False,weak_reasons=[reason]))
            self.assertEqual(policy.decide(.2,busy=False,weak_reasons=[reason]),'guided')
            self.assertIsNone(policy.decide(.3,busy=True,weak_reasons=[reason]))

    def test_pre_loss_handoff_bypasses_healthy_duty_wait_and_failure_backoff(self):
        policy=BackgroundRefresh(failures=2)
        policy.completed(0., 2000.)
        self.assertIsNone(policy.decide(.9, busy=False, weak_reasons=['points_declining']))
        self.assertEqual(policy.decide(1., busy=False, weak_reasons=['points_declining']), 'guided')
        self.assertEqual(policy.reason, 'points_declining')
        self.assertIsNone(policy.decide(1., busy=False, weak_reasons=[]))

    def test_new_region_and_direction_events_do_not_claim_weak_track(self):
        for reason in ('view_direction_changed', 'new_regions_visible', 'view_direction_approaching'):
            policy=BackgroundRefresh(failures=3)
            self.assertEqual(policy.decide(20.,busy=False,weak_reasons=[reason]),'guided')
            self.assertEqual(policy.decide(20.,busy=False,weak_reasons=[],safety_due=True),'global')

    def test_new_view_replenishes_without_waiting_for_healthy_duty_budget(self):
        for reason in ('new_regions_visible', 'view_direction_emerging',
                       'view_direction_changed', 'view_direction_approaching'):
            policy = BackgroundRefresh()
            policy.completed(10., 2000.)
            self.assertIsNone(policy.decide(10.19, busy=False, weak_reasons=[reason]))
            self.assertEqual(policy.decide(10.21, busy=False, weak_reasons=[reason]), 'guided')
            self.assertEqual(policy.reason, reason)
            self.assertIsNone(policy.decide(10.21, busy=True, weak_reasons=[reason]))

    def test_new_view_failure_backoff_and_foreground_priority_still_apply(self):
        policy = BackgroundRefresh(failures=4)
        policy.completed(0., 2000.)
        self.assertIsNone(policy.decide(.9, busy=False, weak_reasons=['view_direction_emerging']))
        self.assertEqual(policy.decide(1., busy=False, weak_reasons=['view_direction_emerging']), 'guided')
        policy.tracking_ms = 250.
        self.assertIsNone(policy.decide(10., busy=False, weak_reasons=['new_regions_visible']))
        self.assertEqual(policy.state, 'foreground_busy')

    def test_new_view_with_coverage_warning_keeps_short_start_delay(self):
        policy = BackgroundRefresh()
        policy.completed(0., 2000.)
        self.assertEqual(policy.decide(.21,busy=False,
            weak_reasons=['view_direction_emerging','image_coverage']), 'guided')


from tests.runtime_dataset import RuntimeDataset


class BackgroundPipelineTests(RuntimeDataset, unittest.TestCase):
    def fixture(self):
        loc = AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        xyz = np.array([[x*50., y*50., 500.] for x in range(10) for y in range(10)])
        image = np.zeros((64,64,3), np.uint8)
        seed = TrackingSeed(image[:,:,0], np.zeros((100,2)), xyz, np.zeros(3), np.zeros(3))
        result = LiveResult('PASS','TRACKING',100,100,100,1.,1.,(0.,0.,0.),(0.,0.,0.),None,1.,'HIDDEN')
        loc.global_localizer = SimpleNamespace(source_label='3DGS', min_inliers=10,
            max_reprojection_error=6., last_tracking_quality={'coverage':.1},
            last_debug=None, last_diagnostics={}, last_tracking_seed=seed,
            last_reference_name='near.png')
        debug = SimpleNamespace(matched_xyz=xyz, inlier_indices=frozenset(range(100)))
        loc.tracker = SimpleNamespace(seed=seed, _copy_seed=lambda value:value,
            _track=Mock(return_value=(result,image)), frames_since_global=0, last_debug=debug)
        loc.replenish_mode='background'
        loc.background_refresh=BackgroundRefresh()
        loc.replenish_anchor=SimpleNamespace(rvec=np.zeros(3), tvec=np.zeros(3))
        loc.replenish_distance_mm=30.
        loc.replenish_height_mm=15.
        loc.replenish_cooldown=1.
        loc.replenish_frames=30
        loc.frames_since_replenish=0
        loc.last_replenish_attempt=0.
        loc.last_refresh_verified=False
        loc.last_search_age_ms=1000.
        loc.max_seed_age=5.
        loc.reseed_frames=900
        loc.future=loc.pending_seed=loc.pending_mode=None
        loc.active_reference_name='old.png'
        loc.search_started=1.
        loc.search_mode='global'
        loc.search_frames=deque(maxlen=48)
        loc._update_coarse=Mock()
        loc._submit=Mock()
        return loc, image

    def test_healthy_pipeline_waits_during_bounded_cooldown(self):
        loc,image=self.fixture()
        loc.background_refresh.completed(1.5,400.)
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            loc.localize(image,np.eye(3),None)
        loc._submit.assert_not_called()
        self.assertEqual(loc.last_replenish_reasons,[])

    def test_low_support_starts_refresh_before_waiting_for_point_loss(self):
        loc,image=self.fixture()
        loc.background_refresh.completed(1.5,400.)
        loc.tracker.seed=replace(loc.tracker.seed,image_points=np.zeros((30,2)))
        with patch('glb_virtual_live.time.monotonic',return_value=1.71):
            loc.localize(image,np.eye(3),None)
        self.assertIn('support_critical',loc.last_replenish_reasons)
        loc._submit.assert_called_once()

    def test_added_points_are_published_in_the_same_current_frame(self):
        loc,image=self.fixture()
        loc.pending_seed=loc.tracker.seed
        loc.pending_mode='guided'
        first=loc.tracker._track.return_value
        final=replace(first[0],inliers=104,matches=104,keypoints=104)
        loc.tracker._track.side_effect=[first,(final,image)]
        def add(*args,**kwargs):
            loc.last_refresh_verified=True
            loc.last_refresh_refinement={}
            loc.last_replenish_diagnostics={'added':4,'reason':'added'}
            return 4
        loc._replenish_current_seed=Mock(side_effect=add)
        answer,_=loc.localize(image,np.eye(3),None)
        self.assertEqual(answer.inliers,104)
        self.assertEqual(answer.detail,'TRACKING_REPLENISHED')

    def test_failed_final_check_rolls_back_points_and_diagnostic_additions(self):
        loc,image=self.fixture()
        seed=loc.tracker.seed
        loc.pending_seed=seed
        loc.pending_mode='guided'
        first=loc.tracker._track.return_value
        loc.tracker._track.side_effect=[first,None]
        def add(*args,**kwargs):
            loc.tracker.seed=replace(seed,image_points=seed.image_points+1.)
            loc.last_refresh_verified=True
            loc.last_refresh_refinement={}
            loc.last_replenish_diagnostics={'added':4,'reason':'added'}
            return 4
        loc._replenish_current_seed=Mock(side_effect=add)
        loc.localize(image,np.eye(3),None)
        self.assertIs(loc.tracker.seed,seed)
        self.assertEqual(loc.last_replenished,0)
        self.assertEqual(loc.last_replenish_diagnostics['added'],0)
        self.assertEqual(loc.last_replenish_diagnostics['reason'],'final_current_frame_qc')

    def test_strict_refresh_accepts_three_pixel_support_only_after_combined_qc(self):
        loc,_=self.fixture()
        uv=np.array([[x,y] for x in (10.,30.,50.,70.) for y in (10.,30.,50.)])
        xyz=np.column_stack((uv,np.full(len(uv),100.)))
        gray=np.zeros((100,100),np.uint8)
        loc.tracker.seed=TrackingSeed(gray,uv,xyz,np.zeros(3),np.zeros(3))
        loc.tracker.flow_window=21;loc.tracker.flow_levels=3;loc.tracker.forward_backward_error=1.5
        loc.global_localizer.ransac_threshold=4.
        additions=np.array([[20.,20.],[40.,20.],[60.,20.],[80.,20.]])
        candidate=TrackingSeed(gray,additions+[3.,0.],
            np.column_stack((additions,np.full(4,100.))),np.zeros(3),np.zeros(3))
        loc.global_localizer.validate_pose=Mock(return_value={'passed':True})
        def flow(a,b,p,_,**kwargs):return p.copy(),np.ones((len(p),1),np.uint8),None
        with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK',side_effect=flow):
            added=loc._replenish_current_seed(candidate,np.zeros((100,100,3),np.uint8),np.diag([100.,100.,1.]),None)
        self.assertEqual(added,4)
        self.assertEqual(loc.last_replenish_diagnostics['residual_limit_px'],4.)
        self.assertEqual(loc.last_replenish_diagnostics['residual_rejected'],0)
        loc.global_localizer.validate_pose.assert_called_once()
        before=loc.tracker.seed
        loc.global_localizer.validate_pose.return_value={'passed':False}
        candidate=replace(candidate,image_points=candidate.image_points+[0.,10.],
            object_points=candidate.object_points+[0.,10.,0.])
        with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK',side_effect=flow):
            self.assertEqual(loc._replenish_current_seed(candidate,np.zeros((100,100,3),np.uint8),np.diag([100.,100.,1.]),None),0)
        self.assertIs(loc.tracker.seed,before)

    def test_geometry_sensitivity_is_diagnostic_not_a_pose_acceptance_or_scheduling_threshold(self):
        loc,image=self.fixture()
        loc.global_localizer.last_tracking_quality['rotation_sensitivity_deg_per_px']=.3
        result,_=loc.localize(image,np.eye(3),None)
        self.assertEqual(result.status,'PASS')
        self.assertNotIn('pose_support_weak',loc.last_replenish_reasons)

    def test_pure_roll_does_not_schedule_refresh(self):
        loc,image=self.fixture()
        loc.background_refresh.completed(1.5,400.)
        loc.global_localizer.manifest=dict(center=(0.,0.,500.))
        loc.replenish_anchor=SimpleNamespace(rvec=np.zeros(3),tvec=np.zeros(3))
        loc.tracker.seed=replace(loc.tracker.seed,rvec=np.array([0.,0.,.5]))
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            loc.localize(image,np.eye(3),None)
        loc._submit.assert_not_called()

    def test_camera_translation_with_same_rotation_schedules_viewpoint_refresh(self):
        loc,image=self.fixture()
        loc.global_localizer.manifest=dict(center=(0.,0.,500.))
        loc.tracker.seed=replace(loc.tracker.seed,tvec=np.array([-200.,0.,0.]))
        loc.localize(image,np.eye(3),None)
        self.assertIn('view_direction_changed',loc.last_replenish_reasons)
        self.assertIs(loc._submit.call_args.kwargs['pose_prior'],loc.tracker.seed)

    def test_background_pipeline_starts_replenishing_at_five_degrees(self):
        loc, image = self.fixture()
        loc.global_localizer.manifest = dict(center=(0.,0.,500.))
        loc.tracker.seed = replace(loc.tracker.seed, tvec=np.array([-44.,0.,0.]))
        loc.background_refresh.completed(0.,2000.)
        with patch('glb_virtual_live.time.monotonic', return_value=.21):
            loc.localize(image,np.eye(3),None)
        self.assertIn('view_direction_emerging', loc.last_replenish_reasons)
        loc._submit.assert_called_once()
        self.assertIs(loc._submit.call_args.kwargs['pose_prior'],loc.tracker.seed)

    def test_unchanged_view_still_allows_periodic_global_review(self):
        loc,image=self.fixture()
        loc.tracker.frames_since_global=loc.reseed_frames
        loc.localize(image,np.eye(3),None)
        self.assertIsNone(loc._submit.call_args.kwargs['pose_prior'])

    def test_new_projected_regions_schedule_guided_refresh_without_orbit(self):
        loc,image=self.fixture()
        xyz=np.array([(x,y,0.) for x in np.linspace(-300,300,33)
                      for y in np.linspace(-80,80,17)])
        loc.global_localizer.map_data=SimpleNamespace(xyz=xyz)
        loc.global_localizer.manifest=dict(center=(0.,0.,0.))
        loc.tracker.seed=replace(loc.tracker.seed,tvec=np.array([0.,0.,300.]))
        rvec=np.array([0.,-.4,0.])
        loc.replenish_anchor=SimpleNamespace(rvec=rvec,tvec=-cv2.Rodrigues(rvec)[0]@np.array([0.,0.,-300.]))
        K=np.array([[150.,0.,100.],[0.,150.,100.],[0.,0.,1.]])
        loc.localize(np.zeros((200,200,3),np.uint8),K,None)
        self.assertIn('new_regions_visible',loc.last_replenish_reasons)
        self.assertNotIn('view_direction_changed',loc.last_replenish_reasons)
        self.assertIs(loc._submit.call_args.kwargs['pose_prior'],loc.tracker.seed)

    def test_pass_pose_is_attached_to_current_search_buffer_frame(self):
        from search_frame_quality import RecentSearchFrames
        loc,image=self.fixture();loc.search_frame_buffer=RecentSearchFrames()
        loc.global_localizer.manifest={}
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            loc.localize(image,np.eye(3),None)
        frame=loc.search_frame_buffer.frames[-1]
        np.testing.assert_array_equal(frame.pose[0],loc.tracker.seed.rvec)
        np.testing.assert_array_equal(frame.pose[1],loc.tracker.seed.tvec)

    def test_seed_expiry_counts_age_before_job_submission(self):
        loc,_=self.fixture()
        loc.search_started=1.;loc.search_source_stamp=.75
        loc.future=Future();loc.future.set_result((SimpleNamespace(status='PASS'),None))
        loc._advance_search_seed=Mock()
        with patch('glb_virtual_live.time.monotonic',return_value=5.9):loc._consume_completed()
        loc._advance_search_seed.assert_not_called()
        self.assertAlmostEqual(loc.last_search_age_ms,5150.)

    def test_single_future_and_foreground_priority(self):
        loc,image=self.fixture()
        loc.future=Future()
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            loc.localize(image,np.eye(3),None)
        loc._submit.assert_not_called()
        loc.future=None
        loc.background_refresh.tracking_ms=250.
        loc.localize(image,np.eye(3),None)
        loc._submit.assert_not_called()
        self.assertEqual(loc.background_refresh.state,'foreground_busy')

    def test_weak_track_and_repeated_rejections_trigger_global(self):
        loc,image=self.fixture()
        loc.background_refresh.failures=3
        loc.global_localizer.last_tracking_quality={'coverage':.001}
        loc.localize(image,np.eye(3),None)
        self.assertIsNone(loc._submit.call_args.kwargs['pose_prior'])

    def test_global_candidate_never_blindly_replaces_working_pose(self):
        loc,image=self.fixture()
        original=loc.tracker.seed
        loc.pending_seed=TrackingSeed(original.gray,original.image_points,original.object_points,
                                      np.ones(3),np.ones(3)*1000)
        loc.pending_mode='global'
        loc._replenish_current_seed=Mock(return_value=0)
        loc.localize(image,np.eye(3),None)
        self.assertIs(loc.tracker.seed,original)
        loc.tracker._track.assert_called_once()
        loc._replenish_current_seed.assert_called_once()
        self.assertEqual(loc.background_refresh.outcome,'current_frame_qc_rejected')

    def test_strong_pose_conflict_hides_overlay_clears_anchor_and_starts_global_search(self):
        from tracking_recovery import ShortTrackRecovery
        loc,image=self.fixture()
        loc.short_recovery=ShortTrackRecovery()
        loc.short_recovery.remember(loc.tracker.seed,'old.png',np.eye(3),None,1.)
        loc.pending_seed=loc.tracker.seed
        loc.pending_mode='guided'
        def conflict(*_):
            loc.last_refresh_refinement=dict(accepted=False,conflict=True,reason='large_pose_change')
            return 0
        loc._replenish_current_seed=Mock(side_effect=conflict)
        result,canvas=loc.localize(image,np.eye(3),None)
        self.assertEqual(result.status,'FAIL')
        self.assertEqual(result.detail,'FRESH_POSE_CONFLICT')
        self.assertEqual(result.mpr_status,'HIDDEN')
        self.assertIsNone(result.rvec)
        self.assertIsNone(loc.tracker.seed)
        self.assertIsNone(loc.short_recovery.seed)
        self.assertIsNone(loc.pending_seed)
        self.assertTrue(loc.last_refresh_refinement['conflict'])
        self.assertEqual(loc.last_track_failure,'fresh_pose_conflict')
        loc._submit.assert_called_once()
        self.assertEqual(loc._submit.call_args.kwargs,{})
        np.testing.assert_array_equal(canvas,image)

    def test_verified_duplicate_refresh_counts_as_verified_not_added(self):
        loc,image=self.fixture()
        loc.pending_seed=loc.tracker.seed
        loc.pending_mode='guided'
        loc.background_refresh.failures=2
        def verified(*_):
            loc.last_refresh_verified=True
            return 0
        loc._replenish_current_seed=Mock(side_effect=verified)
        loc.localize(image,np.eye(3),None)
        self.assertEqual(loc.background_refresh.outcome,'verified')
        self.assertEqual(loc.background_refresh.failures,0)

    def test_stale_search_is_rejected_before_expensive_seed_transport(self):
        loc,_=self.fixture()
        loc.future=Future();loc.future.set_result((SimpleNamespace(status='PASS'),None))
        loc.search_frames.append(np.zeros((64,64),np.uint8))
        loc._advance_search_seed=Mock()
        original=loc.tracker.seed
        with patch('glb_virtual_live.time.monotonic',return_value=10.):
            loc._consume_completed()
        self.assertIs(loc.tracker.seed,original)
        loc._advance_search_seed.assert_not_called()
        self.assertIsNone(loc.pending_seed)
        self.assertEqual(loc.background_refresh.failures,1)

    def test_guided_result_cannot_initialize_after_prior_track_lost(self):
        loc,_=self.fixture();loc.tracker.seed=None
        loc.search_mode='guided'
        loc.future=Future();loc.future.set_result((SimpleNamespace(status='PASS'),None))
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            loc._consume_completed()
        self.assertIsNone(loc.tracker.seed)
        self.assertEqual(loc.last_search_diagnostics['seed_rejected'],'guided_prior_lost')

    def test_worker_exception_does_not_kill_working_track(self):
        loc,image=self.fixture()
        loc.future=Future();loc.future.set_exception(ValueError('test worker failure'))
        original=loc.tracker.seed
        with patch('glb_virtual_live.time.monotonic',return_value=2.):
            result,_=loc.localize(image,np.eye(3),None)
        self.assertEqual(result.status,'PASS')
        self.assertIs(loc.tracker.seed,original)
        self.assertEqual(loc.last_search_result.status,'FAIL')
        self.assertEqual(loc.background_refresh.failures,1)

    def test_submit_guard_does_not_copy_or_queue_when_busy(self):
        loc,image=self.fixture();loc.future=Future()
        loc.executor=SimpleNamespace(submit=Mock())
        AsyncTemporalGLBLocalizer._submit(loc,image,np.eye(3),np.zeros(5))
        loc.executor.submit.assert_not_called()

    def test_motion_hint_is_only_passed_for_view_selection_not_as_pnp_prior(self):
        loc,image=self.fixture()
        from tracking_handoff import TrackingHandoff
        loc.handoff=TrackingHandoff()
        loc.handoff.prediction=(np.array([0.,.2,0.]),np.array([20.,0.,0.]))
        loc.global_localizer.localize_guided=Mock()
        loc.last_replenish_reasons=['new_regions_visible']
        loc.executor=SimpleNamespace(submit=Mock(return_value=Future()))
        seed=loc.tracker.seed
        loc._submit=AsyncTemporalGLBLocalizer._submit.__get__(loc)
        loc._submit(image,np.eye(3),np.zeros(5),pose_prior=seed)
        args=loc.executor.submit.call_args.args
        np.testing.assert_array_equal(args[6],seed.rvec)
        np.testing.assert_array_equal(args[7],seed.tvec)
        hint=loc.executor.submit.call_args.kwargs['view_prediction']
        np.testing.assert_array_equal(hint[0],loc.handoff.prediction[0])
        self.assertIsNot(hint[0],loc.handoff.prediction[0])
        self.assertEqual(loc.global_localizer.diagnostic_context['trigger'], ['new_regions_visible'])
        self.assertIsNot(loc.global_localizer.diagnostic_context['trigger'],loc.last_replenish_reasons)

    def test_pipeline_detects_declining_points_and_schedules_before_loss(self):
        loc,image=self.fixture()
        loc.background_refresh.completed(0.,2000.)
        for now,count in ((.8,60),(.9,52),(1.,42)):
            loc.tracker.seed=replace(loc.tracker.seed,image_points=np.zeros((count,2)))
            with patch('glb_virtual_live.time.monotonic',return_value=now):
                loc.localize(image,np.eye(3),None)
        self.assertIn('points_declining',loc.last_replenish_reasons)
        self.assertEqual(loc._submit.call_count,1)
        self.assertIs(loc._submit.call_args.kwargs['pose_prior'],loc.tracker.seed)
        self.assertGreater(len(loc.tracker.seed.image_points),20)

    def test_continuous_merge_bounds_tracking_points_and_preserves_uv_xyz_pairs(self):
        loc,_=self.fixture()
        uv=np.array([[x,y] for x in np.linspace(5,615,45)
                              for y in np.linspace(5,615,45)],np.float64)
        matrix=np.array([[500.,0,320.],[0,500.,320.],[0,0,1.]])
        def xyz(points):
            return np.column_stack(((points[:,0]-320.)*2.,(points[:,1]-320.)*2.,np.full(len(points),1000.)))
        gray=np.zeros((640,640),np.uint8)
        loc.tracker.seed=TrackingSeed(gray,uv,xyz(uv),np.zeros(3),np.zeros(3))
        loc.tracker.flow_window=21;loc.tracker.flow_levels=3;loc.tracker.forward_backward_error=1.5
        loc.global_localizer.ransac_threshold=8.
        loc.global_localizer.validate_pose=Mock(return_value={'passed':True})
        additions=uv[:80]+7.
        candidate=TrackingSeed(gray,additions,xyz(additions),np.zeros(3),np.zeros(3))
        def flow(before,after,p,_unused,**kwargs):
            return p.copy(),np.ones((len(p),1),np.uint8),None
        for mode in ('background','coverage','interval'):
            with self.subTest(mode=mode):
                loc.replenish_mode=mode
                loc.tracker.seed=TrackingSeed(gray,uv,xyz(uv),np.zeros(3),np.zeros(3))
                with patch('glb_virtual_live.cv2.calcOpticalFlowPyrLK',side_effect=flow):
                    loc._replenish_current_seed(candidate,np.zeros((640,640,3),np.uint8),matrix,None)
                self.assertEqual(len(loc.tracker.seed.image_points),2000)
                projected=cv2.projectPoints(loc.tracker.seed.object_points,np.zeros(3),np.zeros(3),matrix,None)[0].reshape(-1,2)
                np.testing.assert_allclose(projected,loc.tracker.seed.image_points,atol=1e-4)
                self.assertTrue(loc.last_refresh_verified)

    def test_browser_baseline_switch_preserves_camera_and_qc(self):
        default,_,_=runtime_arguments({'mode':'gs'})
        old,_,_=runtime_arguments({'mode':'gs','refresh_policy':'coverage'})
        trial,_,_=runtime_arguments({'mode':'gs','refresh_policy':'orbit-experimental'})
        self.assertFalse(default.tracking_orbit_recovery)
        self.assertTrue(trial.tracking_orbit_recovery)
        self.assertEqual(trial.tracking_replenish_mode,'background')
        self.assertEqual((trial.glb_max_keypoints,trial.virtual_guided_features),(2000,2000))
        self.assertEqual((trial.capture_backend,trial.min_inliers,trial.virtual_policy),
                         (default.capture_backend,default.min_inliers,default.virtual_policy))
        self.assertEqual((default.tracking_replenish_mode,default.tracking_reseed_frames),('background',90))
        self.assertEqual((old.tracking_replenish_mode,old.tracking_reseed_frames),('coverage',90))
        self.assertEqual((default.capture_backend,default.min_inliers,default.virtual_policy),
                         (old.capture_backend,old.min_inliers,old.virtual_policy))
        with self.assertRaises(ValueError):runtime_arguments({'refresh_policy':'unknown'})


if __name__=='__main__':unittest.main()
