"""Small deterministic checks for the new registered-photo live route."""

import sys
import time
import tempfile
import json
import threading
from concurrent.futures import Future
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registered_aliked_live import (candidate_references, rotation_angle,
                                   make_registered_transfers, RegisteredALIKEDLiveLocalizer)
from realtime_pnp import AsyncReferenceLocalizer, build_localizer
from realtime_pnp_server import LiveResult, LocalizationDebug, TrackingSeed


from tests.runtime_dataset import RuntimeDataset


class RegisteredALIKEDLiveTests(RuntimeDataset, unittest.TestCase):
    def test_fallback_reuses_query_workspace_and_preserves_selected_diagnostics(self):
        for retry_status, retry_inliers, adopted in [('PASS',12,True),('FAIL',2,False)]:
            with self.subTest(retry_status=retry_status):
                engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
                engine.min_inliers=10
                image=np.zeros((10,10,3),np.uint8)
                polygon=np.array([[0.,0.],[1.,0.],[1.,1.]])
                full=LiveResult('FAIL','few',10,8,5,.625,1.,None,None,None,100.,'HIDDEN',10.,80.,10.)
                retry=LiveResult(retry_status,'retry',10,15,retry_inliers,.8,1.,None,None,None,50.,'HIDDEN',0.,40.,10.)
                workspaces=[]
                def once(*args,**kwargs):
                    workspaces.append(kwargs['workspace'])
                    self.assertIs(kwargs['excluded_polygon'],polygon)
                    engine.last_debug='retry' if kwargs.get('xyz_only') else 'full'
                    engine.last_reference_name=engine.last_debug
                    engine.last_tracking_seed=engine.last_debug
                    engine.last_diagnostics=dict(selected=engine.last_debug,
                        reference_matching='xyz_only_fallback' if kwargs.get('xyz_only') else 'all_object_features',
                        candidates=[dict(xyz_rejections=dict(missing_xyz=30))])
                    if kwargs.get('xyz_only'):
                        self.assertEqual(kwargs['workspace']['cached_query'],'unchanged')
                        return retry,image+1
                    kwargs['workspace']['cached_query']='unchanged'
                    return full,image
                engine._localize_once=Mock(side_effect=once)
                result,canvas=engine.localize(image,np.eye(3),np.zeros(5),excluded_polygon=polygon)
                self.assertEqual(engine._localize_once.call_count,2)
                self.assertIs(workspaces[0],workspaces[1])
                self.assertEqual(result.status,retry_status if adopted else 'FAIL')
                self.assertEqual(result.processing_ms,150.)
                self.assertEqual(result.sift_ms,10.)
                self.assertEqual(engine.last_debug,'retry' if adopted else 'full')
                self.assertEqual(engine.last_tracking_seed,'retry' if adopted else 'full')
                progress=engine.progress_snapshot()
                self.assertEqual(progress['stage'],'done')
                self.assertEqual(progress['outcome'],result.status)
                self.assertTrue(progress['retry'])
                self.assertEqual(int(canvas[0,0,0]),int(adopted))
                self.assertEqual(engine.last_diagnostics['xyz_only_fallback']['adopted'],adopted)
                # A retained first result must not create a self-referential
                # diagnostic dictionary, which would break HTTP JSON export.
                json.dumps(engine.last_diagnostics)

    def test_no_retry_for_pass_or_non_support_qc_failure(self):
        for status in ('PASS','FAIL'):
            engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
            engine.min_inliers=10
            engine.last_diagnostics=dict(quality=dict(reasons=['ambiguous_pose']),
                candidates=[dict(xyz_rejections=dict(missing_xyz=30))])
            image=np.zeros((10,10,3),np.uint8)
            result=LiveResult(status,'ambiguous',30,25,20,.8,1.,None,None,None,10.,'HIDDEN')
            engine._localize_once=Mock(return_value=(result,image))
            actual,_=engine.localize(image,np.eye(3),np.zeros(5))
            self.assertIs(actual,result)
            self.assertEqual(engine._localize_once.call_count,1)

    def test_progress_can_be_read_while_inference_lock_is_busy(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock()
        engine._progress('matching',candidate_index=2,candidate_total=6)
        with engine.lock:
            values=[]
            reader=threading.Thread(target=lambda:values.append(engine.progress_snapshot()))
            reader.start();reader.join(timeout=1.)
            self.assertFalse(reader.is_alive())
            self.assertEqual(values[0]['stage'],'matching')
            self.assertEqual(values[0]['candidate_total'],6)

    def test_guided_search_projects_landmarks_and_nearby_recovery_can_select_views(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock()
        engine.top_k=3
        engine._handoff_views=Mock(side_effect=AssertionError('No discrete view selection'))
        engine._pose_guided_views=Mock(return_value=[4])
        engine._search=Mock(return_value=('result','frame'))
        engine._localize_landmarks=Mock(return_value=('result','frame'))
        image=np.zeros((10,10,3),np.uint8);k=np.eye(3);d=np.zeros(5)
        engine.localize_guided(image,k,d,np.zeros(3),np.zeros(3))
        engine._localize_landmarks.assert_called_once()
        engine._search.assert_not_called()
        engine._pose_guided_views.assert_not_called()
        engine.localize_nearby(image,k,d,np.zeros(3),np.zeros(3))
        self.assertEqual(engine._search.call_args.kwargs['ranked_override'],[(1.,0.,4)])

    def test_new_view_uses_photo_context_without_four_second_wait(self):
        engine = RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock = threading.RLock()
        engine.last_context_started = 9.8
        engine._localize_landmarks = Mock(return_value=('fast', None))
        engine._localize_pose_references = Mock(return_value=('context', None))
        image = np.zeros((10,10,3), np.uint8)
        for reason in ('new_regions_visible', 'view_direction_emerging',
                       'view_direction_changed', 'view_direction_approaching'):
            engine.diagnostic_context = dict(search_mode='guided', trigger=[reason])
            with patch('registered_aliked_live.time.monotonic', return_value=10.):
                self.assertEqual(engine.localize_guided(image,np.eye(3),None,
                    np.zeros(3),np.zeros(3))[0], 'context')
        engine._localize_landmarks.assert_not_called()
        engine.diagnostic_context = dict(search_mode='guided', trigger=[])
        with patch('registered_aliked_live.time.monotonic', return_value=10.2):
            self.assertEqual(engine.localize_guided(image,np.eye(3),None,
                np.zeros(3),np.zeros(3))[0], 'fast')

    def test_browser_builds_same_temporal_wrapper_and_preserves_ct_gate(self):
        from browser_live import runtime_arguments
        from glb_virtual_live import AsyncTemporalGLBLocalizer
        args,_,_=runtime_arguments({'mpr':False})
        fake=SimpleNamespace(map_data=None,mpr_renderer=None,last_debug=None,
            source_type='3dgs',runtime_transform='identity',pipeline_policy='validated')
        with patch('registered_aliked_live.RegisteredALIKEDLiveLocalizer',return_value=fake) as constructor:
            wrapper=build_localizer(args,temporal_registered=True)
            try:
                self.assertIsInstance(wrapper,AsyncTemporalGLBLocalizer)
                self.assertIs(wrapper.localizer,wrapper.global_localizer)
                self.assertEqual(wrapper.replenish_mode,'background')
                self.assertEqual(wrapper.max_seed_age,10.)
                self.assertIsNotNone(wrapper.search_frame_buffer)
                self.assertEqual(constructor.call_args.kwargs['retrieval_mode'],'dino-patch')
                self.assertEqual(constructor.call_args.kwargs['search_policy'],'adaptive')
            finally:
                wrapper.close()

    def test_completed_search_over_five_seconds_is_not_erased_but_age_is_bounded(self):
        from glb_virtual_live import AsyncTemporalGLBLocalizer
        from browser_live import runtime_arguments
        from realtime_pnp import temporal_view_localizer
        args,_,_=runtime_arguments({'mpr':False})
        gray=np.zeros((32,32),np.uint8)
        seed=TrackingSeed(gray,np.ones((10,2))*16,np.ones((10,3)),np.zeros(3),np.zeros(3))
        passed=LiveResult('PASS','known source',10,10,10,1.,1.,
            (0.,0.,0.),(0.,0.,100.),(0.,0.,-100.),7000.,'HIDDEN')
        for age,accepted in ((7.,True),(11.,False)):
            with self.subTest(age=age):
                fake=SimpleNamespace(map_data=None,mpr_renderer=None,last_debug=None,
                    last_tracking_seed=seed,last_diagnostics={},last_reference_name='ref.png')
                wrapper=temporal_view_localizer(fake,args)
                try:
                    wrapper.search_started=wrapper.search_source_stamp=time.monotonic()-age
                    wrapper.future=Future();wrapper.future.set_result((passed,np.zeros((32,32,3))))
                    wrapper._consume_completed()
                    self.assertIs(wrapper.last_search_result,passed)
                    self.assertEqual(wrapper.tracker.seed is not None,accepted)
                    self.assertIsNone(wrapper.future)
                    if not accepted:
                        self.assertEqual(wrapper.search_progress()['seed_rejected'],'stale_or_failed')
                finally:
                    wrapper.close()

    def test_transfer_preserves_full_reference_indices_and_rejects_missing_xyz(self):
        # Several untriangulated features precede valid ones. Filtering the
        # archive must not shift the index of a match to a different landmark.
        pairs=np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
        scores=np.array([.9,.8,.7,.6,.5])
        uv=np.array([[10.,20.],[30.,40.],[50.,60.],[70.,80.],[90.,100.]])
        ids=np.array([-1,42,99,42,100])
        xyz=np.array([[np.nan]*3,[1.,2.,3.],[np.nan]*3,[1.,2.,3.],[4.,5.,6.]])
        transfers,rejected=make_registered_transfers(pairs,scores,uv,uv+1,ids,xyz)
        self.assertEqual(rejected,dict(missing_xyz=2,duplicate=1))
        self.assertEqual([(t.point3d_id,t.reference_index) for t in transfers],[(42,1),(100,4)])
        self.assertEqual(transfers[0].query_uv,(30.,40.))
        self.assertEqual(transfers[1].xyz,(4.,5.,6.))
        self.assertTrue(all(np.isfinite(t.xyz).all() for t in transfers))

    def test_reference_keeps_untriangulated_features_for_lightglue(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.reference_cache=OrderedDict()
        engine.torch=torch
        engine.device=torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.npz'
            np.savez(path,point_ids=np.array([-1,42,-1,100]),
                     xyz=np.array([[np.nan]*3,[1.,2.,3.],[np.nan]*3,[4.,5.,6.]]),
                     keypoints=np.arange(8,dtype=np.float32).reshape(1,4,2),
                     descriptors=np.zeros((1,4,128),np.float32),
                     keypoint_scores=np.ones((1,4),np.float32),
                     image_size=np.array([[100.,100.]],np.float32))
            engine.registered={'ref.png':{'path':path}}
            features,ids,xyz=engine._reference('ref.png')
            self.assertEqual(features['keypoints'].shape[1],4)
            np.testing.assert_array_equal(ids,[-1,42,-1,100])
            np.testing.assert_array_equal(features['keypoints'][0,3].numpy(),[6.,7.])
            self.assertTrue(np.isnan(xyz[2]).all())
            self.assertIs(engine._reference('ref.png'),engine.reference_cache['ref.png'])

    def test_select_nearby_registered_views_without_duplicate(self):
        virtual = np.eye(4)
        virtual[2, 3] = 10.  # Camera centre is (0, 0, -10).
        views = [{'world_to_camera': virtual.tolist()}]
        registered = {
            'a.png': {'rotation': np.eye(3), 'center': np.array((0., 0., -10.))},
            'b.png': {'rotation': np.eye(3), 'center': np.array((1., 0., -10.))},
            'wrong.png': {'rotation': np.diag((-1., -1., 1.)),
                          'center': np.array((0., 0., -10.))},
        }
        chosen = candidate_references(views, registered, (0., 0., 0.),
                                      [(1., 0., 0)], 1, per_view=2)
        self.assertEqual([item[0] for item in chosen], ['a.png', 'b.png'])
        self.assertAlmostEqual(rotation_angle(np.eye(3), np.eye(3)), 0.)

    def test_async_diagnostic_keeps_current_camera_frame_live(self):
        result = LiveResult('PASS', 'previous sample', 0, 11, 10, .9,
                            1., None, None, None, 1., 'HIDDEN')

        class Fake:
            map_data = None
            mpr_renderer = None
            last_debug = LocalizationDebug(np.empty((0, 3)), frozenset(), None, False)
            source_label = 'RC-ALIKED'

            def localize(self, image, matrix, distortion):
                return result, image

        wrapper = AsyncReferenceLocalizer(Fake(), show_live_during_hold=True)
        try:
            wrapper.last_result = result
            wrapper.last_display = np.zeros((240, 320, 3), np.uint8)
            wrapper.show_result_until = time.monotonic() + 10.
            current = np.full((240, 320, 3), 100, np.uint8)
            _, displayed = wrapper.localize(current, np.eye(3), np.zeros(5))
            self.assertEqual(int(displayed[120, 120, 0]), 100)
        finally:
            wrapper.close()


if __name__ == '__main__':
    unittest.main()
