import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

import cv2
import numpy as np

from pose_landmark_matching import (PoseLandmarkBank, LandmarkCandidates,
                                    match_projected_landmarks, normalized_descriptors)
from registered_aliked_live import RegisteredALIKEDLiveLocalizer
from realtime_pnp_server import LiveResult


class LandmarkMatchingTests(unittest.TestCase):
    def candidates(self, ids=(10, 10, 11), projected=None, descriptors=None):
        projected = [[50., 50.]]*len(ids) if projected is None else projected
        descriptors = [[1., 0., 0.], [.99, .01, 0.], [0., 1., 0.]] if descriptors is None else descriptors
        return LandmarkCandidates(np.asarray(ids), np.zeros((len(ids), 3)), np.asarray(projected),
                                  normalized_descriptors(descriptors), np.ones(len(ids)))

    def test_ratio_runner_up_is_a_different_landmark_not_a_descriptor_variant(self):
        pairs, scores, stats = match_projected_landmarks(
            [[51., 50.]], [[1., 0., 0.]], self.candidates(), radius=5.)
        np.testing.assert_array_equal(pairs, [[0, 0]])
        self.assertAlmostEqual(scores[0], 1.)
        self.assertEqual(stats['mutual_pass'], 1)

    def test_same_descriptor_on_distinct_ids_is_ambiguous(self):
        candidates = self.candidates(ids=(10, 11), descriptors=[[1., 0.], [1., 0.]])
        pairs, _, _ = match_projected_landmarks([[50., 50.]], [[1., 0.]], candidates, radius=5.)
        self.assertEqual(len(pairs), 0)

    def test_mutual_best_prevents_multiple_queries_using_one_landmark(self):
        pairs, _, _ = match_projected_landmarks([[50., 50.], [50., 51.]],
            [[.9, .1, 0.], [1., 0., 0.]], self.candidates(), radius=5.)
        np.testing.assert_array_equal(pairs, [[1, 0]])

    def test_spatial_gate_rejects_similar_far_features(self):
        pairs, _, stats = match_projected_landmarks([[100., 100.]], [[1., 0., 0.]],
                                                   self.candidates(), radius=5.)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(stats['neighbourhood_pairs'], 0)

    def test_empty_nonfinite_and_single_candidate_do_not_invent_correspondences(self):
        candidates = self.candidates(ids=(10,), descriptors=[[1., 0., 0.]])
        for pixels, descriptors in (([], np.empty((0, 3))), ([[50., 50.]], [[1., 0., 0.]]),
                                     ([[np.nan, 50.]], [[1., 0., 0.]])):
            pairs, _, _ = match_projected_landmarks(pixels, descriptors, candidates, radius=5.)
            self.assertEqual(pairs.shape, (0, 2))

    def test_neighbourhood_memory_is_bounded(self):
        count=200
        candidates=self.candidates(ids=np.arange(count), descriptors=np.eye(count),
                                   projected=np.ones((count,2))*50.)
        _, _, stats=match_projected_landmarks([[50.,50.]], np.eye(count)[:1], candidates,
                                             radius=5., neighbours=8)
        self.assertLessEqual(stats['neighbourhood_pairs'],16)

    def test_bank_filters_xyz_bounds_and_observed_view_direction(self):
        xyz=np.array([[0.,0.,10.],[1.,0.,10.],[0.,0.,-10.],[20.,0.,10.]])
        bank=PoseLandmarkBank(xyz,[0,0,1,2,3,-1],np.eye(6),
            [[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,-1.],[0.,0.,-1.],[0.,0.,-1.]],np.ones(6))
        k=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        candidates=bank.candidates(np.zeros(3),np.zeros(3),k,np.zeros(5),(100,100,3))
        np.testing.assert_array_equal(candidates.point_ids,[0,1])
        np.testing.assert_allclose(candidates.projected,[[50.,50.],[60.,50.]])

    def test_per_landmark_variants_and_spatial_budget_are_bounded(self):
        xyz=np.array([[0.,0.,10.],[1.,0.,10.]])
        bank=PoseLandmarkBank(xyz,[0,0,0,1,1,1],np.eye(6),
                              np.tile([0.,0.,-1.],(6,1)),np.ones(6))
        k=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        chosen=bank.candidates(np.zeros(3),np.zeros(3),k,np.zeros(5),(100,100,3),budget=1)
        self.assertEqual(len(set(chosen.point_ids)),1)
        self.assertEqual(len(chosen.point_ids),2)

    def test_archive_id_xyz_inconsistency_is_an_error_not_approximate_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ref.npz'
            np.savez(path,point_ids=np.array([0]),xyz=np.array([[100.,0.,0.]]),
                descriptors=np.ones((1,1,128)),keypoint_scores=np.ones((1,1)))
            registered={'ref.png':dict(path=path,center=np.array([0.,0.,10.]),rotation=np.eye(3))}
            with self.assertRaisesRegex(ValueError,'inconsistent'):
                PoseLandmarkBank.from_registered(np.zeros((1,3)),registered)


class RegisteredValidationTests(unittest.TestCase):
    def test_live_guided_interleaves_context_without_repeating_global_retrieval(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock()
        engine.diagnostic_context={'search_mode':'guided'}
        engine._localize_pose_references=Mock(return_value=('context','canvas'))
        engine._localize_landmarks=Mock(return_value=('fast','canvas'))
        with patch('registered_aliked_live.time.monotonic',return_value=10.):
            self.assertEqual(engine.localize_guided(None,None,None,np.zeros(3),np.zeros(3))[0],'context')
        with patch('registered_aliked_live.time.monotonic',return_value=11.):
            self.assertEqual(engine.localize_guided(None,None,None,np.zeros(3),np.zeros(3))[0],'fast')
        with patch('registered_aliked_live.time.monotonic',return_value=14.):
            self.assertEqual(engine.localize_guided(None,None,None,np.zeros(3),np.zeros(3))[0],'context')
        self.assertEqual(engine._localize_pose_references.call_count,2)

    def test_pose_context_crops_projected_map_and_selects_at_most_two_photos(self):
        engine,frame,k,d,r,t,_,_=self.point_engine()
        engine.views=[dict(image_name='near.png',world_to_camera=np.eye(4))]
        engine.manifest={'center':engine.map_data.xyz.mean(axis=0).tolist()}
        engine.max_coarse_rotation_deg=55.
        engine.registered={str(i):dict(rotation=np.eye(3),center=np.array([i,0.,-300.])) for i in range(4)}
        engine._pose_guided_views=Mock(return_value=[0])
        engine._search_attempts=Mock(return_value=(SimpleNamespace(status='PASS',detail='ok',inliers=20,processing_ms=1.),frame))
        engine.last_diagnostics={}
        engine._localize_pose_references(frame,k,d,r,t)
        workspace=engine._search_attempts.call_args.args[4]
        self.assertEqual(len(workspace['references']),2)
        self.assertEqual(workspace['reference_selection']['mode'],'pose_geometry')
        self.assertIs(workspace['pose_prior'][0],r)
        engine.retriever.assert_not_called()

    def test_context_prior_check_does_not_bypass_qc_or_accept_pose_jump(self):
        engine,_,_,_,r,t,_,_=self.point_engine()
        quality=dict(passed=False,reasons=['clustered_image_support'])
        engine._check_search_pose(quality,r,t,(r,t))
        self.assertFalse(quality['passed'])
        self.assertEqual(quality['reasons'],['clustered_image_support'])
        quality=dict(passed=True,reasons=[])
        engine._check_search_pose(quality,r,t+1000.,(r,t))
        self.assertFalse(quality['passed'])
        self.assertIn('pose_prior_mismatch',quality['reasons'])

    def point_engine(self):
        import torch
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock()
        engine.torch=torch;engine.device='cpu';engine.extractor=None;engine.matcher=None
        engine.resize=1024;engine.ransac_threshold=4.;engine.iterations=1000
        engine.min_inliers=10;engine.min_inlier_ratio=.2;engine.max_reprojection_error=3.
        xyz=np.array([[x,y,10.*(i%2)] for i,(x,y) in enumerate(
            ( (x,y) for x in (-60.,-20.,20.,60.) for y in (-45.,-22.5,0.,22.5,45.) ))])
        k=np.array([[300.,0.,320.],[0.,300.,240.],[0.,0.,1.]])
        r=np.zeros(3);t=np.array([0.,0.,300.]);d=np.zeros(5)
        uv=cv2.projectPoints(xyz,r,t,k,d)[0].reshape(-1,2)
        candidates=LandmarkCandidates(np.arange(20),xyz,uv+4.,np.eye(20,dtype=np.float32),np.ones(20))
        engine.landmark_bank=SimpleNamespace(ids=np.arange(20),candidates=Mock(return_value=candidates))
        engine.map_data=SimpleNamespace(xyz=xyz)
        engine.retriever=Mock(side_effect=AssertionError('No DINO during pose-local search'))
        query={'keypoints':torch.tensor(uv[None],dtype=torch.float32),
               'descriptors':torch.eye(20)[None]}
        frame=np.zeros((480,640,3),np.uint8)
        return engine,frame,k,d,r,t,uv,query

    def test_point_pipeline_uses_measured_pixels_not_pose_predictions(self):
        engine,frame,k,d,r,t,uv,query=self.point_engine()
        pairs=np.column_stack((np.arange(20),np.arange(20)))
        with patch('registered_aliked_live.crop_live_object',return_value=(frame,(0,0,640,480))), \
             patch('registered_aliked_live._extract',return_value=query), \
             patch('registered_aliked_live.match_projected_landmarks',return_value=(pairs,np.ones(20),
                dict(descriptor_pass=20,mutual_pass=20,neighbourhood_pairs=40))):
            result,_=engine.localize_guided(frame,k,d,r,t)
        self.assertEqual(result.status,'PASS')
        self.assertLess(result.mean_reprojection_error,.001)
        expected=cv2.projectPoints(engine.last_debug.matched_xyz,r,t,k,d)[0].reshape(-1,2)
        np.testing.assert_allclose(engine.last_debug.image_points,expected,atol=1e-4)
        self.assertIsNotNone(engine.last_tracking_seed)
        self.assertEqual(engine.last_diagnostics['reference_matching'],'pose_landmarks')
        self.assertIsNone(engine.progress_snapshot()['reference_index'])
        self.assertEqual(engine.progress_snapshot()['stage'],'done')

    def test_failed_point_search_does_not_leave_a_seed_or_start_global_retrieval(self):
        engine,frame,k,d,r,t,uv,query=self.point_engine()
        engine.last_tracking_seed='old'
        engine.landmark_bank.candidates.return_value=LandmarkCandidates(
            np.empty(0,int),np.empty((0,3)),np.empty((0,2)),np.empty((0,20)),np.empty(0))
        result,_=engine.localize_guided(frame,k,d,r,t)
        self.assertEqual(result.detail,'NO_POSE_LOCAL_LANDMARKS')
        self.assertIsNone(engine.last_tracking_seed)
        engine.retriever.assert_not_called()

    def test_face_validation_checks_original_pass_not_unrelated_coarse_orientation(self):
        engine,frame,k,d,r,t,uv,query=self.point_engine()
        engine.fundamental_threshold=3.;engine.top_k=1;engine.max_coarse_rotation_deg=55.
        engine.mpr_renderer=None;engine.retrieval_mode='dino-patch'
        virtual=np.eye(4);virtual[:3,:3]=cv2.Rodrigues(np.array([np.pi,0.,0.]))[0]
        engine.views=[dict(image_name='virtual.png',world_to_camera=virtual)]
        query['image_size']=engine.torch.tensor([[640.,480.]])
        pairs=np.column_stack((np.arange(20),np.arange(20)))
        engine._reference=Mock(return_value=(query,np.arange(20),engine.map_data.xyz))
        def run(validation):
            workspace=dict(retrieval=([(1.,0.,0)],0,[],dict(bbox=[0.,0.,1.,1.])),
                query=query,references=[('old.png',0,0.,0.)])
            if validation:
                workspace['validation_pose']=(r,t)
            return engine._localize_once(frame,k,d,workspace=workspace)[0]
        with patch('registered_aliked_live.crop_live_object',return_value=(frame,(0,0,640,480))), \
             patch('registered_aliked_live._match_pair',return_value=(uv,uv,pairs,np.ones(20))), \
             patch('registered_aliked_live.cv2.findFundamentalMat',return_value=(np.eye(3),np.ones((20,1)))):
            normal=run(False)
            self.assertEqual(normal.status,'FAIL')
            self.assertIn('coarse_pose_mismatch',normal.detail)
            frozen=run(True)
        self.assertEqual(frozen.status,'PASS')
        self.assertEqual(engine.last_diagnostics['quality']['pose_comparison']['reference'],'original_pass')
        self.assertLess(engine.last_diagnostics['quality']['pose_comparison']['rotation_deg'],.001)

    def test_descriptor_selection_includes_camera_roll_not_only_viewing_ray(self):
        xyz=np.array([[0.,0.,10.],[1.,0.,10.]])
        rotations=np.stack([cv2.Rodrigues(np.array([0.,0.,np.pi]))[0],np.eye(3),np.eye(3)])
        bank=PoseLandmarkBank(xyz,[0,0,1],np.eye(3),np.tile([0.,0.,-1.],(3,1)),
                              [1.,.5,1.],rotations)
        k=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        chosen=bank.candidates(np.zeros(3),np.zeros(3),k,np.zeros(5),(100,100,3),variants=1)
        np.testing.assert_array_equal(chosen.descriptors[0],[0.,1.,0.])

    def test_fixed_reference_and_accepted_pose_replace_dino_rerank_and_live_state_is_restored(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock()
        engine.registered={'old.png':{}}
        engine.last_tracking_seed='live'
        engine.max_reprojection_error=3.
        engine._progress('before')
        engine._pose_guided_views=Mock(return_value=[7])
        frame=np.zeros((40,40,3),np.uint8)
        expected=LiveResult('PASS','known',10,10,10,1.,1.,None,None,None,0.,'HIDDEN')
        baseline=(np.zeros(3),np.array([0.,0.,100.]))
        def attempts(image,k,d,polygon,workspace):
            self.assertEqual(workspace['references'],[('old.png',7,0.,0.)])
            self.assertIs(workspace['validation_pose'],baseline)
            self.assertEqual(engine.max_reprojection_error,5.)
            self.assertEqual(engine.min_coverage,0.)
            engine.last_tracking_seed='validation'
            engine.last_debug='result debug'
            engine.last_diagnostics={'quality':{'reasons':[]}}
            return expected,frame
        engine._search_attempts=Mock(side_effect=attempts)
        result,debug,_=engine.localize_validation(frame,np.eye(3),np.zeros(5),
            np.zeros((4,2)),5.,0.,['old.png'],baseline)
        self.assertIs(result,expected)
        self.assertEqual(debug,'result debug')
        self.assertEqual(engine.last_tracking_seed,'live')
        self.assertEqual(engine.max_reprojection_error,3.)
        self.assertFalse(hasattr(engine,'min_coverage'))
        self.assertFalse(hasattr(engine,'last_debug'))
        self.assertEqual(engine.progress_snapshot()['stage'],'before')

    def test_unknown_original_reference_fails_explicitly_and_restores_limits(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock();engine.registered={};engine._progress('before')
        with self.assertRaisesRegex(ValueError,'unavailable'):
            engine.localize_validation(None,None,None,None,5.,0.,['missing.png'],
                                       (np.zeros(3),np.zeros(3)))
        self.assertFalse(hasattr(engine,'max_reprojection_error'))
        self.assertFalse(hasattr(engine,'min_coverage'))

    def test_pose_landmark_validation_reuses_search_pose_and_face_filter(self):
        engine=RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.lock=threading.RLock();engine._progress('before')
        engine.last_debug='debug';engine.last_diagnostics={}
        engine._localize_landmarks=Mock(return_value=('result',None))
        polygon=np.ones((4,2));r=np.zeros(3);t=np.zeros(3)
        actual=engine.localize_validation(None,None,None,polygon,3.,0.,['POSE_LANDMARKS'],(r,t))
        self.assertEqual(actual[0],'result')
        self.assertIs(engine._localize_landmarks.call_args.kwargs['excluded_polygon'],polygon)
        self.assertIs(engine._localize_landmarks.call_args.args[-2],r)


if __name__=='__main__':
    unittest.main()
