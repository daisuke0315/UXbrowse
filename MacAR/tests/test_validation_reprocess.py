import time
import json
import subprocess
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

from browser_live import (BrowserApplication, DEFAULTS, projected_model_faces,
                          validation_qc_diagnosis)
from realtime_pnp import build_parser
from registered_aliked_live import exclude_query_polygon
from glb_virtual_live import GLBVirtualLiveLocalizer, exclude_virtual_query_polygon
from pose_quality import assess_pose
from realtime_pnp_server import LocalizationDebug
from test_browser_live import renderer, result


class ValidationReprocessTests(unittest.TestCase):
    def test_early_roi_failure_is_not_hidden_by_zero_inlier_qc(self):
        failed = SimpleNamespace(status='FAIL', inliers=0, inlier_ratio=0.,
                                 mean_reprojection_error=None, detail='NO_OBJECT_ROI')
        detection = dict(reason='no_confident_object', search_pass='tiles', trials=[{}, {}])
        finding = validation_qc_diagnosis(failed,
            dict(reason='NO_OBJECT_ROI', object_detection=detection), SimpleNamespace())
        self.assertEqual(finding['reasons'][0], 'NO_OBJECT_ROI')
        self.assertEqual(finding['object_detection'], detection)

    def test_detection_failure_and_enlarged_search_are_explained_in_browser(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/validation-diagnostics.js'
        script="""
        const {validationReasons,validationStageSummary}=await import(process.argv[1]);
        const result={status:'FAIL',inliers:0,diagnosis:{reasons:['3DGS_NO_OBJECT_ROI'],
          object_detection:{search_pass:'tiles',trials:[{},{},{},{},{}]}}};
        console.log(JSON.stringify([validationReasons(result),validationStageSummary(result)]));
        """
        reason, stages=json.loads(subprocess.check_output(
            [str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertIn('対象領域を検出できず',reason)
        self.assertIn('対象検出 5回',stages)
        self.assertIn('未検出',stages)

    def test_browser_face_hit_testing(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/measurement-workspace.js'
        script="""
        import {readFileSync} from 'node:fs';
        const {pointInsidePolygon}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        const poly=[[10,10],[30,10],[30,30],[10,30]];
        console.log(JSON.stringify([pointInsidePolygon([20,20],poly),pointInsidePolygon([40,20],poly)]));
        """
        self.assertEqual(json.loads(subprocess.check_output(
            [str(node),'--input-type=module','-e',script,str(source)],text=True)),[True,False])

    def test_exclude_face_matches_before_geometry_and_pnp(self):
        polygon=np.array([[10.,10.],[30.,10.],[30.,30.],[10.,30.]])
        pairs=np.array([[0,0],[1,1],[2,2],[3,3]])
        scores=np.ones(4)
        uv=np.array([[20.,20.],[10.,20.],[35.,20.],[5.,5.]])
        output=exclude_query_polygon(pairs,scores,uv,uv.copy(),polygon)
        self.assertEqual(output[-1],2)
        np.testing.assert_array_equal(output[0],pairs[2:])

    def test_virtual_face_filter_uses_original_image_uv_after_object_crop(self):
        pairs=np.array([[0,0],[1,1],[2,2]])
        scores=np.ones(3)
        cropped_keypoints=np.array([[5.,5.],[25.,25.],[45.,45.]])
        polygon=np.array([[100.,100.],[130.,100.],[130.,130.],[100.,130.]])
        kept, kept_scores, rejected=exclude_virtual_query_polygon(
            pairs,scores,cropped_keypoints,[100.,100.],polygon)
        self.assertEqual(rejected,2)
        np.testing.assert_array_equal(kept,pairs[2:])
        np.testing.assert_array_equal(kept_scores,scores[2:])

    def test_virtual_comparison_reuses_one_view_shortlist(self):
        engine=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        engine.lock=threading.Lock()
        engine.progress_lock=threading.Lock()
        engine.progress={'stage':'before_validation'}
        engine.last_tracking_seed='live_seed'
        engine.max_reprojection_error=3.
        engine.min_coverage=.005
        engine.top_k=3
        engine.views=[None]*10
        image=np.zeros((30,30,3),np.uint8)
        polygon=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
        ranked=[(.9,0.,3),(.8,0.,7)]
        calls=[]
        def rank(_):
            engine.last_crop_debug={'bbox':(.1,.1,.8,.8)}
            return ranked,12
        engine._rank_views=rank
        engine._quality_attempts=lambda attempt:attempt()
        def localize(*args,**kwargs):
            calls.append(kwargs)
            engine.last_debug=f'debug{len(calls)}'
            engine.last_tracking_seed=f'validation_seed{len(calls)}'
            engine.last_diagnostics={'excluded_face_matches':len(calls)-1}
            self.assertEqual((engine.max_reprojection_error,engine.min_coverage,engine.top_k),(4.,.0003,10))
            return result(),None
        engine._localize_locked=localize
        baseline,excluded,debug,diagnostics=engine.compare_excluded_face(
            image,np.eye(3),np.zeros(5),polygon,4.,.0003,10)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['ranked_override'],ranked)
        self.assertEqual(calls[1]['ranked_override'],ranked)
        self.assertEqual(calls[0]['query_workspace']['crop_debug'],
                         calls[1]['query_workspace']['crop_debug'])
        self.assertIsNone(calls[0].get('excluded_polygon'))
        np.testing.assert_array_equal(calls[1]['excluded_polygon'].reshape(-1,2),polygon)
        self.assertEqual(debug,'debug2')
        self.assertEqual(diagnostics['excluded_face_matches'],1)
        self.assertEqual(diagnostics['baseline_diagnostics']['excluded_face_matches'],0)
        self.assertEqual(engine.max_reprojection_error,3.)
        self.assertEqual(engine.min_coverage,.005)
        self.assertEqual(engine.top_k,3)
        self.assertEqual(engine.last_tracking_seed,'live_seed')
        self.assertEqual(engine.progress,{'stage':'before_validation'})

    def test_virtual_comparison_uses_original_pass_views_without_reranking(self):
        engine=GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        engine.lock=threading.Lock()
        engine.progress_lock=threading.Lock()
        engine.progress={}
        engine.max_reprojection_error=3.
        engine.min_coverage=.005
        engine.top_k=3
        engine.views=[{'image_name':'view_a.png'},{'image_name':'view_b.png'}]
        engine.retrieval_auto_roi=False
        engine.retrieval_crop=False
        engine._rank_views=lambda _:self.fail('Original-view mode must not rerank')
        engine._quality_attempts=lambda attempt:attempt()
        calls=[]
        def localize(*args,**kwargs):
            calls.append(kwargs)
            engine.last_debug=None
            engine.last_diagnostics={}
            return result(),None
        engine._localize_locked=localize
        polygon=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
        engine.compare_excluded_face(np.zeros((30,30,3),np.uint8),np.eye(3),
                                     np.zeros(5),polygon,reference_names=('view_b.png',))
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['ranked_override'],[(1.,0.,1)])
        self.assertEqual(calls[1]['ranked_override'],[(1.,0.,1)])
        self.assertIsNone(calls[0].get('excluded_polygon'))
        np.testing.assert_array_equal(calls[1]['excluded_polygon'].reshape(-1,2),polygon)
        self.assertIsNot(calls[0]['query_workspace'],calls[1]['query_workspace'])

    def test_uncalibrated_hfov_defaults_to_72_degrees(self):
        self.assertEqual(DEFAULTS['fov_deg'], 72.)
        self.assertEqual(build_parser().parse_args([]).fov_deg, 72.)

    def setUp(self):
        self.app = BrowserApplication()
        self.app.settings = dict(DEFAULTS, mode='registered', input_source='camera')
        self.app.args = SimpleNamespace(matching_mode='registered-aliked',
                                        max_processing_width=1280, ct_validation=True)
        self.app.renderer = renderer()
        self.app.manifest = {'bounds': [[-20., -20., -20.], [20., 20., 20.]]}
        self.image = np.zeros((100, 100, 3), np.uint8)
        self.matrix = np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]])
        self.distortion = np.zeros(5)
        self.app.capture = SimpleNamespace(read=lambda: (True, self.image.copy()),
                                           stats=lambda: {}, release=lambda: None)
        self.engine = SimpleNamespace(last_debug=None, localize=None)
        self.app.localizer = SimpleNamespace(localizer=self.engine, close=lambda: None)
        self.app.last_valid = (self.image.copy(), result(), self.matrix,
                               self.distortion, time.monotonic(), 10, 'test')
        self.app.controls({'action': 'freeze'})
        self.app.validation_reference_views=('view_0052.png',)

    def tearDown(self):
        self.app.stop()
        self.app.executor.shutdown(wait=True, cancel_futures=True)
        self.app.layer_executor.shutdown(wait=True, cancel_futures=True)

    def process_one(self, matched, limit=None):
        observed = {}
        def localize(image, matrix, distortion, *, excluded_polygon=None):
            observed['image'] = image.copy()
            observed['polygon'] = excluded_polygon.copy()
            return matched, None
        self.engine.localize = localize
        command={'action': 'validation_reprocess', 'face': 'z_min'}
        if limit is not None:
            command['max_mean_error_px']=limit
        self.app.controls(command)
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        np.testing.assert_array_equal(observed['image'], self.image)
        self.assertEqual(observed['polygon'].shape, (4, 2))

    def test_reprocess_pass_replaces_frozen_frame_and_preserves_baseline(self):
        newer = result()
        self.process_one(newer)
        comparison = self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['baseline']['inliers'], 15)
        self.assertEqual(comparison['latest']['status'], 'PASS')
        self.assertTrue(comparison['applied'])
        self.assertEqual(self.app.frozen_source[5], 10)
        self.assertEqual(self.app.frozen_source[1], newer)
        self.assertEqual(self.app.snapshot()['validation_reprocess_status'], 'done')
        self.assertEqual(self.app.validation_baseline['inliers'], 15)
        self.assertEqual(comparison['attempt'], 1)
        self.assertIsNotNone(self.app.validation_baseline_preview)
        self.assertIsNotNone(self.app.validation_latest_preview)
        self.assertEqual(comparison['face'], 'z_min')
        self.assertIsNotNone(self.app.session)
        self.app.publish(*self.app.frozen_source, force=True)
        displayed = self.app.last_published[1]
        self.assertTrue(displayed['frozen'])
        self.assertEqual(displayed['validation_comparison']['attempt'], 1)
        self.assertTrue(displayed['validation_comparison']['applied'])
        self.assertIsNotNone(displayed['pose'])
        self.app.display_plane_revision = self.app.control_revision
        self.app.controls({'action': 'point', 'kind': 'live', 'x': 50, 'y': 50,
                           'plane_revision': self.app.control_revision})
        self.assertEqual(self.app.measurement()['points'][0]['label'], 'LIVE')

    def test_reprocess_fail_keeps_previous_frozen_pose(self):
        failed = result('FAIL')
        original = self.app.frozen_source
        self.process_one(failed)
        self.assertIs(self.app.frozen_source, original)
        comparison = self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['latest']['status'], 'FAIL')
        self.assertEqual(comparison['baseline']['status'], 'PASS')
        self.assertFalse(comparison['applied'])

    def test_occlusion_relocalizes_both_inputs_and_measures_on_original_rgb(self):
        from unittest.mock import patch
        self.image[:] = 200
        self.app.validation_baseline_source[0][:] = 200
        newer = result()
        with patch('browser_live.compare_image_occlusion', return_value=(
                result(), newer, None, {'baseline_diagnostics': {}, 'occluded_fraction': .1})) as run:
            self.app.controls(dict(action='validation_occlusion',
                polygon=[[10,10],[30,10],[30,30],[10,30]]))
            self.app.validation_future.result(timeout=5)
            self.app.finish_validation_reprocess()
        self.assertEqual(run.call_count, 1)
        comparison = self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['kind'], 'occlusion')
        self.assertEqual(comparison['reference_policy'], 'fresh_global')
        self.assertTrue(comparison['applied'])
        self.assertIs(self.app.frozen_source[1], newer)
        self.assertTrue(np.all(self.app.frozen_source[0] == 200))
        self.assertEqual(comparison['occluded_fraction'], .1)

    def test_occlusion_failed_baseline_does_not_apply_masked_pose(self):
        from unittest.mock import patch
        original = self.app.frozen_source
        with patch('browser_live.compare_image_occlusion', return_value=(
                result('FAIL'), result(), None, {'baseline_diagnostics': {}})):
            self.app.controls(dict(action='validation_occlusion',
                polygon=[[10,10],[30,10],[30,30],[10,30]]))
            self.app.validation_future.result(timeout=5)
            self.app.finish_validation_reprocess()
        self.assertIs(self.app.frozen_source, original)
        self.assertFalse(self.app.snapshot()['validation_comparison']['applied'])

    def test_occlusion_original_policy_passes_frozen_refs_and_records_them(self):
        from unittest.mock import patch
        for mode in ('gs','registered'):
            for input_source in ('photo','camera'):
                with self.subTest(mode=mode,input_source=input_source):
                    self.app.settings.update(mode=mode,input_source=input_source)
                    self.app.args.matching_mode='splat-track' if mode=='gs' else 'registered-aliked'
                    self.app.photo_matcher=self.engine
                    self.app.localizer=SimpleNamespace(global_localizer=self.engine,
                        localizer=self.engine,close=lambda:None)
                    self.app.validation_reference_views=('original.png',)
                    with patch('browser_live.compare_image_occlusion',return_value=(
                            result(),result(),None,{'baseline_diagnostics':{}})) as run:
                        self.app.controls(dict(action='validation_occlusion',reference_policy='original',
                            polygon=[[10,10],[30,10],[30,30],[10,30]]))
                        self.app.validation_future.result(timeout=5)
                        self.app.finish_validation_reprocess()
                    self.assertEqual(run.call_args.kwargs['reference_names'],('original.png',))
                    np.testing.assert_allclose(run.call_args.kwargs['baseline_pose'][0],
                        self.app.validation_baseline_source[1].rvec)
                    comparison=self.app.snapshot()['validation_comparison']
                    self.assertEqual(comparison['reference_policy'],'original')
                    self.assertEqual(comparison['reference_views'],['original.png'])
                    self.assertTrue(comparison['applied'])

    def test_occlusion_rejects_invalid_policy_or_missing_fixed_views_before_work(self):
        from unittest.mock import patch
        for policy,names in (('invalid',('original.png',)),('original',()),
                             ('original',('POSE_LANDMARKS',))):
            with self.subTest(policy=policy,names=names):
                self.app.validation_reference_views=names
                with patch('browser_live.compare_image_occlusion') as run:
                    with self.assertRaises(ValueError):
                        self.app.controls(dict(action='validation_occlusion',reference_policy=policy,
                            polygon=[[10,10],[30,10],[30,30],[10,30]]))
                    self.assertFalse(run.called)

    def test_occlusion_supports_gs_and_registered_with_photo_or_camera_sources(self):
        from unittest.mock import patch
        for mode in ('gs', 'registered'):
            for input_source in ('photo', 'camera'):
                with self.subTest(mode=mode, input_source=input_source):
                    self.app.settings.update(mode=mode, input_source=input_source)
                    self.app.args.matching_mode = 'splat-track' if mode == 'gs' else 'registered-aliked'
                    self.app.photo_matcher = self.engine
                    self.app.localizer = SimpleNamespace(global_localizer=self.engine,
                        localizer=self.engine, close=lambda: None)
                    with patch('browser_live.compare_image_occlusion', return_value=(
                            result(), result(), None, {'baseline_diagnostics': {}})) as run:
                        self.app.controls(dict(action='validation_occlusion',
                            polygon=[[10,10],[30,10],[30,30],[10,30]]))
                        self.app.validation_future.result(timeout=5)
                        self.app.finish_validation_reprocess()
                    self.assertIs(run.call_args.args[0], self.engine)
                    self.assertEqual(self.app.validation_reference_views, ('view_0052.png',))
                    self.assertEqual(self.app.validation_comparison['reference_views'], [])
                    self.assertTrue(self.app.validation_comparison['applied'])

    def test_registered_reprocess_preserves_original_reference_and_uses_returned_debug(self):
        seen=[]
        debug=LocalizationDebug(np.zeros((2,3)),frozenset({0}),(0.,0.,-100.),True,
                                np.array([[20.,30.],[70.,70.]]))
        def compare(image,k,d,polygon,max_error,min_coverage,reference_names,baseline_pose):
            seen.append((reference_names,baseline_pose))
            return result(),debug,dict(excluded_face_matches=7,
                quality=dict(reasons=[],pose_comparison=dict(reference='original_pass',rotation_deg=1.)))
        self.engine.localize_validation=compare
        self.engine.last_debug='LIVE STATE MUST NOT BE USED'
        self.app.validation_reference_views=('original.png',)
        self.app.controls(dict(action='validation_reprocess',face='z_min'))
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        self.assertEqual(seen[0][0],('original.png',))
        np.testing.assert_allclose(seen[0][1][0],self.app.validation_baseline_source[1].rvec)
        self.assertIsNotNone(self.app.frozen_debug)
        comparison=self.app.snapshot()['validation_comparison']
        self.assertTrue(comparison['applied'])
        self.assertEqual(comparison['reference_policy'],'original')
        self.assertEqual(comparison['latest']['excluded_face_matches'],7)
        self.assertEqual(comparison['latest']['diagnosis']['pose_comparison']['reference'],'original_pass')

    def test_registered_photo_freeze_records_actual_accepted_rc_reference(self):
        self.app.settings['input_source']='photo'
        self.app.photo_matcher=SimpleNamespace(registered={'original.png':{}},
            last_diagnostics=dict(selected='original.png'))
        self.assertEqual(self.app.current_gs_reference_views(),('original.png',))

    def test_photo_can_reprocess_same_image_but_unfrozen_cannot(self):
        self.app.settings['input_source'] = 'photo'
        self.app.photo_matcher = self.engine
        self.process_one(result('FAIL'))
        self.app.settings['input_source'] = 'camera'
        self.app.controls({'action': 'resume'})
        with self.assertRaisesRegex(ValueError, '固定'):
            self.app.controls({'action': 'validation_reprocess', 'face': 'z_min'})

    def test_only_visible_face_can_be_selected(self):
        faces = projected_model_faces(self.app.manifest['bounds'], result(),
                                       self.matrix, self.distortion)
        self.assertIn('z_min', {face['id'] for face in faces})
        self.assertNotIn('z_max', {face['id'] for face in faces})
        with self.assertRaisesRegex(ValueError, '見える'):
            self.app.controls({'action': 'validation_reprocess', 'face': 'z_max'})

    def test_normal_gs_mode_compares_frozen_image_without_tracking(self):
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        observed={}
        def compare(image,matrix,distortion,polygon,max_error,min_coverage,view_count,reference_names):
            observed['image']=image.copy()
            observed['polygon']=polygon.copy()
            observed['max_error']=max_error
            observed['min_coverage']=min_coverage
            observed['view_count']=view_count
            observed['reference_names']=reference_names
            return result(),result(),None,{'excluded_face_matches':4}
        self.engine.compare_excluded_face=compare
        self.app.localizer=SimpleNamespace(global_localizer=self.engine,
                                          close=lambda:None)
        self.app.controls({'action':'validation_reprocess','face':'z_min',
                           'view_count':10})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        np.testing.assert_array_equal(observed['image'],self.image)
        self.assertEqual(observed['polygon'].shape,(4,2))
        self.assertEqual(observed['max_error'],3.)
        self.assertEqual(observed['min_coverage'],.005)
        self.assertIsNone(observed['view_count'])
        self.assertEqual(observed['reference_names'],('view_0052.png',))
        comparison=self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['baseline']['status'],'PASS')
        self.assertEqual(comparison['latest']['excluded_face_matches'],4)
        self.assertEqual(self.app.snapshot()['validation_reprocess_status'],'done')

    def test_gs_default_requires_original_pass_reference(self):
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        self.app.validation_reference_views=()
        self.app.localizer=SimpleNamespace(global_localizer=self.engine,close=lambda:None)
        with self.assertRaisesRegex(ValueError,'元PASS時の参照view'):
            self.app.controls({'action':'validation_reprocess','face':'z_min'})

    def test_gs_rerank_remains_explicit_diagnostic_option(self):
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        self.app.validation_reference_views=()
        seen=[]
        self.engine.compare_excluded_face=lambda *args: (
            seen.append(args) or (result(),result('FAIL'),None,{}))
        self.app.localizer=SimpleNamespace(global_localizer=self.engine,close=lambda:None)
        self.app.controls({'action':'validation_reprocess','face':'z_min',
                           'reference_policy':'rerank','view_count':5})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        self.assertEqual(seen[0][-2:],(5,None))
        self.assertEqual(self.app.snapshot()['validation_comparison']['reference_policy'],'rerank')

    def test_gs_photo_pass_views_include_merged_sources(self):
        self.app.args.matching_mode='splat-track'
        self.app.settings.update(mode='gs',input_source='photo')
        self.app.manifest['views']=[{'image_name':'view_a.png'},{'image_name':'view_b.png'}]
        self.app.photo_matcher=SimpleNamespace(last_diagnostics={
            'selected':'MERGED_2_view_a.png',
            'selected_reference_views':['view_a.png','view_b.png']})
        self.assertEqual(self.app.current_gs_reference_views(),('view_a.png','view_b.png'))

    def test_freeze_snapshots_original_pass_views_once(self):
        self.app.controls({'action':'resume'})
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        self.app.last_valid=(self.image.copy(),result(),self.matrix,
                             self.distortion,time.monotonic(),11,'test')
        self.app.last_valid_reference_views=('view_a.png','view_b.png')
        self.app.controls({'action':'freeze'})
        self.assertEqual(self.app.validation_reference_views,('view_a.png','view_b.png'))
        self.app.last_valid_reference_views=('view_c.png',)
        self.app.freeze_validation_source(self.app.frozen_source,None)
        self.assertEqual(self.app.validation_reference_views,('view_a.png','view_b.png'))

    def test_normal_gs_baseline_failure_does_not_replace_displayed_pose(self):
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        self.engine.compare_excluded_face=lambda *args: (
            result('FAIL'),result(),None,{'excluded_face_matches':2})
        self.app.localizer=SimpleNamespace(global_localizer=self.engine,
                                          close=lambda:None)
        original=self.app.frozen_source
        self.app.controls({'action':'validation_reprocess','face':'z_min'})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        self.assertIs(self.app.frozen_source,original)
        comparison=self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['baseline']['status'],'FAIL')
        self.assertEqual(comparison['latest']['status'],'PASS')
        self.assertFalse(comparison['applied'])

    def test_failed_face_preview_marks_remaining_inliers_and_outliers(self):
        self.app.settings['mode']='gs'
        self.app.args.matching_mode='splat-track'
        debug=SimpleNamespace(image_points=np.array([[20.,30.],[70.,70.]]),
                              inlier_indices=frozenset({0}))
        self.engine.compare_excluded_face=lambda *args: (
            result(),result('FAIL'),debug,{'excluded_face_matches':2})
        self.app.localizer=SimpleNamespace(global_localizer=self.engine,
                                          close=lambda:None)
        self.app.controls({'action':'validation_reprocess','face':'z_min'})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        preview=cv2.imdecode(np.frombuffer(self.app.validation_latest_preview,np.uint8),
                             cv2.IMREAD_COLOR)
        self.assertGreater(int(preview[30,20,1]),int(preview[30,20,2]))
        self.assertGreater(int(preview[70,70,2]),int(preview[70,70,1]))

    def test_measurement_return_only_uses_applied_face_pose(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source=(Path(__file__).resolve().parents[1]/'browser_ui/measurement-workspace.js').as_uri()
        script="""
        const {MeasurementWorkspace}=await import(process.argv[1]);
        const elements={validationCompare:{open:true},measureOverlay:{checked:false}};
        globalThis.document={getElementById:id=>elements[id]};
        let draws=0,focuses=0;
        const workspace={meta:{frozen:true,validation_comparison:{applied:false}},facePicking:true,
            panes:{live:{canvas:{focus(){focuses++},scrollIntoView(){}}}},draw(){draws++}};
        MeasurementWorkspace.prototype.returnToMeasurement.call(workspace);
        const blocked=[elements.validationCompare.open,elements.measureOverlay.checked,draws];
        workspace.meta.validation_comparison.applied=true;
        MeasurementWorkspace.prototype.returnToMeasurement.call(workspace);
        console.log(JSON.stringify({blocked,after:[elements.validationCompare.open,
            elements.measureOverlay.checked,draws,focuses,workspace.facePicking]}));
        """
        data=json.loads(subprocess.check_output(
            [str(node),'--input-type=module','-e',script,source],text=True))
        self.assertEqual(data['blocked'],[True,False,0])
        self.assertEqual(data['after'],[False,True,1,1,False])

    def test_normal_gs_photo_uses_photo_engine_for_comparison(self):
        self.app.settings.update(mode='gs',input_source='photo')
        self.app.args.matching_mode='splat-track'
        seen=[]
        def compare(image,matrix,distortion,polygon,max_error,min_coverage,view_count,reference_names):
            seen.append(image.copy())
            self.assertEqual(min_coverage,.005)
            self.assertIsNone(view_count)
            self.assertEqual(reference_names,('view_0052.png',))
            return result(),result('FAIL'),None,{'excluded_face_matches':3}
        self.engine.compare_excluded_face=compare
        self.app.photo_matcher=self.engine
        self.app.controls({'action':'validation_reprocess','face':'z_min'})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        self.assertEqual(len(seen),1)
        np.testing.assert_array_equal(seen[0],self.image)
        self.assertEqual(self.app.snapshot()['validation_comparison']['latest']['excluded_face_matches'],3)

    def test_validation_threshold_is_temporary_and_is_reported(self):
        self.engine.max_reprojection_error=3.
        self.engine.min_coverage=.005
        seen=[]
        def localize(image,matrix,distortion,*,excluded_polygon=None):
            seen.append((self.engine.max_reprojection_error,self.engine.min_coverage))
            self.engine.last_diagnostics={'quality':{'reasons':['reprojection_error'],
                'coverage':.008,'p95_error':4.8,'max_p95_error':6.0},
                'excluded_face_matches':12}
            return result('FAIL'),None
        self.engine.localize=localize
        self.app.controls({'action':'validation_reprocess','face':'z_min',
                           'max_mean_error_px':4.,'min_coverage_percent':.03})
        self.app.validation_future.result(timeout=5)
        self.app.finish_validation_reprocess()
        self.assertEqual(seen,[(4.,.0003)])
        self.assertEqual(self.engine.max_reprojection_error,3.)
        self.assertEqual(self.engine.min_coverage,.005)
        comparison=self.app.snapshot()['validation_comparison']
        self.assertEqual(comparison['max_mean_error_px'],4.)
        self.assertAlmostEqual(comparison['min_coverage'],.0003)
        self.assertEqual(comparison['latest']['diagnosis']['limits']['max_mean_px'],4.)
        self.assertAlmostEqual(comparison['latest']['diagnosis']['limits']['min_coverage'],.0003)
        with self.assertRaisesRegex(ValueError,'0.5〜12'):
            self.app.controls({'action':'validation_reprocess','face':'z_min',
                               'max_mean_error_px':20.})
        with self.assertRaisesRegex(ValueError,'0〜2%'):
            self.app.controls({'action':'validation_reprocess','face':'z_min',
                               'min_coverage_percent':3.})

    def test_diagnosis_distinguishes_count_error_and_spatial_support(self):
        failed=SimpleNamespace(status='FAIL',inliers=10,inlier_ratio=10/18,
            mean_reprojection_error=3.05,detail='GS REF image')
        engine=SimpleNamespace(min_inliers=10,min_inlier_ratio=.2,
                               max_reprojection_error=3.)
        diagnostics={'quality':{'reasons':['clustered_image_support'],
            'coverage':.003,'p95_error':5.1,'max_p95_error':4.5},
            'selected':'view_01.png','excluded_face_matches':348,
            'candidates':[{'image_name':'view_01.png','raw_local_matches':30,
                'geometric_inliers':22,'xyz_correspondences':18}]}
        finding=validation_qc_diagnosis(failed,diagnostics,engine)
        self.assertNotIn('few_inliers',finding['reasons'])
        self.assertIn('reprojection_error',finding['reasons'])
        self.assertIn('clustered_image_support',finding['reasons'])
        self.assertEqual(finding['stages']['xyz_correspondences'],18)
        self.assertEqual(finding['candidate_stages'][0]['reference'],'view_01.png')
        self.assertEqual(finding['candidate_stages'][0]['geometric_inliers'],22)

    def test_small_face_support_requires_explicit_validation_relaxation(self):
        uv=np.array([[49.,49.],[51.,49.],[51.,51.],[49.,51.]])
        xyz=np.column_stack((uv[:,0]-50.,uv[:,1]-50.,np.full(4,100.)))
        matrix=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        def check(limit):
            return assess_pose(xyz,uv,np.zeros(3),np.zeros(3),np.arange(4),
                (100,100),matrix,np.zeros(5),4,.2,3.,min_coverage=limit)
        strict=check(.005)
        relaxed=check(.0003)
        self.assertAlmostEqual(strict['coverage'],.0004)
        self.assertIn('clustered_image_support',strict['reasons'])
        self.assertTrue(relaxed['passed'])

    def test_registered_diagnosis_exposes_xyz_shortage_and_fallback(self):
        failed=SimpleNamespace(status='FAIL',inliers=7,inlier_ratio=7/12,
            mean_reprojection_error=2.2,detail='few_inliers')
        engine=SimpleNamespace(min_inliers=10,min_inlier_ratio=.2,max_reprojection_error=3.)
        diagnostics=dict(reference_matching='xyz_only_fallback',selected='ref.png',
            all_feature_result=dict(status='FAIL',inliers=0,matches=0),
            xyz_only_fallback=dict(status='FAIL',inliers=7,adopted=True),
            candidates=[dict(name='ref.png',reference_features=1364,reference_xyz_features=126,
                raw=209,geometric=48,transferred=3,xyz_rejections=dict(missing_xyz=45,duplicate=0))])
        finding=validation_qc_diagnosis(failed,diagnostics,engine)
        self.assertEqual(finding['stages']['missing_xyz'],45)
        self.assertEqual(finding['stages']['reference_features'],1364)
        self.assertEqual(finding['candidate_stages'][0]['reference_xyz_features'],126)
        self.assertEqual(finding['all_feature_result']['inliers'],0)
        self.assertTrue(finding['xyz_only_fallback']['adopted'])

    def test_browser_labels_registered_fallback_and_missing_xyz(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source=(Path(__file__).resolve().parents[1]/'browser_ui/validation-diagnostics.js').as_uri()
        script="""
        const {validationStageSummary}=await import(process.argv[1]);
        console.log(validationStageSummary({inliers:7,diagnosis:{
          reference_matching:'xyz_only_fallback',all_feature_result:{status:'FAIL',inliers:0},
          xyz_only_fallback:{status:'FAIL',inliers:7,adopted:true},
          stages:{reference_features:126,reference_xyz_features:126,missing_xyz:0}}}));
        """
        output=subprocess.check_output([str(node),'--input-type=module','-e',script,source],text=True)
        self.assertIn('XYZ付き再試行 FAIL/7点（再試行を採用）',output)
        self.assertIn('参照特徴 126（XYZ付き 126）',output)
        self.assertIn('XYZ未登録 0',output)

    def test_browser_explains_failed_qc_without_claiming_ct_error(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        source=(Path(__file__).resolve().parents[1]/'browser_ui/validation-diagnostics.js').as_uri()
        script="""
        const {validationReasons,validationStageSummary}=await import(process.argv[1]);
        const result={status:'FAIL',inliers:10,matches:18,inlier_ratio:10/18,
          mean_reprojection_error:3.05,diagnosis:{reasons:['reprojection_error'],
          limits:{min_inliers:10,min_ratio:.2,max_mean_px:3},
          stages:{reference:'view_01.png',raw_matches:30,geometric_inliers:22,xyz_correspondences:18}}};
        console.log(JSON.stringify({reason:validationReasons(result),stages:validationStageSummary(result)}));
        """
        data=json.loads(subprocess.check_output(
            [str(node),'--input-type=module','-e',script,source],text=True))
        self.assertIn('3.05 px > 上限 3.00 px',data['reason'])
        self.assertNotIn('CT',data['reason'])
        self.assertIn('局所照合 30 → 幾何整合 22 → XYZ付与 18 → PnP採用 10',data['stages'])


if __name__ == '__main__':
    unittest.main()
