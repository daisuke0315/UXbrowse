import concurrent.futures
import io
import json
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_live import (BrowserApplication, encode_packet, finite_json, pose_camera,
                          runtime_arguments, make_handler, correspondence_debug, DEFAULTS)
from realtime_pnp_server import LiveResult, MPRRenderer, LocalizationDebug


def result(status='PASS'):
    return LiveResult(status, 'test', 20, 20, 15, .75, 1.,
                      (0., 0., 0.), (0., 0., 100.), (0., 0., -100.), 5., 'HIDDEN')


def renderer():
    value = MPRRenderer.__new__(MPRRenderer)
    value._control_lock = threading.Lock()
    value.slice_fraction = .5
    value.opacity = .82
    value.zoom = 1.
    value._zoom_center = None
    value._measurement_points = []
    value.volume_corners = np.array([(x, y, z) for x in (-20, 20) for y in (-20, 20) for z in (-20, 20)])
    value.volume_center = np.zeros(3)
    value.texture_size = 21
    value.field_of_view = 40.
    value.unit_label = 'mm'
    value.volume = SimpleNamespace(origin=np.array([-20.,-20.,-20.]),spacing=np.ones(3),
                                   dimensions=(41,41,41),inverse_direction=np.eye(3))
    value._sample_plane = lambda *args: (np.full((21,21),120,np.uint8), np.full((21,21),.82,np.float32))
    return value


from tests.runtime_dataset import RuntimeDataset


class BrowserTests(RuntimeDataset, unittest.TestCase):
    def setUp(self):
        self.app = BrowserApplication()
        self.K = np.array([[100.,0,50.],[0,100.,50.],[0,0,1.]])

    def tearDown(self):
        self.app.executor.shutdown(wait=True, cancel_futures=True)

    def test_pose_uses_opencv_camera_convention(self):
        camera = pose_camera(result(), self.K, 100, 100)
        np.testing.assert_allclose(camera['eye'], [0,0,-100])
        np.testing.assert_allclose(camera['target'], [0,0,-99])
        np.testing.assert_allclose(camera['up'], [0,-1,0])
        self.assertIsNone(pose_camera(result('FAIL'), self.K, 100, 100))

    def test_nonfinite_pose_is_not_rendered(self):
        bad = result().__dict__.copy()
        bad['tvec'] = (float('nan'),0.,1.)
        self.assertIsNone(pose_camera(LiveResult(**bad), self.K, 100, 100))

    def test_stop_releases_camera_before_waiting_for_inference(self):
        events=[]
        self.app.capture=SimpleNamespace(release=lambda:events.append('camera_released'))
        self.app.future=concurrent.futures.Future()
        def wait(jobs):
            self.assertIsNone(self.app.capture)
            self.assertIn('camera_released',events)
            events.append('wait')
        with patch('browser_live.concurrent.futures.wait',side_effect=wait):
            self.app.stop()
        self.assertEqual(events[0],'camera_released')
        self.assertIsNone(self.app.args)
        self.assertEqual(self.app.state['status'],'idle')

    def test_same_application_can_start_stop_and_start_with_new_camera(self):
        captures=[Mock(),Mock()]
        image=np.zeros((8,12,3),np.uint8)
        for capture in captures:
            capture.read.return_value=(True,image)
        args=SimpleNamespace(camera=1,ffmpeg='ffmpeg')
        engines=[SimpleNamespace(mpr_renderer=None,close=Mock()) for _ in range(2)]
        with patch('browser_live.runtime_arguments',return_value=(args,{},Path(__file__))), \
             patch('browser_live.open_camera',side_effect=captures) as opened, \
             patch('browser_live.build_localizer',side_effect=engines), \
             patch('browser_live.RuntimeDiagnostics'), patch.object(self.app,'publish'):
            for capture in captures:
                self.app.start(DEFAULTS)
                self.assertIs(self.app.capture,capture)
                self.assertEqual(self.app.state['status'],'running')
                self.app.stop()
                capture.release.assert_called_once()
                self.assertIsNone(self.app.capture)
                self.assertIsNone(self.app.args)
                self.assertEqual(self.app.state['status'],'idle')
        self.assertEqual(opened.call_count,2)

    def test_packet_keeps_image_pose_and_layer_atomic(self):
        packet = encode_packet({'source_id':7,'pose_source_id':7}, b'image', b'layer')
        size = struct.unpack('<I', packet[:4])[0]
        metadata = json.loads(packet[4:4+size])
        self.assertEqual(metadata['source_id'],metadata['pose_source_id'])
        self.assertEqual(metadata['jpeg_bytes'],5)
        self.assertEqual(packet[4+size:],b'imagelayer')

    def test_finite_json_handles_numpy_and_nan(self):
        self.assertEqual(finite_json({'a':np.array([1.,np.nan])}),{'a':[1.,None]})

    def test_layer_has_no_hud_or_display_zoom_and_retains_ct_mapping(self):
        self.app.renderer = renderer()
        layer, status = self.app.renderer.render(np.zeros((100,100,3),np.uint8), self.K,
                                                np.zeros(5), np.zeros((3,1)), np.array([[0.],[0.],[100.]]),layer_only=True)
        self.assertEqual(status,'VISIBLE')
        self.assertEqual(layer.shape,(100,100,4))
        self.assertEqual(layer[50,50,3],209)
        np.testing.assert_allclose(self.app.renderer.display_to_ct_point(50,50)[0],[0,0,0])
        self.assertEqual(layer[0,0,3],0)

    def test_empty_layer_is_transparent(self):
        value=renderer()
        value._sample_plane=lambda *args:(np.zeros((21,21),np.uint8),np.zeros((21,21),np.float32))
        image,status=value.render(np.ones((100,100,3),np.uint8), self.K,np.zeros(5),np.zeros((3,1)),np.zeros((3,1)),layer_only=True)
        self.assertEqual(status,'VOLUME_SAMPLE_EMPTY')
        self.assertEqual(image.shape,(100,100,4))
        self.assertFalse(image.any())

    def prepare_freeze(self):
        self.app.renderer=renderer()
        self.app.args=SimpleNamespace(ct_validation=False)
        self.app.last_valid=(np.zeros((100,100,3),np.uint8),result(),self.K,np.zeros(5),time.monotonic(),12,'test')
        self.app.controls({'action':'freeze'})
        self.app.publish(*self.app.frozen_source,force=True)

    def test_measurement_requires_freeze_and_fresh_plane(self):
        with self.assertRaisesRegex(ValueError,'固定'):
            self.app.controls({'action':'point','x':50,'y':50})
        self.prepare_freeze()
        with self.assertRaisesRegex(ValueError,'更新'):
            self.app.controls({'action':'point','x':50,'y':50,'kind':'ct','plane_revision':-1})

    def test_ct_anchor_stays_world_fixed_when_slice_changes(self):
        self.prepare_freeze()
        self.app.controls({'action':'point','x':50,'y':50,'kind':'ct','plane_revision':self.app.control_revision})
        xyz=self.app.session.ct_xyz.copy()
        self.app.controls({'action':'slice','value':70})
        self.app.publish(*self.app.frozen_source,force=True)
        np.testing.assert_allclose(self.app.session.ct_xyz,xyz)

    def test_frozen_frame_is_not_republished_until_controls_change(self):
        self.prepare_freeze()
        sequence=self.app.sequence
        self.app.publish(*self.app.frozen_source,force=True)
        self.assertEqual(self.app.sequence,sequence)

    def test_distorted_image_is_rectified_before_freeze_measurement(self):
        self.app.renderer=renderer()
        distortion=np.array([.1,0,0,0,0])
        self.app.last_valid=(np.zeros((100,100,3),np.uint8),result(),self.K,distortion,time.monotonic(),1,'calibrated')
        self.app.controls({'action':'freeze'})
        self.assertFalse(self.app.session.distortion.any())
        self.assertFalse(self.app.frozen_source[3].any())

    def test_registered_uses_common_tracker_current_frame_not_old_worker_image(self):
        self.app.args=SimpleNamespace(max_processing_width=100, matching_mode='registered-aliked',process_fps=15)
        frame=np.full((100,100,3),100,np.uint8)
        old=np.full((100,100,3),30,np.uint8)
        self.app.capture=SimpleNamespace(read=lambda:(True,frame))
        localize=Mock(return_value=(result(),frame))
        self.app.localizer=SimpleNamespace(localize=localize,last_debug=None)
        with patch('browser_live.camera_parameters',return_value=(self.K,np.zeros(5),'test')),patch.object(self.app,'publish') as publish:
            self.app.tick()
        self.assertEqual(publish.call_args.args[5],self.app.source_id)
        np.testing.assert_array_equal(publish.call_args.args[0],frame)
        localize.assert_called_once()
        self.assertIsNone(self.app.future)

    def test_slider_bounds(self):
        self.app.controls({'action':'slice','value':-3000})
        self.assertEqual(self.app.slice_percent,-100)
        self.app.controls({'action':'slice','value':2000})
        self.assertEqual(self.app.slice_percent,150)
        with self.assertRaises(ValueError):
            self.app.controls({'action':'opacity','value':float('nan')})

    def test_30hz_tracking_cadence_tolerates_capture_jitter(self):
        self.app.args=SimpleNamespace(max_processing_width=100,process_fps=30)
        frame=np.zeros((100,100,3),np.uint8)
        self.app.capture=SimpleNamespace(read=lambda:(True,frame))
        localize=Mock(return_value=(result('FAIL'),frame))
        self.app.localizer=SimpleNamespace(localize=localize,last_debug=None)
        with patch('browser_live.camera_parameters',return_value=(self.K,np.zeros(5),'test')),patch.object(self.app,'publish'):
            for i in range(120):
                stamp=1000+i/30+(.002 if i%2 else .001)
                with patch('browser_live.time.monotonic',return_value=stamp):
                    self.app.tick()
        self.assertGreaterEqual(localize.call_count,118)
        self.assertEqual(self.app.rate_counts['tracking'],localize.call_count)

    def test_default_preset_uses_enriched_registered_map_and_keeps_ct_bounds_check(self):
        args,manifest,ply=runtime_arguments({})
        self.assertEqual(args.process_fps,30)
        self.assertEqual(DEFAULTS['mode'],'registered')
        self.assertEqual(self.app.settings['mode'],'registered')
        self.assertEqual(args.matching_mode,'registered-aliked')
        self.assertEqual(args.registered_map,Path(DEFAULTS['map_path']))
        self.assertTrue((args.registered_map/'summary.json').is_file())
        self.assertTrue(args.ct_validation)
        self.assertFalse(args.no_mpr)
        self.assertEqual((args.capture_backend,args.width,args.height,args.camera),('opencv',0,0,1))
        self.assertEqual((args.min_inliers,args.ransac_threshold,args.glb_max_reprojection_error),(10,4.,3.))
        self.assertEqual(manifest['source_type'],'3dgs')
        self.assertTrue(ply.is_file())

    def test_photo_default_uses_registered_map_without_forcing_ct_when_disabled(self):
        args,_,_=runtime_arguments({'input_source':'photo','mpr':False})
        self.assertEqual(args.matching_mode,'registered-aliked')
        self.assertFalse(args.ct_validation)
        self.assertTrue(args.no_mpr)

    def test_registered_live_mpr_follows_pose_without_freeze_and_hides_on_fail(self):
        from dataclasses import replace
        self.app.args=SimpleNamespace(ct_validation=True,matching_mode='registered-aliked')
        self.app.renderer=renderer()
        image=np.zeros((100,100,3),np.uint8)
        planes=[]
        for source,distance in enumerate((100.,90.),1):
            self.app.publish(image,replace(result(),tvec=(0.,0.,distance)),self.K,
                             np.zeros(5),time.monotonic(),source,'test',force=True)
            meta=self.app.last_published[1]
            self.assertFalse(meta['frozen'])
            self.assertEqual(meta['mpr_status'],'VISIBLE')
            self.assertIsNotNone(meta['mpr_plane'])
            self.assertEqual(meta['source_id'],meta['pose_source_id'])
            size=struct.unpack('<I',self.app.packet[:4])[0]
            self.assertGreater(json.loads(self.app.packet[4:4+size])['layer_bytes'],0)
            planes.append(meta['mpr_plane']['center'][2])
        np.testing.assert_allclose(planes,[0.,10.])
        self.app.publish(image,result('FAIL'),self.K,np.zeros(5),time.monotonic(),3,'test',force=True)
        meta=self.app.last_published[1]
        self.assertIsNone(meta['pose'])
        self.assertIsNone(meta['mpr_plane'])
        self.assertEqual(meta['mpr_status'],'HIDDEN')
        size=struct.unpack('<I',self.app.packet[:4])[0]
        self.assertEqual(json.loads(self.app.packet[4:4+size])['layer_bytes'],0)

    def test_registered_frozen_measurement_still_available_after_live_overlay(self):
        self.app.renderer=renderer()
        self.app.args=SimpleNamespace(ct_validation=True,matching_mode='registered-aliked')
        self.app.last_valid=(np.zeros((100,100,3),np.uint8),result(),self.K,
                             np.zeros(5),time.monotonic(),1,'test')
        self.app.publish(*self.app.last_valid,force=True)
        self.app.controls(dict(action='freeze'))
        self.app.publish(*self.app.frozen_source,force=True)
        self.assertTrue(self.app.last_published[1]['frozen'])
        self.assertEqual(self.app.last_published[1]['mpr_status'],'VISIBLE')
        self.app.controls(dict(action='resume'))
        self.assertIsNone(self.app.session)

    def test_http_config_initially_selects_registered_enriched_map(self):
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Host':'127.0.0.1:123'}
        handler.server=SimpleNamespace(server_port=123)
        handler.path='/api/config'
        with patch.object(handler,'respond') as response:
            handler.do_GET()
        self.assertEqual(response.call_args.args[0],200)
        config=json.loads(response.call_args.args[1])
        self.assertEqual(config['defaults']['mode'],'registered')
        self.assertEqual(config['defaults']['map_path'],DEFAULTS['map_path'])

    def test_config_offers_original_and_alpha15_with_original_default(self):
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Host':'127.0.0.1:123'}
        handler.server=SimpleNamespace(server_port=123)
        handler.path='/api/config'
        with patch.object(handler,'respond') as response:
            handler.do_GET()
        config=json.loads(response.call_args.args[1])
        presets={p['id']:p for p in config['reference_presets']}
        self.assertEqual(set(presets),{'original','alpha15'})
        self.assertEqual(config['defaults']['views_path'],presets['original']['path'])
        self.assertEqual(Path(presets['original']['path']).name,'output_3dgs_reference_views_72_upper_negz_v2')
        for preset in presets.values():
            self.assertEqual(preset['available'],(Path(preset['path'])/'manifest.json').is_file())
        for mode in ('registered','gs'):
            baseline,_,_=runtime_arguments(dict(mode=mode))
            selected,manifest,_=runtime_arguments(dict(mode=mode,views_path=presets['alpha15']['path']))
            self.assertEqual(selected.splat_views,Path(presets['alpha15']['path']))
            self.assertEqual(manifest['rgb_alpha_gain'],1.5)
            for name in ('camera','capture_backend','ransac_threshold','min_inliers',
                         'min_inlier_ratio','glb_max_reprojection_error','registered_map'):
                self.assertEqual(getattr(baseline,name),getattr(selected,name))

    def test_browser_mode_switch_shows_only_applicable_matching_options(self):
        root=Path(__file__).resolve().parents[1]
        html=(root/'browser_ui/index.html').read_text()
        select=re.search(r'<select id="mode">(.*?)</select>',html).group(1)
        self.assertTrue(select.startswith('<option value="registered" selected>'))
        self.assertIn('<option value="gs">',select)
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():
            self.skipTest('Node unavailable')
        script=r"""
        import {readFileSync} from 'node:fs';
        const source=readFileSync(process.argv[1],'utf8');
        const body=source.match(/function matchingUI\(\)\{([\s\S]*?)\n\}/)[1];
        const elements={mode:{value:'registered'},gsMatchingOptions:{hidden:false},registeredSearchOptions:{hidden:false},matchingInfo:{textContent:''}};
        const update=new Function('$',body),$=id=>elements[id];
        update($);const registered={hidden:elements.gsMatchingOptions.hidden,info:elements.matchingInfo.textContent};
        elements.mode.value='gs';update($);
        console.log(JSON.stringify({registered,gsHidden:elements.gsMatchingOptions.hidden}));
        """
        data=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,
                                                 str(root/'browser_ui/app.js')],text=True))
        self.assertFalse(data['registered']['hidden'])
        self.assertIn('多視点三角測量',data['registered']['info'])
        self.assertIn('固定して測定',data['registered']['info'])
        self.assertFalse(data['gsHidden'])

    def test_gs_comparison_preset_preserves_camera_and_guided_matching(self):
        args,manifest,ply=runtime_arguments({'mode':'gs'})
        self.assertEqual(args.capture_backend,'opencv')
        self.assertEqual(args.width,0)
        self.assertEqual(args.retrieval_mode,'dino-patch')
        self.assertEqual(args.tracking_replenish_mode,'background')
        self.assertEqual(args.tracking_reseed_frames,90)
        self.assertEqual(manifest['up_axis'],'negative-z')
        self.assertTrue(ply.is_file())

    def test_registered_and_gs_share_retrieval_tracking_and_quality_settings(self):
        for profile in ('standard', 'precision', 'fast'):
            registered,_,_=runtime_arguments({'mode':'registered','quality_profile':profile})
            gs,_,_=runtime_arguments({'mode':'gs','quality_profile':profile})
            for name in ('retrieval_mode','retrieval_auto_roi','glb_max_keypoints',
                         'glb_resize','glb_match_threshold','fundamental_threshold',
                         'ransac_threshold','min_inliers','min_inlier_ratio',
                         'glb_max_reprojection_error','tracking_reseed_frames',
                         'tracking_replenish_mode','tracking_min_points',
                         'tracking_replenish_frames','virtual_max_seed_age'):
                self.assertEqual(getattr(registered,name),getattr(gs,name),(profile,name))
            self.assertEqual(registered.glb_max_keypoints,2000)
            self.assertEqual(registered.virtual_max_seed_age,10.)

    def test_invalid_transform_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json'
            path.write_text(json.dumps({'runtime_transform':'flip-z'}))
            with self.assertRaisesRegex(ValueError,'identity'):
                runtime_arguments({'views_path':folder})

    def test_http_post_requires_token(self):
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Origin':'http://evil.test','Host':'127.0.0.1:1','Content-Length':'2'}
        handler.rfile=io.BytesIO(b'{}')
        with patch.object(handler,'respond') as response:
            handler.do_POST()
        self.assertEqual(response.call_args.args[0],403)
        self.assertTrue(self.app.commands.empty())

    def test_browser_serves_every_local_javascript_import(self):
        root=Path(__file__).resolve().parents[1]/'browser_ui'
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Host':'127.0.0.1:123'}
        handler.server=SimpleNamespace(server_port=123)
        imports=set()
        for source in root.glob('*.js'):
            imports.update(re.findall(r"(?:from\s*|import\s*)['\"]\./([^'\"]+\.js)['\"]", source.read_text()))
        self.assertIn('validation-diagnostics.js',imports)
        for filename in imports:
            with self.subTest(filename=filename):
                handler.path='/'+filename
                with patch.object(handler,'respond') as response:
                    handler.do_GET()
                self.assertEqual(response.call_args.args[0],200)
                self.assertEqual(response.call_args.args[2],'text/javascript')

    def test_browser_projection_matches_opencv_pixels_including_half_pixel(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.is_file():
            self.skipTest('Node unavailable for JS projection test')
        source=Path(__file__).resolve().parents[1]/'browser_ui/projection.js'
        script="""
          import {readFileSync} from 'node:fs';
          const {intrinsicProjection}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
          console.log(JSON.stringify(intrinsicProjection([[620,0,300],[0,710,210],[0,0,1]],800,600)));
        """
        output=subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True)
        matrix=np.asarray(json.loads(output)).reshape(4,4,order='F')
        points=np.array([[10.,15.,100.],[-20.,-12.,100.],[0.,0.,100.]])
        gl_points=points*np.array([1.,-1.,-1.])
        clip=(matrix@np.column_stack((gl_points,np.ones(3))).T).T
        ndc=clip[:,:2]/clip[:,3,None]
        uv=np.column_stack(((ndc[:,0]+1)*400-.5,(1-ndc[:,1])*300-.5))
        expected=np.column_stack((620*points[:,0]/points[:,2]+300,710*points[:,1]/points[:,2]+210))
        np.testing.assert_allclose(uv,expected,atol=1e-8)

    def test_resume_discards_inflight_registered_result(self):
        self.prepare_freeze()
        self.app.future=concurrent.futures.Future()
        self.app.controls({'action':'resume'})
        self.assertTrue(self.app.discard_future)
        self.assertIsNone(self.app.current)

    def test_distance_points_must_share_the_same_ct_plane(self):
        self.prepare_freeze()
        self.app.controls({'action':'point','x':50,'y':50,'kind':'distance','plane_revision':self.app.control_revision})
        self.app.controls({'action':'slice','value':70})
        self.app.publish(*self.app.frozen_source,force=True)
        with self.assertRaisesRegex(ValueError,'同じCT断面'):
            self.app.controls({'action':'point','x':51,'y':50,'kind':'distance','plane_revision':self.app.control_revision})

    def test_precision_preserves_camera_and_reduces_live_mpr_work(self):
        args,_,_=runtime_arguments({'mode':'gs','quality_profile':'precision'})
        standard,_,_=runtime_arguments({'mode':'gs','quality_profile':'standard'})
        self.assertEqual((args.mpr_size,args.ransac_threshold,args.glb_max_reprojection_error),(192,4.,3.))
        self.assertEqual(args.tracking_reseed_frames,90)
        self.assertEqual(standard.ransac_threshold,8.)
        self.assertEqual(standard.tracking_reseed_frames,1800)
        self.assertEqual(standard.tracking_replenish_frames,60)
        self.assertEqual(args.min_inliers,10)

    def test_default_precision_thresholds_and_fast_keeps_qc_and_top_k(self):
        normal,_,_=runtime_arguments({'mode':'gs'})
        fast,_,_=runtime_arguments({'mode':'gs','quality_profile':'fast'})
        self.assertEqual((normal.ransac_threshold,normal.glb_max_reprojection_error,normal.tracking_reseed_frames),(4.,3.,90))
        self.assertEqual((fast.glb_max_keypoints,fast.glb_resize),(2000,768))
        for profile in ('standard','fast','precision'):
            selected,_,_=runtime_arguments({'mode':'gs','quality_profile':profile})
            self.assertEqual((selected.glb_max_keypoints,selected.virtual_guided_features),(2000,2000))
        self.assertEqual((fast.glb_top_k,fast.virtual_policy,fast.min_inliers),(3,'validated',10))
        self.assertEqual(fast.ransac_threshold,8.)

    def test_search_preview_requires_exact_generation_and_source_snapshot(self):
        self.app.generation=4
        self.app.search_generation=7
        self.app.search_preview=b'search-jpeg'
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Host':'127.0.0.1:123'}
        handler.server=SimpleNamespace(server_port=123)
        handler.path='/api/search-image?generation=4&preview_id=7'
        with patch.object(handler,'respond') as response:
            handler.do_GET()
        self.assertEqual(response.call_args.args,(200,b'search-jpeg','image/jpeg'))
        handler.path='/api/search-image?generation=4&preview_id=6'
        with patch.object(handler,'respond') as response:
            handler.do_GET()
        self.assertEqual(response.call_args.args[0],404)

    def test_reference_preview_rejects_invalid_index_and_path_escape(self):
        cls=make_handler(self.app)
        handler=cls.__new__(cls)
        handler.headers={'Host':'127.0.0.1:123'}
        handler.server=SimpleNamespace(server_port=123)
        self.app.args=SimpleNamespace(splat_views=Path('/private/tmp'))
        self.app.manifest={'views':[{'image_name':'../../etc/passwd'}]}
        for path in ('/api/reference-image?generation=0&index=-1','/api/reference-image?generation=0&index=0','/api/reference-image?generation=3&index=0'):
            handler.path=path
            with patch.object(handler,'respond') as response:
                handler.do_GET()
            self.assertEqual(response.call_args.args[0],404)

    def test_actual_observation_is_not_replaced_by_reprojection(self):
        xyz=np.array([[0.,0.,0.],[10.,10.,0.]])
        uv=np.array([[53.,50.],[61.,60.]])
        debug=LocalizationDebug(xyz,frozenset([0]),None,True,uv)
        packet=correspondence_debug(debug,result(),self.K,np.zeros(5))
        np.testing.assert_allclose(packet['observed'],uv)
        np.testing.assert_allclose(packet['projected'],[[50.,50.],[60.,60.]])
        self.assertEqual(packet['inliers'],[True,False])
        self.assertTrue(packet['accepted'])
        np.testing.assert_allclose(packet['xyz'],xyz)

    def test_failed_qc_cannot_display_green_accepted_points(self):
        debug=LocalizationDebug(np.zeros((1,3)),frozenset([0]),None,True,np.array([[50.,50.]]))
        packet=correspondence_debug(debug,result('FAIL'),self.K,np.zeros(5))
        self.assertFalse(packet['accepted'])

    def test_feature_coordinates_follow_image_rectification(self):
        uv=np.array([[80.,75.]])
        distortion=np.array([.2,0,0,0,0])
        debug=LocalizationDebug(np.zeros((1,3)),frozenset([0]),None,True,uv)
        packet=correspondence_debug(debug,result(),self.K,distortion)
        expected=cv2.undistortPoints(uv.reshape(-1,1,2),self.K,distortion,P=self.K).reshape(-1,2)
        np.testing.assert_allclose(packet['observed'],expected)

    def test_freeze_uses_high_resolution_only_until_resume(self):
        self.prepare_freeze()
        self.assertEqual(self.app.renderer.texture_size,512)
        self.app.controls({'action':'resume'})
        self.assertEqual(self.app.renderer.texture_size,21)

    def test_manual_retrieval_reset_discards_old_inflight_seed(self):
        from collections import deque
        from glb_virtual_live import AsyncTemporalGLBLocalizer
        localizer=AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
        localizer.tracker=SimpleNamespace(seed=object())
        localizer.search_frames=deque([np.zeros((2,2))])
        localizer.future=concurrent.futures.Future()
        localizer.pending_seed=object()
        localizer.reset_tracking()
        self.assertIsNone(localizer.tracker.seed)
        self.assertIsNone(localizer.pending_seed)
        localizer.future.set_result(('old pose','old image'))
        localizer._consume_completed()
        self.assertIsNone(localizer.future)
        self.assertIsNone(localizer.tracker.seed)

    def test_diagnostic_preserves_exact_display_frame_and_point_metadata(self):
        self.prepare_freeze()
        with tempfile.TemporaryDirectory() as folder,patch('browser_live.DATA_DIR',Path(folder)):
            self.app.controls({'action':'diagnostic'})
            output=list((Path(folder)/'output_browser_diagnostics').iterdir())[0]
            saved=cv2.imread(str(output/'frame.png'))
            np.testing.assert_array_equal(saved,self.app.last_published[0])
            metadata=json.loads((output/'diagnostic.json').read_text())
            self.assertEqual(metadata['source_id'],12)
            self.assertEqual(metadata['pose_source_id'],12)
            self.assertIn('display_K',metadata)

    def test_invalid_quality_controls_are_rejected(self):
        for settings in ({'quality_profile':'unchecked'},{'fov_deg':float('nan')},{'max_processing_width':20000}):
            with self.assertRaises(ValueError):
                runtime_arguments(settings)

    def test_browser_feature_toggles_and_screen_pixel_size(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.is_file():
            self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/feature-overlay.js'
        script="""
          import {readFileSync} from 'node:fs';
          const {drawFeatures,pointCounts}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
          const radii=[],colors=[];
          const ctx={clearRect(){},beginPath(){},arc(x,y,r){radii.push(r)},fill(){colors.push(this.fillStyle)},stroke(){}};
          const canvas={width:100,height:100,getContext:()=>ctx,getBoundingClientRect:()=>({width:50})};
          const data={observed:[[20,20],[30,30]],inliers:[true,false],accepted:true};
          drawFeatures(canvas,data,{pointSize:4,showInliers:true,showOutliers:false});
          drawFeatures(canvas,data,{pointSize:4,showInliers:false,showOutliers:true});
          console.log(JSON.stringify({colors,radii,counts:pointCounts(data),qc:pointCounts({...data,accepted:false})}));
        """
        packet=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(packet['colors'],['#58ff94','#ff6868'])
        self.assertEqual(packet['radii'],[8,8])
        self.assertEqual(packet['counts'],{'total':2,'inliers':1,'rejected':1})
        self.assertEqual(packet['qc']['inliers'],0)

    def test_progress_labels_do_not_claim_search_candidate_is_confirmed_pose(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.is_file():
            self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/localization-progress.js'
        script="""
          import {readFileSync} from 'node:fs';
          const {searchSummary,backgroundSummary,imageQualitySummary}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
          console.log(JSON.stringify({
            search:searchSummary({active:true,stage:'matching',elapsed_ms:1250,candidate_index:2,candidate_total:3,candidates:[{index:12,image_name:'view_0012.png'}]}),
            replenish:searchSummary({active:true,mode:'guided',tracking:true,stage:'pnp'}),
            context:searchSummary({active:true,mode:'guided',tracking:true,stage:'matching',matching_method:'pose_references'}),
            nearby:searchSummary({active:true,mode:'nearby',tracking:false,stage:'pnp'}),
            early:searchSummary({active:false,stage:'done',matching_method:'lightglue',reference_search:{processed:2,total:6,skipped:4,early_exit:true,policy:'adaptive'}}),
            full:searchSummary({active:true,stage:'qc',matching_method:'lightglue',reference_search:{processed:6,total:6,skipped:0,early_exit:false,policy:'exhaustive'}}),
            appearance:searchSummary({active:true,stage:'reference',reference_selection:'gs-real',appearance_cache:'building',appearance_loaded:25,appearance_total:620}),
            replacement:backgroundSummary({background:{state:'cooldown',outcome:'reinitialized',last_added:0,added_total:8,reinitializations:1,failures:0,tracking_ms:5,search_ms:400},refinement:{committed:true,reinitialized:true,method:'common_pool',selected:'fresh',old_points:13,new_points:80}}),
            common:backgroundSummary({background:{state:'cooldown',outcome:'added',last_added:20,added_total:28,failures:0,tracking_ms:5,search_ms:400},refinement:{committed:true,method:'common_pool',selected:'joint',old_points:80,new_points:100}}),
            emerging:backgroundSummary({background:{state:'running',reason:'view_direction_emerging'}}),
            freshWait:imageQualitySummary({image_selection:{waiting:true,max_age_ms:100}}),
            waiting:searchSummary({active:false,tracking:false,background:{state:'cooldown'},stage:'done',candidate_total:3,query_features:2000}),
            trackedIdle:searchSummary({active:false,tracking:true,background:{state:'cooldown'}}),
            qualityWait:searchSummary({active:false,tracking:true,background:{state:'image_wait'}}),
            quality:imageQualitySummary({active:true,photometric_mode:'corrected',image_selection:{age_ms:120,roi_known:true,quality:{sharpness:12,mean:43,contrast:27}}}),
            idle:searchSummary(null),
            background:backgroundSummary({mode:'guided',background:{state:'foreground_busy',reason:'viewpoint_unchanged',outcome:'verified',last_added:0,added_total:8,failures:0,tracking_ms:120,search_ms:400}}),
            noBackground:backgroundSummary(null)}));
        """
        packet=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(packet['search']['elapsed'],'1.3')
        self.assertEqual(packet['search']['step'],2)
        self.assertEqual(packet['search']['reference'],12)
        self.assertIn('候補 2/3',packet['search']['detail'])
        self.assertIn('追跡継続中',packet['replenish']['title'])
        self.assertIn('既知姿勢の実写2枚',packet['context']['detail'])
        self.assertIn('直前の視点付近',packet['nearby']['title'])
        self.assertIn('実写 2/6枚',packet['early']['detail'])
        self.assertIn('早期終了',packet['early']['detail'])
        self.assertIn('省略 4枚',packet['early']['detail'])
        self.assertIn('全候補照合',packet['full']['detail'])
        self.assertIn('実写の見た目',packet['appearance']['detail'])
        self.assertIn('25/620',packet['appearance']['detail'])
        self.assertIn('追跡を再初期化',packet['replacement'])
        self.assertIn('13→80点',packet['replacement'])
        self.assertNotIn('NaN',packet['replacement'])
        self.assertIn('新旧共通の点で姿勢更新',packet['common'])
        self.assertIn('80→100点',packet['common'])
        self.assertNotIn('NaN',packet['common'])
        self.assertIn('新しい面へ先行補充',packet['emerging'])
        self.assertIn('100 ms',packet['freshWait'])
        self.assertFalse(packet['idle']['active'])
        self.assertFalse(packet['idle']['visible'])
        self.assertTrue(packet['waiting']['visible'])
        self.assertTrue(packet['waiting']['waiting'])
        self.assertFalse(packet['waiting']['active'])
        self.assertIn('再探索待機',packet['waiting']['title'])
        self.assertNotIn('候補',packet['waiting']['detail'])
        self.assertNotIn('特徴',packet['waiting']['detail'])
        self.assertTrue(packet['trackedIdle']['visible'])
        self.assertFalse(packet['trackedIdle']['waiting'])
        self.assertTrue(packet['qualityWait']['waiting'])
        self.assertIn('品質待機',packet['qualityWait']['title'])
        self.assertIn('120 ms',packet['quality'])
        self.assertIn('明度補正',packet['quality'])
        self.assertIn('姿勢精度ではありません',packet['quality'])
        self.assertIn('追跡優先で休止',packet['background'])
        self.assertIn('累計追加8点',packet['background'])
        self.assertEqual(packet['noBackground'],'')

    def test_processing_panel_is_below_live_image_not_above(self):
        from html.parser import HTMLParser
        class IDs(HTMLParser):
            def __init__(self): super().__init__(); self.ids=[]
            def handle_starttag(self,tag,attrs):
                self.ids.extend(value for key,value in attrs if key=='id')
        parser=IDs()
        parser.feed((Path(__file__).resolve().parents[1]/'browser_ui/index.html').read_text())
        self.assertLess(parser.ids.index('panels'),parser.ids.index('searchPanel'))
        self.assertLess(parser.ids.index('searchPanel'),parser.ids.index('backgroundStatus'))
        self.assertEqual(len(parser.ids),len(set(parser.ids)))
        css=(Path(__file__).resolve().parents[1]/'browser_ui/styles.css').read_text()
        self.assertIn('#searchPanel{position:fixed;',css)
        self.assertIn('bottom:12px;',css)
        self.assertIn('--search-dock-space',css)


if __name__=='__main__':
    unittest.main()
