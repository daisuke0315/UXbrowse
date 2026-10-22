import io
import json
import struct
import subprocess
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import cv2
import numpy as np
from PIL import Image

from browser_live import BrowserApplication,DEFAULTS,runtime_arguments
from browser_photo import decode_photo,photo_engine,localize_photo
from browser_controls import render_snapshot,render_layer
from test_browser_live import renderer,result


from tests.runtime_dataset import RuntimeDataset


class BrowserPhotoTests(RuntimeDataset, unittest.TestCase):
    def test_measurement_zoom_reads_only_visible_source_pixels(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/measurement-workspace.js'
        script="""
        import {readFileSync} from 'node:fs';
        const {ImageViewTransform,visibleImageRect}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        const view=new ImageViewTransform();view.reset(4284,5712);view.width=700;view.height=700;
        view.zoom=8;view.center=[2142,2856];
        const whole=visibleImageRect(view),ct=visibleImageRect(view,[1900,2600,400,500]);
        console.log(JSON.stringify({whole,ct}));
        """
        values=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertLess(values['whole'][2]*values['whole'][3],4284*5712/20)
        self.assertLessEqual(values['ct'][2]*values['ct'][3],400*500)

    def test_cropped_ct_layer_reconstructs_the_same_full_resolution_pixels(self):
        image=np.zeros((100,100,3),np.uint8)
        K=np.array([[100.,0,50.],[0,100.,50.],[0,0,1.]])
        source=renderer()
        full_png,_,_,full_snapshot=render_layer(*render_snapshot(source,image,K,np.zeros(5),result()))
        crop_png,status,_,snapshot=render_layer(*render_snapshot(
            source,image,K,np.zeros(5),result(),crop_layer=True))
        full=cv2.imdecode(np.frombuffer(full_png,np.uint8),cv2.IMREAD_UNCHANGED)
        crop=cv2.imdecode(np.frombuffer(crop_png,np.uint8),cv2.IMREAD_UNCHANGED)
        x,y,w,h=snapshot._last_layer_rect
        restored=np.zeros_like(full)
        restored[y:y+h,x:x+w]=crop
        self.assertEqual(status,'VISIBLE')
        self.assertLess(w*h,image.shape[0]*image.shape[1])
        np.testing.assert_array_equal(restored,full)
        # Picking still uses full-photo UV, not the cropped layer's origin.
        projected=np.array([50.,50.,1.])
        np.testing.assert_allclose(snapshot._last_inverse_homography@projected,
                                   full_snapshot._last_inverse_homography@projected)

    def test_photo_frame_reuses_static_image_payload(self):
        app=BrowserApplication()
        app.settings=dict(DEFAULTS,input_source='photo')
        app.args=SimpleNamespace(ct_validation=False)
        image=np.full((40,60,3),90,np.uint8)
        K=np.array([[100.,0,30.],[0,100.,20.],[0,0,1.]])
        try:
            app.publish(image,result(),K,np.zeros(5),time.monotonic(),1,'test',force=True)
            header_size=struct.unpack('<I',app.packet[:4])[0]
            first=json.loads(app.packet[4:4+header_size])
            self.assertEqual(first['jpeg_bytes'],0)
            self.assertEqual(first['photo_image_revision'],1)
            self.assertGreater(len(app.photo_encoded),0)
            app.publish(image,result(),K,np.zeros(5),time.monotonic(),2,'test',force=True)
            header_size=struct.unpack('<I',app.packet[:4])[0]
            second=json.loads(app.packet[4:4+header_size])
            self.assertEqual(second['jpeg_bytes'],0)
            self.assertEqual(second['photo_image_revision'],1)
        finally:
            app.executor.shutdown(wait=True)
            app.layer_executor.shutdown(wait=True)

    def test_decode_keeps_resolution_and_applies_exif_rotation(self):
        source=Image.new('RGB',(120,80),'red');exif=source.getexif();exif[274]=6
        stream=io.BytesIO();source.save(stream,format='JPEG',exif=exif)
        self.assertEqual(decode_photo(stream.getvalue()).shape,(120,80,3))
        stream=io.BytesIO();Image.new('RGB',(2301,1703),'blue').save(stream,format='PNG')
        decoded=decode_photo(stream.getvalue())
        self.assertEqual(decoded.shape,(1703,2301,3))
        np.testing.assert_array_equal(decoded[100,100],[255,0,0])
        stream=io.BytesIO();Image.new('RGB',(120,80),'red').save(stream,format='MPO',save_all=True,append_images=[Image.new('RGB',(120,80),'blue')])
        primary=decode_photo(stream.getvalue())
        self.assertEqual(primary.shape,(80,120,3))
        self.assertGreater(int(primary[20,20,2]),240)

    def test_invalid_and_oversized_photos_are_rejected(self):
        with self.assertRaises(ValueError):decode_photo(b'not an image')
        stream=io.BytesIO();Image.new('RGB',(12,12)).save(stream,format='GIF')
        with self.assertRaises(ValueError):decode_photo(stream.getvalue())
        stream=io.BytesIO();Image.new('RGB',(20,20)).save(stream,format='PNG')
        with patch('browser_photo.MAX_PHOTO_PIXELS',100):
            with self.assertRaises(ValueError):decode_photo(stream.getvalue())

    def test_photo_query_resolution_does_not_change_reference_resolution(self):
        engine=SimpleNamespace(resize=1024,retriever=SimpleNamespace())
        self.assertIs(photo_engine(SimpleNamespace(global_localizer=engine),2048),engine)
        self.assertEqual(engine.resize,1024)
        self.assertEqual(engine.query_resize,2048)
        self.assertEqual(engine.retriever.query_max_width,1280)
        self.assertIs(photo_engine(SimpleNamespace(localizer=engine),1600),engine)

    def test_photo_defaults_use_unified_analysis_without_relaxing_quality(self):
        self.assertEqual(DEFAULTS['photo_feature_size'],1024)
        for mode in ('gs','registered'):
            args,_,_=runtime_arguments(dict(DEFAULTS,input_source='photo',mode=mode))
            self.assertEqual(args.glb_resize,1024)
            self.assertEqual(args.glb_max_keypoints,2000)
            self.assertEqual(args.ransac_threshold,4.)
            self.assertEqual(args.glb_max_reprojection_error,3.)

    def test_failed_photo_expands_views_but_keeps_live_budget(self):
        attempted=[]
        engine=SimpleNamespace(top_k=3,source_type='3dgs',views=list(range(72)))
        def localize(*_):
            attempted.append(engine.top_k)
            return SimpleNamespace(status='PASS' if engine.top_k==10 else 'FAIL',
                                   inliers=14 if engine.top_k==10 else 7),None
        engine.localize=localize
        matched,_=localize_photo(engine,None,None,None)
        self.assertEqual(matched.status,'PASS')
        self.assertEqual(attempted,[3,10])
        self.assertEqual(engine.top_k,3)
        self.assertEqual(engine.photo_view_expansion['initial_inliers'],7)
        attempted.clear();engine.top_k=10
        localize_photo(engine,None,None,None)
        self.assertEqual(attempted,[10])
        attempted.clear();engine.top_k=3;engine.source_type='registered'
        localize_photo(engine,None,None,None)
        self.assertEqual(attempted,[3])

    def test_mps_photo_releases_unused_memory_once_before_candidate_expansion(self):
        events=[]
        mps=SimpleNamespace(synchronize=Mock(side_effect=lambda:events.append('sync')),
                            empty_cache=Mock(side_effect=lambda:events.append('release')))
        engine=SimpleNamespace(device=SimpleNamespace(type='mps'),
                               torch=SimpleNamespace(mps=mps),top_k=3,
                               source_type='3dgs',views=list(range(72)))
        image=np.zeros((40,60,3),np.uint8);matrix=np.eye(3);distortion=np.zeros(5)
        final=SimpleNamespace(status='PASS',inliers=14);canvas=object()
        def localize(frame,K,dist):
            self.assertIs(frame,image);self.assertIs(K,matrix);self.assertIs(dist,distortion)
            events.append(('match',engine.top_k))
            return (final if engine.top_k==10 else SimpleNamespace(status='FAIL',inliers=7)),canvas
        engine.localize=localize
        matched,drawn=localize_photo(engine,image,matrix,distortion)
        self.assertIs(matched,final);self.assertIs(drawn,canvas)
        self.assertEqual(events,['sync','release',('match',3),('match',10)])
        mps.synchronize.assert_called_once_with();mps.empty_cache.assert_called_once_with()
        self.assertEqual(engine.top_k,3)

    def test_cpu_photo_does_not_use_mps_memory_operations(self):
        mps=SimpleNamespace(synchronize=Mock(),empty_cache=Mock())
        final=SimpleNamespace(status='PASS',inliers=14)
        engine=SimpleNamespace(device=SimpleNamespace(type='cpu'),torch=SimpleNamespace(mps=mps),
                               localize=Mock(return_value=(final,None)))
        self.assertIs(localize_photo(engine,None,None,None)[0],final)
        mps.synchronize.assert_not_called();mps.empty_cache.assert_not_called()

    def test_photo_runs_once_without_camera_and_preserves_full_resolution(self):
        app=BrowserApplication()
        image=np.full((1248,1664,3),80,np.uint8);app.photo_upload=(image,'test.png')
        events=[]
        mps=SimpleNamespace(synchronize=Mock(side_effect=lambda:events.append('sync')),
                            empty_cache=Mock(side_effect=lambda:events.append('release')))
        def localize(*_):
            events.append('match')
            return result(),None
        R=renderer();engine=SimpleNamespace(localize=Mock(side_effect=localize),last_debug=None,
                                           retriever=SimpleNamespace(),
                                           device=SimpleNamespace(type='mps'),torch=SimpleNamespace(mps=mps))
        wrapper=SimpleNamespace(global_localizer=engine,mpr_renderer=R,close=Mock())
        args=SimpleNamespace(ct_validation=False,matching_mode='splat-track')
        K=np.array([[1000.,0,832.],[0,1000.,624.],[0,0,1.]])
        manifest={'bounds':[[-20,-20,-20],[20,20,20]],'views':[]}
        try:
            with patch('browser_live.runtime_arguments',return_value=(args,manifest,Path(__file__))),patch('browser_live.build_localizer',return_value=wrapper),patch('browser_live.open_camera') as camera,patch('browser_live.camera_parameters',return_value=(K,np.zeros(5),'test')):
                app.start(dict(DEFAULTS,input_source='photo'))
                for _ in range(10):
                    app.tick()
                    if app.session is not None:break
                self.assertIsNotNone(app.session)
                self.assertEqual(app.session.frame.shape,image.shape)
                np.testing.assert_array_equal(app.session.matrix,K)
                self.assertEqual(engine.localize.call_count,1)
                self.assertEqual(events,['sync','release','match'])
                self.assertEqual(engine.localize.call_args.args[0].shape,image.shape)
                self.assertEqual(app.capture,None);camera.assert_not_called()
                # Old capture timestamps are fine for an immutable photo.
                values=list(app.last_valid);values[4]=time.monotonic()-3600;app.last_valid=tuple(values)
                app.controls({'action':'resume'});app.controls({'action':'freeze'})
                for _ in range(3):app.tick()
                self.assertEqual(engine.localize.call_count,1)
                app.publish(*app.frozen_source,force=True)
                metadata=app.last_published[1]
                self.assertEqual(metadata['input_source'],'photo')
                self.assertEqual(metadata['image_mime'],'image/png')
                self.assertEqual((metadata['width'],metadata['height']),(1664,1248))
                self.assertEqual(metadata['photo_feature_size'],1024)
                self.assertEqual(metadata['photo_retrieval_max_width'],1280)
                self.assertFalse(metadata['demo'])
                app.controls({'action':'relocalize'});app.tick()
                if app.future:app.future.result()
                app.tick()
                self.assertEqual(engine.localize.call_count,2)
                self.assertEqual(events,['sync','release','match']*2)
        finally:
            app.stop();app.executor.shutdown(wait=True);app.layer_executor.shutdown(wait=True)

    def test_photo_requires_uploaded_image(self):
        app=BrowserApplication()
        try:
            with patch('browser_live.runtime_arguments',return_value=(SimpleNamespace(),{},Path(__file__))),patch('browser_live.open_camera') as camera:
                with self.assertRaisesRegex(ValueError,'写真'):app.start(dict(DEFAULTS,input_source='photo'))
                camera.assert_not_called()
        finally:app.stop();app.executor.shutdown();app.layer_executor.shutdown()


if __name__=='__main__':unittest.main()
