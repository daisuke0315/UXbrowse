"""Foreground masks belong to the live view, not CT intensity/depth assignment."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_controls import render_snapshot, render_layer
from browser_live import BrowserApplication
from test_browser_live import renderer, result


class ForegroundMaskTests(unittest.TestCase):
    def setUp(self):
        self.app = BrowserApplication()
        self.app.renderer = renderer()
        self.app.manifest = dict(bounds=[[-20.,-20.,-20.],[20.,20.,20.]])
        self.app.args = SimpleNamespace(ct_validation=False)
        self.K = np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        self.image = np.full((100,100,3),200,np.uint8)

    def tearDown(self):
        self.app.executor.shutdown(wait=True)
        self.app.layer_executor.shutdown(wait=True)

    def publish(self, status='PASS'):
        self.app.publish(self.image,result(status),self.K,np.zeros(5),time.monotonic(),self.app.sequence+1,'test',force=True)
        size=struct.unpack('<I',self.app.packet[:4])[0]
        return json.loads(self.app.packet[4:4+size])

    def test_actual_plane_export_and_no_air_mask_png(self):
        metadata=self.publish()
        np.testing.assert_allclose(metadata['mpr_plane']['center'],[0,0,0])
        np.testing.assert_allclose(metadata['mpr_plane']['normal'],[0,0,1])
        self.assertEqual(metadata['model_bounds'],self.app.manifest['bounds'])
        self.assertEqual(metadata['mask_bytes'],0)
        np.testing.assert_array_equal(self.image,np.full_like(self.image,200))

    def test_failed_pose_never_exports_stale_plane(self):
        self.publish()
        self.assertIsNone(self.publish('FAIL')['mpr_plane'])

    def test_air_ct_samples_do_not_make_a_gray_mask(self):
        self.app.renderer._sample_plane=lambda *args:(np.zeros((21,21),np.uint8),np.zeros((21,21),np.float32))
        metadata=self.publish()
        self.assertEqual(metadata['mpr_status'],'VOLUME_SAMPLE_EMPTY')
        self.assertEqual(metadata['mask_bytes'],0)
        self.assertIsNotNone(metadata['mpr_plane'])

    def test_mask_toggle_does_not_change_ct_alpha_or_original_image(self):
        render=lambda:render_layer(*render_snapshot(self.app.renderer,self.image,self.K,np.zeros(5),result()))[0]
        before=render()
        self.app.controls(dict(action='slice_mask',value=False))
        self.assertEqual(before,render())
        self.assertEqual(self.app.opacity,82.)
        self.assertEqual(self.app.renderer.opacity,.82)
        self.assertFalse(self.publish()['slice_mask_enabled'])

    def test_registered_live_pass_has_mask_plane_without_freeze(self):
        self.app.args.ct_validation=True
        self.assertIsNotNone(self.publish()['mpr_plane'])

    def test_actual_plane_tracks_camera_distance_and_rotation(self):
        from dataclasses import replace
        pose=replace(result(),rvec=(0.,np.pi/4,0.))
        self.app.publish(self.image,pose,self.K,np.zeros(5),time.monotonic(),1,'test',force=True)
        metadata=self.app.last_published[1]
        n=np.array(metadata['mpr_plane']['normal'])
        np.testing.assert_allclose(n,cv2.Rodrigues(np.asarray(pose.rvec))[0].T[:,2])
        eye=np.array(metadata['pose']['eye'])
        self.assertAlmostEqual(np.dot(np.array(metadata['mpr_plane']['center'])-eye,n),metadata['slice_position']['camera_depth'])

    def test_box_projection_front_vs_back_matches_full_camera_intrinsics(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/foreground-mask.js'
        script="""
        import {readFileSync} from 'node:fs';
        const text=readFileSync(process.argv[1],'utf8');
        const {projectForegroundBox:project}=await import('data:text/javascript;base64,'+Buffer.from(text).toString('base64'));
        const pose={eye:[0,0,-100],target:[0,0,-99],up:[0,-1,0],K:[[100,0,50],[0,100,50],[0,0,1]],width:100,height:100},bounds=[[-10,-10,-10],[10,10,10]];
        const plane=z=>({center:[0,0,z],normal:[0,0,2]});
        const mid=project(pose,plane(0),bounds);
        console.log(JSON.stringify({before:project(pose,plane(-20),bounds),touch:project(pose,plane(-10),bounds),mid,far:project(pose,plane(20),bounds),behindCamera:project({...pose,eye:[0,0,100],target:[0,0,101]},plane(0),bounds),inside:project({...pose,eye:[0,0,0],target:[0,0,1]},plane(5),bounds),invalid:project(pose,plane(0),[[0,0,0],[0,1,1]])}));
        """
        data=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(data['before'],[])
        self.assertEqual(data['touch'],[])
        self.assertEqual(data['behindCamera'],[])
        self.assertEqual(data['invalid'],[])
        self.assertEqual(data['mid'],data['far'])
        np.testing.assert_allclose(np.min(data['mid'],axis=0),[50-100/9,50-100/9])
        np.testing.assert_allclose(np.max(data['mid'],axis=0),[50+100/9,50+100/9])
        np.testing.assert_allclose(np.min(data['inside'],axis=0),[0,0])
        np.testing.assert_allclose(np.max(data['inside'],axis=0),[100,100])

    def test_rotated_box_projection_agrees_with_yellow_opencv_corners(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/foreground-mask.js'
        bounds=[[-40,-20,-30],[40,20,30]]
        corners=np.array([[x,y,z] for x in (-40,40) for y in (-20,20) for z in (-30,30)],np.float64)
        K=np.array([[150.,0,99.5],[0,170.,79.5],[0,0,1.]])
        cases=[];expected=[]
        for rvec in ([0,0,0],[.3,.7,.1],[-.6,.4,.2],[.2,-.5,-.9]):
            R=cv2.Rodrigues(np.asarray(rvec,np.float64))[0]
            forward=R.T[:,2];eye=-forward*300;tvec=-R@eye
            cases.append(dict(pose=dict(eye=eye.tolist(),target=(eye+forward).tolist(),up=(-R.T[:,1]).tolist(),K=K.tolist(),width=200,height=160),
                              plane=dict(center=(forward*100).tolist(),normal=forward.tolist()),bounds=bounds))
            expected.append(cv2.projectPoints(corners,np.asarray(rvec,np.float64),tvec,K,np.zeros(5))[0].reshape(-1,2))
        script="""
        import {readFileSync} from 'node:fs';
        const {projectForegroundBox:project}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        console.log(JSON.stringify(JSON.parse(process.argv[2]).map(c=>project(c.pose,c.plane,c.bounds))));
        """
        projected=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source),json.dumps(cases)],text=True))
        for polygon,pixels in zip(projected,expected):
            np.testing.assert_allclose(np.min(polygon,axis=0),pixels.min(axis=0),atol=1e-7)
            np.testing.assert_allclose(np.max(polygon,axis=0),pixels.max(axis=0),atol=1e-7)

    def test_oblique_view_mask_grows_as_plane_moves_through_box(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/foreground-mask.js'
        script="""
        import {readFileSync} from 'node:fs';
        const {projectForegroundBox:project}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        const pose={eye:[-100,0,-100],target:[-99,0,-99],up:[0,-1,0],K:[[100,0,50],[0,100,50],[0,0,1]],width:100,height:100},bounds=[[-10,-10,-10],[10,10,10]];
        const area=p=>Math.abs(p.reduce((s,v,i)=>{const q=p[(i+1)%p.length];return s+v[0]*q[1]-v[1]*q[0];},0))*.5;
        console.log(JSON.stringify([-11,-9,-4,0,11].map(c=>area(project(pose,{center:[c,0,c],normal:[1,0,1]},bounds)))));
        """
        areas=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(areas[0],0)
        self.assertTrue(all(a<=b for a,b in zip(areas,areas[1:])),areas)
        self.assertLess(areas[1],areas[3])


if __name__=='__main__':unittest.main()
