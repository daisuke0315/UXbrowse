import concurrent.futures
import json
from pathlib import Path
import queue
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from browser_controls import ControlMailbox, slice_position, render_layer, project_slice_corners
from browser_live import BrowserApplication
from test_browser_live import renderer, result
from types import SimpleNamespace


class ControlsTests(unittest.TestCase):
    def test_pending_slider_values_coalesce_without_crossing_buttons(self):
        mailbox=ControlMailbox(3)
        for value in range(1000):mailbox.put_nowait(('control',dict(action='slice',value=value)))
        mailbox.put_nowait(('control',dict(action='freeze')))
        mailbox.put_nowait(('control',dict(action='slice',value=1)))
        mailbox.put_nowait(('control',dict(action='slice',value=2)))
        with self.assertRaises(queue.Full):mailbox.put_nowait(('stop',{}))
        self.assertEqual(mailbox.get_nowait()[1]['value'],999)
        self.assertEqual(mailbox.get_nowait()[1]['action'],'freeze')
        self.assertEqual(mailbox.get_nowait()[1]['value'],2)
        self.assertTrue(mailbox.empty())

    def test_depth_is_view_direction_distance_not_euclidean_or_object_surface(self):
        model=renderer()
        near=slice_position(model,[0.,0.,0.],[0.,0.,100.],0)
        far=slice_position(model,[0.,0.,0.],[0.,0.,200.],100)
        self.assertEqual(near['span'],40)
        self.assertEqual(near['offset'],0)
        self.assertEqual(near['camera_depth'],80)
        self.assertEqual(far['offset'],40)
        self.assertEqual(far['camera_depth'],220)
        self.assertTrue(slice_position(model,[0.,0.,0.],[0.,0.,100.],150)['outside'])

    def test_camera_approach_advances_slice_and_retreat_reverses_it(self):
        from dataclasses import replace
        positions=[]
        for source, distance in enumerate((100.,90.,110.)):
            pose=replace(result(),tvec=(0.,0.,distance))
            self.app.publish(np.zeros((100,100,3),np.uint8),pose,self.K,np.zeros(5),
                             time.monotonic(),source,'test',force=True)
            positions.append(self.app.renderer._last_validation_plane[0][2])
            self.assertAlmostEqual(self.app.last_published[1]['slice_position']['camera_depth'],100.)
        np.testing.assert_allclose(positions,[0.,10.,-10.])

    def test_manual_mode_keeps_slice_fixed_despite_camera_translation(self):
        from dataclasses import replace
        self.app.controls(dict(action='camera_linked_slice',value=False))
        for source, distance in enumerate((100.,90.,110.)):
            pose=replace(result(),tvec=(0.,0.,distance))
            self.app.publish(np.zeros((100,100,3),np.uint8),pose,self.K,np.zeros(5),
                             time.monotonic(),source,'test',force=True)
            np.testing.assert_allclose(self.app.renderer._last_validation_plane[0],[0.,0.,0.])

    def test_slider_changes_camera_distance_anchor_not_camera_linkage(self):
        self.app.publish(np.zeros((100,100,3),np.uint8),result(),self.K,np.zeros(5),
                         time.monotonic(),1,'test',force=True)
        self.app.controls(dict(action='slice',value=75))
        self.assertAlmostEqual(self.app.camera_slice_distance,110.)
        self.assertTrue(self.app.camera_linked_slice)

    def setUp(self):
        self.app=BrowserApplication()
        self.app.renderer=renderer()
        self.app.args=SimpleNamespace(ct_validation=False)
        self.K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        self.real_executor=self.app.layer_executor
        self.job=concurrent.futures.Future()
        self.app.layer_executor=SimpleNamespace(submit=Mock(return_value=self.job))

    def tearDown(self):
        self.app.executor.shutdown(wait=True,cancel_futures=True)
        self.real_executor.shutdown(wait=True,cancel_futures=True)

    def publish(self,source=10,status='PASS'):
        self.app.last_packet=0
        self.app.publish(np.full((100,100,3),source,np.uint8),result(status),self.K,
                         np.zeros(5),time.monotonic(),source,'test')

    def complete(self):
        args=self.app.layer_executor.submit.call_args.args[1:]
        self.job.set_result(render_layer(*args))

    def test_slow_mpr_never_holds_back_latest_live_image_or_pose(self):
        # Registered mode retains ct_validation for bounds checking, but live
        # packets must use the same asynchronous renderer as GS mode.
        self.app.args=SimpleNamespace(ct_validation=True)
        self.publish()
        self.assertEqual(self.app.last_published[1]['source_id'],10)
        self.assertEqual(self.app.last_published[1]['mpr_status'],'PENDING')
        self.publish(source=11)
        self.app.layer_executor.submit.assert_called_once()
        self.assertEqual(self.app.last_published[1]['pose_source_id'],11)
        self.complete()
        self.publish(source=12)
        image,metadata=self.app.last_published
        self.assertEqual(metadata['source_id'],12)
        self.assertEqual(metadata['pose_source_id'],12)
        self.assertEqual(metadata['mpr_source_id'],10)
        self.assertEqual(metadata['layer_mode'],'projective')
        self.assertEqual(int(image[0,0,0]),12)
        self.assertEqual(metadata['mpr_status'],'VISIBLE')
        self.assertEqual(self.app.layer_executor.submit.call_count,2)

    def test_cached_slice_projects_with_current_pose_and_expires(self):
        from dataclasses import replace
        from unittest.mock import patch
        self.publish(); self.complete(); self.publish(11)
        moved=replace(result(),tvec=(10.,0.,100.))
        self.app.last_packet=0
        self.app.publish(np.zeros((100,100,3),np.uint8),moved,self.K,np.zeros(5),
            time.monotonic(),12,'test')
        meta=self.app.last_published[1]
        quad=np.asarray(meta['mpr_quad'])
        np.testing.assert_allclose(quad[:,0]/quad[:,3],[-.19,.61,.61,-.19],atol=1e-6)
        with patch('browser_live.time.monotonic',return_value=time.monotonic()+2):
            self.app.last_packet=0
            self.app.publish(np.zeros((100,100,3),np.uint8),moved,self.K,np.zeros(5),
                time.monotonic(),13,'test')
        self.assertIsNone(self.app.last_published[1]['mpr_quad'])
        self.assertEqual(self.app.last_published[1]['mpr_status'],'PENDING')

    def test_projection_preserves_pixel_centers_and_rejects_behind_camera(self):
        corners=np.array([[-20.,20.,0.],[20.,20.,0.],[20.,-20.,0.],[-20.,-20.,0.]])
        quad=np.asarray(project_slice_corners(corners,result(),self.K,(100,100)))
        pixels=(quad[:,:2]/quad[:,3:]+1)*50-.5
        pixels[:,1]=99-pixels[:,1]
        import cv2
        expected=cv2.projectPoints(corners,np.zeros(3),np.array([0.,0.,100.]),
                                   self.K,np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_allclose(pixels,expected)
        self.assertIsNone(project_slice_corners(corners-200,result(),self.K,(100,100)))

    def test_live_worker_encodes_native_texture_not_full_camera_image(self):
        self.publish();self.complete()
        png,status,ms,snapshot=self.job.result()
        import cv2
        rgba=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_UNCHANGED)
        self.assertEqual(rgba.shape,(21,21,4))
        self.assertEqual(snapshot._last_layer_corners.shape,(4,3))

    def test_small_capture_jitter_does_not_halve_packet_cadence(self):
        from unittest.mock import patch
        self.app.renderer=None
        self.app.last_packet=0
        frame=np.zeros((100,100,3),np.uint8)
        for i in range(120):
            stamp=1000+i/30+(.002 if i%2 else .001)
            with patch('browser_live.time.monotonic',return_value=stamp):
                self.app.publish(frame,result('FAIL'),self.K,np.zeros(5),stamp,i,'test')
        self.assertGreaterEqual(self.app.sequence,118)

    def test_reset_discards_old_live_plane_without_holding_new_image(self):
        self.publish();self.complete()
        self.app.generation+=1
        self.publish(11)
        meta=self.app.last_published[1]
        self.assertEqual(meta['source_id'],11)
        self.assertIsNone(meta['mpr_source_id'])
        self.assertIsNone(meta['mpr_quad'])

    def test_same_pose_and_controls_do_not_resample_cached_slice(self):
        self.publish();self.complete();self.publish(10)
        self.app.layer_executor.submit.assert_called_once()
        self.assertEqual(self.app.last_published[1]['mpr_status'],'VISIBLE')

    def test_continuous_slice_input_does_not_starve_previews_or_mislabel_plane(self):
        self.publish()
        self.app.controls(dict(action='slice',value=60,request_id=1,interacting=True))
        self.complete();self.publish(11)
        metadata=self.app.last_published[1]
        self.assertEqual(metadata['slice_percent'],50)
        self.assertEqual(metadata['slice_request_id'],0)
        self.assertEqual(metadata['plane_revision'],0)
        self.assertEqual(self.app.control_revision,1)
        # Live projection has no picking inverse; fixed measurement builds one.
        self.assertEqual(self.app.display_plane_revision,-1)

    def test_photo_drops_completed_obsolete_slice_and_starts_latest(self):
        self.app.settings['input_source']='photo'
        self.publish()
        self.app.controls(dict(action='slice',value=60,request_id=1,interacting=True))
        self.complete()
        self.publish(11)
        self.assertEqual(self.app.layer_executor.submit.call_count,2)
        self.assertIsNone(self.app.packet)
        self.assertEqual(self.app.control_revision,1)

    def test_lost_pose_hides_pending_old_mpr_even_if_worker_finishes(self):
        self.publish()
        self.publish(11,'FAIL')
        self.assertIsNone(self.app.last_published[1]['pose'])
        self.complete();self.publish(12,'FAIL')
        self.assertIsNone(self.app.last_published[1]['pose'])
        self.assertEqual(self.app.last_published[1]['source_id'],12)

    def test_opacity_does_not_resample_or_change_plane_revision(self):
        self.app.controls(dict(action='opacity',value=0))
        self.assertEqual(self.app.control_revision,0)
        self.app.layer_executor.submit.assert_not_called()
        self.assertEqual(self.app.renderer.opacity,.82)

    def test_mask_toggle_discards_pending_old_layer(self):
        self.publish()
        self.app.controls(dict(action='slice_mask',value=False))
        self.complete();self.publish(11)
        self.assertEqual(self.app.layer_executor.submit.call_count,2)
        self.assertEqual(self.app.last_published[1]['source_id'],11)
        self.assertEqual(self.app.last_published[1]['mpr_status'],'PENDING')
        self.assertEqual(self.app.render_barrier,1)

    def test_mask_ui_independent_opacity_and_immediate_off(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        source=Path(__file__).resolve().parents[1]/'browser_ui/slice-controls.js'
        if not node.exists():self.skipTest('Node unavailable')
        script="""
        import {readFileSync} from 'node:fs';
        const {installSliceControls}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        const nodes={};globalThis.document={getElementById:id=>nodes[id]??=({style:{},value:'82',checked:true,setAttribute(){}}),querySelectorAll:()=>[]};
        const sent=[];const ui=installSliceControls(async(path,data)=>sent.push(data),()=>{});
        ui.sync({generation:1,opacity:82,slice_mask_enabled:true,slice_mask_available:true});
        nodes.opacity.value='25';nodes.opacity.oninput();
        const alpha={ct:nodes.mprCanvas.style.opacity,mask:nodes.sliceMaskCanvas.style.display};
        nodes.sliceMask.checked=false;nodes.sliceMask.onchange();
        const off=nodes.sliceMaskCanvas.style.display;
        ui.sync({generation:1,slice_mask_enabled:true,slice_mask_available:true});
        const pending=nodes.sliceMask.checked;
        nodes.opacity.value='0';nodes.opacity.oninput();
        console.log(JSON.stringify({alpha,off,pending,zero:nodes.sliceMaskCanvas.style.display,sent}));
        """
        data=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(data['alpha'],dict(ct=.25,mask=''))
        self.assertEqual(data['off'],'none')
        self.assertFalse(data['pending'])
        self.assertEqual(data['zero'],'none')
        self.assertIn(dict(action='slice_mask',value=False,generation=1),data['sent'])

    def test_mpr_visibility_is_independent_of_gs_anatomy_and_global_hide(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        source=Path(__file__).resolve().parents[1]/'browser_ui/slice-controls.js'
        if not node.exists():self.skipTest('Node unavailable')
        script="""
        import {readFileSync} from 'node:fs';
        const {applyMPRVisibility}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        const nodes={showMPR:{checked:true},mprCanvas:{style:{}},sliceMaskCanvas:{style:{}},gsOverlay:{style:{visibility:'visible'}},anatomyCanvas:{style:{visibility:'visible'}}};
        const $=id=>nodes[id],states=[];
        function snapshot(allHidden){const visible=applyMPRVisibility($,allHidden);states.push({visible,ct:nodes.mprCanvas.style.visibility,mask:nodes.sliceMaskCanvas.style.visibility,gs:nodes.gsOverlay.style.visibility,anatomy:nodes.anatomyCanvas.style.visibility});}
        snapshot(false);nodes.showMPR.checked=false;snapshot(false);
        snapshot(true);snapshot(false);nodes.showMPR.checked=true;snapshot(true);snapshot(false);
        console.log(JSON.stringify(states));
        """
        states=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual([s['visible'] for s in states],[True,False,False,False,False,True])
        for state in states:
            self.assertEqual(state['ct'],state['mask'])
            self.assertEqual(state['ct'],'visible' if state['visible'] else 'hidden')
            self.assertEqual(state['gs'],'visible')
            self.assertEqual(state['anatomy'],'visible')
        html=(source.parent/'index.html').read_text()
        self.assertIn('id="showMPR" type="checkbox" checked',html)

    def test_new_session_rejects_old_slider_requests(self):
        with self.assertRaisesRegex(ValueError,'セッション'):
            self.app.controls(dict(action='slice',value=80,generation=-1))

    def test_javascript_sends_during_drag_and_keeps_only_latest_pending(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        source=Path(__file__).resolve().parents[1]/'browser_ui/slice-controls.js'
        script="""
        import {readFileSync} from 'node:fs';
        const {LatestControlSender,depthToPercent}=await import('data:text/javascript;base64,'+readFileSync(process.argv[1]).toString('base64'));
        let release;const sent=[];
        const sender=new LatestControlSender(async v=>{sent.push(v);if(v===0)await new Promise(r=>release=r);},1);
        sender.push(0);await new Promise(r=>setTimeout(r,10));
        for(let i=1;i<=100;i++)sender.push(i);
        const before=sent.slice();release();await new Promise(r=>setTimeout(r,10));
        console.log(JSON.stringify({before,sent,percent:depthToPercent(5,200)}));
        """
        data=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,str(source)],text=True))
        self.assertEqual(data['before'],[0])
        self.assertEqual(data['sent'],[0,100])
        self.assertEqual(data['percent'],2.5)


if __name__=='__main__':unittest.main()
