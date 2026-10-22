"""Offline map audit, job safety, resumable pair caching and browser API."""
import json
import os
import signal
import subprocess
import time
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_preprocessing import (inspect_map, reference_details, reference_preview,
    PreprocessingManager, atomic_json, selected_map)
from preprocess_registered_map import run_job
from enrich_registered_aliked_map import enrich
from browser_live import BrowserApplication, make_handler


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.map = self.root/'map'
        self.dataset = self.root/'colmap'
        (self.map/'reference_features').mkdir(parents=True)
        (self.dataset/'sparse/0').mkdir(parents=True)
        (self.dataset/'images').mkdir()
        self.xyz = np.array([[.2,-.1,5.]], np.float32)
        camera = self.dataset/'sparse/0/Cameras.txt'
        camera.write_text('1 PINHOLE 640 480 500 500 320 240\n')
        image_file = self.dataset/'sparse/0/Images.txt'
        lines = []
        K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
        for i, center in enumerate([(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]):
            name = '%05d.png' % i
            t = -np.array(center)
            lines.extend([f'{i+1} 1 0 0 0 {t[0]} {t[1]} {t[2]} 1 {name}', ''])
            points = np.vstack([self.xyz, [-.2,.1,5.]])
            uv = cv2.projectPoints(points, np.zeros(3), t, K, np.zeros(5))[0].reshape(-1,2)
            values = np.vstack([self.xyz, [np.nan]*3]).astype(np.float32)
            np.savez_compressed(self.map/'reference_features'/('%05d.npz'%i),
                point_ids=np.array([0,-1]), xyz=values, full_uv=uv,
                keypoints=uv[None].astype(np.float32), descriptors=np.ones((1,2,128),np.float32),
                keypoint_scores=np.ones((1,2),np.float32), image_size=np.array([[640.,480.]],np.float32),
                crop_box=np.array([0,0,640,480]), object_mask=np.ones((480,640),np.uint8))
            cv2.imwrite(str(self.dataset/'images'/name), np.full((480,640,3),120,np.uint8))
        image_file.write_text('\n'.join(lines)+'\n')
        np.savez_compressed(self.map/'object_aliked_map.npz', xyz=self.xyz,
            track_length=np.array([3]), max_reprojection_error=np.array([0.],np.float32))
        atomic_json(self.map/'summary.json', dict(image_features={}, input_files=dict(
            images=str(image_file), cameras=str(camera), splat='not used for XYZ')))
        self.manager = PreprocessingManager(self.root/'jobs')

    def tearDown(self):
        self.temp.cleanup()

    def test_audit_counts_unassigned_points_without_fabricating_xyz(self):
        value = inspect_map(self.map, audit=True)
        self.assertTrue(value['geometry_ok'])
        self.assertEqual((value['features'],value['xyz_observations'],value['xyz_landmarks']), (6,3,1))
        self.assertEqual(value['xyz_ratio'], .5)
        self.assertLess(value['reprojection_px']['maximum'], 1e-7)
        self.assertEqual(value['minimum_track_length'], 3)

    def test_audit_rejects_xyz_mismatch_and_large_reprojection(self):
        file = self.map/'reference_features/00000.npz'
        with np.load(file) as a:
            arrays = {k:a[k].copy() for k in a.files}
        arrays['xyz'][0,0] += 1
        arrays['full_uv'][0,0] += 50
        np.savez_compressed(file, **arrays)
        value = inspect_map(self.map, audit=True)
        self.assertFalse(value['geometry_ok'])
        self.assertTrue(any('xyz mismatch' in i for i in value['issues']))
        self.assertTrue(any('reprojection QC' in i for i in value['issues']))

    def test_reference_display_exposes_full_pixels_mask_xyz_and_errors(self):
        value = reference_details(self.map, '00000.png')
        self.assertEqual((value['width'],value['height']), (640,480))
        self.assertIsNotNone(value['points'][0]['xyz'])
        self.assertIsNone(value['points'][1]['xyz'])
        self.assertIsNone(value['points'][1]['reprojection_px'])
        self.assertTrue(value['mask_outlines'])
        self.assertTrue(reference_preview(self.map,'00000.png').startswith(b'\xff\xd8'))
        with self.assertRaises(ValueError):
            reference_preview(self.map, '../00000.png')

    def test_activation_requires_completed_audited_map_and_persists_selection(self):
        job = self.manager.directory/'20261001-test'
        atomic_json(job/'state.json', dict(status='running', output=str(self.map)))
        with self.assertRaises(ValueError):
            self.manager.activate(job.name)
        atomic_json(job/'state.json', dict(status='complete', output=str(self.map)))
        atomic_json(self.map/'audit.json', dict(geometry_ok=False))
        with self.assertRaises(ValueError):
            self.manager.activate(job.name)
        atomic_json(self.map/'audit.json', inspect_map(self.map,audit=True))
        with patch('map_preprocessing.SELECTION',self.root/'selection.json'):
            self.assertEqual(self.manager.activate(job.name)['map_path'],str(self.map))
            self.assertEqual(selected_map(self.root/'fallback'),self.map.resolve())

    def test_job_traversal_and_empty_selection_are_rejected(self):
        for name in ('..','../job','/tmp/x'):
            with self.assertRaises(ValueError):
                self.manager.cancel(name)
            with self.assertRaises(ValueError):
                self.manager.activate(name)
        with self.assertRaises(ValueError):
            self.manager.start(dict(source=str(self.map),scope='selected',target_images=[]))

    def test_modified_features_invalidate_saved_audit(self):
        atomic_json(self.map/'audit.json', inspect_map(self.map,audit=True))
        self.assertTrue(self.manager.inspect(self.map)['geometry_checked'])
        file=self.map/'reference_features/00000.npz'
        with np.load(file) as a:
            arrays={k:a[k].copy() for k in a.files}
        arrays['point_ids'][0]=-1
        arrays['xyz'][0]=np.nan
        np.savez_compressed(file,**arrays)
        result=self.manager.inspect(self.map)
        self.assertFalse(result['geometry_checked'])
        self.assertEqual(result['xyz_observations'],2)

    def test_cancelled_worker_keeps_source_and_does_not_publish_complete(self):
        job = self.root/'cancelled'
        atomic_json(job/'config.json',dict(source=str(self.map),scope='all',max_neighbors=12,device='cpu',coverage_goal=.5))
        (job/'cancel.request').touch()
        original = (self.map/'object_aliked_map.npz').read_bytes()
        value = run_job(job)
        self.assertEqual(value['status'],'cancelled')
        self.assertEqual((self.map/'object_aliked_map.npz').read_bytes(),original)
        self.assertNotIn('output',value)

    @unittest.skipUnless(hasattr(signal,'SIGSTOP'), 'POSIX suspension required')
    def test_pause_resume_preserves_live_worker_and_allows_validation(self):
        job=self.manager.directory/'pause-test'
        job.mkdir(parents=True)
        worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
            'preprocess_registered_map.py','--job-dir',str(job)])
        try:
            for _ in range(50):
                if str(job) in subprocess.check_output(['ps','-p',str(worker.pid),'-o','command='],text=True):
                    break
                time.sleep(.01)
            atomic_json(job/'state.json',dict(status='running',pid=worker.pid,completed=17,total=100))
            self.manager.pause(job.name)
            self.assertEqual(self.manager.jobs()[0]['status'],'paused')
            self.assertFalse(self.manager.busy())
            self.assertEqual(self.manager.jobs()[0]['completed'],17)
            self.assertIn('T',subprocess.check_output(['ps','-p',str(worker.pid),'-o','stat='],text=True))
            with self.assertRaises(ValueError):
                self.manager.start(dict(source=str(self.map)))
            self.manager.resume(job.name)
            self.assertTrue(self.manager.busy())
            self.manager.pause(job.name)
            self.manager.cancel(job.name)
            self.assertTrue((job/'cancel.request').exists())
            self.assertTrue(self.manager.busy())
        finally:
            os.kill(worker.pid,signal.SIGCONT)
            worker.terminate();worker.wait(timeout=5)

    def test_pause_refuses_unrelated_process(self):
        job=self.manager.directory/'wrong-worker'
        atomic_json(job/'state.json',dict(status='running',pid=os.getpid()))
        with self.assertRaises(ValueError):
            self.manager.pause(job.name)
        with self.assertRaises(ValueError):
            self.manager.pause('../unsafe')

    def test_all_scope_processes_all_images_then_weak_images_and_audits(self):
        job = self.root/'all'
        atomic_json(job/'config.json',dict(source=str(self.map),scope='all',max_neighbors=24,
            device='cpu',coverage_goal=.8,target_images=[]))
        calls=[]
        def fake_enrich(args):
            import shutil
            calls.append(args)
            shutil.copytree(args.source,args.output)
            return dict(counts=dict(added_observations=0),processing_seconds=.01)
        with patch('preprocess_registered_map.enrich',side_effect=fake_enrich):
            result=run_job(job)
        self.assertEqual(result['status'],'complete')
        self.assertEqual([c.max_neighbors for c in calls],[12,24])
        self.assertEqual(len(calls[0].target_images),3)
        self.assertEqual(len(calls[1].target_images),3)
        self.assertTrue(all(c.matching_method == 'known-pose' for c in calls))
        self.assertTrue(json.loads((Path(result['output'])/'audit.json').read_text())['original_xyz_preserved'])

    def test_old_job_exposes_baseline_read_only_without_changing_state_file(self):
        job=self.manager.directory/'old'
        state=dict(status='complete',output=str(self.map))
        atomic_json(job/'state.json',state)
        atomic_json(job/'before.json',inspect_map(self.map))
        value=self.manager.jobs()[0]
        self.assertEqual(value['source_overview']['xyz_observations'],3)
        self.assertEqual(json.loads((job/'state.json').read_text()),state)
        self.assertEqual(self.manager.jobs()[0]['source_overview'],value['source_overview'])

    def test_optional_incomplete_baseline_does_not_hide_valid_job(self):
        job=self.manager.directory/'old'
        state=dict(status='complete',output=str(self.map))
        atomic_json(job/'state.json',state)
        atomic_json(job/'before.json',dict(images=3))
        self.assertEqual(self.manager.jobs(),[state])

    def test_phase_units_reset_and_written_map_not_published_until_audit(self):
        import shutil
        job=self.root/'phase-units'
        atomic_json(job/'config.json',dict(source=str(self.map),scope='all',max_neighbors=12,
            device='cpu',coverage_goal=.5,target_images=[]))
        states=[]
        def fake_enrich(args):
            shutil.copytree(args.source,args.output)
            for event in (dict(stage='matching',completed=90,total=100),
                          dict(stage='triangulation'),dict(stage='export',total=3),
                          dict(stage='complete',output=str(args.output))):
                args.progress(event)
                states.append(json.loads((job/'state.json').read_text()))
            return dict(counts=dict(added_observations=0),processing_seconds=.01)
        with patch('preprocess_registered_map.enrich',side_effect=fake_enrich):
            result=run_job(job)
        self.assertEqual((states[1]['completed'],states[1]['total']),(0,0))
        self.assertEqual((states[2]['completed'],states[2]['total']),(0,3))
        self.assertEqual(states[3]['stage'],'audit')
        self.assertEqual(states[3]['status'],'running')
        self.assertNotIn('output',states[3])
        self.assertEqual(states[3]['passes'],[])
        self.assertEqual(states[3]['source_overview']['xyz_observations'],3)
        self.assertEqual(result['stage'],'complete')
        self.assertTrue(result['passes'])

    def test_pair_cache_reuses_matching_but_recomputes_xyz_assignments(self):
        class Tensor:
            def __init__(self,value): self.value=value
            def to(self,device): return self
        args=dict(source=self.map,target_images=None,max_neighbors=12,min_anchor_support=2,max_error=3.,
                  device='cpu',cache_dir=self.root/'cache',matching_method='lightglue')
        fake_torch=SimpleNamespace(from_numpy=Tensor)
        match=Mock(return_value=(None,None,np.array([[0,0],[1,1]]),np.array([.9,.8])))
        with patch('enrich_registered_aliked_map._load_models',return_value=(fake_torch,None,object(),'cpu')):
            with patch('enrich_registered_aliked_map._match_pair',match):
                first=enrich(SimpleNamespace(**args,output=self.root/'first'))
                second=enrich(SimpleNamespace(**args,output=self.root/'second'))
        self.assertEqual(match.call_count,3)
        self.assertEqual(second['counts']['cached_pairs'],3)
        self.assertEqual(first['counts']['new_landmarks'],1)
        self.assertEqual(second['counts']['new_landmarks'],1)
        self.assertTrue(inspect_map(self.root/'first',audit=True)['geometry_ok'])

    def test_known_pose_enrichment_needs_no_lightglue_and_preserves_original_xyz(self):
        # All points on a common epipolar line, distinct appearances provide
        # ambiguity evidence for ratio matching. A third view verifies XYZ.
        K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
        points=np.array([[.2,-.1,5.],[-.2,-.1,5.],[.5,-.1,5.]])
        for i, center in enumerate([(0.,0.,0.),(1.,0.,0.),(-1.,0.,0.)]):
            name='%05d.npz'%i
            path=self.map/'reference_features'/name
            with np.load(path) as a:
                arrays={k:a[k].copy() for k in a.files}
            uv=cv2.projectPoints(points,np.zeros(3),-np.asarray(center),K,np.zeros(5))[0].reshape(-1,2)
            arrays.update(point_ids=np.array([0,-1,-1]),xyz=np.vstack([self.xyz,[np.nan]*3,[np.nan]*3]),
                          full_uv=uv,keypoints=uv[None].astype(np.float32),
                          descriptors=np.eye(3,128,dtype=np.float32)[None],keypoint_scores=np.ones((1,3)))
            np.savez_compressed(path,**arrays)
        camera_file=self.dataset/'sparse/0/Images.txt'
        camera_file.write_text('1 1 0 0 0 0 0 0 1 00000.png\n\n'
                              '2 1 0 0 0 -1 0 0 1 00001.png\n\n'
                              '3 1 0 0 0 1 0 0 1 00002.png\n\n')
        original=(self.map/'object_aliked_map.npz').read_bytes()
        args=dict(source=self.map,target_images=None,max_neighbors=12,min_anchor_support=2,max_error=3.,
                  device='cpu',cache_dir=self.root/'cache',matching_method='known-pose')
        with patch('enrich_registered_aliked_map._load_models',side_effect=AssertionError('No model needed')):
            first=enrich(SimpleNamespace(**args,output=self.root/'known-first'))
            second=enrich(SimpleNamespace(**args,output=self.root/'known-second'))
        self.assertEqual(first['matching_method'],'known-pose')
        self.assertEqual(first['counts']['new_landmarks'],2)
        self.assertEqual(second['counts']['cached_pairs'],3)
        self.assertEqual((self.map/'object_aliked_map.npz').read_bytes(),original)
        audited=inspect_map(self.root/'known-first',audit=True)
        self.assertTrue(audited['geometry_ok'])
        with np.load(self.root/'known-first/object_aliked_map.npz') as a:
            np.testing.assert_array_equal(a['xyz'][:1],self.xyz)

    def test_api_checks_token_and_live_competition_and_serves_preview(self):
        app=BrowserApplication()
        app.preprocessor=self.manager
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(app))
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def post(route,values,token):
            return urlopen(Request(base+route,data=json.dumps(values).encode(),headers={
                'Content-Type':'application/json','X-ViewSync-Token':token}))
        try:
            self.assertIn(b'preprocess.js',urlopen(base+'/preprocess').read())
            with patch('browser_live.choose_local_file',return_value=dict(cancelled=True)) as picker:
                with self.assertRaises(HTTPError) as caught:
                    post('/api/select-file',dict(field='map_path'),'bad')
                self.assertEqual(caught.exception.code,403)
                picker.assert_not_called()
                answer=json.loads(post('/api/select-file',dict(field='map_path'),app.token).read())
                self.assertTrue(answer['cancelled'])
                picker.assert_called_once_with('map_path')
            with patch('browser_live.recommended_map',return_value=self.map), patch('browser_live.audited_map',return_value=(self.map,dict(xyz_ratio=.5,xyz_observations=3))):
                answer=json.loads(post('/api/recommended-map',{},app.token).read())
                self.assertEqual(answer['path'],str(self.map))
                self.assertEqual(answer['xyz_ratio'],.5)
            with self.assertRaises(HTTPError) as caught:
                post('/api/preprocess/start',dict(source=str(self.map)),'bad')
            self.assertEqual(caught.exception.code,403)
            app.update(status='running')
            with patch('browser_live.choose_local_file') as picker:
                with self.assertRaises(HTTPError):
                    post('/api/select-file',dict(field='map_path'),app.token)
                picker.assert_not_called()
            with patch.object(self.manager,'resume') as resume:
                with self.assertRaises(HTTPError):
                    post('/api/preprocess/resume',dict(job_id='paused-job'),app.token)
                resume.assert_not_called()
            with self.assertRaises(HTTPError) as caught:
                post('/api/preprocess/start',dict(source=str(self.map)),app.token)
            self.assertEqual(caught.exception.code,400)
            app.update(status='idle')
            with patch.object(self.manager,'pause',return_value=dict(status='paused')) as pause:
                answer=json.loads(post('/api/preprocess/pause',dict(job_id='test-job'),app.token).read())
                self.assertEqual(answer['status'],'paused')
                pause.assert_called_once_with('test-job')
            with patch.object(self.manager,'busy',return_value=False):
                self.assertEqual(post('/api/start',{},app.token).status,202)
                app.update(status='idle')
            with patch.object(self.manager,'busy',return_value=True):
                with self.assertRaises(HTTPError) as caught:
                    post('/api/start',{},app.token)
                self.assertEqual(caught.exception.code,400)
            from urllib.parse import urlencode
            query=urlencode(dict(source=str(self.map),image='00000.png'))
            value=json.loads(urlopen(base+'/api/preprocess/reference?'+query).read())
            self.assertEqual(len(value['points']),2)
        finally:
            server.shutdown();server.server_close();thread.join()
            app.executor.shutdown();app.layer_executor.shutdown()


if __name__=='__main__':
    unittest.main()
