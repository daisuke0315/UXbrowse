import threading
import unittest
from types import SimpleNamespace

import numpy as np

from image_occlusion import occlude_image, compare_image_occlusion
from test_browser_live import result


class ImageOcclusionTests(unittest.TestCase):
    def test_nonconvex_mask_changes_only_selected_pixels_and_guards_edges(self):
        image = np.full((60, 60, 3), 200, np.uint8)
        polygon = [[10, 10], [40, 10], [40, 20], [20, 20], [20, 40], [10, 40]]
        masked, guard, fraction = occlude_image(image, polygon, 2)
        self.assertEqual(masked[15, 30, 0], 75)
        self.assertEqual(masked[30, 30, 0], 200)
        self.assertTrue(np.all(image == 200))
        self.assertGreater(fraction, 0)
        import cv2
        self.assertGreaterEqual(cv2.pointPolygonTest(guard, (9., 15.), False), 0)

    def test_rejects_degenerate_outside_or_nonfinite_polygons(self):
        image = np.zeros((30, 30, 3), np.uint8)
        for polygon in ([], [[1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
                        [[-1, 1], [20, 1], [20, 20]],
                        [[1,1],[20,20],[20,1],[1,20]],
                        [[1, 1], [20, 1], [float('nan'), 20]]):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                occlude_image(image, polygon)

    def test_each_image_gets_fresh_global_search_and_live_state_is_restored(self):
        for registered in (True, False):
            with self.subTest(registered=registered):
                engine = SimpleNamespace(lock=threading.Lock(), progress_lock=threading.Lock(),
                    progress={'stage': 'live'}, top_k=3, max_reprojection_error=3.,
                    min_coverage=.005, last_debug='live', last_tracking_seed='seed',
                    last_diagnostics={'live': True}, query_tracking_callback='callback',
                    search_policy='adaptive', diagnostic_sink='live_logger')
                calls = []
                def search(image, k, d, **kwargs):
                    calls.append((image.copy(), kwargs))
                    engine.last_debug = 'trial'
                    engine.last_diagnostics = {'virtual_views': [str(len(calls))]}
                    self.assertIsNone(engine.query_tracking_callback)
                    self.assertIsNone(engine.diagnostic_sink)
                    if registered:
                        self.assertEqual(engine.search_policy, 'exhaustive')
                    return result(), None
                if registered:
                    engine.registered = {}
                    engine._search_attempts = lambda im,k,d,poly,workspace: search(
                        im,k,d,excluded_polygon=poly,workspace=workspace)
                else:
                    engine._localize_locked = search
                    engine._quality_attempts = lambda f: f()
                baseline, masked, debug, diagnostics = compare_image_occlusion(
                    engine, np.full((50, 50, 3), 200, np.uint8), np.eye(3),
                    np.zeros(5), [[10,10],[30,10],[30,30],[10,30]], 4., .001)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][0][20,20,0], 200)
                self.assertEqual(calls[1][0][20,20,0], 75)
                self.assertIsNone(calls[0][1]['excluded_polygon'])
                self.assertNotIn('ranked_override', calls[1][1])
                self.assertEqual(diagnostics['baseline_diagnostics']['virtual_views'], ['1'])
                self.assertEqual(debug, 'trial')
                self.assertEqual(engine.last_debug, 'live')
                self.assertEqual(engine.last_tracking_seed, 'seed')
                self.assertEqual(engine.max_reprojection_error, 3.)
                self.assertEqual(engine.query_tracking_callback, 'callback')
                self.assertEqual(engine.diagnostic_sink, 'live_logger')
                self.assertEqual(engine.search_policy, 'adaptive')
                self.assertEqual(engine.progress, {'stage': 'live'})

    def test_failed_search_restores_settings_and_callbacks(self):
        engine = SimpleNamespace(lock=threading.Lock(), progress_lock=threading.Lock(),
            progress={}, max_reprojection_error=3., min_coverage=.005,
            last_debug='live', last_tracking_seed='seed', last_diagnostics={},
            query_tracking_callback='callback', _quality_attempts=lambda f: f())
        def fail(*args, **kwargs):
            engine.last_debug = 'failed_trial'
            raise RuntimeError('test error')
        engine._localize_locked = fail
        with self.assertRaisesRegex(RuntimeError, 'test error'):
            compare_image_occlusion(engine, np.zeros((40,40,3),np.uint8),
                np.eye(3),np.zeros(5),[[10,10],[30,10],[30,30],[10,30]],4.,.001)
        self.assertEqual(engine.max_reprojection_error, 3.)
        self.assertEqual(engine.last_debug, 'live')
        self.assertEqual(engine.query_tracking_callback, 'callback')

    def test_fixed_gs_references_skip_retrieval_but_recompute_each_image(self):
        engine = SimpleNamespace(lock=threading.Lock(), progress_lock=threading.Lock(),
            progress={'stage': 'live'}, views=[{'image_name': 'other.png'},
                {'image_name': 'original.png'}], last_debug='live', last_diagnostics={},
            _quality_attempts=lambda f: f(), max_reprojection_error=3., min_coverage=.005)
        calls = []
        def search(image, k, d, **kwargs):
            calls.append((image.copy(), kwargs))
            engine.last_debug = 'new_debug'
            engine.last_diagnostics = {'retrieval': ['original.png']}
            self.assertNotIn('pose_prior', kwargs)
            return result(), None
        engine._localize_locked = search
        baseline, masked, debug, diagnostics = compare_image_occlusion(
            engine, np.full((50,50,3),200,np.uint8), np.eye(3), np.zeros(5),
            [[10,10],[30,10],[30,30],[10,30]], 4., .001,
            reference_names=('original.png',), baseline_pose=(np.zeros(3),np.ones(3)))
        self.assertEqual(len(calls), 2)
        self.assertEqual([c[1]['ranked_override'] for c in calls],
                         [[(1.,0.,1)],[(1.,0.,1)]])
        self.assertEqual(calls[0][0][20,20,0],200)
        self.assertEqual(calls[1][0][20,20,0],75)
        self.assertIsNone(calls[0][1]['excluded_polygon'])
        self.assertIsNotNone(calls[1][1]['excluded_polygon'])
        self.assertIsNot(calls[0][1].get('query_workspace'),calls[1][1].get('query_workspace'))
        self.assertEqual(diagnostics['search_mode'],'fixed_references')
        self.assertEqual(diagnostics['reference_views'],['original.png'])
        self.assertEqual(debug,'new_debug')
        self.assertEqual(engine.last_debug,'live')
        self.assertEqual(engine.progress,{'stage':'live'})

    def test_fixed_registered_photos_use_validation_path_not_global_search(self):
        engine = SimpleNamespace(lock=threading.RLock(), progress_lock=threading.Lock(),
            progress={}, registered={'original.png':{}}, last_debug='live',
            last_diagnostics={'live':True}, max_reprojection_error=3., min_coverage=.005,
            search_policy='adaptive')
        calls=[]
        pose=(np.array([.1,.2,.3]),np.array([10.,20.,30.]))
        def search(image,k,d,polygon,error,coverage,names,baseline_pose):
            calls.append((image.copy(),polygon,names,baseline_pose))
            self.assertEqual(engine.search_policy,'exhaustive')
            return result(), 'new_debug', {'selected':'original.png'}
        engine.localize_validation=search
        baseline,masked,debug,diagnostics=compare_image_occlusion(engine,
            np.full((50,50,3),200,np.uint8),np.eye(3),np.zeros(5),
            [[10,10],[30,10],[30,30],[10,30]],4.,.001,
            reference_names=('original.png',),baseline_pose=pose)
        self.assertEqual(len(calls),2)
        self.assertIsNone(calls[0][1])
        self.assertIsNotNone(calls[1][1])
        self.assertEqual(calls[0][2],('original.png',))
        self.assertEqual(calls[1][2],('original.png',))
        self.assertIs(calls[1][3],pose)
        self.assertEqual(calls[1][0][20,20,0],75)
        self.assertEqual(diagnostics['baseline_diagnostics']['selected'],'original.png')
        self.assertEqual(debug,'new_debug')
        self.assertEqual(engine.last_diagnostics,{'live':True})
        self.assertEqual(engine.search_policy,'adaptive')

    def test_fixed_references_never_silently_fall_back_when_unavailable(self):
        engine=SimpleNamespace(lock=threading.RLock(),progress_lock=threading.Lock(),
            progress={},views=[{'image_name':'other.png'}],last_debug='live')
        for names in ((),('missing.png',),('POSE_LANDMARKS',)):
            with self.subTest(names=names),self.assertRaisesRegex(ValueError,'参照'):
                compare_image_occlusion(engine,np.zeros((40,40,3),np.uint8),
                    np.eye(3),np.zeros(5),[[10,10],[30,10],[30,30],[10,30]],4.,.001,
                    reference_names=names,baseline_pose=(np.zeros(3),np.ones(3)))
        self.assertEqual(engine.last_debug,'live')
