import sys
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_appearance_retrieval import (select_appearance_references, rank_real_references,
                                      RegisteredAppearanceIndex)
from registered_aliked_live import RegisteredALIKEDLiveLocalizer


def view(degrees, name):
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.radians(degrees)]))[0]
    return dict(image_name=name, world_to_camera=transform.tolist())


class RealAppearanceRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.views = [view(0, 'front'), view(90, 'side'), view(180, 'back')]
        self.registered = {name: dict(rotation=np.asarray(view(deg, name)['world_to_camera'])[:3, :3])
                           for name, deg in [('a', 1), ('b', 12), ('c', 91), ('d', 102), ('e', 179)]}
        self.coarse = [(1., 0., 0), (.9, 0., 1)]

    def test_hybrid_uses_appearance_inside_direction_gate_and_deduplicates(self):
        rankings = {0: [('e', 1.), ('b', .9), ('a', .8)],
                    1: [('b', 1.), ('d', .9), ('c', .8)]}
        refs = select_appearance_references(self.views, self.registered, self.coarse,
                                           rankings, mode='gs-real', per_view=2)
        self.assertEqual([r[0] for r in refs], ['b', 'a', 'd', 'c'])
        self.assertEqual([r[1] for r in refs], [0, 0, 1, 1])
        self.assertTrue(all(r[3] <= 55 for r in refs))

    def test_direct_uses_exact_ranked_photo_not_another_angle_neighbor(self):
        refs = select_appearance_references(self.views, self.registered, self.coarse,
            {None: [('e', 1.), ('b', .9), ('e', .8), ('missing', .7), ('a', .6)]},
            mode='real-direct', budget=3)
        self.assertEqual([r[0] for r in refs], ['e', 'b', 'a'])
        self.assertEqual(refs[0][1], 2)

    def test_hybrid_does_not_fabricate_candidates_when_direction_wrong(self):
        refs = select_appearance_references(self.views, {'e': self.registered['e']},
            self.coarse, {0: [('e', 1.)], 1: [('e', 1.)]}, mode='gs-real')
        self.assertEqual(refs, [])

    def test_ranking_excludes_ineligible_photos_before_shortlist(self):
        names = ['wrong', 'valid', 'valid2']
        vectors = np.array([[1., 0.], [.8, .6], [.6, .8]], np.float32)
        patches = np.tile(np.array([1., 0.], np.float32), (3, 16, 16, 1))
        masks = np.ones((3, 16, 16), bool)
        rows = rank_real_references(names, vectors, patches, masks,
            np.array([1., 0.]), patches[0], masks[0], {'valid', 'valid2'})
        self.assertEqual([r[0] for r in rows], ['valid', 'valid2'])
        self.assertEqual(rank_real_references(names, vectors, patches, masks,
            np.array([1., 0.]), patches[0], masks[0], set()), [])

    def test_live_selector_reuses_coarse_query_and_keeps_direction_gate(self):
        engine = RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.views, engine.registered = self.views, self.registered
        engine.top_k, engine.max_coarse_rotation_deg = 3, 55.
        tokens = (np.array([1., 0.]), np.ones((16,16,2)), np.ones((16,16),bool))
        engine.retriever = SimpleNamespace(last_query_tokens=tokens)
        engine.real_appearance = SimpleNamespace(select=Mock(return_value=[('b',0,.9,12.)]))
        engine.last_crop_debug = dict(bbox=(0.,0.,1.,1.))
        workspace = {'appearance_query': tokens}
        actual = engine._appearance_references(np.zeros((10,10,3),np.uint8), self.coarse, workspace)
        self.assertEqual(actual, [('b',0,.9,12.)])
        self.assertIs(workspace['appearance_query'], tokens)
        arguments = engine.real_appearance.select.call_args
        self.assertIs(arguments.args[3], tokens)
        self.assertEqual(arguments.kwargs, dict(budget=6, max_angle=55.))
        self.assertEqual(engine.progress_snapshot()['reference_selection'], 'gs-real')

    def test_pose_override_encodes_same_roi_once_not_stale_global_tokens(self):
        engine = RegisteredALIKEDLiveLocalizer.__new__(RegisteredALIKEDLiveLocalizer)
        engine.views, engine.registered = self.views, self.registered
        engine.top_k, engine.max_coarse_rotation_deg = 1, 55.
        engine.last_crop_debug = dict(bbox=(.25,.2,.5,.6), detection=None)
        encoder = SimpleNamespace(encode_tokens=Mock(return_value=(np.array([1.,0.]),np.ones((16,16,2)))))
        engine.retriever = SimpleNamespace(last_query_tokens=('stale',))
        engine.real_appearance = SimpleNamespace(encoder=encoder,select=Mock(return_value=[]))
        image=np.zeros((10,20,3),np.uint8)
        workspace = dict(ranked_override=self.coarse)
        engine._appearance_references(image,self.coarse,workspace)
        engine._appearance_references(image,self.coarse,workspace)
        self.assertEqual(encoder.encode_tokens.call_count,1)
        self.assertEqual(encoder.encode_tokens.call_args.args[0].shape,(6,10,3))

    def test_index_is_cached_and_reference_updates_invalidate_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image_dir=root/'custom_RGB';image_dir.mkdir(parents=True)
            pose=root/'colmap/sparse/0/Images.txt';pose.parent.mkdir(parents=True);pose.touch()
            (root/'summary.json').write_text(json.dumps(dict(image_dir=str(image_dir),
                                                            input_files=dict(images=str(pose)))))
            image=np.full((12,16,3),127,np.uint8)
            cv2.imwrite(str(image_dir/'a.png'),image)
            features=root/'a.npz'
            np.savez(features,crop_box=np.array([2,3,14,11]),object_mask=np.ones((8,12),bool))
            registered={'a.png':dict(path=features,rotation=np.eye(3))}
            encoder=SimpleNamespace(model_name='test',repository='test',encode_tokens=Mock(
                return_value=(np.array([1.,0.],np.float32),np.tile([1.,0.],(16,16,1)).astype(np.float32))))
            events=[]
            index=RegisteredAppearanceIndex(root,registered,encoder,cache_root=root/'cache',
                progress=lambda **v:events.append(v))
            self.assertEqual(encoder.encode_tokens.call_count,1)
            self.assertEqual(encoder.encode_tokens.call_args.args[0].shape,(8,12,3))
            self.assertEqual(events[-1]['appearance_loaded'],1)
            encoder.encode_tokens.reset_mock()
            loaded=RegisteredAppearanceIndex(root,registered,encoder,cache_root=root/'cache')
            encoder.encode_tokens.assert_not_called()
            np.testing.assert_array_equal(loaded.patches,index.patches)
            # A bad cache is rebuilt, never silently used or replaced by angle-only selection.
            index.cache_path.write_bytes(b'bad')
            RegisteredAppearanceIndex(root,registered,encoder,cache_root=root/'cache')
            self.assertEqual(encoder.encode_tokens.call_count,1)
            # The active enriched map uses the standard COLMAP RGB layout,
            # with no explicit image_dir. It must work as well as custom dirs.
            image_dir.rename(root/'colmap/images')
            (root/'summary.json').write_text(json.dumps(dict(input_files=dict(images=str(pose)))))
            standard=RegisteredAppearanceIndex(root,registered,encoder,cache_root=root/'cache')
            self.assertEqual(encoder.encode_tokens.call_count,2)
            self.assertNotEqual(standard.cache_path,index.cache_path)
            np.savez(features,crop_box=np.array([2,3,14,11]),object_mask=np.ones((1,1),bool))
            with self.assertRaisesRegex(ValueError,'crop/mask'):
                RegisteredAppearanceIndex(root,registered,encoder,cache_root=root/'cache')


if __name__ == '__main__':
    unittest.main()
