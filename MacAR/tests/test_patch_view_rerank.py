import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patch_view_rerank import GRID_SIDE, patch_metrics, rerank, square_mask, select_diverse
from view_retrieval import ViewRetrievalIndex
from browser_live import runtime_arguments


class FakeTokens:
    model_name='test-tokens'
    repository='test-pinned'
    calls=0

    def encode_tokens(self, image):
        self.calls+=1
        patches=cv2.resize(image,(GRID_SIDE,GRID_SIDE)).astype(np.float32)
        patches/=np.maximum(np.linalg.norm(patches,axis=-1,keepdims=True),1e-8)
        vector=np.ones(3,np.float32)/np.sqrt(3.)
        return vector,patches


from tests.runtime_dataset import RuntimeDataset


class PatchRerankTests(RuntimeDataset, unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(31)
        self.grid=rng.normal(size=(8,8,128)).astype(np.float32)
        self.grid/=np.linalg.norm(self.grid,axis=-1,keepdims=True)
        self.mask=np.ones((8,8),bool)

    def test_correct_layout_beats_equal_global_similarity_wrong_face(self):
        references=np.stack([self.grid[:,::-1],self.grid])
        ranking,rows=rerank([(1.,0.,0),(.99,0.,1)],self.grid,self.mask,references,np.stack([self.mask]*2))
        self.assertEqual(ranking[0][2],1)
        self.assertGreater(rows[0]['inliers'],50)
        self.assertGreater(rows[0]['consistency'],.9)
        self.assertEqual(rows[0]['global_rank'],2)

    def test_reflection_is_not_good_spatial_consistency(self):
        scores=patch_metrics(self.grid,self.grid[:,::-1],self.mask,self.mask)
        self.assertLess(scores['consistency'],.2)

    def test_translated_patch_correspondences_can_support_a_view(self):
        query=np.roll(self.grid,1,axis=1)
        qm=self.mask.copy();qm[:,0]=False
        scores=patch_metrics(query,self.grid,qm,self.mask)
        self.assertGreater(scores['consistency'],.8)
        self.assertGreater(scores['inliers'],40)
        self.assertLess(scores['residual'],.01)

    def test_no_foreground_or_nonfinite_features_do_not_invent_support(self):
        scores=patch_metrics(self.grid,self.grid,np.zeros((8,8),bool),self.mask)
        self.assertEqual((scores['layout'],scores['consistency'],scores['inliers']),(0.,0.,0))
        bad=self.grid.copy();bad[0,0,0]=float('nan')
        self.assertEqual(patch_metrics(bad,self.grid,self.mask,self.mask)['consistency'],0.)

    def test_repeated_constant_patches_have_no_unique_matches(self):
        patches=np.ones_like(self.grid)
        scores=patch_metrics(patches,patches,self.mask,self.mask)
        self.assertEqual(scores['mutual_matches'],0)
        self.assertEqual(scores['consistency'],0.)

    def test_background_tokens_do_not_change_layout_score(self):
        mask=self.mask.copy();mask[:2]=False
        changed=self.grid.copy();changed[:2]*=-1
        original=patch_metrics(self.grid,self.grid,mask,mask)
        background=patch_metrics(changed,self.grid,mask,mask)
        self.assertAlmostEqual(background['layout'],original['layout'],places=5)

    def test_square_mask_excludes_letterbox(self):
        mask=square_mask(np.ones((20,40),np.uint8))
        self.assertEqual(mask.sum(),GRID_SIDE**2//2)
        self.assertFalse(mask[:2].any())
        self.assertFalse(mask[-2:].any())

    def test_index_cache_query_single_forward_and_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);views=[]
            for index in range(12):
                image=np.full((64,64,3),255,np.uint8)
                image[8:56,8:32]=(30,30,200)
                image[8:56,32:56]=(200,30,30)
                if index%2:image=image[:,::-1].copy()
                name=f'view_{index}.png';cv2.imwrite(str(root/name),image)
                views.append(dict(image_name=name))
            encoder=FakeTokens()
            index=ViewRetrievalIndex(root,views,mode='dino-patch',crop=True,roi=(0,0,1,1),encoder=encoder)
            count=encoder.calls
            query=cv2.imread(str(root/'view_0.png'))
            ranked,_,all_ranked,debug=index.rank_debug(query,top_k=3,excluded={0})
            self.assertEqual(encoder.calls,count+1)
            self.assertEqual(len(ranked),3)
            self.assertEqual(len(all_ranked),10)
            self.assertNotIn(0,[entry[2] for entry in all_ranked])
            self.assertEqual(debug['patch_rerank']['shortlist'],10)
            tokens=index.last_query_tokens
            self.assertEqual(tokens[1].shape,(16,16,3))
            self.assertEqual(tokens[2].shape,(16,16))
            captured=index.rank_debug(query,top_k=3,capture_query_tokens=True)[3]['_query_tokens']
            index.rank_debug(query[:,::-1].copy(),top_k=3)
            self.assertIsNot(captured,index.last_query_tokens)
            np.testing.assert_array_equal(captured[1],tokens[1])
            # A no-object query must not expose tokens from the preceding frame.
            index.auto_roi=True
            index.detector=Mock()
            index.detector.detect.return_value=(None,{})
            index.rank_debug(query,top_k=3)
            self.assertIsNone(index.last_query_tokens)
            again_encoder=FakeTokens()
            again=ViewRetrievalIndex(root,views,mode='dino-patch',crop=True,encoder=again_encoder)
            self.assertEqual(again_encoder.calls,0)
            np.testing.assert_allclose(index.reference_patches,again.reference_patches)
            # Interrupted/corrupt archives are rebuilt rather than trusted.
            index._cache_path().write_bytes(b'not an npz archive')
            rebuilt=ViewRetrievalIndex(root,views,mode='dino-patch',crop=True,encoder=FakeTokens())
            self.assertEqual(len(rebuilt.reference),12)

    def test_browser_switches_only_coarse_method_and_keeps_pnp(self):
        args,_,_=runtime_arguments({'mode':'gs'})
        baseline,_,_=runtime_arguments({'mode':'gs','coarse_method':'dino-global'})
        self.assertEqual(args.retrieval_mode,'dino-patch')
        self.assertEqual(baseline.retrieval_mode,'dino-global')
        for field in ('glb_top_k','min_inliers','min_inlier_ratio','ransac_threshold','virtual_policy','tracking_reseed_frames'):
            self.assertEqual(getattr(args,field),getattr(baseline,field))
        with self.assertRaises(ValueError):runtime_arguments({'coarse_method':'unsupported'})

    def test_fine_matching_slots_do_not_all_go_to_same_camera_direction(self):
        views=[]
        for degrees in (0,0,30,90,180):
            transform=np.eye(4)
            transform[:3,:3]=cv2.Rodrigues(np.array([0.,0.,np.radians(degrees)]))[0]
            views.append(dict(world_to_camera=transform.tolist()))
        ranked=[(1.-i*.1,0.,i) for i in range(len(views))]
        chosen=select_diverse(ranked,views,3)
        self.assertEqual([entry[2] for entry in chosen],[0,3,4])
        self.assertEqual(len(chosen),3)

    def test_diversity_falls_back_to_score_when_only_one_direction_exists(self):
        views=[dict(world_to_camera=np.eye(4).tolist()) for _ in range(5)]
        ranked=[(1.-i*.1,0.,i) for i in range(5)]
        self.assertEqual(select_diverse(ranked,views,3),ranked[:3])


if __name__=='__main__':unittest.main()
