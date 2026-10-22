import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enrich_registered_aliked_map import (consensus_assignments, robust_triangulation, enrichment_pairs,
                                         anchor_pair_errors, projected_errors)
from build_aliked_object_map import triangulate_track
from offline_pnp_test import ReferenceImage


class EnrichmentTests(unittest.TestCase):
    def test_anchor_requires_two_independent_images(self):
        ids={'target':np.array([-1,-1])}
        accepted,rejected=consensus_assignments({('target',0):{42:{'a'}},
                                                ('target',1):{43:{'a','b'}}},ids)
        self.assertEqual(accepted,{('target',1):43})
        self.assertEqual(rejected['anchor_low_support'],1)
        np.testing.assert_array_equal(ids['target'],[-1,-1])

    def test_anchor_rejects_conflicting_landmarks_and_duplicate_observations(self):
        ids={'target':np.array([-1,-1,-1,-1,44])}
        proposals={('target',0):{42:{'a','b'},43:{'c','d'}},
                   ('target',1):{45:{'a','b'}},('target',2):{45:{'c','d'}},
                   ('target',3):{44:{'a','b'}}}
        accepted,rejected=consensus_assignments(proposals,ids)
        self.assertEqual(accepted,{})
        self.assertEqual(rejected['anchor_conflict'],1)
        self.assertEqual(rejected['anchor_duplicate_observation'],3)

    def geometry(self):
        K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
        centers=[np.array(c,float) for c in [(0,0,0),(1,0,0),(0,1,0),(-1,0,0)]]
        records={str(i):ReferenceImage(i,1,str(i),(1.,0.,0.,0.,*(-c))) for i,c in enumerate(centers)}
        xyz=np.array([.2,-.1,5.])
        uv={name:cv2.projectPoints(xyz.reshape(1,3),np.zeros(3),np.asarray(rec.pose[4:]),K,np.zeros(5))[0].reshape(1,2)
            for name,rec in records.items()}
        return records,uv,{1:K},{1:np.zeros(5)},np.array([[-1.,-1.,4.],[1.,1.,6.]]),xyz

    def test_robust_triangulation_retains_three_good_views_and_drops_one_bad(self):
        records,uv,K,D,bounds,xyz=self.geometry()
        uv['3']=uv['3']+np.array([0.,60.])
        nodes=[(name,0) for name in records]
        old,_=triangulate_track(nodes,uv,records,K,D,bounds,2.,3.,10.)
        self.assertIsNone(old)
        result=robust_triangulation(nodes,uv,records,K,D,bounds)
        self.assertIsNotNone(result)
        self.assertEqual(set(result[0]),{('0',0),('1',0),('2',0)})
        np.testing.assert_allclose(result[1],xyz,atol=1e-8)
        self.assertLess(max(result[2]),1e-6)

    def test_new_landmark_cannot_be_created_from_only_two_views(self):
        records,uv,K,D,bounds,_=self.geometry()
        self.assertIsNone(robust_triangulation([('0',0),('1',0)],uv,records,K,D,bounds))

    def test_pair_selection_excludes_self_and_has_no_duplicate_edges(self):
        records,_,_,_,_,_=self.geometry()
        pairs=enrichment_pairs(records,np.array([0.,0.,5.]),set(records),3)
        self.assertGreater(len(pairs),0)
        self.assertEqual(len(pairs),len(set(pairs)))
        self.assertTrue(all(a!=b for a,b in pairs))

    def test_batch_anchor_check_equals_individual_fixed_xyz_check(self):
        records,uv,K,D,_,xyz=self.geometry()
        ids={'0':np.array([0]),'1':np.array([-1])}
        got=anchor_pair_errors(np.array([[0,0]]),'0','1',ids,np.array([xyz]),uv,records,K,D)
        expected=projected_errors([('0',0),('1',0)],xyz,uv,records,K,D).max()
        np.testing.assert_allclose(got,[expected],atol=1e-10)
        uv['1'][0,1]+=20
        got=anchor_pair_errors(np.array([[0,0]]),'0','1',ids,np.array([xyz]),uv,records,K,D)
        self.assertAlmostEqual(got[0],20.)


if __name__=='__main__':
    unittest.main()
