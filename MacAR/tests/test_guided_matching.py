import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guided_matching import spatial_indices, subset_features, project_candidates, pair_gate, region_counts, replenishment_indices
from glb_virtual_live import GLBVirtualLiveLocalizer
from glb_virtual_view_pnp import VirtualTransfer
from realtime_pnp_server import LiveResult


class GuidedMatchingTests(unittest.TestCase):
    def test_replenishment_fills_untracked_regions_before_dense_existing_cluster(self):
        candidates=np.array([[10.,10.]]*30+[[90.,90.],[91.,91.],[92.,92.]])
        existing=np.array([[10.,10.]]*70)
        selected=replenishment_indices(candidates,-np.arange(33),existing)
        self.assertEqual(set(selected[:3]),{30,31,32})
        self.assertEqual(sorted(selected.tolist()),list(range(33)))
        self.assertEqual(len(replenishment_indices(np.empty((0,2)),[],existing)),0)
    def test_region_counts_identify_lower_right_without_saving_pixels(self):
        counts=region_counts([[1,1],[99,99],[98,98],[100,50],[float('nan'),2]],(0,0,100,100))
        self.assertEqual(len(counts),16)
        self.assertEqual(counts[0],1)
        self.assertEqual(counts[-1],2)
        self.assertEqual(sum(counts),3)
        self.assertEqual(region_counts([],(0,0,100,100)),[0]*16)
    def test_pose_distance_is_invariant_to_column_vector_shape(self):
        first = SimpleNamespace(rvec=np.zeros((3, 1)), tvec=np.array([[10.], [20.], [100.]]))
        second = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([10., 20., 100.]))
        angle, distance = GLBVirtualLiveLocalizer._pose_distance(first, second)
        self.assertAlmostEqual(angle, 0.)
        self.assertAlmostEqual(distance, 0.)

    def test_spatial_budget_keeps_sparse_region_instead_of_score_cluster(self):
        points = np.vstack((np.zeros((30, 2)), [[100, 100], [0, 100], [100, 0]]))
        selected = spatial_indices(points, np.arange(len(points))[::-1], 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(set([30, 31, 32]) <= set(selected))

    def test_empty_and_under_budget_do_not_change_indices(self):
        self.assertEqual(len(spatial_indices(np.empty((0, 2)), None, 32)), 0)
        np.testing.assert_array_equal(spatial_indices([[1, 2], [3, 4]], None, 32), [0, 1])

    def test_addition_priority_spreads_candidates_without_existing_points(self):
        points=np.vstack((np.zeros((30,2)),[[100,100],[0,100],[100,0]]))
        selected=replenishment_indices(points,np.arange(len(points))[::-1],np.empty((0,2)))
        self.assertTrue(set([30,31,32]) <= set(selected[:4]))
        self.assertEqual(set(selected),set(range(len(points))))
        self.assertEqual(len(replenishment_indices([],[],np.empty((0,2)))),0)

    def test_projection_rejects_invalid_behind_camera_and_outside_crop(self):
        matrix = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
        xyz = [[0, 0, 10], [0, 0, -10], [10, 0, 10], [np.nan, 0, 10]]
        uv, valid = project_candidates(xyz, np.zeros(3), np.zeros(3), matrix,
                                       np.zeros(5), (100, 100, 3), (25, 25, 75, 75))
        np.testing.assert_array_equal(valid, [True, False, False, False])
        np.testing.assert_allclose(uv[0], [50, 50])

    def test_projection_and_match_gate_use_full_image_pixels(self):
        query = np.array([[5, 10], [20, 10]])
        pairs = np.array([[0, 1], [1, 0]])
        projected = np.array([[300, 300], [105, 210]])
        np.testing.assert_array_equal(pair_gate(query, pairs, projected, 10,
                                                np.array([100, 200])), [True, False])
        self.assertEqual(len(pair_gate(query, np.empty((0, 2), int), projected, 10, 0)), 0)

    def test_feature_subset_slices_descriptors_and_preserves_image_size(self):
        import torch
        original = dict(keypoints=torch.arange(8).reshape(1, 4, 2),
                        descriptors=torch.arange(12).reshape(1, 4, 3),
                        keypoint_scores=torch.arange(4).reshape(1, 4),
                        image_size=torch.tensor([[640., 480.]]))
        reduced = subset_features(torch, original, [3, 1])
        self.assertIs(reduced['image_size'], original['image_size'])
        np.testing.assert_array_equal(reduced['keypoint_scores'].numpy(), [[3, 1]])
        np.testing.assert_array_equal(reduced['descriptors'].numpy(), [[[9, 10, 11], [3, 4, 5]]])
        self.assertEqual(original['keypoints'].shape[1], 4)

    def test_strong_early_exit_requires_prior_quality_and_spatial_support(self):
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.min_inliers = 10
        localizer.max_reprojection_error = 6.
        xyz = np.vstack(([[0, 0, 0], [50, 0, 0], [0, 50, 0]], np.zeros((17, 3))))
        transfers = [VirtualTransfer(i, tuple(p), (0, 0), (0, 0), 1.) for i, p in enumerate(xyz)]
        pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0., 0., 100.]),
                               inlier_indices=np.arange(20), mean_error=1.)
        prior = (pose.rvec, pose.tvec)
        self.assertTrue(localizer._guided_strong_pose(pose, transfers, {'passed': True}, None, prior))
        pose.mean_error = 3.5
        self.assertFalse(localizer._guided_strong_pose(pose, transfers, {'passed': True}, None, prior))
        pose.mean_error = 1.
        self.assertFalse(localizer._guided_strong_pose(pose, transfers, {'passed': False}, None, prior))
        self.assertFalse(localizer._guided_strong_pose(pose, transfers, {'passed': True},
                                                      {'passed': False}, prior))
        pose.tvec = np.array([50., 0., 100.])
        self.assertFalse(localizer._guided_strong_pose(pose, transfers, {'passed': True}, None, prior))
        pose.tvec = prior[1]
        pose.inlier_indices = np.arange(10)
        self.assertFalse(localizer._guided_strong_pose(pose, transfers, {'passed': True}, None, prior))

    def test_failed_filtered_attempt_retries_dense_and_counts_total_work(self):
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.guided_optimization = True
        localizer.pipeline_policy = 'validated'
        def attempt(*args, **kwargs):
            dense = kwargs.get('optimize_guided') is False
            localizer.last_diagnostics = dict(status='PASS' if dense else 'FAIL',
                timings_ms=dict(total=20. if dense else 10., query_features=3.), candidates=[])
            result = LiveResult('PASS' if dense else 'FAIL', '', 20, 20, 20, 1., 1.,
                (0., 0., 0.), (0., 0., 100.), (0., 0., -100.),
                20. if dense else 10., 'HIDDEN', 1., 6., 3.)
            return result, None
        localizer._localize_locked = Mock(side_effect=attempt)
        result, _ = localizer._guided_attempts(None, None, None, [(1, 0, 0)], np.zeros(3), np.zeros(3))
        self.assertEqual(result.status, 'PASS')
        self.assertEqual(result.processing_ms, 30.)
        self.assertEqual(result.matching_ms, 12.)
        self.assertEqual(localizer.last_diagnostics['timings_ms']['total'], 30.)
        self.assertFalse(localizer._localize_locked.call_args.kwargs['optimize_guided'])
        first_workspace = localizer._localize_locked.call_args_list[0].kwargs['query_workspace']
        self.assertIs(first_workspace, localizer._localize_locked.call_args.kwargs['query_workspace'])

    def test_guided_baseline_does_not_retry(self):
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.guided_optimization = False
        localizer.pipeline_policy = 'validated'
        localizer._localize_locked = Mock(return_value=(SimpleNamespace(status='FAIL'), None))
        localizer._guided_attempts(None, None, None, [], None, None)
        localizer._localize_locked.assert_called_once()


if __name__ == '__main__':
    unittest.main()
