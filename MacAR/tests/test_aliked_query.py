import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aliked_query import ALIKEDQuery
from guided_matching import spatial_indices


class FakeALIKED:
    preprocess_conf = {'resize': None}

    def __init__(self):
        self.dense_calls = 0
        self.descriptor_batches = []
        self.keypoints = torch.tensor([[-.9, -.8], [-.8, -.7], [-.7, -.6],
                                       [.8, -.8], [-.8, .8], [.8, .8]])
        self.scores = torch.arange(6., 0., -1.)

    def extract_dense_map(self, image):
        self.dense_calls += 1
        return image, image[:, :1]

    def dkd(self, _score_map):
        return [self.keypoints], [self.scores], None

    def desc_head(self, _feature_map, keypoints):
        self.descriptor_batches.append(len(keypoints[0]))
        return [torch.cat((keypoints[0], keypoints[0].sum(-1, keepdim=True)), -1)], None


class ALIKEDQueryTests(unittest.TestCase):
    def test_subpixel_coordinates_and_descriptor_alignment_survive_resize(self):
        model = FakeALIKED()
        query = ALIKEDQuery(torch, model, torch.zeros(3, 30, 40), 80)
        expected = (torch.tensor([79, 59])*(model.keypoints+1)/2+.5)/2-.5
        np.testing.assert_allclose(query.points_numpy, expected.numpy())
        indices = spatial_indices(expected.numpy(), model.scores.numpy(), 3)
        features = query.features(3)
        np.testing.assert_allclose(features['keypoints'][0], expected[indices])
        np.testing.assert_allclose(features['descriptors'][0, :, :2], model.keypoints[indices])
        np.testing.assert_allclose(features['image_size'], [[40., 30.]])

    def test_dense_retry_only_computes_missing_descriptors_and_preserves_order(self):
        model = FakeALIKED()
        query = ALIKEDQuery(torch, model, torch.zeros(3, 30, 40), 0)
        first = query.features(3)
        dense = query.features()
        again = query.features(3)
        self.assertEqual(model.dense_calls, 1)
        self.assertEqual(model.descriptor_batches, [3, 3])
        self.assertEqual(query.descriptor_points_computed, 6)
        np.testing.assert_allclose(dense['descriptors'][0, :, :2], model.keypoints)
        np.testing.assert_allclose(first['descriptors'], again['descriptors'])

    def test_under_budget_computes_all_points_once(self):
        model = FakeALIKED()
        query = ALIKEDQuery(torch, model, torch.zeros(3, 30, 40), 0)
        self.assertEqual(query.features(100)['keypoints'].shape[1], 6)
        query.features()
        self.assertEqual(model.descriptor_batches, [6])


if __name__ == '__main__':
    unittest.main()
