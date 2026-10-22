"""Known camera geometry restricts descriptors before correspondence selection."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from known_pose_matching import prepare_features, match_features, pair_cache_key
from offline_pnp_test import ReferenceImage


class KnownPoseMatchingTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
        self.xyz = np.array([[.1, 0., 5.], [.4, 0., 6.], [.7, 0., 7.], [-.2, 0., 4.]])
        self.descriptors = np.eye(4, dtype=np.float32)
        self.a = ReferenceImage(1, 1, 'a', (1., 0., 0., 0., 0., 0., 0.))
        self.b = ReferenceImage(2, 1, 'b', (1., 0., 0., 0., -1., 0., 0.))

    def prepare(self, rec, xyz=None, descriptors=None, distortion=None):
        xyz = self.xyz if xyz is None else xyz
        desc = self.descriptors if descriptors is None else descriptors
        distortion = np.zeros(5) if distortion is None else distortion
        uv = cv2.projectPoints(xyz, np.zeros(3), np.asarray(rec.pose[4:]),
                               self.K, distortion)[0].reshape(-1, 2)
        return prepare_features(uv, desc, rec, self.K, distortion)

    def test_geometry_rejects_better_looking_impostor_before_nearest_neighbour(self):
        left = self.prepare(self.a)
        xyz = np.vstack([self.xyz, [.1, 1., 5.]])
        desc = np.vstack([self.descriptors, self.descriptors[0]])
        desc[0] = [.99, .01, 0., 0.]
        right = self.prepare(self.b, xyz, desc)
        pairs, scores, info = match_features(left, right)
        np.testing.assert_array_equal(pairs, np.column_stack([np.arange(4), np.arange(4)]))
        self.assertEqual(info['candidate_pairs'], 16)
        self.assertEqual(info['total_pairs'], 20)
        self.assertEqual(len(scores), 4)

    def test_undistortion_and_global_pixels_keep_true_matches(self):
        distortion = np.array([.15, -.03, .002, -.001, 0.])
        pairs, _, _ = match_features(self.prepare(self.a, distortion=distortion),
                                     self.prepare(self.b, distortion=distortion))
        self.assertEqual(len(pairs), 4)

    def test_two_candidates_required_and_equal_descriptors_rejected(self):
        first = self.prepare(self.a, self.xyz[:1], self.descriptors[:1])
        second = self.prepare(self.b, self.xyz[:1], self.descriptors[:1])
        self.assertEqual(len(match_features(first, second)[0]), 0)
        same = np.tile([1., 0., 0., 0.], (4, 1))
        self.assertEqual(len(match_features(self.prepare(self.a, descriptors=same),
                                          self.prepare(self.b, descriptors=same))[0]), 0)

    def test_mutual_nearest_prevents_two_features_claiming_one_point(self):
        desc = self.descriptors.copy(); desc[1] = desc[0]
        pairs, _, _ = match_features(self.prepare(self.a, descriptors=desc), self.prepare(self.b))
        self.assertEqual(len(set(pairs[:, 1])), len(pairs))
        self.assertEqual(sum(pairs[:, 1] == 0), 1)

    def test_same_camera_center_is_degenerate_not_a_free_match(self):
        pairs, _, info = match_features(self.prepare(self.a), self.prepare(self.a))
        self.assertEqual(len(pairs), 0)
        self.assertEqual(info['reason'], 'degenerate_baseline')

    def test_invalid_features_and_empty_images_are_safe(self):
        right = self.prepare(self.b)
        right.descriptors[0] = np.nan
        pairs, _, _ = match_features(self.prepare(self.a), right)
        self.assertNotIn(0, pairs[:, 1])
        empty = prepare_features(np.empty((0, 2)), np.empty((0, 4)), self.a, self.K, None)
        self.assertEqual(len(match_features(empty, right)[0]), 0)

    def test_block_size_does_not_change_matches_or_confidence(self):
        first, second = self.prepare(self.a), self.prepare(self.b)
        small = match_features(first, second, block_size=1)
        big = match_features(first, second, block_size=128)
        np.testing.assert_array_equal(small[0], big[0])
        np.testing.assert_allclose(small[1], big[1])

    def test_cache_separates_methods_thresholds_and_camera_geometry(self):
        default = pair_cache_key('known-pose', 'camera', 'left', 'right')
        self.assertNotEqual(default, pair_cache_key('lightglue', 'camera', 'left', 'right'))
        self.assertNotEqual(default, pair_cache_key('known-pose', 'camera2', 'left', 'right'))
        self.assertNotEqual(default, pair_cache_key('known-pose', 'camera', 'left', 'right', 3.))

    def test_invalid_parameters_fail_explicitly(self):
        for options in [dict(epipolar_px=0), dict(ratio=1.), dict(max_distance=0), dict(block_size=0)]:
            with self.assertRaises(ValueError):
                match_features(self.prepare(self.a), self.prepare(self.b), **options)


if __name__ == '__main__':
    unittest.main()
