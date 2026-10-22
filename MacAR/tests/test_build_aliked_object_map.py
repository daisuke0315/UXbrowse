"""Geometric checks for the registered-photo, object-only ALIKED map."""

import sys
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_aliked_object_map import (TrackUnion, epipolar_inliers, load_scene_transform,
                                     object_mask, triangulate_track, parser)
from offline_pnp_test import ReferenceImage


class ObjectALIKEDMapTests(unittest.TestCase):
    def test_new_map_defaults_to_known_pose_with_lightglue_comparison_available(self):
        self.assertEqual(parser().parse_args([]).matching_method, 'known-pose')
        self.assertEqual(parser().parse_args(['--matching-method', 'lightglue']).matching_method, 'lightglue')

    def setUp(self):
        self.k = np.array(((500., 0., 320.), (0., 500., 240.), (0., 0., 1.)))
        self.d = np.zeros(5)
        self.records = {
            name: ReferenceImage(i, 1, name, (1., 0., 0., 0., *(-center)))
            for i, (name, center) in enumerate((('a', np.array((0., 0., 0.))),
                                                ('b', np.array((1., 0., 0.))),
                                                ('c', np.array((0., 1., 0.)))), 1)
        }
        self.matrices = {1: self.k}
        self.distortions = {1: self.d}

    def project(self, name, xyz):
        return cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3),
                                 np.zeros(3), np.asarray(self.records[name].pose[4:]),
                                 self.k, self.d)[0].reshape(-1, 2)

    def test_projected_ply_mask_covers_object_not_background(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.)))
        mask, box = object_mask((480, 640, 3), points, self.records['a'], self.k,
                                self.d, inset_fraction=.03)
        self.assertIsNotNone(mask)
        x0, y0, x1, y1 = box
        self.assertLess(x0, 320)
        self.assertGreater(x1, 320)
        self.assertNotEqual(mask[240-y0, 320-x0], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_known_pose_epipolar_rejects_wrong_correspondence(self):
        xyz = np.array((.2, -.1, 5.))
        left = self.project('a', xyz)
        right = self.project('b', xyz)
        bad = right + np.array((0., 30.))
        keep = epipolar_inliers(np.vstack((left, left)), np.vstack((right, bad)),
                                self.records['a'], self.records['b'],
                                self.matrices, self.distortions, 2.5)
        np.testing.assert_array_equal(keep, (True, False))

    def test_multiview_triangulation_uses_rc_pose_not_ply_depth(self):
        xyz = np.array((.2, -.1, 5.))
        uv = {name: self.project(name, xyz) for name in self.records}
        nodes = [(name, 0) for name in self.records]
        result, reason = triangulate_track(nodes, uv, self.records, self.matrices,
                                           self.distortions,
                                           np.array(((-1., -1., 4.), (1., 1., 6.))),
                                           2., 3., 0.)
        self.assertIsNone(reason)
        np.testing.assert_allclose(result[0], xyz, atol=1e-8)
        self.assertLess(max(result[1]), 1e-7)

    def test_track_union_rejects_two_features_from_same_image(self):
        tracks = TrackUnion()
        self.assertTrue(tracks.join(('a', 0), ('b', 0)))
        self.assertTrue(tracks.join(('c', 0), ('a', 1)))
        self.assertFalse(tracks.join(('b', 0), ('c', 0)))
        self.assertEqual(sorted(map(len, tracks.components())), [2, 2])

    def test_scene_transform_maps_train_frame_to_colmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scene_transform.json'
            path.write_text(json.dumps({'world_from_train': {
                'matrix_4x4': [[2, 0, 0, 10], [0, 2, 0, 20],
                               [0, 0, 2, 30], [0, 0, 0, 1]]}}))
            matrix = load_scene_transform(path)
        np.testing.assert_allclose(matrix[:3, :3] @ [1, 2, 3] + matrix[:3, 3],
                                   [12, 24, 36])


if __name__ == '__main__':
    unittest.main()
