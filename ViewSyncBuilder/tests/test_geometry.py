import unittest
import numpy as np

from viewsync_builder.geometry import similarity, transform_pose, triangulate


class GeometryTests(unittest.TestCase):
    def test_similarity_has_positive_scale_and_preserves_lps_axes(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 23 * (source @ rotation.T) + [17, -30, 61]
        scale, actual_r, translation = similarity(source, target)
        self.assertAlmostEqual(scale, 23)
        np.testing.assert_allclose(actual_r, rotation, atol=1e-12)
        np.testing.assert_allclose(translation, [17, -30, 61], atol=1e-12)

    def test_world_to_camera_similarity_keeps_projections(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        world_r = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        world_t = np.array([7., 8, 9])
        cam_r, cam_t = transform_pose(rotation, np.array([1., 2, 3]), 4, world_r, world_t)
        x = np.array([3., 4, 5])
        np.testing.assert_allclose(cam_r @ (4 * world_r @ x + world_t) + cam_t,
                                   4 * (rotation @ x + [1, 2, 3]))
        self.assertAlmostEqual(np.linalg.det(cam_r), 1)

    def test_similarity_rejects_collinear_or_reflected_controls(self):
        line = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaises(ValueError):
            similarity(line, line)
        tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            similarity(tetra, tetra * [-1, 1, 1])

    def test_manual_observation_triangulation_and_degeneracy(self):
        p1 = np.column_stack((np.eye(3), [0., 0, 0]))
        p2 = np.column_stack((np.eye(3), [-10., 0, 0]))
        xyz = triangulate([p1, p2], [[.2, .1], [0, .1]])
        np.testing.assert_allclose(xyz, [10, 5, 50], atol=1e-9)
        with self.assertRaises(ValueError):
            triangulate([p1, p1], [[.2, .1], [.2, .1]])


if __name__ == '__main__':
    unittest.main()
