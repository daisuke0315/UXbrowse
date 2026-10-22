import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_observability import rotation_sensitivity


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
        rng = np.random.default_rng(7)
        self.xyz = rng.uniform([-100, -80, 400], [100, 80, 600], (100, 3))

    def sensitivity(self, xyz, rvec=None, tvec=None):
        return rotation_sensitivity(xyz, np.zeros(3) if rvec is None else rvec,
                                    np.zeros(3) if tvec is None else tvec,
                                    self.matrix, None)

    def test_clustered_flat_points_have_worse_sensitivity(self):
        flat = self.xyz.copy()
        flat[:, :2] *= .1
        flat[:, 2] = 500.
        self.assertGreater(self.sensitivity(flat), self.sensitivity(self.xyz)*10)

    def test_units_and_world_frame_do_not_change_local_sensitivity(self):
        expected = self.sensitivity(self.xyz)
        self.assertAlmostEqual(expected, self.sensitivity(self.xyz/1000), places=8)
        rotation = cv2.Rodrigues(np.array([.3, -1., 2.9]))[0]
        translation = np.array([10., 20., -50.])
        world = (self.xyz-translation) @ rotation
        self.assertAlmostEqual(expected, self.sensitivity(world,
                              cv2.Rodrigues(rotation)[0], translation), places=8)

    def test_more_independent_observations_reduce_noise_sensitivity(self):
        twice = np.concatenate([self.xyz, self.xyz])
        self.assertAlmostEqual(self.sensitivity(twice)*np.sqrt(2),
                               self.sensitivity(self.xyz), places=8)

    def test_degenerate_or_invalid_geometry_is_unknown_not_zero(self):
        self.assertIsNone(self.sensitivity(self.xyz[:5]))
        self.assertIsNone(self.sensitivity(np.tile([0., 0., 500.], (10, 1))))
        self.assertIsNone(self.sensitivity(-self.xyz))
        invalid = self.xyz.copy()
        invalid[0, 0] = np.nan
        self.assertIsNone(self.sensitivity(invalid))


if __name__ == '__main__':
    unittest.main()
