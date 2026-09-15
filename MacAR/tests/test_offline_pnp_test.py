import unittest
from types import SimpleNamespace

from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_object_observations import Camera  # noqa: E402
from offline_pnp_test import (  # noqa: E402
    ReferenceImage,
    camera_matrix_and_distortion,
    pose_errors,
    select_unique_point_matches,
)


class OfflinePnPTestTests(unittest.TestCase):
    def test_ratio_test_uses_two_distinct_3d_points(self):
        # The two closest map descriptors belong to the same 3D point. The
        # ratio must compare against the next descriptor from another point.
        neighbors = [[
            SimpleNamespace(trainIdx=0, distance=10.0),
            SimpleNamespace(trainIdx=1, distance=11.0),
            SimpleNamespace(trainIdx=2, distance=20.0),
        ]]
        matches = select_unique_point_matches(
            neighbors, np.asarray((100, 100, 200)), 0.75
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].point3d_id, 100)
        self.assertAlmostEqual(matches[0].ratio, 0.5)

    def test_only_best_query_match_is_kept_for_each_3d_point(self):
        neighbors = [
            [
                SimpleNamespace(trainIdx=0, distance=6.0),
                SimpleNamespace(trainIdx=1, distance=20.0),
            ],
            [
                SimpleNamespace(trainIdx=0, distance=4.0),
                SimpleNamespace(trainIdx=1, distance=20.0),
            ],
        ]
        matches = select_unique_point_matches(
            neighbors, np.asarray((100, 200)), 0.75
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].query_index, 1)

    def test_simple_radial_camera_and_identity_pose_error(self):
        camera = Camera(1, "SIMPLE_RADIAL", 640, 480)
        matrix, distortion = camera_matrix_and_distortion(
            np, camera, ("500", "320", "240", "0.01")
        )
        np.testing.assert_array_equal(
            matrix,
            np.asarray(((500, 0, 320), (0, 500, 240), (0, 0, 1)), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            distortion, np.asarray((0.01, 0, 0, 0, 0), dtype=np.float64)
        )
        reference = ReferenceImage(1, 1, "test.png", (1, 0, 0, 0, 0, 0, 5))
        rotation_error, center_error, center = pose_errors(
            cv2,
            np,
            np.zeros((3, 1), dtype=np.float64),
            np.asarray((0, 0, 5), dtype=np.float64).reshape(3, 1),
            reference,
        )
        self.assertAlmostEqual(rotation_error, 0.0)
        self.assertAlmostEqual(center_error, 0.0)
        self.assertEqual(center, (0.0, 0.0, -5.0))


if __name__ == "__main__":
    unittest.main()
