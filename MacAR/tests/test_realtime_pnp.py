import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_pnp import (  # noqa: E402
    PointCloud3DViewer,
    approximate_camera_matrix,
    resize_for_processing,
    scaled_calibration,
)
from realtime_pnp_server import (  # noqa: E402
    LivePnPError,
    LocalizationDebug,
    colmap_to_rc,
    ray_box_interval,
    rc_to_colmap,
)


class RealtimePnPTests(unittest.TestCase):
    def test_rc_colmap_axis_conversion_round_trip(self):
        rc = np.asarray(
            ((4.5, -201.0, -591.0), (-8.0, 12.0, 30.0)), dtype=np.float64
        )
        expected_colmap = np.asarray(
            ((4.5, 591.0, -201.0), (-8.0, -30.0, 12.0)), dtype=np.float64
        )
        np.testing.assert_array_equal(rc_to_colmap(rc), expected_colmap)
        np.testing.assert_array_equal(colmap_to_rc(expected_colmap), rc)

    def test_ray_box_hit_and_miss(self):
        minimum = np.asarray((-1.0, -1.0, -1.0))
        maximum = np.asarray((1.0, 1.0, 1.0))
        hit = ray_box_interval(
            np.asarray((0.0, 0.0, -3.0)),
            np.asarray((0.0, 0.0, 1.0)),
            minimum,
            maximum,
        )
        self.assertEqual(hit, (2.0, 4.0))
        miss = ray_box_interval(
            np.asarray((3.0, 0.0, -3.0)),
            np.asarray((0.0, 0.0, 1.0)),
            minimum,
            maximum,
        )
        self.assertIsNone(miss)

    def test_approximate_camera_matrix_uses_horizontal_fov(self):
        matrix = approximate_camera_matrix(1920, 1080, 90.0)
        self.assertAlmostEqual(matrix[0, 0], 960.0)
        self.assertAlmostEqual(matrix[1, 1], 960.0)
        self.assertEqual(matrix[0, 2], 960.0)
        self.assertEqual(matrix[1, 2], 540.0)

    def test_calibration_scaling_requires_same_aspect_ratio(self):
        matrix = np.asarray(
            ((1000.0, 0, 960.0), (0, 1000.0, 540.0), (0, 0, 1))
        )
        scaled = scaled_calibration(matrix, (1920, 1080), (1280, 720))
        self.assertAlmostEqual(scaled[0, 0], 1000.0 * 2 / 3)
        with self.assertRaises(LivePnPError):
            scaled_calibration(matrix, (1920, 1080), (1280, 960))

    def test_processing_resize_preserves_aspect_ratio(self):
        frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
        resized, scale = resize_for_processing(frame, 1280)
        self.assertEqual(resized.shape, (960, 1280, 3))
        self.assertAlmostEqual(scale, 2 / 3)
        unchanged, unchanged_scale = resize_for_processing(frame, 0)
        self.assertIs(unchanged, frame)
        self.assertEqual(unchanged_scale, 1.0)

    def test_live_3d_viewer_renders_matches_and_box(self):
        fake_localizer = SimpleNamespace(
            map_data=SimpleNamespace(
                point3d_ids=np.asarray((1, 2, 2, 3)),
                xyz=np.asarray(
                    ((-1, -1, -1), (1, -1, 0), (1, -1, 0), (0, 1, 1)),
                    dtype=np.float64,
                ),
            ),
            mpr_renderer=SimpleNamespace(
                box_minimum=np.asarray((-2.0, -2.0, -2.0)),
                box_maximum=np.asarray((2.0, 2.0, 2.0)),
            ),
        )
        viewer = PointCloud3DViewer(fake_localizer, width=640, height=480)
        debug = LocalizationDebug(
            matched_xyz=np.asarray(((-1, -1, -1), (0, 1, 1)), dtype=np.float64),
            inlier_indices=frozenset((1,)),
            camera_center_model=(0.0, -4.0, 0.0),
            pnp_solved=True,
        )
        image = viewer.render(debug, "PASS")
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertGreater(int(image.max()), 0)


if __name__ == "__main__":
    unittest.main()
