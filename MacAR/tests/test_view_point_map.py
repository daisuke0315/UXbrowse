import tempfile
import unittest
from pathlib import Path

import numpy as np

from view_point_map import load_point_map


class ViewPointMapTests(unittest.TestCase):
    def test_loads_colmap_points_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Points3D.txt"
            path.write_text(
                "# points\n1 1 2 3 10 20 30 0\n2 -1 4 5 40 50 60 0\n",
                encoding="utf-8",
            )
            points, total, box = load_point_map(path, 10)
        self.assertEqual(total, 2)
        self.assertEqual(len(points), 2)
        self.assertEqual(box.center, (0.0, 3.0, 4.0))
        self.assertEqual(box.size, (2.0, 2.0, 2.0))

    def test_loads_unique_feature_map_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object_features.npz"
            np.savez_compressed(
                path,
                point3d_ids=np.asarray((10, 10, 20)),
                xyz=np.asarray(((1, 2, 3), (1, 2, 3), (5, 6, 7)), dtype=float),
            )
            points, total, box = load_point_map(path, 10)
        self.assertEqual(total, 2)
        self.assertEqual(len(points), 2)
        self.assertEqual(box.center, (3.0, 4.0, 5.0))
        self.assertEqual(box.size, (4.0, 4.0, 4.0))


if __name__ == "__main__":
    unittest.main()
