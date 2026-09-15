import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_point_filter import (  # noqa: E402
    FilterError,
    filter_points,
    make_focus_box,
    parse_point_line,
)


HEADER = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 4
"""


class ObjectPointFilterTests(unittest.TestCase):
    def _input(self, directory: Path) -> Path:
        path = directory / "Points3D.txt"
        path.write_text(
            HEADER
            + "10 0 0 0 255 0 0 0.1 1 2\n"
            + "11 1 0 0 0 255 0 0.2 3 4 5 6\n"
            + "12 1.1 0 0 0 0 255 0.3\n"
            + "99 9 9 9 1 2 3 0.4 7 8\n",
            encoding="utf-8",
        )
        return path

    def test_aabb_keeps_original_rows_and_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_path = self._input(directory)
            box = make_focus_box((0, 0, 0), (2, 2, 2))
            result = filter_points(input_path, directory / "out", box)
            self.assertEqual((result.input_points, result.selected_points, result.rejected_points), (4, 2, 2))
            output = result.filtered_path.read_text(encoding="utf-8")
            self.assertIn("# Number of points: 2\n", output)
            self.assertIn("10 0 0 0 255 0 0 0.1 1 2\n", output)
            self.assertIn("11 1 0 0 0 255 0 0.2 3 4 5 6\n", output)
            self.assertNotIn("12 1.1 0 0", output)
            self.assertEqual(result.selected_ids_path.read_text(encoding="utf-8"), "10\n11\n")
            self.assertIn("QC3 selected COLMAP records unchanged: PASS", result.summary_path.read_text())

    def test_obb_rotation_uses_inverse_for_containment(self):
        # A 4x1x1 local box rotated +90 degrees around Z is long along world Y.
        box = make_focus_box((0, 0, 0), (4, 1, 1), (0, 0, 90))
        self.assertTrue(box.contains((0, 1.9, 0)))
        self.assertFalse(box.contains((1.9, 0, 0)))
        self.assertTrue(box.contains((0.5, 0, 0)))  # Boundary is included.

    def test_rotation_matrix_is_supported(self):
        box = make_focus_box(
            (0, 0, 0),
            (4, 1, 1),
            rotation_matrix=(0, -1, 0, 1, 0, 0, 0, 0, 1),
        )
        self.assertTrue(box.contains((0, 1.9, 0)))
        self.assertFalse(box.contains((1.9, 0, 0)))

    def test_rc_coordinates_are_rotated_into_colmap_world(self):
        box = make_focus_box(
            (4.59766, -201.5511, -591.8713),
            (209.423, 176.29451, 261.861969),
            input_coordinate_system="rc",
        )
        self.assertEqual(box.center, (4.59766, 591.8713, -201.5511))
        self.assertEqual(
            box.rotation,
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        )
        self.assertTrue(box.contains(box.center))
        # Half of RC Sy maps along COLMAP Z, while half of RC Sz maps along -Y.
        self.assertTrue(box.contains((4.59766, 591.8713, -201.5511 + 176.29451 / 2)))
        self.assertFalse(box.contains((4.59766, 591.8713, -201.5511 + 176.29451 / 2 + 0.01)))

    def test_zero_selection_does_not_replace_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_path = self._input(directory)
            output_dir = directory / "out"
            output_dir.mkdir()
            existing = output_dir / "filtered_Points3D.txt"
            existing.write_text("previous result\n", encoding="utf-8")
            box = make_focus_box((100, 100, 100), (1, 1, 1))
            with self.assertRaisesRegex(FilterError, "No points found inside focus region"):
                filter_points(input_path, output_dir, box)
            self.assertEqual(existing.read_text(encoding="utf-8"), "previous result\n")

    def test_invalid_colmap_line_is_rejected(self):
        with self.assertRaisesRegex(FilterError, "TRACK"):
            parse_point_line("1 0 0 0 1 2 3 0.1 9", 7)

    def test_nonpositive_size_is_rejected(self):
        with self.assertRaisesRegex(FilterError, "greater than 0"):
            make_focus_box((0, 0, 0), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
