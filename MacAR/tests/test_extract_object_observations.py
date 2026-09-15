import csv
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_object_observations import (  # noqa: E402
    ImageCount,
    ObservationError,
    choose_qc_images,
    extract_observations,
    resolve_image_path,
)


class ExtractObjectObservationsTests(unittest.TestCase):
    def _fixture(self, directory: Path):
        cameras = directory / "Cameras.txt"
        cameras.write_text(
            "# cameras\n1 PINHOLE 100 80 50 50 50 40\n",
            encoding="utf-8",
        )
        selected = directory / "selected_point_ids.txt"
        selected.write_text("10\n11\n", encoding="utf-8")
        points = directory / "filtered_Points3D.txt"
        points.write_text(
            "# points\n"
            "10 1.25 2.5 3.75 1 2 3 0.1 1 0\n"
            "11 -4 -5 -6 4 5 6 0.2 2 0\n",
            encoding="utf-8",
        )
        images = directory / "Images.txt"
        images.write_text(
            "# images\n"
            "1 1 0 0 0 10 20 30 1 first image.png\n"
            "10.5 20.25 10 30 40 -1 90 20 999\n"
            "2 1 0 0 0 11 21 31 1 second.png\n"
            "50 60 11\n"
            "3 1 0 0 0 12 22 32 1 empty.png\n"
            "\n",
            encoding="utf-8",
        )
        return images, cameras, selected, points

    def test_extracts_joined_rows_pose_counts_and_qc(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            images, cameras, selected, points = self._fixture(directory)
            result = extract_observations(
                images,
                cameras,
                selected,
                points,
                directory / "out",
                create_visualizations=False,
            )
            self.assertEqual(result.selected_points, 2)
            self.assertEqual(result.total_images, 3)
            self.assertEqual(result.images_with_observations, 2)
            self.assertEqual(result.total_observations, 2)
            self.assertEqual(result.points_without_observations, 0)
            self.assertEqual((result.minimum_per_image, result.maximum_per_image), (0, 1))

            with result.observations_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["image_name"], "first image.png")
            self.assertEqual(rows[0]["point3d_id"], "10")
            self.assertEqual((rows[0]["X"], rows[0]["Y"], rows[0]["Z"]),
                             ("1.25", "2.5", "3.75"))
            self.assertEqual((rows[0]["qw"], rows[0]["tx"]), ("1", "10"))

            summary = result.summary_path.read_text(encoding="utf-8")
            self.assertIn("QC1 all observation IDs are selected: PASS", summary)
            self.assertIn("empty.png, 0", summary)

            with result.counts_path.open(newline="", encoding="utf-8") as source:
                counts = list(csv.DictReader(source))
            self.assertEqual([row["observation_count"] for row in counts], ["1", "1", "0"])

    def test_out_of_range_observation_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            images, cameras, selected, points = self._fixture(directory)
            images.write_text(
                "1 1 0 0 0 0 0 0 1 bad.png\n100 20 10\n",
                encoding="utf-8",
            )
            output_dir = directory / "out"
            output_dir.mkdir()
            existing = output_dir / "object_observations.csv"
            existing.write_text("previous\n", encoding="utf-8")
            result = extract_observations(
                images,
                cameras,
                selected,
                points,
                output_dir,
                create_visualizations=False,
            )
            self.assertEqual(result.out_of_bounds_observations, 1)
            with result.out_of_bounds_path.open(newline="", encoding="utf-8") as source:
                invalid_rows = list(csv.DictReader(source))
            self.assertEqual(len(invalid_rows), 1)
            self.assertEqual(invalid_rows[0]["point3d_id"], "10")
            self.assertIn("QC3 all selected (u,v) inside camera dimensions: FAIL",
                          result.summary_path.read_text(encoding="utf-8"))

    def test_step1_output_id_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            images, cameras, selected, points = self._fixture(directory)
            selected.write_text("10\n11\n12\n", encoding="utf-8")
            with self.assertRaisesRegex(ObservationError, "Step 1 outputs are inconsistent"):
                extract_observations(
                    images,
                    cameras,
                    selected,
                    points,
                    directory / "out",
                    create_visualizations=False,
                )

    def test_qc_image_selection_uses_low_median_and_high_positive_counts(self):
        counts = [
            ImageCount(1, "one.png", 1, 0),
            ImageCount(2, "two.png", 1, 2),
            ImageCount(3, "three.png", 1, 5),
            ImageCount(4, "four.png", 1, 10),
        ]
        selected = choose_qc_images(counts)
        self.assertEqual([(label, item.image_id) for label, item in selected],
                         [("low", 2), ("medium", 3), ("high", 4)])

    def test_image_template_resolves_numeric_colmap_name(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = directory / "output_image2_0001.jpg"
            image.write_bytes(b"not decoded in this test")
            resolved = resolve_image_path(
                "00000.png", directory, "output_image2_{index:04d}.jpg", 1
            )
            self.assertEqual(resolved, image)

    def test_known_rc_export_preamble_in_cameras_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            images, cameras, selected, points = self._fixture(directory)
            cameras.write_text(
                "The files Cameras.txt, Images.txt and Points3D.txt are in the same folder.\n"
                "1 PINHOLE 100 80 50 50 50 40\n",
                encoding="utf-8",
            )
            result = extract_observations(
                images,
                cameras,
                selected,
                points,
                directory / "out",
                create_visualizations=False,
            )
            self.assertEqual(result.total_observations, 2)


if __name__ == "__main__":
    unittest.main()
