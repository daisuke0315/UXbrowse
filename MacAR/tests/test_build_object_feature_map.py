import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_object_feature_map import (  # noqa: E402
    FeatureMapError,
    Observation,
    build_feature_map,
    match_observations_to_keypoints,
)


class BuildObjectFeatureMapTests(unittest.TestCase):
    def _observation(self, u: float, v: float, point_id: int) -> Observation:
        return Observation(
            1,
            "test.png",
            1,
            u,
            v,
            point_id,
            (float(point_id), 2.0, 3.0),
            (str(point_id), "2", "3"),
            str(u),
            str(v),
        )

    def _write_inputs(
        self,
        directory: Path,
        image: np.ndarray,
        observations,
        *,
        camera_size=None,
    ):
        image_path = directory / "test.png"
        self.assertTrue(cv2.imwrite(str(image_path), image))
        height, width = image.shape[:2]
        camera_width, camera_height = camera_size or (width, height)
        cameras = directory / "Cameras.txt"
        cameras.write_text(
            f"1 PINHOLE {camera_width} {camera_height} 100 100 50 50\n",
            encoding="utf-8",
        )
        csv_path = directory / "object_observations.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("image_id", "image_name", "camera_id", "u", "v", "point3d_id", "X", "Y", "Z")
            )
            for point_id, (u, v) in enumerate(observations, 10):
                writer.writerow((1, "test.png", 1, repr(u), repr(v), point_id, point_id, 2, 3))
        return csv_path, cameras

    def test_nearest_matching_enforces_one_keypoint_per_observation(self):
        observations = [
            self._observation(10.1, 10.0, 10),
            self._observation(10.4, 10.0, 11),
            self._observation(20.0, 20.2, 12),
        ]
        keypoints = [
            SimpleNamespace(pt=(10.0, 10.0)),
            SimpleNamespace(pt=(20.0, 20.0)),
        ]
        matches, unmatched = match_observations_to_keypoints(
            observations, keypoints, 1.0
        )
        self.assertEqual(
            [(match.observation_index, match.keypoint_index) for match in matches],
            [(0, 0), (2, 1)],
        )
        self.assertEqual(unmatched, {1: "SIFT_KEYPOINT_CONFLICT"})

    def test_builds_npz_metadata_and_summary_from_exact_images(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rng = np.random.default_rng(42)
            image = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
            keypoints, descriptors = cv2.SIFT_create().detectAndCompute(image, None)
            self.assertIsNotNone(descriptors)
            unique_positions = []
            seen = set()
            for keypoint in keypoints:
                position = (float(keypoint.pt[0]), float(keypoint.pt[1]))
                rounded = (round(position[0], 5), round(position[1], 5))
                if rounded not in seen:
                    unique_positions.append(position)
                    seen.add(rounded)
                if len(unique_positions) == 3:
                    break
            self.assertEqual(len(unique_positions), 3)
            observations, cameras = self._write_inputs(
                directory, image, unique_positions
            )
            result = build_feature_map(
                observations,
                cameras,
                directory,
                directory / "out",
                max_distance=0.01,
                create_visualizations=True,
            )
            self.assertEqual(result.total_images, 1)
            self.assertEqual(result.processed_images, 1)
            self.assertEqual(result.failed_images, 0)
            self.assertEqual(result.observations_with_descriptor, 3)
            self.assertEqual(result.points_with_descriptor, 3)
            self.assertTrue(result.spatial_qc_path.is_file())
            self.assertEqual(len(result.overlay_paths), 1)

            with np.load(result.features_path) as feature_map:
                self.assertEqual(feature_map["descriptors"].shape, (3, 128))
                self.assertEqual(feature_map["xyz"].shape, (3, 3))
                self.assertEqual(feature_map["point3d_ids"].tolist(), [10, 11, 12])
                self.assertTrue(
                    np.all(feature_map["pixel_distances"] <= 0.01)
                )
            with result.metadata_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["point_observation_count"], "1")
            summary = result.summary_path.read_text(encoding="utf-8")
            self.assertIn("Descriptor acquisition rate: 100.000000%", summary)

    def test_size_mismatch_is_reported_without_resizing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = np.zeros((80, 100), dtype=np.uint8)
            observations, cameras = self._write_inputs(
                directory,
                image,
                [(20.0, 20.0)],
                camera_size=(101, 80),
            )
            result = build_feature_map(
                observations,
                cameras,
                directory,
                directory / "out",
                create_visualizations=False,
            )
            self.assertEqual(result.processed_images, 0)
            self.assertEqual(result.failed_images, 1)
            self.assertEqual(result.observations_with_descriptor, 0)
            with result.failures_path.open(encoding="utf-8", newline="") as source:
                failures = list(csv.DictReader(source))
            self.assertEqual(failures[0]["error_code"], "IMAGE_SIZE_MISMATCH")
            with np.load(result.features_path) as feature_map:
                self.assertEqual(feature_map["descriptors"].shape, (0, 128))

    def test_nonpositive_max_distance_is_rejected(self):
        with self.assertRaisesRegex(FeatureMapError, "positive finite"):
            match_observations_to_keypoints([], [], 0)


if __name__ == "__main__":
    unittest.main()
