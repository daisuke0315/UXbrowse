import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference_guided_pnp import (  # noqa: E402
    PoseRefiner,
    ReferenceResult,
    ReferenceScore,
    TransferredMatch,
    deduplicate_matches,
    merge_matches,
    reference_pose,
)


class ReferenceGuidedPnPTests(unittest.TestCase):
    def test_deduplicate_matches_resolves_query_and_point_conflicts(self):
        matches = deduplicate_matches(
            (
                (
                    TransferredMatch(0, 0, 100, 20.0, 0.5),
                    TransferredMatch(1, 1, 100, 10.0, 0.4),
                ),
                (TransferredMatch(0, 2, 200, 9.0, 0.3),),
            )
        )
        self.assertEqual(
            [(match.query_index, match.point3d_id) for match in matches],
            [(0, 200), (1, 100)],
        )

    def test_guided_merge_prefers_guided_correspondences(self):
        reference = ReferenceScore(1, "one.png", 10, 2.0)
        result = ReferenceResult(
            1,
            reference,
            1,
            1,
            1,
            1,
            [TransferredMatch(0, 0, 100, 10.0, 0.5)],
            guided_transferred=[TransferredMatch(1, 1, 200, 8.0, 0.4)],
        )
        ordinary, _ = merge_matches((result,))
        guided, _ = merge_matches((result,), use_guided=True)
        self.assertEqual(ordinary[0].point3d_id, 100)
        self.assertEqual(guided[0].point3d_id, 200)

    def test_reference_pose_converts_colmap_quaternion(self):
        pose = reference_pose(
            SimpleNamespace(pose=(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
        )
        np.testing.assert_allclose(pose.rvec, np.zeros((3, 1)), atol=1e-12)
        np.testing.assert_allclose(pose.tvec.ravel(), (1.0, 2.0, 3.0))

    def test_pose_refiner_recovers_clean_correspondences(self):
        xyz = np.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.2),
                (1.0, 1.0, -0.1),
                (-1.0, 1.0, 0.3),
                (0.0, 0.0, 1.0),
                (0.5, -0.4, 0.7),
            ),
            dtype=np.float64,
        )
        point_ids = np.arange(100, 106, dtype=np.int64)
        matrix = np.asarray(
            ((700.0, 0.0, 320.0), (0.0, 700.0, 240.0), (0.0, 0.0, 1.0))
        )
        rvec = np.asarray((0.05, -0.08, 0.03), dtype=np.float64).reshape(3, 1)
        tvec = np.asarray((0.1, -0.2, 7.0), dtype=np.float64).reshape(3, 1)
        uv, _ = cv2.projectPoints(xyz, rvec, tvec, matrix, None)
        keypoints = [cv2.KeyPoint(float(u), float(v), 3) for u, v in uv.reshape(-1, 2)]
        matches = [
            TransferredMatch(index, index, int(point_id), 1.0, 0.1)
            for index, point_id in enumerate(point_ids)
        ]
        solved = PoseRefiner(
            point_ids,
            xyz,
            matrix,
            ransac_threshold=2.0,
            iterations=1000,
        ).solve(keypoints, matches)
        self.assertEqual(solved.inliers, len(matches))
        self.assertLess(solved.mean_error, 1e-3)


if __name__ == "__main__":
    unittest.main()
