"""Candidate selection must reward spatial evidence, not feature density."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_pose_scoring import spatial_pose_score, registered_hypothesis_rank, comparable_support


def hypothesis(pixels, errors, *, passed=True):
    score = spatial_pose_score(pixels, errors, (0, 0, 80, 80), 4.)
    return (SimpleNamespace(inlier_indices=np.arange(len(pixels))), [],
            dict(passed=passed, spatial_score=score), 'test', 0.)


class SpatialPoseScoringTests(unittest.TestCase):
    def test_dense_region_has_at_most_one_vote(self):
        sparse = spatial_pose_score([(5., 5.)], [.5], (0, 0, 80, 80), 4.)
        dense = spatial_pose_score([(5., 5.)]*100, [.5]*100, (0, 0, 80, 80), 4.)
        self.assertEqual(sparse, dense)
        self.assertEqual(dense['occupied_cells'], 1)
        self.assertAlmostEqual(dense['score'], 1.-(.5/4.)**2)

    def test_broad_support_beats_many_clustered_points(self):
        dense = hypothesis([(5., 5.)]*100, [.2]*100)
        broad = hypothesis([(x, y) for x in (5, 25, 45, 65)
                            for y in (5, 25, 45, 65)], [1.]*16)
        self.assertIs(max([dense, broad], key=registered_hypothesis_rank), broad)

    def test_equal_regions_prefer_lower_error_regardless_of_count(self):
        uv = [(5., 5.), (25., 25.)]
        dense = hypothesis(uv*50, [2.]*100)
        precise = hypothesis(uv, [.3, .3])
        self.assertGreater(registered_hypothesis_rank(precise), registered_hypothesis_rank(dense))

    def test_identical_spatial_evidence_has_identical_rank_despite_raw_count(self):
        sparse = hypothesis([(5., 5.)], [.5])
        dense = hypothesis([(5., 5.)]*100, [.5]*100)
        self.assertEqual(registered_hypothesis_rank(sparse), registered_hypothesis_rank(dense))

    def test_dense_cell_cannot_dominate_balanced_error(self):
        uv = [(5., 5.), (25., 25.)]
        a = spatial_pose_score(uv, [.5, 2.], (0, 0, 80, 80), 4.)
        b = spatial_pose_score([uv[0]]*200+[uv[1]], [.5]*200+[2.], (0, 0, 80, 80), 4.)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['balanced_error_px'], 1.25)

    def test_grid_is_fixed_to_common_roi_not_inlier_bbox(self):
        a = spatial_pose_score([(21, 21), (22, 22), (23, 23), (24, 24)], [0.]*4,
                               (20, 20, 100, 100), 4.)
        self.assertEqual(a['occupied_cells'], 1)
        shifted = spatial_pose_score([(1, 1), (2, 2), (3, 3), (4, 4)], [0.]*4,
                                     (0, 0, 80, 80), 4.)
        self.assertEqual(a['score'], shifted['score'])

    def test_qc_failure_never_beats_passed_candidate(self):
        passed = hypothesis([(5, 5)], [2.])
        failed = hypothesis([(x, y) for x in range(5, 80, 10)
                             for y in range(5, 80, 10)], [0.]*64, passed=False)
        self.assertGreater(registered_hypothesis_rank(passed), registered_hypothesis_rank(failed))

    def test_competing_support_uses_same_score_and_excludes_failed_qc(self):
        dense = hypothesis([(5, 5)]*100, [.1]*100)
        broad = hypothesis([(5, 5), (25, 25)], [.1, .1])
        self.assertTrue(comparable_support(broad, dense))
        self.assertFalse(comparable_support(dense, broad))
        broad[2]['passed'] = False
        self.assertFalse(comparable_support(broad, dense))

    def test_invalid_pixels_and_errors_cannot_create_support(self):
        score = spatial_pose_score([(5, 5), (-1, 5), (90, 90), (np.nan, 3), (15, 15)],
                                   [4., 0., 0., 0., np.nan], (0, 0, 80, 80), 4.)
        self.assertEqual(score['score'], 0.)
        self.assertEqual(score['occupied_cells'], 1)
        json.dumps(score, allow_nan=False)
        empty = spatial_pose_score([], [], (0, 0, 80, 80), 4.)
        self.assertEqual(empty['occupied_cells'], 0)
        json.dumps(empty, allow_nan=False)

    def test_roi_and_threshold_must_be_valid(self):
        for roi, threshold in [((0, 0, 0, 80), 4.), ((0, 0, 80, 80), 0.)]:
            with self.assertRaises(ValueError):
                spatial_pose_score([(5, 5)], [.1], roi, threshold)

    def test_residual_is_normalized_by_current_ransac_threshold(self):
        a = spatial_pose_score([(5, 5)], [1.], (0, 0, 80, 80), 4.)
        b = spatial_pose_score([(5, 5)], [2.], (0, 0, 80, 80), 8.)
        self.assertEqual(a['score'], b['score'])


if __name__ == '__main__':
    unittest.main()
