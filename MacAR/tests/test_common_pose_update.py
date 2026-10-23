"""A shared current-frame correspondence pool must arbitrate all live poses."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_quality import assess_pose
from realtime_pnp_server import TrackingSeed


class CommonPoseUpdateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(62)
        cv2.setRNGSeed(62)
        self.matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
        self.shape = (480, 640, 3)
        self.gray = np.zeros(self.shape[:2], np.uint8)
        self.engine = SimpleNamespace(min_inliers=10, ransac_threshold=4.,
            min_inlier_ratio=.2, max_reprojection_error=3.)
        self.engine.validate_pose = lambda xyz, uv, r, t, ids, shape, k, d: assess_pose(
            xyz, uv, r, t, ids, shape, k, d, 10, .2, 3., max_p95_error=4.5)

    def seed(self, count, *, shift=0., extent=100., xyz=None):
        if xyz is None:
            xyz = self.rng.uniform([-extent, -extent*.7, 400], [extent, extent*.7, 600], (count, 3))
        t = np.array([shift, 0., 0.])
        uv = cv2.projectPoints(xyz, np.zeros(3), t, self.matrix, None)[0].reshape(-1, 2)
        return TrackingSeed(self.gray, uv, xyz, np.zeros(3), t)

    def select(self, old, fresh):
        from replenishment_refinement import select_pose
        return select_pose(old, fresh, self.shape, self.matrix, None, self.engine)

    def test_twenty_new_points_can_join_eighty_old_points_without_doubling(self):
        old, fresh = self.seed(80), self.seed(20)
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertGreater(len(result.image_points), len(old.image_points))
        self.assertGreaterEqual(detail['added'], 10)
        self.assertLess(np.linalg.norm(result.tvec), .01)
        self.assertEqual({h['name'] for h in detail['hypotheses']}, {'old', 'fresh', 'joint'})
        self.assertEqual(len({h['evaluated_points'] for h in detail['hypotheses']}), 1)
        json.dumps(detail, allow_nan=False)

    def test_broad_forty_new_points_beat_a_hundred_wrong_clustered_old_points(self):
        old, fresh = self.seed(100, shift=80., extent=5.), self.seed(40)
        result, detail, _ = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertLess(np.linalg.norm(result.tvec), .01)
        self.assertGreaterEqual(detail['fresh_inliers'], 30)
        self.assertLess(detail['fresh_inliers'], 2*len(old.image_points))

    def test_drifted_old_observation_of_same_xyz_does_not_veto_fresh_observation(self):
        fresh = self.seed(80)
        old = self.seed(13, shift=-120., xyz=fresh.object_points[:13])
        result, detail, _ = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertLess(np.linalg.norm(result.tvec), .01)
        self.assertGreaterEqual(len(result.image_points), 70)
        self.assertLessEqual(len(result.image_points), 80)

    def test_wrong_fresh_cluster_cannot_displace_broad_old_tracking(self):
        old, fresh = self.seed(80), self.seed(20, shift=80., extent=5.)
        result, detail, _ = self.select(old, fresh)
        self.assertIsNone(result)
        self.assertFalse(detail['accepted'])
        self.assertFalse(detail.get('conflict', False))
        self.assertEqual(detail['reason'], 'no_fresh_support')

    def correctable_old_pose(self):
        # A small rotation around the object moves the estimated camera centre
        # >30 model units while both poses still pass image-space QC.
        xyz = np.array([(x, y, self.rng.uniform(480., 520.))
                        for x in np.linspace(-100., 100., 10)
                        for y in np.linspace(-70., 70., 10)])
        new_ids = np.arange(100) % 5 == 0
        old = self.seed(80, xyz=xyz[~new_ids])
        fresh = self.seed(20, xyz=xyz[new_ids])
        rotation = np.array([0., .08, 0.])
        centre = np.array([0., 0., 500.])
        old = replace(old, rvec=rotation,
                      tvec=centre-cv2.Rodrigues(rotation)[0]@centre)
        return old, fresh, self.seed(100, xyz=xyz)

    def test_qc_valid_old_pose_distance_does_not_veto_agreeing_new_pnp_solutions(self):
        for threshold in (4., 8.):
            with self.subTest(ransac_threshold=threshold):
                self.engine.ransac_threshold = threshold
                old, fresh, _ = self.correctable_old_pose()
                result, detail, pool = self.select(old, fresh)
                baseline = next(h for h in detail['hypotheses'] if h['name'] == 'old')
                self.assertTrue(baseline['quality']['passed'])
                self.assertGreater(np.linalg.norm(old.tvec), 30.)
                self.assertTrue(detail['accepted'], detail)
                self.assertFalse(detail.get('conflict', False))
                self.assertIn(detail['selected'], ('fresh', 'joint'))
                self.assertLess(np.linalg.norm(result.tvec), .01)
                self.assertGreaterEqual(len(result.image_points), 95)
                from replenishment_refinement import final_pose_check
                ok, reason = final_pose_check(result, pool, self.shape,
                                             self.matrix, None, self.engine)
                self.assertTrue(ok, reason)

    def test_final_current_check_does_not_reintroduce_old_pose_distance_veto(self):
        from replenishment_refinement import final_pose_check
        old, fresh, correct = self.correctable_old_pose()
        _, _, pool = self.select(old, fresh)
        ok, detail = final_pose_check(correct, pool, self.shape,
                                     self.matrix, None, self.engine)
        self.assertTrue(ok, detail)

    def test_incompatible_new_and_joint_pnp_select_the_strongest_current_support(self):
        old, fresh = self.seed(60, shift=-60.), self.seed(50, shift=60.)
        # Some new observations support the winning branch. Without these,
        # no_fresh_support correctly keeps the existing track unchanged.
        fresh = replace(fresh, image_points=np.vstack((fresh.image_points, old.image_points[:4])),
                        object_points=np.vstack((fresh.object_points, old.object_points[:4])))
        result, detail, _ = self.select(old, fresh)
        self.assertIsNotNone(result)
        self.assertTrue(detail['accepted'])
        self.assertEqual(detail['reason'], 'common_support_selected')
        current = [h for h in detail['hypotheses'] if h['name'] != 'old']
        self.assertEqual(len(current), 2)
        self.assertGreater(abs(current[0]['pose']['tvec'][0]-current[1]['pose']['tvec'][0]), 100.)
        from spatial_pose_scoring import registered_hypothesis_rank
        strongest = max(detail['hypotheses'],
            key=lambda h: registered_hypothesis_rank((None, None, h['quality'])))
        self.assertEqual(detail['selected'], strongest['name'])
        np.testing.assert_allclose(result.tvec, strongest['pose']['tvec'])

    def test_final_check_uses_current_support_not_distance_from_other_hypotheses(self):
        from replenishment_refinement import final_pose_check
        old, fresh = self.seed(50, shift=-60.), self.seed(50, shift=60.)
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertIn(detail['selected'], ('fresh', 'joint'))
        # The final candidate passes ordinary and common-pool QC and retains
        # baseline spatial support. Distance from another hypothesis is no veto.
        ok, reason = final_pose_check(old, pool, self.shape, self.matrix, None, self.engine)
        self.assertTrue(ok, reason)
        self.assertEqual(reason['reason'], 'final_common_support_verified')

    def test_duplicate_reference_observations_cannot_inflate_votes_or_added_count(self):
        old = self.seed(80)
        fresh = TrackingSeed(self.gray, np.tile(old.image_points, (10, 1)),
            np.tile(old.object_points, (10, 1)), old.rvec, old.tvec)
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertLessEqual(len(pool.image_points), 80)
        self.assertLessEqual(len(result.image_points), 80)
        self.assertEqual(detail['added'], 0)

    def test_same_pixel_with_two_depth_assignments_counts_only_once(self):
        old = self.seed(40)
        fresh = self.seed(40, xyz=old.object_points*2.)
        result, detail, _ = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertLessEqual(len(result.image_points), 40)
        self.assertLessEqual(max(h['inliers'] for h in detail['hypotheses']), 40)

    def test_nearby_but_distinct_features_are_not_removed_as_duplicate_landmarks(self):
        old = self.seed(20)
        fresh = self.seed(20, xyz=old.object_points+[2., 0., 0.])
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        self.assertEqual(len(pool.image_points), 40)
        self.assertEqual(len(result.image_points), 40)
        self.assertEqual(detail['added'], 20)

    def test_final_check_rejects_loss_of_qc_without_requiring_forty_points(self):
        from replenishment_refinement import final_pose_check
        old, fresh = self.seed(13), self.seed(20)
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        ok, _ = final_pose_check(result, pool, self.shape, self.matrix, None, self.engine)
        self.assertTrue(ok)
        weak = TrackingSeed(self.gray, result.image_points[:4], result.object_points[:4],
                            result.rvec, result.tvec)
        ok, reason = final_pose_check(weak, pool, self.shape, self.matrix, None, self.engine)
        self.assertFalse(ok)
        self.assertEqual(reason['reason'], 'final_current_qc')

    def test_nonfinite_and_outside_image_points_are_not_accepted(self):
        old, fresh = self.seed(30), self.seed(15)
        fresh.image_points[:8] = np.nan
        fresh.image_points[8:] = [-10., -10.]
        result, detail, _ = self.select(old, fresh)
        self.assertIsNone(result)
        self.assertFalse(detail['accepted'])

    def test_final_subset_cannot_reduce_support_below_old_pose(self):
        from replenishment_refinement import final_pose_check
        old, fresh = self.seed(80), self.seed(20)
        result, detail, pool = self.select(old, fresh)
        self.assertTrue(detail['accepted'])
        reduced = TrackingSeed(self.gray, result.image_points[:20], result.object_points[:20],
                               result.rvec, result.tvec)
        ok, reason = final_pose_check(reduced, pool, self.shape, self.matrix, None, self.engine)
        self.assertFalse(ok)
        self.assertEqual(reason['reason'], 'final_support_worse_than_old')


if __name__ == '__main__':
    unittest.main()
