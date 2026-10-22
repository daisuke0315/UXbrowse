import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aliked_lightglue_pnp import (  # noqa: E402
    Observation,
    associate_reference_observations,
)
from generate_glb_reference_views import (  # noqa: E402
    depth_to_world_xyz,
    look_at_pose,
)
from glb_virtual_view_pnp import sample_xyz_buffer  # noqa: E402
from glb_virtual_view_pnp import VirtualTransfer  # noqa: E402
from glb_virtual_live import (GLBVirtualLiveLocalizer,
                              object_scale_geometric_filter)  # noqa: E402
from unittest.mock import patch


class LearnedMatchingTests(unittest.TestCase):
    def test_object_geometry_uses_equal_crop_scale_without_changing_source_coordinates(self):
        query = np.asarray([[2000., 1000.]], np.float32)
        reference = np.asarray([[200., 100.]], np.float32)
        scores = np.asarray([.9], np.float32)
        with patch('glb_virtual_live.geometric_filter',
                   side_effect=lambda q, r, s, threshold: (q, r, s)) as matcher:
            output = object_scale_geometric_filter(query, reference, scores, 3.,
                                                  (4000, 3000, 3), (400, 300, 3))
        np.testing.assert_allclose(matcher.call_args.args[0], [[512., 256.]])
        np.testing.assert_allclose(matcher.call_args.args[1], [[512., 256.]])
        np.testing.assert_allclose(output[0], query)
        np.testing.assert_allclose(output[1], reference)
        self.assertEqual(matcher.call_args.args[3], 3.)

    def test_reference_crop_offset_is_restored_for_xyz_sampling(self):
        import torch
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.source_type = '3dgs'
        localizer.pipeline_policy = 'validated'
        localizer.retrieval_crop = True
        localizer.resize = 1024
        localizer.reference_offsets = {2: (100, 50)}
        localizer.reference_feature_xyz = {}
        localizer.geometry_cache = {}
        localizer.xyz_search_radius = 0
        localizer._xyz = lambda index: np.zeros((100, 200, 3))
        localizer._source_xyz_to_runtime = lambda values: values
        features = {'keypoints': torch.tensor([[[10., 15.]]])}
        with patch('glb_virtual_live.sample_xyz_buffer',
                   return_value=([(1., 2., 3.)], [0])) as sample:
            mapped = localizer._feature_xyz(2, features)
        np.testing.assert_allclose(sample.call_args.args[0], [[110., 65.]])
        np.testing.assert_allclose(mapped, [[1., 2., 3.]])

    def test_depth_unprojection_uses_opengl_negative_z(self):
        depth = np.asarray(((2.0,),), dtype=np.float32)
        matrix = np.eye(3, dtype=np.float64)
        xyz = depth_to_world_xyz(depth, np.eye(4), matrix)
        np.testing.assert_allclose(xyz[0, 0], (0.0, 0.0, -2.0))

    def test_look_at_pose_points_negative_camera_z_at_target(self):
        pose = look_at_pose(
            np.asarray((0.0, 0.0, 5.0)),
            np.asarray((0.0, 0.0, 0.0)),
        )
        np.testing.assert_allclose(pose[:3, 2], (0.0, 0.0, 1.0))
        np.testing.assert_allclose(pose[:3, 3], (0.0, 0.0, 5.0))

    def test_sparse_observation_transfer_keeps_best_unique_point(self):
        query = np.asarray(((1, 2), (3, 4), (5, 6)), dtype=np.float32)
        reference = np.asarray(((10.2, 10), (10.4, 10), (30, 30)), dtype=np.float32)
        scores = np.asarray((0.8, 0.9, 0.7), dtype=np.float32)
        observations = (
            Observation(42, (10, 10), (1, 2, 3)),
            Observation(84, (30, 30), (4, 5, 6)),
        )
        result = associate_reference_observations(
            query, reference, scores, observations, maximum_distance=1.0
        )
        self.assertEqual({item.point3d_id for item in result}, {42, 84})
        selected = next(item for item in result if item.point3d_id == 42)
        self.assertAlmostEqual(selected.match_score, 0.9, places=5)

    def test_xyz_sampling_recovers_nearest_valid_pixel(self):
        xyz = np.full((5, 5, 3), np.nan, dtype=np.float32)
        xyz[2, 3] = (1, 2, 3)
        sampled, indices = sample_xyz_buffer(
            np.asarray(((2, 2), (0, 0)), dtype=np.float32), xyz, search_radius=1
        )
        self.assertEqual(indices, [0])
        self.assertEqual(sampled, [(1.0, 2.0, 3.0)])

    def test_live_glb_points_are_converted_to_colmap_axes(self):
        source = VirtualTransfer(0, (4.0, -201.0, -591.0), (1, 2), (3, 4), 0.9)
        converted = GLBVirtualLiveLocalizer._to_colmap((source,))
        self.assertEqual(converted[0].xyz, (4.0, 591.0, -201.0))

    def test_live_3dgs_points_keep_existing_colmap_axes(self):
        source = VirtualTransfer(0, (4.0, 591.0, -201.0), (1, 2), (3, 4), 0.9)
        localizer = GLBVirtualLiveLocalizer.__new__(GLBVirtualLiveLocalizer)
        localizer.runtime_transform = "identity"
        converted = localizer._to_runtime((source,))
        self.assertEqual(converted[0].xyz, source.xyz)

    def test_live_glb_merge_keeps_strongest_match_per_live_keypoint(self):
        weak = VirtualTransfer(0, (1, 2, 3), (10, 20), (1, 1), 0.2)
        strong = VirtualTransfer(1, (4, 5, 6), (10, 20), (2, 2), 0.9)
        other = VirtualTransfer(2, (7, 8, 9), (30, 40), (3, 3), 0.7)
        merged = GLBVirtualLiveLocalizer._merge_transfers(((weak,), (strong, other)))
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].xyz, strong.xyz)
        self.assertEqual({item.query_uv for item in merged}, {(10, 20), (30, 40)})


if __name__ == "__main__":
    unittest.main()
