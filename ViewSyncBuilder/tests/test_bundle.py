import unittest
import numpy as np
import pycolmap as pc

from viewsync_builder.bundle import constrained_bundle, evaluate_points


def fixture():
    pc.set_random_seed(12)
    model = pc.synthesize_dataset(pc.SyntheticDatasetOptions(
        num_rigs=1, num_frames_per_rig=5, num_points3D=40,
        num_points2D_without_point3D=0))
    points = []
    for index, point_id in enumerate(list(model.points3D)[:7]):
        point = model.points3D[point_id]
        observations = []
        for element in point.track.elements:
            image = model.images[element.image_id]
            observations.append(dict(image_id=image.name,
                                     uv=image.points2D[element.point2D_idx].xy.tolist()))
        points.append(dict(id='g'+str(index), xyz=point.xyz.tolist(),
                           role='evaluation' if index == 6 else 'control',
                           enabled=True, observations=observations))
        model.delete_point3D(point_id)
    rng = np.random.default_rng(3)
    for frame in model.frames.values():
        pose = frame.rig_from_world
        frame.rig_from_world = pc.Rigid3d(pose.rotation, pose.translation + rng.normal(0, .025, 3))
    for point in model.points3D.values():
        point.xyz += rng.normal(0, .015, 3)
    return model, points


class BundleTests(unittest.TestCase):
    def test_fixed_ct_points_and_free_cameras_and_sparse_points(self):
        model, controls = fixture()
        original_camera = model.images[1].cam_from_world().translation.copy()
        original_point = next(iter(model.points3D.values())).xyz.copy()
        result, report = constrained_bundle(model, controls, refine_intrinsics=False)
        self.assertTrue(report['converged'])
        self.assertLess(report['after_reprojection_px']['mean'], report['before_reprojection_px']['mean'] / 10)
        self.assertGreater(np.linalg.norm(result.images[1].cam_from_world().translation-original_camera), .001)
        self.assertGreater(np.linalg.norm(next(iter(result.points3D.values())).xyz-original_point), .001)
        for point, point_id in zip(controls[:6], report['fixed_gcp_ids'].values()):
            np.testing.assert_array_equal(result.points3D[point_id].xyz, point['xyz'])
        self.assertEqual(report['gcp_reprojection_px']['failures'], 0)

    def test_evaluation_xyz_never_changes_optimization(self):
        model, controls = fixture()
        result, _ = constrained_bundle(model, controls)
        controls[-1]['xyz'] = [500, -300, 200]
        changed, _ = constrained_bundle(model, controls)
        for image_id in result.images:
            np.testing.assert_allclose(result.images[image_id].cam_from_world().matrix(),
                                       changed.images[image_id].cam_from_world().matrix(), atol=1e-12)
        qc = evaluate_points(result, controls)
        self.assertGreater(qc['tre_mm']['mean'], 100)
        self.assertEqual(qc['evaluation_count'], 1)

    def test_missing_evaluation_is_unverified_and_bad_controls_rejected(self):
        model, controls = fixture()
        self.assertEqual(evaluate_points(model, controls[:6])['accuracy_status'], 'unverified')
        for point in controls[:6]:
            point['xyz'] = [0, 0, 0]
        with self.assertRaises(ValueError):
            constrained_bundle(model, controls)

    def test_interpolated_reference_is_identified_and_never_passes_accuracy(self):
        model, controls = fixture()
        result, _ = constrained_bundle(model, controls)
        for point in controls:
            point['ct_sampling'] = dict(status='original')
        controls[-1]['ct_sampling'] = dict(status='interpolated',interpolated_indices=[2])
        qc = evaluate_points(result, controls)
        self.assertEqual(qc['accuracy_status'], 'unverified')
        self.assertEqual(qc['interpolated_reference_tre_mm']['count'], 1)
        self.assertEqual(qc['original_reference_tre_mm']['count'], 0)
        self.assertEqual(qc['points'][0]['ct_sampling']['status'], 'interpolated')
        controls[-1]['ct_sampling'] = dict(status='original')
        controls[0]['ct_sampling'] = dict(status='interpolated')
        qc = evaluate_points(result, controls)
        self.assertEqual(qc['accuracy_status'], 'unverified')
        self.assertEqual(qc['interpolated_control_ids'], [controls[0]['id']])

    def test_bad_gcp_observation_is_reported_as_an_outlier(self):
        model, controls = fixture()
        controls[0]['observations'][0]['uv'][0] += 100
        _, report = constrained_bundle(model, controls)
        self.assertGreater(report['gcp_outlier_observations'], 0)

    def test_intrinsics_fixed_by_default_and_no_convergence_is_not_success(self):
        model, controls = fixture()
        result, report = constrained_bundle(model, controls)
        for camera_id in model.cameras:
            np.testing.assert_array_equal(model.cameras[camera_id].params, result.cameras[camera_id].params)
        _, stopped = constrained_bundle(model, controls, max_iterations=0)
        self.assertFalse(stopped['converged'])


if __name__ == '__main__':
    unittest.main()
