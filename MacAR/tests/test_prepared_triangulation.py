"""Old/new geometry equivalence: support, XYZ, distortion and rejection QC."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_registered_triangulation import legacy_robust, compare_results
from build_aliked_object_map import triangulate_track
from enrich_registered_aliked_map import robust_triangulation, observation_errors, projected_errors
from offline_pnp_test import ReferenceImage, quaternion_to_rotation
from prepared_triangulation import TriangulationContext


class PreparedTriangulationTests(unittest.TestCase):
    def scene(self, distortion_size=5, count=12):
        records, matrices, distortions, uv = {}, {}, {}, {}
        xyz = np.array([.2, -.1, 5.])
        for i in range(count):
            angle = i*2*np.pi/count
            center = np.array([np.cos(angle), np.sin(angle), .1*np.sin(2*angle)])
            rotation_angle = .12*np.cos(angle)
            quaternion = (np.cos(rotation_angle/2), 0., 0., np.sin(rotation_angle/2))
            rotation = quaternion_to_rotation(np, quaternion)
            translation = -rotation @ center
            name = str(i)
            records[name] = ReferenceImage(i, i, name, (*quaternion, *translation))
            matrices[i] = np.array([[500.+i, 0., 320.], [0., 510.-i, 240.], [0., 0., 1.]])
            distortions[i] = np.array([.02, -.005, .001, -.002, .001, .002, -.001,
                                       .0005, .001, -.0001, .0002, .0003, .01, -.02])[:distortion_size]
            uv[name] = cv2.projectPoints(xyz.reshape(1, 3), cv2.Rodrigues(rotation)[0],
                translation, matrices[i], distortions[i])[0].reshape(-1, 2)
        return records, uv, matrices, distortions, np.array([[-1., -1., 4.], [1., 1., 6.]])

    def test_randomized_consensus_matches_legacy_for_all_opencv_distortion_sizes(self):
        rng = np.random.default_rng(23)
        for size in (4, 5, 8, 12, 14):
            for trial in range(12):
                with self.subTest(distortion_size=size, trial=trial):
                    records, uv, matrices, distortions, bounds = self.scene(size)
                    for pixels in uv.values():
                        pixels += rng.normal(0, .2, pixels.shape)
                        if rng.random() < trial/16:
                            pixels += rng.normal(0, 60, pixels.shape)
                    nodes = [(name, 0) for name in records]
                    old = legacy_robust(nodes, uv, records, matrices, distortions, bounds)
                    new = robust_triangulation(nodes, uv, records, matrices, distortions, bounds)
                    compare_results(old, new)

    def test_refinement_matches_legacy_dlt(self):
        records, uv, matrices, distortions, bounds = self.scene(14)
        nodes = [(name, 0) for name in records]
        prepared = TriangulationContext(uv, records, matrices, distortions).prepare(nodes)
        old, reason = triangulate_track(nodes, uv, records, matrices, distortions, bounds, 2., 3., 10.)
        self.assertIsNone(reason)
        new = prepared.triangulate(np.arange(len(nodes)), bounds, 2., 3.)
        for left, right in zip(old, new):
            np.testing.assert_allclose(left, right, atol=1e-9)

    def test_job_context_undistorts_each_image_only_once(self):
        records, uv, matrices, distortions, bounds = self.scene()
        context = TriangulationContext(uv, records, matrices, distortions)
        nodes = [(name, 0) for name in records]
        with patch('prepared_triangulation.cv2.undistortPoints', wraps=cv2.undistortPoints) as calls:
            for _ in range(3):
                robust_triangulation(nodes, uv, records, matrices, distortions, bounds,
                                     geometry_context=context)
            self.assertEqual(calls.call_count, len(records))

    def test_bounds_parallax_error_and_cheirality_are_not_relaxed(self):
        for kind in ('bounds', 'parallax', 'error', 'behind'):
            with self.subTest(kind=kind):
                records, uv, matrices, distortions, bounds = self.scene()
                kwargs = {}
                if kind == 'bounds':
                    bounds += 100.
                elif kind == 'parallax':
                    kwargs['min_angle'] = 179.
                elif kind == 'error':
                    for i, pixels in enumerate(uv.values()):
                        pixels += [i*10, i*i*8]
                else:
                    point = np.array([.2, -.1, -5.])
                    bounds = np.array([[-1., -1., -6.], [1., 1., -4.]])
                    for name, record in records.items():
                        rotation = quaternion_to_rotation(np, record.pose)
                        uv[name] = cv2.projectPoints(point.reshape(1, 3), cv2.Rodrigues(rotation)[0],
                            np.asarray(record.pose[4:]), matrices[record.camera_id],
                            distortions[record.camera_id])[0].reshape(1, 2)
                nodes = [(name, 0) for name in records]
                old = legacy_robust(nodes, uv, records, matrices, distortions, bounds, **kwargs)
                new = robust_triangulation(nodes, uv, records, matrices, distortions, bounds, **kwargs)
                compare_results(old, new)
                self.assertIsNone(new)

    def test_export_projection_keeps_float32_float64_and_negative_depth_results(self):
        records, _, matrices, distortions, _ = self.scene(14)
        for dtype in (np.float32, np.float64):
            points = np.array([[.2, -.1, 5.], [.3, .4, 5.], [.2, .1, -5.]], dtype=dtype)
            uv = {'0': np.array([[400., 220.], [450., 250.], [200., 200.]])}
            old = [projected_errors([('0', i)], point, uv, records, matrices, distortions)[0]
                   for i, point in enumerate(points)]
            new = observation_errors('0', np.arange(3), points, uv, records, matrices, distortions)
            np.testing.assert_allclose(old, new, atol=1e-9)
            self.assertTrue(np.isinf(new[-1]))

    def test_empty_export(self):
        self.assertEqual(len(observation_errors('missing', [], [], {}, {}, {}, {})), 0)


if __name__ == '__main__':
    unittest.main()
