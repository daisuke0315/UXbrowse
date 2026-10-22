import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_gaussian_pose_render import pose_camera_request


class GaussianPoseRenderTests(unittest.TestCase):
    def test_opencv_world_to_camera_pose_becomes_matching_look_at(self):
        result = SimpleNamespace(status='PASS', rvec=(0., 0., 0.),
                                 tvec=(0., 0., 100.))
        matrix = np.asarray(((800., 0., 639.5),
                             (0., 800., 359.5),
                             (0., 0., 1.)))
        request = pose_camera_request(result, matrix, (720, 1280, 3),
                                      request_id=7)
        self.assertEqual(request['id'], 7)
        np.testing.assert_allclose(request['eye'], (0., 0., -100.))
        np.testing.assert_allclose(request['target'], (0., 0., -99.))
        np.testing.assert_allclose(request['up'], (0., -1., 0.))
        self.assertGreater(request['fov'], 40.)
        self.assertLess(request['fov'], 60.)

    def test_unverified_pose_is_not_rendered(self):
        result = SimpleNamespace(status='FAIL', rvec=(0., 0., 0.),
                                 tvec=(0., 0., 100.))
        self.assertIsNone(pose_camera_request(
            result, np.eye(3), (720, 1280, 3), request_id=1))


if __name__ == '__main__':
    unittest.main()
