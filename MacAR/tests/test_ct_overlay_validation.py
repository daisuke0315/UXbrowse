import sys
import tempfile
import threading
import unittest
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ct_overlay_validation import CTValidationSession
from realtime_pnp_server import LiveResult, MPRRenderer


class CTOverlayValidationTests(unittest.TestCase):
    def test_live_ray_to_ct_marker_reports_pixel_and_metric_lateral_error(self):
        renderer = SimpleNamespace(unit_label='mm', current_validation_plane=lambda: None)
        result = LiveResult('PASS', 'test', 0, 20, 15, .75, 1.5,
                            (0., 0., 0.), (0., 0., 0.), (0., 0., 0.),
                            1., 'HIDDEN')
        matrix = np.asarray(((100., 0., 50.), (0., 100., 50.), (0., 0., 1.)))
        session = CTValidationSession(renderer, np.zeros((100, 100, 3), np.uint8),
                                      matrix, np.zeros(5), result)
        session.live_uv = np.asarray((50., 50.))
        session.ct_xyz = np.asarray((10., 0., 100.))
        session.ct_voxel = np.asarray((60., 50., 100.))
        measured = session.measurement()
        self.assertAlmostEqual(measured['pixel_error'], 10.)
        self.assertAlmostEqual(measured['lateral_error'], 10.)
        np.testing.assert_allclose(measured['ct_physical_xyz'],[10,100,0])
        np.testing.assert_allclose(measured['live_at_ct_depth_physical_xyz'],[0,100,0])
        np.testing.assert_allclose(measured['delta_ct_physical'],[-10,0,0])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'measurements.csv'
            session.save(output, calibration_label='approximate K')
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertIn('live_at_ct_depth_physical_x',output.read_text())

    def test_rotated_camera_reports_ct_physical_coordinates_and_depth_invariant_distance(self):
        from realtime_pnp_server import rc_to_colmap, colmap_to_rc
        renderer=SimpleNamespace(unit_label='mm',current_validation_plane=lambda:None)
        base=LiveResult('PASS','test',0,20,15,.75,1.5,(.2,-.3,.4),(12.,-9.,80.),(0.,0.,0.),1.,'HIDDEN')
        K=np.array([[400.,0,300.],[0,500.,200.],[0,0,1.]])
        session=CTValidationSession(renderer,np.zeros((400,600,3),np.uint8),K,np.zeros(5),base)
        session.ct_xyz=rc_to_colmap(np.array([15.,100.,-8.]))
        R=cv2.Rodrigues(session.rvec)[0];camera=R@session.ct_xyz+session.tvec.ravel()
        ct_uv=cv2.projectPoints(session.ct_xyz.reshape(1,3),session.rvec,session.tvec,K,np.zeros(5))[0].reshape(2)
        session.live_uv=ct_uv+np.array([4.,-5.])
        m=session.measurement()
        expected_camera=np.array([camera[2]*4/400,camera[2]*-5/500,0])
        expected=colmap_to_rc(R.T@expected_camera)
        np.testing.assert_allclose(m['delta_ct_physical'],expected,atol=1e-10)
        np.testing.assert_allclose(m['ct_physical_xyz'],[15,100,-8])
        self.assertAlmostEqual(m['lateral_error'],np.linalg.norm(expected))
        renderer.current_validation_plane=lambda:(np.ones(3)*500,np.array([1,0,0]),np.array([0,1,0]))
        self.assertAlmostEqual(session.measurement()['lateral_error'],m['lateral_error'])

    def test_zoom_is_cursor_anchored_and_does_not_change_selected_image_point(self):
        node=Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
        if not node.exists():self.skipTest('Node unavailable')
        script="""
        const {ImageViewTransform}=await import(process.argv[1]);
        const t=new ImageViewTransform();t.reset(1920,1080);t.width=700;t.height=530;
        const pointer=[137,241],before=t.imageAt(...pointer);t.zoomAt(...pointer,8);
        const after=t.imageAt(...pointer),point=[653.25,456.75];
        t.pan(93,-24);const roundtrip=t.imageAt(...t.screenAt(point));
        t.zoomAt(10,20,100);const maximum=t.zoom;t.zoomAt(10,20,.1);
        console.log(JSON.stringify({before,after,point,roundtrip,maximum,minimum:t.zoom}));
        """
        source=(Path(__file__).resolve().parents[1]/'browser_ui/measurement-workspace.js').as_uri()
        data=json.loads(subprocess.check_output([str(node),'--input-type=module','-e',script,source],text=True))
        np.testing.assert_allclose(data['before'],data['after'])
        np.testing.assert_allclose(data['point'],data['roundtrip'])
        self.assertEqual(data['maximum'],32)
        self.assertEqual(data['minimum'],1)

    def test_ct_click_is_a_fixed_world_point_not_a_reused_texture_pixel(self):
        renderer = MPRRenderer.__new__(MPRRenderer)
        renderer._control_lock = threading.Lock()
        renderer._last_inverse_homography = np.eye(3)
        renderer._last_inverse_view_homography = np.eye(3)
        renderer._last_validation_plane = (np.asarray((0., 0., 10.)),
                                           np.asarray((1., 0., 0.)),
                                           np.asarray((0., 1., 0.)))
        renderer.texture_size = 101
        renderer.field_of_view = 100.
        renderer.volume = SimpleNamespace(origin=np.asarray((-50., -40., -10.)),
            spacing=np.ones(3), inverse_direction=np.eye(3),
            dimensions=(101, 101, 21))
        point, voxel = renderer.display_to_ct_point(50., 50.)
        np.testing.assert_allclose(point, (0., 0., 10.))
        np.testing.assert_allclose(voxel, (50., 50., 10.))
        renderer._last_validation_plane = (np.asarray((0., 0., 12.)),
                                           np.asarray((1., 0., 0.)),
                                           np.asarray((0., 1., 0.)))
        self.assertEqual(float(point[2]), 10.)
        self.assertEqual(float(renderer.display_to_ct_point(50., 50.)[0][2]), 12.)


if __name__ == '__main__':
    unittest.main()
