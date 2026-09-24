import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_global_scan  # noqa: E402

from realtime_pnp import (  # noqa: E402
    AsyncReferenceLocalizer,
    MPRTrackbarControls,
    OpenCVDirectCapture,
    OpenCVLatestFrameCapture,
    PointCloud3DViewer,
    apply_runtime_preset,
    approximate_camera_matrix,
    build_parser,
    center_focus_score,
    open_camera,
    resize_for_processing,
    scaled_calibration,
)
from realtime_pnp_server import (  # noqa: E402
    LiveResult,
    LivePnPError,
    LocalizationDebug,
    MPRRenderer,
    TemporalGlobalPnPLocalizer,
    TrackingSeed,
    colmap_to_rc,
    projected_box_depth_interval,
    ray_box_interval,
    rc_to_colmap,
)


class RealtimePnPTests(unittest.TestCase):
    def test_mpr_controls_change_extended_slice_and_opacity(self):
        renderer = MPRRenderer.__new__(MPRRenderer)
        renderer.slice_fraction = 0.5
        renderer.opacity = 0.82
        renderer.zoom = 1.0
        renderer._control_lock = threading.Lock()
        with mock.patch("realtime_pnp.cv2.createTrackbar") as create, mock.patch(
            "realtime_pnp.cv2.setMouseCallback"
        ) as mouse:
            controls = MPRTrackbarControls("preview", renderer)
        self.assertEqual(create.call_count, 3)
        slice_call = create.call_args_list[0]
        self.assertEqual(slice_call.args[2], 1500)
        self.assertEqual(slice_call.args[3], 2500)
        mouse.assert_called_once()
        controls._slice_changed(1750)
        controls._opacity_changed(35)
        controls._zoom_changed(700)
        self.assertAlmostEqual(renderer.slice_fraction, 0.75)
        self.assertAlmostEqual(renderer.opacity, 0.35)
        self.assertAlmostEqual(renderer.zoom, 8.0)

        controls._slice_changed(0)
        self.assertEqual(renderer.slice_fraction, -1.0)
        controls._slice_changed(1000)
        self.assertEqual(renderer.slice_fraction, 0.0)
        controls._slice_changed(2500)
        self.assertEqual(renderer.slice_fraction, 1.5)

    def test_measurement_single_click_uses_release_and_debounces_double_click(self):
        renderer = mock.Mock()
        renderer.slice_fraction = 0.5
        renderer.opacity = 0.82
        renderer.zoom = 1.0
        renderer.add_measurement_image_point.return_value = None
        renderer.measurement_point_count.return_value = 1
        with mock.patch("realtime_pnp.cv2.createTrackbar"), mock.patch(
            "realtime_pnp.cv2.setMouseCallback"
        ):
            controls = MPRTrackbarControls("preview", renderer)

        controls._mouse_event(cv2.EVENT_LBUTTONDOWN, 120, 80, 0, None)
        renderer.add_measurement_image_point.assert_not_called()
        controls._mouse_event(cv2.EVENT_LBUTTONUP, 120, 80, 0, None)
        controls._mouse_event(cv2.EVENT_LBUTTONUP, 121, 80, 0, None)
        renderer.add_measurement_image_point.assert_called_once_with(120, 80)

        controls._mouse_event(cv2.EVENT_LBUTTONUP, 180, 80, 0, None)
        self.assertEqual(renderer.add_measurement_image_point.call_count, 2)

    def test_mpr_control_values_are_clamped(self):
        renderer = MPRRenderer.__new__(MPRRenderer)
        renderer._control_lock = threading.Lock()
        renderer.zoom = 1.0
        renderer.set_slice_percent(180)
        renderer.set_opacity_percent(-5)
        renderer.set_zoom_percent(2000)
        self.assertEqual(renderer.slice_fraction, 1.5)
        self.assertEqual(renderer.opacity, 0.0)
        self.assertEqual(renderer.zoom, 12.0)
        renderer.set_slice_percent(-140)
        self.assertEqual(renderer.slice_fraction, -1.0)
        renderer.set_zoom_percent(20)
        self.assertEqual(renderer.zoom, 1.0)

    def test_measurement_zoom_maps_display_back_to_original_image(self):
        renderer = MPRRenderer.__new__(MPRRenderer)
        renderer._control_lock = threading.Lock()
        renderer._measurement_enabled = True
        renderer._measurement_points = []
        renderer._zoom_center = None
        renderer.texture_size = 101
        renderer.field_of_view = 100.0
        renderer._last_inverse_homography = np.asarray(
            ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        renderer._last_inverse_view_homography = np.asarray(
            ((0.25, 0.0, 5.0), (0.0, 0.25, 10.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        self.assertIsNone(renderer.add_measurement_image_point(20, 40))
        np.testing.assert_allclose(renderer._zoom_center, (10.0, 20.0))
        distance = renderer.add_measurement_image_point(80, 40)
        self.assertAlmostEqual(distance, 30.0)
        np.testing.assert_allclose(renderer._zoom_center, (10.0, 20.0))

    def test_mpr_measures_two_image_clicks_in_model_units(self):
        renderer = MPRRenderer.__new__(MPRRenderer)
        renderer._control_lock = threading.Lock()
        renderer._measurement_enabled = True
        renderer._measurement_points = []
        renderer._last_inverse_homography = np.eye(3, dtype=np.float64)
        renderer._last_inverse_view_homography = np.eye(3, dtype=np.float64)
        renderer._zoom_center = None
        renderer.texture_size = 101
        renderer.field_of_view = 100.0
        renderer.unit_label = "mm"
        self.assertIsNone(renderer.add_measurement_image_point(10, 20))
        distance = renderer.add_measurement_image_point(40, 60)
        self.assertAlmostEqual(distance, 50.0)
        self.assertEqual(len(renderer._measurement_points), 2)
        self.assertIsNone(renderer.add_measurement_image_point(30, 20))
        self.assertEqual(len(renderer._measurement_points), 1)
        renderer.clear_measurement()
        self.assertEqual(renderer._measurement_points, [])

    def test_mpr_depth_range_does_not_require_center_ray_box_hit(self):
        origin = np.asarray((0.0, 0.0, 0.0))
        forward = np.asarray((0.0, 0.0, 1.0))
        corners = np.asarray(
            [
                (x, y, z)
                for x in (100.0, 110.0)
                for y in (0.0, 10.0)
                for z in (20.0, 30.0)
            ]
        )
        self.assertIsNone(
            ray_box_interval(
                origin,
                forward,
                corners.min(axis=0),
                corners.max(axis=0),
            )
        )
        self.assertEqual(
            projected_box_depth_interval(origin, forward, corners),
            (20.0, 30.0),
        )

    def test_global_scan_entry_point_forces_non_tracking_mode(self):
        with mock.patch.object(
            main_global_scan, "realtime_main", return_value=0
        ) as run:
            status = main_global_scan.main(
                ("--camera", "2", "--matching-mode", "reference")
            )
        self.assertEqual(status, 0)
        run.assert_called_once_with(
            [
                "--camera",
                "2",
                "--matching-mode",
                "reference",
                "--matching-mode",
                "global",
            ]
        )

    def test_default_live_mode_uses_global_tracking(self):
        args = build_parser().parse_args(())
        self.assertEqual(args.matching_mode, "global-track")
        self.assertEqual(args.tracking_reseed_frames, 10)
        self.assertEqual(args.tracking_min_points, 10)
        self.assertEqual(args.min_inliers, 10)

    def test_reference_matching_arguments_are_available(self):
        args = build_parser().parse_args(
            ("--matching-mode", "reference", "--reference-shortlist", "24")
        )
        self.assertEqual(args.matching_mode, "reference")
        self.assertEqual(args.reference_shortlist, 24)
        self.assertEqual(args.reference_workers, 4)

    def test_glb_live_matching_arguments_are_available(self):
        args = build_parser().parse_args(
            (
                "--matching-mode",
                "glb-track",
                "--glb-views",
                "views72",
                "--glb-top-k",
                "3",
                "--glb-max-keypoints",
                "1024",
            )
        )
        self.assertEqual(args.matching_mode, "glb-track")
        self.assertEqual(args.glb_views, Path("views72"))
        self.assertEqual(args.glb_top_k, 3)
        self.assertEqual(args.glb_max_keypoints, 1024)

    def test_3dgs_live_matching_mode_and_views_are_available(self):
        args = build_parser().parse_args(
            (
                "--matching-mode",
                "splat-track",
                "--splat-views",
                "splat72",
                "--glb-top-k",
                "10",
            )
        )
        self.assertEqual(args.matching_mode, "splat-track")
        self.assertEqual(args.splat_views, Path("splat72"))
        self.assertEqual(args.glb_top_k, 10)

    def test_async_reference_localizer_does_not_block_camera_loop(self):
        debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )

        class SlowLocalizer:
            def __init__(self):
                self.map_data = SimpleNamespace()
                self.mpr_renderer = None
                self.last_debug = debug

            def localize(self, image, camera_matrix, distortion):
                time.sleep(0.02)
                result = SimpleNamespace(
                    status="PASS", inliers=25, detail="REF test.png"
                )
                return result, image

        localizer = AsyncReferenceLocalizer(SlowLocalizer(), 0.1)
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        matrix = np.eye(3, dtype=np.float64)
        distortion = np.zeros(5, dtype=np.float64)
        started = time.monotonic()
        working, _ = localizer.localize(frame, matrix, distortion)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.01)
        self.assertEqual(working.detail, "REFERENCE_SEARCH_RUNNING")
        time.sleep(0.03)
        completed, _ = localizer.localize(frame, matrix, distortion)
        self.assertEqual(completed.status, "PASS")
        localizer.close()

    def test_raw_preview_is_opt_in(self):
        args = build_parser().parse_args(())
        self.assertFalse(args.raw_preview)
        args = build_parser().parse_args(("--raw-preview",))
        self.assertTrue(args.raw_preview)

    def test_previous_success_settings_reproduce_original_live_setup(self):
        args = build_parser().parse_args(
            (
                "--previous-success-settings",
                "--capture-backend",
                "ffmpeg",
                "--width",
                "640",
                "--height",
                "480",
                "--ratio",
                "0.9",
            )
        )
        apply_runtime_preset(args)
        self.assertEqual(args.capture_backend, "opencv")
        self.assertEqual((args.width, args.height), (0, 0))
        self.assertEqual(args.max_processing_width, 0)
        self.assertEqual(args.process_fps, 5.0)
        self.assertEqual(args.min_focus_score, 0.0)
        self.assertFalse(args.raw_preview)
        self.assertEqual(args.matching_mode, "global")
        self.assertEqual(args.ratio, 0.75)
        self.assertEqual(args.ransac_threshold, 4.0)
        self.assertEqual(args.min_inliers, 20)
        self.assertEqual(args.min_inlier_ratio, 0.2)

    def test_temporal_global_tracker_reseeds_on_configured_cadence(self):
        debug = LocalizationDebug(
            np.empty((0, 3), dtype=np.float64), frozenset(), None, False
        )
        points_2d = np.zeros((20, 2), dtype=np.float64)
        points_3d = np.zeros((20, 3), dtype=np.float64)

        class FakeGlobalLocalizer:
            def __init__(self):
                self.map_data = SimpleNamespace()
                self.mpr_renderer = None
                self.ransac_threshold = 4.0
                self.min_inliers = 20
                self.min_inlier_ratio = 0.2
                self.last_debug = debug
                self.last_tracking_seed = None
                self.calls = 0

            def localize(self, image, camera_matrix, distortion):
                self.calls += 1
                self.last_tracking_seed = TrackingSeed(
                    np.zeros(image.shape[:2], dtype=np.uint8),
                    points_2d,
                    points_3d,
                    np.zeros((3, 1), dtype=np.float64),
                    np.asarray(((0.0,), (0.0,), (5.0,))),
                )
                result = LiveResult(
                    "PASS", "OK", 20, 20, 20, 1.0, 0.0,
                    (0.0, 0.0, 0.0), (0.0, 0.0, 5.0),
                    (0.0, 0.0, -5.0), 1.0, "HIDDEN",
                )
                return result, image.copy()

        base = FakeGlobalLocalizer()
        tracker = TemporalGlobalPnPLocalizer(base, reseed_frames=2)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        matrix = np.eye(3, dtype=np.float64)
        distortion = np.zeros(5, dtype=np.float64)
        tracker.localize(frame, matrix, distortion)
        tracked_result = LiveResult(
            "PASS", "TRACKING", 20, 20, 20, 1.0, 0.0,
            (0.0, 0.0, 0.0), (0.0, 0.0, 5.0),
            (0.0, 0.0, -5.0), 1.0, "HIDDEN",
        )
        with mock.patch.object(
            tracker, "_track", return_value=(tracked_result, frame.copy())
        ) as track:
            tracker.localize(frame, matrix, distortion)
            tracker.localize(frame, matrix, distortion)
            tracker.localize(frame, matrix, distortion)
        self.assertEqual(track.call_count, 2)
        self.assertEqual(base.calls, 2)

    def test_temporal_tracker_propagates_known_2d_3d_points(self):
        rng = np.random.default_rng(7)
        object_points = rng.uniform(-1.0, 1.0, size=(30, 3)).astype(np.float64)
        object_points[:, 2] *= 0.5
        matrix = np.asarray(
            ((800.0, 0.0, 320.0), (0.0, 800.0, 240.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        old_rvec = np.zeros((3, 1), dtype=np.float64)
        old_tvec = np.asarray(((0.0,), (0.0,), (8.0,)), dtype=np.float64)
        new_rvec = np.asarray(((0.01,), (0.02,), (-0.01,)), dtype=np.float64)
        new_tvec = np.asarray(((0.08,), (-0.03,), (8.0,)), dtype=np.float64)
        previous_points, _ = cv2.projectPoints(
            object_points, old_rvec, old_tvec, matrix, distortion
        )
        current_points, _ = cv2.projectPoints(
            object_points, new_rvec, new_tvec, matrix, distortion
        )
        previous_points = previous_points.reshape((-1, 2))
        current_points = current_points.reshape((-1, 2))
        debug = LocalizationDebug(object_points, frozenset(), None, False)
        base = SimpleNamespace(
            map_data=SimpleNamespace(),
            mpr_renderer=None,
            ransac_threshold=4.0,
            min_inliers=20,
            min_inlier_ratio=0.2,
            last_debug=debug,
        )
        tracker = TemporalGlobalPnPLocalizer(base, reseed_frames=10)
        tracker.seed = TrackingSeed(
            np.zeros((480, 640), dtype=np.uint8),
            previous_points,
            object_points,
            old_rvec,
            old_tvec,
        )
        status = np.ones((len(object_points), 1), dtype=np.uint8)
        flow_results = (
            (current_points.astype(np.float32).reshape((-1, 1, 2)), status, None),
            (previous_points.astype(np.float32).reshape((-1, 1, 2)), status, None),
        )
        with mock.patch(
            "realtime_pnp_server.cv2.calcOpticalFlowPyrLK",
            side_effect=flow_results,
        ):
            result, _ = tracker._track(
                np.zeros((480, 640, 3), dtype=np.uint8), matrix, distortion
            )
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.detail, "TRACKING")
        self.assertGreaterEqual(result.inliers, 20)
        self.assertLess(result.mean_reprojection_error, 0.1)

    def test_opencv_latest_capture_uses_native_settings_and_thread(self):
        class FakeCapture:
            def __init__(self):
                self.released = False
                self.set_calls = []
                self.value = 0

            def isOpened(self):
                return True

            def set(self, key, value):
                self.set_calls.append((key, value))
                return True

            def read(self):
                if self.released:
                    return False, None
                self.value += 1
                time.sleep(0.002)
                return True, np.full(
                    (8, 10, 3), self.value % 255, dtype=np.uint8
                )

            def release(self):
                self.released = True

        fake = FakeCapture()
        with mock.patch("realtime_pnp.cv2.VideoCapture", return_value=fake):
            capture = open_camera(1, 0, 0, capture_backend="opencv-latest")
            self.assertIsInstance(capture, OpenCVLatestFrameCapture)
            ok, first = capture.read()
            time.sleep(0.02)
            ok2, latest = capture.read()
            stats = capture.stats()
            capture.release()
        self.assertTrue(ok and ok2)
        self.assertGreater(int(latest[0, 0, 0]), int(first[0, 0, 0]))
        self.assertGreater(stats["skipped_frames"], 0)
        self.assertEqual(fake.set_calls, [])
        self.assertTrue(fake.released)

    def test_opencv_default_reads_directly_without_property_overrides(self):
        class FakeCapture:
            def __init__(self):
                self.read_count = 0
                self.set_calls = []
                self.released = False

            def isOpened(self):
                return True

            def set(self, key, value):
                self.set_calls.append((key, value))
                return True

            def read(self):
                self.read_count += 1
                return True, np.full((8, 10, 3), self.read_count, dtype=np.uint8)

            def release(self):
                self.released = True

        fake = FakeCapture()
        with mock.patch("realtime_pnp.cv2.VideoCapture", return_value=fake):
            capture = open_camera(1, 0, 0, capture_backend="opencv")
            self.assertIsInstance(capture, OpenCVDirectCapture)
            self.assertEqual(fake.read_count, 0)
            ok, frame = capture.read()
            self.assertEqual(fake.read_count, 1)
            capture.release()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0, 0, 0]), 1)
        self.assertEqual(fake.set_calls, [])
        self.assertTrue(fake.released)

    def test_rc_colmap_axis_conversion_round_trip(self):
        rc = np.asarray(
            ((4.5, -201.0, -591.0), (-8.0, 12.0, 30.0)), dtype=np.float64
        )
        expected_colmap = np.asarray(
            ((4.5, 591.0, -201.0), (-8.0, -30.0, 12.0)), dtype=np.float64
        )
        np.testing.assert_array_equal(rc_to_colmap(rc), expected_colmap)
        np.testing.assert_array_equal(colmap_to_rc(expected_colmap), rc)

    def test_ray_box_hit_and_miss(self):
        minimum = np.asarray((-1.0, -1.0, -1.0))
        maximum = np.asarray((1.0, 1.0, 1.0))
        hit = ray_box_interval(
            np.asarray((0.0, 0.0, -3.0)),
            np.asarray((0.0, 0.0, 1.0)),
            minimum,
            maximum,
        )
        self.assertEqual(hit, (2.0, 4.0))
        miss = ray_box_interval(
            np.asarray((3.0, 0.0, -3.0)),
            np.asarray((0.0, 0.0, 1.0)),
            minimum,
            maximum,
        )
        self.assertIsNone(miss)

    def test_approximate_camera_matrix_uses_horizontal_fov(self):
        matrix = approximate_camera_matrix(1920, 1080, 90.0)
        self.assertAlmostEqual(matrix[0, 0], 960.0)
        self.assertAlmostEqual(matrix[1, 1], 960.0)
        self.assertEqual(matrix[0, 2], 960.0)
        self.assertEqual(matrix[1, 2], 540.0)

    def test_calibration_scaling_requires_same_aspect_ratio(self):
        matrix = np.asarray(
            ((1000.0, 0, 960.0), (0, 1000.0, 540.0), (0, 0, 1))
        )
        scaled = scaled_calibration(matrix, (1920, 1080), (1280, 720))
        self.assertAlmostEqual(scaled[0, 0], 1000.0 * 2 / 3)
        with self.assertRaises(LivePnPError):
            scaled_calibration(matrix, (1920, 1080), (1280, 960))

    def test_processing_resize_preserves_aspect_ratio(self):
        frame = np.zeros((1440, 1920, 3), dtype=np.uint8)
        resized, scale = resize_for_processing(frame, 1280)
        self.assertEqual(resized.shape, (960, 1280, 3))
        self.assertAlmostEqual(scale, 2 / 3)
        unchanged, unchanged_scale = resize_for_processing(frame, 0)
        self.assertIs(unchanged, frame)
        self.assertEqual(unchanged_scale, 1.0)

    def test_center_focus_score_rises_for_sharp_edges(self):
        sharp = np.zeros((300, 300, 3), dtype=np.uint8)
        sharp[100:200, 100:200] = 255
        blurred = cv2.GaussianBlur(sharp, (0, 0), 12)
        self.assertGreater(center_focus_score(sharp), center_focus_score(blurred))

    def test_live_3d_viewer_renders_matches_and_box(self):
        fake_localizer = SimpleNamespace(
            map_data=SimpleNamespace(
                point3d_ids=np.asarray((1, 2, 2, 3)),
                xyz=np.asarray(
                    ((-1, -1, -1), (1, -1, 0), (1, -1, 0), (0, 1, 1)),
                    dtype=np.float64,
                ),
            ),
            mpr_renderer=SimpleNamespace(
                box_minimum=np.asarray((-2.0, -2.0, -2.0)),
                box_maximum=np.asarray((2.0, 2.0, 2.0)),
            ),
        )
        viewer = PointCloud3DViewer(fake_localizer, width=640, height=480)
        debug = LocalizationDebug(
            matched_xyz=np.asarray(((-1, -1, -1), (0, 1, 1)), dtype=np.float64),
            inlier_indices=frozenset((1,)),
            camera_center_model=(0.0, -4.0, 0.0),
            pnp_solved=True,
        )
        image = viewer.render(debug, "PASS")
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertGreater(int(image.max()), 0)


if __name__ == "__main__":
    unittest.main()
