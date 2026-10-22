#!/usr/bin/env python3
"""Calibrate a Mac/Continuity camera with a printed checkerboard."""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore

from realtime_pnp import open_camera
from realtime_pnp_server import LivePnPError




def calibrate(args) -> int:
    capture = open_camera(args.camera, args.width, args.height)
    object_template = np.zeros((args.rows * args.columns, 3), dtype=np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
        * args.square_size
    )
    object_points: list[object] = []
    image_points: list[object] = []
    image_size = None
    latest_corners = None
    print(
        f"Show a {args.columns}x{args.rows} INNER-corner checkerboard. "
        "Space=capture, C=calibrate, R=reset, Q=quit."
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LivePnPError("Camera frame capture failed")
            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray,
                (args.columns, args.rows),
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
            latest_corners = corners if found else None
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(
                    display, (args.columns, args.rows), corners, found
                )
            cv2.rectangle(display, (0, 0), (display.shape[1], 58), (20, 20, 20), -1)
            cv2.putText(
                display,
                f"samples {len(image_points)}/{args.samples} | board {'FOUND' if found else 'NOT FOUND'}",
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (60, 230, 80) if found else (70, 70, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Continuity Camera Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 2
            if key == ord("r"):
                object_points.clear()
                image_points.clear()
                print("Samples reset")
            if key == ord(" "):
                if latest_corners is None:
                    print("Checkerboard not found; sample not captured")
                else:
                    object_points.append(object_template.copy())
                    image_points.append(latest_corners.astype(np.float32))
                    print(f"Captured sample {len(image_points)}/{args.samples}")
            if (key == ord("c") or len(image_points) >= args.samples) and len(image_points) >= args.minimum_samples:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if image_size is None:
        raise LivePnPError("No camera image size was obtained")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise LivePnPError("Calibration produced non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        camera_matrix=matrix.astype(np.float64),
        distortion=distortion.reshape(-1).astype(np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=np.asarray(rms, dtype=np.float64),
        checkerboard=np.asarray((args.columns, args.rows), dtype=np.int32),
        square_size=np.asarray(args.square_size, dtype=np.float64),
    )
    print(f"Calibration saved: {args.output}")
    print(f"RMS reprojection error: {rms:.6f}px")
    print(
        f"fx={matrix[0, 0]:.9g} fy={matrix[1, 1]:.9g} "
        f"cx={matrix[0, 2]:.9g} cy={matrix[1, 2]:.9g}"
    )
    print("distortion:", " ".join(f"{value:.9g}" for value in distortion.ravel()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate an iPhone Continuity Camera for accurate PnP."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=0, help="requested width; 0 uses device default")
    parser.add_argument("--height", type=int, default=0, help="requested height; 0 uses device default")
    parser.add_argument("--columns", type=int, default=9, help="checkerboard inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="checkerboard inner corners down")
    parser.add_argument("--square-size", type=float, default=20.0, help="checker square size in mm")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "output_colmap_image" / "iphone_camera_calibration.npz",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (
            args.columns < 3
            or args.rows < 3
            or args.square_size <= 0
            or args.minimum_samples < 4
            or args.samples < args.minimum_samples
        ):
            raise LivePnPError("Invalid checkerboard or sample parameters")
        return calibrate(args)
    except (LivePnPError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
