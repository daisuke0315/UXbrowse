#!/usr/bin/env python3
"""Render RGB, depth, and per-pixel GLB XYZ reference views from a textured GLB."""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GLB = SCRIPT_DIR.parent / "data" / "260911" / "260108-3.glb"
DEFAULT_OUTPUT = DATA_DIR / "output_glb_reference_views"


class GLBRenderError(RuntimeError):
    """Expected GLB input or rendering error."""


def look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an OpenGL camera-to-world pose looking from eye to target."""
    forward = np.asarray(target - eye, np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError("Camera eye and target must differ")
    forward /= norm
    preferred_up = np.asarray((0.0, 1.0, 0.0), np.float64)
    if abs(float(np.dot(forward, preferred_up))) > 0.98:
        preferred_up = np.asarray((0.0, 0.0, 1.0), np.float64)
    right = np.cross(forward, preferred_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def depth_to_world_xyz(
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Unproject pyrender's positive camera depth into GLB/world coordinates."""
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    z = depth.astype(np.float32, copy=False)
    camera_points = np.stack(
        (
            (xs - cx) * z / fx,
            -(ys - cy) * z / fy,
            -z,
        ),
        axis=-1,
    )
    xyz = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    xyz[z <= 0] = np.nan
    return xyz.astype(np.float32)


def camera_positions(
    center: np.ndarray,
    radius: float,
    azimuth_count: int,
    elevations_deg: Sequence[float],
    distance_multipliers: Sequence[float],
):
    for distance_index, multiplier in enumerate(distance_multipliers):
        distance = radius * multiplier
        for elevation_index, elevation_deg in enumerate(elevations_deg):
            elevation = math.radians(elevation_deg)
            for azimuth_index in range(azimuth_count):
                azimuth_deg = 360.0 * azimuth_index / azimuth_count
                azimuth = math.radians(azimuth_deg)
                offset = distance * np.asarray(
                    (
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                        math.cos(elevation) * math.cos(azimuth),
                    ),
                    np.float64,
                )
                yield (
                    distance_index,
                    elevation_index,
                    azimuth_index,
                    azimuth_deg,
                    elevation_deg,
                    distance,
                    center + offset,
                )


def run(args) -> int:
    if not args.glb.is_file():
        raise GLBRenderError(f"GLB not found: {args.glb}")
    if args.width <= 0 or args.height <= 0 or args.azimuths <= 0:
        raise GLBRenderError("Width, height, and azimuth count must be positive")
    if not args.elevations or not args.distances or min(args.distances) <= 1:
        raise GLBRenderError("Elevations are required and distances must be > 1")
    if not 1 < args.fov_deg < 179:
        raise GLBRenderError("--fov-deg must be between 1 and 179")
    try:
        import pyrender
        import trimesh
        from PIL import Image
    except ImportError as exc:
        raise GLBRenderError(
            "Rendering dependencies are missing. Run: "
            "python3 -m pip install -r requirements-learned-matching.txt"
        ) from exc

    loaded = trimesh.load(str(args.glb), force="scene")
    if not loaded.geometry:
        raise GLBRenderError("GLB contains no mesh geometry")
    bounds = np.asarray(loaded.bounds, np.float64)
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5)
    scene = pyrender.Scene.from_trimesh_scene(
        loaded,
        bg_color=np.asarray((255, 255, 255, 255), np.uint8),
        ambient_light=np.asarray((0.35, 0.35, 0.35), np.float32),
    )
    vertical_fov = math.radians(args.fov_deg)
    camera = pyrender.PerspectiveCamera(
        yfov=vertical_fov,
        aspectRatio=args.width / args.height,
        znear=max(0.1, radius * 0.005),
        zfar=radius * (max(args.distances) + 3.0),
    )
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=args.light_intensity)
    focal = 0.5 * args.height / math.tan(vertical_fov * 0.5)
    camera_matrix = np.asarray(
        (
            (focal, 0.0, args.width * 0.5),
            (0.0, focal, args.height * 0.5),
            (0.0, 0.0, 1.0),
        ),
        np.float64,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = None
    records = []
    try:
        renderer = pyrender.OffscreenRenderer(args.width, args.height)
        positions = list(
            camera_positions(
                center,
                radius,
                args.azimuths,
                args.elevations,
                args.distances,
            )
        )
        print(
            f"GLB: {args.glb}\nBounds: {bounds.tolist()}\n"
            f"Rendering {len(positions)} reference views..."
        )
        for view_index, (
            distance_index,
            elevation_index,
            azimuth_index,
            azimuth_deg,
            elevation_deg,
            distance,
            eye,
        ) in enumerate(positions):
            pose = look_at_pose(eye, center)
            camera_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)
            try:
                color, depth = renderer.render(scene)
            finally:
                scene.remove_node(light_node)
                scene.remove_node(camera_node)
            xyz = depth_to_world_xyz(depth, pose, camera_matrix)
            name = f"view_{view_index:04d}"
            image_name = f"{name}.png"
            data_name = f"{name}.npz"
            Image.fromarray(color, mode="RGB").save(args.output / image_name)
            np.savez_compressed(
                args.output / data_name,
                depth=depth.astype(np.float32),
                xyz=xyz,
                camera_matrix=camera_matrix,
                camera_to_world=pose,
            )
            valid_pixels = int(np.count_nonzero(depth > 0))
            record = {
                "view_id": view_index,
                "image_name": image_name,
                "data_name": data_name,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "distance": distance,
                "distance_multiplier": args.distances[distance_index],
                "eye": eye.tolist(),
                "valid_pixels": valid_pixels,
                "camera_to_world": pose.tolist(),
                "camera_matrix": camera_matrix.tolist(),
            }
            records.append(record)
            print(
                f"[{view_index + 1:03d}/{len(positions):03d}] {image_name} "
                f"az={azimuth_deg:.1f} el={elevation_deg:.1f} "
                f"pixels={valid_pixels}"
            )
    except Exception as exc:
        if isinstance(exc, GLBRenderError):
            raise
        raise GLBRenderError(
            "GLB rendering failed. On macOS run this from VS Code/Terminal with "
            "access to the WindowServer."
        ) from exc
    finally:
        if renderer is not None:
            renderer.delete()

    manifest = {
        "format_version": 1,
        "glb": str(args.glb.resolve()),
        "coordinate_system": "GLB source coordinates (millimeters for dataset 260911)",
        "width": args.width,
        "height": args.height,
        "vertical_fov_deg": args.fov_deg,
        "bounds": bounds.tolist(),
        "center": center.tolist(),
        "radius": radius,
        "views": records,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    print(f"Completed: {args.output / 'manifest.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, default=DEFAULT_GLB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=55.0)
    parser.add_argument("--azimuths", type=int, default=12)
    parser.add_argument("--elevations", nargs="+", type=float, default=(-30.0, 0.0, 30.0))
    parser.add_argument("--distances", nargs="+", type=float, default=(2.5, 3.3))
    parser.add_argument("--light-intensity", type=float, default=3.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (GLBRenderError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
