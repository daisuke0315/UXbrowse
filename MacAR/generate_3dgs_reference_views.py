#!/usr/bin/env python3
"""Render 3DGS RGB and depth moments with the same Gaussian rasterizer.

RGB views are rendered by PlayCanvas' Gaussian rasterizer in Chrome/Metal, so
the PLY scale, quaternion, opacity and spherical harmonics all affect the
reference image. Alpha-composited depth is unprojected on the pixel ray;
transparent or mixed-depth pixels are invalid, not assumed to be a surface.
"""

from __future__ import annotations

from macar_paths import DATA_DIR

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np  # type: ignore
from splat_geometry import moments_to_geometry, unproject


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLY = Path(
    "/Users/daisuketanii/Library/CloudStorage/GoogleDrive-mp.daisuketanii@gmail.com/"
    "マイドライブ/010_splat/data/260918splat/splat.ply"
)
DEFAULT_OUTPUT = DATA_DIR / "output_3dgs_reference_views_72_upper_negz_generated"
DEFAULT_NODE = Path(
    "/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/node/bin/node"
)
DEFAULT_BROWSER_RENDERER = (
    SCRIPT_DIR / "tools" / "gs_browser_renderer" / "capture_batch.mjs"
)

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.asarray(
    (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
     -1.0925484305920792, 0.5462742152960396),
    np.float64,
)
SH_C3 = np.asarray(
    (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
     0.3731763325901154, -0.4570457994644658, 1.445305721320277,
     -0.5900435899266435),
    np.float64,
)


class SplatRenderError(RuntimeError):
    """Expected 3DGS input or rendering error."""


def read_gaussian_ply(path: Path):
    """Memory-map the standard float-only 3D Gaussian PLY layout."""
    if not path.is_file():
        raise SplatRenderError(f"3DGS PLY not found: {path}")
    properties = []
    vertex_count = None
    with path.open("rb") as source:
        first = source.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise SplatRenderError("Input is not a PLY file")
        while True:
            raw = source.readline()
            if not raw:
                raise SplatRenderError("PLY header has no end_header")
            line = raw.decode("ascii", errors="strict").strip()
            parts = line.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                pass
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif parts[:1] == ["property"]:
                if len(parts) != 3 or parts[1] not in {"float", "float32"}:
                    raise SplatRenderError(
                        "Only the standard float-only 3DGS vertex PLY is supported"
                    )
                properties.append(parts[2])
            elif line == "end_header":
                offset = source.tell()
                break
    required = {
        "x", "y", "z", "opacity",
        "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2",
    }
    missing = sorted(required.difference(properties))
    if vertex_count is None or vertex_count <= 0 or missing:
        raise SplatRenderError(
            f"Invalid 3DGS PLY header; missing properties: {', '.join(missing)}"
        )
    dtype = np.dtype([(name, "<f4") for name in properties])
    records = np.memmap(
        path, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,)
    )
    return records, properties


def gaussian_arrays(records, properties, minimum_opacity: float):
    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    scales = np.exp(
        np.column_stack(
            (records["scale_0"], records["scale_1"], records["scale_2"])
        ).astype(np.float64)
    )
    opacity_logits = np.asarray(records["opacity"], np.float64)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logits))
    dc = np.column_stack(
        (records["f_dc_0"], records["f_dc_1"], records["f_dc_2"])
    ).astype(np.float64)
    rest_names = sorted(
        (name for name in properties if name.startswith("f_rest_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    rest = None
    if len(rest_names) >= 45:
        rest = np.column_stack([records[name] for name in rest_names[:45]])
        rest = rest.astype(np.float64).reshape((-1, 3, 15))
    keep = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(dc).all(axis=1)
        & (opacity >= minimum_opacity)
    )
    return (
        xyz[keep].astype(np.float64),
        scales[keep],
        opacity[keep],
        dc[keep],
        None if rest is None else rest[keep],
    )


def evaluate_sh(dc, rest, directions):
    """Evaluate the standard 3DGS real SH basis through degree 3."""
    result = SH_C0 * dc
    if rest is None:
        return np.clip(result + 0.5, 0.0, 1.0)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    result += (
        -SH_C1 * y[:, None] * rest[:, :, 0]
        + SH_C1 * z[:, None] * rest[:, :, 1]
        - SH_C1 * x[:, None] * rest[:, :, 2]
    )
    result += (
        SH_C2[0] * (x * y)[:, None] * rest[:, :, 3]
        + SH_C2[1] * (y * z)[:, None] * rest[:, :, 4]
        + SH_C2[2] * (2.0 * zz - xx - yy)[:, None] * rest[:, :, 5]
        + SH_C2[3] * (x * z)[:, None] * rest[:, :, 6]
        + SH_C2[4] * (xx - yy)[:, None] * rest[:, :, 7]
    )
    result += (
        SH_C3[0] * (y * (3.0 * xx - yy))[:, None] * rest[:, :, 8]
        + SH_C3[1] * (x * y * z)[:, None] * rest[:, :, 9]
        + SH_C3[2] * (y * (4.0 * zz - xx - yy))[:, None] * rest[:, :, 10]
        + SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None]
        * rest[:, :, 11]
        + SH_C3[4] * (x * (4.0 * zz - xx - yy))[:, None] * rest[:, :, 12]
        + SH_C3[5] * (z * (xx - yy))[:, None] * rest[:, :, 13]
        + SH_C3[6] * (x * (xx - 3.0 * yy))[:, None] * rest[:, :, 14]
    )
    return np.clip(result + 0.5, 0.0, 1.0)


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up_sign: float = 1.0,
                   preferred_up: Optional[np.ndarray] = None):
    """Return OpenCV camera transforms with an explicit model vertical sign."""
    forward = np.asarray(target - eye, np.float64)
    forward /= np.linalg.norm(forward)
    if preferred_up is None:
        preferred_up = np.asarray((0.0, 0.0, up_sign), np.float64)
        if abs(float(np.dot(forward, preferred_up))) > 0.98:
            preferred_up = np.asarray((0.0, 1.0, 0.0), np.float64)
    else:
        preferred_up = np.asarray(preferred_up, np.float64)
        if preferred_up.shape != (3,) or not np.isfinite(preferred_up).all():
            raise SplatRenderError('Camera up vector must contain three finite values')
        norm = float(np.linalg.norm(preferred_up))
        if norm < 1e-12:
            raise SplatRenderError('Camera up vector must be nonzero')
        preferred_up /= norm
        if abs(float(np.dot(forward, preferred_up))) > 0.98:
            raise SplatRenderError('Camera up vector is parallel to the view direction')
    right = np.cross(forward, preferred_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.vstack((right, down, forward))
    translation = -rotation @ eye
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return world_to_camera, np.linalg.inv(world_to_camera)


def camera_positions(center, radius, azimuth_count, elevations, distances,
                     azimuths_per_elevation=None, up_sign=1.0,
                     azimuth_offset_deg=0.0):
    for distance_index, multiplier in enumerate(distances):
        distance = radius * multiplier
        for elevation_index, elevation_deg in enumerate(elevations):
            elevation = math.radians(elevation_deg)
            count = (azimuths_per_elevation[elevation_index]
                     if azimuths_per_elevation is not None else azimuth_count)
            for azimuth_index in range(count):
                azimuth_deg = (azimuth_offset_deg + 360.0 * azimuth_index / count) % 360.0
                azimuth = math.radians(azimuth_deg)
                offset = distance * np.asarray(
                    (
                        math.cos(elevation) * math.sin(azimuth),
                        math.cos(elevation) * math.cos(azimuth),
                        up_sign * math.sin(elevation),
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


def render_view(
    xyz,
    scales,
    opacity,
    dc,
    rest,
    eye,
    target,
    camera_matrix,
    width,
    height,
    maximum_radius,
    radius_scale,
    surface_thickness,
    up_sign=1.0,
    preferred_up=None,
):
    world_to_camera, camera_to_world = look_at_opencv(
        eye, target, up_sign, preferred_up)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    # einsum avoids an Accelerate/BLAS warning observed for large non-aligned
    # memmap-derived arrays on macOS while computing the same R @ X + t.
    camera = np.einsum("ni,ji->nj", xyz, rotation) + translation
    depth = camera[:, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    visible = depth > 1e-3
    u = fx * camera[:, 0] / np.maximum(depth, 1e-9) + cx
    v = fy * camera[:, 1] / np.maximum(depth, 1e-9) + cy
    visible &= (u >= -maximum_radius) & (u < width + maximum_radius)
    visible &= (v >= -maximum_radius) & (v < height + maximum_radius)
    indices = np.flatnonzero(visible)
    if len(indices) == 0:
        raise SplatRenderError("No Gaussians project into this virtual view")
    u = np.rint(u[indices]).astype(np.int32)
    v = np.rint(v[indices]).astype(np.int32)
    z = depth[indices]
    projected_scale = np.max(scales[indices], axis=1) * (fx / z) * radius_scale
    radii = np.clip(np.ceil(projected_scale), 1, maximum_radius).astype(np.int32)
    directions = xyz[indices] - eye[None, :]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    colors = evaluate_sh(
        dc[indices], None if rest is None else rest[indices], directions
    )

    pixel_parts = []
    depth_parts = []
    source_parts = []
    weight_parts = []
    for dy in range(-maximum_radius, maximum_radius + 1):
        for dx in range(-maximum_radius, maximum_radius + 1):
            inside_radius = dx * dx + dy * dy <= radii * radii
            px = u + dx
            py = v + dy
            keep = (
                inside_radius
                & (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
            )
            if not np.any(keep):
                continue
            local = np.flatnonzero(keep)
            pixel_parts.append(py[keep].astype(np.int64) * width + px[keep])
            depth_parts.append(z[keep])
            source_parts.append(local)
            sigma = np.maximum(radii[keep] * 0.55, 0.75)
            spatial = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
            weight_parts.append(opacity[indices[local]] * spatial)
    pixels = np.concatenate(pixel_parts)
    depths = np.concatenate(depth_parts)
    sources = np.concatenate(source_parts)
    weights = np.concatenate(weight_parts)

    # Blend Gaussians belonging to the first visible surface.  A small depth
    # slab reproduces the smoothing effect of alpha compositing without mixing
    # the opposite side of the object into the same pixel.
    pixel_count = width * height
    nearest = np.full(pixel_count, np.inf, dtype=np.float64)
    np.minimum.at(nearest, pixels, depths)
    surface = depths <= nearest[pixels] + surface_thickness
    pixels = pixels[surface]
    sources = sources[surface]
    depths = depths[surface]
    weights = weights[surface]
    weight_sum = np.zeros(pixel_count, dtype=np.float64)
    np.add.at(weight_sum, pixels, weights)
    valid_pixels = np.flatnonzero(weight_sum > 1e-12)

    image = np.full((height, width, 3), 245, dtype=np.uint8)
    xyz_buffer = np.full((height, width, 3), np.nan, dtype=np.float32)
    depth_buffer = np.zeros((height, width), dtype=np.float32)
    yy = valid_pixels // width
    xx = valid_pixels % width
    color_sum = np.zeros((pixel_count, 3), dtype=np.float64)
    xyz_sum = np.zeros((pixel_count, 3), dtype=np.float64)
    depth_sum = np.zeros(pixel_count, dtype=np.float64)
    selected_colors = colors[sources]
    selected_xyz = xyz[indices[sources]]
    for component in range(3):
        np.add.at(color_sum[:, component], pixels, selected_colors[:, component] * weights)
        np.add.at(xyz_sum[:, component], pixels, selected_xyz[:, component] * weights)
    np.add.at(depth_sum, pixels, depths * weights)
    denominator = weight_sum[valid_pixels, None]
    rgb = color_sum[valid_pixels] / denominator
    image[yy, xx] = np.rint(np.clip(rgb[:, ::-1], 0, 1) * 255.0).astype(np.uint8)
    xyz_buffer[yy, xx] = (xyz_sum[valid_pixels] / denominator).astype(np.float32)
    depth_buffer[yy, xx] = (depth_sum[valid_pixels] / weight_sum[valid_pixels]).astype(
        np.float32
    )
    return image, depth_buffer, xyz_buffer, world_to_camera, camera_to_world


def run(args) -> int:
    if args.width <= 0 or args.height <= 0 or args.azimuths <= 0:
        raise SplatRenderError("Width, height, and azimuth count must be positive")
    if not args.elevations or not args.distances or min(args.distances) <= 1:
        raise SplatRenderError("Elevations are required and distances must be > 1")
    if args.azimuths_per_elevation is not None and (
        len(args.azimuths_per_elevation) != len(args.elevations)
        or min(args.azimuths_per_elevation) <= 0
    ):
        raise SplatRenderError("--azimuths-per-elevation needs one positive count per elevation")
    if not math.isfinite(args.azimuth_offset_deg):
        raise SplatRenderError("--azimuth-offset-deg must be finite")
    if args.top_rolls is not None:
        rolls = [float(angle) % 360. for angle in args.top_rolls]
        if (len(args.elevations) != 1 or abs(args.elevations[0]-90.) > 1e-6
            or not rolls or not all(math.isfinite(angle) for angle in args.top_rolls)
            or len({round(angle, 6) for angle in rolls}) != len(rolls)):
            raise SplatRenderError('--top-rolls needs elevation 90 and unique finite angles')
    if not 1 < args.fov_deg < 179:
        raise SplatRenderError("--fov-deg must be between 1 and 179")
    if not 0 <= args.minimum_opacity < 1:
        raise SplatRenderError("--minimum-opacity must be in [0, 1)")
    if not 0 < args.geometry_min_alpha <= 1 or not math.isfinite(args.geometry_max_std) or args.geometry_max_std <= 0:
        raise SplatRenderError('Invalid geometry confidence thresholds')
    records, properties = read_gaussian_ply(args.ply)
    xyz, scales, opacity, dc, rest = gaussian_arrays(
        records, properties, args.minimum_opacity
    )
    if len(xyz) < 100:
        raise SplatRenderError("Too few usable Gaussians after opacity filtering")
    bounds = np.asarray((xyz.min(axis=0), xyz.max(axis=0)), np.float64)
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5)
    vertical_fov = math.radians(args.fov_deg)
    focal = 0.5 * args.height / math.tan(vertical_fov * 0.5)
    camera_matrix = np.asarray(
        ((focal, 0, (args.width - 1) * 0.5),
         (0, focal, (args.height - 1) * 0.5),
         (0, 0, 1)),
        np.float64,
    )
    up_sign = -1.0 if args.up_axis == "negative-z" else 1.0
    if args.top_rolls is None:
        positions = [(*position, None, None) for position in camera_positions(
            center, radius, args.azimuths, args.elevations, args.distances,
            args.azimuths_per_elevation, up_sign, args.azimuth_offset_deg)]
    else:
        positions = []
        for distance_index, multiplier in enumerate(args.distances):
            distance = radius * multiplier
            eye = center + np.asarray((0., 0., up_sign * distance))
            for roll_index, roll_deg in enumerate(args.top_rolls):
                roll = math.radians(roll_deg)
                up = np.asarray((math.sin(roll), math.cos(roll), 0.), np.float64)
                positions.append((distance_index, 0, roll_index, 0., 90.,
                                  distance, eye, up, float(roll_deg) % 360.))
    if (args.output / 'manifest.json').exists():
        raise SplatRenderError('Output already contains a manifest; use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"3DGS: {args.ply}\nGaussians: {len(xyz):,}/{len(records):,}\n"
        f"Bounds: {bounds.tolist()}\nRendering {len(positions)} reference views..."
    )
    planned_views = []
    for view_index, position in enumerate(positions):
        planned_views.append(
            {
                "view_id": view_index,
                "image_name": f"view_{view_index:04d}.png",
                "eye": position[6].tolist(),
                **({"up": position[7].tolist()} if position[7] is not None else {}),
            }
        )
    if args.rgb_renderer == "browser":
        node = args.node
        if node is None:
            discovered = shutil.which("node")
            node = Path(discovered) if discovered else DEFAULT_NODE
        if not node.is_file():
            raise SplatRenderError(
                "Node.js not found; install Node.js or pass --node /path/to/node"
            )
        if not args.browser_renderer.is_file():
            raise SplatRenderError(
                f"3DGS browser renderer not found: {args.browser_renderer}"
            )
        render_plan = {
            "ply": str(args.ply.resolve()),
            "width": args.width,
            "height": args.height,
            "vertical_fov_deg": args.fov_deg,
            "target": center.tolist(),
            "up": [0.0, 0.0, -1.0 if args.up_axis == "negative-z" else 1.0],
            "views": planned_views,
            "geometry_renderer": args.geometry_mode,
        }
        plan_path = args.output / "render_plan.json"
        with plan_path.open("w", encoding="utf-8") as target:
            json.dump(render_plan, target, ensure_ascii=False, indent=2)
        print("Rendering true Gaussian RGB views with Chrome/Metal...")
        try:
            subprocess.run(
                [
                    str(node),
                    str(args.browser_renderer),
                    str(plan_path),
                    str(args.output),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SplatRenderError(
                f"True 3DGS browser rendering failed (exit {exc.returncode})"
            ) from exc
    views = []
    for view_index, (
        distance_index,
        elevation_index,
        azimuth_index,
        azimuth_deg,
        elevation_deg,
        distance,
        eye,
        camera_up,
        roll_deg,
    ) in enumerate(positions):
        name = f"view_{view_index:04d}"
        image_name = f"{name}.png"
        data_name = f"{name}.npz"
        geometry = {}
        if args.rgb_renderer == "approximate":
            image, depth, xyz_buffer, world_to_camera, camera_to_world = render_view(
                xyz, scales, opacity, dc, rest, eye, center, camera_matrix,
                args.width, args.height, args.maximum_radius, args.radius_scale,
                args.surface_thickness,
                up_sign, camera_up)
            if not cv2.imwrite(str(args.output / image_name), image):
                raise SplatRenderError(f"Could not write {image_name}")
        else:
            rendered = cv2.imread(str(args.output / image_name), cv2.IMREAD_COLOR)
            if rendered is None or rendered.shape[:2] != (args.height, args.width):
                raise SplatRenderError(
                    f"True 3DGS renderer did not produce a valid {image_name}"
                )
            world_to_camera, camera_to_world = look_at_opencv(
                eye, center, up_sign, camera_up)
            if args.geometry_mode == 'shared-gaussian-moments':
                moments = np.fromfile(args.output / f'{name}.moments.f32', dtype='<f4')
                moments = moments.reshape(args.height, args.width, 4)
                geometry = moments_to_geometry(moments, camera_matrix, camera_to_world,
                                               args.geometry_min_alpha, args.geometry_max_std)
                depth, xyz_buffer = geometry.pop('depth'), geometry.pop('xyz')
            else:
                _, depth, legacy_xyz, _, _ = render_view(
                    xyz, scales, opacity, dc, rest, eye, center, camera_matrix,
                    args.width, args.height, args.maximum_radius, args.radius_scale,
                    args.surface_thickness,
                    up_sign, camera_up)
                valid = np.isfinite(legacy_xyz).all(axis=2) & np.isfinite(depth) & (depth > 0)
                yy, xx = np.indices(depth.shape)
                xyz_buffer = unproject(np.stack((xx, yy), axis=-1), depth,
                                       camera_matrix, camera_to_world).astype(np.float32)
                xyz_buffer[~valid] = np.nan
                geometry = {'valid': valid}
            if not geometry['valid'].any():
                raise SplatRenderError(f'No confident geometry in {image_name}')
        np.savez_compressed(
            args.output / data_name,
            depth=depth,
            xyz=xyz_buffer,
            camera_matrix=camera_matrix,
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            **geometry,
        )
        valid_pixels = int(np.isfinite(xyz_buffer).all(axis=2).sum())
        views.append(
            {
                "view_id": view_index,
                "image_name": image_name,
                "data_name": data_name,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                **({"roll_deg": roll_deg} if roll_deg is not None else {}),
                "distance": distance,
                "distance_multiplier": args.distances[distance_index],
                "eye": eye.tolist(),
                "valid_pixels": valid_pixels,
                "camera_to_world": camera_to_world.tolist(),
                "world_to_camera": world_to_camera.tolist(),
                "camera_matrix": camera_matrix.tolist(),
            }
        )
        print(
            f"[{view_index + 1:03d}/{len(positions):03d}] {image_name} "
            f"az={azimuth_deg:.1f} el={elevation_deg:.1f} pixels={valid_pixels}"
        )
    manifest = {
        "format_version": 2,
        "source_type": "3dgs",
        "ply": str(args.ply.resolve()),
        "coordinate_system": "COLMAP world coordinates (millimeters)",
        "runtime_transform": "identity",
        "up_axis": args.up_axis,
        "geometry_renderer": (args.geometry_mode if args.rgb_renderer == 'browser'
                              else "legacy-cpu-centers"),
        "azimuths_per_elevation": args.azimuths_per_elevation,
        "azimuth_offset_deg": args.azimuth_offset_deg,
        "top_rolls_deg": args.top_rolls,
        "geometry_min_alpha": args.geometry_min_alpha,
        "geometry_max_std": args.geometry_max_std,
        "geometry_note": (
            "Legacy CPU depth proxy unprojected on pixel rays; not shared RGB geometry"
            if args.geometry_mode == 'ray-legacy-depth'
            else "Composited depth estimate; not validated physical surface or CT registration"),
        "pixel_convention": "OpenCV integer pixel centers; principal point (size-1)/2",
        "width": args.width,
        "height": args.height,
        "vertical_fov_deg": args.fov_deg,
        "bounds": bounds.tolist(),
        "center": center.tolist(),
        "radius": radius,
        "minimum_opacity": args.minimum_opacity,
        "rgb_renderer": (
            "playcanvas-browser-gaussian-rasterizer"
            if args.rgb_renderer == "browser"
            else "approximate-cpu-surface"
        ),
        "views": views,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    print(f"Completed: {args.output / 'manifest.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--up-axis", choices=("positive-z", "negative-z"),
                        default="negative-z",
                        help="Model vertical direction for view sampling; does not transform XYZ")
    parser.add_argument("--azimuths", type=int, default=12)
    parser.add_argument("--azimuth-offset-deg", type=float, default=0.0)
    parser.add_argument("--top-rolls", nargs="+", type=float,
                        help="At elevation 90, render one fixed overhead camera per image roll")
    parser.add_argument("--azimuths-per-elevation", nargs="+", type=int,
                        help="Override azimuth count separately for each elevation")
    parser.add_argument(
        "--elevations", nargs="+", type=float, default=(0.0, 25.0, 50.0)
    )
    parser.add_argument(
        "--distances", nargs="+", type=float, default=(1.7, 2.5)
    )
    parser.add_argument("--minimum-opacity", type=float, default=0.005)
    parser.add_argument("--geometry-mode",
                        choices=("shared-gaussian-moments", "ray-legacy-depth"),
                        default="ray-legacy-depth",
                        help="Ray legacy mode keeps the deployed RGB/XYZ proxy protocol")
    parser.add_argument("--maximum-radius", type=int, default=3)
    parser.add_argument("--radius-scale", type=float, default=1.5)
    parser.add_argument("--surface-thickness", type=float, default=5.0)
    parser.add_argument('--geometry-min-alpha', type=float, default=0.9)
    parser.add_argument('--geometry-max-std', type=float, default=3.0,
                        help='Maximum depth std in source units (expected mm)')
    parser.add_argument(
        "--rgb-renderer",
        choices=("browser", "approximate"),
        default="browser",
        help="Use true browser Gaussian rendering (default) or the legacy approximation",
    )
    parser.add_argument("--node", type=Path)
    parser.add_argument(
        "--browser-renderer", type=Path, default=DEFAULT_BROWSER_RENDERER
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (SplatRenderError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
