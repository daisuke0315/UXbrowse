#!/usr/bin/env python3
"""Interactively inspect a COLMAP Points3D text file or a feature-map NPZ."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

from object_point_filter import (
    FilterError,
    FocusBox,
    PointRecord,
    ReservoirSampler,
    parse_point_line,
    show_point_cloud,
)


IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _bounds_box(minimum, maximum) -> FocusBox:
    center = tuple((minimum[index] + maximum[index]) * 0.5 for index in range(3))
    size = tuple(max(maximum[index] - minimum[index], 1e-6) for index in range(3))
    return FocusBox(center, size, IDENTITY_ROTATION, None)


def _load_points3d(path: Path, maximum_points: int):
    sampler = ReservoirSampler(maximum_points, 91367)
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                record = parse_point_line(line, line_number)
                sampler.add(record)
                count += 1
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], record.xyz[axis])
                    maximum[axis] = max(maximum[axis], record.xyz[axis])
    except (OSError, UnicodeError) as exc:
        raise FilterError(f"Cannot read point map: {path}: {exc}") from exc
    if count == 0:
        raise FilterError(f"Point map has no 3D points: {path}")
    return sampler.items, count, _bounds_box(minimum, maximum)


def _load_feature_map(path: Path, maximum_points: int):
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise FilterError("Feature-map viewing requires NumPy") from exc
    try:
        with np.load(path, allow_pickle=False) as data:
            if "xyz" not in data.files or "point3d_ids" not in data.files:
                raise FilterError("Feature map must contain xyz and point3d_ids arrays")
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            point_ids = np.asarray(data["point3d_ids"], dtype=np.int64).reshape(-1)
    except (OSError, ValueError) as exc:
        raise FilterError(f"Cannot read feature map: {path}: {exc}") from exc
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(point_ids):
        raise FilterError("Feature-map xyz/point3d_ids shapes are inconsistent")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise FilterError("Feature map has no finite 3D points")

    # Multiple reference images may store descriptors for the same 3D point.
    # Display every POINT3D_ID once so density reflects the actual 3D map.
    _, first_indices = np.unique(point_ids, return_index=True)
    first_indices.sort()
    unique_xyz = xyz[first_indices]
    unique_ids = point_ids[first_indices]
    sampler = ReservoirSampler(maximum_points, 91367)
    for point_id, point in zip(unique_ids, unique_xyz):
        sampler.add(
            PointRecord(
                int(point_id),
                (float(point[0]), float(point[1]), float(point[2])),
                (80, 220, 255),
            )
        )
    minimum = unique_xyz.min(axis=0)
    maximum = unique_xyz.max(axis=0)
    return (
        sampler.items,
        int(len(unique_xyz)),
        _bounds_box(minimum.tolist(), maximum.tolist()),
    )


def load_point_map(path: Path, maximum_points: int):
    if not path.is_file():
        raise FilterError(f"Point map not found: {path}")
    if maximum_points <= 0:
        raise FilterError("--max-points must be greater than 0")
    if path.suffix.lower() == ".npz":
        return _load_feature_map(path, maximum_points)
    return _load_points3d(path, maximum_points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View COLMAP Points3D.txt or object_features.npz in interactive 3D."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=150_000)
    parser.add_argument("--preview", type=Path, default=Path("point_map_preview.png"))
    parser.add_argument("--no-window", action="store_true", help="Only save the PNG preview")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        points, total, box = load_point_map(args.input, args.max_points)
        print(f"Input: {args.input}")
        print(f"Unique 3D points: {total}")
        print(f"Displayed sample: {len(points)}")
        print("Bounds center: " + " ".join(f"{value:.9g}" for value in box.center))
        print("Bounds size: " + " ".join(f"{value:.9g}" for value in box.size))
        show_point_cloud(
            points,
            (),
            box,
            args.preview,
            interactive=not args.no_window,
        )
        print(f"Preview: {args.preview}")
        return 0
    except FilterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
