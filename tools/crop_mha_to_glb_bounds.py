#!/usr/bin/env python3
"""Crop a LOCAL-binary MHA to the axis-aligned bounds stored in an aligned GLB."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import struct
from pathlib import Path


ELEMENT_BYTES = {
    "MET_CHAR": 1,
    "MET_UCHAR": 1,
    "MET_SHORT": 2,
    "MET_USHORT": 2,
    "MET_INT": 4,
    "MET_UINT": 4,
    "MET_FLOAT": 4,
    "MET_DOUBLE": 8,
}


def parse_mha_header(path: Path):
    header = {}
    with path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                raise ValueError("ElementDataFile = LOCAL was not found")
            text = line.decode("ascii").strip()
            if text and not text.startswith("#") and "=" in text:
                key, value = (part.strip() for part in text.split("=", 1))
                header[key] = value
                if key.lower() == "elementdatafile":
                    if value.upper() != "LOCAL":
                        raise ValueError("Only ElementDataFile = LOCAL is supported")
                    return header, source.tell()


def read_glb_bounds(path: Path):
    with path.open("rb") as source:
        magic, version, _ = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError("Only GLB 2.0 is supported")
        chunk_length, chunk_type = struct.unpack("<I4s", source.read(8))
        if chunk_type != b"JSON":
            raise ValueError("The first GLB chunk is not JSON")
        gltf = json.loads(source.read(chunk_length))

    scene_index = gltf.get("scene", 0)
    root_nodes = gltf.get("scenes", [])[scene_index].get("nodes", [])
    nodes = gltf.get("nodes", [])
    mesh_indices = []

    def visit(node_index):
        node = nodes[node_index]
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise ValueError("GLB node transforms must be baked before cropping")
        if "mesh" in node:
            mesh_indices.append(node["mesh"])
        for child in node.get("children", []):
            visit(child)

    for root in root_nodes:
        visit(root)
    if not mesh_indices:
        raise ValueError("The GLB contains no mesh")

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    accessors = gltf.get("accessors", [])
    meshes = gltf.get("meshes", [])
    for mesh_index in mesh_indices:
        for primitive in meshes[mesh_index].get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None:
                continue
            accessor = accessors[position_index]
            if "min" not in accessor or "max" not in accessor:
                raise ValueError("POSITION accessor must contain min/max")
            for axis in range(3):
                minimum[axis] = min(minimum[axis], accessor["min"][axis])
                maximum[axis] = max(maximum[axis], accessor["max"][axis])
    if not all(math.isfinite(value) for value in minimum + maximum):
        raise ValueError("Unable to determine GLB bounds")
    return minimum, maximum


def invert3(matrix):
    a, b, c, d, e, f, g, h, i = matrix
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise ValueError("MHA TransformMatrix is singular")
    inverse_det = 1.0 / determinant
    return [
        (e * i - f * h) * inverse_det,
        (c * h - b * i) * inverse_det,
        (b * f - c * e) * inverse_det,
        (f * g - d * i) * inverse_det,
        (a * i - c * g) * inverse_det,
        (c * d - a * f) * inverse_det,
        (d * h - e * g) * inverse_det,
        (b * g - a * h) * inverse_det,
        (a * e - b * d) * inverse_det,
    ]


def transform(matrix, vector):
    return [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]


def crop_mha(source_path: Path, output_path: Path, world_min, world_max, margin):
    header, data_offset = parse_mha_header(source_path)
    dimensions = [int(value) for value in header["DimSize"].split()]
    spacing = [float(value) for value in header["ElementSpacing"].split()]
    origin_text = header.get("Offset") or header.get("Origin") or header.get("Position")
    origin = [float(value) for value in origin_text.split()]
    direction = [float(value) for value in header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    inverse_direction = invert3(direction)
    element_type = header["ElementType"].upper()
    bytes_per_voxel = ELEMENT_BYTES.get(element_type)
    if bytes_per_voxel is None:
        raise ValueError(f"Unsupported ElementType: {element_type}")
    if header.get("CompressedData", "False").lower() == "true":
        raise ValueError("Compressed MHA payloads are not supported")

    expanded_min = [value - margin for value in world_min]
    expanded_max = [value + margin for value in world_max]
    voxel_corners = []
    for corner in itertools.product(*zip(expanded_min, expanded_max)):
        relative = [corner[axis] - origin[axis] for axis in range(3)]
        oriented = transform(inverse_direction, relative)
        voxel_corners.append([oriented[axis] / spacing[axis] for axis in range(3)])

    start = []
    end = []
    for axis in range(3):
        start.append(max(0, math.floor(min(corner[axis] for corner in voxel_corners))))
        end.append(min(dimensions[axis] - 1, math.ceil(max(corner[axis] for corner in voxel_corners))))
    cropped_dimensions = [end[axis] - start[axis] + 1 for axis in range(3)]
    if any(value <= 0 for value in cropped_dimensions):
        raise ValueError("GLB bounds do not intersect the MHA volume")

    start_physical = [start[axis] * spacing[axis] for axis in range(3)]
    origin_delta = transform(direction, start_physical)
    cropped_origin = [origin[axis] + origin_delta[axis] for axis in range(3)]

    output_header = [
        "ObjectType = Image",
        "NDims = 3",
        f"BinaryData = {header.get('BinaryData', 'True')}",
        f"BinaryDataByteOrderMSB = {header.get('BinaryDataByteOrderMSB', header.get('ElementByteOrderMSB', 'False'))}",
        "CompressedData = False",
        f"TransformMatrix = {' '.join(format(value, '.12g') for value in direction)}",
        f"Offset = {' '.join(format(value, '.12g') for value in cropped_origin)}",
        f"ElementSpacing = {' '.join(format(value, '.12g') for value in spacing)}",
        f"DimSize = {' '.join(str(value) for value in cropped_dimensions)}",
        f"ElementType = {element_type}",
        "ElementDataFile = LOCAL",
        "",
    ]

    width, height, _ = dimensions
    x_start, y_start, z_start = start
    x_count, y_count, z_count = cropped_dimensions
    row_bytes = x_count * bytes_per_voxel
    with source_path.open("rb") as source, output_path.open("wb") as output:
        output.write("\n".join(output_header).encode("ascii"))
        for z in range(z_start, z_start + z_count):
            for y in range(y_start, y_start + y_count):
                voxel_offset = (z * height * width) + (y * width) + x_start
                source.seek(data_offset + voxel_offset * bytes_per_voxel)
                row = source.read(row_bytes)
                if len(row) != row_bytes:
                    raise ValueError("Unexpected end of MHA payload")
                output.write(row)

    return {
        "source_dimensions": dimensions,
        "cropped_dimensions": cropped_dimensions,
        "voxel_start": start,
        "voxel_end": end,
        "cropped_origin": cropped_origin,
        "world_min": world_min,
        "world_max": world_max,
        "margin_mm": margin,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mha", type=Path)
    parser.add_argument("glb", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    args = parser.parse_args()
    world_min, world_max = read_glb_bounds(args.glb)
    result = crop_mha(args.mha, args.output, world_min, world_max, max(0.0, args.margin_mm))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
