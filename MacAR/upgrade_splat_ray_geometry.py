#!/usr/bin/env python3
"""Separate existing GS appearance from a ray-consistent legacy depth proxy.

Fixes pixel-ray inconsistency ONLY. Legacy depth is still approximate and is
not converted into ground truth by unprojection. Original assets are preserved.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from splat_geometry import unproject


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    manifest = json.loads((a.source/'manifest.json').read_text())
    if manifest.get('source_type') != '3dgs':
        p.error('This migration is for existing 3DGS reference views only')
    a.output.mkdir(parents=True)
    for view in manifest['views']:
        with np.load(a.source/view['data_name']) as old:
            arrays = {key: old[key].copy() for key in old.files}
        depth = arrays['depth']
        # Browser viewport center is at (w/2,h/2) in edge coordinates;
        # OpenCV feature coordinates use integer pixel centers.
        matrix = arrays['camera_matrix'].copy()
        if manifest.get('format_version', 1) == 1:
            matrix[0, 2] -= .5
            matrix[1, 2] -= .5
        valid = np.isfinite(arrays['xyz']).all(axis=2) & np.isfinite(depth) & (depth > 0)
        yy, xx = np.indices(depth.shape)
        xyz = unproject(np.stack((xx, yy), axis=-1), depth, matrix, arrays['camera_to_world'])
        xyz[~valid] = np.nan
        arrays.update(xyz=xyz.astype(np.float32), valid=valid, camera_matrix=matrix)
        np.savez_compressed(a.output/view['data_name'], **arrays)
        shutil.copy2(a.source/view['image_name'], a.output/view['image_name'])
        view['camera_matrix'] = matrix.tolist()
    manifest.update(format_version=2, geometry_renderer='ray-consistent-legacy-depth',
        geometry_source=str(a.source.resolve()),
        geometry_note='Approximate CPU depth proxy, not shared RGB geometry; pixel-ray defect corrected only',
        pixel_convention='OpenCV integer pixel centers; principal point (size-1)/2')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    if (a.source/'.aliked_cache').exists():
        shutil.copytree(a.source/'.aliked_cache', a.output/'.aliked_cache')


if __name__ == '__main__':
    main()
