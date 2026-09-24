#!/usr/bin/env python3
"""Re-evaluate saved Gaussian depth confidence without re-rendering RGB."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from splat_geometry import moments_to_geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--min-alpha', default=.9, type=float)
    p.add_argument('--max-std', required=True, type=float)
    a = p.parse_args()
    if a.output.exists() or not 0 < a.min_alpha <= 1 or a.max_std <= 0:
        p.error('Use a new output directory and positive confidence parameters')
    manifest = json.loads((a.source/'manifest.json').read_text())
    if manifest.get('geometry_renderer') != 'shared-gaussian-moments':
        p.error('Source must contain shared Gaussian depth moments')
    a.output.mkdir(parents=True)
    for view in manifest['views']:
        with np.load(a.source/view['data_name']) as original:
            k, pose = original['camera_matrix'], original['camera_to_world']
            world_to_camera = original['world_to_camera']
        path = a.source / Path(view['image_name']).with_suffix('.moments.f32')
        moments = np.fromfile(path, '<f4').reshape(manifest['height'], manifest['width'], 4)
        geometry = moments_to_geometry(moments, k, pose, a.min_alpha, a.max_std)
        np.savez_compressed(a.output/view['data_name'], **geometry,
                            camera_matrix=k, camera_to_world=pose, world_to_camera=world_to_camera)
        shutil.copy2(a.source/view['image_name'], a.output/view['image_name'])
        view['valid_pixels'] = int(geometry['valid'].sum())
    manifest.update(geometry_min_alpha=a.min_alpha, geometry_max_std=a.max_std,
                    geometry_source=str(a.source.resolve()))
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Images and extractor fingerprint are identical; only geometry changed.
    if (a.source/'.aliked_cache').exists():
        shutil.copytree(a.source/'.aliked_cache', a.output/'.aliked_cache')


if __name__ == '__main__':
    main()
