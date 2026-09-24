#!/usr/bin/env python3
"""Audit pixel-ray consistency; this is NOT a physical surface accuracy test."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new report path')
    reports = []
    for folder in a.views:
        manifest = json.loads((folder/'manifest.json').read_text())
        rows = []
        for view in manifest['views']:
            with np.load(folder/view['data_name']) as data:
                xyz = data['xyz'][::4, ::4]
                yy, xx = np.indices(xyz.shape[:2])
                uv = np.stack((xx*4, yy*4), axis=-1)
                valid = np.isfinite(xyz).all(axis=2)
                world = xyz[valid].astype(np.float64)
                transform = data['world_to_camera']
                camera = np.einsum('ni,ji->nj', world, transform[:3, :3])+transform[:3, 3]
                projection = np.einsum('ni,ji->nj', camera, data['camera_matrix'])
                positive = camera[:, 2] > 0
                error = np.linalg.norm(projection[positive, :2]/projection[positive, 2:3]-uv[valid][positive], axis=1)
                rows.append(dict(view=view['image_name'], valid_fraction=float(valid.mean()),
                    reprojection_p50=float(np.median(error)) if len(error) else None,
                    reprojection_p95=float(np.percentile(error, 95)) if len(error) else None))
        reports.append(dict(views=str(folder), geometry=manifest.get('geometry_renderer', 'legacy'),
                            rows=rows, clinical_accuracy_validated=False))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
