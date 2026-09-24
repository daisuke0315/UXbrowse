#!/usr/bin/env python3
"""Render inferred poses for visual QC; never labels a pose as ground truth."""
import argparse
import json
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np
from generate_3dgs_reference_views import DEFAULT_NODE, DEFAULT_BROWSER_RENDERER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--views', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--references', nargs='+', required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory')
    run = json.loads(a.results.read_text())['results'][0]
    manifest = json.loads((a.views/'manifest.json').read_text())
    image = cv2.imread(run['query'])
    image = cv2.resize(image, (1280, round(image.shape[0]*1280/image.shape[1])))
    h, w = image.shape[:2]
    # Matches benchmark_splat_pipeline's explicitly approximate live intrinsics.
    f = w*.5/math.tan(math.radians(65)*.5)
    vfov = math.degrees(2*math.atan(h*.5/f))
    views = []
    for candidate in run['stages']['candidates']:
        if candidate['image_name'] not in a.references or candidate['pose'] is None:
            continue
        pose = candidate['pose']
        rotation = cv2.Rodrigues(np.asarray(pose['rvec']))[0]
        eye = -rotation.T @ np.asarray(pose['tvec'])
        target = eye + rotation.T @ np.array([0., 0., 100.])
        up = rotation.T @ np.array([0., -1., 0.])
        views.append(dict(image_name=candidate['image_name'], eye=eye.tolist(),
                          target=target.tolist(), up=up.tolist()))
    if not views:
        p.error('No candidate poses available')
    a.output.mkdir(parents=True)
    plan = dict(ply=manifest['ply'], width=w, height=h, vertical_fov_deg=vfov,
                target=views[0]['target'], up=views[0]['up'], views=views)
    plan_path = a.output/'render_plan.json'
    plan_path.write_text(json.dumps(plan, indent=2))
    subprocess.run([str(DEFAULT_NODE), str(DEFAULT_BROWSER_RENDERER), str(plan_path),
                    str(a.output)], check=True)
    panels = [image]
    for view in views:
        rendered = cv2.imread(str(a.output/view['image_name']))
        panels.append(rendered)
    for panel, title in zip(panels, ['Query']+[v['image_name'] for v in views]):
        cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
    cv2.imwrite(str(a.output/'comparison.jpg'), np.hstack(panels))


if __name__ == '__main__':
    main()
