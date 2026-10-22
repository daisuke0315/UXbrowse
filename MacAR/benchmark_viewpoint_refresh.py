"""Replay saved RC poses; measures trigger decisions, not localization accuracy."""

from macar_paths import DATA_DIR
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace

import cv2
import numpy as np

from background_refresh import BackgroundRefresh
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                             camera_matrix_and_distortion, quaternion_to_rotation)
from viewpoint_refresh import ViewpointRefresh, camera_center


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh result JSON path')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a fresh output path')
    metadata = json.loads((DATA_DIR/'output_spatial_scoring_20261004/inputs.json').read_text())
    map_path = Path(metadata['map_path'])
    summary = json.loads((map_path/'summary.json').read_text())
    manifest_path = DATA_DIR/'output_3dgs_reference_views_72_upper_negz_v2/manifest.json'
    center = np.asarray(json.loads(manifest_path.read_text())['center'])
    with np.load(map_path/'object_aliked_map.npz', allow_pickle=False) as data:
        xyz = data['xyz'].copy()
    records = load_reference_images(Path(summary['input_files']['images']))
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
    rows, timings, builds = [], [], []
    for query in metadata['queries']:
        if query['kind'] != 'source':
            continue
        photo = Path(query['path'])
        record = records[photo.name]
        image = cv2.imread(str(photo))
        if image is None:
            raise RuntimeError('Missing source image: '+str(photo))
        matrix, distortion = camera_matrix_and_distortion(np, cameras[record.camera_id],
                                                         params[record.camera_id])
        rotation = quaternion_to_rotation(np, record.pose)
        anchor = SimpleNamespace(rvec=cv2.Rodrigues(rotation)[0].reshape(3),
                                 tvec=np.asarray(record.pose[4:]))
        position = camera_center(anchor)
        roll = cv2.Rodrigues(np.array([0., 0., np.radians(20.)]))[0] @ rotation
        direction = position-center
        axis = np.cross(direction, [0., 0., 1.])
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(direction, [0., 1., 0.])
        orbit = center+cv2.Rodrigues(axis/np.linalg.norm(axis)*np.radians(20.))[0] @ direction
        cases = [('stationary', anchor),
                 ('roll_20deg', SimpleNamespace(rvec=cv2.Rodrigues(roll)[0].reshape(3), tvec=-roll@position)),
                 ('orbit_20deg_same_rotation', SimpleNamespace(rvec=anchor.rvec.copy(), tvec=-rotation@orbit))]
        start = time.perf_counter()
        monitor = ViewpointRefresh(xyz, center)
        builds.append((time.perf_counter()-start)*1000.)
        for name, pose in cases:
            reasons = monitor.reasons(pose, anchor, matrix, distortion, image.shape, 12.)
            for _ in range(20):
                start = time.perf_counter()
                monitor.reasons(pose, anchor, matrix, distortion, image.shape, 12.)
                timings.append((time.perf_counter()-start)*1000.)
            delta = cv2.Rodrigues(cv2.Rodrigues(pose.rvec)[0] @ rotation.T)[0]
            rows.append(dict(photo=photo.name, case=name,
                old_rotation_trigger=bool(np.degrees(np.linalg.norm(delta)) >= 12.),
                new_reasons=reasons, diagnostic=monitor.info.copy()))
    old = runpy.run_path(str(DATA_DIR/'output_viewpoint_refresh_20261004/background_refresh.py.before'))['BackgroundRefresh']
    submissions = {}
    for name, scheduler_type in [('old', old), ('new', BackgroundRefresh)]:
        scheduler = scheduler_type(last_completed=0., last_global=0., search_ms=200.)
        busy_until = None
        jobs = []
        for frame in range(1, 151):
            now = frame/15.
            if busy_until is not None and now >= busy_until:
                scheduler.completed(now, 200.)
                scheduler.accepted()
                busy_until = None
            decision = scheduler.decide(now, busy=busy_until is not None, weak_reasons=[])
            if decision:
                scheduler.submitted(decision, now)
                busy_until = now+.2
                jobs.append(now)
        submissions[name] = jobs
    result = dict(map_points=len(xyz),
        map_sha256=hashlib.sha256((map_path/'object_aliked_map.npz').read_bytes()).hexdigest(),
        manifest=str(manifest_path), object_center=center.tolist(), source_poses=len(rows)//3, rows=rows,
        judge_ms=dict(median=float(np.median(timings)), p95=float(np.percentile(timings, 95)),
                      max=max(timings), samples=len(timings)),
        build_ms=dict(median=float(np.median(builds)), max=max(builds)),
        scheduling_simulation=dict(seconds=10., fps=15., assumed_worker_ms=200., safety_due=False,
                                   submissions=submissions),
        code_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                     ['viewpoint_refresh.py', 'glb_virtual_live.py', 'background_refresh.py', 'tracking_handoff.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(source_poses=result['source_poses'], judge_ms=result['judge_ms'],
                          submissions={key: len(value) for key, value in submissions.items()})))


if __name__ == '__main__':
    main()
