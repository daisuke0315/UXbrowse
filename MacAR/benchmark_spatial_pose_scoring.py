"""Compare old/new rankings on identical exhaustive registered-photo hypotheses.

One fresh MPS process per query. This isolates ranking, not adaptive-search
end-to-end latency. Existing photos/maps/app settings are never modified.
"""

from macar_paths import DATA_DIR
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked_selection(candidates, spatial):
    from registered_aliked_live import rotation_angle
    from spatial_pose_scoring import registered_hypothesis_rank, comparable_support
    pool = []
    for candidate in candidates:
        if 'pose' not in candidate:
            continue
        quality = copy.deepcopy(candidate['quality'])
        quality['reasons'] = [r for r in quality['reasons'] if r != 'ambiguous_pose']
        pose = SimpleNamespace(rvec=np.asarray(candidate['pose']['rvec']),
            tvec=np.asarray(candidate['pose']['tvec']),
            inlier_indices=np.arange(candidate['inliers']), mean_error=quality['mean_error'])
        pool.append((pose, [], quality, candidate['name'], candidate['pose_rotation_deg']))
    rank = registered_hypothesis_rank if spatial else lambda h: (
        h[2]['passed'], len(h[0].inlier_indices), -h[0].mean_error)
    pool.sort(key=rank, reverse=True)
    if not pool:
        return dict(status='FAIL', selected=None, reasons=['no_pnp'])
    best = pool[0]
    p, _, quality, name, _ = best
    conflicts = []
    if quality['passed']:
        for other in pool[1:]:
            competitive = comparable_support(other, best) if spatial else (
                other[2]['passed'] and len(other[0].inlier_indices) >= .7*len(p.inlier_indices))
            if not competitive:
                continue
            R, R2 = cv2.Rodrigues(p.rvec)[0], cv2.Rodrigues(other[0].rvec)[0]
            C, C2 = -R.T@p.tvec, -R2.T@other[0].tvec
            if rotation_angle(R, R2) > 15. or np.linalg.norm(C-C2) > 30.:
                conflicts.append(other[3])
    return dict(status='PASS' if quality['passed'] and not conflicts else 'FAIL',
        selected=name, inliers=len(p.inlier_indices), mean_error_px=p.mean_error,
        spatial_score=quality['spatial_score'], rvec=p.rvec.tolist(), tvec=p.tvec.tolist(),
        reasons=quality['reasons']+(['ambiguous_pose'] if conflicts else []), conflicts=conflicts)


def worker(output, index):
    from browser_photo import decode_photo
    from registered_aliked_live import RegisteredALIKEDLiveLocalizer, rotation_angle
    from offline_pnp_test import (load_reference_images, load_camera_parameters,
                                 camera_matrix_and_distortion, quaternion_to_rotation)
    from realtime_pnp import approximate_camera_matrix
    cv2.setNumThreads(4)
    metadata = json.loads((output/'inputs.json').read_text())
    query = metadata['queries'][index]
    photo = Path(query['path'])
    map_path = Path(metadata['map_path'])
    engine = RegisteredALIKEDLiveLocalizer(map_path, Path(metadata['views_path']),
        device='mps', max_keypoints=2000, top_k=3, resize=1024,
        fundamental_threshold=3., ransac_threshold=4., max_reprojection_error=3.,
        search_policy='exhaustive')
    engine.query_resize = 1600
    summary = json.loads((map_path/'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    image = decode_photo(photo.read_bytes())
    source = query['kind'] == 'source'
    if source:
        record = records[photo.name]
        frame = int(photo.stem)
        engine.registered = {name: item for name, item in engine.registered.items()
                             if abs(int(Path(name).stem)-frame) > 5}
        cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
        K, dist = camera_matrix_and_distortion(np, cameras[record.camera_id], params[record.camera_id])
        true_R = quaternion_to_rotation(np, record.pose)
        true_C = -true_R.T@np.asarray(record.pose[4:])
    else:
        K = approximate_camera_matrix(image.shape[1], image.shape[0], 72.)
        dist = np.zeros(5)

    def truth(selection):
        if selection['selected'] is None:
            return
        R = cv2.Rodrigues(np.asarray(selection['rvec']))[0]
        C = -R.T@np.asarray(selection['tvec'])
        if source:
            angle = rotation_angle(R, true_R)
            distance = float(np.linalg.norm(C-true_C))
            selection.update(rotation_error_deg=angle, center_error_units=distance,
                center_error_object_fraction=distance/engine.scene_scale,
                pose_correct=bool(angle <= 5. and distance <= .05*engine.scene_scale))
        else:
            direction = C-np.asarray(engine.manifest['center'])
            azimuth = float(np.degrees(np.arctan2(direction[0], direction[1]))%360)
            frame = int(photo.stem.split('_')[1])
            start = 90 if frame <= 2591 else 0 if frame <= 2594 else 270 if frame <= 2597 else 180
            selection.update(estimated_azimuth_deg=azimuth,
                             face_correct=bool(0 <= (azimuth-start)%360 <= 90))

    rows, workspace = [], {}
    for top_k in (3, 10):
        engine.top_k = top_k
        cv2.setRNGSeed(0)
        started = time.perf_counter()
        result, _ = engine._localize_once(image, K, dist, workspace=workspace)
        elapsed = (time.perf_counter()-started)*1000
        diagnostics = copy.deepcopy(engine.last_diagnostics)
        old = ranked_selection(diagnostics['candidates'], False)
        new = ranked_selection(diagnostics['candidates'], True)
        if new['status'] != result.status or new['selected'] != diagnostics.get('selected'):
            raise AssertionError('Diagnostic ranking differs from production result')
        truth(old); truth(new)
        rows.append(dict(top_k=top_k, old=old, new=new, exhaustive_wall_ms=elapsed,
                         diagnostics=diagnostics))
        if old['status'] == new['status'] == 'PASS':
            break
        # Same descriptors/ROI; expand both rankings to an identical Top10 pool.
        workspace = dict(query=workspace['query'])
    dump(output/f'query_{index:02d}.json', dict(index=index, name=photo.name,
        kind=query['kind'], photo_sha256=sha(photo), scene_scale=engine.scene_scale, rows=rows))
    print(json.dumps(dict(index=index, name=photo.name, old=rows[-1]['old']['status'],
                         new=rows[-1]['new']['status'], changed=rows[-1]['old']['selected'] !=
                         rows[-1]['new']['selected'])), flush=True)


def summarize(output):
    queries = [json.loads(path.read_text()) for path in sorted(output.glob('query_*.json'))]
    stats = {}
    for kind in ('new', 'source'):
        selected = [q for q in queries if q['kind'] == kind]
        stats[kind] = dict(count=len(selected),
            changed=sum(q['rows'][-1]['old']['selected'] != q['rows'][-1]['new']['selected'] for q in selected))
        for method in ('old', 'new'):
            rows = [q['rows'][-1][method] for q in selected]
            stats[kind][method] = dict(passed=sum(r['status'] == 'PASS' for r in rows),
                correct_passed=sum(r['status'] == 'PASS' and r.get('pose_correct', r.get('face_correct', False)) for r in rows))
            if kind == 'source':
                accepted = [r for r in rows if r['status'] == 'PASS']
                stats[kind][method].update(rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in accepted])),
                    rotation_max_deg=max(r['rotation_error_deg'] for r in accepted),
                    center_median_units=float(np.median([r['center_error_units'] for r in accepted])),
                    center_max_units=max(r['center_error_units'] for r in accepted))
    scores = [c['timings_ms']['spatial_scoring'] for q in queries for row in q['rows']
              for c in row['diagnostics']['candidates'] if 'pose' in c]
    stats['scoring_ms'] = dict(median=float(np.median(scores)), p95=float(np.percentile(scores, 95)),
                               count=len(scores))
    dump(output/'summary.json', dict(stats=stats, queries=queries))
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker is not None:
        worker(output, args.worker)
        return
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    previous = json.loads((DATA_DIR/'output_descriptor_comparison_20261003_isolated/inputs.json').read_text())
    from realtime_pnp import DEFAULT_SPLAT_VIEWS
    previous.update(views_path=str(DEFAULT_SPLAT_VIEWS),
        views_manifest_sha256=sha(DEFAULT_SPLAT_VIEWS/'manifest.json'),
        map_sha256=sha(Path(previous['map_path'])/'object_aliked_map.npz'),
        scoring_code_sha256={file: sha(ROOT/file) for file in (
            'spatial_pose_scoring.py', 'registered_aliked_live.py', 'benchmark_spatial_pose_scoring.py')},
        scoring_comparison='Same exhaustive Top3 pool; Top10 if either ranking fails; no XYZ-only fallback. All query features/ROI/QC/candidates identical between rankings. Pure ranking comparison, not adaptive wall-time comparison.')
    dump(output/'inputs.json', previous)
    for index in range(len(previous['queries'])):
        with (output/f'query_{index:02d}.log').open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                            '--worker', str(index)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed', index+1, '/', len(previous['queries']), flush=True)
    summarize(output)


if __name__ == '__main__':
    main()
