"""Same saved ALIKED features/pairs: LightGlue versus fixed-pose matching.

Audited landmark IDs provide a consistency proxy, not independent ground truth.
No current map or runtime selection is changed.
"""

from macar_paths import DATA_DIR
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from aliked_lightglue_pnp import _load_models, _match_pair
from build_aliked_object_map import epipolar_inliers
from enrich_registered_aliked_map import enrichment_pairs
from known_pose_matching import prepare_features, match_features
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                             camera_matrix_and_distortion, quaternion_to_rotation)


def geometric_quality(pairs, a, b, first, second, xyz, records, matrices, distortions):
    """Cross-project each existing landmark; different IDs can be duplicate XYZ."""
    left, right = a['point_ids'][pairs[:, 0]], b['point_ids'][pairs[:, 1]]
    valid = (left >= 0) & (right >= 0)
    if not valid.any():
        return dict(tested=0, passed=0)
    errors = []
    for name, ids, pixels in [(second, left[valid], b['full_uv'][pairs[valid, 1]]),
                              (first, right[valid], a['full_uv'][pairs[valid, 0]])]:
        rec = records[name]
        rotation = quaternion_to_rotation(np, rec.pose)
        projected = cv2.projectPoints(xyz[ids], cv2.Rodrigues(rotation)[0],
            np.asarray(rec.pose[4:]), matrices[rec.camera_id], distortions[rec.camera_id])[0].reshape(-1, 2)
        value = np.linalg.norm(projected-pixels, axis=1)
        value[(np.einsum('ni,i->n', xyz[ids], rotation[2])+np.asarray(rec.pose[4:])[2]) <= 0] = np.inf
        errors.append(value)
    maximum = np.maximum(*errors)
    return dict(tested=len(maximum), passed=int((maximum <= 3.).sum()),
                median_px=float(np.median(maximum)), p95_px=float(np.percentile(maximum, 95)))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, default=DATA_DIR/'output_map_preprocessing/20261002-153042-c033a9/stage_48')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=48)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--device', default='mps', choices=('cpu', 'mps', 'cuda'))
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 1 or args.repeat < 1:
        parser.error('Use a fresh output path and positive counts')
    args.output.mkdir(parents=True)
    cv2.setNumThreads(4)
    summary = json.loads((args.map/'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    records = {name: rec for name, rec in records.items() if (args.map/'reference_features'/(Path(name).stem+'.npz')).is_file()}
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
    matrices, distortions = {}, {}
    for cid, camera in cameras.items():
        matrices[cid], distortions[cid] = camera_matrix_and_distortion(np, camera, params[cid])
    with np.load(args.map/'object_aliked_map.npz') as archive:
        xyz = archive['xyz'].copy()
    all_pairs = enrichment_pairs(records, np.mean(np.stack([xyz.min(0), xyz.max(0)]), axis=0), set(records), 12)
    selected = [all_pairs[i] for i in np.unique(np.linspace(0, len(all_pairs)-1, min(args.pairs, len(all_pairs))).astype(int))]
    torch, extractor, matcher, device = _load_models(2000, .01, args.device)
    del extractor
    torch.set_num_threads(4)
    def sync():
        if args.device == 'mps':
            torch.mps.synchronize()
        elif args.device == 'cuda':
            torch.cuda.synchronize()
    @lru_cache(maxsize=12)
    def features(name):
        with np.load(args.map/'reference_features'/(Path(name).stem+'.npz')) as archive:
            arrays = {k: archive[k].copy() for k in
                      ['keypoints', 'descriptors', 'keypoint_scores', 'image_size', 'full_uv', 'point_ids']}
        record = records[name]
        prepared = prepare_features(arrays['full_uv'], arrays['descriptors'][0], record,
                                    matrices[record.camera_id], distortions[record.camera_id])
        return arrays, prepared
    def tensor_features(arrays):
        return {k: torch.from_numpy(arrays[k]).to(device) for k in
                ['keypoints', 'descriptors', 'keypoint_scores', 'image_size']}
    rows = []
    for number, (first, second) in enumerate(selected):
        start = time.perf_counter()
        a, pa = features(first); b, pb = features(second)
        load_ms = (time.perf_counter()-start)*1000.
        for method in ['lightglue', 'known-pose']:
            elapsed = []
            # Model warm-up once; include per-pair CPU-to-device transfer in LG.
            if number == 0 and method == 'lightglue':
                _match_pair(torch, matcher, tensor_features(a), tensor_features(b)); sync()
            for repeat in range(args.repeat):
                sync(); start = time.perf_counter()
                detail = {}
                if method == 'lightglue':
                    _, _, pairs, scores = _match_pair(torch, matcher, tensor_features(a), tensor_features(b))
                else:
                    pairs, scores, detail = match_features(pa, pb)
                sync(); elapsed.append((time.perf_counter()-start)*1000.)
            keep = epipolar_inliers(a['full_uv'][pairs[:, 0]], b['full_uv'][pairs[:, 1]],
                                   records[first], records[second], matrices, distortions, 2.5)
            raw = len(pairs); pairs = pairs[keep]
            left_ids, right_ids = a['point_ids'][pairs[:, 0]], b['point_ids'][pairs[:, 1]]
            verified = (left_ids >= 0) & (right_ids >= 0)
            correct = verified & (left_ids == right_ids)
            shared = np.intersect1d(a['point_ids'][a['point_ids'] >= 0], b['point_ids'][b['point_ids'] >= 0])
            row = dict(first=first, second=second, method=method, load_prepare_ms=load_ms,
                       matching_ms=elapsed, raw_matches=raw, epipolar_matches=len(pairs),
                       known_id_pairs=int(verified.sum()), same_id_pairs=int(correct.sum()),
                       conflicting_id_pairs=int((verified & ~correct).sum()),
                       shared_ids=len(shared), shared_id_recall=float(correct.sum()/max(1, len(shared))),
                       geometry=geometric_quality(pairs, a, b, first, second, xyz, records, matrices, distortions),
                       details=detail)
            rows.append(row)
            np.savez_compressed(args.output/f'pair_{number:03d}_{method}.npz', matches=pairs, scores=scores[keep])
            with (args.output/'pairs.jsonl').open('a') as target:
                target.write(json.dumps(row)+'\n')
        print(json.dumps(dict(pair=number+1, first=first, second=second,
                              lightglue_ms=rows[-2]['matching_ms'], known_pose_ms=rows[-1]['matching_ms'])), flush=True)
    totals = {}
    for method in ['lightglue', 'known-pose']:
        group = [r for r in rows if r['method'] == method]
        elapsed = [v for row in group for v in row['matching_ms']]
        known = sum(r['known_id_pairs'] for r in group)
        same = sum(r['same_id_pairs'] for r in group)
        totals[method] = dict(matching_ms_mean=float(np.mean(elapsed)), matching_ms_median=float(np.median(elapsed)),
            matching_ms_p95=float(np.percentile(elapsed, 95)), matches=sum(r['epipolar_matches'] for r in group),
            known_id_pairs=known, same_id_pairs=same, known_id_agreement=same/max(1, known),
            mean_shared_id_recall=float(np.mean([r['shared_id_recall'] for r in group])))
    known = [r for r in rows if r['method'] == 'known-pose']
    totals['candidate_fraction'] = sum(r['details']['candidate_pairs'] for r in known)/sum(r['details']['total_pairs'] for r in known)
    result = dict(map=str(args.map), photos=len(records), all_pair_count=len(all_pairs), sampled_pairs=len(selected),
                  repeats=args.repeat, device=args.device, summary=totals,
                  map_sha256=hashlib.sha256((args.map/'object_aliked_map.npz').read_bytes()).hexdigest(),
                  code_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                      ['known_pose_matching.py', 'enrich_registered_aliked_map.py', 'build_aliked_object_map.py']},
                  caveat='Same saved features; IDs derive from the audited LG map, not independent physical truth; cold ALIKED extraction excluded')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
