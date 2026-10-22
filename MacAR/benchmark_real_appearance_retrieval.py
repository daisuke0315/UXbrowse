"""Controlled GS-angle / real-direct / GS-constrained real-appearance experiment.

One fresh process per photo. Does not modify app settings, maps or neural models.
Common ROI, DINO tokens and ALIKED query, 6-photo cap, exhaustive, no retries.
Reported pipeline cost is component sum, NOT live end-to-end latency.
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
MODES = ('gs-angle', 'real-direct', 'gs-real')
os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def synchronize(engine):
    if engine.device == 'mps':
        engine.torch.mps.synchronize()


def worker(output, index):
    from aliked_lightglue_pnp import _extract, _image_tensor, _match_pair
    from browser_photo import decode_photo
    from glb_virtual_live import crop_live_object
    from offline_pnp_test import (load_reference_images, load_camera_parameters,
                                 camera_matrix_and_distortion, quaternion_to_rotation)
    from patch_view_rerank import square_mask
    from real_appearance_retrieval import rank_real_references, select_appearance_references
    from realtime_pnp import approximate_camera_matrix
    from registered_aliked_live import RegisteredALIKEDLiveLocalizer, candidate_references, rotation_angle
    from view_retrieval import crop_box

    cv2.setNumThreads(4)
    inputs = json.loads((output/'inputs.json').read_text())
    q = inputs['queries'][index]
    path = Path(q['path'])
    engine = RegisteredALIKEDLiveLocalizer(Path(inputs['map_path']), Path(inputs['views_path']),
        device=inputs['device'], max_keypoints=2000, top_k=3, resize=1024,
        fundamental_threshold=3., ransac_threshold=4., max_reprojection_error=3.,
        search_policy='exhaustive')
    engine.query_resize = 1600
    all_registered = dict(engine.registered)
    with np.load(inputs['real_index'], allow_pickle=False) as a:
        names = a['names'].tolist()
        embeddings, patches, masks = [a[k].copy() for k in ('embeddings', 'patches', 'masks')]
        if (names != sorted(all_registered) or str(a['map_path']) != inputs['map_path'] or
                embeddings.shape[0] != len(names) or patches.shape[:3] != (len(names), 16, 16)
                or masks.shape != (len(names), 16, 16)):
            raise ValueError('Real index content mismatch')
    summary = json.loads((Path(inputs['map_path'])/'summary.json').read_text())
    records = load_reference_images(Path(summary['input_files']['images']))
    image_dir = Path(summary['input_files']['images']).parents[2]/'images'
    # Confirm old-index RGB/crop descriptors with the current encoder, not only filenames.
    index_checks = []
    for check in (0, len(names)//2, len(names)-1):
        with np.load(all_registered[names[check]]['path'], allow_pickle=False) as a:
            x0, y0, x1, y1 = a['crop_box']
            mask = a['object_mask'].copy()
        ref = cv2.imread(str(image_dir/names[check]))
        if ref is None:
            raise ValueError('Missing real index source RGB')
        vector, grid = engine.retriever.encoder.encode_tokens(ref[y0:y1, x0:x1])
        delta = float(max(np.max(np.abs(vector-embeddings[check])), np.max(np.abs(grid-patches[check]))))
        if delta > 1e-4 or not np.array_equal(square_mask(mask), masks[check]):
            raise ValueError('Real index encoder/crop/mask mismatch')
        index_checks.append(dict(name=names[check], max_descriptor_delta=delta))

    image = decode_photo(path.read_bytes())
    source = q['kind'] == 'source'
    record = records[path.name] if source else None
    if source:
        frame = int(path.stem)
        engine.registered = {name: r for name, r in all_registered.items()
                             if abs(int(Path(name).stem)-frame) > 5}
        cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
        K, D = camera_matrix_and_distortion(np, cameras[record.camera_id], params[record.camera_id])
        true_R = quaternion_to_rotation(np, record.pose)
        true_C = -true_R.T@np.asarray(record.pose[4:])
    else:
        K = approximate_camera_matrix(image.shape[1], image.shape[0], 72.)
        D = np.zeros(5)

    # Warm ALIKED/LightGlue on a tiny image; exclude from measured preparation.
    with engine.torch.inference_mode():
        warm = _extract(engine.extractor, _image_tensor(engine.torch, np.full((224,224,3),127,np.uint8), engine.device), 224)
    _match_pair(engine.torch, engine.matcher, warm, engine._reference(names[0])[0])
    synchronize(engine)
    engine.reference_cache.clear()

    started = time.perf_counter()
    bbox, detection = engine.retriever.detector.detect(image)
    synchronize(engine)
    detect_ms = (time.perf_counter()-started)*1000
    if bbox is None:
        dump(output/f'query_{index:02d}.json', dict(name=path.name,kind=q['kind'],roi_missing=True,rows=[]))
        return
    dino_crop = crop_box(image, bbox)
    started = time.perf_counter()
    query_vector, query_patches = engine.retriever.encoder.encode_tokens(dino_crop)
    synchronize(engine)
    dino_ms = (time.perf_counter()-started)*1000
    mask = cv2.resize(detection['mask'], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    query_mask = square_mask(crop_box(mask, bbox))
    # Original production ranking, with exact shared ROI/tokens to avoid repeating inference.
    retriever = copy.copy(engine.retriever)
    retriever.detector = SimpleNamespace(detect=lambda _: (bbox, detection))
    retriever.encoder = SimpleNamespace(encode_tokens=lambda _: (query_vector, query_patches))
    started = time.perf_counter()
    retrieval = retriever._rank(image, 3, (), False, False)
    gs_selection_ms = (time.perf_counter()-started)*1000
    coarse = retrieval[0]
    started = time.perf_counter()
    crop, _ = crop_live_object(image, bbox)
    with engine.torch.inference_mode():
        query = _extract(engine.extractor, _image_tensor(engine.torch, crop, engine.device), 1600)
    synchronize(engine)
    features_ms = (time.perf_counter()-started)*1000
    common_ms = detect_ms+dino_ms+features_ms

    eligible = set(engine.registered)
    timings, rankings = {}, {}
    started = time.perf_counter()
    rankings[None] = rank_real_references(names, embeddings, patches, masks,
        query_vector, query_patches, query_mask, eligible)
    direct_refs = select_appearance_references(engine.views, engine.registered, coarse,
        rankings, mode='real-direct')
    timings['real-direct'] = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    for _, _, virtual_index in coarse:
        R = np.asarray(engine.views[virtual_index]['world_to_camera'])[:3,:3]
        compatible = {name for name in eligible if rotation_angle(engine.registered[name]['rotation'], R) <= 55.}
        rankings[virtual_index] = rank_real_references(names, embeddings, patches, masks,
            query_vector, query_patches, query_mask, compatible)
    hybrid_refs = select_appearance_references(engine.views, engine.registered, coarse,
        rankings, mode='gs-real')
    timings['gs-real'] = gs_selection_ms+(time.perf_counter()-started)*1000
    started = time.perf_counter()
    baseline_refs = candidate_references(engine.views, engine.registered, engine.manifest['center'], coarse, 3, 55.)
    timings['gs-angle'] = gs_selection_ms+(time.perf_counter()-started)*1000
    references = dict(zip(MODES, (baseline_refs,direct_refs,hybrid_refs)))

    def pose_truth(result):
        if result.rvec is None:
            return {}
        R = cv2.Rodrigues(np.asarray(result.rvec))[0]
        C = -R.T@np.asarray(result.tvec).reshape(3)
        if source:
            angle = rotation_angle(R, true_R)
            distance = float(np.linalg.norm(C-true_C))
            return dict(rotation_error_deg=angle, center_error_units=distance,
                center_error_object_fraction=distance/engine.scene_scale,
                pose_correct=bool(angle <= 5. and distance <= .05*engine.scene_scale))
        direction = C-np.asarray(engine.manifest['center'])
        azimuth = float(np.degrees(np.arctan2(direction[0],direction[1]))%360)
        n = int(path.stem.split('_')[1])
        start = 90 if n<=2591 else 0 if n<=2594 else 270 if n<=2597 else 180
        return dict(estimated_azimuth_deg=azimuth, face_correct=bool(0 <= (azimuth-start)%360 <= 90))

    def retrieval_truth(ranked_names):
        if not source or not ranked_names:
            return {}
        centre = np.asarray(engine.manifest['center'])
        direction = true_C-centre
        direction /= np.linalg.norm(direction)
        errors = []
        for name, _ in ranked_names:
            rd = engine.registered[name]['center']-centre
            rd /= np.linalg.norm(rd)
            errors.append(float(np.degrees(np.arccos(np.clip(np.dot(direction,rd),-1.,1.)))))
        return {f'top{k}_direction_deg': min(errors[:k]) for k in (1,5,10)}

    rows = []
    for repeat in range(inputs['repeats']):
        shift = (index+repeat)%len(MODES)
        for mode in MODES[shift:]+MODES[:shift]:
            engine.reference_cache.clear()
            cv2.setRNGSeed(0)
            started = time.perf_counter()
            result, canvas = engine._localize_once(image, K, D,
                workspace=dict(retrieval=retrieval, query=query, references=references[mode]))
            synchronize(engine)
            search_ms = (time.perf_counter()-started)*1000
            diagnostics = copy.deepcopy(engine.last_diagnostics)
            selection_ms = timings[mode]
            row = dict(mode=mode,repeat=repeat,status=result.status,inliers=result.inliers,
                matches=result.matches, mean_error_px=result.mean_reprojection_error,
                candidate_selection_ms=selection_ms,search_ms=search_ms,
                pipeline_component_ms=common_ms+selection_ms+search_ms,
                reference_names=[r[0] for r in references[mode]],
                selected=diagnostics.get('selected'),diagnostics=diagnostics,**pose_truth(result))
            rows.append(row)
            print(json.dumps(dict(name=path.name,mode=mode,repeat=repeat,status=result.status,
                inliers=result.inliers,pipeline_ms=row['pipeline_component_ms'],**pose_truth(result))),flush=True)
            if repeat == 0:
                scale = min(1., 1100/canvas.shape[1])
                cv2.imwrite(str(output/f'{path.stem}_{mode}.jpg'), cv2.resize(canvas,
                    (round(canvas.shape[1]*scale),round(canvas.shape[0]*scale))))
    dump(output/f'query_{index:02d}.json',dict(name=path.name,kind=q['kind'],sha256=sha(path),
        roi=bbox,roi_missing=False,index_checks=index_checks,scene_scale=engine.scene_scale,
        shared_cost_ms=dict(detector=detect_ms,dino=dino_ms,aliked=features_ms),
        coarse_gs=[dict(name=engine.views[i]['image_name'],score=s) for s,_,i in coarse],
        real_ranking=rankings[None],real_retrieval_truth=retrieval_truth(rankings[None]),rows=rows))


def summarize(output):
    queries = [json.loads(p.read_text()) for p in sorted(output.glob('query_*.json'))]
    groups = {}
    for kind in ('new','source'):
        subset = [q for q in queries if q['kind']==kind]
        for mode in MODES:
            rows = [r for q in subset for r in q['rows'] if r['mode']==mode]
            passed = [r for r in rows if r['status']=='PASS']
            correct = [r for r in passed if r.get('pose_correct',r.get('face_correct',False))]
            def values(key, selected=rows):
                v = [r[key] for r in selected if r.get(key) is not None]
                return dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(max(v))) if v else None
            groups[f'{kind}_{mode}'] = dict(queries=len(subset),runs=len(rows),
                passed=len(passed),correct_passed=len(correct),
                ambiguous=sum('ambiguous_pose' in (r['diagnostics'].get('quality') or {}).get('reasons',[]) for r in rows),
                failed_names=sorted({q['name'] for q in subset for r in q['rows'] if r['mode']==mode and r['status']!='PASS'}),
                incorrect_names=sorted({q['name'] for q in subset for r in q['rows'] if r['mode']==mode and r['status']=='PASS' and not r.get('pose_correct',r.get('face_correct',False))}),
                pipeline_component_ms=values('pipeline_component_ms'),search_ms=values('search_ms'),
                candidate_selection_ms=values('candidate_selection_ms'),inliers=values('inliers',passed),
                rotation_error_deg=values('rotation_error_deg',passed),center_error_units=values('center_error_units',passed))
    dump(output/'summary.json',dict(groups=groups,queries=queries))
    print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    output=args.output.resolve()
    if args.worker is not None:
        worker(output,args.worker)
        return
    if args.repeats<1:
        parser.error('repeats must be positive')
    output.mkdir(parents=True,exist_ok=False)
    prior=DATA_DIR/'output_coarse_comparison_20261003'
    old=json.loads((prior/'metadata_shortlist.json').read_text())
    if sha(Path(old['map_path'])/'object_aliked_map.npz')!=old['map_sha256']:
        raise ValueError('Previous real index map identity changed; rebuild before comparing')
    # Each worker checks sampled RGB/crop/mask token equality with the current
    # encoder. Whole-file SHA changes (e.g. query-token reuse) are not proof of
    # changed feature representation. Current code SHAs are recorded below.
    from realtime_pnp import DEFAULT_SPLAT_VIEWS
    inputs=dict(map_path=old['map_path'],map_sha256=old['map_sha256'],
        views_path=str(DEFAULT_SPLAT_VIEWS),views_sha256=sha(DEFAULT_SPLAT_VIEWS/'manifest.json'),
        real_index=str(prior/'real_photo_dino_index.npz'),real_index_sha256=sha(prior/'real_photo_dino_index.npz'),
        queries=old['queries'],device=args.device,repeats=args.repeats,
        code_sha256={f:sha(ROOT/f) for f in ('benchmark_real_appearance_retrieval.py',
            'real_appearance_retrieval.py','registered_aliked_live.py','view_retrieval.py','spatial_pose_scoring.py')},
        conditions='Common ROI/DINO/ALIKED; up to 6 photos; exhaustive; no retry; same pose QC. Source excludes own and +/-5 frames. Source RC poses are not external truth. New-photo face sectors are coarse checks only. Pipeline times are component sums without model/index initialization.')
    dump(output/'inputs.json',inputs)
    for index in range(len(inputs['queries'])):
        with (output/f'query_{index:02d}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(output),
                '--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',index+1,'/',len(inputs['queries']),flush=True)
    summarize(output)


if __name__=='__main__':
    main()
