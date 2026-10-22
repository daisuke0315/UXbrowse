"""Standalone coarse-selection ablation; never changes app defaults or maps.

Compare current 128 GS render retrieval with a 620 registered-photo DINO index,
using the same detector, encoder, ALIKED, map, PnP and QC thresholds. Source
queries exclude their own photo and +/-5 numbered frames from both routes.
These source queries still contributed to reconstruction, so are not held out.
An additional exhaustive ablation disables the coarse-rotation gate in ALL
compared arms, and scans every eligible real reference without retrieval.
"""

from macar_paths import DATA_DIR
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import types

import cv2
import numpy as np

from aliked_lightglue_pnp import _extract, _image_tensor
from browser_photo import decode_photo, localize_photo
from glb_virtual_live import crop_live_object
from offline_pnp_test import (load_reference_images, load_camera_parameters,
                             camera_matrix_and_distortion, quaternion_to_rotation)
from patch_view_rerank import square_mask
from realtime_pnp import DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS, approximate_camera_matrix
from registered_aliked_live import RegisteredALIKEDLiveLocalizer, candidate_references, rotation_angle
from view_retrieval import crop_box

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--photo-root', type=Path, required=True)
    p.add_argument('--source-count', type=int, default=12)
    p.add_argument('--phase', choices=['shortlist', 'expanded', 'exhaustive', 'recheck'], default='shortlist')
    p.add_argument('--exhaustive-names', nargs='+', default=['IMG_2593.jpeg', 'IMG_2598.jpeg'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))
    e = RegisteredALIKEDLiveLocalizer(DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS,
        device='mps', max_keypoints=2000, top_k=3, resize=1024,
        fundamental_threshold=3., ransac_threshold=4., max_reprojection_error=3.,
        retrieval_mode='dino-patch', search_policy='adaptive')
    e.query_resize = 1600
    # This historical ablation compares angle-nearest photos, not the current
    # production GS-constrained appearance shortlist.
    e._appearance_references = lambda image, ranked, workspace: candidate_references(
        e.views, e.registered, e.manifest['center'], ranked, e.top_k, e.max_coarse_rotation_deg)
    gs_views, registered = e.views, dict(e.registered)
    gs = copy.copy(e.retriever)
    # No reference-preview image IO in either comparison arm.
    def rank_debug(self, image, top_k=None):
        return self._rank(image, top_k, getattr(self, 'comparison_excluded', ()), False, False)
    gs.rank_debug = types.MethodType(rank_debug, gs)
    real = copy.copy(gs)
    real_names = sorted(registered)
    real_views = []
    for name in real_names:
        ref = registered[name]
        w2c = np.eye(4)
        w2c[:3, :3] = ref['rotation']
        w2c[:3, 3] = -ref['rotation']@ref['center']
        real_views.append(dict(image_name=name, world_to_camera=w2c.tolist()))
    real.views = real_views
    real.rank_debug = types.MethodType(rank_debug, real)
    summary = json.loads((DEFAULT_REGISTERED_MAP/'summary.json').read_text())
    pose_path = Path(summary['input_files']['images'])
    image_dir = pose_path.parents[2]/'images'
    records = load_reference_images(pose_path)
    cameras, params = load_camera_parameters(Path(summary['input_files']['cameras']))
    index_cache = args.output/'real_photo_dino_index.npz'
    index_started = time.perf_counter()
    if index_cache.exists():
        with np.load(index_cache, allow_pickle=False) as a:
            if a['names'].tolist() != real_names or str(a['map_path']) != str(DEFAULT_REGISTERED_MAP):
                raise ValueError('Index identity mismatch')
            real.reference = a['embeddings'].copy()
            real.reference_patches = a['patches'].copy()
            real.reference_masks = a['masks'].copy()
        index_event = 'loaded'
    else:
        values, patches, masks = [], [], []
        for number, name in enumerate(real_names, 1):
            image = cv2.imread(str(image_dir/name))
            if image is None:
                raise RuntimeError('Missing real reference '+name)
            with np.load(registered[name]['path'], allow_pickle=False) as a:
                x0, y0, x1, y1 = a['crop_box']
                mask = a['object_mask'].copy()
            crop = image[y0:y1, x0:x1]
            if crop.shape[:2] != mask.shape:
                raise ValueError('Real reference crop/mask mismatch '+name)
            vector, grid = real.encoder.encode_tokens(crop)
            values.append(vector); patches.append(grid); masks.append(square_mask(mask))
            if number%25 == 0:
                print(json.dumps(dict(event='index_progress', count=number, total=len(real_names))), flush=True)
        real.reference = np.stack(values)
        real.reference_patches = np.stack(patches)
        real.reference_masks = np.stack(masks)
        np.savez_compressed(index_cache, names=np.asarray(real_names), map_path=str(DEFAULT_REGISTERED_MAP),
                            embeddings=real.reference, patches=real.reference_patches, masks=real.reference_masks)
        index_event = 'built'
    index_ms = (time.perf_counter()-index_started)*1000
    print(json.dumps(dict(event='real_index', action=index_event, elapsed_ms=index_ms)), flush=True)
    center = np.asarray(e.manifest['center'])
    photo_paths = sorted(args.photo_root.glob('*/*.jpeg'), key=lambda x:x.name)
    if len(photo_paths) != 12:
        raise ValueError('Expected the same 12 separate photos')
    queries = [(path, None) for path in photo_paths]
    # Before matching, select the source frame closest to each regular azimuth.
    # No retrieval or PnP result enters this query selection.
    source_candidates = []
    for name in real_names:
        ref = registered[name]
        direction = ref['center']-center
        az = float(np.degrees(np.arctan2(direction[0], direction[1]))%360)
        elevation = float(np.degrees(np.arcsin(-direction[2]/np.linalg.norm(direction))))
        source_candidates.append((name, az, elevation))
    picked = set()
    for target in np.linspace(0, 360, args.source_count, endpoint=False):
        chosen = min((v for v in source_candidates if v[0] not in picked),
                     key=lambda v: abs((v[1]-target+180)%360-180)+.1*abs(v[2]-25))
        picked.add(chosen[0])
        queries.append((image_dir/chosen[0], records[chosen[0]]))
    metadata = dict(map_path=str(DEFAULT_REGISTERED_MAP), map_sha256=sha(DEFAULT_REGISTERED_MAP/'object_aliked_map.npz'),
        gs_manifest_sha256=sha(DEFAULT_SPLAT_VIEWS/'manifest.json'), gs_views=len(gs_views), real_views=len(real_views),
        code_sha256={f:sha(ROOT/f) for f in ['compare_coarse_localization.py','registered_aliked_live.py','view_retrieval.py']},
        device=str(e.device), query_resize=1600, max_keypoints=2000, top_k=3,
        fundamental_threshold=3, ransac_threshold=4, max_reprojection_error=3,
        min_inliers=e.min_inliers, min_inlier_ratio=e.min_inlier_ratio,
        source_exclusion='same frame and +/-5 numeric frame IDs',
        known_pose_correct_threshold='rotation <=5deg AND center delta <=5% robust object diagonal',
        source_truth='RC estimated poses; images contributed to reconstruction, not independent holdout',
        new_truth='physical face sector, external numeric pose unknown',
        shared_3dgs='object detector and prebuilt XYZ map/masks remain shared; coarse retrieval representation only is varied',
        real_index_build_or_load_ms=index_ms, real_index_action=index_event,
        queries=[dict(path=str(path), kind='source' if record else 'new') for path,record in queries])
    (args.output/('metadata_'+args.phase+'.json')).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    rows_path = args.output/(args.phase+'.jsonl')
    if rows_path.exists():
        raise ValueError('Refuse to overwrite completed or partial runs')

    def emit(row):
        with rows_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps({k:row[k] for k in ['name','kind','mode','repeat','status','wall_ms','processed','rotation_error_deg','face_correct'] if k in row}), flush=True)

    def truth(result, record, name):
        if result.rvec is None:
            return {}
        rotation = cv2.Rodrigues(np.asarray(result.rvec))[0]
        camera_center = -rotation.T@np.asarray(result.tvec)
        if record:
            true_r = quaternion_to_rotation(np, record.pose)
            true_center = -true_r.T@np.asarray(record.pose[4:])
            error = rotation_angle(rotation, true_r)
            delta = float(np.linalg.norm(camera_center-true_center))
            return dict(rotation_error_deg=error, center_error_units=delta,
                        center_error_object_fraction=delta/e.scene_scale,
                        pose_correct=error<=5 and delta<=.05*e.scene_scale)
        az = float(np.degrees(np.arctan2(*(camera_center-center)[:2]))%360)
        frame = int(Path(name).stem.split('_')[1])
        start = 90 if frame<=2591 else 0 if frame<=2594 else 270 if frame<=2597 else 180
        return dict(estimated_azimuth_deg=az, expected_face_sector=[start, start+90],
                    face_correct=0<=((az-start)%360)<=90)

    # Warm the shared neural models without contributing a reported sample.
    warm_image = decode_photo(photo_paths[4].read_bytes())
    warm_rank = gs.rank_debug(warm_image, top_k=3)
    crop = crop_live_object(warm_image, warm_rank[3]['bbox'])[0]
    with e.torch.inference_mode():
        warm_query = _extract(e.extractor, _image_tensor(e.torch, crop, e.device), 1600)
    from aliked_lightglue_pnp import _match_pair
    _match_pair(e.torch,e.matcher,warm_query,e._reference(real_names[0])[0])
    del warm_query
    e.reference_cache.clear()
    e.torch.mps.empty_cache()

    def setup(path, record):
        excluded = set()
        if record:
            frame = int(path.stem)
            excluded = {name for name in registered if abs(int(Path(name).stem)-frame)<=5}
        e.registered = {n:r for n,r in registered.items() if n not in excluded}
        real.comparison_excluded = [i for i,n in enumerate(real_names) if n in excluded]
        image = decode_photo(path.read_bytes())
        if record:
            K, dist = camera_matrix_and_distortion(np, cameras[record.camera_id], params[record.camera_id])
        else:
            K = approximate_camera_matrix(image.shape[1],image.shape[0],72.)
            dist = np.zeros(5)
        return image, K, dist, excluded

    def run(path, record, mode, repeat, image, K, dist, excluded, *, controlled=False, workspace=None):
        use_real = mode!='gs'
        e.views, e.retriever = (real_views, real) if use_real else (gs_views, gs)
        e.search_policy = 'exhaustive' if controlled else 'adaptive'
        e.max_coarse_rotation_deg = 180. if controlled else 55.
        e.reference_cache.clear()
        cv2.setRNGSeed(0)
        workspace = {} if workspace is None else workspace
        started = time.perf_counter()
        if args.phase=='expanded':
            result, canvas = localize_photo(e,image,K,dist,max_views=10)
        else:
            result, canvas = e._search_attempts(image, K, dist, None, workspace)
        e.torch.mps.synchronize()
        wall = (time.perf_counter()-started)*1000
        diag = copy.deepcopy(e.last_diagnostics)
        search = diag.get('reference_search', {})
        row = dict(name=path.name,kind='source' if record else 'new',mode=mode,repeat=repeat,
                   controlled=controlled,wall_ms=wall,result=result.as_dict(),status=result.status,
                   processed=search.get('processed',len(diag.get('candidates',[]))),
                   total=search.get('total'), diagnostics=diag, excluded=sorted(excluded),
                   **truth(result,record,path.name))
        emit(row)
        if repeat==0:
            preview=cv2.resize(canvas,(int(canvas.shape[1]*min(1,1100/canvas.shape[1])),int(canvas.shape[0]*min(1,1100/canvas.shape[1]))))
            cv2.imwrite(str(args.output/(path.stem+'_'+mode+('_controlled' if controlled else '')+'.jpg')),preview)
        e.reference_cache.clear()
        e.torch.mps.empty_cache()
        return workspace

    if args.phase=='shortlist':
        for index,(path,record) in enumerate(queries):
            image,K,dist,excluded=setup(path,record)
            modes=['gs','real'] if index%2==0 else ['real','gs']
            for mode in modes:
                run(path,record,mode,0,image,K,dist,excluded)
        # Repeated same-photo comparisons, alternating first arm across repeats.
        for repeat in [1,2]:
            for path in photo_paths:
                if path.name not in args.exhaustive_names:
                    continue
                image,K,dist,excluded=setup(path,None)
                for mode in (['real','gs'] if repeat%2 else ['gs','real']):
                    run(path,None,mode,repeat,image,K,dist,excluded)
    elif args.phase=='expanded':
        initial=[json.loads(l) for l in (args.output/'shortlist.jsonl').read_text().splitlines()]
        failed={r['name'] for r in initial if r['repeat']==0 and r['status']!='PASS'}
        for index,(path,record) in enumerate(queries):
            if path.name not in failed:
                continue
            image,K,dist,excluded=setup(path,record)
            for mode in (['gs','real'] if index%2==0 else ['real','gs']):
                run(path,record,mode,0,image,K,dist,excluded)
    else:
        selected=[q for q in queries if q[0].name in args.exhaustive_names]
        if len(selected)!=len(args.exhaustive_names):
            raise ValueError('Unknown exhaustive query name')
        for path,record in selected:
            candidate_counter = [0]
            def diagnostic_sink(event, **fields):
                if event == 'reference_candidate':
                    candidate_counter[0] += 1
                    if candidate_counter[0] % 25 == 0:
                        print(json.dumps(dict(event='scan_progress',name=path.name,
                                              processed=candidate_counter[0])), flush=True)
            e.diagnostic_sink = diagnostic_sink
            image,K,dist,excluded=setup(path,record)
            # Controlled budget comparison: common numeric QC, no directional
            # prior rejection or early exit. All arms use identical ALIKED
            # features and ROI, avoiding feature-extraction timing confounds.
            e.views,e.retriever=gs_views,gs
            retrieval_started=time.perf_counter()
            retrieval=gs.rank_debug(image,top_k=3)
            gs_retrieval_ms=(time.perf_counter()-retrieval_started)*1000
            detector_started=time.perf_counter()
            bbox,_=gs.detector.detect(image)
            detector_ms=(time.perf_counter()-detector_started)*1000
            if not np.allclose(bbox,retrieval[3]['bbox']):
                raise ValueError('ROI changed in detector-only ablation')
            cropped=crop_live_object(image,retrieval[3]['bbox'])[0]
            feature_started=time.perf_counter()
            with e.torch.inference_mode():
                query=_extract(e.extractor,_image_tensor(e.torch,cropped,e.device),1600)
            e.torch.mps.synchronize()
            extraction_ms=(time.perf_counter()-feature_started)*1000
            # Detect again for real retrieval, but use the common saved ROI in
            # ALIKED/PnP. Deterministic detector equality is required.
            retrieval_started=time.perf_counter()
            real_retrieval=real.rank_debug(image,top_k=3)
            real_retrieval_ms=(time.perf_counter()-retrieval_started)*1000
            if not np.allclose(retrieval[3]['bbox'],real_retrieval[3]['bbox']):
                raise ValueError('ROI changed between comparison arms')
            for mode,ret,views in [('gs',retrieval,gs_views),('real',real_retrieval,real_views)]:
                refs=candidate_references(views,e.registered,e.manifest['center'],ret[0],3,55.)
                run(path,record,mode,0,image,K,dist,excluded,controlled=True,
                    workspace=dict(retrieval=ret,query=query,references=refs))
            # No image descriptor retrieval or shortlist for this arm. The
            # registered cameras are only labels; max_rotation=180 disables
            # the view-angle rejection in all controlled arms.
            refs=[(name,i,0.,0.) for i,name in enumerate(real_names) if name not in excluded]
            if args.phase!='recheck':
                run(path,record,'all',0,image,K,dist,excluded,controlled=True,
                    workspace=dict(retrieval=real_retrieval,query=query,references=refs))
            suffix='_recheck' if args.phase=='recheck' else ''
            (args.output/(path.stem+suffix+'_common_costs.json')).write_text(json.dumps(dict(
                gs_retrieval_ms=gs_retrieval_ms,real_retrieval_ms=real_retrieval_ms,
                shared_features_ms=extraction_ms,all_retrieval_ms=detector_ms,
                comment='Add the shared extraction and each retrieval/detection cost to the query-reused search time.'),indent=2))
            del query
            e.torch.mps.empty_cache()


if __name__=='__main__':
    main()
