"""Paired alpha-reference ablation using the browser photo pipeline, no camera/CT.

New-photo QC and reprojection statistics are NOT independent pose accuracy.
Writes separate results and feature caches; never changes source RGB or XYZ.
"""

from macar_paths import DATA_DIR
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent


def validate_reference_geometry(conditions):
    baseline = None
    hashes = None
    for path in conditions.values():
        data = json.loads((path/'manifest.json').read_text())
        if data.get('ply'):
            ply = Path(data['ply'])
            data['ply'] = str((ply if ply.is_absolute() else path/ply).resolve())
        geometry = {key:data.get(key) for key in ('width','height','views','center','bounds',
            'up_axis','runtime_transform','source_type','ply','geometry_renderer')}
        current = [hashlib.sha256((path/v['data_name']).read_bytes()).hexdigest() for v in data['views']]
        if baseline is not None and (geometry != baseline or current != hashes):
            raise ValueError('RGB-only ablation requires identical cameras and geometry')
        baseline, hashes = geometry, current


def summarize(rows):
    accepted = [row['result'] for row in rows if row['result']['status'] == 'PASS']
    def median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None
    return dict(cases=len(rows), pass_count=len(accepted),
        ambiguous_count=sum('AMBIGUOUS' in row['result']['status'] or
                            'AMBIGUOUS' in row['result'].get('detail', '') for row in rows),
        inliers_median_all=median([r['result']['inliers'] for r in rows]),
        inliers_median_pass=median([r['inliers'] for r in accepted]),
        reprojection_median_pass_px=median([r['mean_reprojection_error'] for r in accepted]),
        processing_median_ms=median([r.get('wall_ms', r['result']['processing_ms']) for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo-root', type=Path, required=True)
    parser.add_argument('--extra-photo', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--original', type=Path, default=DATA_DIR/'output_3dgs_reference_views_72_upper_negz_v2')
    parser.add_argument('--alpha10', type=Path, default=DATA_DIR/'output_3dgs_reference_views_upper_rgb_20261003_alpha10')
    parser.add_argument('--alpha15', type=Path, default=DATA_DIR/'output_3dgs_reference_views_upper_rgb_20261003_alpha15')
    parser.add_argument('--conditions', nargs='+', default=['original', 'alpha10', 'alpha15'],
                        choices=['original', 'alpha10', 'alpha15'])
    parser.add_argument('--modes', nargs='+', default=['registered', 'gs'], choices=['registered', 'gs'])
    parser.add_argument('--source-limit', type=int, default=48,
                        help='Deterministic reconstruction-source subset for coarse direction audit')
    options = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))
    import cv2
    import numpy as np
    import torch
    from browser_live import runtime_arguments, finite_json
    from browser_photo import decode_photo, photo_engine, localize_photo
    from realtime_pnp import DEFAULT_REGISTERED_MAP, approximate_camera_matrix, build_localizer
    from offline_pnp_test import load_reference_images, quaternion_to_rotation
    cv2.setNumThreads(4)
    if options.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS unavailable: run with Mac GPU access, or explicitly choose --device cpu')
    options.output.mkdir(parents=True, exist_ok=False)
    conditions = dict(original=options.original.resolve(), alpha10=options.alpha10.resolve(),
                      alpha15=options.alpha15.resolve())
    validate_reference_geometry({key:conditions[key] for key in options.conditions})
    map_path = (options.map or DEFAULT_REGISTERED_MAP).resolve()
    photos = sorted(set(options.photo_root.glob('*/*.jpeg')) | set(options.extra_photo), key=lambda p:p.name)
    if not photos:
        parser.error('No query photos found')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in photos}
    settings = dict(input_source='photo', photo_feature_size=1600, top_k=3,
                    fov_deg=72, quality_profile='precision', mpr=False,
                    map_path=str(map_path), coarse_method='dino-patch')
    (options.output/'configuration.json').write_text(json.dumps(dict(settings=settings,
        photo_hashes=hashes, device=options.device, reference_paths={k:str(v) for k,v in conditions.items()},
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['glb_virtual_live.py','registered_aliked_live.py','view_retrieval.py','browser_photo.py']},
        truth='No independent new-photo pose truth; RC source subset is in-distribution only'), indent=2))
    rows, summaries = [], {}
    def emit(row):
        clean = finite_json(row)
        with (options.output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(clean, ensure_ascii=False, allow_nan=False)+'\n')
        return clean
    for mode in options.modes:
        for condition in options.conditions:
            init_started = time.perf_counter()
            args, manifest, _ = runtime_arguments(dict(settings, mode=mode, views_path=str(conditions[condition])))
            args.glb_device = options.device
            wrapper = build_localizer(args, temporal_registered=True)
            engine = photo_engine(wrapper, 1600)
            init_ms = (time.perf_counter()-init_started)*1000
            prep_started = time.perf_counter()
            # Precompute ALL virtual reference features so cold disk/model work
            # does not selectively penalize alpha15 during timed photo matching.
            if mode == 'gs':
                for index in range(len(engine.views)):
                    engine._reference_features(index)
            print(f'EVAL ready: {mode}/{condition}, map={map_path}', flush=True)
            warm = decode_photo(photos[0].read_bytes())
            localize_photo(engine, warm, approximate_camera_matrix(warm.shape[1], warm.shape[0], 72), np.zeros(5))
            prep_ms = (time.perf_counter()-prep_started)*1000
            group = []
            for number, path in enumerate(photos, 1):
                image = decode_photo(path.read_bytes())
                matrix = approximate_camera_matrix(image.shape[1], image.shape[0], 72)
                cv2.setRNGSeed(0); torch.manual_seed(0)
                # Synchronous single-image path: no optical flow / prior pose.
                started = time.perf_counter()
                result, display = localize_photo(engine, image, matrix, np.zeros(5))
                wall_ms = (time.perf_counter()-started)*1000
                diagnostic = engine.last_diagnostics
                row = emit(dict(kind='new_photo', mode=mode, condition=condition, name=path.name,
                    result=result.as_dict(), wall_ms=wall_ms, stages=diagnostic, retrieval=engine.last_retrieval,
                    roi=engine.last_crop_debug.get('bbox'), device=str(engine.device),
                    expansion=getattr(engine,'photo_view_expansion',None)))
                rows.append(row); group.append(row)
                preview = cv2.resize(display, (round(display.shape[1]*min(1,1280/display.shape[1])),
                                               round(display.shape[0]*min(1,1280/display.shape[1]))))
                cv2.imwrite(str(options.output/f'{mode}_{condition}_{path.stem}.jpg'), preview)
                print(f'EVAL {mode}/{condition} {number}/{len(photos)} {path.name}: '
                      f'{result.status}, {result.inliers}/{result.matches}, '
                      f'{result.mean_reprojection_error}px, {result.processing_ms:.0f}ms', flush=True)
            summaries[f'{mode}/{condition}'] = dict(summarize(group), init_ms=init_ms,
                                                     reference_prepare_and_warmup_ms=prep_ms)
            # Coarse retrieval has a pose truth only for reconstruction-source
            # photographs. Audit each appearance once, NOT as held-out accuracy.
            if mode == options.modes[0] and options.source_limit > 0:
                source = json.loads((map_path/'summary.json').read_text())['input_files']
                pose_path = Path(source['images'])
                records = sorted(load_reference_images(pose_path).items())
                image_dir = pose_path.parents[2]/'images'
                indices = np.linspace(0, len(records)-1, min(options.source_limit,len(records))).astype(int)
                center = np.asarray(manifest['center'])
                directions = np.asarray([v['eye'] for v in engine.views])-center
                directions /= np.linalg.norm(directions,axis=1,keepdims=True)
                coarse_rows=[]
                for index in indices:
                    name, record = records[index]
                    image = decode_photo((image_dir/name).read_bytes())
                    ranked, _, all_ranked, debug = engine.retriever.rank_debug(image, top_k=10)
                    rotation=quaternion_to_rotation(np,record.pose)
                    eye=-rotation.T@np.asarray(record.pose[4:])
                    direction=eye-center;direction/=np.linalg.norm(direction)
                    errors=np.degrees(np.arccos(np.clip(directions@direction,-1,1)))
                    selected=[i for _,_,i in ranked]
                    row=emit(dict(kind='source_coarse',condition=condition,name=name,
                        top1_direction_deg=float(errors[selected[0]]) if selected else None,
                        top5_near=any(errors[i]<=30 for i in selected[:5]),
                        top10_near=any(errors[i]<=30 for i in selected[:10]),
                        roi=debug.get('bbox')))
                    coarse_rows.append(row)
                summaries[f'coarse/{condition}']=dict(cases=len(coarse_rows),
                    top1_median_direction_deg=statistics.median([r['top1_direction_deg']
                        for r in coarse_rows if r['top1_direction_deg'] is not None]),
                    top5_near_rate=sum(r['top5_near'] for r in coarse_rows)/len(coarse_rows),
                    top10_near_rate=sum(r['top10_near'] for r in coarse_rows)/len(coarse_rows),
                    truth='Reconstruction-source fixed RC pose, NOT held-out')
            (options.output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
            wrapper.close()
            del wrapper, engine
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    main()
