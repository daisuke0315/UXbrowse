"""Audit current coarse retrieval against fixed RC poses and separate new photos.

Reconstruction-source photos are NOT independent held-out data. New photos
have no numeric pose truth; contact sheets support a separate visual audit.
No ALIKED/PnP output is used as retrieval ground truth.
"""

from macar_paths import DATA_DIR
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from browser_photo import decode_photo
from glb_virtual_view_pnp import load_manifest
from offline_pnp_test import load_reference_images, quaternion_to_rotation
from view_retrieval import ViewRetrievalIndex, crop_box




def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.einsum('ij,j->i', a, b), -1., 1.)))


def tile(bgr, label, width=360, height=275):
    canvas = Image.new('RGB', (width, height), '#eeeeee')
    picture = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    picture.thumbnail((width, height-42))
    canvas.paste(picture, ((width-picture.width)//2, 36))
    ImageDraw.Draw(canvas).text((8, 8), label, fill='black')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--photo-root', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--source-limit', type=int, default=None,
                        help='Uniform deterministic subset for backend agreement checks')
    args = parser.parse_args()
    if args.source_limit is not None and args.source_limit < 1:
        parser.error('--source-limit must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('TORCH_HOME', str(DATA_DIR/'.model_cache/torch'))
    views_path = DATA_DIR/'output_3dgs_reference_views_72_upper_negz_v2'
    manifest, views = load_manifest(views_path)
    center = np.asarray(manifest['center'], np.float64)
    transforms = np.asarray([v['world_to_camera'] for v in views])
    rotations = transforms[:, :3, :3]
    eyes = -np.einsum('nji,nj->ni', rotations, transforms[:, :3, 3])
    directions = eyes-center
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    retriever = ViewRetrievalIndex(views_path, views, mode='dino-patch',
                                  crop=True, auto_roi=True, device=args.device)
    output_file = args.output/'queries.jsonl'

    def emit(row):
        with output_file.open('a', encoding='utf-8') as target:
            target.write(json.dumps(row, ensure_ascii=False)+'\n')

    def retrieve(path, kind, pose=None):
        image = decode_photo(path.read_bytes())
        started = time.perf_counter()
        selected, _, ranked, debug = retriever.rank_debug(image, top_k=3)
        row = dict(kind=kind, name=path.name, path=str(path),
                   retrieval_ms=(time.perf_counter()-started)*1000.,
                   bbox=debug.get('bbox'),
                   candidates=[dict(index=i, name=views[i]['image_name'],
                                    azimuth=views[i]['azimuth_deg'],
                                    elevation=views[i]['elevation_deg'], score=float(s))
                               for s, _, i in selected],
                   ranked_indices=[i for _, _, i in ranked],
                   global_top1=debug.get('patch_rerank', {}).get('global_top1_index'))
        if pose is not None:
            rotation = quaternion_to_rotation(np, pose)
            eye = -rotation.T@np.asarray(pose[4:], np.float64)
            direction = eye-center
            direction /= np.linalg.norm(direction)
            errors = angle(directions, direction)
            rotation_errors = np.degrees(np.arccos(np.clip(
                (np.einsum('nij,ij->n', rotations, rotation)-1.)/2., -1., 1.)))
            row.update(oracle_direction_deg=float(errors.min()),
                       oracle_rotation_deg=float(rotation_errors.min()),
                       query_azimuth=float(np.degrees(np.arctan2(direction[0], direction[1]))%360),
                       query_elevation=float(np.degrees(np.arcsin(-direction[2]))))
            if selected:
                row.update(top1_direction_deg=float(errors[selected[0][2]]),
                           top3_direction_deg=float(min(errors[i] for _, _, i in selected)),
                           top1_rotation_deg=float(rotation_errors[selected[0][2]]),
                           top3_rotation_deg=float(min(rotation_errors[i] for _, _, i in selected)),
                           global_top1_direction_deg=float(errors[row['global_top1']]),
                           global_top1_rotation_deg=float(rotation_errors[row['global_top1']]),
                           top10_direction_deg=float(min(errors[i] for _, _, i in ranked[:10])))
                for candidate in row['candidates']:
                    candidate['direction_error_deg'] = float(errors[candidate['index']])
                    candidate['rotation_error_deg'] = float(rotation_errors[candidate['index']])
        emit(row)
        return row, image, debug

    # New photos first, so their visual evidence is available during the larger audit.
    new_rows = []
    new_photos = sorted(args.photo_root.glob('*/*.jpeg'), key=lambda p:p.name)
    for number, path in enumerate(new_photos):
        row, image, debug = retrieve(path, 'new_photo')
        new_rows.append(row)
        sheet = Image.new('RGB', (1440, 275), 'white')
        crop = crop_box(image, debug.get('bbox'))
        sheet.paste(tile(crop, path.name+' / detected crop'), (0, 0))
        cv2.imwrite(str(args.output/(path.stem+'_crop.jpg')), crop)
        for column, candidate in enumerate(row['candidates'], 1):
            ref = cv2.imread(str(views_path/candidate['name']))
            ref = retriever._reference_crop(ref, views[candidate['index']])
            label = 'Top%d %s az%.0f el%.0f' % (column, candidate['name'],
                candidate['azimuth'], candidate['elevation'])
            sheet.paste(tile(ref, label), (column*360, 0))
        sheet.save(args.output/(path.stem+'_candidates.jpg'))
        print(json.dumps(dict(event='new_photo', name=path.name, candidates=row['candidates'])), flush=True)
    for batch in range((len(new_rows)+2)//3):
        sheet = Image.new('RGB', (1440, 825), 'white')
        for line, row in enumerate(new_rows[batch*3:batch*3+3]):
            sheet.paste(Image.open(args.output/(Path(row['name']).stem+'_candidates.jpg')), (0, line*275))
        sheet.save(args.output/('new_photos_%d.jpg'%batch))

    summary_path = DATA_DIR/'output_aliked_object_map_enriched_pilot/summary.json'
    source = json.loads(summary_path.read_text())
    pose_path = Path(source['input_files']['images'])
    image_dir = pose_path.parents[2]/'images'
    references = load_reference_images(pose_path)
    source_items = sorted(references.items())
    if args.source_limit is not None and args.source_limit < len(source_items):
        indices = np.linspace(0, len(source_items)-1, args.source_limit, dtype=int)
        source_items = [source_items[i] for i in indices]
    rows = []
    for number, (name, record) in enumerate(source_items):
        path = image_dir/name
        if not path.is_file():
            raise RuntimeError('Missing source image: '+str(path))
        row, _, _ = retrieve(path, 'reconstruction_source', record.pose)
        rows.append(row)
        if number%25 == 0 or number+1 == len(source_items):
            print(json.dumps(dict(event='progress', completed=number+1, total=len(source_items),
                top1_near=sum(r.get('top1_direction_deg', 999)<=30.0001 for r in rows),
                top3_near=sum(r.get('top3_direction_deg', 999)<=30.0001 for r in rows))), flush=True)
    def summarize(subset):
        result = dict(queries=len(subset), no_roi=sum(not r['candidates'] for r in subset))
        for metric in ['top1_direction_deg', 'top3_direction_deg', 'top10_direction_deg',
                       'top1_rotation_deg', 'top3_rotation_deg', 'global_top1_direction_deg',
                       'global_top1_rotation_deg', 'oracle_direction_deg', 'oracle_rotation_deg']:
            values = [r[metric] for r in subset if metric in r]
            result[metric] = dict(near30=sum(r.get(metric, 999)<=30.0001 for r in subset),
                over60=sum(r.get(metric, 999)>60.0001 for r in subset),
                median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None)
        durations = [r['retrieval_ms'] for r in subset]
        result['retrieval_ms'] = dict(median=float(np.median(durations)),
                                     p95=float(np.percentile(durations, 95)))
        return result
    groups = {}
    for sector in range(12):
        subset = [r for r in rows if int(r['query_azimuth']//30)==sector]
        if subset:
            groups[str(sector*30)] = summarize(subset)
    result = dict(device=args.device, mode='dino-patch', top_k=3,
        source_total=len(references), view_count=len(views), source_limit=args.source_limit,
        note='Source photos contributed to 3DGS reconstruction; correlated video frames. '
             'RC estimated poses are the numeric reference, not calibrated external truth. '
             'Separate new photos have visual face-layout truth only.',
        summary=summarize(rows), by_azimuth=groups,
        representable=summarize([r for r in rows if r['oracle_direction_deg']<=30.0001]),
        source_pose_file=str(pose_path), views=str(views_path), new_photos=new_rows)
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(event='complete', summary=result['summary'])), flush=True)


if __name__ == '__main__':
    main()
