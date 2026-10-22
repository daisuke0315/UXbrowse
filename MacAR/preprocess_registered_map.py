#!/usr/bin/env python3
"""Thorough offline enrichment: all views, then progressively weak views.

CLI and browser use the same durable worker. Existing XYZ/IDs stay intact.
The source map is never overwritten; reusable pair caches survive cancellation.
"""

from macar_paths import DATA_DIR
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from enrich_registered_aliked_map import enrich
from map_preprocessing import ROOT, JOBS, atomic_json, inspect_map


def run_job(job_dir):
    job_dir = Path(job_dir).resolve()
    config = json.loads((job_dir/'config.json').read_text())
    source = Path(config['source']).resolve()
    started = time.monotonic()
    state = dict(job_id=job_dir.name, pid=os.getpid(), source=str(source), status='running',
                 stage='load', passes=[], config=config, started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    def progress(event):
        if (job_dir/'cancel.request').exists():
            raise InterruptedError('Cancelled; completed passes and pair cache retained')
        event = dict(event)
        # Enrichment "complete" means files written, NOT job/QC complete.
        if event.get('stage') == 'complete' and state['status'] == 'running':
            event.update(stage='audit', completed=0, total=state.get('source_overview', {}).get('images', 0),
                         detail='保存した全XYZの幾何監査を開始します。監査合格までは使用しません。')
            if 'output' in event:
                event['pending_output'] = event.pop('output')
        if event.get('stage') != state.get('stage'):
            # Pair, track and image counters use different units. Never carry
            # a previous phase's numerator into the next phase's progress bar.
            state.update(completed=0, total=0, detail='', phase_started_seconds=round(time.monotonic()-started, 1))
        state.update(event, elapsed_seconds=round(time.monotonic()-started, 1))
        atomic_json(job_dir/'state.json', state)
    try:
        progress(dict(stage='inspect'))
        before = inspect_map(source, progress=progress)
        atomic_json(job_dir/'before.json', before)
        state['source_overview'] = {k:before[k] for k in ('images','features','xyz_observations','xyz_ratio','xyz_landmarks')}
        with np.load(source/'object_aliked_map.npz', allow_pickle=False) as a:
            original_xyz = a['xyz'].copy()
        explicit = set(config.get('target_images', []))
        if explicit - {r['image'] for r in before['rows']}:
            raise ValueError('Unknown selected reference')
        schedules = sorted(set([12, min(24, config['max_neighbors']), config['max_neighbors']]))
        current, latest = source, before
        for index, neighbors in enumerate(schedules):
            if config['scope'] == 'selected':
                targets = [r['image'] for r in latest['rows'] if r['image'] in explicit]
            elif index == 0 and config['scope'] == 'all':
                targets = [r['image'] for r in latest['rows']]
            else:
                targets = [r['image'] for r in latest['rows'] if r['ratio'] < config['coverage_goal'] or r['xyz'] < 300]
            if not targets:
                break
            output = job_dir / ('stage_%02d' % neighbors)
            state.update(pass_number=index+1, total_passes=len(schedules), neighbors=neighbors,
                         target_images=len(targets), completed=0, total=0, counts={}, timings={}, new_landmarks=0)
            progress(dict(stage='load', detail=f'Pass {index+1}: {len(targets)} images / {neighbors} neighbors'))
            report = enrich(SimpleNamespace(source=current, output=output, target_images=targets,
                max_neighbors=neighbors, min_anchor_support=2, max_error=3., device=config['device'],
                matching_method=config.get('matching_method', 'known-pose'),
                cache_dir=job_dir.parent/'pair_cache', progress=progress))
            progress(dict(stage='audit', completed=0, total=before['images'],
                          detail='保存した全XYZを登録カメラへ再投影して監査中'))
            latest = inspect_map(output, audit=True, progress=progress)
            with np.load(output/'object_aliked_map.npz', allow_pickle=False) as a:
                latest['original_xyz_preserved'] = bool(np.array_equal(a['xyz'][:len(original_xyz)], original_xyz))
            latest['geometry_ok'] = latest['geometry_ok'] and latest['original_xyz_preserved']
            atomic_json(output/'audit.json', latest)
            if not latest['geometry_ok']:
                raise ValueError('Geometric audit failed; result is not activated')
            for row in latest['rows']:
                original = next(r for r in before['rows'] if r['image'] == row['image'])
                row['before_xyz'] = original['xyz']
                row['added_xyz'] = row['xyz'] - original['xyz']
            atomic_json(output/'audit.json', latest)
            state['passes'].append(dict(neighbors=neighbors, targets=len(targets), output=str(output),
                xyz_landmarks=latest['xyz_landmarks'], xyz_observations=latest['xyz_observations'],
                xyz_ratio=latest['xyz_ratio'], zero_images=latest['images_without_xyz'],
                added_observations=report['counts']['added_observations'], seconds=report['processing_seconds']))
            progress(dict(stage='pass_complete', output=str(output), completed=0, total=0,
                          detail='幾何監査に合格した地図を保存しました。次は不足画像のみ補強します。'))
            current = output
        state.update(status='complete', stage='complete', completed=0, total=0, output=str(current), geometry_ok=True,
            overview={k:latest[k] for k in ('images','features','xyz_landmarks','xyz_observations','xyz_ratio',
                     'images_under_10_percent','images_without_xyz')},
            unresolved_images=sum(r['ratio'] < config['coverage_goal'] or r['xyz'] < 300 for r in latest['rows']),
            detail='All scheduled passes audited. Unresolved views are reported; completeness is not CT accuracy.')
        if current == source:
            # Nothing to process: still audit before allowing selection.
            latest = inspect_map(source, audit=True, progress=progress)
            if not latest['geometry_ok']:
                raise ValueError('Source failed geometric audit')
            # Do not mutate source; no-op jobs cannot be activated as a new map.
            state.update(status='no_changes', geometry_ok=True)
        atomic_json(job_dir/'state.json', state)
        return state
    except (InterruptedError, KeyboardInterrupt) as exc:
        state.update(status='cancelled', detail=str(exc) or 'Interrupted', elapsed_seconds=round(time.monotonic()-started,1))
        atomic_json(job_dir/'state.json', state)
        return state
    except Exception as exc:
        state.update(status='error', detail=f'{type(exc).__name__}: {exc}', elapsed_seconds=round(time.monotonic()-started,1))
        atomic_json(job_dir/'state.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path)
    parser.add_argument('--source', type=Path, default=DATA_DIR/'output_aliked_object_map_enriched_pilot')
    parser.add_argument('--max-neighbors', type=int, default=48)
    parser.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    parser.add_argument('--matching-method', choices=('known-pose','lightglue'), default='known-pose')
    args = parser.parse_args()
    if args.job_dir:
        path = args.job_dir
    else:
        if not 12 <= args.max_neighbors <= 96:
            parser.error('max-neighbors must be 12–96')
        path = JOBS/(time.strftime('%Y%m%d-%H%M%S')+'-full')
        path.mkdir(parents=True)
        atomic_json(path/'config.json', dict(source=str(args.source.resolve()), max_neighbors=args.max_neighbors,
                   device=args.device, matching_method=args.matching_method,
                   coverage_goal=.5, scope='all', target_images=[]))
    print('PREPROCESS JOB: ' + str(path), flush=True)
    print(json.dumps(run_job(path), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
