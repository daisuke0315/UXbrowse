"""Reproducible photo-only product-entry benchmark; never modifies input data."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGES = ('retrieval', 'query_features', 'reference_selection', 'reference_load',
          'lightglue', 'geometry', 'xyz_assignment', 'pnp', 'qc', 'other')


def summarize_attempts(attempts, wall_ms):
    totals = {key + '_ms': 0. for key in STAGES}
    calls = 0
    for attempt in attempts:
        diagnostic = attempt['diagnostics']
        for key in ('retrieval', 'query_features'):
            totals[key + '_ms'] += diagnostic.get('timings_ms', {}).get(key, 0.)
        if attempt['selection_computed']:
            totals['reference_selection_ms'] += diagnostic.get('reference_selection', {}).get('elapsed_ms', 0.)
        for candidate in diagnostic.get('candidates', []):
            calls += 1
            for key in ('reference_load', 'lightglue', 'geometry', 'xyz_assignment', 'pnp', 'qc'):
                totals[key + '_ms'] += candidate.get('timings_ms', {}).get(key, 0.)
    totals['other_ms'] = wall_ms - sum(totals.values())
    totals.update(candidate_calls=calls, retry_attempts=max(0, len(attempts)-1))
    return totals


def image_summary(rows):
    warm = [r for r in rows if r['repeat'] > 0 and r['completed']]
    result = dict(warm_completed=len(warm), warm_pass_count=sum(r['status'] == 'PASS' for r in warm),
                  incomplete=sum(not r['completed'] for r in rows))
    for key in ('wall_ms', 'candidate_calls', 'retry_attempts', 'inliers') + tuple(s + '_ms' for s in STAGES):
        values = [r[key] for r in warm if key in r]
        result['warm_' + key.replace('_ms', '') + '_median' + ('_ms' if key.endswith('_ms') else '')] = (
            statistics.median(values) if values else None)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    # Runtime result output, not editing source or input files.
    from browser_live import finite_json
    path.write_text(json.dumps(finite_json(data), ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def worker(photo, output):
    import cv2
    import numpy as np
    import torch
    from browser_live import DEFAULTS, runtime_arguments
    from browser_photo import decode_photo, photo_engine, localize_photo
    from realtime_pnp import build_localizer, camera_parameters

    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required for this benchmark protocol')
    settings = dict(DEFAULTS, input_source='photo')
    args, _, _ = runtime_arguments(settings)
    started = time.perf_counter()
    raw = photo.read_bytes()
    read_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    image = decode_photo(raw)
    decode_ms = (time.perf_counter()-started)*1000
    K, dist, label = camera_parameters(args, (image.shape[1], image.shape[0]))
    started = time.perf_counter()
    wrapper = build_localizer(args, temporal_registered=True)
    for owner in (wrapper, getattr(wrapper, 'tracker', None), getattr(wrapper, 'localizer', None),
                  getattr(wrapper, 'global_localizer', None)):
        if owner is not None:
            owner.mpr_renderer = None
    if hasattr(wrapper, 'viewer_coarse_enabled'):
        wrapper.viewer_coarse_enabled = False
    engine = photo_engine(wrapper, settings['photo_feature_size'])
    torch.mps.synchronize()
    config = dict(settings=settings, path=str(photo), sha256=hashlib.sha256(raw).hexdigest(),
                  shape=list(image.shape), K=K.tolist(), distortion=np.asarray(dist).tolist(),
                  calibration_label=label, init_ms=(time.perf_counter()-started)*1000,
                  read_ms=read_ms, decode_ms=decode_ms,
                  python=sys.version, platform=platform.platform(), machine=platform.machine(),
                  torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
                  device=str(engine.device), opencv_threads=cv2.getNumThreads(),
                  torch_threads=torch.get_num_threads(), seed=0,
                  source_sha256={n: digest(ROOT/n) for n in ('benchmark_photo_pipeline.py',
                      'browser_photo.py', 'browser_live.py', 'view_retrieval.py',
                      'registered_aliked_live.py', 'glb_virtual_live.py', 'real_appearance_retrieval.py')
                      if (ROOT/n).is_file()},
                  data_sha256={str(p): digest(p) for p in
                      (Path(settings['map_path'])/'summary.json', Path(settings['map_path'])/'audit.json',
                       Path(settings['map_path'])/'object_aliked_map.npz',
                       Path(settings['views_path'])/'manifest.json')})
    attempts, events, rows = [], [], []
    original = engine._localize_once

    def probe(*values, **kwargs):
        workspace = kwargs.get('workspace') or {}
        computed = 'references' not in workspace
        result = original(*values, **kwargs)
        attempts.append(dict(diagnostics=engine.last_diagnostics, selection_computed=computed))
        return result

    def sink(event, **fields):
        events.append(dict(event=event, monotonic=time.perf_counter(), **fields))
        if event == 'reference_candidate' and torch.mps.driver_allocated_memory() > 8_000_000_000:
            raise RuntimeError('RESOURCE_STOP_DRIVER_8GB')

    engine._localize_once = probe
    engine.diagnostic_sink = sink
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for repeat in range(4):
                attempts.clear()
                events.clear()
                cv2.setRNGSeed(0)
                torch.manual_seed(0)
                torch.mps.synchronize()
                start = time.perf_counter()
                result, error = None, None
                try:
                    result, _ = executor.submit(localize_photo, engine, image, K, dist).result()
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                torch.mps.synchronize()
                wall_ms = (time.perf_counter()-start)*1000
                row = dict(photo=photo.name, repeat=repeat, completed=result is not None,
                           status=result.status if result else 'INCOMPLETE', error=error,
                           wall_ms=wall_ms, result=result.as_dict() if result else None,
                           attempts=list(attempts), events=list(events),
                           memory=dict(current=torch.mps.current_allocated_memory(),
                                       driver=torch.mps.driver_allocated_memory()))
                row.update(summarize_attempts(attempts, wall_ms))
                if result is not None:
                    row.update(inliers=result.inliers, matches=result.matches, reason=result.detail,
                               mean_reprojection_error=result.mean_reprojection_error)
                    assert hashlib.sha256(image.tobytes()).hexdigest() == image_hash
                rows.append(row)
                save(output, dict(config=config, rows=rows, summary=image_summary(rows)))
                print(json.dumps({k: row[k] for k in ('photo', 'repeat', 'status', 'wall_ms',
                                                      'candidate_calls', 'retry_attempts')}) , flush=True)
                if error:
                    break
    finally:
        engine._localize_once = original
        wrapper.close()


def report(output):
    documents = [json.loads(p.read_text()) for p in sorted(output.glob('IMG_*.json'))]
    fields = ['photo', 'repeat', 'completed', 'status', 'wall_ms'] + [s+'_ms' for s in STAGES] + [
        'candidate_calls', 'retry_attempts', 'inliers', 'matches', 'mean_reprojection_error', 'reason', 'error']
    with (output/'measurements.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r for doc in documents for r in doc['rows'])
    lines = ['# 写真12画像：工程別ベンチマーク実測', '',
             '登録実写ALIKED／現行既定・厳格QC。初回は写真ごとの新規プロセス、反復は同一写真3回。',
             '時間は姿勢決定の壁時計（GPU待機・キャッシュ解放・debug画像含む）。読込・初期化・MPR・ブラウザ表示は別。',
             'FAILも完走時間に含む。QC PASSは真値精度ではない。開発に使った写真群であり外部評価ではない。', '',
             '|写真|初回 秒|反復中央値 秒|反復PASS/完走|候補照合 中央値|再試行 中央値|PnP採用 中央値|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for doc in documents:
        s, rows = doc['summary'], doc['rows']
        lines.append(f"|{Path(doc['config']['path']).name}|{rows[0]['wall_ms']/1000:.3f}|"
                     f"{(s['warm_wall_median_ms']/1000 if s['warm_wall_median_ms'] is not None else float('nan')):.3f}|"
                     f"{s['warm_pass_count']}/{s['warm_completed']}|{s.get('warm_candidate_calls_median')}|"
                     f"{s.get('warm_retry_attempts_median')}|{s.get('warm_inliers_median')}|")
    lines += ['', '## 工程別（画像ごとの反復中央値を集計、ms）', '',
              '|工程|中央値|最小|最大|', '|---|---:|---:|---:|']
    for stage in ('wall',) + STAGES:
        values = [d['summary'].get('warm_'+stage+'_median_ms') for d in documents]
        values = [v for v in values if v is not None]
        if values:
            lines.append(f'|{stage}|{statistics.median(values):.1f}|{min(values):.1f}|{max(values):.1f}|')
    lines += ['', '画像別の工程・全48回の成否・再投影誤差・理由は measurements.csv、全候補・設定・hashは各JSON。',
              '工程中央値の合計は総時間中央値とは一致しない（各々別の中央値）。QCはspatial採点を内包し二重加算しない。',
              'プロセスcoldはOS/ディスクcache coldではない。例外・安全停止・timeoutは未完走として扱う。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker-photo', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker_photo:
        worker(options.worker_photo, options.output)
        return
    photos = sorted(p for p in options.photo_root.rglob('*') if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    if len(photos) != 12 or len({p.stem for p in photos}) != 12:
        raise ValueError('Exactly 12 uniquely named photos required')
    options.output.mkdir(parents=True, exist_ok=False)
    (options.output/'inputs.json').write_text(json.dumps([
        dict(path=str(p), sha256=digest(p)) for p in photos], ensure_ascii=False, indent=2))
    for photo in photos:
        with (options.output/(photo.stem+'.log')).open('w') as log:
            try:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--worker-photo', str(photo), '--output', str(options.output/(photo.stem+'.json'))],
                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
                print(photo.name, 'process exit', completed.returncode, flush=True)
            except subprocess.TimeoutExpired:
                print(photo.name, 'TIMEOUT 300s', flush=True)
                (options.output/(photo.stem+'.timeout')).write_text('300s process timeout; incomplete\n')
        report(options.output)
    assert all(digest(p) == r['sha256'] for p, r in zip(photos, json.loads((options.output/'inputs.json').read_text())))


if __name__ == '__main__':
    main()
