"""Read-only runtime log audit. Keep photo, full search and local refresh separate.

Candidate spans include synchronization, not just GPU kernel time. Historical
logs without candidate timers cannot attribute those spans to individual calls.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def read_session(path):
    groups, active = [], {}
    settings, frame = {}, {}
    for line in path.open(encoding='utf-8'):
        try:
            r = json.loads(line)
        except ValueError:
            continue  # An active writer may have a partial final line.
        event = r.get('event')
        if event == 'configuration':
            settings = r.get('settings', {})
        if event == 'frame_sample':
            frame = r
            for g in groups:
                if (g.get('done') and g.get('first_pass') is None
                    and r.get('status') == 'PASS' and r.get('result')
                    and abs(r['result'].get('processing_ms', -1)-g['done'].get('total_ms', -2)) < .01):
                    g['first_pass'] = r['monotonic']
            continue
        if event not in ('search_stage', 'reference_candidate', 'reference_search_decision',
                         'search_completed', 'search_submitted'):
            continue
        sid = r.get('search_id')
        if (sid not in active or
            (event == 'search_stage' and r.get('stage') != 'done' and active[sid].get('done'))):
            g = dict(file=path.name, search_id=sid, mode=r.get('search_mode'),
                     source=settings.get('input_source'), profile=frame.get('quality_profile') or settings.get('quality_profile'),
                     photo_feature_size=settings.get('photo_feature_size'), map_path=settings.get('map_path'),
                     settings=settings.copy(), stages=[], candidates=[], decisions=[], first_pass=None)
            active[sid] = g
            groups.append(g)
        g = active[sid]
        if event == 'search_stage':
            g['stages'].append(r)
            if r.get('stage') == 'done':
                g['done'] = r
        elif event == 'reference_candidate':
            g['candidates'].append(r['candidate'])
        elif event == 'reference_search_decision':
            g['decisions'].append(r)
        elif event == 'search_completed':
            g['completed'] = r
        elif event == 'search_submitted':
            g['submitted'] = r
    result = []
    for g in groups:
        if not g.get('done') or g['done'].get('outcome') not in ('PASS', 'FAIL'):
            continue
        stages = g['stages']
        begin = next((r for r in stages if r.get('stage') == 'retrieval'), stages[0])
        done = g['done']
        row = {k:g[k] for k in ['file', 'search_id', 'mode', 'source', 'profile',
                               'photo_feature_size', 'map_path']}
        row.update(start_wall=begin['wall_time'], outcome=done.get('outcome'),
                   reason=done.get('reason'), inliers=done.get('inliers'),
                   total_ms=done.get('total_ms'), wall_span_ms=(done['monotonic']-begin['monotonic'])*1000,
                   coarse_ms=0., features_ms=0., candidates=g['candidates'],
                   candidate_timing_sums_ms=dict(), reference_search=[{k:r.get(k) for k in
                        ['policy','processed','total','skipped','early_exit','reason','checkpoints']}
                        for r in g['decisions']],
                   full_search=any(r.get('stage')=='retrieval' for r in stages),
                   xyz_retry=any(r.get('retry') for r in stages),
                   stages=[dict(stage=r.get('stage'), ms=(r['monotonic']-begin['monotonic'])*1000,
                            candidate_index=r.get('candidate_index'), retry=r.get('retry'),
                            query_features=r.get('query_features'), correspondences=r.get('correspondences'))
                           for r in stages])
        for a, b in zip(stages, stages[1:]):
            elapsed = (b['monotonic']-a['monotonic'])*1000
            if a.get('stage') == 'retrieval' and b.get('stage') == 'features':
                row['coarse_ms'] += elapsed
            if a.get('stage') == 'features':
                row['features_ms'] += elapsed
        times = Counter()
        for c in g['candidates']:
            times.update(c.get('timings_ms', {}))
        row['candidate_timing_sums_ms'] = dict(times)
        row['candidate_count'] = len(g['candidates'])
        row['cached_candidates'] = sum(bool(c.get('reference_cached')) for c in g['candidates'])
        row['timing_unattributed_ms'] = (row['wall_span_ms']-row['coarse_ms']-row['features_ms']-sum(times.values()))
        if g['first_pass'] is not None:
            row['done_to_first_pass_ms'] = (g['first_pass']-done['monotonic'])*1000
        if g.get('completed'):
            row['result'] = g['completed'].get('result')
            row['capture_age_ms'] = g['completed'].get('age_ms')
            row['diagnostics'] = g['completed'].get('diagnostics')
        result.append(row)
    return result


def stats(rows):
    result = dict(n=len(rows), pass_count=sum(r['outcome']=='PASS' for r in rows),
                  retries=sum(r['xyz_retry'] for r in rows),
                  early_exits=sum(any(d['early_exit'] for d in r['reference_search']) for r in rows))
    for key in ['total_ms','wall_span_ms','coarse_ms','features_ms','candidate_count',
                'done_to_first_pass_ms','timing_unattributed_ms']:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),
                               max=float(max(values)))
    for key in ['reference_load','lightglue','geometry','xyz_assignment','pnp','qc']:
        values = [r['candidate_timing_sums_ms'][key] for r in rows
                  if key in r['candidate_timing_sums_ms']]
        if values:
            result[key] = dict(n=len(values), median=float(np.median(values)),
                               p95=float(np.percentile(values,95)), max=float(max(values)))
    return result


def expansion_duplicates(rows):
    result = []
    for a, b in zip(rows, rows[1:]):
        if not (a['source'] == 'photo' and a['file'] == b['file']
                and a['search_id'] == b['search_id'] and a['full_search'] and b['full_search']):
            continue
        previous = {c['name']: c for c in a['candidates']}
        duplicate = [c for c in b['candidates'] if c['name'] in previous]
        result.append(dict(file=a['file'], outcomes=[a['outcome'], b['outcome']],
            first_candidates=a['candidate_count'], expanded_candidates=b['candidate_count'],
            duplicate_references=len(duplicate),
            duplicate_raw_count_equal=sum(c['raw'] == previous[c['name']]['raw'] for c in duplicate),
            repeated_features_ms=b['features_ms'], repeated_coarse_ms=b['coarse_ms'],
            duplicate_lightglue_ms=sum(c.get('timings_ms', {}).get('lightglue', 0) for c in duplicate),
            duplicate_candidate_stages_ms=sum(sum(c.get('timings_ms', {}).values()) for c in duplicate),
            duplicate_work_ms=b['features_ms']+b['coarse_ms']+
                              sum(sum(c.get('timings_ms', {}).values()) for c in duplicate),
            combined_attempt_ms=a['total_ms']+b['total_ms']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a fresh output file')
    rows = [row for p in sorted(args.logs.glob('*.jsonl')) for row in read_session(p)]
    grouped = defaultdict(list)
    for r in rows:
        if r['file'].startswith('20261002'):
            grouped['%s / %s / %s / %s'%(r['source'],r['profile'],r['mode'],
                     'full' if r['full_search'] else 'local')].append(r)
    summaries = {k:stats(v) for k,v in grouped.items()}
    output = dict(note='Sums of separately timed candidates are wall-clock spans; GPU attribution requires synchronization-aware profiling. Logs/configurations differ; this is not a controlled benchmark.',
                  sessions=len(list(args.logs.glob('*.jsonl'))), rows=rows, summaries=summaries,
                  photo_expansion_duplicates=expansion_duplicates(rows))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(dict(searches=len(rows),summaries=summaries),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
