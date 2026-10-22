"""Summarize recorded standalone comparisons; no model inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    root=args.output
    def read(name):
        return [json.loads(l) for l in (root/name).read_text().splitlines()]
    initial=read('shortlist.jsonl')
    expanded=read('expanded.jsonl')
    exhaustive=read('exhaustive.jsonl')
    recheck=read('recheck.jsonl')
    if len(initial)!=56 or len(expanded)!=6 or len(exhaustive)!=5 or len(recheck)!=2:
        raise ValueError('Incomplete experiment; refuse final summary')
    groups={}
    for kind in ['new','source']:
        for mode in ['gs','real']:
            rows=[r for r in initial if r['kind']==kind and r['mode']==mode and r['repeat']==0]
            def median(values):return float(np.median(values))
            group=dict(count=len(rows),passed=sum(r['status']=='PASS' for r in rows),
                accepted_correct=sum(r['status']=='PASS' and r.get('pose_correct',r.get('face_correct',False)) for r in rows),
                wall_median_ms=median([r['wall_ms'] for r in rows]),
                processed_median=median([r['processed'] for r in rows]),
                retrieval_median_ms=median([r['diagnostics']['timings_ms']['retrieval'] for r in rows]),
                extraction_median_ms=median([r['diagnostics']['timings_ms']['query_features'] for r in rows]),
                lightglue_median_ms=median([sum(c['timings_ms'].get('lightglue',0) for c in r['diagnostics']['candidates']) for r in rows]),
                pnp_median_ms=median([r['result']['pnp_ms'] for r in rows]))
            if kind=='source':
                group.update(rotation_median_deg=median([r['rotation_error_deg'] for r in rows]),
                             rotation_max_deg=max(r['rotation_error_deg'] for r in rows),
                             center_median_units=median([r['center_error_units'] for r in rows]),
                             center_max_units=max(r['center_error_units'] for r in rows))
            groups[kind+'_'+mode]=group
    expanded_by={(r['name'],r['mode']):r for r in expanded}
    final_new={}
    for mode in ['gs','real']:
        first=[r for r in initial if r['kind']=='new' and r['mode']==mode and r['repeat']==0]
        final=[expanded_by.get((r['name'],mode),r) for r in first]
        final_new[mode]=dict(count=len(final),passed=sum(r['status']=='PASS' for r in final),
            accepted_correct=sum(r['status']=='PASS' and r['face_correct'] for r in final),
            false_face_pass=[r['name'] for r in final if r['status']=='PASS' and not r['face_correct']],
            fail=[r['name'] for r in final if r['status']!='PASS'])
    controlled=[]
    for r in exhaustive[:3]+recheck:
        suffix='_recheck' if r in recheck else ''
        costs=json.loads((root/(Path(r['name']).stem+suffix+'_common_costs.json')).read_text())
        preparation=costs['shared_features_ms']+costs[r['mode']+'_retrieval_ms']
        controlled.append(dict(name=r['name'],mode=r['mode'],status=r['status'],
            face_correct=r.get('face_correct'),processed=r['processed'],
            search_wall_ms=r['wall_ms'],preparation_measured_ms=preparation,
            summed_pipeline_ms=r['wall_ms']+preparation,
            phase='recheck' if r in recheck else 'exhaustive',
            comment='Component sum from shared preparation; not a separate end-to-end latency sample.',
            selected=r['diagnostics'].get('selected'),inliers=r['result']['inliers'],
            lightglue_ms=sum(c['timings_ms'].get('lightglue',0) for c in r['diagnostics']['candidates']),
            pnp_ms=r['result']['pnp_ms']))
    result=dict(initial=groups,final_new=final_new,controlled=controlled,
                post_scan_controls=[{k:r[k] for k in ['name','mode','status','wall_ms','processed','face_correct']} for r in exhaustive[3:]],
                expanded_cases=[{k:r[k] for k in ['name','mode','status','wall_ms','processed','face_correct']} for r in expanded],
                repeat_cases=[{k:r[k] for k in ['name','mode','repeat','status','wall_ms','processed','face_correct']} for r in initial if r['name'] in ['IMG_2593.jpeg','IMG_2598.jpeg']],
                limitations=['Source images contributed to RC/3DGS/map; exclude +/-5 frames but not held-out landmarks.',
                    'Separate photos have face identity truth only, not numeric 6DoF pose truth.',
                    'MPS standalone warm process without its own GS rendering/UI; other host workloads were not stopped or controlled. Model/index startup and decode excluded.',
                    'Initial arms use their own coarse camera rotation prior with the same thresholds.',
                    'Controlled arms disable the rotation rejection and early exit in all three routes.',
                    'Second 620-reference scan was stopped after macOS process footprint reached ~38GB; first full scan completed. Six-reference controls were rechecked in a fresh process.',
                    'Shared GS-derived detector, masks and XYZ map; this is not complete removal of 3DGS.',
                    'Timings vary across repeated shapes; this experiment does not resolve the previous 23-second MPS stall.',
                    'JSON legacy pose_comparison.reference=3dgs_view names the registered-photo camera in real/all arms.'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
