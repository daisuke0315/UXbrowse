"""Summarize completed isolated matching comparisons, including false poses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    args=p.parse_args();root=args.output
    metadata=json.loads((root/'inputs.json').read_text())
    rows=[]
    for f in root.glob('*_results.json'):rows.extend(json.loads(f.read_text()))
    if len(list(root.glob('*_complete.json')))!=72 or len(rows)!=96:
        raise ValueError('Incomplete comparison; refuse final aggregate')
    groups={}
    modes=['lightglue','photo_nn','global_nn','global_nn_relaxed']
    for kind in ['new','source']:
        for mode in modes:
            x=[r for r in rows if r['kind']==kind and r['mode']==mode]
            correct=lambda r:r['status']=='PASS' and r.get('pose_correct',r.get('face_correct',False))
            g=dict(count=len(x),pass_count=sum(r['status']=='PASS' for r in x),
                accepted_correct=sum(correct(r) for r in x),
                failed=[r['name'] for r in x if r['status']!='PASS'],
                accepted_incorrect=[r['name'] for r in x if r['status']=='PASS' and not correct(r)],
                pipeline_component_median_ms=float(np.median([r['pipeline_component_sum_ms'] for r in x])),
                pipeline_component_max_ms=max(r['pipeline_component_sum_ms'] for r in x),
                inlier_median=float(np.median([r['inliers'] for r in x])),
                footprint_max_bytes=max(r['footprint_bytes'] for r in x))
            if kind=='source':
                poses=[r for r in x if 'rotation_error_deg' in r]
                g.update(rotation_error_median_deg=float(np.median([r['rotation_error_deg'] for r in poses])),
                    rotation_error_max_deg=max(r['rotation_error_deg'] for r in poses),
                    center_error_median_units=float(np.median([r['center_error_units'] for r in poses])),
                    center_error_max_units=max(r['center_error_units'] for r in poses))
            groups[kind+'_'+mode]=g
    agreement=[]
    for name in sorted({r['name'] for r in rows}):
        x=[r for r in rows if r['name']==name]
        same_features=len({r['query_feature_sha256'] for r in x})==1
        same_roi=len({tuple(r['roi']) for r in x})==1
        if not same_features or not same_roi:raise ValueError('Feature or ROI mismatch '+name)
        agreement.append(dict(name=name,features_identical=same_features,roi_identical=same_roi))
    checks={}
    for f,h in metadata['comparison_code_sha256'].items():
        checks[f]=hashlib.sha256((Path(__file__).resolve().parent/f).read_bytes()).hexdigest()==h
    checks['object_aliked_map.npz']=hashlib.sha256((Path(metadata['map_path'])/'object_aliked_map.npz').read_bytes()).hexdigest()==metadata['map_sha256']
    if not all(checks.values()):raise ValueError('Source/map fingerprint changed')
    result=dict(groups=groups,query_agreement=agreement,source_integrity=checks,
        conditions=metadata['matcher_comparison'],
        paired_rows=sorted(rows,key=lambda r:(r['query_index'],modes.index(r['mode']))),
        limitations=['12 separate photos represent four view sectors, with nearby triplets, not 12 independent environments.',
            'Separate photos have physical face identity truth, not calibrated numeric 6DoF pose truth.',
            'Source photos contributed to reconstruction; same query and +/-5 frames excluded from both reference observations and global bank, but landmark geometry and GS still use them.',
            'RC saved estimated poses are the numeric comparison reference, not external measurement.',
            'Only the photo-NN arm preserves the current shortlist, image-pair geometry, directional gate and competing-reference checks.',
            'Global NN uses common PnP/numeric QC without coarse direction gate, per-photo fundamental test or competing-photo hypotheses.',
            'Component sums exclude startup/warmup and photo decode; cold map/index startup and app browser/network latency are not evaluated.',
            'NN workers still load the same localizer and perform one LightGlue warmup outside measurement; measured NN matching invokes no LightGlue. Full removal memory/startup effect is not measured.',
            'Other workloads on the host are not stopped; benchmark does not certify resolution of the earlier 23-second MPS wait.'])
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(groups,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
