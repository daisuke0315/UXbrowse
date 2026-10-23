"""Replay a rendered 360-degree orbit through the actual asynchronous tracker.

This is a known-pose synthetic regression, not a live-camera accuracy claim.
Reference map geometry is unchanged. Query views are newly rendered RGB only.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from browser_live import runtime_arguments, build_localizer
from generate_3dgs_reference_views import look_at_opencv

ROOT = Path(__file__).resolve().parent


def generate(folder, count):
    folder.mkdir(parents=True, exist_ok=False)
    _, manifest, ply = runtime_arguments({'mpr': False})
    center = np.asarray(manifest['center'])
    distance = manifest['radius']*2.2
    views = []
    width, height, fov = 960, 720, 65.
    focal = height/(2*math.tan(math.radians(fov)/2))
    matrix = [[focal,0,(width-1)/2], [0,focal,(height-1)/2], [0,0,1]]
    for i in range(count):
        azimuth = i/count*2*math.pi
        elevation = math.radians(35+8*math.sin(azimuth))
        eye = center+distance*np.array([math.sin(azimuth)*math.cos(elevation),
                                        math.cos(azimuth)*math.cos(elevation),
                                        -math.sin(elevation)])
        transform, _ = look_at_opencv(eye, center, -1.)
        views.append(dict(view_id=i, image_name=f'query_{i:04d}.png', eye=eye.tolist(),
                          world_to_camera=transform.tolist(), camera_matrix=matrix))
    plan = dict(ply=str(ply), width=width, height=height, vertical_fov_deg=fov,
                target=center.tolist(), up=[0,0,-1], views=views)
    path = folder/'plan.json'
    path.write_text(json.dumps(plan, indent=2))
    node = Path('/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    with (folder/'render.log').open('w') as log:
        subprocess.run([str(node),str(ROOT/'tools/gs_browser_renderer/capture_batch.mjs'),
                        str(path),str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Rendered {count} novel query views in {folder}', flush=True)


def replay(folder, output, fps, occlusion, baseline=False):
    if output.exists():
        raise ValueError('Use a new output filename')
    plan = json.loads((folder/'plan.json').read_text())
    source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('glb_virtual_live.py','realtime_pnp_server.py','tracking_handoff.py','background_refresh.py',
         'tracking_recovery.py','replenishment_refinement.py','benchmark_orbit_tracking.py',
         'pose_quality.py','pose_observability.py')}
    images = [cv2.imread(str(folder/v['image_name'])) for v in plan['views']]
    if any(image is None for image in images):
        raise ValueError('Missing rendered frames')
    args, _, _ = runtime_arguments({'mpr':False,
                                  'refresh_policy':'background' if baseline else 'orbit-experimental'})
    args.virtual_diagnostics = None
    cv2.setNumThreads(4)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.with_suffix('.log').open('x') as log, contextlib.redirect_stdout(log):
        actor = build_localizer(args)
        if baseline:
            actor.short_recovery = None
            actor.resilient_transport = False
            actor.background_refresh.fast_recovery = False
            # Preserve previous scheduling for a paired comparison.
            decide = actor.background_refresh.decide
            actor.background_refresh.decide = lambda now, **kw: decide(now,
                **dict(kw,weak_reasons=[r for r in kw['weak_reasons']
                                      if r not in ('support_critical','pose_support_weak')]))
        try:
            K = np.asarray(plan['views'][0]['camera_matrix'],np.float64)
            D = np.zeros(5)
            boot = time.monotonic()
            while time.monotonic()-boot < 20:
                start = time.monotonic()
                actor.record_capture(images[0],K,D,start)
                result, _ = actor.localize(images[0],K,D)
                if result.status == 'PASS':
                    break
                time.sleep(max(0.,1/fps-(time.monotonic()-start)))
            boot_seconds = time.monotonic()-boot
            bootstrap_passed = result.status == 'PASS'
            sequence_started = time.monotonic()
            for index,(view,original) in enumerate(zip(plan['views'],images)):
                sample = original.copy()
                hidden = occlusion and index in (60,61,120)
                if hidden:
                    sample[:] = 0  # Short total occlusion; those frames must NOT PASS.
                started = time.monotonic()
                actor.record_capture(sample,K,D,started,index)
                result, _ = actor.localize(sample,K,D)
                elapsed = (time.monotonic()-started)*1000
                row = dict(index=index, status=result.status, inliers=result.inliers,
                    sequence_seconds=started-sequence_started,
                    detail=result.detail, foreground_ms=elapsed, occluded=hidden,
                    failure=actor.last_track_failure, added=actor.last_replenished,
                    search_active=actor.future is not None, search_mode=actor.search_mode,
                    background=actor.background_refresh.snapshot(),
                    recovery=actor.short_recovery.snapshot(time.monotonic()) if actor.short_recovery else {},
                    refinement=getattr(actor,'last_refresh_refinement',{}).copy(),
                    pose_quality=getattr(actor.global_localizer,'last_tracking_quality',{}).copy())
                if result.status == 'PASS':
                    expected = np.asarray(view['world_to_camera'])
                    rotation = cv2.Rodrigues(np.asarray(result.rvec))[0]
                    row['rotation_error_deg'] = float(np.degrees(np.arccos(np.clip(
                        (np.trace(rotation@expected[:3,:3].T)-1)/2,-1,1))))
                    center = -rotation.T@np.asarray(result.tvec).reshape(3)
                    row['center_error_mm'] = float(np.linalg.norm(center-np.asarray(view['eye'])))
                rows.append(row)
                time.sleep(max(0.,1/fps-(time.monotonic()-started)))
            sequence_seconds = time.monotonic()-sequence_started
        finally:
            actor.executor.shutdown(wait=True, cancel_futures=True)
            actor.coarse_executor.shutdown(wait=True, cancel_futures=True)
    visible = [r for r in rows if not r['occluded']]
    accepted = [r for r in visible if r['status']=='PASS']
    longest, run, lost_events, previous = 0,0,0,False
    for row in visible:
        lost = row['status']!='PASS'
        run = run+1 if lost else 0
        longest = max(longest,run)
        lost_events += int(lost and not previous)
        previous = lost
    summary = dict(frames=len(rows), visible_pass=len(accepted), visible_frames=len(visible),
        pass_ratio=len(accepted)/max(1,len(visible)), lost_events=lost_events,
        sequence_seconds=sequence_seconds,
        longest_lost_seconds=longest/fps, bootstrap_seconds=boot_seconds,
        bootstrap_passed=bootstrap_passed,
        occluded_pass=sum(r['occluded'] and r['status']=='PASS' for r in rows),
        foreground_p50_ms=float(np.median([r['foreground_ms'] for r in rows])),
        foreground_p95_ms=float(np.percentile([r['foreground_ms'] for r in rows],95)),
        added_points=sum(r['added'] for r in rows),
        rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in accepted])) if accepted else None,
        rotation_p95_deg=float(np.percentile([r['rotation_error_deg'] for r in accepted],95)) if accepted else None,
        center_median_mm=float(np.median([r['center_error_mm'] for r in accepted])) if accepted else None,
        center_p95_mm=float(np.percentile([r['center_error_mm'] for r in accepted],95)) if accepted else None)
    with output.open('x') as stream:
        json.dump(dict(summary=summary, fps=fps, occlusion=occlusion, baseline=baseline,
            note='Synthetic GS RGB orbit; known K; no MPR/browser load; not independent live accuracy.',
            source_hashes=source_hashes,
            rows=rows),stream,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','replay'])
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--count',type=int,default=180)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fps',type=float,default=10.)
    parser.add_argument('--occlusion',action='store_true')
    parser.add_argument('--baseline',action='store_true',help='Disable new short recovery and drift refinement')
    args=parser.parse_args()
    if args.count<12 or not 0<args.fps<=60 or (args.command=='replay' and not args.output):
        parser.error('Require count >= 12, 0 < fps <= 60 and replay --output')
    if args.command=='generate':generate(args.frames,args.count)
    else:replay(args.frames,args.output,args.fps,args.occlusion,args.baseline)
