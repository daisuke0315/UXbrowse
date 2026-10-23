"""Recorded-screen still-image check, not a raw-camera tracking benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from browser_live import DEFAULTS, runtime_arguments, finite_json
from realtime_pnp import build_localizer, camera_parameters
from realtime_pnp_server import TrackingSeed, TemporalGlobalPnPLocalizer
from glb_virtual_live import AsyncTemporalGLBLocalizer
from live_search_resources import run_live_search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--camera-rect',type=int,nargs=4,required=True,metavar=('X','Y','W','H'))
    parser.add_argument('--region',type=int,nargs=4,required=True,metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--quality',choices=('standard','precision'),default='standard')
    parser.add_argument('--event-context',action='store_true',
        help='Route context through the live new-view event, inside the 4-second interval')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    capture=cv2.VideoCapture(str(args.video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC,args.seconds*1000)
        ok,frame=capture.read()
    finally:capture.release()
    if not ok:raise ValueError('Video frame unavailable')
    x,y,w,h=args.camera_rect
    if x<0 or y<0 or w<=0 or h<=0 or x+w>frame.shape[1] or y+h>frame.shape[0]:
        raise ValueError('Camera rectangle outside recorded frame')
    image=cv2.resize(frame[y:y+h,x:x+w],(1280,720),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.output/'camera.png'),image)
    runtime,_,_=runtime_arguments(dict(DEFAULTS,mpr=False,quality_profile=args.quality))
    wrapper=build_localizer(runtime,temporal_registered=True)
    engine=wrapper.global_localizer
    K,dist,_=camera_parameters(runtime,(1280,720))
    box=np.array(args.region)
    def in_region(uv):
        uv=np.asarray(uv).reshape(-1,2)
        return ((uv[:,0]>=box[0]) & (uv[:,0]<box[2]) &
                (uv[:,1]>=box[1]) & (uv[:,1]<box[3]))
    report=dict(source=dict(video=str(args.video),seconds=args.seconds,camera_rect=args.camera_rect,
        region=args.region,sha256=hashlib.sha256(args.video.read_bytes()).hexdigest()),
        quality=args.quality,event_context=args.event_context,
        note='Screen-recording still image; synthetic upper-only seed; not raw live replay',
        cases={},sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('registered_aliked_live.py','glb_virtual_live.py','replenishment_refinement.py',
             'guided_matching.py','background_refresh.py','viewpoint_refresh.py',
             'search_frame_quality.py','benchmark_regional_replenishment.py')})
    seeds={}
    try:
        initial,_=run_live_search(engine,engine.localize,image,K,dist)
        report['initial']=initial.as_dict()
        if initial.status!='PASS':raise ValueError('Initial screen frame failed QC')
        prior=np.array(initial.rvec),np.array(initial.tvec)
        for name,method in [('fast',engine._localize_landmarks),('context',engine._localize_pose_references)]:
            if name == 'context' and args.event_context:
                engine.last_context_started = time.monotonic()
                engine.diagnostic_context = dict(search_mode='guided', trigger=['view_direction_emerging'])
                method = engine.localize_guided
            started=time.perf_counter()
            answer,_=run_live_search(engine,method,image,K,dist,*prior)
            elapsed=(time.perf_counter()-started)*1000
            debug=engine.last_debug
            uv=debug.image_points[np.array(sorted(debug.inlier_indices),int)]
            seeds[name]=engine.last_tracking_seed
            canvas=image.copy()
            for point in uv:
                cv2.circle(canvas,tuple(np.round(point).astype(int)),3,(0,255,0),1)
            cv2.rectangle(canvas,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
            cv2.imwrite(str(args.output/f'{name}.png'),canvas)
            report['cases'][name]=dict(result=answer.as_dict(),wall_ms=elapsed,
                region_inliers=int(in_region(uv).sum()),diagnostics=engine.last_diagnostics)
            np.savez_compressed(args.output/f'{name}.npz',uv=debug.image_points,xyz=debug.matched_xyz,
                inliers=np.array(sorted(debug.inlier_indices),int),K=K,dist=dist,
                rvec=np.asarray(answer.rvec),tvec=np.asarray(answer.tvec))
        base=seeds['fast']
        if base is None:raise ValueError('Fast reference seed unavailable')
        keep=~in_region(base.image_points)
        old=TrackingSeed(base.gray.copy(),base.image_points[keep].copy(),base.object_points[keep].copy(),
                        base.rvec.copy(),base.tvec.copy())
        for name in ('fast','context'):
            if seeds[name] is None:continue
            actor=AsyncTemporalGLBLocalizer.__new__(AsyncTemporalGLBLocalizer)
            actor.global_localizer=engine
            actor.tracker=TemporalGlobalPnPLocalizer(engine)
            actor.tracker.seed=TemporalGlobalPnPLocalizer._copy_seed(old)
            actor.last_search_diagnostics={'reference_matching':'pose_references' if name=='context' else 'pose_landmarks'}
            added=actor._replenish_current_seed(seeds[name],image,K,dist)
            checked=actor.tracker._track(image,K,dist)
            points=actor.tracker.seed.image_points if checked is not None else np.empty((0,2))
            report['cases'][name]['integration']=dict(added=added,final_pass=checked is not None,
                region_after_final_pnp=int(in_region(points).sum()),diagnostics=actor.last_replenish_diagnostics,
                refinement=actor.last_refresh_refinement,seed_before=len(old.image_points))
            canvas=image.copy()
            for point in points:
                cv2.circle(canvas,tuple(np.round(point).astype(int)),3,
                    (0,255,0) if in_region([point])[0] else (180,180,180),1)
            cv2.rectangle(canvas,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
            cv2.imwrite(str(args.output/f'integrated-{name}.png'),canvas)
        (args.output/'report.json').write_text(json.dumps(finite_json(report),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({name:dict(inliers=data['result']['inliers'],region_inliers=data['region_inliers'],
              integration=data.get('integration')) for name,data in report['cases'].items()},ensure_ascii=False),flush=True)
    finally:wrapper.close()


if __name__=='__main__':main()
