#!/usr/bin/env python3
"""Static-photo timing comparison; NOT a live tracking or CT accuracy test."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from registered_aliked_live import RegisteredALIKEDLiveLocalizer
from realtime_pnp import (DEFAULT_REGISTERED_MAP, DEFAULT_SPLAT_VIEWS,
                          approximate_camera_matrix, resize_for_processing)
from browser_live import projected_model_faces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo',type=Path)
    parser.add_argument('--map',type=Path,default=DEFAULT_REGISTERED_MAP)
    parser.add_argument('--views',type=Path,default=DEFAULT_SPLAT_VIEWS)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--feature-size',type=int,default=1024)
    parser.add_argument('--validation-faces',action='store_true')
    args=parser.parse_args()
    image=cv2.imread(str(args.photo))
    if image is None:
        raise ValueError(f'Photo unreadable: {args.photo}')
    image,_=resize_for_processing(image,args.width)
    h,w=image.shape[:2]
    matrix=approximate_camera_matrix(w,h,72.)
    distortion=np.zeros(5)
    engine=RegisteredALIKEDLiveLocalizer(args.map,args.views,device=args.device,
        max_keypoints=2000,resize=1024,fundamental_threshold=3.,ransac_threshold=4.,
        max_reprojection_error=3.,retrieval_mode='dino-patch')
    engine.query_resize=args.feature_size
    initial,_=engine.localize(image,matrix,distortion)
    print('INITIAL '+json.dumps(initial.as_dict()),flush=True)
    if initial.status!='PASS':
        raise ValueError('Initial photo must PASS before testing pose-local replenishment')
    prior=(np.asarray(initial.rvec),np.asarray(initial.tvec))
    initial_references=engine.last_diagnostics['selected_reference_views']
    if args.validation_faces:
        for face in projected_model_faces(engine.manifest['bounds'],initial,matrix,distortion):
            polygon=np.asarray(face['polygon'],np.float32)
            before,_=engine.localize(image,matrix,distortion,excluded_polygon=polygon)
            before_quality=engine.last_diagnostics.get('quality')
            after,_,diagnosis=engine.localize_validation(image,matrix,distortion,polygon,3.,0.,
                                                        initial_references,prior)
            print('FACE_VALIDATION '+json.dumps(dict(face=face['id'],old=before.as_dict(),
                old_quality=before_quality,new=after.as_dict(),quality=diagnosis.get('quality'))),flush=True)
    ranked=[(1.,0.,i) for i in engine._handoff_views(*prior,None)]
    old,_=engine._search(image,matrix,distortion,ranked_override=ranked)
    print('OLD_GUIDED '+json.dumps(old.as_dict()),flush=True)
    for run in range(args.runs):
        answer,_=engine.localize_guided(image,matrix,distortion,*prior)
        print('POSE_LOCAL '+json.dumps(dict(run=run,cold_bank=run==0,
            result=answer.as_dict(),diagnostics=engine.last_diagnostics)),flush=True)


if __name__=='__main__':
    main()
