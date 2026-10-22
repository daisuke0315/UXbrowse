"""Standalone LightGlue / same-photo NN / global 3D-descriptor NN comparison.

Global NN is an exact, chunked float32 dot-product search over every saved
XYZ-backed descriptor observation. It uses no photo shortlist or pose truth.
Variants of one landmark are not each other's ratio-test runner-up. Each query
runs in a fresh process, bounding the lifetime of Torch/MPS retained state.
No production implementation, map or application settings are changed.
"""

from macar_paths import DATA_DIR
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME',str(DATA_DIR/'.model_cache/torch'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output,metadata):
    output.mkdir(parents=True,exist_ok=False)
    from offline_pnp_test import load_reference_images
    map_path=Path(metadata['map_path'])
    summary=json.loads((map_path/'summary.json').read_text())
    records=load_reference_images(Path(summary['input_files']['images']))
    xyz=np.load(map_path/'object_aliked_map.npz',allow_pickle=False)['xyz']
    files=[(name,map_path/'reference_features'/(Path(name).stem+'.npz'))
           for name in sorted(records) if (map_path/'reference_features'/(Path(name).stem+'.npz')).is_file()]
    counts=[]
    fingerprint=[]
    started=time.perf_counter()
    def read(file):
        with np.load(file,allow_pickle=False) as a:
            ids=a['point_ids'];positions=a['xyz'];descriptors=a['descriptors'][0]
            norms=np.linalg.norm(descriptors,axis=1)
            keep=(ids>=0)&(ids<len(xyz))&np.isfinite(positions).all(axis=1)&np.isfinite(descriptors).all(axis=1)&(norms>1e-8)
            indices=np.flatnonzero(keep)
            if not np.allclose(positions[indices],xyz[ids[indices]],rtol=1e-5,atol=1e-4):
                raise ValueError('Inconsistent XYZ '+str(file))
            return ids[indices],descriptors[indices]/norms[indices,None]
    for n,(name,file) in enumerate(files):
        ids,_=read(file);counts.append(len(ids))
        st=file.stat();fingerprint.append((name,st.st_size,st.st_mtime_ns))
        if (n+1)%100==0:print('Bank audit',n+1,'/',len(files),flush=True)
    total=sum(counts)
    desc=np.lib.format.open_memmap(output/'bank_descriptors.npy',mode='w+',dtype=np.float32,shape=(total,128))
    bank_ids=np.lib.format.open_memmap(output/'bank_ids.npy',mode='w+',dtype=np.int32,shape=(total,))
    frames=np.lib.format.open_memmap(output/'bank_frames.npy',mode='w+',dtype=np.int32,shape=(total,))
    offset=0
    for n,((name,file),count) in enumerate(zip(files,counts)):
        ids,values=read(file)
        desc[offset:offset+count]=values;bank_ids[offset:offset+count]=ids;frames[offset:offset+count]=int(Path(name).stem)
        offset+=count
    desc.flush();bank_ids.flush();frames.flush()
    dump(output/'bank_metadata.json',dict(observations=total,landmarks=len(xyz),
        covered_landmarks=len(np.unique(bank_ids)),map_sha256=sha(map_path/'object_aliked_map.npz'),
        reference_stat_sha256=hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest(),
        build_ms=(time.perf_counter()-started)*1000,source='All valid saved ALIKED observations, normalized separately; no averaging of opposing views'))
    dump(output/'inputs.json',metadata)


def cpu_values(tensor):
    return tensor[0].detach().cpu().numpy()


def nearest_photo(torch,matcher,query,reference):
    from pose_landmark_matching import normalized_descriptors
    q=normalized_descriptors(cpu_values(query['descriptors']))
    r=normalized_descriptors(cpu_values(reference['descriptors']))
    bf=cv2.BFMatcher(cv2.NORM_L2)
    forward=bf.knnMatch(q,r,k=2)
    reverse=bf.match(r,q)
    reverse_q={m.queryIdx:m.trainIdx for m in reverse}
    pairs=[];scores=[]
    for neighbours in forward:
        if len(neighbours)<2:continue
        a,b=neighbours
        if a.distance<=.9 and a.distance<.85*b.distance and reverse_q.get(a.trainIdx)==a.queryIdx:
            pairs.append((a.queryIdx,a.trainIdx));scores.append(1.-.5*a.distance*a.distance)
    return cpu_values(query['keypoints']),cpu_values(reference['keypoints']),np.asarray(pairs,np.int64).reshape(-1,2),np.asarray(scores,np.float64)


def global_search(torch,query,descriptors,point_ids,frames,excluded_frame,landmarks,device):
    from pose_landmark_matching import normalized_descriptors
    started=time.perf_counter()
    q=normalized_descriptors(cpu_values(query['descriptors']))
    q_gpu=torch.from_numpy(q).to(device)
    valid=np.ones(len(descriptors),bool) if excluded_frame is None else np.abs(frames-excluded_frame)>5
    count=len(q);chunk=4096;k=64
    best=torch.full((count,k),-float('inf'),device=device)
    best_index=torch.full((count,k),-1,dtype=torch.int64,device=device)
    scratch=torch.zeros((chunk,128),device=device)
    scratch_mask=torch.zeros(chunk,dtype=torch.bool,device=device)
    # Stream a fixed-size CPU buffer. Slicing one large MPS tensor repeatedly
    # can retain substantial backend state; no full-bank GPU views are needed.
    cpu_chunk=np.zeros((chunk,128),np.float32)
    cpu_mask=np.zeros(chunk,bool)
    rev_scores=np.empty(len(descriptors),np.float32)
    rev_query=np.empty(len(descriptors),np.int64)
    with torch.inference_mode():
        for start in range(0,len(descriptors),chunk):
            n=min(chunk,len(descriptors)-start)
            cpu_chunk.fill(0);cpu_mask.fill(False)
            cpu_chunk[:n]=descriptors[start:start+n];cpu_mask[:n]=valid[start:start+n]
            scratch.copy_(torch.from_numpy(cpu_chunk).to(device))
            scratch_mask.copy_(torch.from_numpy(cpu_mask).to(device))
            similarity=q_gpu@scratch.T
            similarity=torch.where(scratch_mask.unsqueeze(0),similarity,-float('inf'))
            values,indices=torch.topk(similarity,k,dim=1,sorted=True)
            joined_values=torch.cat((best,values),dim=1)
            joined_indices=torch.cat((best_index,indices+start),dim=1)
            best,selection=torch.topk(joined_values,k,dim=1,sorted=True)
            best_index=joined_indices.gather(1,selection)
            maxima,queries=similarity.max(dim=0)
            rev_scores[start:start+n]=maxima[:n].cpu().numpy()
            rev_query[start:start+n]=queries[:n].cpu().numpy()
            if start//chunk%50==0:print('Global descriptors',start+n,'/',len(descriptors),flush=True)
        nearest=best_index.cpu().numpy();similarities=best.cpu().numpy()
    point_best_score=np.full(landmarks,-np.inf,np.float32)
    np.maximum.at(point_best_score,point_ids[valid],rev_scores[valid])
    tie=valid&(rev_scores==point_best_score[point_ids])
    point_best_query=np.full(landmarks,count,np.int32)
    np.minimum.at(point_best_query,point_ids[tie],rev_query[tie])
    elapsed=(time.perf_counter()-started)*1000
    del q_gpu,best,best_index,scratch,scratch_mask,similarity
    torch.mps.empty_cache()
    return nearest,similarities,point_best_query,elapsed


def global_correspondences(query,indices,similarities,reverse,bank_ids,xyz,offset,ratio):
    from aliked_lightglue_pnp import Transfer
    pixels=cpu_values(query['keypoints'])+np.asarray(offset)
    transfers=[];stats=dict(no_distinct_runner_up=0,ratio_pass=0,mutual_pass=0)
    for qi,(observations,scores) in enumerate(zip(indices,similarities)):
        finite=np.isfinite(scores)&(observations>=0)&(observations<len(bank_ids))
        obs=observations[finite];ss=scores[finite]
        if not len(obs):continue
        pid=int(bank_ids[obs[0]])
        distinct=np.flatnonzero(bank_ids[obs]!=pid)
        if not len(distinct):stats['no_distinct_runner_up']+=1;continue
        d0=float(np.sqrt(max(0.,2.-2.*float(ss[0]))))
        d1=float(np.sqrt(max(0.,2.-2.*float(ss[distinct[0]]))))
        if d0>.9 or d0>=ratio*d1:continue
        stats['ratio_pass']+=1
        if reverse[pid]!=qi:continue
        stats['mutual_pass']+=1
        transfers.append(Transfer(qi,int(obs[0]),pid,tuple(float(v) for v in xyz[pid]),
            tuple(float(v) for v in pixels[qi]),(0.,0.),(0.,0.),0.,float(ss[0])))
    transfers.sort(key=lambda t:t.match_score,reverse=True)
    seen=set();unique=[]
    for t in transfers:
        if t.point3d_id in seen:continue
        seen.add(t.point3d_id);unique.append(t)
    stats['unique_xyz']=len(unique)
    return unique,stats


def worker(output,query_index,only_mode):
    from aliked_lightglue_pnp import _extract,_image_tensor,solve_pose
    from browser_photo import decode_photo
    from glb_virtual_live import crop_live_object
    from offline_pnp_test import load_reference_images,load_camera_parameters,camera_matrix_and_distortion,quaternion_to_rotation
    from realtime_pnp import approximate_camera_matrix,DEFAULT_SPLAT_VIEWS
    from registered_aliked_live import RegisteredALIKEDLiveLocalizer,rotation_angle
    import registered_aliked_live as module
    from profile_host_memory import process_memory
    cv2.setNumThreads(4)
    metadata=json.loads((output/'inputs.json').read_text())
    item=metadata['queries'][query_index];path=Path(item['path']);map_path=Path(metadata['map_path'])
    e=RegisteredALIKEDLiveLocalizer(map_path,DEFAULT_SPLAT_VIEWS,device='mps',max_keypoints=2000,
        top_k=3,resize=1024,fundamental_threshold=3.,ransac_threshold=4.,max_reprojection_error=3.,search_policy='adaptive')
    e.query_resize=1600
    summary=json.loads((map_path/'summary.json').read_text())
    records=load_reference_images(Path(summary['input_files']['images']))
    source=item['kind']=='source';record=records[path.name] if source else None
    excluded_frame=int(path.stem) if source else None
    if source:e.registered={n:r for n,r in e.registered.items() if abs(int(Path(n).stem)-excluded_frame)>5}
    image=decode_photo(path.read_bytes())
    if source:
        cameras,params=load_camera_parameters(Path(summary['input_files']['cameras']))
        K,dist=camera_matrix_and_distortion(np,cameras[record.camera_id],params[record.camera_id])
    else:K=approximate_camera_matrix(image.shape[1],image.shape[0],72.);dist=np.zeros(5)
    # Warm extractor and one fixed reference shape, outside measured work.
    retrieval=e.retriever.rank_debug(image,top_k=3)
    cropped=crop_live_object(image,retrieval[3]['bbox'])
    if cropped is None:raise ValueError('No ROI: '+path.name)
    crop,(x0,y0,_,_)=cropped
    with e.torch.inference_mode():q=_extract(e.extractor,_image_tensor(e.torch,crop,e.device),1600)
    module._match_pair(e.torch,e.matcher,q,e._reference(sorted(e.registered)[0])[0])
    del q;e.reference_cache.clear();e.torch.mps.empty_cache()
    started=time.perf_counter();bbox,_=e.retriever.detector.detect(image);detector_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();retrieval=e.retriever.rank_debug(image,top_k=3);retrieval_ms=(time.perf_counter()-started)*1000
    if not np.allclose(bbox,retrieval[3]['bbox']):raise ValueError('ROI mismatch')
    cropped=crop_live_object(image,bbox);crop,(x0,y0,_,_)=cropped
    started=time.perf_counter()
    with e.torch.inference_mode():query=_extract(e.extractor,_image_tensor(e.torch,crop,e.device),1600)
    e.torch.mps.synchronize();features_ms=(time.perf_counter()-started)*1000
    e.torch.mps.empty_cache()
    center=np.asarray(e.manifest['center'])

    def truth(rvec,tvec):
        if rvec is None:return {}
        R=cv2.Rodrigues(np.asarray(rvec))[0];camera=-R.T@np.asarray(tvec).reshape(3)
        if source:
            true_R=quaternion_to_rotation(np,record.pose);true_center=-true_R.T@np.asarray(record.pose[4:])
            angle=rotation_angle(R,true_R);delta=float(np.linalg.norm(camera-true_center))
            return dict(rotation_error_deg=angle,center_error_units=delta,center_error_object_fraction=delta/e.scene_scale,
                        pose_correct=bool(angle<=5 and delta<=.05*e.scene_scale))
        az=float(np.degrees(np.arctan2(*(camera-center)[:2]))%360)
        frame=int(path.stem.split('_')[1]);start=90 if frame<=2591 else 0 if frame<=2594 else 270 if frame<=2597 else 180
        return dict(estimated_azimuth_deg=az,face_correct=bool(0<=(az-start)%360<=90))

    rows=[]
    def save(row):
        row.update(name=path.name,kind=item['kind'],query_index=query_index,
            preparation_ms=dict(detector=detector_ms,retrieval=retrieval_ms,features=features_ms),
            footprint_bytes=process_memory(os.getpid())['phys_footprint'])
        row.update(query_feature_sha256=hashlib.sha256(cpu_values(query['keypoints']).tobytes()+cpu_values(query['descriptors']).tobytes()).hexdigest(),
                   roi=list(bbox))
        rows.append(row);dump(output/(path.stem+'_'+only_mode+'_results.json'),rows)
        print(json.dumps({k:row.get(k) for k in ['name','mode','status','pipeline_component_sum_ms','inliers','face_correct','rotation_error_deg','footprint_bytes']}),flush=True)

    def photo_mode(mode):
        original=module._match_pair
        module._match_pair=nearest_photo if mode=='photo_nn' else original
        e.top_k=3;e.reference_cache.clear();attempts=[];extra_retrieval=0.
        started=time.perf_counter()
        try:
            result,_=e._search_attempts(image,K,dist,None,dict(retrieval=retrieval,query=query))
            attempts.append(copy.deepcopy(e.last_diagnostics))
            if result.status!='PASS':
                e.top_k=10
                ret_started=time.perf_counter();expanded=e.retriever.rank_debug(image,top_k=10)
                extra_retrieval=(time.perf_counter()-ret_started)*1000
                # Same current retry behavior: extraction repeated on expansion.
                result,_=e._search_attempts(image,K,dist,None,dict(retrieval=expanded))
                attempts.append(copy.deepcopy(e.last_diagnostics))
            e.torch.mps.synchronize();elapsed=(time.perf_counter()-started)*1000
        finally:module._match_pair=original;e.top_k=3
        save(dict(mode=mode,status=result.status,inliers=result.inliers,result=result.as_dict(),
            matching_and_retry_wall_ms=elapsed,pipeline_component_sum_ms=retrieval_ms+features_ms+elapsed,
            extra_retrieval_ms=extra_retrieval,attempts=attempts,**truth(result.rvec,result.tvec)))
        e.reference_cache.clear();e.torch.mps.empty_cache()

    def global_mode():
        descriptors=np.load(output/'bank_descriptors.npy',mmap_mode='c')
        ids=np.load(output/'bank_ids.npy',mmap_mode='c');frames=np.load(output/'bank_frames.npy',mmap_mode='c')
        nearest,similarity,reverse,search_ms=global_search(e.torch,query,descriptors,ids,frames,excluded_frame,len(e.map_data.xyz),e.device)
        np.savez_compressed(output/(path.stem+'_global_neighbours.npz'),indices=nearest,similarities=similarity)
        for ratio in [.85,.95]:
            started=time.perf_counter()
            transfers,stats=global_correspondences(query,nearest,similarity,reverse,ids,e.map_data.xyz,(x0,y0),ratio)
            filtering_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter();pose=solve_pose(transfers,K,dist,4.,10000);pnp_ms=(time.perf_counter()-started)*1000
            qc_started=time.perf_counter()
            if pose is None:quality=dict(passed=False,reasons=['no_pnp']);rv=tv=None;inliers=0
            else:
                rv=pose.rvec;tv=pose.tvec;inliers=len(pose.inlier_indices)
                quality=e.validate_pose(np.asarray([t.xyz for t in transfers]),np.asarray([t.query_uv for t in transfers]),rv,tv,pose.inlier_indices,image.shape,K,dist)
            qc_ms=(time.perf_counter()-qc_started)*1000
            save(dict(mode='global_nn' if ratio==.85 else 'global_nn_relaxed',ratio=ratio,
                status='PASS' if quality['passed'] else 'FAIL',inliers=inliers,correspondences=len(transfers),quality=quality,
                descriptor_search_ms=search_ms,filtering_ms=filtering_ms,pnp_ms=pnp_ms,qc_ms=qc_ms,
                pipeline_component_sum_ms=detector_ms+features_ms+search_ms+filtering_ms+pnp_ms+qc_ms,
                matching_stats=stats,rvec=None if rv is None else np.asarray(rv).ravel().tolist(),
                tvec=None if tv is None else np.asarray(tv).ravel().tolist(),**truth(rv,tv)))

    for mode in [only_mode]:
        if mode=='global':global_mode()
        else:photo_mode(mode)
    dump(output/(path.stem+'_'+only_mode+'_complete.json'),dict(rows=len(rows),name=path.name))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=int)
    p.add_argument('--mode',choices=['lightglue','photo_nn','global'])
    p.add_argument('--limit',type=int)
    args=p.parse_args();output=args.output.resolve()
    if args.worker is not None:
        if args.mode is None:raise ValueError('Worker mode required')
        worker(output,args.worker,args.mode);return
    previous=DATA_DIR/'output_coarse_comparison_20261003/metadata_shortlist.json'
    metadata=json.loads(previous.read_text())
    metadata.update(matcher_comparison=dict(photo_nn='Exact BF L2, mutual, ratio .85, distance .9; same GS shortlist and image geometry as current.',
        global_nn='Exact float32 GPU dot products, 4096-observation chunks, global top64; ratio .85 across distinct XYZ IDs, distance .9, global reciprocal nearest query per XYZ.',
        global_nn_relaxed='Same exact search, ratio .95 sensitivity check; no per-query threshold tuning.',
        query_control='Same ALIKED extraction and ROI for initial searches; current GS routes retry Top10 and reextract on failure.',
        qc_control='Same numerical PnP/QC. Global route has no coarse-rotation prior or per-reference fundamental test.',
        timing='Component sums, not independently timed full end-to-end pipelines. Model startup, bank build/load and photo decoding excluded; query descriptor transfer to GPU included.',
        lifecycle='One fresh process per query AND mode; CPU streaming into fixed-shape global chunks.'),
        comparison_code_sha256={f:sha(ROOT/f) for f in ['compare_descriptor_matching.py','registered_aliked_live.py','aliked_lightglue_pnp.py','pose_quality.py']})
    if not output.exists():prepare(output,metadata)
    else:
        old=json.loads((output/'inputs.json').read_text())
        if old['comparison_code_sha256']!=metadata['comparison_code_sha256']:raise ValueError('Code changed; use a fresh output')
    for index,item in enumerate(metadata['queries'][:args.limit]):
        name=Path(item['path']).stem
        modes=['lightglue','photo_nn','global'] if index%2==0 else ['global','photo_nn','lightglue']
        for mode in modes:
            if (output/(name+'_'+mode+'_complete.json')).exists():continue
            print('QUERY',index,name,mode,flush=True)
            log_path=output/(name+'_'+mode+'_console.log')
            with log_path.open('w') as log:
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker',str(index),'--mode',mode],stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:raise RuntimeError('Worker failed; inspect '+str(log_path))
            print('COMPLETE',name,mode,flush=True)


if __name__=='__main__':main()
