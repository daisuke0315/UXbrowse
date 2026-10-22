"""Bounded sequential live-resolution still-image stress check; no camera UI."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from browser_live import DEFAULTS, runtime_arguments, finite_json
from browser_photo import decode_photo
from realtime_pnp import build_localizer, camera_parameters
from live_search_resources import run_live_search


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--photo-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cleanup',action='store_true')
    parser.add_argument('--cpu-extraction',action='store_true')
    parser.add_argument('--manual-attention',action='store_true')
    parser.add_argument('--rounds',type=int,default=2)
    options=parser.parse_args()
    if options.output.exists():raise ValueError('Never overwrite a saved run')
    settings=dict(DEFAULTS,mpr=False)
    args,_,_=runtime_arguments(settings)
    wrapper=build_localizer(args,temporal_registered=True)
    engine=wrapper.global_localizer
    if options.manual_attention:
        for module in engine.matcher.modules():
            if hasattr(module,'has_sdp'):module.has_sdp=False
    if options.cpu_extraction:
        import registered_aliked_live
        original=registered_aliked_live._extract
        engine.extractor.to('cpu')
        def extract_cpu(extractor,tensor,resize):
            features=original(extractor,tensor.cpu(),resize)
            return {key:value.to(engine.device) for key,value in features.items()}
        registered_aliked_live._extract=extract_cpu
    engine.query_resize=1024
    assert engine.device.type=='mps'
    paths=sorted(options.photo_root.rglob('IMG_*.jpeg'))
    events=[]
    def sink(event,**fields):
        events.append(dict(event=event,**fields))
        if event=='reference_candidate' and torch.mps.driver_allocated_memory()>8_000_000_000:
            raise RuntimeError('RESOURCE_STOP_DRIVER_8GB')
    engine.diagnostic_sink=sink
    rows=[]
    config=dict(settings=settings,cleanup=options.cleanup,cpu_extraction=options.cpu_extraction,
                manual_attention=options.manual_attention,rounds=options.rounds,
                shape_policy='whole image resized to width 1280; native live UV/K',
                note='Independent still images, not physical tracking or a photo benchmark',
                sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                         ('live_search_resources.py','glb_virtual_live.py','background_refresh.py',
                          'registered_aliked_live.py','benchmark_live_resources.py')})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for repeat in range(options.rounds):
                for path in paths:
                    image=decode_photo(path.read_bytes())
                    image=cv2.resize(image,(1280,round(image.shape[0]*1280/image.shape[1])),interpolation=cv2.INTER_AREA)
                    K,dist,_=camera_parameters(args,(image.shape[1],image.shape[0]))
                    events.clear();cv2.setRNGSeed(0);torch.manual_seed(0)
                    torch.mps.synchronize();started=time.perf_counter()
                    result,error=None,None
                    try:
                        if options.cleanup:
                            result,_=executor.submit(run_live_search,engine,engine.localize,image,K,dist).result()
                        else:
                            result,_=executor.submit(engine.localize,image,K,dist).result()
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                    torch.mps.synchronize()
                    row=dict(photo=path.name,repeat=repeat,error=error,
                        wall_ms=(time.perf_counter()-started)*1000,
                        result=result.as_dict() if result else None,
                        current_bytes=torch.mps.current_allocated_memory(),
                        driver_bytes=torch.mps.driver_allocated_memory(),events=list(events))
                    rows.append(row)
                    options.output.parent.mkdir(parents=True,exist_ok=True)
                    options.output.write_text(json.dumps(finite_json(dict(config=config,rows=rows)),
                        ensure_ascii=False,allow_nan=False,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k not in ('events','result')},ensure_ascii=False),
                          result.status if result else 'INCOMPLETE',flush=True)
                    if error or row['driver_bytes']>8_000_000_000:return
    finally:wrapper.close()


if __name__=='__main__':main()
