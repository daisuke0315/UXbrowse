"""Offline real-photo rotation replay with an intentional 20-second match delay.

Uses IMG_2593 and the pinned comparison map; never opens a physical camera.
The synthetic homography provides relative rotation truth, not CT accuracy.
"""

from macar_paths import DATA_DIR
import os
os.environ['PYTHONPYCACHEPREFIX']='/private/tmp/macar-concurrent-pycache'
import sys,time,json,threading,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from browser_photo import decode_photo
from registered_aliked_live import RegisteredALIKEDLiveLocalizer,rotation_angle
import registered_aliked_live as module
from glb_virtual_live import AsyncTemporalGLBLocalizer
from realtime_pnp import approximate_camera_matrix,DEFAULT_SPLAT_VIEWS
cv2.setNumThreads(4)
meta=json.loads((DATA_DIR/'output_descriptor_comparison_20261003_isolated/inputs.json').read_text())
photo=Path(meta['queries'][4]['path'])
raw=decode_photo(photo.read_bytes())
image=cv2.resize(raw,(960,round(raw.shape[0]*960/raw.shape[1])))
K=approximate_camera_matrix(image.shape[1],image.shape[0],72.)
e=RegisteredALIKEDLiveLocalizer(Path(meta['map_path']),DEFAULT_SPLAT_VIEWS,
    device='mps',max_keypoints=2000,top_k=3,resize=1024,
    fundamental_threshold=3.,ransac_threshold=4.,max_reprojection_error=3.,search_policy='adaptive')
e.query_resize=1600
loc=AsyncTemporalGLBLocalizer(e,reseed_frames=900,min_track_points=8,
    flow_window=31,flow_levels=3,forward_backward_error=1.5,tracking_iterations=150,
    max_seed_age=10.,replenish_mode='background')
original=module._match_pair
entered=threading.Event();release=threading.Event()
def delayed(*args):
    if not entered.is_set():
        entered.set()
        if not release.wait(35.):raise TimeoutError('Benchmark gate')
    return original(*args)
module._match_pair=delayed
loc.localize(image,K,None)
started=time.monotonic();motion_start=None;last_print=-10.;angle=0.;records=[];result=None
try:
    while time.monotonic()-started < 50.:
        tick=time.monotonic();elapsed=tick-started
        if entered.is_set() and motion_start is None:motion_start=tick
        motion_age=0. if motion_start is None else tick-motion_start
        angle=8.*min(1.,motion_age/20.)
        rotation=cv2.Rodrigues(np.array([0.,np.deg2rad(angle),0.]))[0]
        H=K@rotation@np.linalg.inv(K)
        live=cv2.warpPerspective(image,H,(image.shape[1],image.shape[0]))
        if motion_start is not None and motion_age>=20.:release.set()
        result,_=loc.localize(live,K,None)
        snapshot=loc.query_tracks.snapshot()
        records.append(dict(elapsed_s=elapsed,angle_deg=angle,query=snapshot,status=result.status,
            frame_ms=(time.monotonic()-tick)*1000))
        if elapsed-last_print>=3.:
            print(json.dumps(records[-1]),flush=True);last_print=elapsed
        if loc.last_search_result is not None and loc.future is None:
            if result.status!='PASS':raise RuntimeError('Current frame failed '+result.detail)
            break
        time.sleep(max(0.,1./15.-(time.monotonic()-tick)))
    else:raise TimeoutError('Replay search timed out')
    source=e.last_tracking_seed;current=loc.tracker.seed
    if source is None or current is None:raise RuntimeError('No accepted source/current seed')
    expected_R=rotation@cv2.Rodrigues(source.rvec)[0]
    expected_t=rotation@source.tvec.reshape(3)
    current_R=cv2.Rodrigues(current.rvec)[0]
    rotation_delta=rotation_angle(current_R,expected_R)
    source_center=-cv2.Rodrigues(source.rvec)[0].T@source.tvec.reshape(3)
    current_center=-current_R.T@current.tvec.reshape(3)
    summary=dict(code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['concurrent_query_tracking.py','registered_aliked_live.py','glb_virtual_live.py']},
        photo=str(photo),photo_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
        map_path=meta['map_path'],views_path=str(DEFAULT_SPLAT_VIEWS),
        views_manifest_sha256=hashlib.sha256((DEFAULT_SPLAT_VIEWS/'manifest.json').read_bytes()).hexdigest(),
        input_size=[image.shape[1],image.shape[0]],
        simulation='Real photo pure camera rotation H=K R K^-1; no physical camera ground truth',
        delay_s=20.,source_status=loc.last_search_result.status,current_status=result.status,
        search_age_ms=loc.last_search_age_ms,rotation_deg=angle,
        rotation_vs_expected_deg=rotation_delta,
        center_vs_source_units=float(np.linalg.norm(current_center-source_center)),
        source_inliers=len(source.image_points),current_inliers=len(current.image_points),
        transport=loc.last_search_diagnostics.get('seed_transport'),
        foreground_median_ms=float(np.median([r['frame_ms'] for r in records[:-1]])),
        foreground_p95_ms=float(np.percentile([r['frame_ms'] for r in records[:-1]],95)),
        records=records)
    assert summary['search_age_ms']>20000 and summary['transport']['method']=='continuous_query_lk'
    assert summary['transport']['reason']=='current_pnp_verified' and rotation_delta<1.
    out=DATA_DIR/'output_concurrent_tracking_20261003';out.mkdir(exist_ok=True)
    (out/'real_photo_rotation_replay.json').write_text(json.dumps(summary,indent=2))
    cv2.imwrite(str(out/'current.jpg'),live)
    print('REPLAY_RESULT',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
finally:
    release.set();module._match_pair=original;loc.close()
