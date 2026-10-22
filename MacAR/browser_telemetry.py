"""Validate bounded, untrusted browser timing reports before JSONL enqueue."""
import math
import re


def _number(value, *, nullable=False, integer=False):
    if value is None and nullable:return
    if isinstance(value,bool) or not isinstance(value,(int,float)) or abs(value)>1e15 or not math.isfinite(value):
        raise ValueError('Invalid browser telemetry number')
    if integer and (value<0 or int(value)!=value):raise ValueError('Invalid telemetry counter')


def validate_browser_batch(value):
    if set(value)!={'client_id','session_id','generation','batch_id','window','frames','counters','environment'}:
        raise ValueError('Invalid browser telemetry fields')
    for name in ('client_id','session_id'):
        if not isinstance(value[name],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value[name]):
            raise ValueError('Invalid telemetry identity')
    for name in ('generation','batch_id'):_number(value[name],integer=True)
    frames=value['frames']
    if not isinstance(frames,list) or len(frames)>64:raise ValueError('Too many telemetry frames')
    window=value['window']
    allowed={'start_ms','end_ms','received_frames','new_pose_frames','painted_frames','gs_frames',
             'receive_fps','pose_fps','paint_fps','gs_fps','gs_active','gs_ready','pending_frames',
             'visibility','gs_mode','gs_layout','mpr_visible'}
    if not isinstance(window,dict) or not set(window)<=allowed:raise ValueError('Invalid telemetry window')
    required_window={'start_ms','end_ms','received_frames','new_pose_frames','painted_frames','gs_frames',
                     'receive_fps','pose_fps','paint_fps','gs_fps'}
    if not required_window<=set(window):raise ValueError('Incomplete telemetry window')
    for key,number in window.items():
        if key in ('gs_active','gs_ready','mpr_visible'):
            if not isinstance(number,bool):raise ValueError('Invalid GS state')
        elif key in ('visibility','gs_mode','gs_layout'):
            choices={'visibility':('visible','hidden','unknown'),'gs_mode':('matched','orbit'),
                     'gs_layout':('side','overlay','off')}
            if number not in choices[key]:raise ValueError('Invalid browser display state')
        else:
            _number(number,integer=key.endswith('_frames'))
            if number<0:raise ValueError('Negative telemetry rate/time')
    if window.get('end_ms',-1)<window.get('start_ms',0):raise ValueError('Invalid window times')
    counters=value['counters']
    if not isinstance(counters,dict) or not set(counters)<={'dropped_frames','http_failures','transport_errors','clock_failures','previous_session_unsent_frames'}:
        raise ValueError('Invalid browser counters')
    for number in counters.values():_number(number,integer=True)
    environment=value['environment']
    if not isinstance(environment,dict) or not set(environment)<={'user_agent','width','height','pixel_ratio','hardware_concurrency'}:
        raise ValueError('Invalid browser environment')
    for key,item in environment.items():
        if key=='user_agent':
            if not isinstance(item,str) or len(item)>400:raise ValueError('Invalid user agent')
        else:_number(item,nullable=True)
    floats={'received_ms','decoded_ms','drawn_ms','paint_ms','capture_monotonic','sent_at','mpr_age_ms',
            'python_receive_to_browser_receive_ms','python_receive_to_paint_opportunity_ms',
            'python_publish_to_browser_receive_ms','localize_ms','mpr_ms','image_coverage',
            'python_tracking_fps','python_mpr_fps','python_packet_fps'}
    integers={'sequence','source_id','pose_source_id','mpr_source_id','skipped_server_packets','width','height','inliers','matches'}
    flags={'frozen','demo','has_pose','clock_inconsistent','paint_clock_valid'}
    enums={'status':{'PASS','FAIL','SEARCH'},'input_source':{'camera','photo'},
           'visibility':{'visible','hidden'},'paint_reason':{'hidden','superseded','capacity','final_flush','paint_opportunity'},
            'layer_mode':{'image','projective',None}}
    for frame in frames:
        if not isinstance(frame,dict) or not set(frame)<=floats|integers|flags|set(enums)|{'clock','layer_id','sensor_exposure_time','physical_presentation_time','mpr_status'}:
            raise ValueError('Invalid frame fields')
        if not {'sequence','source_id','status','input_source','visibility','received_ms','decoded_ms','drawn_ms','paint_ms','paint_reason'}<=set(frame):
            raise ValueError('Incomplete browser frame')
        for key,item in frame.items():
            if key in floats:_number(item,nullable=True)
            elif key in integers:_number(item,nullable=True,integer=True)
            elif key in flags:
                if not isinstance(item,bool):raise ValueError('Invalid telemetry flag')
            elif key in enums:
                if not isinstance(item,(str,type(None))) or item not in enums[key]:raise ValueError('Invalid telemetry state')
            elif key=='layer_id':
                if item is not None and (not isinstance(item,str) or len(item)>120):raise ValueError('Invalid layer ID')
            elif key=='mpr_status':
                if item is not None and (not isinstance(item,str) or len(item)>64):raise ValueError('Invalid MPR state')
            elif key in ('sensor_exposure_time','physical_presentation_time'):
                if item is not None:raise ValueError('Physical timing is not available')
            elif key=='clock' and item is not None:
                if not isinstance(item,dict) or not set(item)<={'valid','reason','offset_ms','round_trip_ms','uncertainty_ms',
                    'at_ms','age_ms','client_send_ms','server_receive_ms','server_send_ms','client_receive_ms'}:
                    raise ValueError('Invalid clock fields')
                for field,number in item.items():
                    if field=='valid':
                        if not isinstance(number,bool):raise ValueError('Invalid clock flag')
                    elif field=='reason':
                        if number not in ('ready','missing','stale'):raise ValueError('Invalid clock reason')
                    else:_number(number)
        times=[frame[k] for k in ('received_ms','decoded_ms','drawn_ms','paint_ms') if frame.get(k) is not None]
        if times!=sorted(times):raise ValueError('Invalid browser frame time order')
    return value
