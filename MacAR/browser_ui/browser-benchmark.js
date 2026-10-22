// Application latency only: neither sensor exposure nor physical display timing.
export class ClockModel {
    constructor(){this.samples=[];}
    add(c0,s1,s2,c3){
        if(![c0,s1,s2,c3].every(Number.isFinite)||c3<c0||s2<s1||c3-c0<s2-s1)throw Error('Invalid clock sample');
        this.samples.push({offset_ms:((s1-c0)+(s2-c3))/2,round_trip_ms:c3-c0,
            uncertainty_ms:((c3-c0)-(s2-s1))/2,at_ms:c3,
            client_send_ms:c0,server_receive_ms:s1,server_send_ms:s2,client_receive_ms:c3});
        this.samples=this.samples.slice(-5);
    }
    snapshot(now){
        const fresh=this.samples.filter(s=>now>=s.at_ms&&now-s.at_ms<=30000);
        if(!fresh.length)return {valid:false,reason:this.samples.length?'stale':'missing'};
        const best=fresh.reduce((a,b)=>a.uncertainty_ms<=b.uncertainty_ms?a:b);
        return {...best,valid:true,reason:'ready',age_ms:now-best.at_ms};
    }
}

export class BrowserBenchmark {
    constructor({clientId,now=()=>performance.now(),raf=f=>requestAnimationFrame(f),send,
                 clockRequest,gsState=()=>({render_count:0}),environment={}}){
        this.clientId=clientId;this.now=now;this.raf=raf;this.send=send;
        this.clockRequest=clockRequest;this.gsState=gsState;this.environment=environment;
        this.clock=new ClockModel();this.session=null;this.endedKey=null;this.inflight=null;
        this.queue=[];this.pending=new Set();this.drawToken=0;this.batch=0;
        this.rates={receive_fps:0,pose_fps:0,paint_fps:0,gs_fps:0};
        this.resetCounters();this.lastError=null;this.syncing=false;
    }
    resetCounters(){this.counters={dropped_frames:0,http_failures:0,transport_errors:0,clock_failures:0};}
    resetWindow(){
        this.windowStart=this.now();this.counts={received_frames:0,new_pose_frames:0,painted_frames:0};
        this.gsStart=this.gsState().render_count||0;
    }
    async syncClock(){
        if(this.syncing||!this.clockRequest)return;
        this.syncing=true;
        try{const c0=this.now(),value=await this.clockRequest(),c3=this.now();
            this.clock.add(c0,value.server_receive_ms,value.server_send_ms,c3);
        }catch(_){this.counters.clock_failures++;}finally{this.syncing=false;}
    }
    transportError(){this.counters.transport_errors++;}
    drawn(meta,received,decoded,drawn,visible=true){
        const key=`${meta.generation}:${meta.diagnostic_log?.session_id||''}`;
        if(key===this.endedKey)return;
        if(this.session?.key!==key){
            const discarded=this.queue.length+this.pending.size;
            for(const r of this.pending)r.done=true;
            this.pending.clear();this.queue=[];this.resetCounters();
            if(discarded)this.counters.previous_session_unsent_frames=discarded;
            this.session={key,generation:meta.generation,id:meta.diagnostic_log?.session_id||null};
            this.batch=0;this.lastPose=null;this.lastSequence=null;this.resetWindow();
        }
        this.counts.received_frames++;
        const hasPose=Boolean(meta.pose)&&meta.pose_source_id===meta.source_id;
        if(hasPose&&meta.pose_source_id!==this.lastPose){this.counts.new_pose_frames++;this.lastPose=meta.pose_source_id;}
        const skipped=this.lastSequence===null?0:Math.max(0,meta.sequence-this.lastSequence-1);
        this.lastSequence=meta.sequence;
        const record={sequence:meta.sequence,source_id:meta.source_id,pose_source_id:meta.pose_source_id??null,
            status:meta.status,input_source:meta.input_source,frozen:Boolean(meta.frozen),demo:Boolean(meta.demo),
            visibility:visible?'visible':'hidden',has_pose:hasPose,width:meta.width??null,height:meta.height??null,received_ms:received,decoded_ms:decoded,
            drawn_ms:drawn,paint_ms:null,paint_reason:null,capture_monotonic:meta.capture_monotonic??null,
            sent_at:meta.sent_at??null,mpr_source_id:meta.mpr_source_id??null,mpr_age_ms:meta.mpr_age_ms??null,
            layer_mode:meta.layer_mode??null,layer_id:meta.layer_id??null,skipped_server_packets:skipped,
            localize_ms:meta.localize_ms??null,mpr_ms:meta.mpr_ms??null,mpr_status:meta.mpr_status??null,
            inliers:meta.result?.inliers??null,matches:meta.result?.matches??null,
            image_coverage:meta.correspondences?.image_coverage??null,
            python_tracking_fps:meta.pipeline_fps?.tracking??null,
            python_mpr_fps:meta.pipeline_fps?.mpr??null,python_packet_fps:meta.pipeline_fps?.packets??null,
            clock:this.clock.snapshot(received),sensor_exposure_time:null,physical_presentation_time:null};
        const token=++this.drawToken,epoch=this.session;
        const pending={done:false,finish:(paint,reason)=>{
            if(pending.done)return;pending.done=true;this.pending.delete(pending);
            if(this.session!==epoch)return;
            record.paint_ms=paint;record.paint_reason=reason;
            if(reason==='hidden')record.visibility='hidden';
            if(paint!==null)this.counts.painted_frames++;
            const eligible=record.clock.valid&&record.visibility==='visible';
            const latency=t=>eligible&&t!==null&&record.capture_monotonic!==null
                ?t+record.clock.offset_ms-record.capture_monotonic*1000:null;
            record.python_receive_to_browser_receive_ms=latency(received);
            record.paint_clock_valid=eligible&&paint!==null&&paint-record.clock.at_ms<=30000;
            record.python_receive_to_paint_opportunity_ms=record.paint_clock_valid?latency(paint):null;
            record.python_publish_to_browser_receive_ms=eligible&&record.sent_at!==null
                ?received+record.clock.offset_ms-record.sent_at*1000:null;
            record.clock_inconsistent=record.python_receive_to_browser_receive_ms!==null&&
                record.python_receive_to_browser_receive_ms < -record.clock.uncertainty_ms;
            if(this.queue.length>=64){this.queue.shift();this.counters.dropped_frames++;}
            this.queue.push(record);
        }};
        this.pending.add(pending);
        if(this.pending.size>64)this.pending.values().next().value.finish(null,'capacity');
        if(!visible){pending.finish(null,'hidden');return;}
        this.raf(()=>{
            if(pending.done)return;
            if(this.drawToken!==token){pending.finish(null,'superseded');return;}
            this.raf(()=>pending.finish(this.now(),'paint_opportunity'));
        });
    }
    hidden(){for(const r of [...this.pending])r.finish(null,'hidden');}
    async flush({final=false,keepalive=false}={}){
        if(final)for(const r of [...this.pending])r.finish(null,'final_flush');
        if(this.inflight){if(final){await this.inflight;return this.flush({final,keepalive});}return;}
        if(!this.session)return;
        const end=this.now(),duration=end-this.windowStart;
        if(duration<=0&&!this.queue.length)return;
        const gs=this.gsState(),count=Math.max(0,(gs.render_count||0)-this.gsStart);
        const window={start_ms:this.windowStart,end_ms:end,...this.counts,gs_frames:count,
            receive_fps:duration>0?this.counts.received_frames*1000/duration:0,
            pose_fps:duration>0?this.counts.new_pose_frames*1000/duration:0,
            paint_fps:duration>0?this.counts.painted_frames*1000/duration:0,
            gs_fps:duration>0?count*1000/duration:0,
            gs_active:Boolean(gs.active),gs_ready:Boolean(gs.ready),pending_frames:this.pending.size,
            visibility:gs.visibility||'unknown',gs_mode:gs.gs_mode||'matched',
            gs_layout:gs.gs_layout||'off',mpr_visible:Boolean(gs.mpr_visible)};
        this.rates=window;this.resetWindow();
        if(!this.session.id){this.queue=[];return;}
        const frames=this.queue.splice(0,32),counters=this.counters;
        const payload={client_id:this.clientId,session_id:this.session.id,generation:this.session.generation,
            batch_id:++this.batch,window,frames,counters:{...counters},environment:this.environment};
        this.inflight=(async()=>{
            try{await this.send(payload,{keepalive});this.lastError=null;}
            catch(e){counters.http_failures++;counters.dropped_frames+=frames.length;this.lastError=String(e.message||e);}
        })();
        try{await this.inflight;}finally{this.inflight=null;}
        if(final&&this.queue.length)return this.flush({final,keepalive});
    }
    async endSession({keepalive=false}={}){
        this.endedKey=this.session?.key;
        await this.flush({final:true,keepalive});
    }
}
