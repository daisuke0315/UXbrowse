// Leading + trailing throttle: continuous dragging sends continuously, one
// request at a time; unsent intermediate values are replaced, not queued.
export class LatestControlSender {
    constructor(send, interval=50, onError=()=>{}) {
        this.send=send;this.interval=interval;this.onError=onError;
        this.pending=null;this.busy=false;this.timer=null;this.last=-Infinity;
    }
    push(value) {this.pending=value;this.schedule();}
    schedule() {
        if(this.busy||this.timer!==null||this.pending===null)return;
        this.timer=setTimeout(()=>this.flush(),Math.max(0,this.interval-(performance.now()-this.last)));
    }
    async flush() {
        this.timer=null;if(this.pending===null)return;
        const value=this.pending;this.pending=null;this.busy=true;this.last=performance.now();
        try {await this.send(value);} catch(error) {this.onError(error);}
        finally {this.busy=false;this.schedule();}
    }
}

export const depthToPercent=(depth,span)=>Math.max(-100,Math.min(150,100*depth/span));
export const percentToDepth=(percent,span)=>span*percent/100;

// Display only: no capture, matching, pose or measurement state is changed.
// The foreground cutaway belongs to the CT overlay, not the independent GS
// or anatomy display. Respect the separate all-overlays utility as well.
export function applyMPRVisibility($, allHidden=false) {
    const visible=$('showMPR').checked&&!allHidden;
    for(const id of ['mprCanvas','sliceMaskCanvas'])$(id).style.visibility=visible?'visible':'hidden';
    return visible;
}

export function installSliceControls(post,onError,onMaskChange=()=>{}) {
    const $=id=>document.getElementById(id),slider=$('slice');
    let span=0,unit='mm',generation=null,requested=0,nextId=0,dragging=false,settle,pendingMask=null;
    const sender=new LatestControlSender(data=>post('/api/control',data),50,onError);
    const opacitySender=new LatestControlSender(data=>post('/api/control',data),50,onError);
    function describe() {
        $('sliceValue').value=`${Number(slider.value).toFixed(1)} ${unit}`;
        slider.setAttribute('aria-valuetext',`CT手前から${slider.value}${unit}、右で奥へ`);
    }
    function send(interacting) {
        if(!span||generation===null)return;
        requested=++nextId;
        sender.push({action:'slice',value:depthToPercent(Number(slider.value),span),
                     interacting,request_id:requested,generation});
        describe();$('sliceStatus').textContent='断面を更新中…（表示中の画像は確定した断面）';
    }
    const finish=()=>{clearTimeout(settle);dragging=false;send(false);};
    slider.oninput=()=>{dragging=true;send(true);clearTimeout(settle);settle=setTimeout(finish,180);};
    slider.onchange=finish;
    slider.onpointercancel=finish;
    for(const button of document.querySelectorAll('[data-slice-step], [data-slice-percent]')) {
        button.onclick=()=>{
            slider.value=button.dataset.slicePercent!==undefined
                ?percentToDepth(Number(button.dataset.slicePercent),span)
                :Number(slider.value)+Number(button.dataset.sliceStep);
            finish();
        };
    }
    $('sliceExtended').onchange=()=>{range();finish();};
    $('cameraLinkedSlice').onchange=()=>post('/api/control',{
        action:'camera_linked_slice',value:$('cameraLinkedSlice').checked,generation
    }).catch(onError);
    const maskVisibility=()=>{
        $('sliceMaskCanvas').style.display=$('sliceMask').checked&&Number($('opacity').value)>0?'':'none';
        $('sliceMaskCanvas').style.opacity=Number($('maskStrength').value)/100;
        $('maskStrengthValue').value=`${$('maskStrength').value}%`;
    };
    $('maskStrength').oninput=maskVisibility;
    $('sliceMask').onchange=()=>{
        const value=$('sliceMask').checked;pendingMask=value;maskVisibility();onMaskChange();
        post('/api/control',{action:'slice_mask',value,generation}).catch(error=>{pendingMask=null;onError(error);});
    };
    function range() {
        slider.min=$('sliceExtended').checked?-span:0;
        slider.max=$('sliceExtended').checked?span*1.5:span;
        $('sliceNear').textContent=`← 手前（0 ${unit}）`;
        $('sliceFar').textContent=`奥（${span.toFixed(1)} ${unit}）→`;
    }
    function opacity() {
        $('opacityValue').value=`${$('opacity').value}%`;
        $('mprCanvas').style.opacity=Number($('opacity').value)/100;
        maskVisibility();
    }
    $('opacity').oninput=()=>{
        opacity();opacitySender.push({action:'opacity',value:Number($('opacity').value),generation});
    };
    opacity();
    return {
        sync(meta) {
            if(meta.generation!==generation) {
                generation=meta.generation;requested=0;dragging=false;clearTimeout(settle);
                sender.pending=opacitySender.pending=null;
                $('opacity').value=meta.opacity??82;opacity();
                pendingMask=null;
            }
            if(pendingMask===meta.slice_mask_enabled)pendingMask=null;
            $('sliceMask').checked=pendingMask??meta.slice_mask_enabled??true;
            $('sliceMask').disabled=!meta.slice_mask_available;
            maskVisibility();
            const position=meta.slice_position,ready=Boolean(position&&position.span>0&&meta.mpr_loaded);
            $('cameraLinkedSlice').checked=meta.camera_linked_slice??true;
            $('cameraLinkedSlice').disabled=!ready;
            slider.disabled=!ready;$('sliceExtended').disabled=!ready;
            for(const b of document.querySelectorAll('[data-slice-step], [data-slice-percent]'))b.disabled=!ready;
            if(!ready){$('sliceStatus').textContent='姿勢確定後にCTの距離を表示します。';return;}
            const pending=(meta.slice_request_id??0)<requested;
            if(!dragging&&!pending) {
                span=position.span;unit=position.unit;
                if(position.outside)$('sliceExtended').checked=true;
                range();slider.value=position.offset;describe();
            }
            const where=position.outside?'CTデータ範囲外（通常は断面なし）':meta.mpr_status==='VISIBLE'?'CT範囲内':'この位置に表示できる断面なし';
            $('sliceStatus').textContent=`${meta.camera_linked_slice?'カメラ連動 · ':''}表示中：CT手前から ${position.offset.toFixed(1)} ${position.unit} · ${where}${meta.camera_linked_slice?` · カメラ前方 ${position.camera_depth.toFixed(1)} ${position.unit}`:''}${pending?' · 操作位置へ更新中…':''}`;
        }
    };
}
