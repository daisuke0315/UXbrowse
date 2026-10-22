// Frozen LIVE and CT images, with optional CT compositing for visual comparison.
// Every cursor is stored
// in original image pixels; pan/zoom never changes a selected coordinate.
export class ImageViewTransform {
    constructor(){this.center=[0,0];this.zoom=1;this.imageWidth=1;this.imageHeight=1;this.width=1;this.height=1;}
    reset(w,h){this.imageWidth=w;this.imageHeight=h;this.center=[w/2,h/2];this.zoom=1;}
    get scale(){return Math.min(this.width/this.imageWidth,this.height/this.imageHeight)*this.zoom;}
    imageAt(x,y){return [this.center[0]+(x-this.width/2)/this.scale,this.center[1]+(y-this.height/2)/this.scale];}
    screenAt(p){return [(p[0]-this.center[0])*this.scale+this.width/2,(p[1]-this.center[1])*this.scale+this.height/2];}
    zoomAt(x,y,value){const p=this.imageAt(x,y);this.zoom=Math.max(1,Math.min(32,value));this.center=[p[0]-(x-this.width/2)/this.scale,p[1]-(y-this.height/2)/this.scale];}
    pan(dx,dy){this.center=[this.center[0]-dx/this.scale,this.center[1]-dy/this.scale];}
}

export function visibleImageRect(view, content=null) {
    const scale=view.scale;
    if(!Number.isFinite(scale)||scale<=0)return null;
    const left=view.center[0]-view.width/(2*scale),top=view.center[1]-view.height/(2*scale);
    const right=view.center[0]+view.width/(2*scale),bottom=view.center[1]+view.height/(2*scale);
    const bounds=content||[0,0,view.imageWidth,view.imageHeight];
    const x0=Math.max(0,bounds[0],Math.floor(left)-1),y0=Math.max(0,bounds[1],Math.floor(top)-1);
    const x1=Math.min(view.imageWidth,bounds[0]+bounds[2],Math.ceil(right)+1);
    const y1=Math.min(view.imageHeight,bounds[1]+bounds[3],Math.ceil(bottom)+1);
    return x1>x0&&y1>y0?[x0,y0,x1-x0,y1-y0]:null;
}

export function pointInsidePolygon(point, polygon) {
    let inside=false;
    for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
        const a=polygon[i],b=polygon[j];
        if((a[1]>point[1])!==(b[1]>point[1])&&
            point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;
    }
    return inside;
}

const $=id=>document.getElementById(id);
const names={live:'ライブ',ct:'CT'},colors={live:'#ffe066',ct:'#fb82ff'};
export class MeasurementWorkspace {
    constructor(send,onError){
        this.send=send;this.onError=onError;this.dialog=$('measureDialog');this.pendingOpen=false;this.meta=null;this.session=null;
        this.panes={};this.committing=false;this.pendingPoint=null;this.selectedFace=null;this.facePicking=false;
        this.occlusionPoints=[];this.occlusionPicking=false;
        for(const kind of ['live','ct']){
            const article=document.createElement('article');article.className='measure-pane';
            article.innerHTML=`<h3><b id="${kind}MeasureTitle">${kind==='live'?'① ライブ画像のマーカー':'② CT断面の同じマーカー'}</b> <span id="${kind}PickState"></span></h3>
                <canvas id="${kind}MeasureCanvas" tabindex="0" aria-label="${names[kind]}画像。クリックで仮選択、ドラッグで移動、ホイールで拡大、矢印で点を微調整"></canvas>
                <div class="measure-pane-tools"><button data-fit>全体</button><button data-minus>−</button><output data-zoom>1×</output><button data-plus>＋</button><button data-focus>選択点を拡大</button><button data-confirm class="primary">この${names[kind]}点を確定</button></div>
                <p data-coordinate>画像上をクリックしてマーカーを選択</p>`;
            $('measurePanes').appendChild(article);
            const pane={kind,article,canvas:$(`${kind}MeasureCanvas`),view:new ImageViewTransform(),draft:null,pin:null};this.panes[kind]=pane;
            pane.canvas.onpointerdown=e=>{if(e.button!==0)return;pane.canvas.focus();pane.canvas.setPointerCapture(e.pointerId);pane.drag={x:e.clientX,y:e.clientY,lastX:e.clientX,lastY:e.clientY,moved:false};};
            pane.canvas.onpointermove=e=>{const d=pane.drag;if(!d)return;const dx=e.clientX-d.lastX,dy=e.clientY-d.lastY;d.lastX=e.clientX;d.lastY=e.clientY;if(Math.hypot(e.clientX-d.x,e.clientY-d.y)>4)d.moved=true;if(d.moved){pane.view.pan(dx,dy);this.link(pane);this.draw();}};
            pane.canvas.onpointerup=e=>{const d=pane.drag;pane.drag=null;if(!d||d.moved)return;
                const rect=pane.canvas.getBoundingClientRect(),p=pane.view.imageAt(e.clientX-rect.left,e.clientY-rect.top);
                if(p[0]<0||p[1]<0||p[0]>=this.meta.width||p[1]>=this.meta.height)return;
                if(this.occlusionPicking&&pane.kind==='live'){
                    if(this.occlusionPoints.length>=64){this.onError('頂点は64点までです。');return;}
                    this.occlusionPoints.push(p);this.draw();return;
                }
                if(this.facePicking&&pane.kind==='live'){
                    const face=(this.meta.validation_faces||[]).filter(f=>pointInsidePolygon(p,f.polygon)).sort((a,b)=>a.depth-b.depth)[0];
                    if(!face){this.onError('黄色のモデル枠内で、見えている面をクリックしてください。');return;}
                    this.selectedFace=face.id;this.facePicking=false;this.draw();return;
                }
                pane.draft=p;this.draw();};
            pane.canvas.onpointercancel=()=>{pane.drag=null;};
            pane.canvas.onwheel=e=>{e.preventDefault();const rect=pane.canvas.getBoundingClientRect();pane.view.zoomAt(e.clientX-rect.left,e.clientY-rect.top,pane.view.zoom*Math.exp(-Math.max(-200,Math.min(200,e.deltaY))*.004));this.link(pane);this.draw();};
            pane.canvas.onkeydown=e=>{
                if(this.occlusionPicking){
                    if(e.key==='Enter'||e.key==='Escape'){e.preventDefault();this.occlusionPicking=false;this.draw();}
                    return;
                }
                const dirs={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
                if(e.key==='Enter'){e.preventDefault();this.confirm(pane);return;}
                if(!dirs[e.key])return;e.preventDefault();
                const p=pane.draft||pane.pin||pane.view.center,step=e.shiftKey?1:.25;
                pane.draft=p.map((v,i)=>Math.max(0,Math.min((i?this.meta.height:this.meta.width)-1e-6,v+dirs[e.key][i]*step)));this.draw();
            };
            article.querySelector('[data-fit]').onclick=()=>{pane.view.reset(this.meta.width,this.meta.height);this.link(pane);this.draw();};
            for(const [selector,factor] of [['[data-plus]',2],['[data-minus]',.5]])article.querySelector(selector).onclick=()=>{pane.view.zoomAt(pane.view.width/2,pane.view.height/2,pane.view.zoom*factor);this.link(pane);this.draw();};
            article.querySelector('[data-focus]').onclick=()=>{pane.view.center=(pane.draft||pane.pin||pane.view.center).slice();pane.view.zoom=Math.max(8,pane.view.zoom);this.link(pane);this.draw();pane.canvas.focus();};
            article.querySelector('[data-confirm]').onclick=()=>this.confirm(pane);
        }
        this.observer=new ResizeObserver(()=>this.draw());for(const p of Object.values(this.panes))this.observer.observe(p.canvas);
        $('openMeasurement').onclick=()=>this.open();
        $('closeMeasurement').onclick=()=>this.dialog.close();
        $('measureResume').onclick=async()=>{await this.command({action:'resume'});this.dialog.close();};
        $('measureClear').onclick=async()=>{for(const p of Object.values(this.panes))p.draft=null;this.pendingPoint=null;await this.command({action:'clear'});};
        $('measureReturn').onclick=()=>this.command({action:'return'});
        $('measureSave').onclick=()=>this.command({action:'save'});
        $('validationPickFace').onclick=()=>{this.facePicking=!this.facePicking;this.draw();this.panes.live.canvas.focus();};
        $('validationMode').onchange=()=>{this.facePicking=this.occlusionPicking=false;$('validationCompare').open=true;this.draw();};
        $('occlusionReferencePolicy').onchange=()=>{$('validationCompare').open=true;this.draw();};
        $('occlusionDraw').onclick=()=>{this.facePicking=false;this.occlusionPicking=!this.occlusionPicking;this.draw();this.panes.live.canvas.focus();};
        $('occlusionUndo').onclick=()=>{this.occlusionPoints.pop();this.draw();};
        $('occlusionClear').onclick=()=>{this.occlusionPoints=[];this.occlusionPicking=false;this.draw();};
        $('occlusionRun').onclick=()=>{this.occlusionPicking=false;$('validationCompare').open=true;this.draw();return this.command({
            action:'validation_occlusion',polygon:this.occlusionPoints.map(p=>p.slice()),
            reference_policy:$('occlusionReferencePolicy').value,
            max_mean_error_px:Number($('validationLimit').value),
            min_coverage_percent:Number($('validationCoverage').value)});};
        $('validationReprocess').onclick=()=>this.command({action:'validation_reprocess',
            face:this.selectedFace,max_mean_error_px:Number($('validationLimit').value),
            min_coverage_percent:Number($('validationCoverage').value),
            view_count:Number($('validationViews').value),
            reference_policy:$('validationReferencePolicy').value});
        $('validationLimit').oninput=()=>{$('validationLimitValue').value=`${Number($('validationLimit').value).toFixed(1)} px`;};
        $('validationCoverage').oninput=()=>{$('validationCoverageValue').value=`${Number($('validationCoverage').value).toFixed(2)}%`;};
        $('validationCompare').ontoggle=()=>this.draw();
        $('validationMeasureResult').onclick=()=>this.returnToMeasurement();
        $('measureSyncZoom').onchange=()=>{this.link(this.panes.live);this.draw();};
        $('measureOverlay').onchange=()=>this.draw();
        $('measureOverlayOpacity').oninput=()=>this.draw();
        const forwardSlice=type=>{const original=$('slice');original.value=$('measureSlice').value;original.dispatchEvent(new Event(type));this.draw();};
        $('measureSlice').oninput=()=>forwardSlice('input');$('measureSlice').onchange=()=>forwardSlice('change');
        for(const button of document.querySelectorAll('[data-measure-step]'))button.onclick=()=>{const slider=$('measureSlice');slider.value=Number(slider.value)+Number(button.dataset.measureStep);forwardSlice('input');forwardSlice('change');};
    }
    async command(data){try{this.onError('');await this.send({...data,generation:this.meta?.generation});return true;}catch(e){this.onError(e.message);return false;}}
    async open(){
        if(this.meta?.frozen){this.dialog.showModal();this.draw();return;}
        this.pendingOpen=true;if(!await this.command({action:'freeze'}))this.pendingOpen=false;
    }
    returnToMeasurement(){
        if(!this.meta?.frozen||!this.meta.validation_comparison?.applied)return;
        $('validationCompare').open=false;
        this.facePicking=false;
        this.occlusionPicking=false;
        $('measureOverlay').checked=true;
        this.draw();
        this.panes.live.canvas.focus();
        this.panes.live.canvas.scrollIntoView({block:'nearest'});
    }
    link(pane){if(!$('measureSyncZoom').checked)return;const other=this.panes[pane.kind==='live'?'ct':'live'];other.view.center=pane.view.center.slice();other.view.zoom=pane.view.zoom;}
    async confirm(pane){
        if(!pane.draft||this.committing||this.pendingPoint||!this.meta?.frozen)return;
        if($('sliceStatus').textContent.includes('更新中')){this.onError('断面の更新が終わってから確定してください。');return;}
        this.committing=true;
        const uv=pane.draft.slice();this.pendingPoint={kind:pane.kind,uv,after:this.meta.sequence};this.draw();
        const ok=await this.command({action:'point',kind:pane.kind,x:uv[0],y:uv[1],sequence:this.meta.sequence,plane_revision:this.meta.plane_revision});
        if(!ok)this.pendingPoint=null;this.committing=false;this.draw();
    }
    sync(meta){
        const old=this.meta;this.meta=meta;
        $('openMeasurement').disabled=!meta?.pose||!meta?.mpr_loaded;
        $('openMeasurement').textContent=meta?.input_source==='photo'?'写真・CTのずれを拡大して測る':'拡大してライブ・CTのずれを測る';
        if(!meta?.frozen){if(this.dialog.open)this.dialog.close();return;}
        const session=`${meta.generation}:${meta.measurement?.frozen_at}`;
        const sessionChanged=session!==this.session;
        if(sessionChanged){
            if(!old?.frozen||old.generation!==meta.generation){
                this.selectedFace=null;
                const comparison=meta.validation_comparison;
                this.occlusionPoints=comparison?.kind==='occlusion'?(comparison.polygon||[]).map(p=>p.slice()):[];
                if(comparison){
                    $('validationMode').value=comparison.kind==='occlusion'?'occlusion':'face';
                    if(comparison.kind==='occlusion')$('occlusionReferencePolicy').value=comparison.reference_policy;
                }
            }
            this.session=session;this.pendingPoint=null;this.facePicking=false;
            for(const p of Object.values(this.panes)){p.view.reset(meta.width,meta.height);p.draft=p.pin=null;}
            if(!meta.validation_comparison){
                $('validationLimit').value=meta.validation_default_max_error_px??3;
                $('validationLimit').dispatchEvent(new Event('input'));
                $('validationCoverage').value=meta.validation_default_min_coverage_percent??.5;
                $('validationCoverage').dispatchEvent(new Event('input'));
                $('validationViews').value=meta.validation_default_view_count??3;
                $('validationReferencePolicy').value='original';
                $('occlusionReferencePolicy').value='fresh_global';
            }
        }
        if(this.selectedFace&&!(meta.validation_faces||[]).some(face=>face.id===this.selectedFace))this.selectedFace=null;
        // A draft CT click refers to the currently shown slice, not a future one.
        if(old?.frozen&&JSON.stringify(old.mpr_plane)!==JSON.stringify(meta.mpr_plane))this.panes.ct.draft=null;
        const points=meta.measurement?.points||[];
        for(const p of Object.values(this.panes))p.pin=points.find(q=>q.label===p.kind.toUpperCase())?.uv||null;
        const pending=this.pendingPoint;
        if(pending&&meta.sequence>pending.after){
            const pane=this.panes[pending.kind];
            if(pane.pin&&Math.hypot(pane.pin[0]-pending.uv[0],pane.pin[1]-pending.uv[1])<.1){pane.draft=null;this.pendingPoint=null;}
            else if(meta.plane_revision>old?.plane_revision)this.pendingPoint=null;
        }
        if(this.pendingOpen){this.pendingOpen=false;if(!this.dialog.open)this.dialog.showModal();}
        const staticLive=(!sessionChanged&&old?.frozen&&meta.input_source==='photo'&&
            old.generation===meta.generation&&old.source_id===meta.source_id&&
            old.photo_image_revision===meta.photo_image_revision&&
            JSON.stringify(old.measurement?.points)===JSON.stringify(meta.measurement?.points)&&
            !$('measureOverlay').checked);
        this.draw(staticLive);
    }
    draw(skipLive=false){
        if(!this.dialog.open||!this.meta?.frozen)return;
        const meta=this.meta,m=meta.measurement,c=m?.consistency;
        const overlay=$('measureOverlay').checked,alpha=Number($('measureOverlayOpacity').value)/100;
        const inputName=meta.input_source==='photo'?'写真':'ライブ';
        const occlusion=$('validationMode').value==='occlusion';
        const busy=Boolean(this.validationBusy);
        $('occlusionTools').hidden=!occlusion;
        $('occlusionDraw').textContent=this.occlusionPicking?'囲みを完了（Enterでも可）':'領域を囲む';
        $('occlusionHint').textContent=`頂点 ${this.occlusionPoints.length} 点 · ${this.occlusionPicking?'左画像で頂点を順にクリック。ドラッグで移動、ホイールで拡大。':'3点以上で再照合できます。凹形も可能。辺が交差しない順に選んでください。'}`;
        $('occlusionRun').disabled=busy||this.occlusionPoints.length<3;
        $('measureTitle').textContent=`${inputName}・CT マーカーの表示ずれ`;
        $('measureOverlayLabel').textContent=`左の${inputName}画像にCTを重ねる`;
        this.panes.live.article.querySelector('[data-confirm]').textContent=`この${inputName}点を確定`;
        $('measureOverlayOpacity').disabled=!overlay;
        $('measureOverlayOpacityValue').value=`${Math.round(alpha*100)}%`;
        $('liveMeasureTitle').textContent=overlay?`① ${inputName}＋CT（${inputName}点を選択）`:`① ${inputName}画像のマーカー`;
        $('validationPickFace').textContent=this.facePicking?'面をクリックしてください（解除）':'左画像で面を選ぶ';
        $('validationSelectedFace').textContent=`除外する面：${this.selectedFace||'未選択'}${this.facePicking?' · 左画像をクリック':''}`;
        for(const pane of Object.values(this.panes)){
            const canvas=pane.canvas,rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)continue;
            pane.view.width=rect.width;pane.view.height=rect.height;
            const dpr=Math.min(2,window.devicePixelRatio||1);
            const resized=canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr);
            if(resized){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
            const redraw=!(skipLive&&pane.kind==='live'&&!resized);
            const ctx=canvas.getContext('2d');
            if(redraw){
                ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#070b12';ctx.fillRect(0,0,canvas.width/dpr,canvas.height/dpr);
                const scale=pane.view.scale;
                const drawVisible=(source,content=null)=>{
                    const area=visibleImageRect(pane.view,content);if(!area)return;
                    const [sx,sy,sw,sh]=area,[dx,dy]=pane.view.screenAt([sx,sy]);
                    ctx.drawImage(source,sx,sy,sw,sh,dx,dy,sw*scale,sh*scale);
                };
                drawVisible($(pane.kind==='live'?'liveCanvas':'mprCanvas'),pane.kind==='ct'?meta.layer_rect:null);
                if(overlay&&pane.kind==='live'){
                    ctx.save();ctx.globalAlpha=alpha;drawVisible($('mprCanvas'),meta.layer_rect);ctx.restore();
                    const ct=this.panes.ct.pin;
                    if(pane.pin&&ct){
                        const a=pane.view.screenAt(pane.pin),b=pane.view.screenAt(ct);
                        ctx.strokeStyle='#ffffff';ctx.lineWidth=1;ctx.setLineDash([4,4]);
                        ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);
                    }
                }
                if(pane.kind==='live'&&occlusion&&this.occlusionPoints.length&&
                    (this.occlusionPicking||$('validationCompare').open)){
                    ctx.beginPath();this.occlusionPoints.forEach((point,index)=>{const [x,y]=pane.view.screenAt(point);if(index)ctx.lineTo(x,y);else ctx.moveTo(x,y);});
                    if(this.occlusionPoints.length>=3){ctx.closePath();ctx.fillStyle='#4b4b4bcc';ctx.fill();}
                    ctx.strokeStyle='#ffca55';ctx.lineWidth=2;ctx.stroke();
                    for(const p of this.occlusionPoints){const [x,y]=pane.view.screenAt(p);ctx.fillStyle='#ffca55';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();}
                }
                if(pane.kind==='live'&&!occlusion&&this.selectedFace&&$('validationCompare').open){
                    const face=(meta.validation_faces||[]).find(f=>f.id===this.selectedFace);
                    if(face){ctx.beginPath();face.polygon.forEach((point,index)=>{const [x,y]=pane.view.screenAt(point);if(index)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.closePath();ctx.fillStyle='#f06b6b55';ctx.fill();ctx.strokeStyle='#ff7777';ctx.lineWidth=2;ctx.stroke();}
                }
            }
            const mark=(point,draft,color=colors[pane.kind],label='')=>{if(!point)return;const [x,y]=pane.view.screenAt(point);ctx.strokeStyle=draft?'#ffffff':color;ctx.lineWidth=1.5;ctx.setLineDash(draft?[3,3]:[]);ctx.beginPath();ctx.moveTo(x-14,y);ctx.lineTo(x-4,y);ctx.moveTo(x+4,y);ctx.lineTo(x+14,y);ctx.moveTo(x,y-14);ctx.lineTo(x,y-4);ctx.moveTo(x,y+4);ctx.lineTo(x,y+14);ctx.stroke();ctx.setLineDash([]);ctx.strokeRect(x-1,y-1,2,2);if(label){ctx.font='12px system-ui';ctx.strokeStyle='#07101c';ctx.lineWidth=3;ctx.strokeText(label,x+17,y-8);ctx.fillStyle=color;ctx.fillText(label,x+17,y-8);}};
            if(redraw){
                if(overlay&&pane.kind==='live'){
                    mark(this.panes.ct.pin,false,colors.ct,'CT');
                    mark(pane.pin,false,colors.live,'LIVE');
                }else mark(pane.pin,false);
                mark(pane.draft,true);
            }
            pane.article.querySelector('[data-zoom]').value=`${pane.view.zoom.toFixed(1)}×`;
            const point=pane.draft||pane.pin;
            pane.article.querySelector('[data-coordinate]').textContent=point?`${pane.draft?'仮選択':'確定'} · u ${point[0].toFixed(2)}, v ${point[1].toFixed(2)} px`:'クリックで仮選択 → 拡大・微調整 → 確定';
            $(`${pane.kind}PickState`).textContent=this.pendingPoint?.kind===pane.kind?'確定中…':pane.pin?'✓ 確定済み':'未選択';
            pane.article.querySelector('[data-confirm]').disabled=!pane.draft||this.committing||Boolean(this.pendingPoint)||$('sliceStatus').textContent.includes('更新中');
        }
        $('measureValue').textContent=c?`${c.lateral_error.toFixed(2)} ${c.unit}`:'—';
        $('measurePixels').textContent=c?`画面上 ${c.pixel_error.toFixed(2)} px`:'左右で同じマーカーを指定';
        $('measureCoordinates').textContent=c?`CT点 (${c.ct_physical_xyz.map(v=>v.toFixed(2)).join(', ')}) / ライブ点のCT深度換算 (${c.live_at_ct_depth_physical_xyz.map(v=>v.toFixed(2)).join(', ')}) / 差 Δ (${c.delta_ct_physical.map(v=>v.toFixed(2)).join(', ')}) ${c.unit}`:'';
        const offset=m?.slice_offset;
        $('measureOffset').textContent=offset==null?'':Math.abs(offset)>.01?`CT確定点は現在の断面から ${offset.toFixed(2)} ${m.unit} 離れています。「CT点の断面へ」で戻れます。`:'CT確定点は現在の断面上にあります。';
        $('measureSave').disabled=!c||Boolean(this.pendingPoint);$('measureReturn').disabled=!this.panes.ct.pin;
        const slider=$('measureSlice'),original=$('slice');
        for(const key of ['min','max','step'])slider[key]=original[key];slider.value=original.value;slider.disabled=original.disabled;
        $('measureSliceValue').value=$('sliceValue').value;$('measureSliceStatus').textContent=$('sliceStatus').textContent;
        for(const b of document.querySelectorAll('[data-measure-step]'))b.disabled=original.disabled;
    }
}
