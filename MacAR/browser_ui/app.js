import {GaussianView} from './gs-view.js';
import {drawFeatures,pointCounts} from './feature-overlay.js';
import {searchSummary,backgroundSummary,imageQualitySummary,diagnosticLogSummary} from './localization-progress.js';
import {installSliceControls,applyMPRVisibility} from './slice-controls.js';
import {ForegroundMask} from './foreground-mask.js';
import {AnatomyDemo} from './anatomy-demo.js';
import {MeasurementWorkspace} from './measurement-workspace.js';
import {validationReasons,validationStageSummary} from './validation-diagnostics.js';
import {installReferenceSelection} from './reference-selection.js';
import {MPRPlane} from './mpr-plane.js';
import {BrowserBenchmark} from './browser-benchmark.js';
const $=id=>document.getElementById(id);
const inputUI=()=>{const photo=$('input_source').value==='photo';$('photoInputs').hidden=!photo;$('start').textContent=photo?'写真を照合':'開始';};
$('input_source').onchange=inputUI;
inputUI();
function matchingUI(){
    const registered=$('mode').value==='registered';
    $('gsMatchingOptions').hidden=false;
    $('registeredSearchOptions').hidden=!registered;
    $('matchingInfo').textContent=registered
        ?'初期化・復帰は3DGSで粗い視点を選び、登録実写ALIKEDと照合。追跡中は姿勢からXYZを投影して近傍点を検索し、定期的に同方向の実写2枚までをLightGlue照合して別領域も補充します（DINO再検索なし）。XYZは実写の多視点三角測量。PASS中はCT/MPRをリアルタイム重畳、ロスト時は非表示。「固定して測定」で表示ずれを確認できます。'
        :'比較用：同じ3DGS view選択・追跡・近傍補充・QCを使い、仮想画像ALIKEDと近似XYZでPnP。PASS中はCT/MPRをリアルタイム重畳、ロスト時は非表示。「固定して測定」も使えます。';
}
$('mode').onchange=matchingUI;
matchingUI();
const foregroundMask=new ForegroundMask($('sliceMaskCanvas'));
const anatomyDemo=new AnatomyDemo($('anatomyCanvas'),$('anatomyStatus'));
const mprPlane=new MPRPlane();
let measurementWorkspace=null,starting=false;
let config;
try{
    const response=await fetch('/api/config');
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    config=await response.json();
}catch(e){
    $('connection').textContent='Pythonに接続できません';
    $('error').hidden=false;
    $('error').textContent='Pythonサーバーに接続できません。VS Codeの実行を確認し、新しく表示されたURLで開き直してください。';
    $('start').disabled=true;
    throw e;
}
for(const [name,value] of Object.entries(config.defaults)){if($(name)){if(typeof value==='boolean')$(name).checked=value;else $(name).value=value;}}
const syncReference=installReferenceSelection($('referencePreset'),$('views_path'),config.reference_presets);
inputUI();
matchingUI();
if(config.demo)$('connection').textContent='DEMO · 既知姿勢';
let gs=null, sequence=0, frame=null, generation=null, zoomCenter=[.5,.5], shutdown=false, lastError='';
let photoImageGeneration=null,photoImageRevision=null,lastLayerRect=null,validationImageKey='',lastCompletedComparisonKey=null;
let fileSelectionBusy=false;
try{gs=new GaussianView($('gsCanvas'),$('gsStatus'));}catch(e){$('gsStatus').textContent=`GPU初期化失敗: ${e.message}`;}
async function benchmarkFetch(path,options={}){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),1500);
    try{const response=await fetch(path,{...options,signal:controller.signal});
        if(!response.ok)throw Error(`Browser log HTTP ${response.status}`);return response;
    }finally{clearTimeout(timer);}
}
const browserBenchmark=new BrowserBenchmark({clientId:crypto.randomUUID(),
    clockRequest:async()=>await(await benchmarkFetch('/api/clock')).json(),
    send:async(payload,{keepalive})=>{
        const body=JSON.stringify(payload);
        if(new TextEncoder().encode(body).length>65536)throw Error('Browser log batch too large');
        await benchmarkFetch('/api/browser-telemetry',{method:'POST',keepalive,
            headers:{'Content-Type':'application/json','X-ViewSync-Token':config.token},body});
    },
    gsState:()=>({render_count:gs?.renderCount||0,active:gs?.active||false,ready:gs?.ready||false,
        visibility:document.visibilityState,gs_mode:gs?.orbit?'orbit':'matched',
        gs_layout:$('layout').value,mpr_visible:$('showMPR').checked}),
    environment:{user_agent:navigator.userAgent.slice(0,400),width:innerWidth,height:innerHeight,
        pixel_ratio:devicePixelRatio,hardware_concurrency:navigator.hardwareConcurrency??null}});
browserBenchmark.syncClock();
const clockTimer=setInterval(()=>browserBenchmark.syncClock(),10000);
const benchmarkTimer=setInterval(()=>browserBenchmark.flush(),1000);
document.addEventListener('visibilitychange',()=>{
    if(document.hidden)browserBenchmark.hidden();else browserBenchmark.syncClock();
    browserBenchmark.flush();
});
const post=async(path,data={})=>{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-ViewSync-Token':config.token},body:JSON.stringify(data)});if(!response.ok)throw Error((await response.json()).error||response.statusText);};
async function selectRegistrationFile(field,recommended=false){
    if(fileSelectionBusy||starting)return;
    fileSelectionBusy=true;error('');
    const buttons=[...document.querySelectorAll('[data-picker]'),$('recommendedMap'),$('clearCalibration'),$('referencePreset')];
    buttons.forEach(button=>button.disabled=true);
    try{
        const response=await fetch(recommended?'/api/recommended-map':'/api/select-file',{method:'POST',headers:{'Content-Type':'application/json','X-ViewSync-Token':config.token},body:JSON.stringify({field})});
        const result=await response.json();if(!response.ok)throw Error(result.error||'ファイル選択に失敗しました');
        if(!result.cancelled){$(field).value=result.path;$(field).title=result.path;
            if(field==='views_path')syncReference();
            if(field==='map_path')$('mapSelectionInfo').textContent=`${recommended?'推奨地図':'選択した地図'}：${result.name} · XYZ付与率 ${(result.xyz_ratio*100).toFixed(1)}% · XYZ観測 ${result.xyz_observations.toLocaleString()} 件（CT実測精度の保証ではありません）`;
        }
    }catch(e){error(e.message);}finally{fileSelectionBusy=false;}
}
for(const button of document.querySelectorAll('[data-picker]'))button.onclick=()=>selectRegistrationFile(button.dataset.picker);
$('recommendedMap').onclick=()=>selectRegistrationFile('map_path',true);
$('clearCalibration').onclick=()=>{$('calibration_path').value='';$('calibration_path').title='';};
for(const id of ['map_path','views_path','volume_path','coordinates_path','calibration_path'])$(id).title=$(id).value;
$('mapSelectionInfo').textContent=`使用する地図：${$('map_path').value.split('/').slice(-2).join('/')}。関連データは同じフォルダから読み込みます。「推奨地図に戻す」で監査済み・XYZ付与率の高い地図を選択できます。`;
function error(message){$('error').hidden=!message;$('error').textContent=message;$('measureError').hidden=!message;$('measureError').textContent=message;if(message&&measurementWorkspace){measurementWorkspace.pendingPoint=null;measurementWorkspace.pendingOpen=false;measurementWorkspace.draw();}}
async function control(action,extra={}){try{error('');await post('/api/control',{action,...extra});}catch(e){error(e.message);}}
$('photoFile').onchange=()=>{$('photoInfo').textContent=$('photoFile').files[0]?.name||'写真を選択してください。';};
$('start').onclick=async()=>{const settings={};for(const key of Object.keys(config.defaults))settings[key]=typeof config.defaults[key]==='boolean'?$(key).checked:$(key).value;try{starting=true;$('start').disabled=true;error('');
    if(settings.input_source==='photo'){
        const file=$('photoFile').files[0];if(!file)throw Error('入力するJPEG・PNG写真を選んでください。');if(file.size>64*1024*1024)throw Error('写真は64 MB以下を選んでください。');
        $('photoInfo').textContent='写真を読み込み中…';
        const response=await fetch('/api/photo',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-ViewSync-Token':config.token,'X-Photo-Name':encodeURIComponent(file.name)},body:file});
        const uploaded=await response.json();if(!response.ok)throw Error(uploaded.error||'写真の読み込みに失敗しました');
        $('photoInfo').textContent=`${uploaded.name} · ${uploaded.width} × ${uploaded.height} px（元解像度）`;
    }
    await post('/api/start',settings);
}catch(e){error(e.message);$('start').disabled=false;}finally{starting=false;}};
$('stop').onclick=async()=>{await browserBenchmark.endSession();post('/api/stop').catch(e=>error(e.message));};
for(const name of ['freeze','resume','clear','return','save'])$(name).onclick=()=>control(name);
for(const name of ['relocalize','diagnostic'])$(name).onclick=()=>control(name);
const featureSettings=()=>Object.fromEntries(['showInliers','showOutliers','showResiduals','showPointIds','showModelBox'].map(id=>[id,$(id).checked]).concat([['pointSize',$('pointSize').value]]));
function features(){drawFeatures($('featureCanvas'),frame?.correspondences,featureSettings());$('pointSizeValue').value=`${$('pointSize').value} px`;}
for(const name of ['showInliers','showOutliers','showResiduals','showPointIds','showModelBox','pointSize'])$(name).oninput=features;
let searchKey='',referenceKey='';
// Reserve document space for the fixed dock, including its expanded details.
const searchDockObserver=new ResizeObserver(()=>{
    const panel=$('searchPanel'),height=panel.hidden?0:panel.getBoundingClientRect().height+24;
    document.documentElement.style.setProperty('--search-dock-space',`${height}px`);
});
searchDockObserver.observe($('searchPanel'));
function drawSearchROI(){
    const canvas=$('searchROI'),image=$('searchImage'),roi=frame?.search?.roi;
    canvas.width=image.naturalWidth||320;canvas.height=image.naturalHeight||180;
    const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
    if(!roi||roi.length!==4||!roi.every(Number.isFinite))return;
    ctx.strokeStyle='#ffe066';ctx.lineWidth=2;ctx.strokeRect(roi[0]*canvas.width,roi[1]*canvas.height,roi[2]*canvas.width,roi[3]*canvas.height);
}
$('searchImage').onload=drawSearchROI;
function searchDisplay(meta){
    const s=meta?.search,info=searchSummary(s),visible=$('showSearch').checked&&info.visible&&!meta?.frozen;
    $('searchPanel').hidden=!visible;if(!visible)return;
    if(!meta.pose)$('poseBadge').textContent='SEARCH';
    $('searchPanel').classList.toggle('waiting',!info.active);
    $('searchBar').hidden=$('searchSteps').hidden=!info.active;
    $('searchTitle').textContent=info.title;$('searchElapsed').textContent=info.active?`${info.elapsed} 秒`:info.waiting?'自動再試行':'追跡継続';
    $('searchDetail').textContent=`${info.label}${info.detail?' · '+info.detail:''}`;
    $('searchDetail').title=$('searchDetail').textContent;
    const quality=imageQualitySummary(s);
    $('searchQuality').hidden=!quality;$('searchQuality').textContent=quality;
    const ranking=info.active?s.patch_rerank:null;
    $('rerankDetail').hidden=$('rerankTable').hidden=!ranking;
    if(ranking){
        $('rerankDetail').textContent=`全体Top-${ranking.shortlist} → パッチ＋方向分散 ${ranking.rerank_ms.toFixed(1)} ms → 局所照合${ranking.fine_candidates}枚 · Top-1 view ${ranking.global_top1_index} → ${ranking.reranked_top1_index}（未確定） · 照合対象 ${ranking.selected_images.join(', ')}`;
        $('rerankTable').replaceChildren(...ranking.rows.slice(0,5).map((row,i)=>{
            const line=document.createElement('div');
            line.textContent=`${row.fine_selected?'✓照合':'候補'} ${i+1}: ${row.image_name} · 元${row.global_rank}位 · 全体${row.global_score.toFixed(3)} / 配置${row.layout.toFixed(3)} / 整合${row.consistency.toFixed(3)} · パッチ支持${row.inliers}/${row.mutual_matches}`;
            return line;
        }));
    }
    for(const step of $('searchSteps').children)step.className=Number(step.dataset.step)===info.step?'active':Number(step.dataset.step)<info.step?'done':'';
    $('searchSteps').children[0].textContent=s.matching_method==='pose_landmarks'?'① 姿勢から点投影':'① 検出 / view選択';
    $('queryPreview').hidden=!info.active||s.source_id==null;
    const key=`${s.generation}:${s.preview_id}`;
    if(info.active&&s.source_id!=null&&key!==searchKey){searchKey=key;$('searchImage').src=`/api/search-image?generation=${s.generation}&preview_id=${s.preview_id}`;}
    const corrected=s.photometric_mode&&s.photometric_mode!=='raw';
    $('queryCaption').textContent=meta.input_source==='photo'?'入力写真（探索用プレビュー）':`探索画像 #${s.source_id}（${corrected?'明度補正前・':''}現在映像とは別）`;drawSearchROI();
    $('referencePreview').hidden=!info.active||info.reference===null;
    const refkey=`${s.generation}:${info.reference}`;
    if(info.active&&info.reference!==null&&refkey!==referenceKey){referenceKey=refkey;$('referenceImage').src=`/api/reference-image?generation=${s.generation}&index=${info.reference}`;}
    $('referenceCaption').textContent=`参照候補 ${info.referenceName}`;
}
$('showSearch').onchange=()=>searchDisplay(frame);
let overlaysHidden=false;
$('hideOverlays').onclick=()=>{overlaysHidden=!overlaysHidden;for(const id of ['anatomyCanvas','gsOverlay'])$(id).style.visibility=overlaysHidden?'hidden':'visible';$('hideOverlays').textContent=overlaysHidden?'重畳を戻す':'重畳を隠す';layout();};
$('resetDisplay').onclick=()=>{overlaysHidden=false;$('showMPR').checked=true;for(const id of ['anatomyCanvas','gsOverlay'])$(id).style.visibility='visible';$('hideOverlays').textContent='重畳を隠す';$('layout').value='side';$('gsViewMode').value='matched';$('zoom').value=1;zoomCenter=[.5,.5];zoom();for(const name of ['showInliers','showOutliers'])$(name).checked=true;for(const name of ['showResiduals','showPointIds','showModelBox'])$(name).checked=false;$('pointSize').value=4;$('anatomyMode').value='off';$('anatomyOpacity').value=65;for(const id of ['anatomyTumor','anatomyArtery','anatomyVein'])$(id).checked=true;features();layout();};
const sliceControls=installSliceControls(post,e=>error(e.message),()=>layout());
measurementWorkspace=new MeasurementWorkspace(data=>post('/api/control',data),error);
function layout(){
    const mprVisible=applyMPRVisibility($,overlaysHidden);
    $('mprDisplayStatus').textContent=!mprVisible?'CT / MPR：非表示（表示設定のみ。追跡は継続）'
        :!frame?.mpr_loaded?'CT / MPR：読込前（開始時に「CT / MPRを読み込む」を選択）'
        :!frame?.pose?'CT / MPR：姿勢未確定・ロスト中は非表示'
        :frame?.mpr_status==='PENDING'?'CT / MPR：断面生成中（映像・追跡は継続）'
        :frame?.mpr_status==='GPU_UNAVAILABLE'?'CT / MPR：GPU投影が利用できません'
        :frame?.mpr_status!=='VISIBLE'?'CT / MPR：この断面位置に表示できるデータなし'
        :frame?.frozen?'CT / MPR：固定画像に重畳・測定中':`CT / MPR：現在姿勢へ投影 · 断面内容の経過 ${Math.round(frame?.mpr_age_ms||0)} ms`;
    // Live-image masking depends only on the yellow box + accepted pose/plane.
    // It also works while PLY is loading or when the GS viewer is unavailable.
    foregroundMask.configure(frame?.pose,frame?.mpr_plane,frame?.model_bounds,mprVisible&&$('sliceMask').checked&&Number($('opacity').value)>0);
    foregroundMask.render();
    $('anatomyOpacityValue').value=`${$('anatomyOpacity').value}%`;
    anatomyDemo.render(frame?.pose,frame?.mpr_plane,frame?.model_bounds,{mode:$('anatomyMode').value,opacity:$('anatomyOpacity').value,tumor:$('anatomyTumor').checked,artery:$('anatomyArtery').checked,vein:$('anatomyVein').checked});
    const mode=$('layout').value;document.body.classList.toggle('overlay',mode==='overlay');document.body.classList.toggle('gs-off',mode==='off');$('gsViewMode').disabled=mode!=='side';
    if(gs){(mode==='overlay'?$('gsOverlay'):$('gsViewport')).appendChild(gs.canvas);const orbit=mode==='side'&&$('gsViewMode').value==='orbit',pose=frame?.pose||(orbit?frame?.inspection_camera:null);gs.setOrbit(orbit,pose,frame?.model_center,frame?.world_up);gs.update(pose,Boolean(pose)&&mode!=='off',mode==='overlay'?Number($('gsOpacity').value)/100:1);gs.commit();$('gsCanvas').style.cursor=orbit?'grab':'default';$('gsTitle').textContent=orbit?'3DGS · 自由視点（推定姿勢ではありません）':'3DGS · 同じ画像・同じ姿勢';$('gsEmpty').hidden=Boolean(pose)&&Boolean(gs.ready);}
}
$('showMPR').onchange=layout;
$('gsViewMode').onchange=layout;
$('layout').onchange=layout;
for(const id of ['anatomyMode','anatomyTumor','anatomyArtery','anatomyVein','anatomyOpacity'])$(id).oninput=layout;
$('showAnatomyDemo').onclick=()=>{$('anatomyMode').value=frame?.mpr_plane?'cutaway':'transparent';$('anatomyOpacity').value=65;for(const id of ['anatomyTumor','anatomyArtery','anatomyVein'])$(id).checked=true;layout();};
$('gsOpacity').oninput=()=>{$('gsOpacityValue').value=`${$('gsOpacity').value}%`;layout();};
function zoom(){const amount=Number($('zoom').value);$('zoomValue').value=`${amount.toFixed(1)}×`;$('stack').style.transformOrigin=`${zoomCenter[0]*100}% ${zoomCenter[1]*100}%`;$('stack').style.transform=`scale(${amount})`;}
$('zoom').oninput=zoom;$('resetZoom').onclick=()=>{$('zoom').value=1;zoomCenter=[.5,.5];zoom();};
$('markers').onclick=event=>{
    if(!frame?.frozen){error('「固定して測定」を押してからクリックしてください。');return;}
    const rect=$('markers').getBoundingClientRect();const x=(event.clientX-rect.left)*frame.width/rect.width,y=(event.clientY-rect.top)*frame.height/rect.height;
    if(!frame.measurement?.points?.length){zoomCenter=[x/frame.width,y/frame.height];zoom();}
    control('point',{x,y,kind:$('pointKind').value,sequence:frame.sequence,plane_revision:frame.plane_revision});
};
$('markers').oncontextmenu=event=>{event.preventDefault();control('clear');};
$('liveViewport').onwheel=event=>{if(!frame?.frozen)return;event.preventDefault();const rect=$('markers').getBoundingClientRect();zoomCenter=[(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height];$('zoom').value=Math.max(1,Math.min(12,Number($('zoom').value)+(event.deltaY<0?.5:-.5)));zoom();};
function drawMarkers(meta){const canvas=$('markers'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const points=meta.measurement?.points||[];for(const p of points){const [x,y]=p.uv;ctx.strokeStyle=p.color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-10,y);ctx.lineTo(x+10,y);ctx.moveTo(x,y-10);ctx.lineTo(x,y+10);ctx.stroke();ctx.fillStyle=p.color;ctx.font='14px system-ui';ctx.fillText(p.label,x+12,y-12);}if(points.length===2){ctx.strokeStyle='#ffe066';ctx.beginPath();ctx.moveTo(...points[0].uv);ctx.lineTo(...points[1].uv);ctx.stroke();}}
function telemetry(meta){const result=meta.result, age=meta.source_age_ms||0,rates=browserBenchmark.rates;$('poseBadge').textContent=meta.demo?'DEMO':meta.frozen?'FROZEN':meta.status;$('poseBadge').className='tag '+(meta.pose?'pass':'fail');$('liveTitle').textContent=meta.frozen?'固定画像 / CT検証':age>200?'PnP対象画像 / 経過時間に注意':'カメラ / MPR';$('metrics').textContent=`受信 ${rates.receive_fps.toFixed(1)} fps · 姿勢 ${rates.pose_fps.toFixed(1)} fps · GS GPU ${rates.gs_fps.toFixed(1)} fps · capture ${(meta.capture_fps||0).toFixed(1)} fps | localize ${meta.localize_ms.toFixed(1)} ms · MPR ${meta.mpr_ms.toFixed(1)} ms | 画像経過 ${age.toFixed(0)} ms`;
    searchDisplay(meta);
    const background=meta.frozen?'':backgroundSummary(meta.search);
    $('backgroundStatus').hidden=!background;$('backgroundStatus').textContent=background;
    const diagnostic=diagnosticLogSummary(meta.diagnostic_log,meta.sequence);
    $('diagnosticLog').hidden=!diagnostic;$('diagnosticLog').textContent=diagnostic+(browserBenchmark.lastError?' · ブラウザログ送信失敗':' · ブラウザ計測ログ有効');
    $('diagnosticLog').title=meta.diagnostic_log?.path||'';
    $('detail').textContent=`${meta.calibration} | ${result?`${result.inliers} inliers / ${result.matches} matches · 誤差 ${result.mean_reprojection_error?.toFixed(2)??'—'} px · ${result.detail}`:'姿勢未確定・重畳なし'} | MPR ${meta.mpr_status}`;
    const counts=pointCounts(meta.correspondences);$('pointStats').textContent=meta.demo?'DEMO：既知姿勢のため実際の照合点なし':`${meta.quality_profile==='precision'?'厳格判定':meta.quality_profile==='fast'?'速度優先':'安定優先'} | 対応候補 ${counts.total} · 採用 ${counts.inliers} · 未採用 ${counts.rejected} | 採用点の画像分布 ${((meta.correspondences?.image_coverage||0)*100).toFixed(2)}%（画像全体に対する凸包面積）`;
    const measurement=meta.measurement;let text=measurement?.points?.length?`測定点 ${measurement.points.length} 個を指定済み。対応する点を追加してください。`:'測定点：未選択';if(measurement?.consistency){const c=measurement.consistency;text=`表示ずれ ${c.pixel_error.toFixed(2)} px / CT深度換算 ${c.lateral_error.toFixed(2)} ${c.unit}（奥行き誤差ではありません）`;if(measurement.slice_offset!==null)text+=` · CT点の断面距離 ${measurement.slice_offset.toFixed(1)} ${c.unit}`;}if(measurement?.distance!==null&&measurement?.distance!==undefined)text=`CT上の2点距離 ${measurement.distance.toFixed(2)} ${measurement.unit}（固定3D点）`;$('measurement').textContent=text;
    $('freeze').disabled=!meta.pose||meta.frozen||!meta.mpr_loaded;$('resume').disabled=!meta.frozen;for(const id of ['pointKind','clear','return','save'])$(id).disabled=!meta.frozen;for(const id of ['slice','opacity'])$(id).disabled=!meta.mpr_loaded;
    $('relocalize').disabled=meta.demo||meta.frozen;$('diagnostic').disabled=false;
    sliceControls.sync(meta);
    if(meta.input_source==='photo'){
        $('liveTitle').textContent=`写真 / CT · ${meta.photo_name}`;
        $('poseBadge').textContent=meta.pose?'PHOTO · PASS':meta.status;
        $('metrics').textContent=`写真 ${meta.width} × ${meta.height} px · 粗検索 幅≤${meta.photo_retrieval_max_width} px · ALIKED入力 ${meta.photo_feature_size} px · 照合 ${meta.localize_ms.toFixed(0)} ms · MPR ${meta.mpr_ms.toFixed(0)} ms`;
        $('relocalize').textContent='写真を再照合';$('relocalize').disabled=Boolean(meta.search?.active);
        $('resume').textContent='写真の重畳画面へ';$('measureResume').textContent='写真の重畳画面へ';
    }else{$('relocalize').textContent='全体を再探索';$('resume').textContent=$('measureResume').textContent='ライブに戻す';}
    if(meta.input_source!=='photo'){
        const fps=meta.pipeline_fps||{};
        $('metrics').textContent+=` | Python追跡 ${(fps.tracking||0).toFixed(1)} / MPR ${(fps.mpr||0).toFixed(1)} / 送出 ${(fps.packets||0).toFixed(1)} fps`;
    }
    measurementWorkspace.sync(meta);
}
async function frames(){while(!shutdown){try{const response=await fetch(`/api/frame?after=${sequence}`);if(response.status===204)continue;if(!response.ok)throw Error('映像接続失敗');const buffer=await response.arrayBuffer(),receivedAt=performance.now();const headerBytes=new DataView(buffer).getUint32(0,true), offset=4+headerBytes;const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,headerBytes)));const layerOffset=offset+meta.jpeg_bytes,maskOffset=layerOffset+meta.layer_bytes;
    const photoChanged=meta.input_source==='photo'&&(meta.generation!==photoImageGeneration||meta.photo_image_revision!==photoImageRevision);
    const imageRequest=photoChanged
        ?fetch(`/api/photo-image?generation=${meta.generation}&revision=${meta.photo_image_revision}`).then(async response=>{if(!response.ok)throw Error('写真の再取得に失敗しました');return createImageBitmap(await response.blob());})
        :meta.input_source==='photo'?Promise.resolve(null)
        :createImageBitmap(new Blob([buffer.slice(offset,layerOffset)],{type:meta.image_mime||'image/jpeg'}));
    const projective=meta.layer_mode==='projective';
    const needsTexture=!projective||meta.generation!==generation||mprPlane.key!==meta.layer_id;
    const bitmaps=await Promise.all([imageRequest,meta.layer_bytes&&needsTexture?createImageBitmap(new Blob([buffer.slice(layerOffset,maskOffset)],{type:'image/png'}),{premultiplyAlpha:'none'}):Promise.resolve(null)]);
    const decodedAt=performance.now();
    if(meta.generation!==generation){generation=meta.generation;lastLayerRect=null;gs?.load(generation).then(layout).catch(e=>{$('gsStatus').textContent=`3DGS読込失敗: ${e.message}`;});}
    for(const id of ['liveCanvas','mprCanvas','sliceMaskCanvas','anatomyCanvas','markers','featureCanvas']){const c=$(id);if(c.width!==meta.width||c.height!==meta.height){c.width=meta.width;c.height=meta.height;}}
    $('liveViewport').style.aspectRatio=`${meta.width}/${meta.height}`;$('gsViewport').style.aspectRatio=`${meta.width}/${meta.height}`;
    if(bitmaps[0]){$('liveCanvas').getContext('2d').drawImage(bitmaps[0],0,0);bitmaps[0].close();if(meta.input_source==='photo'){photoImageGeneration=meta.generation;photoImageRevision=meta.photo_image_revision;}}
    const layer=$('mprCanvas').getContext('2d');
    if(projective){
        try{mprPlane.draw(layer,bitmaps[1],meta.mpr_quad,meta.layer_id,meta.width,meta.height);}
        catch(e){meta.mpr_status='GPU_UNAVAILABLE';meta.mpr_plane=null;error(e.message);}
        finally{bitmaps[1]?.close();}
        lastLayerRect=null;
    }else{
        mprPlane.reset();
        if(lastLayerRect)layer.clearRect(...lastLayerRect);else layer.clearRect(0,0,meta.width,meta.height);
        if(bitmaps[1]){const rect=meta.layer_rect||[0,0,meta.width,meta.height];layer.drawImage(bitmaps[1],rect[0],rect[1]);bitmaps[1].close();lastLayerRect=rect;}else lastLayerRect=null;
    }
    frame=meta;sequence=meta.sequence;$('empty').hidden=true;drawMarkers(meta);features();layout();telemetry(meta);
    browserBenchmark.drawn(meta,receivedAt,decodedAt,performance.now(),!document.hidden);
}catch(e){browserBenchmark.transportError();error(e.message);await new Promise(resolve=>setTimeout(resolve,1000));}}}
async function stateLoop() {
    while(!shutdown) {
        try {
            const state=await(await fetch('/api/state')).json();
            const idle=['idle','error'].includes(state.status), running=state.status==='running';
            $('notice').textContent=state.detail;
            $('measureSaveStatus').textContent=state.detail?.startsWith('保存しました')?state.detail:'';
            $('start').disabled=!idle||starting||fileSelectionBusy;
            $('stop').disabled=idle||state.status==='stopping';
            for(const key of Object.keys(config.defaults))$(key).disabled=!idle||starting;
            $('referencePreset').disabled=!idle||starting||fileSelectionBusy;
            $('photoFile').disabled=!idle||starting;
            for(const button of [...document.querySelectorAll('[data-picker]'),$('recommendedMap'),$('clearCalibration')])button.disabled=!idle||starting||fileSelectionBusy;
            if(idle&&!starting){const photo=$('input_source').value==='photo';$('camera').disabled=photo;$('max_processing_width').disabled=photo;}
            $('relocalize').disabled=!running||(frame?.input_source==='photo'?Boolean(frame.search?.active):frame?.frozen||config.demo);
            $('diagnostic').disabled=!running||!frame||frame.generation!==state.generation;
            const compareVisible=running&&['gs','registered'].includes(state.mode)&&frame?.frozen&&frame.generation===state.generation;
            const occlusionMode=$('validationMode').value==='occlusion';
            const occlusionFixed=$('occlusionReferencePolicy').value==='original';
            const occlusionResult=state.validation_comparison?.kind==='occlusion';
            const trialName=occlusionResult?'遮蔽後':'面除外';
            $('validationCompare').hidden=!compareVisible;
            $('validationReferencePolicyLabel').hidden=!compareVisible||state.mode!=='gs'||occlusionMode;
            $('occlusionReferencePolicyLabel').hidden=!compareVisible||!occlusionMode;
            const compareBusy=['pending','running'].includes(state.validation_reprocess_status);
            measurementWorkspace.validationBusy=compareBusy;
            $('validationMode').disabled=!compareVisible||compareBusy;
            $('occlusionReferencePolicy').disabled=!compareVisible||compareBusy;
            $('occlusionOriginalViews').textContent=occlusionFixed
                ?`固定参照: ${(frame?.validation_reference_views||[]).join(', ')||'参照なし：全体再照合でPASS後に固定し直してください'}`:'';
            $('occlusionRun').textContent=occlusionFixed?'参照を固定して遮蔽前後を照合':'遮蔽前後を最初から照合';
            $('occlusionPolicyHint').textContent=occlusionFixed
                ?'元PASS時の参照view／登録写真だけを使用。ROI検出・特徴抽出・局所照合・PnPは前後で再実行します。カメラの回転・位置は固定せず、QCも維持します。測定時は元画像を表示。'
                :'照合入力を灰色で遮蔽して、前後を独立に検出・粗検索・PnPします。測定時は元画像を表示。形状変化や内部露出は再現しません。';
            $('occlusionTools').hidden=!compareVisible||!occlusionMode;
            for(const id of ['occlusionDraw','occlusionUndo','occlusionClear'])$(id).disabled=!compareVisible||compareBusy;
            $('occlusionRun').disabled=!compareVisible||compareBusy||measurementWorkspace.occlusionPoints.length<3;
            $('validationPickFace').hidden=occlusionMode;
            $('validationReprocess').hidden=occlusionMode;
            $('validationSelectedFace').hidden=occlusionMode;
            $('validationReprocess').disabled=!compareVisible||compareBusy||!measurementWorkspace.selectedFace;
            $('validationPickFace').disabled=!compareVisible||compareBusy;
            $('validationLimit').disabled=!compareVisible||compareBusy;
            $('validationCoverage').disabled=!compareVisible||compareBusy;
            const rerank=state.mode==='gs'&&$('validationReferencePolicy').value==='rerank';
            $('validationReferencePolicy').disabled=!compareVisible||compareBusy;
            $('validationOriginalViews').textContent=!rerank
                ?`${state.mode==='gs'?'固定view':'固定実写／点検索'}: ${(frame?.validation_reference_views||[]).join(', ')||'元PASS姿勢で比較'}`:'';
            $('validationViewsLabel').hidden=!rerank||occlusionMode;
            $('validationViews').disabled=!compareVisible||compareBusy||!rerank;
            const formatResult=(label,r)=>!r?'':`${label}: ${r.status} · ${r.inliers??'—'}/${r.matches??'—'} 点 · 平均 ${r.mean_reprojection_error==null?'—':r.mean_reprojection_error.toFixed(2)} px`;
            const comparison=state.validation_comparison;
            const completedKey=compareVisible&&state.validation_reprocess_status==='done'&&comparison?.attempt
                ?`${state.generation}:${comparison.attempt}`:null;
            if(completedKey&&completedKey!==lastCompletedComparisonKey){
                $('validationCompare').open=false;
                lastCompletedComparisonKey=completedKey;
            }
            if(!comparison)lastCompletedComparisonKey=null;
            const resultReady=Boolean(comparison?.applied&&frame?.validation_comparison?.attempt===comparison.attempt&&
                frame?.validation_comparison?.applied&&frame.pose&&frame.frozen&&frame.layer_bytes>0);
            $('validationMeasureBar').hidden=!compareVisible||compareBusy||!comparison?.attempt;
            $('validationMeasureResult').disabled=compareBusy||!resultReady;
            $('validationMeasureResult').textContent=`${trialName}の姿勢でずれを測る`;
            const failureReason=validationReasons(comparison?.latest);
            $('validationMeasureStatus').textContent=comparison?.applied
                ?resultReady?`${trialName}の姿勢とCTが表示に反映されました。測定画像は元画像です。`:`${trialName}の姿勢でCTを更新中…`
                :comparison?.attempt?`${trialName}の姿勢は不採用：${comparison.baseline?.status!=='PASS'?'遮蔽なしの再照合もFAILのため比較不能':failureReason}。直前の固定姿勢を維持。`:'';
            $('validationCompareSummary').textContent=comparison?.attempt
                ?`${trialName} ${comparison.face||''}：${comparison.latest?.status||'処理中'}${comparison.applied?' · 姿勢採用':''}（クリックで診断）`
                :occlusionMode?'画像遮蔽の条件・結果（クリックで開閉）':'面を除外して再照合（クリックで開閉）';
            const retrievalText=r=>(r?.retrieval||[]).map(v=>typeof v==='string'?v:v.image_name).join(', ')||'候補なし';
            $('validationCompareText').textContent=compareBusy?state.detail:comparison&&occlusionResult?
                `${formatResult('遮蔽なし',comparison.baseline)} → ${formatResult('遮蔽あり',comparison.latest)} · 全画像に対する遮蔽 ${((comparison.occluded_fraction||0)*100).toFixed(2)}% · ${comparison.reference_policy==='original'?`元PASS参照固定: ${(comparison.reference_views||[]).join(', ')}`:'各画像で粗検索から再実行'}\n遮蔽なし ${Math.round(comparison.baseline?.processing_ms||0)} ms / view: ${retrievalText(comparison.baseline)}\n遮蔽あり ${Math.round(comparison.latest?.processing_ms||0)} ms / view: ${retrievalText(comparison.latest)}`:comparison?
                `${formatResult('全点',comparison.baseline)} → ${formatResult('面除外',comparison.latest)} · 除外 ${comparison.latest?.excluded_face_matches??'—'} 候補 · 誤差上限 ${comparison.max_mean_error_px??'—'} px · 画像分布下限 ${comparison.min_coverage==null?'—':(comparison.min_coverage*100).toFixed(2)+'%'} · ${comparison.reference_policy==='original'?`元PASS view固定: ${(comparison.reference_views||[]).join(', ')}`:comparison.view_count==null?'登録実写':`再検索 Top-${comparison.view_count} view`}${comparison.applied?' · 除外後の姿勢を表示中':' · 元の固定姿勢を表示中'}`:
                occlusionMode?'左画像の任意領域を囲み、選択した参照条件で遮蔽前後を照合してください。':'面選択を押し、左の対象物の面をクリックしてください。';
            const results=$('validationCompare').querySelector('.validation-compare-results');
            const displayedOcclusion=comparison?occlusionResult:occlusionMode;
            const displayedFixed=comparison?comparison.reference_policy==='original':occlusionFixed;
            results.querySelector('b').textContent=displayedOcclusion
                ?displayedFixed?'参照を固定し、遮蔽前後の局所照合・PnPを比較':'実際の入力画像を遮蔽してローカライズから比較'
                :'同じ画像で一面の対応点を除外して比較';
            const note=results.querySelector(':scope > small');
            if(!note.dataset.faceNote)note.dataset.faceNote=note.textContent;
            note.textContent=displayedOcclusion
                ?`灰色領域は実際の照合入力です。${displayedFixed?'参照は元PASS時と同じものに固定し、ROI検出・特徴抽出・PnPは各画像で再実行します。姿勢自体は固定しません。':'ROIとviewは前後で独立に検索します。'}遮蔽域と境界付近の対応点はPnPから除外します。測定画面は元画像に遮蔽後の姿勢でCTを重ねます。切開変形・内部露出は再現しません。`
                :note.dataset.faceNote;
            $('validationMetricCards').children[0].querySelector('b').textContent=occlusionResult?'遮蔽なし':'全点';
            $('validationMetricCards').children[1].querySelector('b').textContent=occlusionResult?'遮蔽あり':'面除外';
            $('validationLatestImage').nextElementSibling.textContent=occlusionResult?'灰：実際の遮蔽入力 ／ 緑：PnP採用 ／ 橙：不採用':'灰：除外面 ／ 緑：PnP採用 ／ 橙：不採用の対応点';
            $('validationMetricCards').hidden=!compareVisible||!comparison?.attempt;
            const stageText=r=>{
                const first=validationStageSummary(r);
                const candidates=r?.diagnosis?.candidate_stages||[];
                if(!candidates.length)return first;
                const count=v=>v==null?'—':v;
                return `${first}\n候補別（生照合 → ${occlusionResult?'遮蔽域除外後':'面除外後'} → 幾何 → XYZ → PnP）:\n`+
                    candidates.map(c=>{
                        const lost=c.xyz_rejections;
                        const reason=lost&&Object.values(lost).some(Number)
                            ?lost.missing_xyz!=null
                                ?`（XYZ未登録 ${lost.missing_xyz} / 重複 ${lost.duplicate||0}）`
                                :`（XYZ除外: 領域外 ${lost.outside||0} / 深度無効 ${lost.invalid_support||0} / 深度境界 ${lost.depth_edge||0}）`:'';
                        const reference=c.reference_features==null?'':`参照 ${c.reference_features}（XYZ付き ${c.reference_xyz_features}） / `;
                        return `${c.reference||'—'}: ${reference}${count(c.raw_matches)} → ${count(c.after_face_filter)} → ${count(c.geometric_inliers)} → ${count(c.xyz_correspondences)} → ${count(c.pnp_inliers)}${reason}`;
                    }).join('\n');
            };
            $('validationBaselineStages').textContent=stageText(comparison?.baseline);
            $('validationLatestStages').textContent=stageText(comparison?.latest);
            const qualityText=r=>!r?'':`採用 ${r.inliers??'—'}/${r.matches??'—'} · 平均 ${r.mean_reprojection_error==null?'—':r.mean_reprojection_error.toFixed(2)} px · 画像分布 ${r.diagnosis?.coverage==null?'—':(r.diagnosis.coverage*100).toFixed(2)+'%'}（下限 ${r.diagnosis?.limits?.min_coverage==null?'—':(r.diagnosis.limits.min_coverage*100).toFixed(2)+'%'}） · P95 ${r.diagnosis?.p95_error==null?'—':r.diagnosis.p95_error.toFixed(2)+' px'}`;
            $('validationBaselineQuality').textContent=qualityText(comparison?.baseline);
            $('validationLatestQuality').textContent=qualityText(comparison?.latest);
            $('validationFailureReason').textContent=failureReason?`FAIL理由：${failureReason}`:'QC：PASS';
            $('validationFailureReason').classList.toggle('validation-fail',Boolean(failureReason));
            $('validationCompareImages').hidden=!compareVisible||!comparison?.attempt;
            const validationKey=`${state.generation}:${comparison?.attempt}`;
            if(compareVisible&&comparison?.attempt&&validationKey!==validationImageKey){
                validationImageKey=validationKey;
                for(const [kind,id] of [['baseline','validationBaselineImage'],['latest','validationLatestImage']])
                    $(id).src=`/api/validation-image?generation=${state.generation}&attempt=${comparison.attempt}&kind=${kind}`;
            }
            const errorKey=`${state.error_revision}:${state.error}`;
            if(state.error&&errorKey!==lastError){error(state.error);lastError=errorKey;}
            $('connection').textContent=`${config.demo&&$('input_source').value!=='photo'?'DEMO · ':''}Python ${state.status}`;
            if(idle){$('cameraLinkedSlice').disabled=true;measurementWorkspace.sync(null);}
            if(idle){frame=null;if(gs)gs.update(null,false);foregroundMask.configure(null,null,null,false);foregroundMask.render();anatomyDemo.clear('模擬データ：停止中');$('sliceMask').disabled=true;$('searchPanel').hidden=true;$('poseBadge').textContent='STOP';$('freeze').disabled=true;$('resume').disabled=true;for(const id of ['slice','opacity','sliceExtended'])$(id).disabled=true;for(const b of document.querySelectorAll('[data-slice-step], [data-slice-percent]'))b.disabled=true;$('sliceStatus').textContent='停止中。開始して姿勢が確定すると断面を操作できます。';}
            if(idle)layout();
        } catch(e) {$('connection').textContent='Python接続切れ';error(e.message);}
        await new Promise(resolve=>setTimeout(resolve,500));
    }
}
window.addEventListener('pagehide',()=>{browserBenchmark.hidden();browserBenchmark.endSession({keepalive:true});});
window.addEventListener('beforeunload',()=>{shutdown=true;clearInterval(clockTimer);clearInterval(benchmarkTimer);});frames();stateLoop();
