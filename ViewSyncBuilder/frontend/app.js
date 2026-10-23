import * as THREE from 'three';
import {OrbitControls} from '/vendor/three/OrbitControls.js';
import {PLYLoader} from '/vendor/three/PLYLoader.js';
import {GLTFLoader} from '/vendor/three/GLTFLoader.js';

const $ = id => document.getElementById(id);
const state = {token:'', project:null, points:[], selectedPoint:'', photo:null, photoImage:null,
  ct:null, mpr:null, roiMode:false, roi:[], jobs:[], job:null, preview:null, result:null};
function node(tag, text, attributes={}) { const e=document.createElement(tag); if(text!==null)e.textContent=text; Object.assign(e,attributes); return e; }
function notice(message,error=false){$('notice').hidden=false;$('notice').textContent=message;$('notice').classList.toggle('error',error);}
function on(id,event,callback){$(id).addEventListener(event,async e=>{try{await callback(e);}catch(error){notice(error.message,true);}});}
async function api(path, options={}){
  const headers={'X-ViewSync-Token':state.token,...options.headers};
  if(options.body && !(options.body instanceof FormData)){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  const response=await fetch('/api'+path,{...options,headers});
  if(!response.ok){let value;try{value=await response.json();}catch{value={error:response.statusText};}throw new Error(value.error||JSON.stringify(value.detail)||'処理に失敗しました');}
  return response.json();
}
function page(name){
  for(const element of document.querySelectorAll('.page'))element.hidden=element.id!=='page-'+name;
  for(const button of document.querySelectorAll('nav button'))button.classList.toggle('active',button.dataset.page===name);
  if(name==='result' && state.preview)state.preview.resize();
}
for(const button of document.querySelectorAll('nav button'))button.addEventListener('click',()=>page(button.dataset.page));
function requireProject(){if(!state.project)throw new Error('プロジェクトを作成または開いてください。');}
async function openProject(mode,path){
  const project=await api('/project',{method:'POST',body:{mode,path:path||$('project-path').value,name:$('new-name').value}});
  await applyProject(project);
  notice(mode==='create'?'新しいプロジェクトを作成しました。':'プロジェクトを再開しました。');
}
async function applyProject(project){
  state.project=project;state.job=null;
  state.result=null;if(state.preview){state.preview.dispose();state.preview=null;}$('preview').replaceChildren();
  $('download').hidden=true;$('download').removeAttribute('href');$('qc-metrics').replaceChildren();$('qc-points').replaceChildren();
  $('coordinate-label').textContent='モデル未選択';$('qc-status').textContent='未評価';$('result-layer').replaceChildren(node('option','Sparse',{value:'sparse'}));
  state.points=state.project.points;state.selectedPoint=state.points[0]?.id||'';state.photo=null;state.ct=null;state.mpr=null;
  $('project-name').textContent=state.project.manifest.name;$('project-path').value=state.project.path;
  renderPhotos();renderPoints();await refreshJobs();
  if(state.project.manifest.ct){state.ct=await api('/ct');await configureMPR();}
  else{$('mpr-canvas').getContext('2d').clearRect(0,0,512,512);$('ct-summary').textContent='CT未選択';$('ct-interpolation').replaceChildren();$('mpr-quality').textContent='';}
}
on('create-project','click',()=>openProject('create'));
on('open-project','click',()=>openProject('open'));
on('photo-files','change',async()=>{
  requireProject();const data=new FormData();for(const file of $('photo-files').files)data.append('files',file);
  if(!data.has('files'))return;
  await api('/images',{method:'POST',body:data});state.project=await api('/project');renderPhotos();
  notice('写真を取り込み、原本を保持したままEXIF回転を適用しました。');$('photo-files').value='';
});
function renderPhotos(){
  const images=state.project?.manifest.images||[];
  $('image-count').textContent=`${images.filter(i=>i.enabled).length} / ${images.length}`;
  $('photo-grid').replaceChildren();$('photo-select').replaceChildren(node('option','写真を選択',{value:''}));
  for(const image of images){
    const tile=node('div',null,{className:'photo-tile'+(image.enabled?'':' excluded')});
    const img=node('img',null,{src:'/api/file?path='+encodeURIComponent(image.path),loading:'lazy',alt:image.name});
    img.addEventListener('click',()=>{page('control');$('photo-select').value=image.id;loadPhoto(image.id).catch(e=>notice(e.message,true));});
    const check=node('input',null,{type:'checkbox',checked:image.enabled});
    check.addEventListener('change',async()=>{try{await api('/images/'+image.id,{method:'PATCH',body:{enabled:check.checked}});image.enabled=check.checked;renderPhotos();}catch(e){check.checked=image.enabled;notice(e.message,true);}});
    const label=node('label',null);label.append(check,node('span',image.name));tile.append(img,label,node('div',`${image.width} × ${image.height}`));$('photo-grid').append(tile);
    $('photo-select').append(node('option',image.name+(image.enabled?'':'（除外中）'),{value:image.id}));
  }
  if(state.photo)$('photo-select').value=state.photo.id;
}
on('scan-ct','click',async()=>{
  requireProject();const groups=await api('/ct/series',{method:'POST',body:{path:$('ct-folder').value}});
  $('ct-series').replaceChildren(...groups.map(g=>node('option',`Series ${g.series_number || '—'} · ${g.description || '説明なし'} · ${g.slices} slices · ${g.series_uid}`,{value:g.series_uid})));
  if(!groups.length)notice('CTシリーズが見つかりませんでした。',true);
});
on('load-ct','click',async()=>{
  requireProject();notice('CTシリーズを検証・取り込み中…');
  state.ct=await api('/ct',{method:'POST',body:{path:$('ct-folder').value,series_uid:$('ct-series').value||null}});
  state.project=await api('/project');state.points=state.project.points;renderPoints();await configureMPR();notice('CTを取り込みました。補間の記録とGCP設定の断面を確認できます。');
});
async function configureMPR(){
  if(!state.ct)return;
  const axis={axial:2,coronal:1,sagittal:0}[$('mpr-plane').value],bounds=state.ct.bounds;
  const slider=$('mpr-position');slider.min=bounds[0][axis];slider.max=bounds[1][axis];slider.value=(+slider.min + +slider.max)/2;
  $('ct-summary').textContent=`${state.ct.metadata.dimensions.join(' × ')} voxel · DICOM LPS / mm`;
  const plan=state.ct.metadata.regularization;$('ct-interpolation').replaceChildren();
  if(plan){
    $('ct-interpolation').append(node('p',`原DICOM ${plan.original_slices}枚 → ${plan.output_slices}断面 · 欠損補間 ${plan.interpolated_slices.length}枚 · 間隔 ${plan.spacing_mm.toFixed(3)} mm`));
    if(plan.interpolated_slices.length){
      $('ct-interpolation').append(node('p','HU線形補間。原DICOMは保持しています。補間影響のある点を使う結果のCT精度は未検証です。'));
      const details=node('details',null),list=node('ul',null);details.append(node('summary','補間した断面のLPS位置'));
      for(const item of plan.interpolated_slices)list.append(node('li',`断面 ${item.index} · XYZ (${item.position_lps_mm.map(v=>v.toFixed(3)).join(', ')}) mm · 元断面 ${item.source_indices.join(' / ')}`));
      details.append(list);$('ct-interpolation').append(details);
    }
  }
  await renderMPR();
}
let mprSequence=0;
async function renderMPR(){
  if(!state.ct)return;
  const sequence=++mprSequence;
  const query=new URLSearchParams({plane:$('mpr-plane').value,position:$('mpr-position').value,level:$('mpr-level').value,width:$('mpr-width').value});
  const response=await fetch('/api/ct/mpr?'+query);if(!response.ok)throw new Error((await response.json()).error);
  const mapping=JSON.parse(response.headers.get('X-MPR-Mapping')),bitmap=await createImageBitmap(await response.blob());
  if(sequence!==mprSequence){bitmap.close();return;}
  state.mpr=mapping;const canvas=$('mpr-canvas'),ctx=canvas.getContext('2d');canvas.width=mapping.width;canvas.height=mapping.height;
  ctx.drawImage(bitmap,0,0);bitmap.close();$('mpr-position-label').textContent=(+$('mpr-position').value).toFixed(2)+' mm';
  $('mpr-quality').textContent=mapping.interpolation_affected_pixels
    ? `欠損補間の影響あり：この断面の ${mapping.interpolation_affected_pixels}画素（${(100*mapping.interpolation_affected_pixels/(mapping.width*mapping.height)).toFixed(1)}%）。参照点の補間影響は保存時にも判定します。`
    : 'この断面の画素は欠損補間の影響を受けません。';
  const point=state.points.find(p=>p.id===state.selectedPoint);
  if(point){
    const delta=point.xyz.map((v,i)=>v-mapping.origin[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
    const u=dot(delta,mapping.u)/dot(mapping.u,mapping.u),v=dot(delta,mapping.v)/dot(mapping.v,mapping.v);
    const projected=delta.map((value,i)=>value-u*mapping.u[i]-v*mapping.v[i]);
    if(Math.hypot(...projected)<Math.hypot(...mapping.u)*1.5)marker(ctx,u,v,point.role==='control'?'#ffd166':'#6ee7e7',point.id,1);
  }
}
on('mpr-plane','change',configureMPR);let mprTimer;
on('mpr-position','input',()=>{clearTimeout(mprTimer);mprTimer=setTimeout(()=>renderMPR().catch(e=>notice(e.message,true)),100);});
on('mpr-level','change',renderMPR);on('mpr-width','change',renderMPR);
function canvasUV(event,canvas){const box=canvas.getBoundingClientRect();return[(event.clientX-box.left)*canvas.width/box.width,(event.clientY-box.top)*canvas.height/box.height];}
on('mpr-canvas','click',async event=>{
  if(!state.mpr)throw new Error('CTを読み込んでください。');const point=selectedPoint(),[u,v]=canvasUV(event,$('mpr-canvas'));
  point.xyz=state.mpr.origin.map((a,i)=>a+u*state.mpr.u[i]+v*state.mpr.v[i]);
  point.source={type:'CT_MPR',series_uid:state.ct.metadata.series_uid,frame_of_reference_uid:state.ct.metadata.frame_of_reference_uid};
  await savePoints();await renderMPR();
});
function selectedPoint(){const point=state.points.find(p=>p.id===state.selectedPoint);if(!point)throw new Error('先にGCPまたは評価点を追加・選択してください。');return point;}
async function savePoints(){try{state.points=await api('/points',{method:'POST',body:state.points});}catch(error){state.points=await api('/points');renderPoints();drawPhoto();throw error;}renderPoints();drawPhoto();}
function samplingLabel(sampling){return {original:'元断面から参照',interpolated:'補間影響あり',outside:'CT範囲外',unavailable:'CT未確認'}[sampling?.status]||'CT未確認';}
on('add-point','click',async()=>{
  requireProject();let index=state.points.length+1;while(state.points.some(p=>p.id==='P'+index))index++;
  const point={id:'P'+index,role:'control',xyz:[0,0,0],enabled:true,observations:[]};
  state.points.push(point);state.selectedPoint=point.id;await savePoints();
});
on('delete-point','click',async()=>{const point=selectedPoint();state.points=state.points.filter(p=>p!==point);state.selectedPoint=state.points[0]?.id||'';await savePoints();});
on('point-select','change',async()=>{state.selectedPoint=$('point-select').value;renderPoints();drawPhoto();await renderMPR();});
function renderPoints(){
  $('point-select').replaceChildren(...state.points.map(p=>node('option',`${p.id} · ${p.role==='control'?'拘束':'評価'} · ${p.observations.length}画像`,{value:p.id})));
  $('point-select').value=state.selectedPoint;$('point-table').replaceChildren();
  for(const point of state.points){
    const tr=node('tr',null),cell=element=>{const td=node('td',null);td.append(element);tr.append(td);};
    const enabled=node('input',null,{type:'checkbox',checked:point.enabled});enabled.addEventListener('change',async()=>{point.enabled=enabled.checked;try{await savePoints();}catch(e){notice(e.message,true);}});cell(enabled);
    const id=node('input',null,{value:point.id});id.addEventListener('change',async()=>{const old=point.id;point.id=id.value;try{if(state.selectedPoint===old)state.selectedPoint=point.id;await savePoints();}catch(e){point.id=old;state.selectedPoint=old;renderPoints();notice(e.message,true);}});cell(id);
    const role=node('select',null);role.append(node('option','拘束GCP',{value:'control'}),node('option','独立評価点',{value:'evaluation'}));role.value=point.role;role.addEventListener('change',async()=>{point.role=role.value;try{await savePoints();}catch(e){notice(e.message,true);}});cell(role);
    point.xyz.forEach((value,index)=>{const input=node('input',null,{type:'number',step:'.001',value:value});input.addEventListener('change',async()=>{const old=point.xyz[index];point.xyz[index]=Number(input.value);point.source={type:'manual_CT_LPS_mm'};try{await savePoints();}catch(e){point.xyz[index]=old;renderPoints();notice(e.message,true);}});cell(input);});
    tr.append(node('td',String(point.observations.length)),node('td',samplingLabel(point.ct_sampling)));$('point-table').append(tr);
  }
  const current=state.points.find(p=>p.id===state.selectedPoint);$('observations').replaceChildren();
  $('point-summary').textContent=`拘束 ${state.points.filter(p=>p.enabled&&p.role==='control').length} / 評価 ${state.points.filter(p=>p.enabled&&p.role==='evaluation').length}`;
  if(current)for(const observation of current.observations){
    const image=state.project?.manifest.images.find(i=>i.id===observation.image_id),row=node('div',null,{className:'observation'});
    row.append(node('span',`${image?.name||observation.image_id} · (${observation.uv.map(v=>v.toFixed(1)).join(', ')})`));
    const remove=node('button','削除');remove.addEventListener('click',async()=>{current.observations=current.observations.filter(o=>o!==observation);try{await savePoints();}catch(e){notice(e.message,true);}});row.append(remove);$('observations').append(row);
  }
}
async function loadPhoto(id){
  state.photo=state.project?.manifest.images.find(i=>i.id===id);state.roi=[];state.roiMode=false;
  $('roi-apply').hidden=true;$('roi-mode').textContent='ROIを描く';if(!state.photo)return;
  const image=new Image();image.src='/api/file?path='+encodeURIComponent(state.photo.path);await image.decode();state.photoImage=image;
  $('photo-info').textContent=`${state.photo.name} · ${image.width} × ${image.height}${state.photo.mask?' · ROI設定済み':''}`;drawPhoto();
}
on('photo-select','change',()=>loadPhoto($('photo-select').value));
function marker(ctx,u,v,color,label,scale){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5*scale;ctx.beginPath();ctx.arc(u,v,5*scale,0,Math.PI*2);ctx.moveTo(u-10*scale,v);ctx.lineTo(u+10*scale,v);ctx.moveTo(u,v-10*scale);ctx.lineTo(u,v+10*scale);ctx.stroke();ctx.font=`${12*scale}px sans-serif`;ctx.fillText(label,u+10*scale,v-8*scale);}
function drawPhoto(){
  if(!state.photo||!state.photoImage)return;const canvas=$('photo-canvas'),ctx=canvas.getContext('2d'),image=state.photoImage;
  canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);const scale=image.width/700;
  for(const point of state.points){const observation=point.observations.find(o=>o.image_id===state.photo.id);if(observation)marker(ctx,...observation.uv,point.role==='control'?'#ffd166':'#6ee7e7',point.id,scale);}
  if(state.roi.length){ctx.strokeStyle='#b9ed8e';ctx.lineWidth=2*scale;ctx.beginPath();state.roi.forEach(([u,v],i)=>i?ctx.lineTo(u,v):ctx.moveTo(u,v));ctx.stroke();}
}
on('photo-canvas','click',async event=>{
  if(!state.photo)return;const uv=canvasUV(event,$('photo-canvas'));
  if(state.roiMode){state.roi.push(uv);drawPhoto();return;}
  const point=selectedPoint(),observation=point.observations.find(o=>o.image_id===state.photo.id);
  if(observation)observation.uv=uv;else point.observations.push({image_id:state.photo.id,uv});await savePoints();
});
on('photo-zoom','input',()=>{$('photo-canvas').style.width=$('photo-zoom').value+'%';});
on('roi-mode','click',()=>{if(!state.photo)throw new Error('写真を選択してください。');state.roiMode=!state.roiMode;state.roi=[];$('roi-apply').hidden=!state.roiMode;$('roi-mode').textContent=state.roiMode?'描画をやめる':'ROIを描く';drawPhoto();});
on('roi-apply','click',async()=>{if(state.roi.length<3)throw new Error('3頂点以上を指定してください。');Object.assign(state.photo,await api('/images/'+state.photo.id,{method:'PATCH',body:{roi:state.roi}}));state.roiMode=false;state.roi=[];$('roi-apply').hidden=true;$('roi-mode').textContent='ROIを描く';drawPhoto();notice('ROIマスクを別ファイルで保存しました。');});
on('roi-clear','click',async()=>{if(!state.photo)return;Object.assign(state.photo,await api('/images/'+state.photo.id,{method:'PATCH',body:{roi:null}}));notice('ROIを解除しました。原画像は保持しています。');});
function config(){return{coordinate_mode:$('coordinate-mode').value,surface:$('surface-mode').value,surface_max_size:+$('surface-max-size').value,dense_resolution_level:+$('dense-resolution-level').value,mesh_faces:+$('mesh-faces').value,texture_size:+$('texture-size').value,aliked_model:$('aliked-model').value,max_features:+$('max-features').value,max_image_size:+$('max-image-size').value,pairing:$('pairing').value,overlap:+$('overlap').value,camera_model:$('camera-model').value,camera_params:$('camera-params').value,single_camera:$('single-camera').checked,refine_intrinsics:$('refine-intrinsics').checked,gcp_max_error_px:+$('gcp-max-error').value};}
on('start-job','click',async()=>{requireProject();state.job=await api('/jobs',{method:'POST',body:{settings:config(),stop_after:$('stop-after').value}});await refreshJobs();renderJob(state.job);notice('生成を開始しました。');});
on('cancel-job','click',async()=>{if(state.job){state.job=await api('/jobs/'+state.job.id+'/cancel',{method:'POST'});renderJob(state.job);}});
async function refreshJobs(){
  if(!state.project)return;state.jobs=await api('/jobs');
  for(const id of ['job-select','result-select']){const select=$(id),old=select.value;select.replaceChildren(...state.jobs.map(j=>node('option',`${j.id.slice(0,8)} · ${j.stage} · ${j.status}`,{value:j.id})));if(state.jobs.some(j=>j.id===old))select.value=old;}
  if(state.jobs.length&&!state.job){state.job=state.jobs[0];renderJob(state.job);}
}
function renderJob(job){
  const status={queued:'待機中',running:'実行中',completed:'完了',failed:'失敗',interrupted:'中断・再実行可能',cancelled:'キャンセル済み'};
  $('job-status').textContent=`${status[job.status]||job.status} · ${job.stage}${job.elapsed_seconds?' · '+job.elapsed_seconds.toFixed(1)+'秒':''}`;
  const running=job.status==='running'||job.status==='queued';$('cancel-job').disabled=!running;$('start-job').disabled=running;
  for(const item of $('stages').children){item.classList.toggle('done',!!job.stages[item.dataset.stage]);item.classList.toggle('running',running&&job.stage===item.dataset.stage);}
  $('job-metrics').textContent=[job.processed_images?`抽出 ${job.processed_images}/${job.total_images}画像`:'',job.feature_counts?'特徴点 '+Object.values(job.feature_counts).reduce((a,b)=>a+b,0):'',job.matching?'幾何検証済み '+job.matching.verified_pairs+'ペア':'',job.registered_images!==undefined?`登録 ${job.registered_images}/${job.input_images}画像 · ${job.points3d}点`:'',job.error||''].filter(Boolean).join(' / ');
}
on('job-select','change',async()=>{state.job=await api('/jobs/'+$('job-select').value);renderJob(state.job);await refreshLog();});
on('reuse-settings','click',()=>{const job=state.jobs.find(j=>j.id===$('job-select').value);if(!job)return;
  const mapping={'coordinate_mode':'coordinate-mode','surface':'surface-mode','surface_max_size':'surface-max-size','dense_resolution_level':'dense-resolution-level','mesh_faces':'mesh-faces','texture_size':'texture-size','aliked_model':'aliked-model','max_features':'max-features','max_image_size':'max-image-size','pairing':'pairing','overlap':'overlap','camera_model':'camera-model','camera_params':'camera-params'};
  for(const [key,id] of Object.entries(mapping))if(job.settings[key]!==undefined)$(id).value=job.settings[key];$('single-camera').checked=job.settings.single_camera;$('refine-intrinsics').checked=job.settings.refine_intrinsics;$('stop-after').value=job.stop_after;$('gcp-max-error').value=job.settings.gcp_max_error_px||4;
  notice('実行条件を復元しました。現在の写真・GCPで再実行すると、同じ条件の完了キャッシュを再利用します。');});
async function refreshLog(){if(state.job){const logs=await api('/jobs/'+state.job.id+'/log');$('job-log').textContent=Object.entries(logs).map(([name,text])=>name+'\n'+text).join('\n');}}
setInterval(async()=>{try{if(!state.project)return;if(state.job&&['running','queued'].includes(state.job.status)){const previous=state.job.status;state.job=await api('/jobs/'+state.job.id);renderJob(state.job);await refreshLog();if(state.job.status!==previous&&!['running','queued'].includes(state.job.status)){await refreshJobs();notice(state.job.status==='completed'?'選択した工程が完了しました。結果確認でモデルを開けます。':(state.job.error||'処理を中断しました。'),state.job.status==='failed');}}}catch(e){notice(e.message,true);}},2000);
function metric(label,value,unit=''){const row=node('div',null,{className:'metric'}),strong=node('strong',value===null||value===undefined?'—':typeof value==='number'?value.toFixed(3):String(value));strong.append(node('small',unit));row.append(node('span',label),strong);$('qc-metrics').append(row);}
on('load-result','click',async()=>{
  const runId=$('result-select').value;if(!runId)throw new Error('実行を選択してください。');const result=await api('/results/'+runId);state.result=result;const layers={sparse:'Sparse',dense:'Dense',mesh:'メッシュ',texture:'テクスチャ'};$('result-layer').replaceChildren(...Object.entries(layers).filter(([key])=>key==='sparse'||result.surface?.assets[key]).map(([key,label])=>node('option',label,{value:key})));$('result-layer').value=result.surface?.assets.texture?'texture':'sparse';await showPreview(result);
  $('coordinate-label').textContent=result.coordinate_system==='DICOM_LPS_mm'?'DICOM LPS / mm · 拘束BA後':'初期SfM · 任意座標・任意単位';
  $('download').hidden=!result.run.output;$('download').href='/api/results/'+runId+'/download';$('qc-metrics').replaceChildren();$('qc-points').replaceChildren();
  const qc=result.qc;$('qc-status').textContent=qc?({unverified:'CT位置精度：未検証',within_provisional_target:'この入力では暫定TRE目標内（性能未検証）',failed:'CT位置精度：不合格'}[qc.accuracy_status]||qc.accuracy_status):'初期SfM · CT精度は未検証';
  if(result.surface){metric('Dense点数',result.surface.manifest.dense?.vertices,'点');metric('メッシュ面数',result.surface.manifest.mesh?.faces,'面');}metric('登録成功率',result.run.registration_rate*100,'%');metric('Sparse点数',result.xyz.length,'点');
  if(qc){metric('独立評価 TRE 平均',qc.tre_mm.mean,'mm');metric('TRE 標準偏差',qc.tre_mm.std,'mm');metric('TRE 中央値',qc.tre_mm.median,'mm');metric('TRE 最大',qc.tre_mm.maximum,'mm');metric('評価失敗数',qc.tre_mm.failures,'件');if(qc.bundle_adjustment){metric('全観測 再投影誤差',qc.bundle_adjustment.after_reprojection_px.mean,'px');metric('GCP 再投影誤差',qc.bundle_adjustment.gcp_reprojection_px.mean,'px');}
    if(qc.original_reference_tre_mm){metric('元断面参照 TRE 平均',qc.original_reference_tre_mm.mean,'mm');metric('補間影響あり TRE 平均',qc.interpolated_reference_tre_mm.mean,'mm');}
    if(qc.reference_sampling_uncertain)$('qc-metrics').append(node('p','CT参照に補間影響・範囲外・未確認の点があるため、物理精度は未検証です。',{className:'hint'}));
    const table=node('table',null),head=node('tr',null);for(const title of ['評価点','TRE / mm','CT投影誤差 / px','CT参照','状態'])head.append(node('th',title));table.append(head);
    for(const point of qc.points){const row=node('tr',null);for(const text of [point.id,point.tre_mm?.toFixed(3)||'—',point.ct_reprojection_px?.mean?.toFixed(3)||'—',samplingLabel(point.ct_sampling),point.status==='measured'?'測定済み':point.reason])row.append(node('td',text));table.append(row);}$('qc-points').append(table);
  }
});
async function showPreview(result){
  const layer=$('result-layer').value;let asset;
  if(layer==='texture'){
    asset=(await new GLTFLoader().loadAsync(result.surface.assets.texture)).scene;
    const inverse=result.surface.manifest.source_from_glb;
    if(inverse)asset.applyMatrix4(new THREE.Matrix4().fromArray(inverse));
  }else{
    let geometry;
    if(layer==='sparse'){
      geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(result.xyz.flat(),3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(result.rgb.flat().map(v=>v/255),3));
    }else geometry=await new PLYLoader().loadAsync(result.surface.assets[layer]);
    geometry.computeBoundingSphere();
    if(layer==='mesh'){geometry.computeVertexNormals();asset=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:'#bbcabd',roughness:.8,side:THREE.DoubleSide}));}
    else asset=new THREE.Points(geometry,new THREE.PointsMaterial({size:Math.max(geometry.boundingSphere.radius*.003,.00001),vertexColors:!!geometry.attributes.color,color:geometry.attributes.color?'#ffffff':'#bdd5ca'}));
  }
  if(state.preview)state.preview.dispose();const host=$('preview');host.replaceChildren();
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.append(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color('#12221c');scene.add(asset,new THREE.HemisphereLight('#ffffff','#4b524b',2));
  const light=new THREE.DirectionalLight('#ffffff',2);light.position.set(1,-1,2);scene.add(light);
  const camera=new THREE.PerspectiveCamera(50,1,.01,100000);camera.up.set(0,0,1);
  const bounds=new THREE.Box3().setFromObject(asset).getBoundingSphere(new THREE.Sphere()),radius=Math.max(bounds.radius,1e-3);
  const cameras=new THREE.Group(),controlPoints=new THREE.Group();scene.add(cameras,controlPoints);
  for(const item of result.cameras){const center=new THREE.Vector3(...item.center),r=new THREE.Matrix3().set(...item.rotation).transpose();const corners=[[-1,-.7,1.5],[1,-.7,1.5],[1,.7,1.5],[-1,.7,1.5]].map(c=>new THREE.Vector3(...c).multiplyScalar(radius*.04).applyMatrix3(r).add(center));const vertices=[];for(let i=0;i<4;i++)vertices.push(...center.toArray(),...corners[i].toArray(),...corners[i].toArray(),...corners[(i+1)%4].toArray());const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));cameras.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:'#81b6a4'})));}
  if(result.coordinate_system==='DICOM_LPS_mm')for(const point of result.points.filter(p=>p.enabled)){const mesh=new THREE.Mesh(new THREE.SphereGeometry(radius*.012,12,8),new THREE.MeshBasicMaterial({color:point.role==='control'?'#ffd166':'#6ee7e7'}));mesh.position.set(...point.xyz);controlPoints.add(mesh);}
  const axes=new THREE.AxesHelper(radius*.25);scene.add(axes);const orbit=new OrbitControls(camera,renderer.domElement);orbit.target.copy(bounds.center);camera.position.copy(bounds.center).add(new THREE.Vector3(radius*1.5,-radius*1.8,radius*1.2));camera.near=radius/1000;camera.far=radius*100;camera.updateProjectionMatrix();orbit.update();
  const render=()=>renderer.render(scene,camera);const resize=()=>{const w=host.clientWidth,h=host.clientHeight;if(w){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();render();}};orbit.addEventListener('change',render);const observer=new ResizeObserver(resize);observer.observe(host);resize();
  state.preview={resize,dispose(){observer.disconnect();orbit.dispose();scene.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material]).filter(Boolean)){for(const value of Object.values(m))if(value?.isTexture)value.dispose();m.dispose();}});renderer.dispose();},setVisibility(){cameras.visible=$('show-cameras').checked;controlPoints.visible=$('show-points').checked;render();}};state.preview.setVisibility();
}
on('result-layer','change',async()=>{if(state.result)await showPreview(state.result);});
on('show-cameras','change',()=>state.preview?.setVisibility());on('show-points','change',()=>state.preview?.setVisibility());
async function boot(){
  const session=await api('/session');state.token=session.token;$('project-path').value=session.default_projects+'/Project001';
  if(session.recent.length){$('recent-projects').replaceChildren(...session.recent.map(path=>{const button=node('button',path);button.addEventListener('click',()=>openProject('open',path).catch(e=>notice(e.message,true)));return button;}));}
  const caps=await api('/capabilities');$('engine-state').textContent=caps.aliked&&caps.lightglue?'● ALIKED + LightGlue / CPU':'● エンジンの準備が必要';
  $('capabilities').replaceChildren(node('p',caps.engine_version||'COLMAP未導入'),node('p',`N16ROT: ${caps.models['aliked-n16rot.onnx']?'準備済み':'未導入'} / N32: ${caps.models['aliked-n32.onnx']?'準備済み':'未導入'} / LightGlue: ${caps.models['aliked-lightglue.onnx']?'準備済み':'未導入'}`),node('p',`Dense / Mesh / Texture：${caps.texture?(caps.surface_engine+' 準備済み'):'エンジン未導入'}。検出スコア・対応信頼度：標準COLMAP DBでは取得不可。`));
  const current=await api('/project');if(current)await applyProject(current);
}
boot().catch(error=>notice(error.message,true));
