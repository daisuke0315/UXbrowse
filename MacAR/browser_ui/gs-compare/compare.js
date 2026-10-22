import {Application, Asset, AssetListLoader, Color, Entity, Vec3, RESOLUTION_FIXED,
    createGraphicsDevice, GSPLAT_RENDERER_RASTER_GPU_SORT} from '/playcanvas.mjs';
import {profiles, referencePose} from './profiles.js';

const $=id=>document.getElementById(id), config=await (await fetch('/config.json')).json();
let baseline=null, candidate=null, serial=0, rotating=false, exactPose=null,lastTime=performance.now();
const state={azimuth:30,elevation:20,distance:380};
const gainShader=`uniform float opacityGain;
void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}
void modifySplatColor(vec3 center, inout vec4 color) {color.a = min(1.0, color.a * opacityGain);}`;

class ComparisonView {
    async initialize(canvas,status,profile) {
        this.status=status;this.canvas=canvas;this.profile=profile;this.fps=0;this.settled=false;
        canvas.width=config.width;canvas.height=config.height;
        const device=await createGraphicsDevice(canvas,{deviceTypes:[profile.backend],antialias:false,alpha:false,powerPreference:'high-performance'});
        if(device.deviceType==='null')throw Error('このブラウザではGPU描画を開始できません');
        this.app=new Application(canvas,{graphicsDevice:device});
        this.app.setCanvasResolution(RESOLUTION_FIXED,config.width,config.height);
        this.app.autoRender=false;
        this.camera=new Entity('camera');
        this.camera.addComponent('camera',{clearColor:new Color(.09,.12,.16,1),nearClip:.1,farClip:100000,fov:config.fov});
        this.app.root.addChild(this.camera);
        const p=this.app.scene.gsplat;
        p.minPixelSize=profile.minPixelSize;p.minContribution=profile.minContribution;p.dataFormat=profile.format;
        if(profile.backend==='webgpu')p.renderer=GSPLAT_RENDERER_RASTER_GPU_SORT;
        if(profile.gain!==1){p.material.getShaderChunks('glsl').set('gsplatModifyVS',gainShader);p.material.setParameter('opacityGain',profile.gain);p.material.update();}
        this.app.systems.gsplat.on('frame:request',()=>{this.app.renderNextFrame=true;});
        this.app.systems.gsplat.on('frame:ready',(camera,layer,ready,loading)=>{this.settled=ready&&loading===0;});
        let count=0,start=performance.now(),last=0;
        this.app.on('update',()=>{const now=performance.now();if(now-last>=1000/30-1)this.app.renderNextFrame=true;});
        this.app.on('postrender',()=>{count++;last=performance.now();if(last-start>=1000){this.fps=count*1000/(last-start);count=0;start=last;this.report();}});
        this.app.start();
        status.textContent='PLY読込中…';
        this.asset=new Asset('same-model','gsplat',{url:'/asset.ply',filename:'splat.ply'});
        await new Promise((resolve,reject)=>new AssetListLoader([this.asset],this.app.assets).load(error=>error?reject(Error(String(error))):resolve()));
        this.entity=new Entity('same-model');this.entity.addComponent('gsplat',{asset:this.asset,unified:true});this.app.root.addChild(this.entity);
        this.pose();this.report();return this;
    }
    pose(){
        if(!this.app)return;
        const a=state.azimuth*Math.PI/180,e=state.elevation*Math.PI/180,r=state.distance,c=config.center;
        const next=exactPose||{eye:[c[0]+r*Math.cos(e)*Math.sin(a),c[1]+r*Math.cos(e)*Math.cos(a),c[2]+config.up[2]*r*Math.sin(e)],target:c,up:config.up};
        this.camera.setPosition(...next.eye);
        this.camera.lookAt(new Vec3(...next.target),new Vec3(...next.up));
        this.settled=false;this.app.renderNextFrame=true;
    }
    report(){
        const gpu=this.app.graphicsDevice.isWebGPU;
        const unavailable=this.profile.backend==='webgpu'&&!gpu?' · WebGPU非対応：比較不成立':'';
        this.status.textContent=`${gpu?'WebGPU / GPU sort':'WebGL2 / CPU sort'} · ${this.fps.toFixed(1)} fps · ${this.settled?'sort完了':'sort更新待ち'}${unavailable}`;
    }
    capture(){
        return new Promise((resolve,reject)=>{
            let frames=0;
            const timeout=setTimeout(()=>{this.app.off('postrender',sample);reject(Error('sort完了待ちがタイムアウト'));},15000);
            const sample=()=>{
                if(++frames<8||!this.settled)return;
                this.app.off('postrender',sample);clearTimeout(timeout);
                // Request the snapshot while the framebuffer is still valid.
                this.canvas.toBlob(blob=>blob?resolve(blob):reject(Error('PNG取得失敗')),'image/png');
            };
            this.app.on('postrender',sample);this.app.renderNextFrame=true;
        });
    }
    destroy(){if(this.asset){this.app.assets.remove(this.asset);this.asset.unload();}this.app?.destroy();}
}

function pose(){for(const name of ['azimuth','elevation','distance']){$(name).value=state[name];$(name+'Value').textContent=Math.round(state[name])+(name==='distance'?' mm':'°');}baseline?.pose();candidate?.pose();}
function view(){const v=config.views[Number($('view').value)];if(!v)return;exactPose=referencePose(config,v);const d=v.eye.map((x,i)=>x-config.center[i]),r=Math.hypot(...d);state.azimuth=Math.atan2(d[0],d[1])*180/Math.PI;state.elevation=Math.asin(config.up[2]*d[2]/r)*180/Math.PI;state.distance=r;pose();}
async function changeProfile(){
    const token=++serial,profile=profiles[$('profile').value];candidate?.destroy();candidate=null;
    // A graphics context cannot change between WebGL and WebGPU: replace canvas.
    const canvas=document.createElement('canvas');canvas.id='candidate';$('candidate').replaceWith(canvas);attachDrag(canvas);
    $('candidateHeading').textContent='B · '+profile.label;
    $('note').textContent=profile.gain!==1?'alpha増強は表示比較です。モデルのopacityを描画時に変更するため、実表面／精度の改善を意味しません。この画面の条件切替だけでは検索用の参照RGBは変わりません。':'静止後の透けと回転中だけの乱れを分けて比較してください。元のalpha合成は変更していません。';
    const next=new ComparisonView();
    try{await next.initialize(canvas,$('candidateStatus'),profile);if(token!==serial)next.destroy();else candidate=next;}
    catch(error){next.destroy();$('error').textContent=String(error);}
}
function attachDrag(canvas){let drag=null;canvas.onpointerdown=event=>{drag=[event.clientX,event.clientY];canvas.setPointerCapture(event.pointerId);};canvas.onpointermove=event=>{if(!drag)return;exactPose=null;state.azimuth=(state.azimuth+(event.clientX-drag[0])*.3+540)%360-180;state.elevation=Math.max(-10,Math.min(89,state.elevation+(event.clientY-drag[1])*.3));drag=[event.clientX,event.clientY];pose();};canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};}
for(const [key,p] of Object.entries(profiles))$('profile').add(new Option(p.label,key));
config.views.forEach((v,i)=>$('view').add(new Option(`${v.image_name||'view '+i} · 方位${v.azimuth_deg??'?'}° 上方${v.elevation_deg??'?'}°${v.roll_deg!==undefined?' roll '+v.roll_deg+'°':''}`,i)));
$('profile').value='detail';
for(const name of ['azimuth','elevation','distance'])$(name).oninput=()=>{exactPose=null;state[name]=Number($(name).value);pose();};
$('view').onchange=view;$('reset').onclick=view;$('profile').onchange=changeProfile;
$('rotate').onclick=()=>{rotating=!rotating;$('rotate').textContent=rotating?'停止して比較':'回転して比較';};
$('source').textContent=`PLY: ${config.ply} | 座標変換なし | 上方: ${config.up.join(', ')} | ${config.width}×${config.height}`;
if(config.capture){
    $('capturePanel').hidden=false;
    $('capture').onclick=async()=>{
        $('capture').disabled=true;rotating=false;
        for(const input of document.querySelectorAll('#controls input,#controls select,#controls button'))input.disabled=true;
        try{
            $('profile').value='alpha15';await changeProfile();
            if(!baseline||!candidate)throw Error('両方の描画が必要です');
            for(const renderer of [baseline,candidate])renderer.camera.camera.clearColor=new Color(.96,.96,.96,1);
            const headers={'X-Capture-Token':config.capture.token};
            for(let i=0;i<config.views.length;i++){
                $('captureProgress').textContent=`参照RGB生成 ${i+1}/${config.views.length} · alpha 1.0 / 1.5（XYZ固定）`;
                exactPose=referencePose(config,config.views[i]);baseline.pose();candidate.pose();
                const blobs=await Promise.all([baseline.capture(),candidate.capture()]);
                for(const [j,key] of ['alpha10','alpha15'].entries()){
                    const result=await fetch(`/capture/${key}/${i}`,{method:'POST',headers,body:blobs[j]});
                    if(!result.ok)throw Error(await result.text());
                }
            }
            $('captureProgress').textContent='全RGB保存済み · XYZのコピーとmanifest確定中…';
            const response=await fetch('/capture/finish',{method:'POST',headers});
            if(!response.ok)throw Error(await response.text());
            const result=await response.json();
            $('captureProgress').textContent='生成完了：'+result.outputs.join(' / ');
            $('note').textContent='alpha 1.0対照と1.5参照を別フォルダへ生成しました。元のRGB・XYZは保持しています。照合比較前なので精度改善は未判定です。';
        }catch(error){$('captureProgress').textContent='生成失敗：'+String(error);}
        finally{
            for(const input of document.querySelectorAll('#controls input,#controls select,#controls button'))input.disabled=false;
            view();
        }
    };
}
attachDrag($('baseline'));
try{baseline=await new ComparisonView().initialize($('baseline'),$('baselineStatus'),profiles.default);view();await changeProfile();}
catch(error){$('error').textContent=String(error);}
function animate(now){const dt=Math.min(.1,(now-lastTime)/1000);lastTime=now;if(rotating){exactPose=null;state.azimuth=(state.azimuth+dt*18+180)%360-180;pose();}requestAnimationFrame(animate);}
requestAnimationFrame(animate);
