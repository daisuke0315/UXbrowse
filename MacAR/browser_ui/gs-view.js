import {Application, Asset, AssetListLoader, Entity, Color, Vec3, RESOLUTION_FIXED} from '/playcanvas.mjs';
import {intrinsicProjection} from './projection.js';

// OpenCV (right/down/forward) -> PlayCanvas (right/up/backward).
// This uses the full K, including principal point and unequal fx/fy, not just FOV.

export class GaussianView {
    constructor(canvas, status) {
        this.canvas=canvas;this.status=status;this.generation=null;this.ready=false;
        this.app=new Application(canvas,{graphicsDeviceOptions:{antialias:false,alpha:true,powerPreference:'high-performance'}});
        this.app.autoRender=false;
        this.camera=new Entity('camera');
        this.camera.addComponent('camera',{clearColor:new Color(0,0,0,0),nearClip:.1,farClip:100000});
        this.app.root.addChild(this.camera);
        this.projection=null;
        this.camera.camera.calculateProjection=(matrix)=>{if(this.projection)matrix.set(this.projection);};
        this.lastRenderAt=0;this.renderCount=0;
        this.app.on('update',()=>{const now=performance.now();if(this.active&&now-this.lastRenderAt>=1000/30-1){this.app.renderNextFrame=true;}});
        // frameend fires even when rendering is skipped; count actual renders.
        this.app.on('postrender',()=>{this.renderCount++;this.lastRenderAt=performance.now();});
        this.app.start();
        this.orbit=false;this.orbitState=null;
        let drag=null;
        canvas.addEventListener('pointerdown',event=>{if(!this.orbit)return;drag=[event.clientX,event.clientY];canvas.setPointerCapture(event.pointerId);});
        canvas.addEventListener('pointermove',event=>{if(!drag||!this.orbitState)return;this.orbitState.azimuth+=(event.clientX-drag[0])*.008;this.orbitState.elevation=Math.max(-1.45,Math.min(1.45,this.orbitState.elevation+(event.clientY-drag[1])*.008));drag=[event.clientX,event.clientY];this.applyOrbit();});
        canvas.addEventListener('pointerup',()=>{drag=null;});canvas.addEventListener('pointercancel',()=>{drag=null;});
        canvas.addEventListener('wheel',event=>{if(!this.orbitState||!this.orbit)return;event.preventDefault();this.orbitState.distance=Math.max(1,Math.min(100000,this.orbitState.distance*Math.exp(event.deltaY*.001)));this.applyOrbit();},{passive:false});
    }
    async load(generation) {
        if(this.generation===generation)return;
        this.generation=generation;this.ready=false;this.canvas.style.opacity='0';
        this.status.textContent='3DGS読込中…';
        if(this.entity){this.entity.destroy();this.entity=null;}
        if(this.asset){this.app.assets.remove(this.asset);this.asset.unload();}
        const asset=new Asset('reference','gsplat',{url:`/asset.ply?generation=${generation}`,filename:'splat.ply'});
        this.asset=asset;
        await new Promise((resolve,reject)=>{
            const loader=new AssetListLoader([asset],this.app.assets);
            loader.load(error=>error?reject(Error(String(error))):resolve());
        });
        if(this.generation!==generation){asset.unload();return;}
        this.entity=new Entity('patient-reference');this.entity.addComponent('gsplat',{asset,unified:true});
        this.app.root.addChild(this.entity);this.ready=true;this.status.textContent='GPU · 30 fps目標';
    }
    update(pose, show, opacity=1) {
        this.active=show&&this.ready;
        this.canvas.style.opacity=show&&this.ready?String(opacity):'0';
        if(!pose)return;
        if(this.canvas.width!==pose.width||this.canvas.height!==pose.height)this.app.setCanvasResolution(RESOLUTION_FIXED,pose.width,pose.height);
        this.projection=intrinsicProjection(pose.K,pose.width,pose.height);
        if(this.orbit&&this.orbitState)this.applyOrbit();
        else {this.camera.setPosition(...pose.eye);this.camera.lookAt(new Vec3(...pose.target),new Vec3(...pose.up));}
        // Invalidate the cached projection after changing K/resolution.
        this.camera.camera.fov=2*Math.atan(pose.height/(2*pose.K[1][1]))*180/Math.PI;
        this.camera.camera.camera._projMatDirty=true;
    }
    setOrbit(enabled, pose, center, worldUp) {
        if(enabled&&(!this.orbit||!this.orbitState)&&pose&&center){
            const d=pose.eye.map((value,i)=>value-center[i]),distance=Math.hypot(...d),sign=worldUp[2];
            this.orbitState={center,sign,distance,azimuth:Math.atan2(d[1],d[0]),elevation:Math.asin(sign*d[2]/distance)};
        }
        this.orbit=enabled;
    }
    commit() {
        // Commit the new pose before the browser paints the new image. A RAF
        // running at a separate frequency could otherwise retain the old GS.
        if(this.ready&&this.active){this.app.render();this.app.renderNextFrame=false;}
    }
    applyOrbit(){const p=this.orbitState;if(!p)return;this.camera.setPosition(p.center[0]+p.distance*Math.cos(p.elevation)*Math.cos(p.azimuth),p.center[1]+p.distance*Math.cos(p.elevation)*Math.sin(p.azimuth),p.center[2]+p.sign*p.distance*Math.sin(p.elevation));this.camera.lookAt(new Vec3(...p.center),new Vec3(0,0,p.sign));}
}
