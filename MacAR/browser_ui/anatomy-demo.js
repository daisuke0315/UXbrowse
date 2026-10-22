// Display-only synthetic anatomy in the same model/world coordinates as PnP.
// Never used as observations, CT voxels, depth, segmentation, or measurements.
import {clipFaces} from './foreground-mask.js';

const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>{const n=Math.hypot(...v);return n>1e-10?v.map(x=>x/n):null;};
const finite3=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const validBounds=b=>Array.isArray(b)&&b.length===2&&b.every(finite3)&&b[0].every((v,i)=>v<b[1][i]);

function ellipsoid(center,radii) {
    const rings=12,sectors=24,faces=[];
    const p=(i,j)=>{const t=Math.PI*i/rings,a=2*Math.PI*j/sectors;
        return center.map((c,k)=>c+radii[k]*[Math.sin(t)*Math.cos(a),Math.sin(t)*Math.sin(a),Math.cos(t)][k]);};
    for(let i=0;i<rings;i++)for(let j=0;j<sectors;j++){
        // Remove duplicate pole vertices so normal/plane clipping stays stable.
        if(i===0)faces.push([p(0,j),p(1,j),p(1,j+1)]);
        else if(i===rings-1)faces.push([p(i,j),p(rings,j),p(i,j+1)]);
        else faces.push([p(i,j),p(i+1,j),p(i+1,j+1),p(i,j+1)]);
    }
    return faces;
}

function cylinder(a,b,radius) {
    const n=unit(sub(b,a)),u=unit(cross(n,Math.abs(n[0])<.9?[1,0,0]:[0,1,0])),v=cross(n,u);
    const count=12,ring=c=>Array.from({length:count},(_,i)=>c.map((x,j)=>x+radius*(u[j]*Math.cos(i*2*Math.PI/count)+v[j]*Math.sin(i*2*Math.PI/count))));
    const r=ring(a),s=ring(b),faces=[r.slice().reverse(),s];
    for(let i=0;i<count;i++){const j=(i+1)%count;faces.push([r[i],r[j],s[j],s[i]]);}
    return faces;
}

export function buildAnatomy(bounds) {
    if(!validBounds(bounds))return [];
    const [lo,hi]=bounds,size=hi.map((v,i)=>v-lo[i]),scale=Math.min(...size);
    const at=q=>q.map((v,i)=>lo[i]+v*size[i]),meshes=[];
    const tumor=(q,r)=>meshes.push({kind:'tumor',color:[246,157,49],faces:ellipsoid(at(q),r.map(v=>v*scale))});
    tumor([.55,.50,.53],[.16,.13,.17]);
    tumor([.30,.66,.35],[.07,.065,.075]);
    const vessel=(kind,path,radius)=>{
        const color=kind==='artery'?[236,69,83]:[64,159,244];
        for(let i=1;i<path.length;i++)meshes.push({kind,color,faces:cylinder(at(path[i-1]),at(path[i]),radius*scale)});
    };
    vessel('artery',[[.20,.30,.12],[.27,.35,.32],[.38,.39,.51],[.43,.42,.69],[.50,.44,.87]],.025);
    vessel('artery',[[.38,.39,.51],[.57,.30,.62],[.74,.25,.79]],.018);
    vessel('artery',[[.27,.35,.32],[.43,.23,.33],[.65,.16,.42]],.014);
    vessel('artery',[[.43,.42,.69],[.27,.58,.76],[.18,.69,.85]],.016);
    vessel('vein',[[.78,.73,.14],[.70,.66,.33],[.69,.63,.51],[.65,.60,.70],[.73,.56,.88]],.032);
    vessel('vein',[[.70,.66,.33],[.53,.75,.43],[.36,.83,.62]],.019);
    vessel('vein',[[.65,.60,.70],[.77,.77,.77],[.86,.83,.87]],.021);
    return meshes;
}

// Produce camera-projected, depth-sorted polygons; the source geometry is immutable.
export function anatomyPolygons(meshes,pose,plane,mode,visible={tumor:true,artery:true,vein:true}) {
    if(!pose||!finite3(pose.eye)||!finite3(pose.target)||!finite3(pose.up)
        ||!Array.isArray(pose.K)||pose.K.length!==3||!pose.K.every(finite3)
        ||!Number.isFinite(pose.width)||!Number.isFinite(pose.height)||pose.width<=0||pose.height<=0
        ||!['transparent','cutaway','section'].includes(mode))return [];
    const forward=unit(sub(pose.target,pose.eye));if(!forward)return [];
    const right=unit(cross(forward,pose.up));if(!right)return [];
    const down=cross(forward,right),normal=finite3(plane?.normal)?unit(plane.normal):null;
    if(mode!=='transparent'&&(!normal||!finite3(plane?.center)))return [];
    const out=[],eps=1e-6;
    for(const mesh of meshes) {
        if(!visible[mesh.kind])continue;
        let faces=mesh.faces;
        if(mode!=='transparent') {
            // Remove the CAMERA SIDE: keep geometry beyond the current CT plane.
            const n=normal.map(v=>-v),offset=dot(normal,plane.center);
            faces=clipFaces(faces,n,offset);
            if(mode==='section')faces=faces.filter(face=>face.every(p=>Math.abs(dot(normal,sub(p,plane.center)))<eps));
        }
        const near=forward.map(v=>-v);
        faces=clipFaces(faces,near,dot(forward,pose.eye)+.1);
        for(const face of faces) {
            const camera=face.map(p=>{const d=sub(p,pose.eye);return [dot(d,right),dot(d,down),dot(d,forward)];});
            const pixels=camera.map(p=>{const z=dot(pose.K[2],p);return [dot(pose.K[0],p)/z,dot(pose.K[1],p)/z];});
            if(!pixels.every(p=>p.every(Number.isFinite)))continue;
            const cap=mode!=='transparent'&&face.every(p=>Math.abs(dot(normal,sub(p,plane.center)))<eps);
            const faceNormal=unit(cross(sub(face[1],face[0]),sub(face[2],face[0])));
            if(!faceNormal)continue;
            const light=cap?1:.48+.52*Math.abs(dot(faceNormal,unit([-.3,-.5,-1])));
            out.push({pixels,depth:camera.reduce((s,p)=>s+p[2],0)/camera.length,
                color:mesh.color.map(v=>Math.round(v*light)),cap,kind:mesh.kind});
        }
    }
    return out.sort((a,b)=>b.depth-a.depth);
}

export class AnatomyDemo {
    constructor(canvas,status) {
        this.canvas=canvas;this.status=status;this.geometryKey='';this.meshes=[];
        // Composite opacity ONCE, avoiding translucent tessellation seams.
        this.surface=document.createElement('canvas');
    }
    clear(message='模擬データ：非表示') {
        this.canvas.getContext('2d').clearRect(0,0,this.canvas.width,this.canvas.height);
        this.canvas.dataset.state='hidden';this.status.textContent=message;
    }
    render(pose,plane,bounds,settings) {
        this.clear();if(settings.mode==='off')return;
        if(!pose){this.clear('模擬データ：姿勢確定待ち（ロスト中は表示しません）');return;}
        if(!validBounds(bounds)){this.clear('模擬データ：対象の3D範囲がありません');return;}
        if(settings.mode!=='transparent'&&(!finite3(plane?.center)||!finite3(plane?.normal)||!unit(plane.normal))){
            this.clear('模擬データ：MPR断面待ち。半透明3DはCTなしでも表示できます');return;
        }
        const key=JSON.stringify(bounds);
        if(key!==this.geometryKey){this.meshes=buildAnatomy(bounds);this.geometryKey=key;}
        const start=performance.now(),polygons=anatomyPolygons(this.meshes,pose,plane,settings.mode,settings);
        const surface=this.surface;
        if(surface.width!==this.canvas.width||surface.height!==this.canvas.height){surface.width=this.canvas.width;surface.height=this.canvas.height;}
        const ctx=surface.getContext('2d');ctx.clearRect(0,0,surface.width,surface.height);
        for(const face of polygons) {
            ctx.fillStyle=`rgb(${face.color.join(',')})`;ctx.beginPath();
            face.pixels.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();
            if(face.cap){ctx.strokeStyle='rgba(255,237,184,.9)';ctx.lineWidth=1.5;ctx.stroke();}
        }
        const target=this.canvas.getContext('2d');target.save();
        target.globalAlpha=Math.max(0,Math.min(1,Number(settings.opacity)/100));target.drawImage(surface,0,0);target.restore();
        // Label is independent of model opacity and always identifies fictional data.
        target.fillStyle='rgba(15,19,29,.8)';target.fillRect(10,this.canvas.height-34,240,24);
        target.fillStyle='#ffcd81';target.font='14px system-ui';target.fillText('DEMO · 模擬腫瘍 / 血管（架空）',18,this.canvas.height-17);
        const ms=performance.now()-start;
        this.canvas.dataset.state=polygons.length?'visible':'empty';this.canvas.dataset.renderMs=ms.toFixed(2);
        const name={transparent:'半透明3D',cutaway:'手前を切り取る3D',section:'断面のみ'}[settings.mode];
        this.status.textContent=`模擬データ · ${name} · ${ms.toFixed(1)} ms${polygons.length?'':' · この断面／視野にはありません'}`;
    }
}
