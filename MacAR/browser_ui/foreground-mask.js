// The same COLMAP-world bounds as the yellow model box. No PLY, depth map,
// splat render pass or GPU context is needed for this display-only mask.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>{const length=Math.hypot(...v);return length>1e-10?v.map(x=>x/length):null;};
const finite3=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);

export function clipFaces(faces,normal,offset) {
    const result=[],cap=[],distance=p=>dot(normal,p)+offset;
    for(const face of faces) {
        const clipped=[];
        for(let i=0;i<face.length;i++) {
            const a=face[i],b=face[(i+1)%face.length],da=distance(a),db=distance(b);
            if(da<=0)clipped.push(a);
            if((da<=0)!==(db<=0)) {
                const t=da/(da-db),p=a.map((v,j)=>v+t*(b[j]-v));
                clipped.push(p);
                if(!cap.some(q=>Math.hypot(...sub(p,q))<1e-7))cap.push(p);
            }
        }
        if(clipped.length>=3)result.push(clipped);
    }
    if(cap.length>=3) {
        const center=[0,1,2].map(i=>cap.reduce((s,p)=>s+p[i],0)/cap.length);
        const u=unit(cross(normal,Math.abs(normal[0])<.9?[1,0,0]:[0,1,0])),v=cross(normal,u);
        cap.sort((a,b)=>Math.atan2(dot(sub(a,center),v),dot(sub(a,center),u))-Math.atan2(dot(sub(b,center),v),dot(sub(b,center),u)));
        result.push(cap);
    }
    return result;
}

function hull(points) {
    const sorted=points.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const turn=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const half=list=>{const out=[];for(const p of list){while(out.length>=2&&turn(out[out.length-2],out[out.length-1],p)<=0)out.pop();out.push(p);}return out;};
    const low=half(sorted),high=half(sorted.slice().reverse());low.pop();high.pop();return low.concat(high);
}

function clipViewport(polygon,width,height) {
    for(const [axis,limit,sign] of [[0,0,-1],[0,width,1],[1,0,-1],[1,height,1]]) {
        const out=[];
        for(let i=0;i<polygon.length;i++) {
            const a=polygon[i],b=polygon[(i+1)%polygon.length],da=sign*(a[axis]-limit),db=sign*(b[axis]-limit);
            if(da<=0)out.push(a);
            if((da<=0)!==(db<=0)){const t=da/(da-db);out.push(a.map((v,j)=>v+t*(b[j]-v)));}
        }
        polygon=out;
    }
    return polygon;
}

export function projectForegroundBox(pose,plane,bounds) {
    if(!pose||!plane||!finite3(pose.eye)||!finite3(pose.target)||!finite3(pose.up)
        ||!finite3(plane.center)||!finite3(plane.normal)||!Array.isArray(bounds)||bounds.length!==2
        ||!bounds.every(finite3)||bounds[0].some((v,i)=>v>=bounds[1][i])
        ||!Array.isArray(pose.K)||pose.K.length!==3||!pose.K.every(finite3)
        ||!Number.isFinite(pose.width)||!Number.isFinite(pose.height)||pose.width<=0||pose.height<=0)return [];
    const normal=unit(plane.normal),forward=unit(sub(pose.target,pose.eye));
    if(!normal||!forward)return [];
    const right=unit(cross(forward,pose.up));if(!right)return [];
    const down=cross(forward,right),[lo,hi]=bounds;
    const corners=Array.from({length:8},(_,i)=>[i&4?hi[0]:lo[0],i&2?hi[1]:lo[1],i&1?hi[2]:lo[2]]);
    const offset=-dot(normal,plane.center);
    // A plane touching the front face has no foreground volume to mask.
    if(!corners.some(p=>dot(normal,p)+offset < -1e-8))return [];
    let faces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]].map(ids=>ids.map(i=>corners[i]));
    faces=clipFaces(faces,normal,offset); // Keep camera side of the slice.
    const backwards=forward.map(v=>-v);
    faces=clipFaces(faces,backwards,dot(forward,pose.eye)+.1); // Clip behind-camera geometry before division.
    const pixels=faces.flat().map(p=>{
        const delta=sub(p,pose.eye),camera=[dot(delta,right),dot(delta,down),dot(delta,forward)];
        const denominator=dot(pose.K[2],camera);
        return [dot(pose.K[0],camera)/denominator,dot(pose.K[1],camera)/denominator];
    }).filter(p=>p.every(Number.isFinite));
    if(pixels.length<3)return [];
    const polygon=clipViewport(hull(pixels),pose.width,pose.height);
    const area=Math.abs(polygon.reduce((s,p,i)=>{const q=polygon[(i+1)%polygon.length];return s+p[0]*q[1]-p[1]*q[0];},0))*.5;
    return area>1e-5?polygon:[];
}

export class ForegroundMask {
    constructor(canvas) {this.canvas=canvas;}
    clear() {
        this.canvas.getContext('2d').clearRect(0,0,this.canvas.width,this.canvas.height);
        this.canvas.dataset.state='hidden';
    }
    configure(pose,plane,bounds,enabled) {this.pose=pose;this.plane=plane;this.bounds=bounds;this.enabled=Boolean(enabled);}
    render() {
        this.clear();if(!this.enabled)return;
        const polygon=projectForegroundBox(this.pose,this.plane,this.bounds);if(polygon.length<3)return;
        const ctx=this.canvas.getContext('2d');ctx.save();ctx.fillStyle='rgb(128,128,128)';
        ctx.beginPath();polygon.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();ctx.restore();
        this.canvas.dataset.state='foreground';
    }
}
