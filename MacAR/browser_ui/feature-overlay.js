// All points belong to the packet's source image, and are rectified by Python.
// Green is an accepted PnP inlier, not a certified physical landmark identity.
export function pointCounts(data) {
    if (!data) return {total:0,inliers:0,rejected:0};
    const total=data.observed.length, inliers=data.inliers.filter(Boolean).length;
    return {total,inliers: data.accepted?inliers:0,rejected:data.accepted?total-inliers:total};
}

export function drawFeatures(canvas, data, settings) {
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (!data) return;
    // Keep dots/labels legible when a 1280px frame is shown in a small pane.
    const scale=canvas.width/(canvas.getBoundingClientRect?.().width||canvas.width);
    const radius=Number(settings.pointSize)*scale, points=data.observed;
    const visible=([x,y])=>Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&y>=0&&x<canvas.width&&y<canvas.height;
    for(let i=0;i<points.length;i++) {
        const uv=points[i], accepted=data.accepted&&data.inliers[i];
        if(!(accepted?settings.showInliers:settings.showOutliers)||!visible(uv)) continue;
        const color=accepted?'#58ff94':(data.inliers[i]?'#ffbe55':'#ff6868');
        const [x,y]=uv;
        if(settings.showResiduals&&data.projected&&visible(data.projected[i])) {
            const [px,py]=data.projected[i];
            ctx.strokeStyle='#45dcff';ctx.lineWidth=1.5*scale;
            ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(px,py);ctx.stroke();
            ctx.beginPath();ctx.moveTo(px-3*scale,py);ctx.lineTo(px+3*scale,py);ctx.moveTo(px,py-3*scale);ctx.lineTo(px,py+3*scale);ctx.stroke();
        }
        ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);
        ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#08111e';ctx.lineWidth=2*scale;ctx.stroke();
        if(settings.showPointIds) {
            ctx.font=`${13*scale}px ui-monospace,monospace`;ctx.lineWidth=3*scale;ctx.strokeStyle='#08111e';
            ctx.strokeText(String(i+1),x+radius+3,y-3);ctx.fillStyle=color;ctx.fillText(String(i+1),x+radius+3,y-3);
        }
    }
    if(settings.showModelBox&&data.box) {
        ctx.strokeStyle='#ffe16b';ctx.lineWidth=1.5*scale;
        // Corners were enumerated in xyz binary order. Flip one bit per edge.
        for(let i=0;i<8;i++) for(const bit of [1,2,4]) {const j=i^bit;if(j<=i)continue;const a=data.box[i],b=data.box[j];if(!a.every(Number.isFinite)||!b.every(Number.isFinite))continue;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    }
}
