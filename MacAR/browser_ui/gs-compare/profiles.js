// Changes apply only to the comparison renderer; no PLY or reference writes.
export const profiles = {
    default: {label:'現行条件', backend:'webgl2', minPixelSize:2, minContribution:3, format:'compact', gain:1},
    detail: {label:'小粒子カットなし', backend:'webgl2', minPixelSize:0, minContribution:0, format:'compact', gain:1},
    precision: {label:'高精度カラー格納', backend:'webgl2', minPixelSize:2, minContribution:3, format:'large', gain:1},
    gpu: {label:'WebGPU・毎フレームGPUソート', backend:'webgpu', minPixelSize:2, minContribution:3, format:'compact', gain:1},
    alpha15: {label:'alpha 1.5倍（見た目だけの実験）', backend:'webgl2', minPixelSize:2, minContribution:3, format:'compact', gain:1.5},
    alpha25: {label:'alpha 2.5倍（見た目だけの実験）', backend:'webgl2', minPixelSize:2, minContribution:3, format:'compact', gain:2.5},
};

export function referencePose(config, view) {
    const matrix=view.camera_to_world, eye=view.eye;
    if(!matrix)return {eye,target:config.center,up:config.up};
    const distance=Math.hypot(...eye.map((x,i)=>x-config.center[i]));
    return {eye,target:eye.map((x,i)=>x+distance*matrix[i][2]),up:matrix.slice(0,3).map(row=>-row[1])};
}
