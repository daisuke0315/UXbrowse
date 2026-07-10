import { add, cross, dot, normalize, scale, sub } from "./math3d.js";
import { volumeCenter } from "./VolumeGeometry.js";

export class MprPlaneCalculator {
  calculate({ eye, target, up, normalOffset = 0 }) {
    const warnings = [];
    const n = normalize(sub(target, eye));
    let upVector = normalize(up);
    if (Math.abs(dot(upVector, n)) > 0.99) {
      warnings.push("upと視線が平行に近いため代替upを使用しました");
      upVector = Math.abs(dot([1, 0, 0], n)) > 0.9 ? [0, 1, 0] : [1, 0, 0];
    }
    const u = normalize(cross(upVector, n));
    const v = normalize(cross(n, u));
    const center = add(target, scale(n, normalOffset));
    return { center, u, v, n, warnings };
  }

  axial(metadata) {
    const center = volumeCenter(metadata);
    return cameraFromPlane(center, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
  }

  coronal(metadata) {
    const center = volumeCenter(metadata);
    return cameraFromPlane(center, [1, 0, 0], [0, 0, -1], [0, 1, 0]);
  }

  sagittal(metadata) {
    const center = volumeCenter(metadata);
    return cameraFromPlane(center, [0, 1, 0], [0, 0, -1], [1, 0, 0]);
  }
}

function cameraFromPlane(center, u, v, n) {
  return {
    target: center,
    eye: [center[0] - n[0] * 500, center[1] - n[1] * 500, center[2] - n[2] * 500],
    up: v,
  };
}
