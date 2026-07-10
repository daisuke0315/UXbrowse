import { normalize, sub } from "../math3d.js";

export class ThreeCameraController {
  constructor(THREE, camera, controls, getTargetDistance = null) {
    this.THREE = THREE;
    this.camera = camera;
    this.controls = controls;
    this.getTargetDistance = getTargetDistance;
    this.tmpEye = new THREE.Vector3();
    this.tmpTarget = new THREE.Vector3();
    this.tmpForward = new THREE.Vector3();
    this.tmpUp = new THREE.Vector3();
  }

  getCameraInput() {
    this.camera.getWorldPosition(this.tmpEye);
    const eye = vecToArray(this.tmpEye);
    const orbitDistance = this.tmpEye.distanceTo(this.controls.target);
    const targetDistance = Math.max(0.001, Number(this.getTargetDistance?.() ?? orbitDistance));
    const forward = this.tmpForward.set(0, 0, -1).applyQuaternion(this.camera.quaternion).normalize();
    // ViewSync MPRの正規仕様:
    // MPRのtargetはOrbitControlsの回転中心ではなく、カメラ前方の固定距離点。
    // ズームでcamera.positionが変わるとtargetも動き、MPR断面位置が更新される。
    this.tmpTarget.copy(this.tmpEye).addScaledVector(forward, targetDistance);
    const target = vecToArray(this.tmpTarget);
    const up = vecToArray(this.tmpUp.set(0, 1, 0).applyQuaternion(this.camera.quaternion).normalize());
    const n = normalize(sub(target, eye));
    return {
      eye,
      target,
      up,
      n,
      distance: targetDistance,
      orbitDistance,
      targetMode: "camera-forward",
      fov: this.camera.fov,
      quaternion: [this.camera.quaternion.x, this.camera.quaternion.y, this.camera.quaternion.z, this.camera.quaternion.w],
    };
  }
}

function vecToArray(v) {
  return [v.x, v.y, v.z];
}
