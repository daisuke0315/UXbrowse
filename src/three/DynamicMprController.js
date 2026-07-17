import { formatVector } from "../math3d.js";
import { volumeVoxelCenterSpan } from "../VolumeGeometry.js";

export class DynamicMprController {
  constructor({ renderer, mprCanvas, cameraController, mapper, getBaseRequest, getPlaneMode, logger, onResult }) {
    this.renderer = renderer;
    this.mprCanvas = mprCanvas;
    this.cameraController = cameraController;
    this.mapper = mapper;
    this.getBaseRequest = getBaseRequest;
    this.getPlaneMode = getPlaneMode ?? (() => "view");
    this.logger = logger;
    this.onResult = onResult;
    this.enabled = true;
    this.lastInfo = null;
  }

  setEnabled(enabled) {
    this.enabled = enabled;
  }

  update(volume, reason = "dynamic") {
    if (!this.enabled || !volume) return null;
    const totalStart = performance.now();
    const cameraStart = performance.now();
    const threeInput = this.cameraController.getCameraInput();
    const cameraMs = performance.now() - cameraStart;
    const mapStart = performance.now();
    const planeMode = this.getPlaneMode();
    const dicomInput = constrainDicomInputToPlane(volume, this.mapper.cameraInputToDicom(threeInput), planeMode);
    const mapMs = performance.now() - mapStart;
    return this.updateFromDicomInput(volume, threeInput, dicomInput, { reason, totalStart, cameraMs, mapMs, planeMode });
  }

  updateFromThreeInput(volume, threeInput, reason = "object") {
    if (!this.enabled || !volume || !threeInput) return null;
    const totalStart = performance.now();
    const mapStart = performance.now();
    const planeMode = this.getPlaneMode();
    const dicomInput = constrainDicomInputToPlane(volume, this.mapper.cameraInputToDicom(threeInput), planeMode);
    const mapMs = performance.now() - mapStart;
    return this.updateFromDicomInput(volume, threeInput, dicomInput, { reason, totalStart, cameraMs: 0, mapMs, planeMode });
  }

  updateFromDicomInput(volume, threeInput, dicomInput, timingContext) {
    const request = {
      ...this.getBaseRequest(),
      eye: dicomInput.eye,
      target: dicomInput.target,
      up: dicomInput.up,
    };
    const renderStart = performance.now();
    const result = this.renderer.render(volume, request);
    const renderMs = performance.now() - renderStart;
    const drawStart = performance.now();
    this.renderer.draw(this.mprCanvas, result);
    const drawMs = performance.now() - drawStart;
    const totalMs = performance.now() - timingContext.totalStart;
    this.lastInfo = {
      reason: timingContext.reason,
      planeMode: timingContext.planeMode ?? "view",
      threeInput,
      dicomInput,
      request,
      result,
      timings: { cameraMs: timingContext.cameraMs, mapMs: timingContext.mapMs, renderMs, drawMs, totalMs },
      text: this.formatDebug(threeInput, dicomInput, result, { cameraMs: timingContext.cameraMs, mapMs: timingContext.mapMs, renderMs, drawMs, totalMs }, timingContext.planeMode),
    };
    this.onResult?.(this.lastInfo);
    return this.lastInfo;
  }

  formatDebug(threeInput, dicomInput, result, timings, planeMode = "view") {
    return [
      "Three.js",
      `eyeThree: ${formatVector(threeInput.eye)}`,
      `targetThree: ${formatVector(threeInput.target)}`,
      `upThree: ${formatVector(threeInput.up)}`,
      `nThree: ${formatVector(threeInput.n)}`,
      `targetMode: ${threeInput.targetMode ?? "orbit-target"}`,
      `distance: ${threeInput.distance.toFixed(2)} mm`,
      `orbitDistance: ${threeInput.orbitDistance?.toFixed?.(2) ?? "n/a"} mm`,
      `modelCenterThree: ${threeInput.modelCenter ? formatVector(threeInput.modelCenter) : "n/a"}`,
      `modelCenterDistance: ${threeInput.modelCenterDistance?.toFixed?.(2) ?? "n/a"} mm`,
      `quaternion: ${threeInput.quaternion ? threeInput.quaternion.map((x) => x.toFixed(4)).join(", ") : "n/a"}`,
      "",
      "Volume",
      `planeMode: ${planeMode}`,
      `eyeVolume: ${formatVector(dicomInput.eye)}`,
      `targetVolume: ${formatVector(dicomInput.target)}`,
      `upVolume: ${formatVector(dicomInput.up)}`,
      `nVolume: ${formatVector(dicomInput.n)}`,
      `uVolume: ${formatVector(result.plane.u)}`,
      `vVolume: ${formatVector(result.plane.v)}`,
      `center: ${formatVector(result.plane.center)}`,
      "",
      "Performance",
      `coord: ${timings.cameraMs.toFixed(1)} ms`,
      `map: ${timings.mapMs.toFixed(1)} ms`,
      `mpr: ${timings.renderMs.toFixed(1)} ms`,
      `draw: ${timings.drawMs.toFixed(1)} ms`,
      `total: ${timings.totalMs.toFixed(1)} ms`,
      `outside / invalid: ${result.outsideCount} / ${result.invalidCount}`,
    ].join("\n");
  }
}

const FIXED_PLANE_MODES = {
  axial: { axis: 2, n: [0, 0, 1], up: [0, 1, 0] },
  coronal: { axis: 1, n: [0, 1, 0], up: [0, 0, -1] },
  sagittal: { axis: 0, n: [1, 0, 0], up: [0, 0, -1] },
};

export function constrainDicomInputToPlane(volume, input, mode = "view") {
  const fixed = FIXED_PLANE_MODES[mode];
  if (!fixed || !volume?.metadata || !input?.target || !input?.eye) return input;
  // 固定するのは断面の向きだけ。断面のFOV中心はカメラ前方の注視点へ追従させる。
  // 以前は面内2軸をvolume中心へ戻していたため、Sagittal/Coronalでは
  // 視線を移動しても常にCT中央付近のFOVしか表示できなかった。
  const target = input.target.map((value, axis) => {
    const [minimum, maximum] = volumeWorldAxisBounds(volume.metadata, axis);
    return Math.max(minimum, Math.min(maximum, value));
  });
  const distance = Math.max(1, Math.hypot(
    input.target[0] - input.eye[0],
    input.target[1] - input.eye[1],
    input.target[2] - input.eye[2]
  ));
  const eye = target.map((value, index) => value - fixed.n[index] * distance);
  return {
    ...input,
    eye,
    target,
    up: [...fixed.up],
    n: [...fixed.n],
  };
}

function volumeWorldAxisBounds(metadata, axis) {
  const spans = volumeVoxelCenterSpan(metadata);
  const direction = metadata.direction;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const x of [0, spans[0]]) {
    for (const y of [0, spans[1]]) {
      for (const z of [0, spans[2]]) {
        const point = [
          metadata.origin[0] + direction[0] * x + direction[1] * y + direction[2] * z,
          metadata.origin[1] + direction[3] * x + direction[4] * y + direction[5] * z,
          metadata.origin[2] + direction[6] * x + direction[7] * y + direction[8] * z,
        ];
        minimum = Math.min(minimum, point[axis]);
        maximum = Math.max(maximum, point[axis]);
      }
    }
  }
  return [minimum, maximum];
}
