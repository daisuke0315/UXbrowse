import { formatVector } from "../math3d.js";

export class DynamicMprController {
  constructor({ renderer, mprCanvas, cameraController, mapper, getBaseRequest, logger, onResult }) {
    this.renderer = renderer;
    this.mprCanvas = mprCanvas;
    this.cameraController = cameraController;
    this.mapper = mapper;
    this.getBaseRequest = getBaseRequest;
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
    const dicomInput = this.mapper.cameraInputToDicom(threeInput);
    const mapMs = performance.now() - mapStart;
    return this.updateFromDicomInput(volume, threeInput, dicomInput, { reason, totalStart, cameraMs, mapMs });
  }

  updateFromThreeInput(volume, threeInput, reason = "object") {
    if (!this.enabled || !volume || !threeInput) return null;
    const totalStart = performance.now();
    const mapStart = performance.now();
    const dicomInput = this.mapper.cameraInputToDicom(threeInput);
    const mapMs = performance.now() - mapStart;
    return this.updateFromDicomInput(volume, threeInput, dicomInput, { reason, totalStart, cameraMs: 0, mapMs });
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
      threeInput,
      dicomInput,
      request,
      result,
      timings: { cameraMs: timingContext.cameraMs, mapMs: timingContext.mapMs, renderMs, drawMs, totalMs },
      text: this.formatDebug(threeInput, dicomInput, result, { cameraMs: timingContext.cameraMs, mapMs: timingContext.mapMs, renderMs, drawMs, totalMs }),
    };
    this.onResult?.(this.lastInfo);
    return this.lastInfo;
  }

  formatDebug(threeInput, dicomInput, result, timings) {
    return [
      "Three.js",
      `eyeThree: ${formatVector(threeInput.eye)}`,
      `targetThree: ${formatVector(threeInput.target)}`,
      `upThree: ${formatVector(threeInput.up)}`,
      `nThree: ${formatVector(threeInput.n)}`,
      `targetMode: ${threeInput.targetMode ?? "orbit-target"}`,
      `distance: ${threeInput.distance.toFixed(2)} mm`,
      `orbitDistance: ${threeInput.orbitDistance?.toFixed?.(2) ?? "n/a"} mm`,
      `quaternion: ${threeInput.quaternion ? threeInput.quaternion.map((x) => x.toFixed(4)).join(", ") : "n/a"}`,
      "",
      "Volume",
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
