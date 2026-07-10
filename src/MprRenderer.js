import { add, scale } from "./math3d.js";
import { CoordinateTransformer } from "./CoordinateTransformer.js";
import { Interpolator } from "./Interpolator.js";
import { MprPlaneCalculator } from "./MprPlaneCalculator.js";
import { WindowingProcessor } from "./WindowingProcessor.js";

export class MprRenderer {
  constructor(logger) {
    this.logger = logger;
    this.planeCalculator = new MprPlaneCalculator();
    this.windowing = new WindowingProcessor();
  }

  render(volume, request) {
    const start = performance.now();
    const plane = this.planeCalculator.calculate(request);
    for (const warning of plane.warnings) this.logger.warn(warning);
    const transformer = new CoordinateTransformer(volume.metadata);
    const interpolator = new Interpolator(volume, request.outsideValue);
    const scalarType = volume.metadata.scalarType || "";
    const PixelArray = /FLOAT|DOUBLE/.test(scalarType) ? Float32Array : Int16Array;
    const pixels = new PixelArray(request.width * request.height);
    let outsideCount = 0;
    let invalidCount = 0;
    const halfU = (request.width - 1) / 2;
    const halfV = (request.height - 1) / 2;
    for (let j = 0; j < request.height; j += 1) {
      const offsetV = (j - halfV) * request.spacingV;
      for (let i = 0; i < request.width; i += 1) {
        const offsetU = (i - halfU) * request.spacingU;
        const point = add(add(plane.center, scale(plane.u, offsetU)), scale(plane.v, offsetV));
        const voxel = transformer.worldToVoxel(point);
        const value = request.interpolation === "nearest"
          ? interpolator.nearest(voxel[0], voxel[1], voxel[2])
          : interpolator.linear(voxel[0], voxel[1], voxel[2]);
        const index = j * request.width + i;
        if (value === request.outsideValue) outsideCount += 1;
        if (!Number.isFinite(value)) {
          pixels[index] = request.outsideValue;
          invalidCount += 1;
        } else {
          // DICOM CTは整数HUとして保持し、MHAの浮動小数スカラーはwindow処理まで精度を残す。
          pixels[index] = PixelArray === Float32Array ? value : Math.round(value);
        }
      }
    }
    const rgba = this.windowing.toRgba(pixels, request.windowLevel, request.windowWidth, volume.metadata.photometricInterpretation);
    const elapsedMs = performance.now() - start;
    this.logger.info("MPR生成完了", { ms: Math.round(elapsedMs), outsideCount, invalidCount });
    return { pixels, rgba, plane, width: request.width, height: request.height, elapsedMs, outsideCount, invalidCount };
  }

  draw(canvas, result) {
    if (canvas.width !== result.width) canvas.width = result.width;
    if (canvas.height !== result.height) canvas.height = result.height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(result.rgba, result.width, result.height);
    ctx.putImageData(imageData, 0, 0);
  }
}
