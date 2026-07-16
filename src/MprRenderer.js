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
    const scalarType = volume.metadata.scalarType || "";
    const PixelArray = /FLOAT|DOUBLE/.test(scalarType) ? Float32Array : Int16Array;
    const pixels = new PixelArray(request.width * request.height);
    const grid = createVoxelSamplingGrid(volume.metadata, plane, request);
    const stats = request.interpolation === "nearest"
      ? renderNearest(volume, request, grid, pixels, PixelArray === Float32Array)
      : renderLinear(volume, request, grid, pixels, PixelArray === Float32Array);
    const rgba = this.windowing.toRgba(pixels, request.windowLevel, request.windowWidth, volume.metadata.photometricInterpretation);
    const elapsedMs = performance.now() - start;
    this.logger.info("MPR生成完了", { ms: Math.round(elapsedMs), outsideCount: stats.outsideCount, invalidCount: stats.invalidCount });
    return { pixels, rgba, plane, width: request.width, height: request.height, elapsedMs, outsideCount: stats.outsideCount, invalidCount: stats.invalidCount };
  }

  draw(canvas, result) {
    if (canvas.width !== result.width) canvas.width = result.width;
    if (canvas.height !== result.height) canvas.height = result.height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(result.rgba, result.width, result.height);
    ctx.putImageData(imageData, 0, 0);
  }
}

function createVoxelSamplingGrid(metadata, plane, request) {
  const inverse = metadata.inverseDirection;
  const spacing = metadata.spacing;
  const relativeCenter = [
    plane.center[0] - metadata.origin[0],
    plane.center[1] - metadata.origin[1],
    plane.center[2] - metadata.origin[2],
  ];
  const center = transformToVoxel(inverse, spacing, relativeCenter);
  const stepU = transformToVoxel(inverse, spacing, plane.u).map((value) => value * request.spacingU);
  const stepV = transformToVoxel(inverse, spacing, plane.v).map((value) => value * request.spacingV);
  const halfU = (request.width - 1) / 2;
  const halfV = (request.height - 1) / 2;
  return {
    start: [
      center[0] - halfU * stepU[0] - halfV * stepV[0],
      center[1] - halfU * stepU[1] - halfV * stepV[1],
      center[2] - halfU * stepU[2] - halfV * stepV[2],
    ],
    stepU,
    stepV,
  };
}

function transformToVoxel(inverse, spacing, vector) {
  return [
    (inverse[0] * vector[0] + inverse[1] * vector[1] + inverse[2] * vector[2]) / spacing[0],
    (inverse[3] * vector[0] + inverse[4] * vector[1] + inverse[5] * vector[2]) / spacing[1],
    (inverse[6] * vector[0] + inverse[7] * vector[1] + inverse[8] * vector[2]) / spacing[2],
  ];
}

function renderNearest(volume, request, grid, pixels, preserveFloat) {
  const data = volume.data;
  const [width, height, depth] = volume.metadata.dimensions;
  const sliceStride = width * height;
  const [startX, startY, startZ] = grid.start;
  const [stepUx, stepUy, stepUz] = grid.stepU;
  const [stepVx, stepVy, stepVz] = grid.stepV;
  let outsideCount = 0;
  let invalidCount = 0;
  let index = 0;

  for (let j = 0; j < request.height; j += 1) {
    const rowX = startX + j * stepVx;
    const rowY = startY + j * stepVy;
    const rowZ = startZ + j * stepVz;
    for (let i = 0; i < request.width; i += 1, index += 1) {
      const x = Math.round(rowX + i * stepUx);
      const y = Math.round(rowY + i * stepUy);
      const z = Math.round(rowZ + i * stepUz);
      let value = request.outsideValue;
      if (x >= 0 && y >= 0 && z >= 0 && x < width && y < height && z < depth) {
        value = data[z * sliceStride + y * width + x];
      }
      if (value === request.outsideValue) outsideCount += 1;
      if (!Number.isFinite(value)) {
        pixels[index] = request.outsideValue;
        invalidCount += 1;
      } else {
        pixels[index] = preserveFloat ? value : Math.round(value);
      }
    }
  }
  return { outsideCount, invalidCount };
}

function renderLinear(volume, request, grid, pixels, preserveFloat) {
  const data = volume.data;
  const [width, height, depth] = volume.metadata.dimensions;
  const sliceStride = width * height;
  const [startX, startY, startZ] = grid.start;
  const [stepUx, stepUy, stepUz] = grid.stepU;
  const [stepVx, stepVy, stepVz] = grid.stepV;
  let outsideCount = 0;
  let invalidCount = 0;
  let index = 0;

  for (let j = 0; j < request.height; j += 1) {
    const rowX = startX + j * stepVx;
    const rowY = startY + j * stepVy;
    const rowZ = startZ + j * stepVz;
    for (let i = 0; i < request.width; i += 1, index += 1) {
      const x = rowX + i * stepUx;
      const y = rowY + i * stepUy;
      const z = rowZ + i * stepUz;
      let value = request.outsideValue;
      if (x >= 0 && y >= 0 && z >= 0 && x <= width - 1 && y <= height - 1 && z <= depth - 1) {
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const z0 = Math.floor(z);
        const x1 = Math.min(x0 + 1, width - 1);
        const y1 = Math.min(y0 + 1, height - 1);
        const z1 = Math.min(z0 + 1, depth - 1);
        const xd = x - x0;
        const yd = y - y0;
        const zd = z - z0;
        const z0Offset = z0 * sliceStride;
        const z1Offset = z1 * sliceStride;
        const y0Offset = y0 * width;
        const y1Offset = y1 * width;
        const c000 = data[z0Offset + y0Offset + x0];
        const c100 = data[z0Offset + y0Offset + x1];
        const c010 = data[z0Offset + y1Offset + x0];
        const c110 = data[z0Offset + y1Offset + x1];
        const c001 = data[z1Offset + y0Offset + x0];
        const c101 = data[z1Offset + y0Offset + x1];
        const c011 = data[z1Offset + y1Offset + x0];
        const c111 = data[z1Offset + y1Offset + x1];
        const c00 = c000 + (c100 - c000) * xd;
        const c10 = c010 + (c110 - c010) * xd;
        const c01 = c001 + (c101 - c001) * xd;
        const c11 = c011 + (c111 - c011) * xd;
        const c0 = c00 + (c10 - c00) * yd;
        const c1 = c01 + (c11 - c01) * yd;
        value = c0 + (c1 - c0) * zd;
      }
      if (value === request.outsideValue) outsideCount += 1;
      if (!Number.isFinite(value)) {
        pixels[index] = request.outsideValue;
        invalidCount += 1;
      } else {
        pixels[index] = preserveFloat ? value : Math.round(value);
      }
    }
  }
  return { outsideCount, invalidCount };
}
