import { cross, dot, invert3, length, normalize, nearlyEqual } from "./math3d.js";

export class VolumeBuilder {
  constructor(logger) {
    this.logger = logger;
  }

  build(slices) {
    if (!slices.length) throw new Error("読み込めたDICOMスライスがありません");
    const start = performance.now();
    const seriesUids = new Set(slices.map((slice) => slice.seriesInstanceUid));
    if (seriesUids.size > 1) {
      this.logger.warn("異なるSeries Instance UIDが混在しています。先頭シリーズのみ使用します。", [...seriesUids]);
      const firstUid = slices[0].seriesInstanceUid;
      slices = slices.filter((slice) => slice.seriesInstanceUid === firstUid);
    }

    const first = slices[0];
    const rowDirection = normalize(first.imageOrientation.slice(0, 3));
    const columnDirection = normalize(first.imageOrientation.slice(3, 6));
    const sliceDirection = normalize(cross(rowDirection, columnDirection));

    for (const slice of slices) {
      if (slice.rows !== first.rows || slice.columns !== first.columns) throw new Error("画像サイズが異なるスライスがあります");
      if (!nearlyEqual(slice.pixelSpacing[0], first.pixelSpacing[0]) || !nearlyEqual(slice.pixelSpacing[1], first.pixelSpacing[1])) throw new Error("Pixel Spacingが異なるスライスがあります");
      for (let i = 0; i < 6; i += 1) {
        if (!nearlyEqual(slice.imageOrientation[i], first.imageOrientation[i], 1e-4)) throw new Error("Image Orientation Patientが異なるスライスがあります");
      }
    }

    slices = [...slices].sort((a, b) => a.slicePosition - b.slicePosition);
    const sliceSpacings = [];
    for (let i = 1; i < slices.length; i += 1) {
      sliceSpacings.push(dot(subVec(slices[i].imagePosition, slices[i - 1].imagePosition), sliceDirection));
    }
    const spacingZ = median(sliceSpacings.map(Math.abs)) || 1;
    const maxSpacingError = sliceSpacings.reduce((max, value) => Math.max(max, Math.abs(Math.abs(value) - spacingZ)), 0);
    if (maxSpacingError > Math.max(0.01, spacingZ * 0.02)) {
      this.logger.warn("スライス間隔が不均一です", { spacingZ, maxSpacingError });
    }

    const width = first.columns;
    const height = first.rows;
    const depth = slices.length;
    const data = new Int16Array(width * height * depth);
    let minimumHU = Infinity;
    let maximumHU = -Infinity;
    for (let z = 0; z < depth; z += 1) {
      const slice = slices[z];
      const offset = z * width * height;
      for (let i = 0; i < slice.pixelData.length; i += 1) {
        const hu = Math.round(slice.pixelData[i] * slice.rescaleSlope + slice.rescaleIntercept);
        data[offset + i] = clampInt16(hu);
        if (hu < minimumHU) minimumHU = hu;
        if (hu > maximumHU) maximumHU = hu;
      }
    }

    const direction = [
      rowDirection[0], columnDirection[0], sliceDirection[0],
      rowDirection[1], columnDirection[1], sliceDirection[1],
      rowDirection[2], columnDirection[2], sliceDirection[2],
    ];
    const volume = {
      data,
      metadata: {
        dimensions: [width, height, depth],
        origin: slices[0].imagePosition,
        spacing: [first.pixelSpacing[1], first.pixelSpacing[0], spacingZ],
        direction,
        inverseDirection: invert3(direction),
        minimumValue: minimumHU,
        maximumValue: maximumHU,
        minimumHU,
        maximumHU,
        modality: "CT",
        scalarType: "MET_SHORT",
        sourceFormat: "DICOM",
        seriesInstanceUid: first.seriesInstanceUid,
        studyInstanceUid: first.studyInstanceUid,
        windowCenter: first.windowCenter,
        windowWidth: first.windowWidth,
        photometricInterpretation: first.photometricInterpretation,
      },
    };
    this.logger.info("CTボリューム構築完了", { dimensions: volume.metadata.dimensions, ms: Math.round(performance.now() - start) });
    return volume;
  }
}

function subVec(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function clampInt16(value) {
  return Math.max(-32768, Math.min(32767, value));
}
