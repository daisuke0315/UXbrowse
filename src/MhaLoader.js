import { invert3 } from "./math3d.js";

const MHA_COORDINATE_WARNING = "MHA座標系はDICOM患者座標系であるとは限りません。DICOM由来MHAの場合は、origin / spacing / direction が保持されていることを確認してください。";

const ELEMENT_TYPES = {
  MET_CHAR: { bytes: 1, ArrayType: Int8Array, reader: "getInt8", scalarType: "MET_CHAR" },
  MET_UCHAR: { bytes: 1, ArrayType: Uint8Array, reader: "getUint8", scalarType: "MET_UCHAR" },
  MET_SHORT: { bytes: 2, ArrayType: Int16Array, reader: "getInt16", scalarType: "MET_SHORT" },
  MET_USHORT: { bytes: 2, ArrayType: Uint16Array, reader: "getUint16", scalarType: "MET_USHORT" },
  MET_INT: { bytes: 4, ArrayType: Int32Array, reader: "getInt32", scalarType: "MET_INT" },
  MET_UINT: { bytes: 4, ArrayType: Uint32Array, reader: "getUint32", scalarType: "MET_UINT" },
  MET_FLOAT: { bytes: 4, ArrayType: Float32Array, reader: "getFloat32", scalarType: "MET_FLOAT" },
  MET_DOUBLE: { bytes: 8, ArrayType: Float64Array, reader: "getFloat64", scalarType: "MET_DOUBLE" },
};

export class MhaLoader {
  constructor(logger) {
    this.logger = logger;
  }

  async loadFile(file) {
    if (!file) throw new Error("MHAファイルが指定されていません");
    if (!file.name.toLowerCase().endsWith(".mha")) throw new Error(".mhaファイルを選択してください");
    return this.parse(await file.arrayBuffer(), file.name);
  }

  parse(arrayBuffer, fileName = "volume.mha") {
    const start = performance.now();
    const { header, dataOffset } = parseHeader(arrayBuffer);

    // MetaImageのヘッダは大小文字が混ざることがあるため、検索時だけlower-caseのMapを使う。
    const headerValue = (key) => getHeaderValue(header, key);
    if ((headerValue("ObjectType") || "").toLowerCase() !== "image") throw new Error("ObjectType = Image のMHAのみ対応しています");
    if (readIntField(header, "NDims", 0) !== 3) throw new Error("NDims = 3 のMHAのみ対応しています");
    if (readBoolField(header, "CompressedData", false)) throw new Error("CompressedData = True のMHAは未対応です");
    if (getHeaderValue(header, "BinaryData") == null || !readBoolField(header, "BinaryData", false)) throw new Error("BinaryData = True のMHAのみ対応しています");
    if ((headerValue("ElementDataFile") || "").toUpperCase() !== "LOCAL") throw new Error("ElementDataFile = LOCAL の.mhaのみ対応しています");

    const dimensions = readNumberList(header, "DimSize", 3, { integer: true });
    const spacing = readNumberList(header, "ElementSpacing", 3);
    const origin = readOrigin(header);
    const direction = readDirection(header);
    const elementType = (headerValue("ElementType") || "").toUpperCase();
    const elementInfo = ELEMENT_TYPES[elementType];
    if (!elementInfo) throw new Error(`未対応のElementTypeです: ${elementType || "(missing)"}`);

    const voxelCount = dimensions[0] * dimensions[1] * dimensions[2];
    const expectedBytes = voxelCount * elementInfo.bytes;
    if (arrayBuffer.byteLength - dataOffset < expectedBytes) {
      throw new Error(`MHAバイナリ長が不足しています: expected=${expectedBytes}, actual=${arrayBuffer.byteLength - dataOffset}`);
    }

    // MHAはElementByteOrderMSBで保存endianを示す。DataView経由で読むことで
    // ホスト環境のendianに依存せず、big-endian MHAも同じTypedArrayへ正規化できる。
    const littleEndian = !readBoolField(header, "ElementByteOrderMSB", readBoolField(header, "BinaryDataByteOrderMSB", false));
    const dataView = new DataView(arrayBuffer, dataOffset, expectedBytes);
    const data = new elementInfo.ArrayType(voxelCount);
    let minimumValue = Infinity;
    let maximumValue = -Infinity;
    let invalidCount = 0;
    for (let i = 0; i < voxelCount; i += 1) {
      const value = readScalar(dataView, i * elementInfo.bytes, elementInfo, littleEndian);
      data[i] = value;
      if (Number.isFinite(value)) {
        if (value < minimumValue) minimumValue = value;
        if (value > maximumValue) maximumValue = value;
      } else {
        invalidCount += 1;
      }
    }
    if (invalidCount > voxelCount * 0.01) throw new Error(`MHAにNaN/Infinityが多すぎます: ${invalidCount}/${voxelCount}`);
    if (!Number.isFinite(minimumValue) || !Number.isFinite(maximumValue)) throw new Error("MHAの有効なスカラー値を取得できませんでした");

    const windowWidth = Math.max(1, maximumValue - minimumValue);
    const windowCenter = minimumValue + windowWidth / 2;
    const volume = {
      data,
      metadata: {
        dimensions,
        origin,
        spacing,
        direction,
        inverseDirection: invert3(direction),
        minimumValue,
        maximumValue,
        minimumHU: minimumValue,
        maximumHU: maximumValue,
        modality: "MHA",
        scalarType: elementInfo.scalarType,
        sourceFormat: "MHA",
        sourceFileName: fileName,
        seriesInstanceUid: `mha:${fileName}`,
        studyInstanceUid: "mha",
        windowCenter,
        windowWidth,
        photometricInterpretation: "MONOCHROME2",
        coordinateWarning: MHA_COORDINATE_WARNING,
      },
    };
    this.logger?.warn?.(MHA_COORDINATE_WARNING);
    this.logger?.info?.("MHAボリューム読込完了", {
      fileName,
      dimensions,
      spacing,
      scalarType: elementInfo.scalarType,
      min: minimumValue,
      max: maximumValue,
      ms: Math.round(performance.now() - start),
    });
    return volume;
  }
}

function parseHeader(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const decoder = new TextDecoder("ascii");
  const header = new Map();
  let lineStart = 0;
  for (let i = 0; i < bytes.length; i += 1) {
    if (bytes[i] !== 10) continue;
    const lineBytes = bytes.subarray(lineStart, bytes[i - 1] === 13 ? i - 1 : i);
    const line = decoder.decode(lineBytes).trim();
    lineStart = i + 1;
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    const value = line.slice(eq + 1).trim();
    header.set(key, value);
    if (key.toLowerCase() === "elementdatafile") {
      return { header, dataOffset: lineStart };
    }
  }
  throw new Error("ElementDataFile = LOCAL を含むMHAヘッダを検出できませんでした");
}

function getHeaderValue(header, key) {
  const target = key.toLowerCase();
  for (const [candidate, value] of header) {
    if (candidate.toLowerCase() === target) return value;
  }
  return undefined;
}

function readNumberList(header, key, count, { fallback = null, integer = false } = {}) {
  const raw = getHeaderValue(header, key);
  if (!raw) {
    if (fallback) return [...fallback];
    throw new Error(`${key} がMHAヘッダにありません`);
  }
  const values = raw.split(/\s+/).map(Number);
  if (values.length !== count || values.some((value) => !Number.isFinite(value))) throw new Error(`${key} が不正です: ${raw}`);
  if (integer && values.some((value) => !Number.isInteger(value) || value <= 0)) throw new Error(`${key} は正の整数である必要があります: ${raw}`);
  if (!integer && values.some((value) => value <= 0)) throw new Error(`${key} は正の値である必要があります: ${raw}`);
  return values;
}

function readOrigin(header) {
  for (const key of ["Offset", "Position", "Origin"]) {
    const raw = getHeaderValue(header, key);
    if (!raw) continue;
    const values = raw.split(/\s+/).map(Number);
    if (values.length === 3 && values.every(Number.isFinite)) return values;
    throw new Error(`${key} が不正です: ${raw}`);
  }
  return [0, 0, 0];
}

function readDirection(header) {
  const raw = getHeaderValue(header, "TransformMatrix");
  if (!raw) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const values = raw.split(/\s+/).map(Number);
  if (values.length !== 9 || values.some((value) => !Number.isFinite(value))) throw new Error(`TransformMatrixが不正です: ${raw}`);
  return values;
}

function readIntField(header, key, fallback) {
  const raw = getHeaderValue(header, key);
  if (!raw) return fallback;
  const value = Number.parseInt(raw, 10);
  if (!Number.isInteger(value)) throw new Error(`${key} が不正です: ${raw}`);
  return value;
}

function readBoolField(header, key, fallback) {
  const raw = getHeaderValue(header, key);
  if (raw == null) return fallback;
  return /^(true|1|yes)$/i.test(raw);
}

function readScalar(dataView, byteOffset, elementInfo, littleEndian) {
  if (elementInfo.bytes === 1) return dataView[elementInfo.reader](byteOffset);
  return dataView[elementInfo.reader](byteOffset, littleEndian);
}
