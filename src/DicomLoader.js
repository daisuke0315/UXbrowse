const LONG_VR = new Set(["OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"]);
const TEXT_DECODER = new TextDecoder("ascii");

const TAGS = {
  transferSyntaxUid: "0002,0010",
  studyInstanceUid: "0020,000D",
  seriesInstanceUid: "0020,000E",
  imagePosition: "0020,0032",
  imageOrientation: "0020,0037",
  instanceNumber: "0020,0013",
  sliceLocation: "0020,1041",
  rows: "0028,0010",
  columns: "0028,0011",
  pixelSpacing: "0028,0030",
  bitsAllocated: "0028,0100",
  bitsStored: "0028,0101",
  pixelRepresentation: "0028,0103",
  photometricInterpretation: "0028,0004",
  windowCenter: "0028,1050",
  windowWidth: "0028,1051",
  rescaleIntercept: "0028,1052",
  rescaleSlope: "0028,1053",
  pixelData: "7FE0,0010",
};

export class DicomLoader {
  constructor(logger) {
    this.logger = logger;
  }

  async loadFiles(files) {
    const parsed = [];
    const rejected = [];
    const start = performance.now();
    for (const file of files) {
      try {
        const buffer = await file.arrayBuffer();
        parsed.push(this.parse(buffer, file.name));
      } catch (error) {
        rejected.push({ name: file.name, reason: error.message });
      }
    }
    this.logger.info("DICOM読込完了", { parsed: parsed.length, rejected: rejected.length, ms: Math.round(performance.now() - start) });
    if (rejected.length) this.logger.warn("DICOMとして扱えないファイルがあります", rejected.slice(0, 8));
    return { slices: parsed, rejected };
  }

  parse(buffer, fileName) {
    const view = new DataView(buffer);
    if (buffer.byteLength < 132 || TEXT_DECODER.decode(new Uint8Array(buffer, 128, 4)) !== "DICM") {
      throw new Error("DICOM preamble not found");
    }

    const meta = new Map();
    let offset = 132;
    let transferSyntaxUid = "1.2.840.10008.1.2.1";
    while (offset + 8 <= buffer.byteLength) {
      const group = view.getUint16(offset, true);
      if (group !== 0x0002) break;
      const element = this.readElement(view, buffer, offset, true, true);
      meta.set(element.tag, element);
      offset = element.nextOffset;
      if (element.tag === TAGS.transferSyntaxUid) transferSyntaxUid = this.readString(element);
    }

    const explicit = transferSyntaxUid !== "1.2.840.10008.1.2";
    const littleEndian = transferSyntaxUid !== "1.2.840.10008.1.2.2";
    if (!littleEndian) throw new Error("Big Endian DICOM is not supported in this phase");

    const dataset = new Map(meta);
    while (offset + 8 <= buffer.byteLength) {
      const element = this.readElement(view, buffer, offset, explicit, littleEndian);
      dataset.set(element.tag, element);
      offset = element.nextOffset;
      if (element.tag === TAGS.pixelData) break;
    }

    return this.toSlice(dataset, transferSyntaxUid, fileName);
  }

  readElement(view, buffer, offset, explicit, littleEndian) {
    const group = view.getUint16(offset, littleEndian);
    const element = view.getUint16(offset + 2, littleEndian);
    const tag = `${group.toString(16).padStart(4, "0").toUpperCase()},${element.toString(16).padStart(4, "0").toUpperCase()}`;
    let vr = "UN";
    let length;
    let valueOffset;
    if (explicit) {
      vr = TEXT_DECODER.decode(new Uint8Array(buffer, offset + 4, 2));
      if (LONG_VR.has(vr)) {
        length = view.getUint32(offset + 8, littleEndian);
        valueOffset = offset + 12;
      } else {
        length = view.getUint16(offset + 6, littleEndian);
        valueOffset = offset + 8;
      }
    } else {
      length = view.getUint32(offset + 4, littleEndian);
      valueOffset = offset + 8;
    }
    if (length === 0xffffffff) throw new Error(`Undefined length is not supported: ${tag}`);
    const nextOffset = valueOffset + length + (length % 2);
    return { tag, vr, valueOffset, length, nextOffset, littleEndian, view, buffer };
  }

  toSlice(dataset, transferSyntaxUid, fileName) {
    const required = [TAGS.rows, TAGS.columns, TAGS.pixelSpacing, TAGS.imagePosition, TAGS.imageOrientation, TAGS.seriesInstanceUid, TAGS.pixelData];
    for (const tag of required) {
      if (!dataset.has(tag)) throw new Error(`Required DICOM tag missing: ${tag}`);
    }
    const bitsAllocated = this.readUShort(dataset.get(TAGS.bitsAllocated) ?? fail("Bits Allocated missing"));
    const pixelRepresentation = this.readUShort(dataset.get(TAGS.pixelRepresentation) ?? fail("Pixel Representation missing"));
    if (bitsAllocated !== 16) throw new Error(`Only 16-bit Pixel Data is supported: ${bitsAllocated}`);

    const rows = this.readUShort(dataset.get(TAGS.rows));
    const columns = this.readUShort(dataset.get(TAGS.columns));
    const pixelData = this.readPixelData(dataset.get(TAGS.pixelData), rows * columns, pixelRepresentation);
    const imageOrientation = this.readNumbers(dataset.get(TAGS.imageOrientation));
    const imagePosition = this.readNumbers(dataset.get(TAGS.imagePosition));
    const row = imageOrientation.slice(0, 3);
    const col = imageOrientation.slice(3, 6);
    const normal = [
      row[1] * col[2] - row[2] * col[1],
      row[2] * col[0] - row[0] * col[2],
      row[0] * col[1] - row[1] * col[0],
    ];
    return {
      rows,
      columns,
      pixelSpacing: this.readNumbers(dataset.get(TAGS.pixelSpacing)).slice(0, 2),
      imagePosition,
      imageOrientation,
      slicePosition: imagePosition[0] * normal[0] + imagePosition[1] * normal[1] + imagePosition[2] * normal[2],
      instanceNumber: this.readOptionalNumber(dataset.get(TAGS.instanceNumber)),
      seriesInstanceUid: this.readString(dataset.get(TAGS.seriesInstanceUid)),
      studyInstanceUid: this.readString(dataset.get(TAGS.studyInstanceUid)) || "",
      rescaleSlope: this.readOptionalNumber(dataset.get(TAGS.rescaleSlope)) ?? 1,
      rescaleIntercept: this.readOptionalNumber(dataset.get(TAGS.rescaleIntercept)) ?? 0,
      windowCenter: firstNumber(this.readOptionalNumbers(dataset.get(TAGS.windowCenter))),
      windowWidth: firstNumber(this.readOptionalNumbers(dataset.get(TAGS.windowWidth))),
      bitsAllocated,
      bitsStored: this.readUShort(dataset.get(TAGS.bitsStored) ?? fail("Bits Stored missing")),
      pixelRepresentation,
      photometricInterpretation: this.readString(dataset.get(TAGS.photometricInterpretation)) || "MONOCHROME2",
      transferSyntaxUid,
      pixelData,
      fileName,
    };
  }

  readPixelData(element, expectedLength, pixelRepresentation) {
    const byteOffset = element.valueOffset;
    if (element.length < expectedLength * 2) throw new Error("Pixel Data is shorter than expected");
    const source = pixelRepresentation === 1
      ? new Int16Array(element.buffer, byteOffset, expectedLength)
      : new Uint16Array(element.buffer, byteOffset, expectedLength);
    return new (pixelRepresentation === 1 ? Int16Array : Uint16Array)(source);
  }

  readString(element) {
    if (!element) return "";
    return TEXT_DECODER.decode(new Uint8Array(element.buffer, element.valueOffset, element.length)).replace(/\0/g, "").trim();
  }

  readNumbers(element) {
    return this.readString(element).split("\\").filter(Boolean).map(Number);
  }

  readOptionalNumbers(element) {
    return element ? this.readNumbers(element).filter(Number.isFinite) : [];
  }

  readOptionalNumber(element) {
    const numbers = this.readOptionalNumbers(element);
    if (numbers.length) return numbers[0];
    if (!element) return null;
    if (element.length === 2) return this.readUShort(element);
    return null;
  }

  readUShort(element) {
    return element.view.getUint16(element.valueOffset, element.littleEndian);
  }
}

function firstNumber(numbers) {
  return numbers.length ? numbers[0] : null;
}

function fail(message) {
  throw new Error(message);
}
