export class WindowingProcessor {
  apply(hu, windowLevel, windowWidth) {
    const width = Math.max(1, windowWidth);
    const lower = windowLevel - width / 2;
    const upper = windowLevel + width / 2;
    if (hu <= lower) return 0;
    if (hu >= upper) return 255;
    return Math.round(((hu - lower) / (upper - lower)) * 255);
  }

  toRgba(pixels, windowLevel, windowWidth, photometricInterpretation = "MONOCHROME2") {
    const rgba = new Uint8ClampedArray(pixels.length * 4);
    const inverse = photometricInterpretation === "MONOCHROME1";
    for (let i = 0; i < pixels.length; i += 1) {
      const gray = inverse ? 255 - this.apply(pixels[i], windowLevel, windowWidth) : this.apply(pixels[i], windowLevel, windowWidth);
      const offset = i * 4;
      rgba[offset] = gray;
      rgba[offset + 1] = gray;
      rgba[offset + 2] = gray;
      rgba[offset + 3] = 255;
    }
    return rgba;
  }
}
