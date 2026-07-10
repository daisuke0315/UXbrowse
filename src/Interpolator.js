export class Interpolator {
  constructor(volume, outsideValue = -1024) {
    this.volume = volume;
    this.outsideValue = outsideValue;
    this.width = volume.metadata.dimensions[0];
    this.height = volume.metadata.dimensions[1];
    this.depth = volume.metadata.dimensions[2];
  }

  nearest(x, y, z) {
    const xi = Math.round(x);
    const yi = Math.round(y);
    const zi = Math.round(z);
    if (!this.inBounds(xi, yi, zi)) return this.outsideValue;
    return this.volume.data[this.index(xi, yi, zi)];
  }

  linear(x, y, z) {
    if (x < 0 || y < 0 || z < 0 || x > this.width - 1 || y > this.height - 1 || z > this.depth - 1) return this.outsideValue;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const x1 = Math.min(x0 + 1, this.width - 1);
    const y1 = Math.min(y0 + 1, this.height - 1);
    const z1 = Math.min(z0 + 1, this.depth - 1);
    const xd = x - x0;
    const yd = y - y0;
    const zd = z - z0;
    const c00 = lerp(this.at(x0, y0, z0), this.at(x1, y0, z0), xd);
    const c10 = lerp(this.at(x0, y1, z0), this.at(x1, y1, z0), xd);
    const c01 = lerp(this.at(x0, y0, z1), this.at(x1, y0, z1), xd);
    const c11 = lerp(this.at(x0, y1, z1), this.at(x1, y1, z1), xd);
    const c0 = lerp(c00, c10, yd);
    const c1 = lerp(c01, c11, yd);
    return lerp(c0, c1, zd);
  }

  at(x, y, z) {
    return this.volume.data[this.index(x, y, z)];
  }

  index(x, y, z) {
    return z * this.width * this.height + y * this.width + x;
  }

  inBounds(x, y, z) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.width && y < this.height && z < this.depth;
  }
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}
