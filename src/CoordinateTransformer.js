import { mulMat3Vec3, sub } from "./math3d.js";

export class CoordinateTransformer {
  constructor(metadata) {
    this.metadata = metadata;
  }

  worldToVoxel(point) {
    const relative = sub(point, this.metadata.origin);
    const oriented = mulMat3Vec3(this.metadata.inverseDirection, relative);
    return [
      oriented[0] / this.metadata.spacing[0],
      oriented[1] / this.metadata.spacing[1],
      oriented[2] / this.metadata.spacing[2],
    ];
  }

  voxelToWorld(index) {
    const scaled = [
      index[0] * this.metadata.spacing[0],
      index[1] * this.metadata.spacing[1],
      index[2] * this.metadata.spacing[2],
    ];
    const d = this.metadata.direction;
    return [
      this.metadata.origin[0] + d[0] * scaled[0] + d[1] * scaled[1] + d[2] * scaled[2],
      this.metadata.origin[1] + d[3] * scaled[0] + d[4] * scaled[1] + d[5] * scaled[2],
      this.metadata.origin[2] + d[6] * scaled[0] + d[7] * scaled[1] + d[8] * scaled[2],
    ];
  }
}
