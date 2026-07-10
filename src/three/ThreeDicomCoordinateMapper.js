import { identity4, invert4, transformPoint, transformVector } from "./matrix4.js";

export class ThreeDicomCoordinateMapper {
  constructor(threeToDicomMatrix = identity4()) {
    this.setThreeToDicomMatrix(threeToDicomMatrix);
  }

  setThreeToDicomMatrix(matrix) {
    if (!Array.isArray(matrix) || matrix.length !== 16) throw new Error("threeToDicomMatrix must be a 4x4 matrix");
    this.threeToDicomMatrix = [...matrix];
    this.dicomToThreeMatrix = invert4(this.threeToDicomMatrix);
  }

  threePointToDicom(point) {
    return transformPoint(this.threeToDicomMatrix, point);
  }

  threeVectorToDicom(vector) {
    return transformVector(this.threeToDicomMatrix, vector);
  }

  dicomPointToThree(point) {
    return transformPoint(this.dicomToThreeMatrix, point);
  }

  dicomVectorToThree(vector) {
    return transformVector(this.dicomToThreeMatrix, vector);
  }

  cameraInputToDicom(input) {
    return {
      eye: this.threePointToDicom(input.eye),
      target: this.threePointToDicom(input.target),
      up: this.threeVectorToDicom(input.up),
      n: this.threeVectorToDicom(input.n),
    };
  }
}
