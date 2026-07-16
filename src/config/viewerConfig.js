export const viewerConfig = {
  three: {
    fov: 45,
    near: 0.1,
    farPaddingFactor: 8,
    background: 0x101820,
    rotateSpeed: 0.8,
    panSpeed: 0.2,
    zoomSpeed: 0.9,
    inertia: 0.8,
    maxPixelRatio: 2,
    interactivePixelRatio: 1,
    qualityRestoreDelayMs: 180,
  },
  mapper: {
    // Identity means Three.js world units are DICOM patient-space millimeters.
    threeToDicomMatrix: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ],
  },
  dynamicMpr: {
    enabledByDefault: true,
    minUpdateIntervalMs: 80,
    performanceWarnMs: 200,
  },
};
