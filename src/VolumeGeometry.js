export function volumeCenter(metadata) {
  const [width, height, depth] = metadata.dimensions;
  const [spanX, spanY, spanZ] = volumeVoxelCenterSpan(metadata);
  const d = metadata.direction;

  // DICOMのImage Position PatientやMHAのOffset/Originは、先頭voxel中心のworld座標として扱う。
  // そのためVolume中心は dimensions * spacing / 2 ではなく、
  // index範囲 0..(dimension-1) の中央、つまり (dimension-1) * spacing / 2 で求める。
  return [
    metadata.origin[0] + d[0] * spanX / 2 + d[1] * spanY / 2 + d[2] * spanZ / 2,
    metadata.origin[1] + d[3] * spanX / 2 + d[4] * spanY / 2 + d[5] * spanZ / 2,
    metadata.origin[2] + d[6] * spanX / 2 + d[7] * spanY / 2 + d[8] * spanZ / 2,
  ];
}

export function volumeVoxelCenterSpan(metadata) {
  return metadata.dimensions.map((dimension, index) => Math.max(0, dimension - 1) * metadata.spacing[index]);
}

export function volumeDisplaySpan(metadata) {
  return volumeVoxelCenterSpan(metadata).map((span, index) => {
    // 1スライスや1画素幅のVolumeでは中心間距離が0になる。Boxを可視化するためだけに
    // 最小1 voxelぶんの厚みを持たせるが、MPR計算で使うworld座標はvolumeCenterの値を正とする。
    if (span > 0) return span;
    return metadata.spacing[index] || 1;
  });
}
