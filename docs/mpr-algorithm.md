# MPRアルゴリズム設計

1. 複数DICOMを読み込み、DICOM preambleと主要タグを検査する。
2. `Image Orientation Patient` のrow/column方向からslice法線を作る。
3. `dot(Image Position Patient, sliceNormal)` でスライス位置を求め、物理位置順にソートする。
4. Stored Valueを `HU = value * RescaleSlope + RescaleIntercept` でCT値へ変換し、`Int16Array` に詰める。
5. `eye`、`target`、`up` から正規直交基底 `u`、`v`、`n` を作る。
6. 各MPR画素の物理座標を算出する。
7. 物理座標を連続voxel座標へ逆アフィン変換する。
8. 最近傍又は三線形補間でHU値を取得する。範囲外と非数は `-1024 HU` にする。
9. Window Level / Widthで8-bitグレースケールへ変換し、Canvasへ描画する。

CPU実装のため、標準CTでの応答が不足する場合は次段階でWeb Worker、WASM、WebGL2/WebGPUを検討します。
