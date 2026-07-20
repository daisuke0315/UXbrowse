# MPRアルゴリズム設計

## 処理手順

1. DICOMまたはMHAからVolumeと座標情報を読み込みます。
2. DICOMはslice法線上の物理位置で並べ替えます。
3. DICOM Stored ValueをSlope／Interceptで実値へ変換します。
4. `eye / target / up`からMPR基底`u / v / n`を作ります。
5. 出力画像の各画素に対応する物理座標を計算します。
6. 物理座標を連続voxel座標へ逆変換します。
7. 最近傍または三線形補間で値を取得します。
8. 範囲外または無効値は既定値`-1024`として扱います。
9. Window Level／Widthで8-bit画像へ変換します。
10. 同じ画像をMPRサブ画面とThree.js重畳面へ表示します。

## 断面位置

```text
center = target + normalOffset * n
pixel = center + offsetU * u + offsetV * v
```

- 視線連動では位置と方向がカメラに追従します。
- Axial／Coronal／Sagittalでは方向だけを固定し、位置は注視点へ追従します。
- FOVは`画像幅 × spacingU`、`画像高さ × spacingV`です。

## 実装上の制限

- MPR生成は現在CPU処理です。
- 更新要求は`requestAnimationFrame`と最小更新間隔で集約します。
- 大規模データで不足する場合は、Worker、WASM、WebGL2、WebGPUを検討します。
