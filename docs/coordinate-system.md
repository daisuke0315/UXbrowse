# 座標系定義

## 基本規約

- Volumeは一次元TypedArrayで保持し、論理順序は`volume[z][y][x]`です。
- `x`は画像の列、`y`は行、`z`は物理位置順のスライスです。
- インデックスは`z * width * height + y * width + x`です。
- 長さの単位はmm、Three.jsの1 world unitも1 mmです。

## DICOM患者座標

- `origin`：先頭voxel中心の`Image Position Patient`
- `direction`：row、column、slice方向を列ベクトルに持つ3×3行列
- `spacing`：`[columnSpacing, rowSpacing, sliceSpacing]`
- slice方向：`cross(rowDirection, columnDirection)`
- slice順：`dot(Image Position Patient, sliceDirection)`で並べ替えます。

## voxel座標への変換

```text
relative = worldPoint - origin
oriented = inverse(direction) * relative
voxel = oriented / spacing
```

- `origin`はvoxelの角ではなく中心です。
- Volumeの中心範囲は`(dimensions - 1) * spacing`で計算します。

## MPR平面

```text
n = normalize(target - eye)
u = normalize(cross(up, n))
v = normalize(cross(n, u))
center = target + normalOffset * n
```

- `u`は画面横、`v`は画面縦、`n`は断面法線です。
- `up`と`n`が平行に近い場合は代替upを使用します。
- 画素中心は`center + offsetU * u + offsetV * v`で求めます。

## Three.jsとの対応

- 既定の`threeToDicomMatrix`は恒等変換です。
- 点の変換には平行移動を含め、方向ベクトルの変換には含めません。
- カメラの画面上方向は`camera.quaternion`を反映して求めます。
- 座標整合済みモデルは`Scale = 1`、位置・回転を`0`のまま読み込みます。
- X／Y／Z反転は確認用の手動補正であり、正しい座標整合の代替ではありません。

設定値は[`src/config/viewerConfig.js`](../src/config/viewerConfig.js)で管理します。
