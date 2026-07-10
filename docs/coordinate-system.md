# 座標系定義

## 配列

ボリューム配列は一次元 `Int16Array` で保持し、論理順序は `volume[z][y][x]` です。

```text
index = z * width * height + y * width + x
```

`x` はDICOM画像の列、`y` は行、`z` はスライス順を表します。

## DICOM患者座標

DICOMの `Image Position Patient`、`Image Orientation Patient`、`Pixel Spacing` を基準にします。

- `origin`: 物理順に並び替えた先頭スライスの `Image Position Patient`
- `direction`: 列ベクトルとして `[rowDirection, columnDirection, sliceDirection]` を持つ3x3行列
- `spacing`: `[columnSpacing, rowSpacing, sliceSpacing]`

JavaScript配列上は行優先の9要素配列として保持します。

```text
direction = [
  row.x, column.x, slice.x,
  row.y, column.y, slice.y,
  row.z, column.z, slice.z
]
```

## 物理座標からvoxel座標

MPR上の物理座標 `X` は以下で連続voxel座標へ変換します。

```text
relative = X - origin
oriented = inverseDirection * relative
voxel = oriented / spacing
```

## MPR断面

```text
n = normalize(target - eye)
u = normalize(cross(up, n))
v = normalize(cross(n, u))
center = target + normalOffset * n
```

`up` と `n` が平行に近い場合は代替upを使います。画素中心は以下です。

```text
offsetU = (i - (width - 1) / 2) * spacingU
offsetV = (j - (height - 1) / 2) * spacingV
X(i,j) = center + offsetU * u + offsetV * v
```

## Three.js連動

初期設定では、Three.jsワールド座標の1単位を1 mmとし、Three.js座標とDICOM患者座標は同じ向きの恒等変換として扱います。

```text
Pdicom = threeToDicomMatrix * Pthree
```

`threeToDicomMatrix` は [viewerConfig.js](/Users/daisuketanii/Library/Mobile%20Documents/com~apple~CloudDocs/splatClone/browser-mpr/src/config/viewerConfig.js) で一元管理します。点と方向ベクトルは分けて変換します。

```javascript
threePointToDicom(point)    // 平行移動を含む
threeVectorToDicom(vector)  // 平行移動を含まない
dicomPointToThree(point)
dicomVectorToThree(vector)
```

Three.js側の `up` は固定の `camera.up` ではなく、カメラ姿勢を反映した以下の画面上方向です。

```javascript
upThree = [0, 1, 0] applied by camera.quaternion
```

Volume範囲BoxはDICOM/MHAボリュームの物理中心、物理寸法、方向行列に合わせてThree.js空間へ配置します。

重要: `origin` は先頭voxel中心のworld座標として扱います。そのためVolume中心やBoxの座標基準は `dimensions * spacing / 2` ではなく、voxel中心の範囲である `(dimensions - 1) * spacing / 2` です。これによりMPRでサンプルするCT/MHA画像座標と、Three.js上のモデル座標が同じworld座標を参照します。

座標整合済みOBJ/GLBを読み込む場合は、モデル頂点がすでにVolume world座標にあるものとして扱います。`注視点へ中心合わせ` はOFF、`Scale = 1`、`位置 = [0,0,0]`、`回転 = [0,0,0]` にしてください。ONにするとモデルのバウンディングボックス中心を現在の注視点へ移動するため、元のCT/MHA座標は保持されません。
