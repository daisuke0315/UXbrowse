# iOS 27 Object Tracking移行契約

## ゴール

Create MLで3D digital twinから生成した`.referenceobject`を認識し、ARKitから得た6DoF poseをViewSync/CT座標へ接続する。CT/MPRはこの座標chainと物理精度の検証に合格した場合だけ表示する。

## 資産

- 学習用: `data/260911/260108-3-training.usdz`（元の8192×8192 texture）
- 実行表示用: `data/260911/260108-3-runtime.usdz`（4096×4096 texture）
- 学習済み参照: `data/260911/260911.referenceobject`（Upright）
- 座標契約: `data/260911/model-coordinate.json`
- Create ML project: `ViewSync260911ObjectTracker.mlproj`

2つのUSDZは同じroot transformを持つ。

```text
p_tracking_m = 0.001 * (p_glb_mm - [10.0868454, -170.5942688, -579.3869629])
```

tracking座標のAABBは次の通り。

```text
min = [-0.1088395, -0.1030565, -0.1256737] m
max = [ 0.1088395,  0.1030565,  0.1256737] m
```

## Create ML（完了）

`Tools/train_object_tracker.command`で`Upright`学習を完了した。生成物の検証結果は次の通り。

- `visibleName`: `260911`
- 埋め込み`object.usdz`のSHA-256: `24b932803e696e48a2e3fd46b10fc3239666414c8985be0a8c5df3154f21687e`
- 学習入力USDZと埋め込みUSDZのSHA-256は完全一致
- 埋め込みAABB: `[-0.10883951, -0.10305649, -0.12567377]`〜`[0.10883951, 0.10305649, 0.12567368]` m
- `model-coordinate.json`との差はfloat丸め範囲内
- `usdchecker --arkit`成功

## iOS 27ランタイム

`.referenceobject`をbundleへ追加すると、現在のコードもiOS 27上では旧`ARObjectScanningConfiguration`と端末内`260911.arobject`より学習済み参照を優先して読み、`ARWorldTrackingConfiguration.detectionObjects`へ設定する。Xcode 27 SDKへ更新後、移動対象も要件に含む場合だけ`trackingObjects`へ切り替える。

- 静止した対象: `detectionObjects`で初期poseを得て、その後はARKit VIOでworld poseを維持する。
- 動かす可能性がある対象: `trackingObjects`を使い、選択したvideo formatの各frameでobject poseを更新する。
- `ARObjectAnchor.transform`を`T_world_reference`として扱う。
- 埋め込みUSDZとruntime USDZの座標一致を確認後、`T_world_GLB = T_world_reference * T_reference_GLB`を構成する。
- `T_reference_GLB`を暗黙のidentityにしない。検証結果を固定値として保存する。

## 表示許可条件

以下をすべて満たすまでは透明モデルだけを表示し、MPRを出さない。

1. `.referenceobject`が認識され、anchorがtracking中。
2. reference objectのscale/origin/axesが`model-coordinate.json`と一致。
3. ARKit camera trackingがnormal。
4. 物理ファントム上の既知ランドマークでTREを測定し、合意した許容値以下。

## 受入テスト

- 初回認識時間と再認識時間をログへ記録。
- 30°刻みの視点と複数距離でrecognition success rateを測定。
- 既知ランドマーク3点以上で各軸のTREを測定。
- 30秒静止時と周回時のpose jitter/driftを測定。
- ARKit camera trackingがlimited/notAvailableになった場合、MPRが即時停止すること。
- reference/runtime USDZのAABB 8隅が同じworld位置へ写ること。
