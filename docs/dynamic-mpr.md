# Three.js連動MPR

## 流れ

1. Three.jsの `PerspectiveCamera` と `OrbitControls` を初期化する。
2. DICOM / MHA読込又はデモファントム生成後、Volume物理範囲に一致するBoxを配置する。
3. MPR注視点マーカーを `camera.position + camera.forward * MPR距離` に同期する。
4. カメラ位置、MPR注視点、画面上方向を取得する。
5. `ThreeDicomCoordinateMapper` でVolume world座標へ変換する。
6. 変換後の `eyeVolume / targetVolume / upVolume` を既存 `MprRenderer` へ渡す。
7. `OrbitControls` の `change` イベントを `MprUpdateScheduler` で集約し、多重更新を避ける。

## デバッグ

画面下部に以下を表示します。

- `eyeThree / targetThree / upThree / nThree`
- `eyeVolume / targetVolume / upVolume / nVolume / uVolume / vVolume`
- 座標取得、変換、MPR生成、Canvas描画、総更新時間
- 範囲外画素数、無効値数

`デモファントム` は実DICOMがない状態でもThree.js連動とMPR更新を確認するための合成ボリュームです。

## MPR注視点の正規仕様

ViewSync MPRでは、MPR断面中心として使う `targetThree` は `OrbitControls.target` でもモデル中心でもありません。

```text
targetThree = camera.position + camera.forward * mprDistance
```

ズーム操作はカメラ位置の移動として扱います。そのため、カメラが近づくと `targetThree` も移動し、MPR断面位置も更新されます。`OrbitControls.target` は回転中心、`modelCenterThree` はモデル位置確認用の参照値として扱います。

## 3Dモデル

`OBJLoader` / `GLTFLoader` で単体 `.obj` / `.glb` を読み込み、Three.jsシーンへ追加します。`モデル連動MPR` がONの場合でも、モデル中心をMPR中心にはしません。MPR中心は常にカメラ前方固定距離の `targetThree` として計算し、モデル中心は `modelCenterThree` としてデバッグ表示します。

既定ではモデルの元座標を保持します。CT/MHA座標と一致済みのモデルを確認する場合は `注視点へ中心合わせ` をOFF、Scale/位置/回転を初期値のままにします。`注視点へ中心合わせ` をONにすると、座標を持たない確認用モデルを現在の注視点へ仮配置します。
