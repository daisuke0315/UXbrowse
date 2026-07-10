# ViewSync MPR

外観モデル連動MPRビューア

ブラウザ内だけで非圧縮CT DICOMシリーズ又は単一`.mha`ボリュームを読み込み、`eye`、`target`、`up` から任意断面MPRを生成する試作です。Three.jsのカメラ、OrbitControls、カメラ前方固定距離のMPR注視点、読み込んだ3DモデルにMPRを連動させます。外部サーバー送信、IndexedDB、LocalStorage保存は行いません。

## 起動

`browser-mpr/index.html` をChromeで開きます。ローカルファイルのES Modulesが制限される環境では、任意の静的サーバーで `browser-mpr/` を配信してください。

```bash
cd browser-mpr
python3 -m http.server 5173
```

その場合は `http://localhost:5173/` を開きます。

Three.js本体、OrbitControls、OBJLoader、GLTFLoaderは `vendor/` にローカル配置しているため、起動後はCDNに依存しません。

## 試し方

実データがない場合は `デモファントム` を押すと、合成CTボリューム、Volume範囲Box、動的MPRをすぐ確認できます。

実DICOMで試す場合は、`DICOMファイル` 又は `フォルダ` から同一CTシリーズを選択します。MHAで試す場合は、`MHAファイル` から `ElementDataFile = LOCAL` の単一`.mha`を選択します。読込後、左のThree.js表示をドラッグ、右ドラッグ、ホイール操作すると、カメラの `eye / target / up` がVolume座標へ変換され、MPRが更新されます。

3Dモデルは `OBJモデル` 又は `GLBモデル` から読み込みます。`モデル連動MPR` がONの場合でも、MPR中心はモデル中心固定ではなく `camera.position + camera.forward * MPR距離` で定義します。ズームでカメラ位置が変わるとMPR中心も移動します。モデル中心はデバッグ用の参照値として表示します。

## 注視点仕様

ViewSync MPRのMPR注視点は、Three.js標準の `OrbitControls.target` でも、3Dモデルのバウンディングボックス中心でもありません。現在の正規仕様は以下です。

```text
MPR注視点 = camera.position + camera.forward * MPR距離
```

この仕様により、ズーム操作でカメラがモデルへ近づくと、カメラ前方に固定されたMPRプレーンもCT/MHA空間内で移動します。これはUnity版の「カメラ前方に固定したプレーンをtranslateして重畳する」挙動に合わせるための仕様です。

用語は次のように分けます。

- `OrbitControls.target`: 3D操作の回転中心です。MPR中心ではありません。
- `modelCenter`: OBJ / GLBモデルの現在のワールド中心です。MPR中心ではなく、位置確認用の参照値です。
- `targetThree` / `targetVolume`: MPR注視点です。`camera-forward` 系の値として計算し、MPR断面中心になります。
- `注視点距離/MPR距離`: カメラ位置からMPR注視点までの距離です。

## 現在の確定状態

2026-07-10時点で、以下の動作を現状仕様として保存しています。

- DICOM / MHAボリューム読込後、Three.jsカメラ操作にMPRが追従します。
- OBJ / GLBモデル読込後も、Three.jsカメラ操作はカメラ基準のMPR更新として動き続けます。
- Canvas上に生成したMPR画像は、Three.js空間内のMPRプレーンにもCanvasTextureとして重畳表示されます。
- MPRプレーンは `plane.center / u / v / n` をVolume座標からThree.js座標へ変換し、断面と同じ位置・姿勢で配置されます。
- MPRプレーンの表示、Opacity、境界線表示をMPRパネルから切り替えできます。
- `モデル連動MPR` がONの場合、MPR中心は `camera.position + camera.forward * MPR距離` で生成します。モデル中心はMPR中心ではなく、デバッグ欄の `modelCenterThree` として確認します。
- ズームでカメラ位置が動いた場合、MPR注視点とMPR断面も移動します。これはViewSync MPRの正規仕様です。
- `MPRプレーン前景透過` がONの場合、MPRプレーンよりカメラ手前側にあるOBJ / GLBメッシュをshaderで半透明にします。
- 前景透過の `前景Alpha`、`前景透過距離 mm`、`フェード幅 mm`、`完全切断` はMPRパネルから調整できます。`前景透過距離 = 0` の場合は、MPRプレーンよりカメラ手前側全体が透過対象です。
- MPRプレーン側では `CT値しきい値透過` により、指定したCT値以下の画素を透明化できます。初期値は `-500` です。
- `モデル連動MPR` をOFFにすると、MPR更新はカメラ基準へ戻ります。
- CT/MHA座標とモデル座標が合わない場合の手動確認用に、`CT座標反転` の `X反転`、`Y反転`、`Z反転`、`反転リセット` を使えます。
- CT座標反転はVolume中心まわりに適用されます。モデル座標は変更しません。
- 反転中の軸はボタンのON表示とデバッグ欄の `CT Axis Flip` で確認できます。

座標整合済みモデルを確認する時の基本設定は、`注視点へ中心合わせ OFF`、`Scale = 1`、`位置 = 0,0,0`、`回転 = 0,0,0` です。

## 対応範囲

- 単一フレームCT DICOM、非圧縮 Little Endian Explicit VR / Implicit VR、16-bit Pixel Data
- 同一SeriesのRows、Columns、Pixel Spacing、Image Orientation Patientが一致するDICOMシリーズ
- 単一`.mha`、`NDims = 3`、`BinaryData = True`、`CompressedData = False`、`ElementDataFile = LOCAL`
- MHA ElementType: `MET_SHORT`、`MET_USHORT`、`MET_FLOAT`、`MET_CHAR`、`MET_UCHAR`、`MET_INT`、`MET_UINT`、`MET_DOUBLE`
- Axial、Coronal、Sagittal、任意斜断面、Three.jsカメラ連動MPR、3Dモデル連動MPR
- MPR画像のThree.js空間内プレーン重畳表示
- MPRプレーンよりカメラ手前側にあるOBJ / GLBモデルのshader透過
- MPRプレーン上の任意CT値しきい値以下のalpha透過
- 最近傍補間、三線形補間、Window Level / Width、Min-Maxプリセット
- OBJ / GLBモデル読込、表示、中心合わせ、Scale、Opacity、位置、回転変更
- CT/MHA座標のX/Y/Z軸反転切り替え

JPEG/JPEG-LS/JPEG2000圧縮DICOM、Enhanced CT、圧縮MHA、`.mhd + .raw`、GLB外部ファイル、Draco/KTX2、PACS/DICOMweb、患者情報管理、保存機能は対象外です。

## モジュール

- `src/DicomLoader.js`: DICOM判定、タグ解析、Pixel Data展開
- `src/MhaLoader.js`: MHAヘッダ解析、LOCALバイナリ展開、Volume metadata生成
- `src/VolumeBuilder.js`: DICOMスライス検証、並び替え、HU変換、ボリューム構築
- `src/CoordinateTransformer.js`: Volume world座標と連続voxel座標の変換
- `src/MprPlaneCalculator.js`: `eye`、`target`、`up` から `n`、`u`、`v` を算出
- `src/Interpolator.js`: 最近傍補間、三線形補間
- `src/WindowingProcessor.js`: スカラー値から8-bitグレースケールへのWindow処理
- `src/MprRenderer.js`: MPR画像生成とCanvas描画
- `src/ValidationLogger.js`: ログ保持と画面表示
- `src/VolumeGeometry.js`: Volume中心、voxel中心範囲、表示用Box寸法の共通計算
- `src/three/ThreeSceneManager.js`: Three.jsシーン、Volume Box、補助表示、OBJ/GLBモデル表示
- `src/three/MprPlaneOverlay.js`: MPR CanvasTexture、MPRプレーンMesh、境界線表示の管理
- `src/three/ModelVisibilityShaderController.js`: MPRプレーン前景側モデル透過shaderの管理
- `src/three/ThreeCameraController.js`: `eyeThree`、`targetThree`、`upThree`、`nThree` 取得
- `src/three/ThreeDicomCoordinateMapper.js`: Three.js座標とVolume座標の点/ベクトル変換
- `src/three/DynamicMprController.js`: Three.js入力からMPR生成への連携
- `src/three/MprUpdateScheduler.js`: `requestAnimationFrame` による更新集約

## 3Dモデル

`OBJモデル` から `.obj`、`GLBモデル` から `.glb` を選ぶと、Three.jsシーンへ読み込みます。`3Dモデル` パネルで中心合わせ、Scale、Opacity、位置、回転、モデル連動MPRを調整できます。

Volumeと既に同じ座標系で作られたモデルをそのまま重ねたい場合は、`注視点へ中心合わせ` をOFF、`Scale = 1`、`位置 = 0,0,0`、`回転 = 0,0,0` にしてください。Three.jsの1単位は1 mmとして扱います。

GLB読込後も、通常のカメラ操作ではカメラ前方固定距離のMPRを更新します。`モデル連動MPR` がONでも、MPR中心はモデル中心固定にはしません。モデル中心は `modelCenterThree` としてデバッグ表示し、MPR中心は `targetThree` / `targetVolume` として表示します。

MPRプレーンがモデル内部に入り込む場合は、`前景透過` をONにしておくと、カメラ手前側のモデルだけが半透明になり、MPR断面を見やすくできます。`前景Alpha = 0` 又は `完全切断` で手前側をほぼ非表示にできます。前景透過は「プレーンから離れるほど不透明に戻す」のではなく、手前側を基本的に `前景Alpha` まで落とし、プレーン近傍だけ `フェード幅 mm` で滑らかに遷移させる実装です。

可視光モデルの断面にCT/MHA由来MPRを直感的に重ねたい場合は、`CT値しきい値透過` をONにして、空気や低吸収域に相当する値を透明化してください。例えばCTなら `-500` 付近を初期確認値として使えます。

OBJは単体 `.obj` のみ対応です。`.mtl`、OBJ外部テクスチャ、複数ファイル一括読込は未対応です。GLBは単一 `.glb` のみ対応で、外部 `.gltf + .bin + texture`、Draco、KTX2は未対応です。

## テスト

`browser-mpr/tests.html` を開くと、ベクトル計算、直交性、座標変換、補間、Window処理、Three.js-Volume座標変換、MHA Loaderの最小単体テストを実行します。
