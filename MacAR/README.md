# Object-only 3D Map 作成ツール

RealityCapture / RealityScanからCOLMAP Text形式で出力したデータを使い、次の2工程を実行するPythonツールです。

1. `Points3D.txt`を指定した直方体（AABBまたはOBB）で抽出する
2. 抽出した3D点に対応する画像上の`(u,v)`を`Images.txt`から抽出する

どちらの工程でも`POINT3D_ID`は再採番せず、元のCOLMAPファイルは変更しません。

## セットアップ

Python 3.9以降を使用します。`MacAR`フォルダで次を実行してください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

OpenCVとNumPyはプレビュー／3Dビューに使用します。テキストの抽出処理自体はPython標準ライブラリだけで動作します。

## Step 1: 入力点群の範囲を確認

```bash
python3 object_point_filter.py analyze --input colmap/Points3D.txt
```

点数、COLMAP XYZの最小値・最大値、全体boundsのcenter/sizeを表示します。これはフォーカス領域を決めるための補助情報です。

## Step 1: RealityCaptureの領域値で抽出

`filter`のCenter / Size / Rotationは、既定でRealityCapture座標として受け取ります。内部で実データから確認した次の軸変換を適用します。

```text
(X, Y, Z) RealityCapture -> (X, -Z, Y) COLMAP
```

RCで表示された値をそのまま入力できます。

```bash
python3 object_point_filter.py filter \
  --input colmap/Points3D.txt \
  --output-dir output \
  --center 4.59766 -201.5511 -591.8713 \
  --size 209.423 176.29451 261.861969 \
  --preview
```

`--rotation`を省略するとRC側でRotationが0のboxとして扱います。COLMAP空間では軸変換後のboxになります。

すでにCOLMAP座標へ変換済みの値を直接入力する場合だけ、次を追加します。

```bash
--input-coordinates colmap
```

## Step 1: OBBで抽出

Euler角を度で指定する例：

```bash
python3 object_point_filter.py filter \
  --center Cx Cy Cz \
  --size Sx Sy Sz \
  --rotation Rx Ry Rz \
  --preview \
  --visualize
```

回転規約は入力座標内で`R = Rz @ Ry @ Rx`となる、ローカル座標からワールド座標へのactive rotationです。RC入力の場合はこの回転へRC→COLMAP変換を合成します。点の判定時には変換後の`R^T (p - C)`でローカル座標へ戻します。角度の既定単位は度です。ラジアンの場合は`--radians`を追加してください。

回転行列を直接指定する場合は、ローカル→ワールドの直交行列をrow-majorで9値渡します。

```bash
python3 object_point_filter.py filter \
  --center Cx Cy Cz --size Sx Sy Sz \
  --rotation-matrix R00 R01 R02 R10 R11 R12 R20 R21 R22
```

## Step 1の出力

`--output-dir`（既定は`MacAR/output`）へ次を作成します。

- `filtered_Points3D.txt`: 領域内の点だけを含むCOLMAP互換ファイル
- `selected_point_ids.txt`: 採用した`POINT3D_ID`（入力順、再採番なし）
- `filter_summary.txt`: 点数、比率、フォーカス領域、回転、QC結果
- `point_cloud_preview.png`: `--preview`または`--visualize`指定時のQC画像

抽出点が0の場合や入力形式が不正な場合はエラー終了し、既存の正常な出力ファイルを置き換えません。

## Step 1の3D QCビュー

`--visualize`を付けると、PNG保存後にOpenCVウィンドウを開きます。

- 左ドラッグ: 視点回転
- マウスホイール: ズーム
- `A`: 除外点（グレー）の表示切替
- `F`: フォーカスboxにフィット
- `G`: サンプルした点群全体にフィット
- `R`: 視点をリセット
- `S`: 現在のビューをPNGへ保存
- `Q` / `Esc`: 終了

黄色の線がフォーカスbox、元のRGBで描かれた点が採用点、グレーが除外点です。大規模点群でも操作できるよう、表示だけは採用／除外からそれぞれ最大10万点をreservoir samplingします。抽出ファイルには全採用点が入ります。

## Step 1の自動QC

処理完了前に次を検証します。

1. `Input = Selected + Rejected`
2. 出力の全点がフォーカス領域内
3. `selected_point_ids.txt`と出力点IDの一致
4. 採用レコードのSHA-256が入力時と出力時で一致（XYZ/RGB/ERROR/TRACKを含む元行が不変）

QC4（背景点が大量に残っていないか）は`point_cloud_preview.png`または3Dビューで目視確認してください。点群自体の座標は変更しません。RC入力値はboxだけをCOLMAP世界座標へ変換し、採用したCOLMAPレコードは元のまま出力します。

## Step 2: 2D–3D observationを抽出

Step 1が完了した状態なら、`MacAR`フォルダで次を実行します。

```bash
source .venv/bin/activate
python3 extract_object_observations.py
```

既定では以下を入力として使用します。

- `colmap/Images.txt`
- `colmap/Cameras.txt`
- `output/selected_point_ids.txt`
- `output/filtered_Points3D.txt`
- QC用画像: `image/`以下にある、`Images.txt`の`NAME`と同名・同寸法の画像

`Images.txt`は画像単位でストリーミング処理するため、ファイル全体をメモリへ読み込みません。

### 現在の連番JPEGで近似QCを作る場合

現在の`image/output_image2_0001.jpg`形式を利用する場合は、次を実行します。

```bash
python3 extract_object_observations.py \
  --image-template 'output_image2_{index:04d}.jpg' \
  --image-index-offset 1 \
  --allow-rescaled-images
```

このJPEGは`Cameras.txt`記載の寸法と異なるため、座標を線形拡大して描く近似表示です。画像には赤い`APPROXIMATE`バナーが入ります。画素単位で正確なQCには、COLMAPの`Images.txt`が参照する画像（例: `00000.png`）を同じ寸法のまま用意し、次のように指定してください。

```bash
python3 extract_object_observations.py --image-dir /path/to/exported/images
```

可視化だけ不要な場合は`--no-visualization`を追加します。

### Step 2の出力

`output/`へ次を作成します。

- `object_observations.csv`: 1行1 observation。画像ID、名前、カメラID、姿勢、`u,v`、`POINT3D_ID`、`X,Y,Z`を保持
- `image_observation_counts.csv`: 画像別observation数
- `observation_summary.txt`: 全体集計、画像別集計、QC結果
- `out_of_bounds_observations.csv`: `Cameras.txt`の画像範囲外だったobservationの確認用一覧
- `observation_qc/`: observation数が少・中・多の3画像に採用点を重畳したQC画像

範囲外observationは元データを黙って変更しないため、`object_observations.csv`から削除しません。該当行を`out_of_bounds_observations.csv`にも記録し、サマリーのQC3を`FAIL`にします。次工程で除外するかは、このレポートを確認して判断してください。

## Step 2の自動QC

1. CSV内の全`POINT3D_ID`がStep 1の選択IDに含まれる
2. 同一`POINT3D_ID`のXYZが`filtered_Points3D.txt`と一致する
3. `(u,v)`が`Cameras.txt`の`WIDTH, HEIGHT`内にあるかを全件確認する
4. 画像別observation数を0件の画像も含めて記録する

## Step 3: Object-only 3D Feature Mapを生成

Step 2で生成した`object_observations.csv`と、RC/RSのalignmentで使用した元画像からSIFT descriptorを生成します。

元画像が`Images.txt`の`image_name`と同じ名前で`image/`にある場合は、次で実行します。

```bash
source .venv/bin/activate
python3 build_object_feature_map.py
```

RC observationとSIFT keypointの許容距離は既定で3 pixelです。変更する場合は次のように指定します。

```bash
python3 build_object_feature_map.py --max-distance 5
```

処理は画像を1枚ずつ読み、画像全体でSIFTを1回だけ計算します。各RC observationに最も近いSIFT keypointが許容距離内にある場合だけ対応を保存します。同じSIFT keypointを複数observationへ割り当てず、競合時は距離が最小の組を採用します。同じ`POINT3D_ID`が複数画像から得たdescriptorは平均化せず、別レコードとして保持します。

### Step 3の画像要件

Step 3では画像の近似リサイズを許可しません。画像の幅と高さが`Cameras.txt`と完全一致する場合だけSIFTを計算します。

現在の`output_image2_0001.jpg`形式を検査対象にするには次の指定ができます。

```bash
python3 build_object_feature_map.py \
  --image-template 'output_image2_{index:04d}.jpg' \
  --image-index-offset 1
```

ただし、現在配置されているJPEGは3840×2160、`Cameras.txt`側は画像ごとに約3720×2090であり一致しません。そのため全画像が`IMAGE_SIZE_MISMATCH`となり、descriptorは生成されません。RC/COLMAPエクスポート時に使われた同名・同寸法画像（例: `00000.png`）を用意してください。別フォルダなら次のように指定します。

```bash
python3 build_object_feature_map.py --image-dir /path/to/exact/rc-images
```

### Step 3の出力

- `object_features.npz`: `point3d_ids`, `xyz`, 128次元`descriptors`, `image_ids`, `rc_uv`, `sift_uv`, `pixel_distances`, `scales`, `angles`, `responses`, `point_observation_counts`
- `object_features_metadata.csv`: descriptorを除いた、人間が確認しやすい対応情報
- `feature_summary.txt`: 取得率、3D点数、画像エラー、距離統計
- `feature_failures.csv`: 画像なし、寸法不一致、SIFT失敗などの画像別エラー
- `feature_unmatched_observations.csv`: 近傍SIFTなし、割当競合、座標範囲外のobservation
- `feature_qc/`: RC点、対応SIFT点、両者を結ぶ線を重畳した少・中・多のQC画像
- `feature_spatial_distribution.png`: descriptor付き3D点の空間分布

画像単位のエラーでは全体を停止せず、次の画像へ進みます。descriptorが1件も生成されなかった場合も診断ファイルは保存しますが、コマンドは終了コード2で未完成を通知します。

### 現在の`colmap_image/`データで再生成

`colmap_image/`には画像と、それらに対応する別エクスポートのCOLMAPパラメータが入っています。旧パラメータ由来のStep 1・2出力とは混在させず、次の順序で`output_colmap_image/`へ再生成します。

```bash
python3 object_point_filter.py filter \
  --input colmap_image/Points3D.txt \
  --output-dir output_colmap_image \
  --center 4.59766 -201.5511 -591.8713 \
  --size 209.423 176.29451 261.861969 \
  --preview

python3 extract_object_observations.py \
  --images colmap_image/Images.txt \
  --cameras colmap_image/Cameras.txt \
  --selected-ids output_colmap_image/selected_point_ids.txt \
  --filtered-points output_colmap_image/filtered_Points3D.txt \
  --output-dir output_colmap_image \
  --image-dir colmap_image

python3 build_object_feature_map.py \
  --observations output_colmap_image/object_observations.csv \
  --cameras colmap_image/Cameras.txt \
  --image-dir colmap_image \
  --output-dir output_colmap_image
```

## Step 4: オフラインSIFT Matching / RANSAC-PnP試験

まず既知姿勢を持つ参照画像をクエリとして使い、その画像自身から登録されたdescriptorだけをFeature Mapから除外するleave-one-image-out試験を行います。他画像から得たdescriptorだけで再localizeし、RC姿勢との誤差を測定できます。

```bash
python3 offline_pnp_test.py \
  --query-image 00414.png \
  --query-image 00366.png \
  --query-image 00240.png
```

処理内容は、クエリ画像のSIFT抽出、Feature MapとのFLANN matching、異なる3D点を使ったLowe ratio test、3D点の重複排除、RANSAC-PnP、inlierによるLM pose refinementです。同一クエリ画像由来のmap descriptorは既定で除外します。

既定の合格条件は次の通りです。

- Lowe ratio: 0.75未満
- RANSAC再投影誤差: 4px以内
- PnP inlier: 20点以上
- inlier率: 20%以上

出力は`output_colmap_image/pnp_test/`へ作成します。

- `localization_results.csv`: 画像別の成否、match数、inlier数、推定姿勢、既知姿勢との誤差
- `localization_matches.csv`: PnPへ渡した2D–3D対応と再投影誤差
- `localization_summary.txt`: 全体集計と画像別結果
- `qc/*_pnp.jpg`: 緑がRANSAC inlier、赤がoutlierの重畳画像

### 本当に新しく撮影した画像

`Images.txt`に存在しない画像には既知のカメラ内部パラメータがないため、`fx fy cx cy`を明示します。この場合は正解姿勢がないため、inlier数・inlier率・再投影誤差で成否を判定します。

```bash
python3 offline_pnp_test.py \
  --query-dir /path/to/new-images \
  --query-image query.jpg \
  --intrinsics FX FY CX CY \
  --distortion K1 K2 P1 P2 K3
```

新規画像をリサイズ・クロップ・回転した場合は、その画像に対応する内部パラメータも同じpixel geometryへ変換する必要があります。

## Step 5: iPhone連係カメラでリアルタイムPnP + MPR

iPhoneをmacOSの連係カメラとして接続すれば、専用iOSアプリや画像転送サーバーは不要です。OpenCVがMacのカメラ入力を直接読み、SIFT matching、RANSAC-PnP、MHAのoblique MPR生成、カメラ画像への重畳までMac上で行います。

### 1. カメラ番号を確認

iPhoneを連係カメラとしてMacへ接続した状態で、`MacAR`フォルダから実行します。

```bash
source .venv/bin/activate
python3 realtime_pnp.py --list-cameras
```

macOSからカメラアクセス確認が表示された場合は許可します。見つからない場合は「システム設定 → プライバシーとセキュリティ → カメラ」で、使用しているTerminalまたはPythonにカメラアクセスを許可してください。既定ではFFmpegのAVFoundation入力を使い、番号とデバイス名を表示します。OpenCVのAVFoundation入力は連係カメラの列挙が不安定なため、カメラ取得には使用せず、取得後のPnP・MPR処理だけをOpenCVで行います。FFmpegがない場合は`brew install ffmpeg`で導入できます。

### 2. まず近似内部パラメータでライブ確認

```bash
python3 realtime_pnp.py --camera 0
```

既定ではカメラ固有のネイティブ解像度で取得し、SIFT/PnPへ渡す直前にアスペクト比を維持したまま横1280pxへ縮小します。処理上限は10fps、水平画角は65度の近似値です。現在のiPhone連係カメラでは、入力1920×1440、処理1280×960になります。表示ウィンドウで`Q`または`Esc`を押すと終了します。別のカメラ番号なら`--camera 1`などに変更します。

画面上部にはPnPの`matches`、`inliers`、MPR状態に加えて、カメラFPS、実処理FPS、最新フレームへ追いつくために捨てたフレーム数、キュー滞留時間、アプリ内遅延、SIFT・matching・PnP・MPR・3Dデバッグ描画の個別時間、入力/処理解像度、判定閾値を表示します。`app latency`はFFmpegでフレームを復号してから表示処理が終わるまでの値で、iPhoneの撮像・エンコード・無線転送時間は含みません。`latest-frame drops`は異常ではなく、古い映像を順番に処理して遅延が増えないよう意図的に破棄した累積数です。緑点がRANSAC inlier、赤点がoutlier、黄色枠内の半透明グレースケール画像がMPRです。

3Dマッチ範囲のデバッグ画面は既定で有効ですが、PnP処理を妨げないよう2Hzだけ更新します。不要なら`--no-3d`で停止できます。さらに低遅延を優先する場合は次を使います。

```bash
python3 realtime_pnp.py --camera 1 --max-processing-width 960 --process-fps 15 --no-3d
```

縮小しすぎるとSIFTのマッチ数が減るため、まず既定の1280pxで`SIFT`、`match`、`inliers`を確認し、余裕がある場合だけ960pxへ下げます。元解像度のまま処理する場合は`--max-processing-width 0`です。

同時に`ViewSync Live 3D Match Coverage`ウィンドウを表示します。灰色がFeature Map全体、緑が現在のPnP inlier、赤がoutlier、オレンジがPnP未成立の候補match、黄色線が対象boxです。`A/D`で水平回転、`W/S`で上下回転、`R`で初期視点へ戻せます。不要な場合だけ`--no-3d`を付けます。

### 3. 正確な重畳用にカメラをキャリブレーション

連係カメラを通常のMacカメラとして読む場合、OpenCVはiPhoneの正確な`fx, fy, cx, cy`を取得できません。近似値のままでもlocalization試験はできますが、MPRの画素位置と位置誤差の評価には使えません。

9×6の「内側コーナー」を持つチェッカーボードを用意し、次を実行します。

```bash
python3 calibrate_camera.py --camera 0
```

ボードの距離、傾き、画面内位置を変えながら、検出枠が出た状態でSpaceを15回押します。`C`で早期確定、`R`で撮り直し、`Q`で中止できます。結果は`output_colmap_image/iphone_camera_calibration.npz`へ保存されます。

その後は同じカメラ、同じ解像度、同じズームで次を実行します。

```bash
python3 realtime_pnp.py \
  --camera 0 \
  --calibration output_colmap_image/iphone_camera_calibration.npz
```

カメラのズーム、解像度、アスペクト比、手ぶれ補正によるクロップが変わると内部パラメータも変わります。精度確認時はこれらを固定してください。本出力は現段階では研究・QC用であり、臨床判断には使用しません。

### 主な調整

```bash
# 内部パラメータが既知の場合
python3 realtime_pnp.py --camera 0 --intrinsics FX FY CX CY

# CT表示のwindow level / widthと透明度
python3 realtime_pnp.py --camera 0 --window-level 40 --window-width 400 --mpr-opacity 0.7

# PnPだけを確認しMPRを非表示
python3 realtime_pnp.py --camera 0 --no-mpr
```

## テスト

```bash
python3 -m unittest discover -s tests -v
```
