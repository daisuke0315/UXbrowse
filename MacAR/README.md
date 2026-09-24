# Object-only 3D Map 作成ツール

現在の3DGSライブMPR構成・起動方法・ファイルの役割は、[SYSTEM_OVERVIEW.md](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/SYSTEM_OVERVIEW.md)にまとめています。以下には初期のCOLMAP処理や比較用の旧方式も含まれます。

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

## Reference-guided PnP静止画PoC

Feature Map全体への直接matchingでは反復模様の誤対応が増えるため、登録画像を介して2D–3D対応を作る静止画PoCを追加しています。

```bash
python3 reference_guided_pnp.py \
  --query /Users/daisuketanii/Downloads/IMG_2532.jpeg
```

処理は、候補登録画像の検索、live–reference SIFT matching、Fundamental Matrix RANSAC、POINT3D_ID転写、暫定姿勢または登録画像姿勢からの投影近傍再探索、RANSAC-PnPの順です。異なる候補画像の対応を無条件に混ぜず、最もPnP inlier支持が多い単一姿勢仮説を最終結果として選びます。

既定の`IMG_2532.jpeg`では、従来のglobal matchingが0〜7 inlierだったのに対し、登録画像`00314.png`を経由した仮説で71 correspondence中36 inlier、inlier率0.507、平均inlier再投影誤差3.44pxとなり、20 inlier / 0.20の条件をPASSしました。これは近似画角65度での成立確認であり、絶対的な位置精度を保証するものではありません。

出力は`output_colmap_image/reference_guided_pnp/`です。

- `summary.txt`: 最良姿勢とTop-K比較
- `reference_results.csv`: 候補画像ごとの検索・幾何検証・PnP件数
- `best_guided_matches.csv`: 最良仮説の2D–3D対応、再投影誤差、inlier判定
- `best_guided_overlay.jpg`: query画像上のinlier（緑）とoutlier（赤）
- `best_pose.npz`: `rvec`、`tvec`、近似camera matrix
- `qc/`: live–referenceのFundamental Matrix inlier表示

このPoCはまだ`realtime_pnp.py`へ接続していません。まず静止画で候補選択と姿勢仮説の成立を確認するための実装です。

## Step 4.5: GLB仮想Reference + ALIKED/LightGlue試験

現行のSIFT Feature Mapとリアルタイム経路を変更せず、別系統で次の新方式を試せます。

```text
GLB → 仮想Reference RGB/depth/XYZ
ライブ相当の静止画 → SIFTでReference候補検索
→ ALIKED + LightGlue → Fundamental RANSAC
→ Reference画素のGLB XYZを転写 → RANSAC-PnP + LM
```

使用するモデルは`data/260911/260108-3.glb`です。学習済みモデルとレンダラーの追加依存を一度だけ導入します。

```bash
source .venv/bin/activate
python3 -m pip install -r requirements-learned-matching.txt
```

まず軽量な8方向PoC Referenceを生成します。各ViewにつきPNGと、depth・画素ごとのGLB XYZ・K・camera poseを含むNPZを保存します。

```bash
python3 generate_glb_reference_views.py \
  --glb ../data/260911/260108-3.glb \
  --output output_glb_reference_views_poc \
  --azimuths 8 \
  --elevations 0 \
  --distances 2.7 \
  --width 640 \
  --height 480
```

macOSではOpenGLのWindowServer接続が必要なため、VS Codeの「実行とデバッグ」から`MacAR: Generate GLB Reference Views (PoC 8)`を選ぶ方法を推奨します。

生成後、`IMG_2532.jpeg`で新方式のPnPを実行します。

```bash
python3 glb_virtual_view_pnp.py \
  --query /Users/daisuketanii/Downloads/IMG_2532.jpeg \
  --views output_glb_reference_views_poc \
  --top-k 5
```

VS Codeでは`MacAR: GLB ALIKED LightGlue PnP`を選びます。結果は`output_glb_virtual_pnp/summary.json`、候補別の`view_results.csv`、対応線を描いた`best_matches.jpg`へ保存されます。推定カメラ位置はGLB元座標（このデータではmm）です。

PoC成立後の標準View数は、例えば`--azimuths 12 --elevations -30 0 30 --distances 2.5 3.3`の72 Viewです。View数を増やすほど視点差への耐性は上がりますが、生成容量と候補検索時間も増えます。

### 72 View生成とライブ実行

VS Codeの「実行とデバッグ」で、次の順に起動します。

1. `MacAR: Generate GLB Reference Views (72)`
2. `MacAR: GLB ALIKED LightGlue Live (72 Views)`

72 Viewは方位12方向、仰角−30/0/+30度、距離2段階で生成されます。ライブ処理は72枚すべてへLightGlueをかけず、各フレームのSIFT投票でTop-3を選択してからALIKED + LightGlueを実行します。Top-3から得た対応は、同じライブ特徴点の競合をLightGlue scoreで解決して統合し、個別候補と統合候補のうち再投影誤差が正常な最良PnPを採用します。姿勢がPASSした後は、その2D–3D inlierを既存のLucas–Kanade Optical Flowへ渡して追跡し、150処理フレームごと、または追跡点が8点未満になったときにGLB Reference検索へ戻ります。

Terminalからの同等コマンドは次です。

```bash
python3 generate_glb_reference_views.py \
  --glb ../data/260911/260108-3.glb \
  --output output_glb_reference_views_72 \
  --azimuths 12 \
  --elevations -30 0 30 \
  --distances 2.5 3.3 \
  --width 960 \
  --height 720

python3 main.py \
  --camera 1 \
  --matching-mode glb-track \
  --glb-views output_glb_reference_views_72 \
  --glb-top-k 3 \
  --glb-max-keypoints 2048 \
  --glb-resize 1024 \
  --glb-match-threshold 0.01 \
  --max-processing-width 1280 \
  --process-fps 15 \
  --ransac-threshold 8 \
  --iterations 10000 \
  --min-inliers 10 \
  --tracking-reseed-frames 150 \
  --tracking-min-points 8
```

GLBの画素XYZはRC/GLB座標から既存MPRが使用するCOLMAP座標`(x, -z, y)`へ変換してからPnPへ渡します。そのため、PASS後の姿勢を既存MPR rendererへそのまま入力できます。画面上では初回・再探索時に`GLB ALIKED+LightGlue`、追跡中は`optical-flow tracking`と表示されます。

比較用として、従来のSIFT global Feature Map経路と、RC登録画像をReferenceにする`reference_guided_pnp.py`は削除していません。GLB方式はライブカメラ、Optical Flow追跡、既存MPR rendererへ接続済みです。

### 3DGS PLYからのライブPnP + MPR

二段階方式（view選択→local matching→XYZ→PnP→QC）は維持しています。
現在のライブ設定は`output_3dgs_reference_views_72_ray_v2`です。既存3DGS RGBは変更せず、従来の近似depthを画素の視線上へ逆投影し、サブピクセル位置でXYZを取得します。穴の近傍XYZ流用・depth不連続を除外します。これは画素とXYZの不整合の修正であり、depthが実物表面と一致する保証ではありません。

`generate_3dgs_reference_views.py`には同じGaussian描画からRGBとdepth momentsを取得する実験経路も実装しました。scale/quaternion/opacity/SHを含むRGBと同じfootprint・alpha・順序で深度を合成します。ただし、透明度合成depthは表面depthではなく、今回の検証では誤差や対応不足が出たため、ライブの既定にはしていません。詳しくは`SPLAT_PIPELINE_AUDIT.md`を参照してください。

データ`260918splat/splat.ply`のXYZはすでにCOLMAP/MPR座標系です。GLB用の`(x, -z, y)`変換は適用せず、manifestの`runtime_transform: identity`で明示します。

VS Codeでは`MacAR: 3DGS ALIKED LightGlue Live (72 Views)`を起動します。今回の補正版72視点は生成済みです。`MacAR: Generate 3DGS Shared Depth (Experimental)`は別の実験用生成処理であり、ライブ前に実行する必要はありません。

実験用に`MacAR: 3DGS DINOv2 Crop Live (Experimental)`も追加しています。従来のSIFT検索は既定のまま残し、こちらだけ対象物ROI→DINOv2（4分割patch特徴＋全体特徴）→Top-10検索に切り替えます。ALIKED/LightGlue、XYZ、PnP、QCは共通です。ライブ側のROIは画面幅・高さに対する`--retrieval-roi X Y WIDTH HEIGHT`で、既定は中央80%です。対象物が外れた場合は値を調整してください。reference側はRGBの対象物シルエットからbboxを作ります。初回は公式DINOv2重みのダウンロードが必要で、以降は`TORCH_HOME`と`.view_retrieval_cache`を再利用します。表示される類似度は姿勢の確信度ではありません。

### −Z上方視点・自動ROIを使う新しい実験経路

この患者モデルの実際の上方向は**−Z**です。旧72視点は`+Z up`として描画され、上方視点のつもりで下面を見ていました。新しい`output_3dgs_reference_views_72_upper_negz_v2`は`--up-axis negative-z`で、仰角0°/25°/50°に各12方位×2距離、計72視点を作成しました。これは**撮影位置とカメラの画面上方向だけ**の変更で、PLY・画素XYZ・COLMAP/CT座標への変換は行いません。ライブの3DGS姿勢ビューもmanifestの`up_axis`を読んで−Zを画面上へ表示します。

VS Codeでは`MacAR: 3DGS DINOv2 Auto ROI Upper Live`を選択します。ライブ画像からDINOv2 patch特徴と3DGS参考画像の前景・背景特徴で対象物bboxを推定し、reference画像はRGBシルエットからbboxを取得します。双方を同じ224×224のアスペクト保持・灰色余白条件へ正規化してglobal embeddingを比較します。ROIに自信がない場合は中央cropへ黙って戻らず検索を失敗扱いにします。`ViewSync Embedding Crops`ウィンドウは実際にembeddingへ渡したlive/ref画像を並べて表示します。SIFT経路は従来通り選択可能です。

独立した24枚の3DGS合成問い合わせで検索を比較するコマンド：

```bash
TORCH_HOME=.model_cache/torch .venv/bin/python3 benchmark_view_retrieval.py \
  --views output_3dgs_reference_views_72_upper_negz_v2 \
  --query-views output_3dgs_queries_24_negz_v1 \
  --output output_splat_audit/retrieval_upper_negz_v2_24_compare.json
```

評価の正解は**合成問い合わせの既知カメラ姿勢**です。Top-5/10近傍率は30°以内で判定します。同じrendererのため実写での成功率を意味せず、PnPのPASSもCT重畳の正確さを保証しません。旧マップの上下反転、仰角分布、ROI方式は別の要因なので、同じ−Z規約の均等配置`output_3dgs_reference_views_72_uniform_negz_v1`でも比較しています。

24枚の独立合成問い合わせでの検索結果（平均Top-1角度 / Top-5近傍率 / 検索時間中央値）：

| 参照配置 | 検索 | 平均角度 | Top-5 | 中央値 |
|---|---|---:|---:|---:|
| 旧+Z 72視点 | DINO自動ROI | 174.8° | 0/24 | 101 ms |
| −Z 均等配置72視点 | DINO自動ROI | 12.3° | 24/24 | 101 ms |
| −Z 上方72視点 v2 | SIFT全画面 | 10.2° | 24/24 | 34 ms |
| −Z 上方72視点 v2 | DINO中央固定crop | 10.5° | 24/24 | 24 ms |
| −Z 上方72視点 v2 | DINO自動ROI | 10.2° | 24/24 | 101 ms |

自動ROIは今回の合成画像では中央固定cropを上回らず、検索単体で約77 ms重くなります。背景がある実写での効果はまだ1枚しか確認していません。`IMG_2532.jpeg`のcrop確認画像は`output_splat_audit/roi_img2532_upper_negz_v2.png`を参照してください。実写ライブを検証するまで新経路を既定にしません。

同じ−Z上方マップ、同じ独立合成12枚、Top-5、ALIKED/LightGlue、PnP/QC、参照特徴キャッシュ事前生成での後段比較は、SIFT全画面・DINO中央固定crop・DINO自動ROIの順にPASSが12/12・12/12・12/12、AMBIGUOUSが0・0・0でした。処理時間中央値は2.81秒・2.97秒・2.99秒で、local matchingがそれぞれ約2.74秒・2.84秒・2.79秒を占めます。検索自体は約48 ms・49 ms・129 msです。合成正解から10°または30モデル座標単位を超える誤PASSは、この12枚では0件でした。結果は`output_splat_audit/pnp_upper_negz_v2_{sift_full,dino_fixed,dino_auto}_warm_12`に保存しています。

実写`IMG_2532.jpeg`の静止画では、DINO自動ROIが392対応/319 PnP inlier、中央固定cropが298対応/168 inlier、SIFT全画面が369対応/298 inlierで、全方式PASSでした。自動ROIとSIFTの姿勢差は0.42°・カメラ中心3.7モデル座標単位ですが、**独立した実写正解姿勢はない**ので、どちらが正しいかやCT重畳精度は判定できません。この静止画試験ではMPRを描画しておらず、ライブ動作も未検証です。

ライブ用VS Code設定はTop-3にしています。同じ合成12枚で12/12 PASS・誤PASS 0、処理時間中央値2.14秒（Top-5は2.99秒）でした。実写1枚も286 inlierでPASS、処理時間2.12秒でした。ただしTop-3は候補を減らすため、ライブで視点が大きく変化した際はTop-5へ戻す余地があります。速度の主因は約1.97秒のlocal matchingで、検索の約0.13秒より大きいです。

SIFTからembeddingへの変更とcropの効果を分ける場合、まず`benchmark_view_retrieval.py`で全72視点を6条件で検索評価します。各問い合わせについて同一方位・仰角の距離違い2枚を両方除外します。次に、同じholdout番号・同じ`--top-k`で以下の後段PnP比較を実行します。各出力の`summary.json`、`results.json`を比較します。実写画像を`--query`に指定した場合、その画像の独立した正解姿勢がない限り角度誤差は計算できません。

```bash
python3 benchmark_view_retrieval.py --views output_3dgs_reference_views_72_ray_v2 --output output_splat_audit/retrieval_72_comparison.json
```

初回の72視点・合成holdout実測では、SIFT全文検索のTop-5近傍率は88.9%、DINOv2 spatial＋cropは45.8%でした。12視点の後段PnP比較（Top-5）ではPASSがそれぞれ11/12、6/12。さらに合成正解姿勢と10度または30座標単位以上ずれたPASSがSIFTで3件、DINOで1件ありました。後者は現時点で**既定方式へ昇格しません**。QCのPASSも正確な重畳の証明ではありません。これは同じ3DGS renderer同士の比較であり、実写での優劣や臨床精度を証明するものではありません。実写`IMG_2532.jpeg`では両方式ともPnPはAMBIGUOUSでした。

```bash
python3 benchmark_splat_pipeline.py --views output_3dgs_reference_views_72_ray_v2 --output output_splat_audit/retrieval_sift_full --holdout 0 1 2 3 4 --top-k 10
python3 benchmark_splat_pipeline.py --views output_3dgs_reference_views_72_ray_v2 --output output_splat_audit/retrieval_sift_crop --holdout 0 1 2 3 4 --top-k 10 --retrieval-crop
python3 benchmark_splat_pipeline.py --views output_3dgs_reference_views_72_ray_v2 --output output_splat_audit/retrieval_dino_full --holdout 0 1 2 3 4 --top-k 10 --retrieval-mode dino-spatial
python3 benchmark_splat_pipeline.py --views output_3dgs_reference_views_72_ray_v2 --output output_splat_audit/retrieval_dino_crop --holdout 0 1 2 3 4 --top-k 10 --retrieval-mode dino-spatial --retrieval-crop
```

新しい環境では、最初に一度だけブラウザGaussian rendererの依存関係を準備します。現在のMacでは実行済みです。

```bash
./setup_3dgs_renderer.sh
```

Terminalからの同等コマンドは次です。

```bash
python3 generate_3dgs_reference_views.py \
  --ply "/Users/daisuketanii/Library/CloudStorage/GoogleDrive-mp.daisuketanii@gmail.com/マイドライブ/010_splat/data/260918splat/splat.ply" \
  --output output_3dgs_reference_views_72_geometry_new \
  --azimuths 12 \
  --elevations -30 0 30 \
  --distances 1.7 2.5 \
  --width 960 \
  --height 720

python3 main.py \
  --camera 1 \
  --matching-mode splat-track \
  --splat-views output_3dgs_reference_views_72_ray_v2 \
  --virtual-policy validated \
  --virtual-diagnostics output_splat_audit/live_stages.jsonl \
  --glb-top-k 10 \
  --glb-max-keypoints 2048 \
  --glb-resize 1024 \
  --glb-match-threshold 0.01 \
  --max-processing-width 1280 \
  --process-fps 15 \
  --fundamental-threshold 3 \
  --ransac-threshold 8 \
  --iterations 10000 \
  --min-inliers 10 \
  --min-inlier-ratio 0.20 \
  --virtual-max-coarse-rotation 35 \
  --tracking-reseed-frames 150 \
  --tracking-min-points 8
```

初回のTop-10 learned matchingはバックグラウンドで実行されます。カメラ表示は継続し、PnPがPASSすると既存のLucas–Kanade trackingとMPR重畳へ切り替わります。

同じVS Code起動設定で、別ウィンドウ`ViewSync Live | 3DGS Camera Pose`も開きます。左にライブ映像、中央に3DGS RGB/XYZから作った色付き3D概形とカメラ位置、右に選択された3DGS仮想視点を並べます。青は約1秒間隔で更新するSIFT Top-1の**粗い候補**で、姿勢確定ではありません。右にはその仮想画像とTop-1/Top-2スコア比を表示します。緑は現フレームでQCを通過したカメラ、オレンジは過去の探索で得た**未確定のPnP候補**です。`UNKNOWN`、`COARSE CANDIDATE`、`UNVERIFIED`の間は現在位置が確定しておらず、MPRは表示されません。3DGS仮想視点とPnPの回転差が既定35度を超える候補もPASSさせません。A/Dで3D概形を左右回転、W/Sで上下回転、Rで視点を戻せます。中央の概形は診断用に既存の3DGS RGB/XYZを間引いた表示で、Gaussianそのものの再レンダリングではありません。`--no-3d`でこの診断ウィンドウを非表示にできます。

reference ALIKED特徴は画像内容・モデル重み・設定のhash付きで`.aliked_cache`へ保存します。画像/設定変更時は再抽出します。起動前に全viewを準備したい場合（`--output`は未使用ディレクトリを指定）:

```bash
python3 benchmark_splat_pipeline.py --views output_3dgs_reference_views_72_ray_v2 \
  --output output_splat_audit/cache_preparation --prepare-only --device auto
```

既存3DGSデータからray補正版を新規作成する場合:

```bash
python3 upgrade_splat_ray_geometry.py --source output_3dgs_reference_views_72_gpu \
  --output output_3dgs_reference_views_72_ray_v2
```

既存出力は上書きしません。作成済みなら省略してください。`--virtual-policy legacy`は比較用に従来の候補統合・判定を使います。通常は`validated`を使用します。候補の姿勢が競合する場合は`AMBIGUOUS`としてMPRを表示しません。再探索結果が5秒を超えて古い場合は追跡のseedに使いません（`--virtual-max-seed-age`）。高速打ち切り`--virtual-fast-consensus`は実験用・既定OFFです。

`--virtual-appearance-audit`はPnP点の周辺を除外した広域appearance比較をログへ追加します。現在は診断専用で、PASS判定には使用しません。

## Step 5: iPhone連係カメラでリアルタイムPnP + MPR

iPhoneをmacOSの連係カメラとして接続すれば、専用iOSアプリや画像転送サーバーは不要です。OpenCVがMacのカメラ入力を直接読み、SIFT matching、RANSAC-PnP、MHAのoblique MPR生成、カメラ画像への重畳までMac上で行います。

### VS Codeから起動

Terminal起動時とVS Code起動時でiPhoneのフォーカス挙動が異なる環境では、リポジトリの`UXbrowse`フォルダをVS Codeで開き、「実行とデバッグ」から`MacAR: Realtime PnP Main`を選んで起動します。この構成は`MacAR/.venv/bin/python3`、作業ディレクトリ`MacAR`、カメラ1、OpenCV/AVFoundation直接読取、ネイティブ解像度、フォーカスゲートなし、global Feature Map matching + temporal trackingを固定します。

Mainでは、最初にSIFTをFeature Map全体へ照合してRANSAC-PnPを実行します。PASSした2D–3D inlierはPOINT3D_IDを維持したままLucas–Kanade Optical Flowで次フレームへ伝播し、forward/backward誤差でドリフトを除去して、前回姿勢を初期値に高速PnPを実行します。10追跡フレームごと、または追跡点が10点未満になった時だけglobal matchingで点を補充・再localizeします。global再取得が1回失敗しても有効な追跡は破棄しません。通常のMPR表示条件はinlier 10点以上かつinlier ratio 20%以上です。

追跡なしで各処理フレームを都度走査する場合は`MacAR: Global Scan PnP (No Tracking)`、reference-guided方式を比較する場合は`MacAR: Reference Guided (Experimental)`を選びます。Terminalでは`python3 main_global_scan.py --camera 1`で起動できます。この経路は毎回SIFT抽出、Feature Map全体とのmatching、RANSAC-PnPを実行し、前フレームの点や姿勢を再利用しません。

切り分けには`MacAR: Simple Camera Focus Test`、RAWとPnP画面の同時比較には`MacAR: Realtime PnP + RAW Compare`を使用します。VS Codeの統合Terminalから起動するため、カメラ権限もVS Code側の条件に揃います。

### 1. カメラ番号を確認

iPhoneを連係カメラとしてMacへ接続した状態で、`MacAR`フォルダから実行します。

```bash
source .venv/bin/activate
python3 realtime_pnp.py --list-cameras
```

macOSからカメラアクセス確認が表示された場合は許可します。見つからない場合は「システム設定 → プライバシーとセキュリティ → カメラ」で、使用しているTerminalまたはPythonにカメラアクセスを許可してください。現在の既定は、フォーカス挙動を確認済みの`cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)`によるOpenCV直接取得です。

### 2. まず近似内部パラメータでライブ確認

```bash
python3 realtime_pnp.py --camera 0
```

reference-guided方式をTerminalから明示する場合は次のとおりです。

```bash
python3 realtime_pnp.py \
  --camera 1 \
  --matching-mode reference \
  --ransac-threshold 8 \
  --iterations 10000
```

既定では、`cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)`で直接取得したカメラのネイティブ解像度を、縮小せずPnPへ渡します。解像度・バッファ・autofocusプロパティは変更しません。Terminalの既定処理上限は5fps、VS Code Mainは追跡を滑らかにするため15fps、水平画角は65度の近似値です。表示ウィンドウで`Q`または`Esc`を押すと終了します。別のカメラ番号なら`--camera 1`などに変更します。

以前に一瞬localizationできた際の取得・マッチング条件を正確に再現する場合は、次のプリセットを使用します。

```bash
python3 realtime_pnp.py --camera 1 --previous-success-settings
```

このプリセットは、FFmpeg導入前のOpenCV/AVFoundation直接読取、ネイティブ解像度、縮小なし、5fps、追跡なし、Lowe ratio 0.75、RANSAC 4px、最小inlier 20点へ固定します。フォーカスゲートとRAW診断画面は無効です。通常の`python3 realtime_pnp.py --camera 1`は同じカメラ取得条件にtemporal trackingを追加した構成です。

バックグラウンドで最新フレームだけを保持する比較経路は`--capture-backend opencv-latest`、FFmpeg経路は`--capture-backend ffmpeg`で明示できます。フォーカスゲートは既定では無効で、必要な場合だけ`--min-focus-score 15`のように指定します。MPRは姿勢推定に使った同一フレームへ重畳します。

RAW比較が必要な場合だけ`--raw-preview`を追加します。これは取得フレームを、リサイズ・focus判定・SIFT・PnP・描画より前にそのまま`imshow()`する診断画面です。

画面上部にはPnPの`matches`、`inliers`、MPR状態に加えて、カメラFPS、実処理FPS、最新フレームへ追いつくために捨てたフレーム数、キュー滞留時間、アプリ内遅延、SIFT・matching・PnP・MPR・3Dデバッグ描画の個別時間、SIFT keypoint数、中央領域の相対的な`center focus`値、入力/処理解像度、判定閾値を表示します。`center focus`は同じ構図で値が高いほど鮮明です。絶対的な合否値ではないため、カメラを静止してAFが落ち着く前後の比較に使います。`app latency`はFFmpegでフレームを復号してから表示処理が終わるまでの値で、iPhoneの撮像・エンコード・無線転送時間は含みません。`latest-frame drops`は異常ではなく、古い映像を順番に処理して遅延が増えないよう意図的に破棄した累積数です。緑点がRANSAC inlier、赤点がoutlier、黄色枠内の半透明グレースケール画像がMPRです。

MPR有効時はカメラ画面に`MPR Slice x0.1%`と`MPR Opacity %`のトラックバーが表示されます。断面位置はカメラ側の手前面0%から奥面100%を基準とし、手前側−100%から奥側150%まで0.1%刻みで移動できます。macOSで負値トラックバーに依存しないよう、UI上の`0〜2500`を内部で`-1000〜1500`へ変換します。UI値の`0 = -100%`、`1000 = 手前面0%`、`1500 = 中央50%`、`2000 = 奥面100%`、`2500 = 奥側150%`です。断面探索範囲にはモデルboxではなくMHA全体のboundsを使用します。データ範囲外まで送った場合は空の断面になります。透過率は0～100%です。

MPR面の外枠は表示しません。断面はカメラの中央光線ではなくCTボリューム全体を基準に配置されるため、対象が画面中央から外れても画面内にあるCT断面は重畳されます。

距離計測時は`Measure Zoom +100%`トラックバーで1倍から12倍まで拡大できます。計測の1点目を左クリックすると、その位置が自動的に拡大中心になります。2点目では中心を動かさず、計測結果を表示します。`Shift+左クリック`またはホイール操作でも拡大中心を変更できます。拡大後の左クリック座標は元のMPR座標へ逆変換されるため、距離の単位と計算縮尺は変わりません。右クリックで計測点をクリアします。
計測点は左ボタンを離した時点で確定します。macOSが発行するダブルクリックの重複イベントは同一の1点として扱います。

断面上の距離は画像上で直接計測します。MPRが表示されている状態で左クリックした点を始点、2回目の左クリックを終点として、画像座標を断面座標へ逆変換した距離を表示します。3回目の左クリックで新しい計測を開始し、右クリックで消去します。単位は`model-coordinate.json`のmillimeter座標に基づくmmです。これはMPR断面内の距離であり、画面上の実物との絶対位置・寸法精度評価にはカメラキャリブレーションが必要です。

3Dマッチ範囲のデバッグ画面は既定で有効ですが、PnP処理を妨げないよう2Hzだけ更新します。不要なら`--no-3d`で停止できます。マッチングより低遅延を優先する場合は次を使います。

```bash
python3 realtime_pnp.py --camera 1 --max-processing-width 1280 --process-fps 10 --no-3d
```

縮小すると小さな模様のSIFTマッチ数が減る場合があります。まず既定の元解像度で`keypoints`、`matches`、`inliers`を確認し、十分な余裕がある場合だけ1280pxへ下げます。

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
