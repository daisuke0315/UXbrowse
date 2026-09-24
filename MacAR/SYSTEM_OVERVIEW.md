# 現在のMacAR：3DGSを参照するライブMPR重畳

2026-09-23時点。これは現在採用している構成の説明です。実験結果の詳細は[SPLAT_PIPELINE_AUDIT.md](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/SPLAT_PIPELINE_AUDIT.md)を参照してください。

## 1. 何をするプログラムか

iPhoneをMacの外部カメラとして使い、撮影した物体が「カメラに対してどの位置・向きにあるか」を推定し、その姿勢に合わせてCTの断面を映像へ重ねるPythonプログラムです。

3DGSは、物体がいろいろな角度からどう見えるかを調べるための参照です。ライブ中に毎フレーム3DGSを再構築しているわけではありません。

**72種類の角度からひとつを選んで終わりではありません。** 近い参照画像を手掛かりに点を対応付け、PnPが連続的な位置3成分・回転3成分（6DoF）を推定します。

## 2. 起動方法

1. iPhoneをMacの連係カメラとして使用できる状態にする。
2. VS CodeでUXbrowseフォルダを開く。
3. 「実行とデバッグ」で **MacAR: 3DGS ALIKED LightGlue Live (72 Views)** を選び、F5で起動する。
4. 終了はOpenCVの映像ウィンドウでQまたはEsc。

設定は[launch.json](/Users/daisuketanii/Documents/GitHub/UXbrowse/.vscode/launch.json)、入口は[main.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/main.py)です。

**単にmain.pyの「Pythonファイルを実行」を押すと、この起動設定は使われません。** 引数なしの既定方式は旧`global-track`なので、上記の名前の構成を選んでください。

現在はcamera index=1です。Macの接続状態によって番号は変わります。新しい参照画像を生成し直す作業は、通常のライブ起動には不要です。

## 3. 処理の流れ

### 準備済みのデータ

```text
3DGS（splat.ply）
  ↓ 事前にさまざまな方向・距離から描画
72枚の参照RGB画像 ＋ 画素に対応する3D情報
  ↓
参照画像の特徴を抽出してキャッシュ
```

ライブで使うのは[output_3dgs_reference_views_72_ray_v2](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/output_3dgs_reference_views_72_ray_v2)です。
RGBは従来の3DGS描画画像を維持し、XYZは画素の視線に合わせた補正版です。ただし奥行き自体には近似が残ります。

### 起動時・見失ったときの再探索

```text
ライブ画像
  ↓ SIFTで見え方の近い参照画像を検索
上位10枚を選ぶ                    ← coarse selection
  ↓ 各画像内でALIKED＋LightGlueを使って点を対応付ける
2D画像同士の対応                  ← local matching
  ↓ 幾何的に矛盾する対応を除く
参照画素のXYZを取得
  ↓
ライブ画像の(u,v) ↔ 物体上の(X,Y,Z)
  ↓ PnP＋RANSAC
位置・向きの候補
  ↓ 点の分布・誤差・候補間の競合を確認
採用可能な姿勢 → 追跡開始 → MPR重畳
```

SIFTは参照viewの検索用、ALIKEDは局所特徴の抽出用、LightGlueはその対応付け用です。全Sparse Point Cloudへ直接SIFT照合する旧方式には戻していません。

複数viewの点を無条件に混ぜることも避けています。推定姿勢が整合する候補だけを統合します。

### 位置が分かった後

毎フレーム重いTop-10探索を繰り返すのではなく、採用された2D点をLucas–Kanade optical flowで追跡します。対応するXYZは保持し、新しい2D位置からPnPを更新します。

- 通常時：2D点の追跡 → PnP → QC → MPR表示。
- 定期的：バックグラウンドで参照画像との再照合。
- 見失ったとき：探索表示へ戻る。

現在は成功した追跡150フレームを目安に再照合します。実時間固定ではありません。検索結果が古すぎる場合（既定5秒超）は新しい追跡の初期状態に使いません。

カメラ取得自体はOpenCV/AVFoundationの直接readです。非同期なのは重い再探索であり、現在のカメラ経路は`OpenCVLatestFrameCapture`ではなく`OpenCVDirectCapture`です。AFはカメラ側に任せています。

## 4. 画面の見方

|表示|意味|
|---|---|
|SEARCH|参照画像を検索・照合中。映像表示は継続する|
|PASS / optical-flow tracking|採用された対応を追跡し、現在フレームの姿勢を更新している|
|FAIL|点数・誤差・分布などの条件を満たさない|
|AMBIGUOUS|十分な支持を持つ別の姿勢が存在し、ひとつに決められない|
|MPR VISIBLE / HIDDEN|断面を重畳している／していない|

新しい探索結果がFAILでも、既存のtrackingが有効なら、その追跡姿勢による表示は継続します。FAILした探索姿勢を採用するという意味ではありません。

MPRには断面送り（−100%〜150%）、透明度、計測時の拡大があります。表示する断面はカメラに正対する斜断面です。患者の固定axial/coronal/sagittal断面ではありません。

2点クリックの計測は「表示中のMPR平面上の距離」です。単眼カメラで任意の実物上の2点の奥行きを測っているわけではありません。

別の3Dウィンドウは照合点のデバッグ用です。3DGS全体をそのまま表示するものではなく、背景点表示には参照viewから抽出した点を使っています。

## 5. 主なファイル

ファイル名にGLBと書いてあるものも、現在は3DGSで共用しています。

|ファイル|担当|
|---|---|
|[main.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/main.py)|VS Codeからの入口|
|[realtime_pnp.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/realtime_pnp.py)|カメラ取得、起動引数、画面、スライダー、処理の接続|
|[glb_virtual_live.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/glb_virtual_live.py)|3DGS/GLBのview検索、local matching、候補統合、非同期再探索|
|[aliked_lightglue_pnp.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/aliked_lightglue_pnp.py)|ALIKED/LightGlue、幾何フィルタ、PnP推定・改善|
|[glb_virtual_view_pnp.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/glb_virtual_view_pnp.py)|対応画素からXYZを取り出して2D–3D対応を作る|
|[splat_geometry.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/splat_geometry.py)|サブピクセルdepth補間、XYZ逆投影、無効領域の除外|
|[pose_quality.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/pose_quality.py)|姿勢の誤差・点の分布・カメラ前方にあるかの確認|
|[realtime_pnp_server.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/realtime_pnp_server.py)|LK追跡、MHA読み込み、座標変換、MPR描画。現在の構成ではHTTP serverとして起動する必要はない|
|[upgrade_splat_ray_geometry.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/upgrade_splat_ray_geometry.py)|既存3DGS RGBを維持してray補正版XYZを事前生成|
|[benchmark_splat_pipeline.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/benchmark_splat_pipeline.py)|保存画像による再現試験、処理時間・候補の記録、特徴cache準備|

通常はmain.pyから起動するだけで、これらを手作業で順番に実行する必要はありません。

## 6. 現在の設定とデータ

|項目|VS Codeの3DGSライブ設定|
|---|---|
|カメラ|index 1、OpenCV/AVFoundation、ネイティブ取得解像度|
|処理画像|幅最大1280px、処理上限15fps（実測保証ではない）|
|候補数|72viewからTop-10|
|局所特徴|ALIKED最大2048、抽出時resize1024|
|幾何判定|Fundamental RANSAC 3px、PnP RANSAC 8px|
|採用条件|10 inliers以上・比率20%以上に加え、誤差・分布・候補競合等を検査|
|カメラ内部パラメータ|概算HFOV65°、歪みゼロ。現在の起動構成に校正ファイル指定なし|
|CTボリューム|[volume_preprocessed.model-bounds.mha](/Users/daisuketanii/Documents/GitHub/UXbrowse/data/260911/volume_preprocessed.model-bounds.mha)|
|座標メタデータ|[model-coordinate.json](/Users/daisuketanii/Documents/GitHub/UXbrowse/data/260911/model-coordinate.json)|
|ライブログ|`MacAR/output_splat_audit/live_stages.jsonl`。起動後に追記される|

PnPはマップ座標→カメラ座標の変換を推定します。3DGSのXYZは現行manifestでCOLMAP系として扱い、追加軸変換はしません。MPR側はMHA物理座標とRC座標が一致している前提で、RC↔COLMAPの軸変換を行います。CTとの登録精度をこの前提だけで保証しているわけではありません。

## 7. 採用していない実験・旧方式

- RGBと同じGaussian描画から合成depthを取得する経路：実装済みだが、対応不足・姿勢誤差の悪化があり、ライブ既定には未採用。
- 広い範囲のappearance一致を使う追加QC：`pose_appearance.py`。診断専用、既定OFF。PASS判定には使わない。
- 候補の早期打ち切り：実験用、既定OFF。Top-10内の競合を見逃さない設定を優先。
- Sparse global、RC reference、GLB方式：比較用コードとして残っている。現在の選択は`splat-track`。

**RGB描画とgeometry生成が完全に統一された版が、現在稼働しているわけではありません。** 現在の採用版は「3DGS RGB＋補正した近似depth」です。

## 8. どこまでできていて、何が残っているか

できていること：参照view選択からMPR重畳までの接続、点の追跡、段階別ログ、XYZのpixel-ray整合性修正、特徴cache、競合姿勢の拒否。直近の単体試験は77件合格。

測定：仮想画像5例で、同一の改善済みコードを使ったXYZ変更前後の平均カメラ位置誤差は2.21→1.66モデル単位。Apple GPUでの実写真の再探索は初回約3.55秒、ウォーム時約1.75秒。これは患者座標のmm精度や映像のend-to-end遅延ではありません。

残っていること：

1. 実写真IMG_2532では、約78°異なる姿勢を38点・29点が支持する。現在はAMBIGUOUSとして重畳を止める。
2. XYZの奥行きは近似であり、実物の正しい表面かは未保証。
3. カメラ校正、CTとの登録精度、実カメラの長時間追跡・遮蔽復帰・実遅延は追加評価が必要。
4. 器具先端の位置や患者内部の変形は推定していない。

つまり現状は、**3DGSを手掛かりにしたマーカーレスMPR重畳の研究用試作であり、誤った姿勢を表示しないための判定を強化している段階**です。
