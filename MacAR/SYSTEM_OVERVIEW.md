# 現在のMacAR：3DGSを参照するライブMPR重畳

2026-10-23改訂：補充は新旧の現在UV–XYZを共通集合にし、旧・新点由来・統合PnP姿勢を同じ空間支持で選ぶ。候補姿勢間の距離によるリセットと登録実写高速補充の検索元姿勢差による拒否は削除する。QC・最終確認とrollback・追跡上限2000点を維持し、補充失敗だけでは既存追跡をリセットしない。登録実写と3DGSの両方式の最終統合が対象。初期照合・参照間競合・3DGSソース照合固有のQCは保持。[現行仕様](docs/regional-replenishment.md)。

2026-10-22配置：コードは`MacAR/`、ローカル画像・モデル・地図・結果・ログは`data/MacAR/`（Git対象外）。データルートは`VIEWSYNC_MACAR_DATA_DIR`で変更できます。[配置・移動記録・GitHub公開の注意](docs/data-layout.md)。以下の過去の`MacAR/output_*`表記は変更前の配置です。

2026-10-22現行：[継続補充・ライブMPS資源管理・カメラ再接続](docs/live-refresh-recovery.md)。同一視点も負荷管理付きで補充し、点支持低下では先行起動する。厳格QCと登録実写／3DGS両方式を維持する。以下の過去日付は履歴を含む。

2026-10-05：登録実写の前処理では、既知RC姿勢によるエピポーラ候補制限をALIKED descriptor照合より先に行う。初期地図と増補に適用し、ライブのLightGlueと現在使用中の地図は変更しない。[前処理の仕様・検証](docs/known-pose-preprocessing.md)。

元写真620枚の新規作成から全点監査まで実測9分18秒。全体の旧方式対照とCT精度は未検証で、対応数は減る。[測定報告](KNOWN_POSE_PREPROCESSING_20261005.md)。

2026-10-04：[実写appearance検索](docs/real-appearance-retrieval.md)を登録実写モードの既定として採用。3DGS方向候補を維持し、その方向内の実写をDINO類似度・パッチ配置で各2枚選びます。写真・ライブ初期化・全体再探索・近傍view復帰が対象。姿勢からの点補充・追跡・XYZ・PnP/QC・表示ゲートは維持します。初回のみ実写索引を生成します。[採用前の比較と制約](REAL_APPEARANCE_RETRIEVAL_20261004.md)。

検証画面には[任意領域の画像遮蔽検証](docs/image-occlusion-validation.md)を追加。遮蔽前後でROI検出・粗検索から独立して再実行し、両方PASSなら遮蔽後の姿勢でCT表示ずれを測定します。既存の面対応点除外とは別モードです。

2026-10-03の参照RGB：現在の既定は元の `output_3dgs_reference_views_72_upper_negz_v2`（128視点）。ブラウザ左側の「3DGS参照画像」でアルファ1.5も選択可能です。登録実写・仮想画像の両方式に使い、停止中に選択して次の開始へ反映します。アルファ1.5は同じカメラ／XYZを持ちますが、粗検索の改善は立証されていません。[条件と比較結果](docs/gs-rendering-comparison.md#現行の選択と比較結果)。以下の日時付き説明・実験は履歴を含みます。

## ブラウザ版の現在の既定（2026-10-03）

2026-10-04の点補充（履歴）：同一視点では停止し、方向・新区画・追跡支持を起動条件にしていた。2026-10-22からはこれらを優先度へ使用し、同一視点の維持補充も行う。[視点イベントと履歴](docs/viewpoint-refresh.md)。

2026-10-04の登録実写候補評価：共通対象cropの8×8領域を各最大1票とし、領域内の再投影誤差で減点した支持点数で順位を決めます。QC通過を優先し、早期終了・競合判定もこの点数に揃えます。PnP内部とQCの閾値は維持します。[現行採点仕様](docs/spatial-pose-scoring.md)。

VS Codeで **MacAR: Browser Live (3DGS + MPR)** → F5 → ブラウザで「登録実写 · ALIKED」のまま開始します。以下の2026-09-24版の説明には、比較用の仮想画像／OpenCV表示の旧構成も含まれます。

```text
初期化・全体復帰：対象crop → 3DGS/DINO視点候補 → 実写ALIKED/LightGlue → 三角測量XYZ → PnP/QC
並行2D追跡：ALIKED抽出直後のUVを連続LK → 照合XYZを付与 → 現在PnP/QC
追跡：採用UVをLKで更新 → 同じXYZでPnP/QC → ブラウザへ映像と姿勢を同時送信
点補充：現在姿勢 → 高速な近傍ALIKED検索／同方向の実写2枚までのLightGlue照合 → 独立PnP/QC → 現在画像へ転送・統合
         → 最新画像までLK転送 → 新旧共通集合で旧/新/統合PnPを評価 → 一意inlierを採用 → 最終確認
表示：ブラウザGPUで3DGSを推定姿勢に合わせて描画。CTは登録実写では固定検証時に表示
```

点補充にDINOや離散参照viewの再選択は不要になりました。予測投影は候補の絞り込みだけに使い、PnPに入力するUVは実際の検出点です。初期化・ロスト復帰・定期安全確認では全体検索を残します。比較用の「3DGS仮想画像」モードは従来の近傍view/LightGlue経路です。

今回の主なコード：[registered_aliked_live.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/registered_aliked_live.py) が実写経路を接続し、[pose_landmark_matching.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/pose_landmark_matching.py) がXYZ付きdescriptor索引・投影候補・局所相互最近傍を担当します。追跡への安全な統合は既存の [glb_virtual_live.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/glb_virtual_live.py)、ブラウザ操作・固定検証は [browser_live.py](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/browser_live.py) です。初回のみ点索引を読み込み、進捗欄に件数を表示します。速度比較と制約はREADME冒頭に記録しています。

面除外の固定検証では、元PASSの実写参照／点検索経路を維持し、姿勢比較先も元PASSにします。独立PnP/QCを通らない結果は表示へ採用しません。

ブラウザ版は開始ごとに`output_runtime_logs`へJSONLログを自動保存します。画面下の記録時刻・セッション経過秒・表示番号と照合すると、後から画面収録に対応する検索／照合／転送／補充拒否・追加点数を調べられます。`runtime_diagnostics.py`が有限キュー＋別スレッドのファイルIOを担当し、画像やdescriptorは保存しません。

2026-09-24時点。現在使用する参照マップは上下を修正した`upper_negz_v2`です。実験結果の詳細は[README.md](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/README.md)を参照してください。

ブラウザ版の更新（2026-09-28）：粗検索はDINO Top-10 → パッチ配置＋方向分散 → ALIKED/LightGlue 3候補。追跡中は負荷を調整しながら近傍から対応点を継続補充します。探索は同時1件、現在画像のQC後に統合し、追跡中の姿勢を無条件に上書きしません。近傍検索の連続失敗＋追跡悪化で全体探索し、900成功処理フレームでも安全確認します。ブラウザの「対応点更新」で従来方式へ戻せます。

補足（2026-09-27）：登録実写由来の三角測量済みALIKED地図はライブ比較・固定CT検証へ接続済みです。また、ブラウザで操作・表示し、Pythonがカメラ取得と局在・MPR生成を担当する[ browser_live.py ](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/browser_live.py)を追加しました。VS Codeの **MacAR: Browser Live (3DGS + MPR)** → F5 → ブラウザの「開始」で起動できます。3DGSはブラウザGPUで直接描画し、従来のスクリーンショット転送を使いません。新UIの操作方法と制約は[README.md](/Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR/README.md)冒頭を参照してください。以下のOpenCV表示経路も残しています。

2026-10-01 前処理画面：ブラウザ左上の「登録実写の前処理・特徴点／XYZを確認」から `/preprocess` を開きます。参照全画像のXYZ付与率、対象マスク内の全ALIKED点／XYZ付き点、3D点群、カメラ方向分布を表示し、全体／不足画像／選択画像の増補と幾何監査を操作できます。`preprocess_registered_map.py` は12→24→48近傍へ段階的に照合相手を増やし、不整合観測を除いて三角測量します。元XYZ／IDは保存し、ジョブとpair cacheは `output_map_preprocessing` に残します。完了した監査済み地図だけを次の照合へ選択できます。処理完了は未登録XYZがゼロになる保証やCTの位置精度保証ではありません。

## 1. 何をするプログラムか

iPhoneをMacの外部カメラとして使い、撮影した物体が「カメラに対してどの位置・向きにあるか」を推定し、その姿勢に合わせてCTの断面を映像へ重ねるPythonプログラムです。

3DGSは、物体がいろいろな角度からどう見えるかを調べるための参照です。ライブ中に毎フレーム3DGSを再構築しているわけではありません。

**72種類の角度からひとつを選んで終わりではありません。** 近い参照画像を手掛かりに点を対応付け、PnPが連続的な位置3成分・回転3成分（6DoF）を推定します。

## 2. 起動方法

1. iPhoneをMacの連係カメラとして使用できる状態にする。
2. VS CodeでUXbrowseフォルダを開く。
3. 「実行とデバッグ」で **MacAR: 3DGS DINOv2 Auto ROI Upper Live** を選び、F5で起動する。
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

既定のライブ参照は[元upper_negz_v2](../data/MacAR/output_3dgs_reference_views_72_upper_negz_v2)です。アルファ1.5も選択可能です。実物の上側はモデルの−Z方向で、仮想カメラの上方視点もその規約で生成済みです。XYZ自体は反転していません。
どちらもXYZはupper_negz_v2の画素の視線に合わせた補正版を維持しています。ただし奥行き自体には近似が残ります。

### 起動時・見失ったときの再探索

```text
ライブ画像
  ↓ 対象物を自動cropしてDINOv2で近い参照画像を検索
上位3枚を選ぶ                     ← coarse selection
  ↓ 各画像内でALIKED＋LightGlueを使って点を対応付ける
2D画像同士の対応                  ← local matching
  ↓ 幾何的に矛盾する対応を除く
参照画素のXYZを取得
  ↓
ライブ画像の(u,v) ↔ 物体上の(X,Y,Z)
  ↓ PnP＋RANSAC
位置・向きの候補
  ↓ 点の分布・誤差・候補間の競合を確認
ALIKED抽出直後から2D点を並行追跡
照合で採用されたXYZ → 生存点の現在UV → 現在PnP/QC → 3D追跡・MPR重畳
```

DINOv2は対象物cropと参照viewの検索用、ALIKEDは局所特徴の抽出用、LightGlueはその対応付け用です。SIFT view検索も比較用として残しますが、全Sparse Point Cloudへ直接SIFT照合する旧方式には戻していません。

複数viewの点を無条件に混ぜることも避けています。推定姿勢が整合する候補だけを統合します。

### 位置が分かった後

毎フレーム重いTop-3探索を繰り返すのではなく、採用された2D点をLucas–Kanade optical flowで追跡します。対応するXYZは保持し、新しい2D位置からPnPを更新します。

- 通常時：2D点の追跡 → PnP → QC → MPR表示。
- ブラウザの背景補充：同一視点も継続補充し、見る方向の4°変化から先行起動（従来の12°条件・新投影区画・点支持の悪化も維持）。新視点は通常待機を短縮し、100ms以内の画像から照合。登録実写はこの時contextの4秒制限も待ちません。前景負荷優先・同時1件・現在画像への転送・再投影・重複除去・QCを維持します。上限は全品質設定で2000点、実際の採用数は検出・照合・QCで決まります。条件・制約は[領域補充](docs/regional-replenishment.md)を参照。
- 全体再探索：近傍照合が3回連続拒否され、追跡にも弱い兆候がある場合。900成功処理フレームでの安全確認も残します。既に処理中なら新しい探索を追加せず、探索完了後も負荷調整の待機を優先します。
- 見失ったとき：探索表示へ戻る。

900は追跡成功処理フレームの目安で、実時間固定ではありません。近傍照合が失敗しても現在の追跡を置き換えません。ライブではALIKED抽出直後から、未照合の点も照合と並行して2D追跡します。照合完了時にXYZを生存点へ対応づけ、現在画像で独立PnP/QCを通します。この経路で確認できた結果は検索時間だけでは拒否しません。連続追跡できない結果には年齢上限（CLI既定5秒、ブラウザ10秒）を適用します。静止写真には追跡を使いません。coverage方式もブラウザから選べます。native CLIでは`--tracking-replenish-mode background --tracking-reseed-frames 900`で視点補充を選択できます。interval方式だけは`--tracking-replenish-frames`の周期（既定30、0で周期補充なし）を使います。

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
|処理画像|ブラウザは幅最大1280px、追跡上限30fps。MPRと送出を分離（実測保証ではない）。OpenCV Mainの従来設定とは別|
|候補数|72viewからDINOv2自動ROIでTop-3|
|局所特徴|ALIKED最大2048、抽出時resize1024|
|幾何判定|Fundamental RANSAC 3px、PnP RANSAC 8px|
|採用条件|10 inliers以上・比率20%以上に加え、誤差・分布・候補競合等を検査|
|カメラ内部パラメータ|未校正時は概算HFOV72°、歪みゼロ。校正ファイルを指定した場合はその値を使用|
|CTボリューム|[volume_preprocessed.model-bounds.mha](/Users/daisuketanii/Documents/GitHub/UXbrowse/data/260911/volume_preprocessed.model-bounds.mha)|
|座標メタデータ|[model-coordinate.json](/Users/daisuketanii/Documents/GitHub/UXbrowse/data/260911/model-coordinate.json)|
|ライブログ|`MacAR/output_splat_audit/live_dino_auto_upper_stages.jsonl`。起動後に追記される|

PnPはマップ座標→カメラ座標の変換を推定します。3DGSのXYZは現行manifestでCOLMAP系として扱い、追加軸変換はしません。MPR側はMHA物理座標とRC座標が一致している前提で、RC↔COLMAPの軸変換を行います。CTとの登録精度をこの前提だけで保証しているわけではありません。

## 7. 採用していない実験・旧方式

- RGBと同じGaussian描画から合成depthを取得する経路：実装済みだが、対応不足・姿勢誤差の悪化があり、ライブ既定には未採用。
- 広い範囲のappearance一致を使う追加QC：`pose_appearance.py`。診断専用、既定OFF。PASS判定には使わない。
- 候補の早期打ち切り：実験用、既定OFF。Top-3内の競合を見逃さない設定を優先。
- Sparse global、RC reference、GLB方式：比較用コードとして残っている。現在の選択は`splat-track`。

**RGB描画とgeometry生成が完全に統一された版が、現在稼働しているわけではありません。** 現在の採用版は「3DGS RGB＋補正した近似depth」です。

## 8. どこまでできていて、何が残っているか

できていること：参照view選択からMPR重畳までの接続、点の追跡と姿勢ガイドによる補充、段階別ログ、XYZのpixel-ray整合性修正、特徴cache、競合姿勢の拒否。直近の単体試験は97件合格。

測定：現在のupper参照マップとTop-3では、独立した3DGS合成12視点で12/12 PASS、処理時間中央値は約2.14秒です。実写`IMG_2532.jpeg`ではPnP 286 inlierでPASSしました。これは同一rendererの合成評価と実写1枚の結果であり、患者座標のmm精度や映像のend-to-end遅延ではありません。

補充の静止画確認：同じ実写フレームでは近傍view照合297 inlierでPASSし、現在の姿勢への再投影で検証して21点を追加、追跡集合は286→307点になりました。これは動きのあるライブでの補充成功率・遅延を証明するものではありません。

残っていること：

1. 旧マップでは、実写真IMG_2532に誤った下方カメラ姿勢でもPASSした例がある。新マップでは支持点が増えたが、QCの物理妥当性チェックは残課題。
2. XYZの奥行きは近似であり、実物の正しい表面かは未保証。
3. カメラ校正、CTとの登録精度、実カメラの長時間追跡・遮蔽復帰・実遅延は追加評価が必要。
4. 器具先端の位置や患者内部の変形は推定していない。

つまり現状は、**3DGSを手掛かりにしたマーカーレスMPR重畳の研究用試作であり、誤った姿勢を表示しないための判定を強化している段階**です。
