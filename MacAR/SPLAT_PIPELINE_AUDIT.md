# 3DGS二段階matching 改善・測定記録（2026-09-23）

この文書は旧参照マップでの実験履歴です。記載された旧マップは2026-09-24にゴミ箱へ移動しました。現在のライブ・再現試験は`output_3dgs_reference_views_72_upper_negz_v2`を使用します。旧コマンドをそのまま実行しないでください。

## 維持した設計

live image → coarse view selection → ALIKED/LightGlue local matching → XYZ → PnP → quality check。
3DGSのappearance reference、OpenCV/AVFoundation capture、AF設定、MPR renderer、CTへの座標変換は変更していません。Sparse global matchingへの置換はしていません。

## 採用した変更

- 既存の72枚の3DGS RGBをバイト同一で維持。近似CPU depthを各画素の視線へ逆投影するgeometryを別データセット`output_3dgs_reference_views_72_ray_v2`に作成。
- OpenGLの画素端座標とOpenCVの整数画素中心の差を補正（principal pointを0.5px補正）。照合されたサブピクセル位置のdepthを補間し、その位置の視線へ逆投影する。
- XYZの穴を隣接する別表面のXYZで埋めない。4画素の有効支持を要求し、5 source unitsを超えるdepth段差を除外。
- ALIKED reference featuresを画像内容・extractor重み・設定hashで永続キャッシュ。pickleは使用しない。read-onlyディレクトリならRAM cacheへフォールバック。
- PnP LMが元のRANSAC支持点の誤差を悪化させた場合は元の姿勢を保持。最終姿勢で再投影誤差・正depthからinlierを再集計。
- QC: count/ratio/mean errorに加え、画像上の支持面積、3D点の非共線性、正depthを確認。trackingにも同じQCを適用。
- 全Top-KのXYZ一括統合をやめ、独立PnPが整合する候補だけを統合。複数の異なる姿勢を別仮説として保持。
- 選択姿勢の70%以上のinlier支持がある他候補が15°または30 source unitsを超えて異なる場合はAMBIGUOUSでFAIL。
- 非同期の古いseedを拒否（取得後の処理経過時間、既定5秒）。既存の正常trackingは古い結果で置き換えない。
- retrieval/local geometric filter/XYZ/PnP/QCの件数、候補pose、理由、時間をJSONLへ記録。従来のHUDはPnP時間をmatching時間へ含めていたので修正。
- Top-K早期終了は実装したが既定OFF。速度のために未評価候補の競合を見逃さない設定を採用。

品質閾値は研究用の初期値であり、臨床上の誤差保証ではない。source unitsは今回の入力がmmを想定しているが、CT登録精度は別途検証が必要。

## 5段階の評価結果

### 1. View retrieval

最終版では既存のSIFT ratio-matchランキングを維持。unique-voteランキングも試したが実写真の正解ラベルがなく、改善を示せないため既定にはしなかった。
既知姿勢のvirtual query 5枚（13/28/2/37/60）をreference候補から除外して試験。同方向・異距離のviewはTop-10に5/5入り、順位は1/1/1/1/5。
これはsynthetic queryでの限定試験。新しい実画像の正解view recall、完全に未知の角度、領域の部分遮蔽に対する成功率は未測定。

### 2. Local matching

IMG_2532を幅1280へ縮小、ALIKED 2048、resize1024、Top-10、F-RANSAC3px。
候補view_0067ではLightGlue 200 → geometric inliers 91。
候補view_0005では95 → 54。この両方に十分な2D対応があり、異なるPnP姿勢を支持していた。
Fのinlierは真の対応の証明ではない。現時点では実写真の対応点ground truthがなく、local matching単独のprecisionは確定していない。

### 3. XYZ assignment

従来はGaussian中心XYZを半径最大3pxへ広げて平均していたため、そのXYZが画素の視線上にない。
72視点の4px間隔サンプルで「view別再投影誤差p95の中央値」は2.981px。
ray補正版は同じ指標が0.000053px（float丸めレベル）。
これは構成上満たすべきpixel-ray整合性の修正であり、物理的なdepthの精度を証明する値ではない。
RGBは同じで、depth近似は残る。したがって「RGBとgeometry完全一致を解決済み」とは扱わない。

同じコード・設定で既知姿勢5例を比較（CPU、source units、真の患者TREではない）:

|query|従来XYZでのカメラ中心誤差|ray補正版|
|---|---:|---:|
|view_0013|3.629|2.899|
|view_0028|0.947|0.769|
|view_0002|1.209|1.112|
|view_0037|3.474|1.906|
|view_0060|1.814|1.612|
|平均|2.215|1.660|

全件PASS。回転誤差は全例で改善したわけではない（view_0037では0.078°→0.221°）。5例のみなので一般化性能は未確定。
比較は同一RGB、同一の改善済みlocalizerを使用。ray版にはpixel-center規約の補正も含む。元コード全体からの純粋な単一要因比較ではない。

### 共有Gaussian描画depthを採用しなかった理由

実験経路`generate_3dgs_reference_views.py`はRGBと同じGaussian footprint、alpha cutoff、sortを使い、float render targetへΣ(w*z)、Σ(w*z²)、alphaを出力する。XYZは合成depthから逆投影。GPU shaderで色payloadだけを変更するため、旧CPUの円形footprint近似とは異なる。
PlayCanvas 2.22.2のper-instance描画経路を使用。既存unified経路とRGBの差もあるため、この実験と旧版の比較はdepthだけのablationではない。

- alpha>=0.9、depth std<=3: 平均有効画素率は約0.66%。実写真のXYZ対応が4点となりFAIL。
- std<=10に緩めると実写真はPASSしたが、virtual view_0028の中心誤差は約6.56 units。PASSや見た目の改善だけでは採用できない。
- Gaussianのopacity合成depthは実物の最前面surfaceではない。深度分布が広い領域を安易に採用すると、正しい画素ray上でも奥行きが誤る。

実験データ`output_3dgs_reference_views_72_geometry_v2`は当時保存していたが、現在はゴミ箱へ移動済み。幾何proxyの物理精度、前景/表面depth推定、独立した幾何との整合は残課題。

### 4. PnP

従来の候補ログには最適化後に40px超へ悪化する仮説があった。LM悪化のrollbackとinlier再集計を追加。
既知3D点・既知pose＋外れ値、LM強制発散、背面点、集中的支持、共線支持の単体試験を追加。
PnP自体の実行時間は小さく、主な遅延要因ではなかった。

### 5. Quality control

IMG_2532にはray補正後も競合が残る。最終GPU試験でview_0067は38 inliers、view_0005は29。姿勢差77.55°、カメラ中心差754.72 units。
最終設定ではこれをAMBIGUOUSとしてFAILにし、MPRを表示しない。どちらが正しいかをground truthなしに断定していない。
中間実験の70%より緩い80%支持閾値ではこの画像がPASSしたが、最終コードでは使っていない。単なる再投影誤差低下を成功として扱わない。
実画像におけるfalse-PASS率の統計的評価は未実施。単体試験とこの競合例の拒否だけで保証しない。

## 遅延

- 変更前CPU・reference未キャッシュ・実写真: 11.60秒。
- 改善版CPU・referenceキャッシュあり: 約4.10秒。
- 改善版Apple MPS・実写真: 初回約3.55秒、同一プロセスの再実行約1.75秒。

未キャッシュ/キャッシュ、CPU/GPUを混同しない。reference抽出を前処理へ移した効果であり、全計算が無料になったわけではない。
既存コードにもRAM cacheはあった。今回の改善は再起動後にも使える永続cache。warm CPUの旧policy約4.30秒→ray版約4.10秒で、毎回のmatchingが桁違いに速くなったわけではない。
MPSのwarm時はquery抽出約80ms、retrieval約110ms、PnP約30ms程度。残りの大半は候補10枚とのlocal matching。これは再局在時間であり、カメラ映像のfps・end-to-end遅延ではない。
カメラcapture/AFは変更せず、通常trackingは既存LK経路を維持した。実カメラのmotion-to-photon latency、遮蔽復帰、長時間trackingは今回未測定。

## 起動・ログ・再現

以下は当時の測定を記録したものです。現在のVS Codeライブ設定は`MacAR: 3DGS DINOv2 Auto ROI Upper Live`で、`upper_negz_v2`を使用します。
当時のライブログは`output_splat_audit/live_stages.jsonl`です。

```bash
cd /Users/daisuketanii/Documents/GitHub/UXbrowse/MacAR
source .venv/bin/activate
python3 benchmark_splat_pipeline.py \
  --views output_3dgs_reference_views_72_upper_negz_v2 \
  --query /Users/daisuketanii/Downloads/IMG_2532.jpeg \
  --holdout 13 28 2 37 60 --device auto --repeat 2 \
  --output output_splat_audit/new_run
```

出力先は未使用ディレクトリを指定。候補ごとの件数/pose/タイミング、合成queryのpose誤差、使用コードhashをresults.jsonへ保存。
`--legacy`は比較用。全候補一括統合など旧policyへ戻すので通常運用には使用しない。
必要なら`--prepare-only`で全reference特徴を先にcacheできる。

主な測定ファイル:
- `output_splat_audit/baseline/results.json`: 変更前コードでの測定
- `output_splat_audit/legacy_warm/results.json`: legacy policy、cold/warm比較
- `output_splat_audit/release_existing/results.json`: 改善済みコード＋従来XYZ
- `output_splat_audit/release_ray/results.json`: 同コード＋ray XYZ、5例
- `output_splat_audit/final_mps/results.json`: 最終ambiguity gate込みGPU再検証
- `output_splat_audit/geometry_audit.json`: 全72viewのpixel-ray診断
- `output_splat_audit/shared_geometry*/results.json`: 不採用の共有描画depth実験

77単体試験合格。Python構文チェック合格。git status/diffはこのMacのXcode license未同意で実行不可だったため、git clean等は確認・実行していない。元RGB/XYZ/PLYは上書きせず、新規出力へ保存した（元referenceディレクトリには特徴cacheを追加）。

## 追加確認：競合poseから再描画

`render_splat_hypotheses.py`で2候補の推定カメラから3DGS RGBを再描画した。
`output_splat_audit/pose_renders/comparison.jpg`は左から実写真、view_0005由来pose、view_0067由来pose。
0005由来poseでは物体全体の向き・投影サイズが明らかに異なるが、局所対応だけでは29 inliersになっていた。0067由来は実写真に近く見えるが、この視覚所見だけで正しい物理poseとは認定していない。

`pose_appearance.py`に、reference RGBを候補poseのXYZでqueryへ投影し、照明の低周波成分を除去したpatch相関を測る診断を追加した。PnPに使用した特徴の周囲8pxを除外して独立した広域支持を調べる。
実写真では除外後の相関中央値が両候補とも約0.001と弱く、信頼できる識別器とは確認できなかった。geometry近似、可視性、見え方の差の影響も残るため、**診断専用・既定OFF**であり、この値で無理にPASSへ変更しない。

再現: benchmarkに`--appearance-audit`、ライブに`--virtual-appearance-audit`を付ける。結果は候補ごとの`appearance_audit`、所要時間は`timings_ms.appearance_audit`。記録例は`output_splat_audit/appearance_check/results.json`。

## 残るボトルネック

1. 実写真の反復模様によるview仮説の競合。view選択→local matchingは維持したまま、候補間の大域的構造・複数フレーム整合の検証が必要。
2. ray補正版のdepthは依然近似。3DGS appearanceと独立した幾何proxyを使うなら、その登録誤差・可視性を独立検証する必要がある。
3. 10候補のLightGlueが再局在遅延の主因。候補打ち切りを既定にするには、false-PASSとrecallを評価できる別のテスト集合が必要。
4. CT登録・カメラ内部パラメータ・患者変形は今回の修正範囲外。PASSは臨床ナビゲーションの精度保証ではない。

## 2026-09-24 ライブ録画での再確認

`画面収録 2026-09-24 8.57.05.mov`ではカメラは約30 fpsで更新されていた。表示が`SEARCH`のままなのは映像取得停止ではない。録画時の診断ではview_0067がretrievalの上位に入り、例えばSIFT 171件、LightGlue 82件、幾何inlier 27件まで得られるが、XYZ 16件、PnP inlier 7件でFAIL。探索は約5〜8秒。録画フレームを画面収録から切り出した再現（元のカメラフレームではない）でも、Top-10・2048 keypoints・depth jump 5 source unitsでview_0067は37幾何inlier→18 XYZ→9 PnP inlierとなった。

depth jumpを10へ緩めると切り出しフレームの一部で11 inlierのPASSが出るが、近接フレーム間の推定カメラ中心が大きく変動した。このため閾値緩和は採用していない。4096 keypointsでは16〜19 inlierへ増えたものの、反対側view_0061にも13 inlierの競合姿勢があり、view_0067との差は約179°。現行のambiguity gateがFAILにするのは意図どおり。したがって特徴点数増加もライブ既定には採用していない。

探索処理時間が5秒を超えたとき、従来は初回取得の正しいseedまで時間だけで破棄していた。現在は既存trackingを古いreseedで上書きしない一方、初回seedは現在フレームに対するLK往復検査＋PnP QCへ渡す。これで将来PASSした場合の初期化機会は失われない。ただし今回の録画で不足している対応点・反復模様の曖昧さ自体は未解決であり、MPRを強制表示する修正ではない。`SEARCH`画面には直前のQC理由を表示する。
