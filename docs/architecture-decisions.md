# 設計判断記録

この文書は、現在の実装がなぜこの構成と仕様になったのかを記録します。実装方法の説明だけでなく、将来の変更時に維持すべき意図を残すことが目的です。

## ADR-001: MPR処理をブラウザ内で完結させる

### 状態

採用済み

### 背景

過去のPoCでは、Unityで外観モデルを操作し、カメラ情報をWebSocketでPythonへ送り、Pythonで生成したMPR画像をUnityへ戻していました。この構成は技術検証には有効でしたが、複数の実行環境、通信、画像転送、座標変換、非同期更新を同時に管理する必要がありました。

### 判断

現行PoCでは、Three.jsによる3D表示、DICOM／MHA読込、MPR再構成、Canvas描画をJavaScriptへ集約し、ブラウザ内で完結させます。

### 結果

- WebSocketやHTTP画像転送が不要になる。
- カメラ、MPR、重畳プレーンが同じ更新周期で動作する。
- Python／Unity間の座標変換と非同期状態が減る。
- 一方、CPUによるMPR生成は大規模ボリュームで性能上の制約になり得る。

性能が不足する場合は、Web Worker、WASM、WebGL2又はWebGPUへの移行を検討します。

## ADR-002: DICOM患者座標を基準座標とする

### 状態

採用済み

### 背景

CTと外観モデルを見た目だけで重ねると、表示上の一致に物理的な意味を持たせられません。また、voxel配列、Three.js、Unity、外観モデル生成ソフトでは、原点、軸方向、単位、右手系／左手系が異なる場合があります。

### 判断

DICOMのImage Position Patient、Image Orientation Patient、Pixel Spacingから定義される患者座標（mm）を基準とします。Three.jsの1 world unitも1 mmとして扱い、初期変換は恒等変換とします。

`origin`は先頭voxelの中心です。Volume中心と表示用Boxは、`dimensions * spacing`ではなくvoxel中心範囲である`(dimensions - 1) * spacing`を基準に計算します。

### 結果

座標整合済みモデルは、Scale、位置、回転を変更せずに重畳できます。手動の軸反転は確認・調整機能であり、モデル生成時の正しい座標整合を置き換えるものではありません。

## ADR-003: MPR中心をカメラ前方固定距離に置く

### 状態

採用済み・現行の正規仕様

### 背景

MPR中心の候補には、OrbitControlsの回転中心、外観モデルの中心、カメラ前方の点があります。モデル中心を使用すると、ズームしても断面の奥行きが変わらず、過去Unity版の「カメラ前方に固定したプレーンを移動させる」操作感と一致しません。

### 判断

MPR中心を次の式で定義します。

```text
target = camera.position + camera.forward * MPR距離
```

`OrbitControls.target`は回転中心、`modelCenter`は位置確認用の参照値として分離します。「モデル連動MPR」がONでも、モデル中心をMPR中心にはしません。

現在の既定距離は`1000 mm`です。MPR textureの既定方向は、外観モデルとの表示方向を合わせるため`V反転 ON`とします。

### 結果

ズームでカメラ位置が移動すると、MPR断面も視線方向へ移動します。この挙動を変更する場合は、単なるバグ修正ではなく操作仕様の変更として扱う必要があります。

## ADR-004: MPR画像を3D空間の実平面として配置する

### 状態

採用済み

### 背景

画面上へ2D画像を重ねるだけでは、外観モデルと断面の位置、姿勢、物理サイズを一致させられません。

### 判断

MPR計算で得た`center / u / v / n`をVolume座標からThree.js座標へ変換し、`画像幅 × spacingU`、`画像高さ × spacingV`の物理寸法を持つプレーンとして配置します。同じMPR結果をCanvas表示とCanvasTextureに使用します。

### 結果

2D表示と3Dプレーンが同じ断面結果を共有します。UV反転は画像方向の確認用に明示的に管理します。

## ADR-005: 断面手前のモデルをshaderで透過する

### 状態

採用済み

### 背景

外観モデル内部にMPRプレーンを配置すると、手前の表面によって断面が隠れます。モデル全体の表示切替では、断面周辺の空間関係も失われます。

### 判断

MPR平面の中心と法線をshaderへ渡し、各fragmentと平面の符号付き距離に基づいて手前側のalphaを制御します。前景alpha、対象距離、フェード幅、完全切断を調整可能にします。

通常のalpha blendでは単一mesh内の三角形を奥行き順に並べられないため、前景透過にはThree.jsのAlpha Hashを使用します。可視fragmentは通常どおりdepthを書き、確率的にdiscardされたfragmentからMPRを見せます。完全切断もalphaを0にするだけでなくdiscardとして処理します。

### 結果

外観の文脈を残しつつ内部断面を表示できます。透明objectの通常ソートに依存しないため、巨大な単一GLB内で内部面が外表面より前に合成される現象を抑制できます。Alpha Hash特有の粒状パターンは残るため、必要に応じてTAA等を検討します。

## ADR-006: GSを必須実装形式に固定しない

### 状態

現時点の判断

### 背景

Gaussian Splattingは写実的な外観表現とリアルタイム表示の候補ですが、本プロジェクトの中核課題は、特定の3D表現方式ではなく、外観モデルとCT断面を同じ物理座標で連動させることです。

### 判断

現行PoCではOBJ／GLBを入力形式とし、MPRと座標統合の成立を先に検証します。GSを導入する場合も、同じ患者座標、カメラ、断面平面の契約へ接続できる構成を維持します。

### 結果

GSの採否や実装ライブラリを変更しても、MPRロジックと研究上の中心命題を分離できます。

## ADR-007: 外観モデルを既定でUnlit表示する

### 状態

採用済み

### 背景

外観モデルのtextureには撮影時の明暗や陰影が含まれます。さらに、RealityCapture等から出力されたモデルにvertex normalやPBR用textureが含まれない場合、通常のLit表示では再構成面の細かなノイズや面境界が照明によって強調されます。

### 判断

OBJ／GLBの外観モデルは`MeshBasicMaterial`相当のUnlit表示を既定とします。GLBのbase color、texture、vertex color、alpha設定は可能な範囲で引き継ぎます。通常表示はOpacity 1、前景透過OFF、両面描画、depth test／depth write有効とします。フォトグラメトリmeshは薄い開放面や局所的に向きが反転した三角形を含み得るため、`FrontSide`のback-face cullingで手前面を消さず、両面をdepthへ参加させます。MPR前景透過は必要時にUnlit materialのshaderへ適用します。

### 結果

撮影済みtextureを照明条件に左右されず表示でき、Unity版のUnlit表示に近い外観になります。一方、動的な陰影による形状把握は弱くなるため、将来Lit表示を追加する場合は確認用の任意モードとして扱います。

## ADR-008: Reverse-Zと動的clip範囲で深度精度を確保する

### 状態

採用済み

### 背景

PerspectiveCameraの深度精度はnear／farの比に依存し、farを固定するとズームアウト時にモデルが消え、単純にfarだけを拡大すると遠距離側の深度分解能が下がります。Three.js r166にはReverse-Z対応がありませんでした。

### 判断

Three.jsをr185へ更新し、`WebGLRenderer({ reversedDepthBuffer: true })`を使用します。`EXT_clip_control`対応GPUではReverse-Zを有効にし、未対応環境ではThree.js標準の通常depthへフォールバックします。

同時に、モデル／Volumeのbounding sphereとカメラ距離からnear／farを毎フレーム調整します。farは最遠点より外側、nearは最寄り表示対象とMPRプレーンを切らない範囲で可能な限り前進させます。

### 結果

ズームアウト時のfar clippingを防ぎつつ、通常depthへフォールバックした環境でもnear／far比を抑えます。実データ`260108-3.glb`で、Reverse-Z有効、通常モデルdepth write有効、前景透過時Alpha Hash有効を確認済みです。
