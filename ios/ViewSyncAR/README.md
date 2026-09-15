# ViewSync AR for iOS

ViewSyncの要件をiPhone上で検証するための、SwiftUI + ARKit + RealityKit製PoCです。

## 現在動くもの（iOS 26 SDKビルド）

- 現行Xcode 26 SDKでは、初回だけ`ARObjectScanningConfiguration`で実物を周回スキャンし、`260911.arobject`として端末内へ保存
- 2回目以降は保存済み参照をデータセットより先に読み、モデル/CT読込と並行して対象認識を開始
- GLBを起動ごとに解析せず、事前変換済みの`data/260911/260108-3-runtime.usdz`を非同期読込
- `data/260911/volume_preprocessed.model-bounds.mha`（235×223×245、Float32）を読み込み
- USDZとMHAを並列に読み、アプリ起動中はデコード済みEntity/volumeをメモリへ保持
- `data/260911/model-coordinate.json`を読んでGLB(mm)→tracking(m)の固定変換を検証
- カメラの位置・方向へ追従する実ボリュームのoblique MPRを三線形補間で約20 Hz更新
- DICOM/ViewSync座標のZ軸を重力上方向、Y軸を対象奥行きへ変換してGLBとMHAを同じ姿勢で配置
- `TARGET NOT REGISTERED / SCANNING / SEARCHING / LOCKING / LEGACY PREVIEW / TRACKING / LOW_CONFIDENCE / LOST`状態管理
- 対象未認識またはARKit tracking低下時のMPR即時非表示（failure safe）
- 旧`.arobject`ではモデルの位置確認だけを許可し、座標chainを保証できないCT/MPRは表示しない
- FPS、confidence、対象距離のデバッグ表示
- 画面録画と時刻を照合できる登録診断トレース（anchor姿勢、画面中心ずれ、ray-box判定、MPR成功/遮断回数）
- 診断イベントを端末内の`Application Support/ViewSyncAR/registration-<session>.log`へ保存し、デバッグカード右上から共有

> 現段階は研究用PoCであり、診断用途には使用できません。

## 実機で実行

1. Xcodeで `ViewSyncAR.xcodeproj` を開く。
2. `ViewSyncAR` targetのSigning Teamを選ぶ。
3. iOS 17以降のiPhoneを接続してRunする。
4. 起動後にデバッグ表示の`dataset`が`260911 ready`になるまで待つ。
5. 初回は中央の円を実物の正面中央へ合わせ、位置が安定したら「スキャン開始」を押す。
6. 4〜12秒かけて対象の側面、上、下へゆっくり回り込み、「スキャン完了」を押す。
7. 登録後、同じ実物を再び画面へ入れる。現在の旧方式では`LEGACY PREVIEW`になり、透明モデルの位置だけを確認できる。MPRは誤重畳防止のため停止する。

登録内容をやり直す場合はゴミ箱ボタンで端末内の`260911.arobject`を削除します。SimulatorではカメラARセッションを実行できません。

## iOS 27 Object Trackingへの移行

Create ML 6.2で8Kテクスチャを保持した`data/260911/260108-3-training.usdz`を`Upright`モードで学習し、`data/260911/260911.referenceobject`を生成してアプリbundleへ組み込み済みです。再学習用プロジェクトは[ViewSync260911ObjectTracker.mlproj](../../ViewSync260911ObjectTracker.mlproj)にあります。

再学習は`Tools/train_object_tracker.command`から実行できます。`data/260911/CreateMLCheckpoint`へcheckpointを残し、`data/260911/260911.referenceobject`を出力します。

生成物内の`object.usdz`は学習入力USDZとSHA-256が完全一致し、AABBも`model-coordinate.json`の座標契約と一致しています。iOS 27上では旧`.arobject`より優先して読み、静止対象向けの`detectionObjects`経路で使用します。このMacの現在のSDKはiPhoneOS 26.5なので、SDKにまだ存在しない移動対象向け`trackingObjects`だけは実装していません。Xcode 27導入後の切替境界と検証手順は[OBJECT_TRACKING_IOS27.md](OBJECT_TRACKING_IOS27.md)に固定しています。

## 録画用の診断表示

右下の波形ボタンで`REGISTRATION TRACE`を表示してから画面収録します。最上段のコードが、その時点でMPRを止めている主な理由です。

- `W220 MPR_RAY_MISS`: カメラ中央線が登録モデル境界へ当たっていない（実物とモデル座標のずれを疑う）
- `E221`〜`E223`: MHA断面生成またはGPUテクスチャ更新に失敗
- `I300 MPR_VISIBLE`: MPR生成と表示が成功

`center offset`は登録モデル中心と画面中央の差、`ray → bbox`はカメラ中央線とモデル境界との交差結果です。右上の共有ボタンから、画面に収まらないイベントを含むフルログを送れます。

## 現方式の限界

旧`.arobject`は実物スキャン時に決めた参照座標であり、GLB/CT座標と同一だとは保証できません。そのため旧方式で`TRACKING LOCKED`やMPRを出す処理は無効化しました。学習済み`.referenceobject`から得たARKitの6DoF poseを唯一の姿勢源として使います。

Apple Object Trackingで不足する場合だけ、次を追加候補にします。

1. OpenCVでORB/SIFT feature detection + matching
2. `solvePnPRansac`で `T(C←M)` を推定
3. `camera.transform`と合成して `T(W←M)` を算出

現在は実MHAをCPUで三線形補間しています。次の性能段階ではvoxel dataをMetal 3D textureへ転送し、同じMPR面基底をcompute shaderでsamplingします。

## データセットのスモークテスト

`Tools/DatasetSmokeTest.swift`は、MHA・GLBの読込、件数検証、中央MPR生成を実データで確認します。通常はXcodeビルド時には含まれません。USDZは`usdchecker --arkit`で検証済みです。
