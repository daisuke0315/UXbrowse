# ViewSync Builder

CT-GCP拘束型フォトグラメトリ生成用の独立したローカルWebアプリケーション。
既存のViewSyncビューア・MacARの地図・3DGS/実写方式は変更しない。

## 仕様と実装順序

- [要件定義書 v1.0](docs/requirements-v1.md)を要求の基準とする。
- [設計・検証状況](docs/implementation.md)で実装済み範囲と未確認事項を区別する。
- Phase 0（固定GCPを用いるCeres BA）→Phase 1（Sparse MVP）→Phase 2（Dense/メッシュ/テクスチャ）→Phase 3（配布）の順で検証する。

## 起動

このフォルダの専用環境では、`start.command`をダブルクリックするか、次を実行する。

```sh
cd ViewSyncBuilder
.venv/bin/python -m viewsync_builder serve
```

ブラウザが`http://127.0.0.1:5180/`を開く。終了する場合は実行ターミナルでCtrl+Cを押す。別のHTTPサービスが同じポートを使用している場合は`--port 5182`などで変更する。

開発環境から作り直す場合（エンドユーザー用ZIPでは不要）：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python setup_engines.py
.venv/bin/python -m viewsync_builder doctor
```

`setup_engines.py`だけが開発時に公式配布物を取得する。アプリ実行時にはネットワークからモデルやパッケージを取得しない。

## 操作

1. 絶対パスで新しいプロジェクトフォルダを指定するか、既存プロジェクトを開く。
2. JPEG/PNGを取り込む。原本は保持し、EXIF回転適用済みPNGを処理に使う。必要なら写真を除外する。
3. GCP入力前に試す場合は「モデル生成」→「初期SfMまで」を実行する。
4. DICOMフォルダから番号・説明・スライス数を確認してCTシリーズを選ぶ。内部の整数間隔欠損はアプリ内でHU線形補間し、元枚数・補間枚数・位置を表示する。Axial/Coronal/Sagittal MPRとWL/WWを確認する。
5. GCP設定で点を追加する。CTクリックでLPS/mm座標を設定し、同じ点を2画像以上で指定する。拘束点は4点以上を推奨し、独立評価点は別の役割にする。
6. 「座標と用途」でCT拘束か写真だけの形状試作を選び、「生成物」でSparseのみかテクスチャまでを選択する。CT拘束ではGCP BA後のカメラからDenseを新規生成する。写真試作は任意座標・任意単位で、CT精度は未検証。
7. 選択した生成物の出力まで実行し、結果画面でSparse/Dense/メッシュ/テクスチャを切り替える。回転・拡大して確認し、成果物ZIPを保存する。中断後は新しい実行で再開する。特徴抽出・照合の完了キャッシュは条件が同じ場合に再利用する。

写真は取り込み順に処理用連番を付ける。近傍照合を使う場合は撮影順で取り込む。ROIは写真の「ROIを描く」で3頂点以上を指定して確定する。マスクは別ファイルで保存する。

## 出力とCLI

- `output/<run-id>/sparse/`：COLMAP binary/text、PLY
- `feature_map/`：画像別NPZ（UV、float32 ALIKED descriptor、特徴ID、最終XYZ、validity、再投影誤差）、トラックJSON、観測CSV
- `cameras/`：`T_camera_from_CT_LPS`と内部パラメータ
- `qc/`：BA前後の誤差、カメラ変化、独立評価TREと統計
- `alignment.json`、`coordinates.json`、`run.json`：Sim(3)、座標契約、入力・CT・設定・版・モデルハッシュ
- `appearance/`：Dense PLY、メッシュPLY、テクスチャ付きOBJ/MTL/画像、テクスチャを埋め込んだ`model.glb`、外観生成条件と元カメラのハッシュ

CTモードのPLY/OBJはLPS/mm。GLBは`0.001×(-X,Z,Y)`でY-up/mへ変換し、正逆行列を記録する。アプリ内では逆変換してCT/Sparseに重ねる。写真試作は任意座標を保持し、CT名付きのカメラ/特徴地図は出力しない。

未三角測量の特徴は`point3d_ids=-1`、`validity=False`、`xyz=NaN`で表す。NPZは`allow_pickle=False`で読める。

```sh
.venv/bin/python -m viewsync_builder run /absolute/project/folder --stop-after sfm
.venv/bin/python -m viewsync_builder run /absolute/project/folder --settings '{"pairing":"exhaustive"}'
.venv/bin/python -m viewsync_builder run /absolute/project/folder --settings '{"coordinate_mode":"sfm_preview","surface":"texture","pairing":"exhaustive"}'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python build_package.py
```

## 現在の制約

中核のSparse生成と固定GCP Ceres BAを実装し、合成データの数値テスト・既存PNGのSfM・仮想GCPによる出力接続・ブラウザ操作を確認した。追加の4K JPEG 9枚の試験では共有カメラ・4096点・2400px・全ペアの条件で9枚/435点を再構成し、実CT 4シリーズの復号とMPRを確認した。CTとの対応・物理精度は未確認。詳細と実測値は[検証記録](docs/validation-20261023.md)を参照。

- 実測CT-GCP・独立評価点による物理精度検証は未実施。独立評価点なしは「未検証」と表示する。
- CTの内部欠損を読み込み時に補間し、原DICOMと補間由来を保存する。基本間隔を特定できない場合・非整数間隔・重複位置・座標不整合はエラー。先頭・末尾の未観測欠損は推定しない。補間影響のあるGCP/評価点を使う場合はCT精度「未検証」とする。詳細は設計・検証記録を参照。
- 標準COLMAP DBが検出スコア・対応ごとのLightGlue信頼度を保持しないため、これらは未出力。値を代入して補わない。
- OpenMVS 2.4.0のCPU経路でDense/Mesh/Texture・OBJ/GLBを生成する。点群0点・面0件・テクスチャ欠落は失敗にする。撮影範囲・重なり・姿勢が不足する入力では生成に失敗する。平滑化・穴埋めは初期状態で無効。
- 同梱OpenMVSはAGPL-3.0で、研究用途に限定された第三者コードの記載もある。対応ソース・ライセンスを同梱するが、商用配布条件は未確認。[公式ライセンス](https://github.com/cdcseacave/openMVS/blob/v2.4.0/COPYRIGHT.md)。
- 特徴地図は要件のデータ項目を持つ独立形式。既存MacARのライブ経路は3DGS粗検索と既存地図形式を必要とするため、直接ロード用アダプターは未実装。
- 元写真に歪みがある場合は対応するカメラモデルを選ぶ。圧縮DICOMのデコードは同梱pydicom/Pillowで対応できる転送構文に限り、Enhanced CTは非対応。
- ZIPはmacOS arm64向け開発用プレビュー。別PCの受入試験、署名・公証、Windows実機検証、完全な再配布ライセンス監査は未完了。[配布の境界](docs/distribution.md)。

全要件の完成・精度達成は主張しない。
