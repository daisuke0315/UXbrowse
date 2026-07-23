# ViewSync

CT断面画像（MPR）と可視光3Dモデルを、ブラウザ上で重ねて観察するための研究開発用ビューアです。

## 概要

- 3Dモデルをメイン画面、MPRをサブ画面として表示します。
- カメラの視線に連動した断面、またはAxial・Coronal・Sagittal固定断面を表示できます。
- MPRより手前のモデルを透過し、外観と内部構造の位置関係を確認できます。
- DICOM・MHA・GLB・OBJをブラウザ内で読み込みます。
- 読み込んだデータを外部サーバーへ送信しません。
- 本リポジトリは、あらかじめ整合されたCTと3Dモデルを観察するビューアです。CTとモデルの位置合わせ処理自体は含みません。

## オンラインデモ

- [ViewSyncを開く](https://daisuke0315.github.io/UXbrowse/)
- 初回は「みかん」デモを自動で読み込みます。
- 画面左上の「みかん」「バナナ」ボタンでデモを切り替えられます。
- デモの注視点距離は300 mmです。
- 初回表示は3DモデルとCTデータの読み込みに時間がかかる場合があります。

## 主な機能

- DICOM CTシリーズと単一MHAボリュームの読み込み
- GLB・OBJ可視光モデルの読み込み
- 視線連動MPR
- Axial・Coronal・Sagittal固定MPR
- MPR画像の3D空間への重畳
- OITによるモデル前景の半透明表示
- CT値しきい値によるMPR背景の透過
- Window Level・Window Width調整
- 最近傍補間・三線形補間
- モデルの位置・回転・スケール調整
- 回転・平行移動・ズームと操作状態のロック
- Phase 1検証モード（MPR条件と提案ビューア条件の比較）

## Phase 1検証モード

- `For Specialist` → `検証` → `検証モードを開始` から開始します。
- 研究用ID、みかん／バナナ、条件順A／Bを設定します。
- 観察時間に制限はなく、参加者が操作ボタンを押したときだけ回答へ進みます。
- MPR三断面または提案ビューアでターゲットを観察します。MPR条件では各断面をスライダーまたはホイールで移動できます。
- 観察後はMPR・CT・正解情報を隠し、モデル中央に表示される透過型XYZ軸ハンドルで3次元位置を調整します。
- 2試行の完了後、Excelで開けるCSVまたは監査用JSONを保存できます。
- 研究設定と仮ターゲットは `data/validation/study-config.phase1.json` で変更できます。
- 現在のターゲットは動作確認用です。研究開始前に確定したDICOM物理座標相当の定義へ差し替えてください。

## 基本操作

- ドラッグ：3Dモデルを回転
- Shift＋ドラッグ：画面に沿って平行移動
- ホイール／トラックパッド：ズーム
- 鍵アイコン：現在の重畳状態を固定・解除
- `みかん`／`バナナ`：デモデータを切り替え
- `For Specialist`：詳細設定を表示
- `For Specialist` → `データ読込`：手元のCT・モデルを選択
- 読み込むCTと可視光モデルは、同じ座標系へ高精度に整合されている必要があります。

## ローカル起動

Node.jsやビルド処理は不要です。

```bash
git clone https://github.com/daisuke0315/UXbrowse.git
cd UXbrowse
python3 -m http.server 5173
```

- ブラウザで `http://localhost:5173/` を開きます。
- 最新版のChrome、Edge、Safariを推奨します。
- 大きなCTや3Dモデルでは、端末のメモリ容量やGPU性能により動作が重くなる場合があります。

## 対応データ

- DICOM：単一フレームCT、非圧縮Little Endian、16-bit Pixel Data
- MHA：3次元、バイナリ、`ElementDataFile = LOCAL`
- 3Dモデル：GLB、OBJ
- MHA ElementType：`MET_SHORT`、`MET_USHORT`、`MET_FLOAT`、`MET_CHAR`、`MET_UCHAR`、`MET_INT`、`MET_UINT`、`MET_DOUBLE`

## 未対応

- JPEG・JPEG-LS・JPEG 2000圧縮DICOM
- Enhanced CT
- `.mhd + .raw`
- PACS・DICOMweb接続
- 患者情報管理やデータ保存
- CTと可視光モデルの自動位置合わせ

## ディレクトリ

- `src/`：ビューア、MPR、座標変換、Three.js連携
- `vendor/`：ブラウザ実行に必要な外部ライブラリ
- `data/`：公開デモデータ
- `docs/`：設計・座標系・研究開発状況
- `tools/`：データ処理とデスクトップDICOMビューア
- `tests.html`：ブラウザ単体テスト

## 技術資料

- [プロジェクト概要](docs/project-overview.md)
- [設計判断](docs/architecture-decisions.md)
- [研究・開発状況](docs/research-status.md)
- [座標系](docs/coordinate-system.md)
- [MPRアルゴリズム](docs/mpr-algorithm.md)
- [Three.js連動MPR](docs/dynamic-mpr.md)

## 注意事項

- 研究・技術検証段階のPoCです。
- 診断用途の医療機器ソフトウェアではありません。
- 臨床的な有用性、精度、安全性は今後の検証対象です。
- 公開デモには果物のデータのみを使用しています。
- ライセンスは現在未設定です。利用・改変・再配布を希望する場合は、事前にリポジトリ所有者へ確認してください。

## テスト

- `http://localhost:5173/tests.html` を開くと、座標変換・補間・MHA読込などの単体テストを実行できます。
- 現在のテスト数は34件です。
