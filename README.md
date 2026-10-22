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
- 画面左上の「みかん」「バナナ」「260911」ボタンでデモを切り替えられます。
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

## 基本操作

- ドラッグ：3Dモデルを回転
- Shift＋ドラッグ：画面に沿って平行移動
- ホイール／トラックパッド：ズーム
- 鍵アイコン：現在の重畳状態を固定・解除
- `みかん`／`バナナ`／`260911`：デモデータを切り替え
- `For Specialist`：詳細設定を表示
- `For Specialist` → `データ読込`：手元のCT・モデルを選択
- 読み込むCTと可視光モデルは、同じ座標系へ高精度に整合されている必要があります。

## ローカル起動

MacARのPythonライブ処理は[MacARの説明](MacAR/README.md)を参照してください。コードは`MacAR/`、ローカル画像・特徴地図・モデル・ログはGit対象外の`data/MacAR/`に分離しています。[データ配置](MacAR/docs/data-layout.md)。以下は独立したブラウザビューアの起動方法です。

Node.jsやビルド処理は不要です。

```bash
git clone https://github.com/daisuke0315/UXbrowse.git
cd UXbrowse
python3 -m http.server 5173
```

- ブラウザで `http://localhost:5173/` を開きます。
- 最新版のChrome、Edge、Safariを推奨します。
- 大きなCTや3Dモデルでは、端末のメモリ容量やGPU性能により動作が重くなる場合があります。

## PSP連携モック

既存のデスクトップDICOMビューアをPSP風のモックとして利用し、表示中の検査にViewSyncデータがある場合だけ外部ブラウザで開けます。

```bash
tools/start_dicom_viewer.command /path/to/dicom-folder
```

検査の対応付けとデモ方法は[PSP Mock Viewerの説明](tools/README_psp_mock.md)を参照してください。

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
- 現在のテスト数は32件です。
