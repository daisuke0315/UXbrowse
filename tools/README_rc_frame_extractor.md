# RealityCapture向け MOVフレーム抽出

標準モードは、以前使用していた `ffmpeg -vf fps=10 -q:v 1` と同じ固定fpsの直接抽出です。GUIの初期値も10fps・最高品質です。必要な場合だけ、低解像度プレビューを使ったブレ・重複の自動選別へ切り替えられます。

## GUI版を開く

Finderで `tools/start_rc_gui.command` をダブルクリックします。準備ができるとブラウザが開きます。GUIの使用中は起動したターミナルを開いたままにし、終了するときはターミナルで `control + C` を押します。

初回にmacOSのセキュリティ警告が出た場合は、Finderでファイルをcontrolクリックして「開く」を選択します。

## 準備（macOS）

```bash
brew install ffmpeg
python3 -m venv .venv
.venv/bin/python -m pip install pillow numpy
```

## 実行

```bash
.venv/bin/python tools/rc_frame_extractor.py /path/to/input.mov -o /path/to/rc_frames
```

初期値では毎秒3枚を検査します。カメラ移動が速い動画では候補を増やします。

```bash
.venv/bin/python tools/rc_frame_extractor.py input.mov -o rc_frames --sample-fps 5
```

画像が多すぎる場合は `--min-difference` を上げ、抜けが生じる場合は下げます。

```bash
# 強めに間引く
.venv/bin/python tools/rc_frame_extractor.py input.mov -o rc_frames --min-difference 0.03

# 似た画像も多めに残す
.venv/bin/python tools/rc_frame_extractor.py input.mov -o rc_frames --min-difference 0.01
```

ブレ判定は撮影条件に依存します。`selection_report.csv` の `sharpness` を見て、鮮明な画像まで除外されるなら `--min-sharpness` を下げてください。RealityCaptureへ投入する前に、出力画像をサムネイル表示して目視確認することを推奨します。

## 主なオプション

- `--sample-fps 3`: 1秒あたりの候補数
- `--min-sharpness 18`: ブレ判定の下限
- `--min-difference 0.018`: 前回採用画像との差の下限
- `--max-gap 1.5`: 画像間隔が空きすぎないための救済秒数
- `--quality 95`: JPEG品質

このツールの差分判定は被写体の重複率を直接測るものではありません。最終的にはRealityCaptureのAlignment後に、カメラ位置が連続しているか、複数コンポーネントに分かれていないかを確認してください。
