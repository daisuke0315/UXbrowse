# PSP Mock Viewer

既存のDICOMビューアとブラウザ版ViewSyncをつなぎ、PSPからの外部アプリ連携を検証するモックです。

## 起動

Finderで `start_dicom_viewer.command` をダブルクリックします。初回だけ必要なPythonライブラリをセットアップし、ViewSync用ローカルサーバーとPSP Mock Viewerを起動します。

PSPがDICOM保存フォルダを引数として渡す動作は、ターミナルから次のように再現できます。

```bash
tools/start_dicom_viewer.command /path/to/dicom-folder
```

CTを選択すると、表示中シリーズの `StudyInstanceUID` を使って `viewsync_studies.json` を検索します。

- 対応あり：ツールバーと右クリックメニューの「ViewSyncで表示」が有効
- 対応なし：ボタンが無効になり、理由を右側へ表示
- ボタンを押す：患者IDをURLへ含めず、既定ブラウザで対応するViewSyncデモを表示

## 検査とViewSyncデータの対応

`viewsync_studies.json` の `studies` に検査を登録します。

```json
"1.2.826.0.1.3680043.10.543.1001": {
  "available": true,
  "dataset": "mikan",
  "label": "ViewSyncデータあり（みかんデモ）"
}
```

キーはDICOMの `StudyInstanceUID` と完全に一致させます。`dataset` は現在 `mikan`、`banana`、`all` を指定できます。`available: false` の検査は利用不可として扱います。

手元のDICOMで起動フローだけを確認する場合は、開発用オプションで可否判定を一時的に上書きできます。

```bash
tools/start_dicom_viewer.command --demo-dataset mikan /path/to/dicom-folder
```

このオプションは連携確認専用です。本番相当の確認では対応表による判定を使用します。
