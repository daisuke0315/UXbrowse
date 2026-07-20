#!/bin/zsh
set -e
cd "${0:A:h}"

VENV_DIR=".dicom_viewer_venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "DICOM Viewerの初回セットアップを行います…"
  python3 -m venv "$VENV_DIR"
fi

if ! "$VENV_DIR/bin/python" -c 'import numpy, pydicom, PyQt6' >/dev/null 2>&1; then
  echo "必要なライブラリをインストールしています…"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r requirements_dicom_viewer.txt
fi

echo "DICOM Viewerを起動します…"
exec "$VENV_DIR/bin/python" dicom_viewer.py
