#!/bin/zsh
set -e
TOOLS_DIR="${0:A:h}"
PROJECT_DIR="${TOOLS_DIR:h}"
cd "$TOOLS_DIR"

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

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if ! curl --silent --fail --max-time 1 http://127.0.0.1:5173/ >/dev/null 2>&1; then
  echo "ViewSyncローカルサーバーを起動します…"
  "$VENV_DIR/bin/python" -m http.server 5173 --bind 127.0.0.1 --directory "$PROJECT_DIR" >/tmp/viewsync-psp-mock-server.log 2>&1 &
  SERVER_PID=$!
fi

echo "PSP Mock Viewerを起動します…"
"$VENV_DIR/bin/python" dicom_viewer.py "$@"
