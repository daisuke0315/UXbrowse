#!/bin/zsh
set -u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8501
URL="http://localhost:${PORT}"
HEALTH_URL="${URL}/_stcore/health"
LOG_FILE="${PROJECT_DIR}/.streamlit/rc_gui.log"
PID_FILE="${PROJECT_DIR}/.streamlit/rc_gui.pid"

cd "$PROJECT_DIR"
export PATH="/opt/homebrew/bin:$PATH"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "GUIの実行環境が見つかりません。"
  echo "場所: ${PROJECT_DIR}/.venv"
  read -k 1 "?何かキーを押すと閉じます。"
  exit 1
fi

# 二重起動せず、すでに動いていればその画面を開きます。
if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  echo "RC Frame Studioはすでに起動しています。"
  open "$URL"
  exit 0
fi

echo "RC Frame Studioを起動しています…"
: > "$LOG_FILE"
.venv/bin/streamlit run tools/rc_frame_gui.py \
  --server.headless true \
  --server.port "$PORT" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for attempt in {1..40}; do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "起動しました。ブラウザを開きます。"
    open "$URL"
    echo ""
    echo "RC Frame Studioの使用中は、このターミナルを閉じないでください。"
    echo "終了するときは control + C を押します。"
    trap 'kill "$SERVER_PID" >/dev/null 2>&1' HUP INT TERM EXIT
    wait "$SERVER_PID"
    exit $?
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "起動に失敗しました。エラー内容："
    tail -40 "$LOG_FILE"
    read -k 1 "?何かキーを押すと閉じます。"
    exit 1
  fi
  sleep 0.25
done

echo "起動確認がタイムアウトしました。エラー内容："
tail -40 "$LOG_FILE"
read -k 1 "?何かキーを押すと閉じます。"
exit 1
