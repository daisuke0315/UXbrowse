#!/bin/zsh
set -eu
VIEWSYNC_BUILDER_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$VIEWSYNC_BUILDER_DIR"
if [[ ! -x .venv/bin/python ]]; then
  print '開発用Python環境がありません。README.mdのセットアップ手順を参照してください。'
  exit 1
fi
exec .venv/bin/python -m viewsync_builder serve "$@"
