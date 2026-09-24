#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
TOOLS_DIR="$SCRIPT_DIR/.tools/gs-browser"
CODEX_PNPM="/Users/daisuketanii/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/pnpm"

if command -v pnpm >/dev/null 2>&1; then
  PNPM=$(command -v pnpm)
elif [[ -x "$CODEX_PNPM" ]]; then
  PNPM="$CODEX_PNPM"
else
  echo "Error: pnpm was not found. Install Node.js/pnpm first." >&2
  exit 1
fi

if [[ ! -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  echo "Error: Google Chrome is required for Metal 3DGS rendering." >&2
  exit 1
fi

mkdir -p "$TOOLS_DIR"
"$PNPM" install --dir "$TOOLS_DIR"
echo "3DGS browser renderer is ready."
