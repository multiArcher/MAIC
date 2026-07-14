#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_cc_switch_wsl.py"
BASHRC="$HOME/.bashrc"

if [[ ! -x "$SYNC_SCRIPT" ]]; then
  chmod +x "$SYNC_SCRIPT"
fi

START="# >>> cc-switch wsl sync >>>"
END="# <<< cc-switch wsl sync <<<"

if grep -Fq "$START" "$BASHRC" 2>/dev/null; then
  echo "cc-switch WSL sync is already installed in $BASHRC"
  exit 0
fi

cat >>"$BASHRC" <<EOF

$START
export CC_SWITCH_WSL_SYNC_SCRIPT="$SYNC_SCRIPT"
ccswitch-sync() {
  "\$CC_SWITCH_WSL_SYNC_SCRIPT" "\$@"
}
codex() {
  ccswitch-sync >/dev/null 2>&1 || true
  command codex "\$@"
}
$END
EOF

echo "Installed cc-switch WSL sync in $BASHRC"
echo "Run: source $BASHRC"
