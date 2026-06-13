#!/bin/bash
set -euo pipefail

SC2PATH="${SC2PATH:-$HOME/.local/share/StarCraftII}"
MAP_DIR="$SC2PATH/Maps/SMAC_Maps"
URL="https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ ! -d "$SC2PATH" ]]; then
    echo "ERROR: SC2PATH does not exist: $SC2PATH" >&2
    echo "Set SC2PATH to your StarCraftII install, then rerun this script." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required." >&2
    exit 1
fi

mkdir -p "$MAP_DIR"

echo "Downloading SMACv2 maps from $URL"
curl -L "$URL" -o "$TMP_DIR/SMAC_Maps.zip"

echo "Installing SMACv2 maps into $MAP_DIR"
if command -v unzip >/dev/null 2>&1; then
    unzip -o "$TMP_DIR/SMAC_Maps.zip" -d "$TMP_DIR" >/dev/null
else
    python3 - "$TMP_DIR/SMAC_Maps.zip" "$TMP_DIR" <<'PY'
import sys
import zipfile

zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
PY
fi
while IFS= read -r map_file; do
    target="$MAP_DIR/$(basename "$map_file")"
    if [[ ! -f "$target" ]]; then
        cp "$map_file" "$target"
    fi
done < <(find "$TMP_DIR" -type f -name "*.SC2Map" | sort)

if [[ ! -f "$MAP_DIR/32x32_flat.SC2Map" ]]; then
    echo "ERROR: 32x32_flat.SC2Map was not installed into $MAP_DIR" >&2
    exit 1
fi

echo "SMACv2 maps installed successfully."
echo "Verified: $MAP_DIR/32x32_flat.SC2Map"
