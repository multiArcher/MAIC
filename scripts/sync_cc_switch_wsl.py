#!/usr/bin/env python3
"""Sync Windows CC Switch Codex settings into WSL Codex config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path


DEFAULT_DB = Path("/mnt/c/Users/Jacob/.cc-switch/cc-switch.db")


def backup_and_write(path: Path, data: bytes, dry_run: bool) -> str:
    old = path.read_bytes() if path.exists() else None
    if old == data:
        return "unchanged"

    if dry_run:
        return "would update"

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = time.strftime("%Y%m%d%H%M%S")
        backup = path.with_name(f"{path.name}.cc-switch-backup-{stamp}")
        shutil.copy2(path, backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "updated"


def current_codex_provider(db_path: Path) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"CC Switch DB not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            """
            SELECT id, name, settings_config
            FROM providers
            WHERE app_type = 'codex' AND is_current = 1
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise RuntimeError("No current Codex provider found in CC Switch DB")

    provider_id, name, settings_config = row
    settings = json.loads(settings_config)
    return {"id": provider_id, "name": name, "settings": settings}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync the current Windows CC Switch Codex provider into WSL."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    provider = current_codex_provider(args.db)
    settings = provider["settings"]
    config = settings.get("config")
    auth = settings.get("auth")

    if not isinstance(config, str) or not config.strip():
        raise RuntimeError(f"Current provider {provider['name']} has no Codex config")
    if not isinstance(auth, dict):
        raise RuntimeError(f"Current provider {provider['name']} has no auth object")

    auth_data = json.dumps(auth, separators=(",", ":")).encode("utf-8") + b"\n"
    config_data = config.rstrip().encode("utf-8") + b"\n"

    config_status = backup_and_write(
        args.codex_home / "config.toml", config_data, args.dry_run
    )
    auth_status = backup_and_write(args.codex_home / "auth.json", auth_data, args.dry_run)

    print(f"provider: {provider['name']} ({provider['id']})")
    print(f"config.toml: {config_status}")
    print(f"auth.json: {auth_status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
