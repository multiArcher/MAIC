#!/usr/bin/env python3
"""
Remove the substring "loss_weight" from file and directory names under a tree.

By default this is a dry run. Add --apply to actually rename paths.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DEFAULT_ROOT = r"D:\Users\Jacob\Downloads\results"
TOKEN = "loss_weight"


def normalize_root(raw_path: str) -> Path:
    """Accept either a Windows path or a normal local path."""
    path = os.path.expanduser(raw_path)

    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

    return Path(path)


def replacement_name(name: str) -> str | None:
    new_name = name.replace(TOKEN, "")
    if new_name == name:
        return None
    if not new_name:
        raise ValueError(
            f'Cannot remove "{TOKEN}" from "{name}" because the result is empty.'
        )
    return new_name


def collect_renames(root: Path) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []

    for current_root, dir_names, file_names in os.walk(root, topdown=False):
        current = Path(current_root)

        for name in file_names:
            new_name = replacement_name(name)
            if new_name is not None:
                renames.append((current / name, current / new_name))

        for name in dir_names:
            new_name = replacement_name(name)
            if new_name is not None:
                renames.append((current / name, current / new_name))

    return renames


def apply_renames(renames: list[tuple[Path, Path]], dry_run: bool) -> int:
    failures = 0

    for old_path, new_path in renames:
        print(f"{old_path} -> {new_path}")

        if dry_run:
            continue

        if new_path.exists():
            print(f"ERROR: target already exists, skipped: {new_path}", file=sys.stderr)
            failures += 1
            continue

        try:
            old_path.rename(new_path)
        except OSError as exc:
            print(f"ERROR: failed to rename {old_path}: {exc}", file=sys.stderr)
            failures += 1

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f'Recursively remove "{TOKEN}" from file and directory names. '
            "Runs as a dry run unless --apply is provided."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Root directory to process. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files and directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = normalize_root(args.root)

    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 1

    try:
        renames = collect_renames(root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not renames:
        print(f'No paths containing "{TOKEN}" found under {root}.')
        return 0

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: {len(renames)} path(s) will be renamed under {root}.")

    failures = apply_renames(renames, dry_run=not args.apply)
    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    if not args.apply:
        print("Dry run only. Re-run with --apply to make these changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
