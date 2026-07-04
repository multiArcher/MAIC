#!/usr/bin/env python3
# python scripts/fix_result_loss_weight_names.py --apply --replace-existing
import argparse
import re
import shutil
import sys
from pathlib import Path


LOSS_KEYS = (
    "td_loss_weight",
    "action_loss_weight",
    "continue_loss_weight",
    "aux_loss_weight",
    "entropy_loss_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair result artifact names whose loss-weight suffix does not match "
            "the actual configuration recorded in results/logs/*.log. Dry-run by default."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--map-name",
        default=None,
        help=(
            "Optional map/environment name filter. If omitted, all logs under "
            "results/logs are processed."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "If the target path already exists, delete it before renaming. "
            "This is destructive for existing result directories."
        ),
    )
    return parser.parse_args()


def format_value(value: str) -> str:
    value = value.strip().rstrip(",")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number)) if value in {"0", "1"} else f"{number:.1f}"
    return f"{number:g}"


def parse_actual_weights(log_path: Path) -> dict[str, str]:
    weights: dict[str, str] = {}
    text = log_path.read_text(errors="ignore")
    for key in LOSS_KEYS:
        match = re.search(rf"'{key}':\s*([^,\n]+)", text)
        if match:
            weights[key] = format_value(match.group(1))
    return weights


def strip_loss_suffix(name: str) -> str:
    stem = name[:-4] if name.endswith(".log") else name
    stem = re.sub(r"-td_loss_weight=.*$", "", stem)
    stem = re.sub(r"-td_=.*$", "", stem)
    return stem


def canonical_name(base: str, weights: dict[str, str], suffix: str = "") -> str:
    loss_suffix = "".join(f"-{key}={weights[key]}" for key in LOSS_KEYS if key in weights)
    return f"{base}{loss_suffix}{suffix}"


def matching_artifacts(results_dir: Path, old_token: str) -> list[tuple[Path, Path]]:
    artifacts: list[tuple[Path, Path]] = []
    for subdir in ("models", "tensorboard_logs", "sacred"):
        path = results_dir / subdir / old_token
        if path.exists():
            artifacts.append((path, path))
    log_path = results_dir / "logs" / f"{old_token}.log"
    if log_path.exists():
        artifacts.append((log_path, log_path))
    return artifacts


def target_for(path: Path, old_token: str, new_token: str) -> Path:
    if path.name == f"{old_token}.log":
        return path.with_name(f"{new_token}.log")
    if path.name == old_token:
        return path.with_name(new_token)
    raise ValueError(f"Unsupported artifact path: {path}")


def remove_existing_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    if not logs_dir.exists():
        print(f"ERROR: missing logs directory: {logs_dir}", file=sys.stderr)
        return 1

    renames: list[tuple[Path, Path]] = []
    seen_old_tokens: set[str] = set()
    log_pattern = f"*{args.map_name}*.log" if args.map_name else "*.log"
    for log_path in sorted(logs_dir.glob(log_pattern)):
        weights = parse_actual_weights(log_path)
        if set(weights) != set(LOSS_KEYS):
            continue

        old_token = log_path.name[:-4]
        if old_token in seen_old_tokens:
            continue
        seen_old_tokens.add(old_token)

        base = strip_loss_suffix(old_token)
        new_token = canonical_name(base, weights)
        if old_token == new_token:
            continue

        for old_path, _ in matching_artifacts(results_dir, old_token):
            renames.append((old_path, target_for(old_path, old_token, new_token)))

    if not renames:
        print("No mismatched result names found.")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: {len(renames)} path(s) would be renamed.")
    failures = 0
    for old_path, new_path in renames:
        print(f"{old_path} -> {new_path}")
        if not args.apply:
            continue
        if new_path.exists():
            if not args.replace_existing:
                print(f"ERROR: target exists, skipped: {new_path}", file=sys.stderr)
                failures += 1
                continue
            print(f"REPLACE: deleting existing target: {new_path}", file=sys.stderr)
            remove_existing_target(new_path)
        try:
            old_path.rename(new_path)
        except OSError as exc:
            print(f"ERROR: failed to rename {old_path}: {exc}", file=sys.stderr)
            failures += 1

    if not args.apply:
        print("Dry-run only. Re-run with --apply to rename.")
        print("Use --apply --replace-existing to overwrite existing targets.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
