#!/usr/bin/env python3
# 默认执行python scripts/cleanup_short_runs.py --threshold 500000 --delete
import argparse
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


STEP_RE = re.compile(r"\bt_env:\s*([0-9][0-9_]*)")
MODEL_SAVE_RE = re.compile(r"Saving models to\s+(.+?/results/models/|results/models/)(.+?)/([0-9][0-9_]*)\b")


@dataclass
class RunArtifacts:
    token: str
    paths: set[Path] = field(default_factory=set)
    step_candidates: list[int] = field(default_factory=list)

    @property
    def max_steps(self) -> int | None:
        if not self.step_candidates:
            return None
        return max(self.step_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove run artifacts whose inferred interaction steps are below a threshold. "
            "Default mode is dry-run."
        )
    )
    parser.add_argument("--threshold", type=int, default=500_000, help="Maximum steps to delete below.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    parser.add_argument("--root-log-dir", default="log", help="Root stdout/stderr log directory.")
    parser.add_argument("--delete", action="store_true", help="Actually delete matched artifacts.")
    parser.add_argument(
        "--no-root-log",
        action="store_true",
        help="Do not include matching files under the root log directory.",
    )
    parser.add_argument(
        "--delete-unknown",
        action="store_true",
        help="Delete runs whose step count cannot be inferred. Off by default.",
    )
    return parser.parse_args()


def normalise_step(raw: str) -> int:
    return int(raw.replace("_", ""))


def add_run(runs: dict[str, RunArtifacts], token: str) -> RunArtifacts:
    if token not in runs:
        runs[token] = RunArtifacts(token=token)
    return runs[token]


def collect_result_artifacts(results_dir: Path) -> dict[str, RunArtifacts]:
    runs: dict[str, RunArtifacts] = {}

    for subdir in ("sacred", "tensorboard_logs", "models"):
        base = results_dir / subdir
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.name.startswith("."):
                continue
            run = add_run(runs, path.name)
            run.paths.add(path)

    logs_dir = results_dir / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("*.log"):
            token = path.name[:-4]
            run = add_run(runs, token)
            run.paths.add(path)
            run.step_candidates.extend(parse_steps_from_log(path))

    models_dir = results_dir / "models"
    if models_dir.exists():
        for run_dir in models_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = add_run(runs, run_dir.name)
            run.step_candidates.extend(parse_steps_from_model_dir(run_dir))

    return runs


def parse_steps_from_log(path: Path) -> list[int]:
    steps: list[int] = []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return steps

    for match in STEP_RE.finditer(text):
        steps.append(normalise_step(match.group(1)))
    for match in MODEL_SAVE_RE.finditer(text):
        steps.append(normalise_step(match.group(3)))
    return steps


def parse_steps_from_model_dir(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return steps


def collect_root_logs(root_log_dir: Path, runs: dict[str, RunArtifacts]) -> None:
    if not root_log_dir.exists():
        return

    token_by_model_path = {f"results/models/{token}/": token for token in runs}
    for path in root_log_dir.glob("*.log"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue

        matched_tokens = {
            token
            for marker, token in token_by_model_path.items()
            if marker in text
        }
        for token in matched_tokens:
            runs[token].paths.add(path)
            if path.name.endswith("_log.log"):
                err_path = path.with_name(path.name.replace("_log.log", "_err.log"))
                if err_path.exists():
                    runs[token].paths.add(err_path)
            elif path.name.endswith("_err.log"):
                log_path = path.with_name(path.name.replace("_err.log", "_log.log"))
                if log_path.exists():
                    runs[token].paths.add(log_path)
            runs[token].step_candidates.extend(parse_steps_from_log(path))


def choose_deletion_candidates(
        runs: dict[str, RunArtifacts],
        threshold: int,
        delete_unknown: bool,
    ) -> tuple[list[RunArtifacts], list[RunArtifacts]]:
    candidates: list[RunArtifacts] = []
    skipped_unknown: list[RunArtifacts] = []

    for run in runs.values():
        max_steps = run.max_steps
        if max_steps is None:
            if delete_unknown:
                candidates.append(run)
            else:
                skipped_unknown.append(run)
        elif max_steps < threshold:
            candidates.append(run)

    candidates.sort(key=lambda r: (r.max_steps is None, r.max_steps or -1, r.token))
    skipped_unknown.sort(key=lambda r: r.token)
    return candidates, skipped_unknown


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    root_log_dir = Path(args.root_log_dir)

    runs = collect_result_artifacts(results_dir)
    if not args.no_root_log:
        collect_root_logs(root_log_dir, runs)

    candidates, skipped_unknown = choose_deletion_candidates(
        runs, args.threshold, args.delete_unknown
    )

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"{mode}: threshold={args.threshold}, candidates={len(candidates)}, unknown_skipped={len(skipped_unknown)}")

    for run in candidates:
        step_text = "unknown" if run.max_steps is None else str(run.max_steps)
        print(f"\n[{step_text} steps] {run.token}")
        for path in sorted(run.paths):
            print(f"  {path}")
            if args.delete:
                remove_path(path)

    if skipped_unknown:
        print("\nSkipped runs with unknown step count. Use --delete-unknown to include them:")
        for run in skipped_unknown[:50]:
            print(f"  {run.token}")
        if len(skipped_unknown) > 50:
            print(f"  ... {len(skipped_unknown) - 50} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
