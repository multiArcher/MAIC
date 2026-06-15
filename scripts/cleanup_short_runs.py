#!/usr/bin/env python3
"exec" "python3" "$0" "$@"
# 默认执行python scripts/cleanup_short_runs.py --threshold 500000 --delete
import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


STEP_RE = re.compile(r"\bt_env:\s*([0-9][0-9_]*)")
MODEL_SAVE_RE = re.compile(r"Saving models to\s+(.+?/results/models/|results/models/)(.+?)/([0-9][0-9_]*)\b")
FAIL_RE = re.compile(r"\b(Failed after|Run Failed|Traceback|FATAL|ERROR)\b")

LOSS_TOKEN_REPLACEMENTS = (
    ("td_loss_weight=", "td_="),
    ("action_loss_weight=", "action_="),
    ("continue_loss_weight=", "continue_="),
    ("aux_loss_weight=", "aux_="),
    ("entropy_loss_weight=", "entropy_="),
)
ABNORMAL_SACRED_STATUSES = {"FAILED", "INTERRUPTED", "TIMEOUT"}


@dataclass
class RunArtifacts:
    token: str
    paths: set[Path] = field(default_factory=set)
    step_candidates: list[int] = field(default_factory=list)
    failed: bool = False

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


def normalise_token(token: str) -> str:
    for old, new in LOSS_TOKEN_REPLACEMENTS:
        token = token.replace(old, new)
    return token


def add_run(runs: dict[str, RunArtifacts], token: str) -> RunArtifacts:
    token = normalise_token(token)
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
            if subdir == "sacred" and path.is_dir():
                parse_sacred_run_dir(path, run)

    logs_dir = results_dir / "logs"
    if logs_dir.exists():
        for path in logs_dir.glob("*.log"):
            token = path.name[:-4]
            run = add_run(runs, token)
            run.paths.add(path)
            run.step_candidates.extend(parse_steps_from_log(path))
            if log_indicates_failure(path):
                run.failed = True

    models_dir = results_dir / "models"
    if models_dir.exists():
        for run_dir in models_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run = add_run(runs, run_dir.name)
            run.step_candidates.extend(parse_steps_from_model_dir(run_dir))

    return runs


def parse_sacred_run_dir(path: Path, run: RunArtifacts) -> None:
    for sacred_id_dir in path.iterdir():
        if not sacred_id_dir.is_dir():
            continue

        metrics_path = sacred_id_dir / "metrics.json"
        if metrics_path.exists():
            run.step_candidates.extend(parse_steps_from_metrics(metrics_path))

        cout_path = sacred_id_dir / "cout.txt"
        if cout_path.exists():
            run.step_candidates.extend(parse_steps_from_log(cout_path))
            if log_indicates_failure(cout_path):
                run.failed = True

        run_path = sacred_id_dir / "run.json"
        if run_path.exists():
            status, steps = parse_sacred_run_json(run_path)
            run.step_candidates.extend(steps)
            if status in ABNORMAL_SACRED_STATUSES:
                run.failed = True


def parse_steps_from_metrics(path: Path) -> list[int]:
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return []

    steps: list[int] = []
    if not isinstance(data, dict):
        return steps

    for metric in data.values():
        if not isinstance(metric, dict):
            continue
        raw_steps = metric.get("steps")
        if isinstance(raw_steps, list):
            steps.extend(step for step in raw_steps if isinstance(step, int))
    return steps


def parse_sacred_run_json(path: Path) -> tuple[str | None, list[int]]:
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return None, []

    if not isinstance(data, dict):
        return None, []

    steps: list[int] = []
    status = data.get("status")
    captured_out = data.get("captured_out")
    if isinstance(captured_out, str):
        steps.extend(parse_steps_from_text(captured_out))
    return status if isinstance(status, str) else None, steps


def parse_steps_from_log(path: Path) -> list[int]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []

    return parse_steps_from_text(text)


def parse_steps_from_text(text: str) -> list[int]:
    steps: list[int] = []

    for match in STEP_RE.finditer(text):
        steps.append(normalise_step(match.group(1)))
    for match in MODEL_SAVE_RE.finditer(text):
        steps.append(normalise_step(match.group(3)))
    return steps


def log_indicates_failure(path: Path) -> bool:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return FAIL_RE.search(text) is not None


def parse_steps_from_model_dir(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return steps


def collect_root_logs(root_log_dir: Path, runs: dict[str, RunArtifacts]) -> None:
    if not root_log_dir.exists():
        return

    for path in root_log_dir.glob("*.log"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue

        matched_tokens = {
            normalise_token(match.group(2))
            for match in MODEL_SAVE_RE.finditer(text)
            if normalise_token(match.group(2)) in runs
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
            if FAIL_RE.search(text):
                runs[token].failed = True


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
            if run.failed:
                run.step_candidates.append(0)
                candidates.append(run)
            elif delete_unknown:
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
