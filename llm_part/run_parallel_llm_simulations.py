#!/usr/bin/env python3
"""Run one isolated sequential simulation worker per mission in parallel."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_SCT_ROOT = SCRIPT_DIR.parent
YAML_ROOT = LLM_SCT_ROOT / "automata" / "resulting_automata" / "YAML"
RESULTS_ROOT = LLM_SCT_ROOT / "new_results" / "llm"
RUNNER = SCRIPT_DIR / "run_leo_with_llm_yaml.py"
MISSIONS = ("exploration", "patrolling", "delivery")
RESULT_NAMES = {
    "exploration": "results_exploration",
    "patrolling": "results_patrolling",
    "delivery": "results_delivery",
}


def active_processes_in_group(process_group_id: int) -> list[int]:
    """Return live (non-zombie) PIDs belonging to a process group."""
    active: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            state, group = fields[0], int(fields[2])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if group == process_group_id and state != "Z":
            active.append(int(stat_path.parent.name))
    return active


def stop_process_group(process_group_id: int) -> None:
    """Stop only the process group created for one simulation launch."""
    for sig, timeout in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not active_processes_in_group(process_group_id):
                return
            time.sleep(0.1)
    raise RuntimeError(
        f"simulation process group {process_group_id} did not terminate"
    )


def generated_yamls(mission: str) -> list[Path]:
    return sorted(
        YAML_ROOT.glob(f"{mission}/prompt_*/generation_*/S_*.yaml"),
        key=lambda path: (
            int(path.parents[1].name.removeprefix("prompt_")),
            int(path.parent.name.removeprefix("generation_")),
        ),
    )


def has_saved_run(
    mission: str, yaml_path: Path, seed: str, duration: float
) -> bool:
    run_root = (
        RESULTS_ROOT / RESULT_NAMES[mission] / yaml_path.stem / "robots_3"
    )
    for status_path in run_root.glob("run_*/SAVE_STATUS.txt"):
        status = status_path.read_text(encoding="utf-8", errors="replace")
        result_path = status_path.parent / "task_result.csv"
        if (
            f"random_seed: {seed}" in status
            and f"run_duration: {duration}" in status
            and "Saving OK" in status
            and result_path.is_file()
        ):
            with result_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows and (
                rows[-1]["success"].casefold() == "true"
                or float(rows[-1]["duration_s"]) >= duration - 1.0
            ):
                return True
    return False


def run_worker(
    mission: str,
    domain_id: int,
    seed: str,
    duration: float,
    resume: bool,
) -> tuple[str, int, int]:
    environment = os.environ.copy()
    partition = f"llm_{mission}_{domain_id}"
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    environment["IGN_PARTITION"] = partition
    environment["GZ_PARTITION"] = partition

    yamls = generated_yamls(mission)
    failures = 0
    completed = 0
    for index, yaml_path in enumerate(yamls, start=1):
        label = f"[{mission} {index}/{len(yamls)}] {yaml_path}"
        if resume and has_saved_run(mission, yaml_path, seed, duration):
            print(f"{label}: seed {seed} already saved; skipping", flush=True)
            continue
        print(f"{label}: starting with seed {seed}", flush=True)
        command = [
            sys.executable,
            str(RUNNER),
            mission,
            "--yaml",
            str(yaml_path),
            "--total-robots",
            "3",
            "--run-duration",
            str(duration),
            "--random-seed",
            seed,
            "--headless",
            "--no-record-video",
        ]
        process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )
        try:
            return_code = process.wait()
        finally:
            stop_process_group(process.pid)
        if has_saved_run(mission, yaml_path, seed, duration):
            completed += 1
            print(f"{label}: saved", flush=True)
        else:
            failures += 1
            print(
                f"{label}: FAILED (launcher exit code {return_code})",
                flush=True,
            )
    return mission, completed, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="1001")
    parser.add_argument("--exploration-duration", type=float, default=300.0)
    parser.add_argument("--patrolling-duration", type=float, default=600.0)
    parser.add_argument("--delivery-duration", type=float, default=600.0)
    parser.add_argument("--domain-base", type=int, default=31)
    parser.add_argument(
        "--mission",
        choices=MISSIONS,
        action="append",
        dest="missions",
        help="Run only this mission (repeat to select multiple missions)",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    selected_missions = tuple(args.missions or MISSIONS)
    counts = {
        mission: len(generated_yamls(mission)) for mission in selected_missions
    }
    if any(count != 15 for count in counts.values()):
        print(f"error: expected 15 YAMLs per mission, found {counts}", file=sys.stderr)
        return 2

    durations = {
        "exploration": args.exploration_duration,
        "patrolling": args.patrolling_duration,
        "delivery": args.delivery_duration,
    }
    if any(duration <= 0 for duration in durations.values()):
        parser.error("all mission durations must be positive")

    failures = 0
    with ThreadPoolExecutor(max_workers=len(selected_missions)) as executor:
        futures = [
            executor.submit(
                run_worker,
                mission,
                args.domain_base + offset,
                str(args.seed),
                durations[mission],
                args.resume,
            )
            for offset, mission in enumerate(selected_missions)
        ]
        for future in as_completed(futures):
            mission, completed, worker_failures = future.result()
            failures += worker_failures
            print(
                f"[{mission}] worker finished: {completed} new, "
                f"{worker_failures} failed",
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
