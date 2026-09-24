#!/usr/bin/env python3
"""Run baseline missions (mission × seed) in parallel across isolated slots.

Each slot owns one P-core (2 HT threads), its own ROS domain ID, and its own
Ignition partition so instances never interfere.  Jobs from all missions and
seeds are pooled and dispatched to whichever slot finishes first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import SimpleQueue


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_SCT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = LLM_SCT_ROOT.parent.parent        # /home/ecem/sct_ws
RESULTS_ROOT = LLM_SCT_ROOT / "new_results" / "baseline"
INSTALL_SETUP = WORKSPACE_ROOT / "install" / "setup.bash"


def sourced_environment() -> dict[str, str]:
    """Return an environment dict with the sct_ws workspace sourced."""
    if not INSTALL_SETUP.is_file():
        raise FileNotFoundError(
            f"Workspace setup not found: {INSTALL_SETUP}\n"
            f"Build the workspace first with: colcon build"
        )
    shell_cmd = (
        f"source {INSTALL_SETUP} >/dev/null 2>&1 && "
        f"{sys.executable} -c 'import os, json; print(json.dumps(dict(os.environ)))'"
    )
    result = subprocess.run(["bash", "-c", shell_cmd], check=True, capture_output=True, text=True)
    env = json.loads(result.stdout)
    env.update({k: v for k, v in os.environ.items() if k not in env})
    return env


# Initialized eagerly in main() before the thread pool starts, so all worker
# threads see a fully populated dict and never race to call sourced_environment().
_SOURCED_ENV: dict[str, str] = {}


def get_env() -> dict[str, str]:
    return _SOURCED_ENV
MISSIONS = ("exploration", "patrolling", "delivery")
PACKAGES = {
    "exploration": "leo_exploration",
    "patrolling":  "leo_patrolling",
    "delivery":    "leo_delivery",
}

# ---------------------------------------------------------------------------
# CPU affinity — Intel Core Ultra 7 155H (same slots as LLM script)
# ---------------------------------------------------------------------------
_SLOT_CPUS: tuple[frozenset[int], ...] = (
    frozenset({0, 5}),
    frozenset({1, 2}),
    frozenset({3, 4}),
    frozenset({6, 7}),
    frozenset({8, 9}),
    frozenset({10, 11}),
)
DEFAULT_PARALLEL = len(_SLOT_CPUS)


# ---------------------------------------------------------------------------
# Process-group helpers
# ---------------------------------------------------------------------------

def active_processes_in_group(process_group_id: int) -> list[int]:
    active: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2:].split()
            state, group = fields[0], int(fields[2])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if group == process_group_id and state != "Z":
            active.append(int(stat_path.parent.name))
    return active


def stop_process_group(process_group_id: int) -> None:
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
    raise RuntimeError(f"process group {process_group_id} did not terminate")


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def has_saved_run(mission: str, seed: str, duration: float) -> bool:
    """Return True if a valid completed run for this (mission, seed) exists."""
    run_root = RESULTS_ROOT / f"results_{mission}" / f"seed_{seed}"
    for status_path in run_root.glob("**/SAVE_STATUS.txt"):
        status = status_path.read_text(encoding="utf-8", errors="replace")
        result_path = status_path.parent / "task_result.csv"
        if (
            f"random_seed: {seed}" in status
            and "Saving OK" in status
            and result_path.is_file()
        ):
            with result_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                last = rows[-1]
                if last["success"].casefold() == "true":
                    return True
                if float(last["duration_s"]) >= duration - 1.0:
                    return True
    return False


# ---------------------------------------------------------------------------
# Single-job runner
# ---------------------------------------------------------------------------

def run_job(
    *,
    mission: str,
    seed: str,
    duration: float,
    slot: int,
    domain_base: int,
    label: str,
    yaml_path: str | None = None,
) -> bool:
    results_dir = RESULTS_ROOT / f"results_{mission}" / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = RESULTS_ROOT / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = log_dir / f"ros_slot{slot}"
    ros_log_dir.mkdir(parents=True, exist_ok=True)

    domain_id = domain_base + slot
    partition = f"baseline_slot{slot}"

    env = get_env().copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["IGN_PARTITION"] = partition
    env["GZ_PARTITION"] = partition
    env["ROS_LOG_DIR"] = str(ros_log_dir)

    cpus = ",".join(str(c) for c in sorted(_SLOT_CPUS[slot]))

    cmd = [
        "taskset", "-c", cpus,
        "ros2", "launch", PACKAGES[mission], "leo_gz.launch.py",
        "total_robots:=3",
        f"run_duration:={duration}",
        f"random_seed:={seed}",
        "headless:=true",
        "overhead_camera:=false",
        "record_video:=false",
        f"results_dir:={results_dir}",
    ]
    if yaml_path:
        cmd.append(f"metadata_yaml_path:={yaml_path}")

    log_path = log_dir / f"{mission}_seed_{seed}_slot{slot}.log"
    wall_start = time.monotonic()
    print(f"{label}: starting", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait()
        finally:
            stop_process_group(proc.pid)

    wall_elapsed = time.monotonic() - wall_start
    ok = has_saved_run(mission, seed, duration)
    status = "saved ✓" if ok else "FAILED"
    print(f"{label}: {status}  wall={wall_elapsed:.0f}s", flush=True)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default="1001,1002,1003,1004,1005,1006,1007,1008,1009,1010",
        help="Comma-separated random seeds (default: 1001–1010)",
    )
    parser.add_argument("--exploration-duration", type=float, default=60.0,
                        help="Wall-clock seconds for exploration (default 60 = 120 sim-sec at RTF=2)")
    parser.add_argument("--patrolling-duration",  type=float, default=300.0,
                        help="Wall-clock seconds for patrolling  (default 300 = 600 sim-sec at RTF=2)")
    parser.add_argument("--delivery-duration",    type=float, default=300.0,
                        help="Wall-clock seconds for delivery    (default 300 = 600 sim-sec at RTF=2)")
    parser.add_argument(
        "--mission", choices=MISSIONS, action="append", dest="missions",
        help="Run only this mission (repeat for multiple). Default: all three.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL,
        help=f"Simultaneous Gazebo instances (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument("--domain-base", type=int, default=51,
                        help="First ROS domain ID for baseline slots (default 51)")
    parser.add_argument(
        "--yaml-path", default=None,
        help="Supervisor YAML for ALL missions (metadata_yaml_path). "
             "Overridden per-mission by --exploration-yaml-path etc.",
    )
    parser.add_argument("--exploration-yaml-path", default=None,
                        help="Supervisor YAML for exploration (overrides --yaml-path).")
    parser.add_argument("--patrolling-yaml-path",  default=None,
                        help="Supervisor YAML for patrolling (overrides --yaml-path).")
    parser.add_argument("--delivery-yaml-path",    default=None,
                        help="Supervisor YAML for delivery (overrides --yaml-path).")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one seed")

    # Per-mission YAML paths: specific flag > --yaml-path > None (package default)
    yaml_paths = {
        "exploration": args.exploration_yaml_path or args.yaml_path,
        "patrolling":  args.patrolling_yaml_path  or args.yaml_path,
        "delivery":    args.delivery_yaml_path     or args.yaml_path,
    }

    selected = tuple(args.missions or MISSIONS)
    durations = {
        "exploration": args.exploration_duration,
        "patrolling":  args.patrolling_duration,
        "delivery":    args.delivery_duration,
    }

    all_jobs: list[tuple[str, str]] = [
        (mission, seed)
        for mission in selected
        for seed in seeds
    ]

    pending = [
        (mission, seed)
        for mission, seed in all_jobs
        if not (args.resume and has_saved_run(mission, seed, durations[mission]))
    ]

    n_slots = min(args.parallel, len(pending))
    print(
        f"Jobs total={len(all_jobs)}  pending={len(pending)}  "
        f"parallel={n_slots}  seeds={seeds}",
        flush=True,
    )
    if not pending:
        print("All jobs already saved — nothing to do.", flush=True)
        return 0

    # Source the workspace once on the main thread before workers start,
    # so every thread sees a fully populated environment (no race condition).
    global _SOURCED_ENV
    print(f"Sourcing workspace: {INSTALL_SETUP}", flush=True)
    _SOURCED_ENV = sourced_environment()
    if "leo_exploration" not in _SOURCED_ENV.get("AMENT_PREFIX_PATH", ""):
        print(
            "WARNING: leo_exploration not in AMENT_PREFIX_PATH after sourcing.\n"
            f"  AMENT_PREFIX_PATH={_SOURCED_ENV.get('AMENT_PREFIX_PATH', '(unset)')}",
            flush=True,
        )

    slot_pool: SimpleQueue[int] = SimpleQueue()
    for i in range(n_slots):
        slot_pool.put(i)

    total = len(pending)

    def _run(indexed_job: tuple[int, tuple[str, str]]) -> bool:
        job_num, (mission, seed) = indexed_job
        label = f"[{job_num}/{total} {mission}] seed={seed}"
        slot = slot_pool.get()
        try:
            return run_job(
                mission=mission,
                seed=seed,
                duration=durations[mission],
                slot=slot,
                domain_base=args.domain_base,
                label=label,
                yaml_path=yaml_paths[mission],
            )
        finally:
            slot_pool.put(slot)

    indexed_pending = list(enumerate(pending, start=1))
    failures = 0

    with ThreadPoolExecutor(max_workers=n_slots) as executor:
        futures = {executor.submit(_run, item): item for item in indexed_pending}
        for future in as_completed(futures):
            if not future.result():
                failures += 1

    print(
        f"\nDone. {total - failures}/{total} jobs saved, {failures} failed.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
