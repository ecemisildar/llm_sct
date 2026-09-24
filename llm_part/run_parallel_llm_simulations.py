#!/usr/bin/env python3
"""Run LLM simulation jobs (mission × YAML × seed) in parallel across isolated slots.

Each slot owns one P-core (2 HT threads), its own ROS domain ID, and its own
Ignition partition, so instances never interfere with each other.  Jobs from
all missions and seeds are pooled together and dispatched to whichever slot
finishes first, keeping every slot busy until all 900 (or however many) runs
are done.
"""

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
from queue import SimpleQueue


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_SCT_ROOT = SCRIPT_DIR.parent
YAML_ROOT = LLM_SCT_ROOT / "automata" / "resulting_automata" / "YAML"
RESULTS_ROOT = LLM_SCT_ROOT / "new_results" / "llm"
RUNNER = SCRIPT_DIR / "run_leo_with_llm_yaml.py"
MISSIONS = ("exploration", "patrolling", "delivery")

# ---------------------------------------------------------------------------
# CPU affinity — Intel Core Ultra 7 155H
#
# Each slot gets ONE P-core (2 HT threads).  Profiling showed each Gazebo
# instance uses only ~0.3 of a P-core at RTF=1.0, so 6 slots across 6
# P-cores comfortably fits with headroom.  E-cores (12-21) are intentionally
# left free for the OS and ROS daemon overhead.
#
# P-core HT pairs on this chip:
#   P-core 0 → threads {0, 5}
#   P-core 1 → threads {1, 2}
#   P-core 2 → threads {3, 4}
#   P-core 3 → threads {6, 7}
#   P-core 4 → threads {8, 9}
#   P-core 5 → threads {10, 11}
# ---------------------------------------------------------------------------
_SLOT_CPUS: tuple[frozenset[int], ...] = (
    frozenset({0, 5}),    # slot 0 → P-core 0
    frozenset({1, 2}),    # slot 1 → P-core 1
    frozenset({3, 4}),    # slot 2 → P-core 2
    frozenset({6, 7}),    # slot 3 → P-core 3
    frozenset({8, 9}),    # slot 4 → P-core 4
    frozenset({10, 11}),  # slot 5 → P-core 5
)
DEFAULT_PARALLEL = len(_SLOT_CPUS)  # 6


# ---------------------------------------------------------------------------
# Process-group helpers (unchanged)
# ---------------------------------------------------------------------------

def active_processes_in_group(process_group_id: int) -> list[int]:
    """Return live (non-zombie) PIDs belonging to a process group."""
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
    """SIGTERM → SIGKILL the process group created for one simulation launch."""
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


# ---------------------------------------------------------------------------
# YAML discovery and result checking (unchanged)
# ---------------------------------------------------------------------------

def generated_yamls(
    mission: str,
    generations: list[int] | None = None,
    yaml_root: Path | None = None,
) -> list[Path]:
    root = yaml_root if yaml_root is not None else YAML_ROOT
    pattern = f"{mission}/*/prompt_*/generation_*/S_*.yaml"
    paths = root.glob(pattern)
    if generations:
        paths = (p for p in paths if p.parent.name in {f"generation_{g}" for g in generations})
    import re as _re
    def _prompt_key(path: Path) -> tuple[int, str]:
        name = path.parents[1].name  # e.g. "prompt_1", "prompt_1_old", "prompt_6"
        m = _re.match(r"prompt_(\d+)(.*)", name)
        return (int(m.group(1)), m.group(2)) if m else (0, name)

    return sorted(
        paths,
        key=lambda path: (
            _prompt_key(path),
            int(path.parent.name.removeprefix("generation_")),
        ),
    )


def has_saved_run(
    mission: str, yaml_path: Path, seed: str, duration: float,
    results_root: Path | None = None,
) -> bool:
    root = results_root if results_root is not None else RESULTS_ROOT
    # collision_mode is always 3 levels above the YAML file:
    # .../collision_mode/prompt_N/generation_N/S_*.yaml
    collision_mode = yaml_path.parents[2].name
    run_root = (
        root / mission / collision_mode / f"seed_{seed}" / yaml_path.stem
    )
    # Support both leo_gz structure (run_*/SAVE_STATUS.txt) and
    # balanced_delivery structure (robots_N/run_*/SAVE_STATUS.txt)
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
                # Always accept a run where the task completed successfully —
                # no need to re-run with a longer duration.
                if last["success"].casefold() == "true":
                    return True
                # Accept a timed-out run only if it ran for at least the
                # currently requested duration (so shorter-duration runs are
                # re-tried when we raise the limit).
                # Allow 3 s tolerance — the sim timer can stop slightly early.
                if float(last["duration_s"]) >= duration - 3.0:
                    return True
    return False


# ---------------------------------------------------------------------------
# Single-job runner
# ---------------------------------------------------------------------------

def run_job(
    *,
    mission: str,
    yaml_path: Path,
    seed: str,
    duration: float,
    slot: int,
    domain_base: int,
    label: str,
    results_root: Path | None = None,
    color_order: str | None = None,
    balanced: bool = False,
    partition_prefix: str = "llm",
) -> bool:
    """Run one (mission, yaml, seed) simulation in the given slot.  Returns True on success."""
    domain_id = domain_base + slot
    partition = f"{partition_prefix}_slot{slot}"

    environment = os.environ.copy()
    environment.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "IGN_PARTITION": partition,
        "GZ_PARTITION": partition,
        # Route OGRE2 headless EGL rendering to the NVIDIA GPU (renderD129)
        # instead of the default Intel Arc (renderD128).
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "__NV_PRIME_RENDER_OFFLOAD": "1",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        "DRI_PRIME": "1",
    })

    # Isolate ROS logs per slot so concurrent workers never write to the same dir.
    ros_log_dir = LLM_SCT_ROOT / "new_results" / "llm_run_logs" / f"slot_{slot}"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)

    cpu_set = _SLOT_CPUS[slot] if slot < len(_SLOT_CPUS) else None

    def _pin_cpus() -> None:
        if cpu_set is not None:
            os.sched_setaffinity(0, cpu_set)

    command = [
        sys.executable, str(RUNNER),
        mission,
        "--yaml", str(yaml_path),
        "--total-robots", "3",
        "--run-duration", str(duration),
        "--random-seed", seed,
        "--headless",
        "--no-record-video",
    ]
    if results_root is not None:
        command += ["--results-root", str(results_root)]
    if color_order is not None:
        command += ["--color-order", color_order]
    if balanced:
        command += ["--balanced"]

    print(f"{label}: start  slot={slot} domain={domain_id}", flush=True)
    wall_start = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        preexec_fn=_pin_cpus,
    )
    pgid = process.pid
    try:
        return_code = process.wait()
    finally:
        stop_process_group(pgid)

    wall_elapsed = time.monotonic() - wall_start
    rtf = duration / wall_elapsed if wall_elapsed > 0 else 0.0
    ok = has_saved_run(mission, yaml_path, seed, duration, results_root=results_root)
    status = "saved ✓" if ok else f"FAILED (rc={return_code})"
    print(
        f"{label}: {status}  wall={wall_elapsed:.0f}s sim={duration:.0f}s RTF={rtf:.2f}",
        flush=True,
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default="1001",
        help="Comma-separated random seeds, e.g. 1001,1002,1003,... (default: 1001)",
    )
    parser.add_argument("--exploration-duration", type=float, default=300.0)
    parser.add_argument("--patrolling-duration", type=float, default=600.0)
    parser.add_argument("--delivery-duration", type=float, default=600.0)
    parser.add_argument("--partition-prefix", default="llm",
                        help="Gazebo partition prefix; use a unique value for concurrent batches")
    parser.add_argument("--domain-base", type=int, default=31,
                        help="First ROS domain ID; each slot gets domain_base+slot")
    parser.add_argument(
        "--mission", choices=MISSIONS, action="append", dest="missions",
        help="Run only this mission (repeat for multiple). Default: all three.",
    )
    parser.add_argument(
        "--generation", type=int, action="append", dest="generations",
        help="Only run YAMLs from this generation (repeat for multiple, e.g. --generation 1 --generation 2). Default: all.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL,
        help=f"Simultaneous Gazebo instances (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument(
        "--exclude-yaml-stems", default="",
        help="Comma-separated YAML stems to skip, e.g. S_20260904_144334,S_20260904_145959",
    )
    parser.add_argument(
        "--yaml-stems", default="",
        help="Comma-separated YAML stems to run exclusively (allowlist); all others are skipped",
    )
    parser.add_argument(
        "--exploration-yaml-path", default=None,
        help="Run a single explicit YAML for exploration instead of auto-discovery.",
    )
    parser.add_argument(
        "--patrolling-yaml-path", default=None,
        help="Run a single explicit YAML for patrolling instead of auto-discovery.",
    )
    parser.add_argument(
        "--delivery-yaml-path", default=None,
        help="Run a single explicit YAML for delivery instead of auto-discovery.",
    )
    parser.add_argument(
        "--color-order", default=None,
        help="Override patrolling color visit order for all jobs, e.g. red,green,blue. "
             "If omitted, each YAML's sibling .prompt.txt is used.",
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="Use balanced_delivery.launch.py (6 boxes, 2 zones) for delivery jobs.",
    )
    parser.add_argument(
        "--yaml-root", default=None,
        help="Root directory to discover YAMLs from instead of the default "
             "automata/resulting_automata/YAML.",
    )
    parser.add_argument(
        "--collision-mode", default=None,
        help="Only run YAMLs under this collision-mode folder, e.g. with_fixed_spec or fixed_collision.",
    )
    parser.add_argument(
        "--results-root", default=None,
        help="Root directory to save results into instead of new_results/llm.",
    )
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one seed")

    excluded_stems = {s.strip() for s in args.exclude_yaml_stems.split(",") if s.strip()}
    allowed_stems = {s.strip() for s in args.yaml_stems.split(",") if s.strip()}

    # Explicit per-mission YAML overrides (bypass directory discovery).
    explicit_yamls: dict[str, list[Path]] = {}
    for mission, flag in (
        ("exploration", args.exploration_yaml_path),
        ("patrolling",  args.patrolling_yaml_path),
        ("delivery",    args.delivery_yaml_path),
    ):
        if flag:
            p = Path(flag).expanduser().resolve()
            if not p.is_file():
                parser.error(f"--{mission}-yaml-path does not exist: {p}")
            explicit_yamls[mission] = [p]

    yaml_root     = Path(args.yaml_root).expanduser().resolve()     if args.yaml_root     else None
    results_root  = Path(args.results_root).expanduser().resolve()  if args.results_root  else None
    if yaml_root and not yaml_root.is_dir():
        parser.error(f"--yaml-root does not exist: {yaml_root}")
    if results_root:
        results_root.mkdir(parents=True, exist_ok=True)

    selected = tuple(args.missions or MISSIONS)
    durations = {
        "exploration": args.exploration_duration,
        "patrolling": args.patrolling_duration,
        "delivery": args.delivery_duration,
    }
    if any(d <= 0 for d in durations.values()):
        parser.error("all durations must be positive")

    # ── Build the full job list: every mission × yaml × seed combination ──
    all_jobs: list[tuple[str, Path, str]] = [
        (mission, yaml_path, seed)
        for mission in selected
        for yaml_path in (
            explicit_yamls[mission] if mission in explicit_yamls
            else generated_yamls(mission, args.generations, yaml_root=yaml_root)
        )
        if yaml_path.stem not in excluded_stems
        if not allowed_stems or yaml_path.stem in allowed_stems
        if not args.collision_mode or yaml_path.parents[2].name == args.collision_mode
        for seed in seeds
    ]
    if not all_jobs:
        print("error: no YAMLs found for the selected missions", file=sys.stderr)
        return 2

    # ── Skip already-completed jobs (--resume) ────────────────────────────
    pending = [
        (mission, yaml_path, seed)
        for mission, yaml_path, seed in all_jobs
        if not (
            args.resume
            and has_saved_run(mission, yaml_path, seed, durations[mission],
                              results_root=results_root)
        )
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

    # ── Slot pool: maps running workers to a fixed slot number ────────────
    # With max_workers=n_slots threads and n_slots slots, get() never blocks.
    slot_pool: SimpleQueue[int] = SimpleQueue()
    for i in range(n_slots):
        slot_pool.put(i)

    total = len(pending)

    def _run(indexed_job: tuple[int, tuple[str, Path, str]]) -> bool:
        job_num, (mission, yaml_path, seed) = indexed_job
        label = (
            f"[{job_num}/{total} {mission}] {yaml_path.name} seed={seed}"
        )
        slot = slot_pool.get()
        try:
            return run_job(
                mission=mission,
                yaml_path=yaml_path,
                seed=seed,
                duration=durations[mission],
                slot=slot,
                domain_base=args.domain_base,
                label=label,
                results_root=results_root,
                color_order=args.color_order,
                balanced=args.balanced,
                partition_prefix=args.partition_prefix,
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
