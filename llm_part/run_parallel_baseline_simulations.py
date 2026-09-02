#!/usr/bin/env python3
"""Run the three baseline missions in isolated parallel Gazebo workers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "new_results" / "baseline"
MISSIONS = {
    "exploration": ("leo_exploration", 300.0),
    "patrolling": ("leo_patrolling", 600.0),
    "delivery": ("leo_delivery", 600.0),
}


def active_processes_in_group(group_id: int) -> list[int]:
    active = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text().split(")", 1)[1].split()
            state, process_group = fields[0], int(fields[2])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if process_group == group_id and state != "Z":
            active.append(int(stat_path.parent.name))
    return active


def stop_process_group(group_id: int) -> None:
    for sig, timeout in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not active_processes_in_group(group_id):
                return
            time.sleep(0.1)
    raise RuntimeError(f"simulation process group {group_id} did not terminate")


def run_mission(
    mission: str,
    seeds: tuple[int, ...],
    domain_id: int,
    duration: float,
) -> tuple[str, int]:
    package, _ = MISSIONS[mission]
    environment = os.environ.copy()
    partition = f"baseline_{mission}_{domain_id}"
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    environment["IGN_PARTITION"] = partition
    environment["GZ_PARTITION"] = partition
    result_dir = RESULTS_ROOT / f"results_{mission}"
    log_dir = RESULTS_ROOT / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ros_log_dir = log_dir / f"ros_{mission}"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    failures = 0

    for index, seed in enumerate(seeds, 1):
        label = f"[{mission} {index}/{len(seeds)} seed={seed}]"
        command = [
            "ros2", "launch", package, "leo_gz.launch.py",
            "total_robots:=3",
            f"run_duration:={duration}",
            f"random_seed:={seed}",
            "headless:=true",
            "overhead_camera:=false",
            "record_video:=false",
            f"results_dir:={result_dir}",
        ]
        log_path = log_dir / f"{mission}_seed_{seed}.log"
        print(f"{label} starting; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait()
            finally:
                stop_process_group(process.pid)
        if return_code:
            failures += 1
            print(f"{label} FAILED with exit code {return_code}", flush=True)
        else:
            print(f"{label} finished", flush=True)
    return mission, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1001,1002,1003,1004,1005")
    parser.add_argument(
        "--missions",
        default=",".join(MISSIONS),
        help="Comma-separated missions to run: exploration, patrolling, delivery",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--domain-base", type=int, default=41)
    parser.add_argument("--exploration-duration", type=float, default=300.0)
    parser.add_argument("--patrolling-duration", type=float, default=600.0)
    parser.add_argument("--delivery-duration", type=float, default=600.0)
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if len(seeds) != 5 or len(set(seeds)) != 5:
        parser.error("--seeds must contain exactly five unique integers")
    missions = tuple(
        value.strip() for value in args.missions.split(",") if value.strip()
    )
    if not missions or any(mission not in MISSIONS for mission in missions):
        parser.error(
            "--missions must contain one or more of: " + ", ".join(MISSIONS)
        )
    if len(set(missions)) != len(missions):
        parser.error("--missions must not contain duplicates")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    durations = {
        mission: getattr(args, f"{mission}_duration") for mission in MISSIONS
    }
    if any(duration <= 0 for duration in durations.values()):
        parser.error("durations must be positive")

    failures = 0
    jobs = [
        (mission, seed)
        for mission in missions
        for seed in seeds
    ]
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as executor:
        futures = [
            executor.submit(
                run_mission,
                mission,
                (seed,),
                args.domain_base + index,
                durations[mission],
            )
            for index, (mission, seed) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            mission, mission_failures = future.result()
            failures += mission_failures
            print(f"[{mission}] worker complete; failures={mission_failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
