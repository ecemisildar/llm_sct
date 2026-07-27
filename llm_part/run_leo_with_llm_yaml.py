#!/usr/bin/env python3
"""Launch a Leo mission with an LLM-generated supervisor YAML."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


LLM_SCT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = LLM_SCT_ROOT.parent.parent
DEFAULT_YAML_DIR = LLM_SCT_ROOT / "automata" / "resulting_automata" / "YAML"
DEFAULT_RESULTS_ROOT = LLM_SCT_ROOT / "results" / "llm"
INSTALL_SETUP = WORKSPACE_ROOT / "install" / "setup.bash"

MISSIONS = {
    "exploration": {
        "package": "leo_exploration",
        "results": "results_exploration",
    },
    "patrolling": {
        "package": "leo_patrolling",
        "results": "results_patrolling",
    },
    "delivery": {
        "package": "leo_delivery",
        "results": "results_delivery",
    },
}


def newest_yaml(directory: Path = DEFAULT_YAML_DIR) -> Path:
    candidates = sorted(directory.glob("S_[0-9]*_[0-9]*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No generated S_<timestamp>.yaml found in {directory}")
    return candidates[-1]


def color_order_for_run(args: argparse.Namespace, yaml_path: Path) -> tuple[str, ...]:
    if args.color_order:
        colors = tuple(
            color.strip().casefold()
            for color in args.color_order.split(",")
            if color.strip()
        )
    else:
        prompt_path = yaml_path.with_suffix(".prompt.txt")
        prompt = (
            prompt_path.read_text(encoding="utf-8")
            if prompt_path.is_file()
            else ""
        )
        colors = tuple(
            match.casefold()
            for match in re.findall(r"\b(red|green|blue)\b", prompt, re.IGNORECASE)
        )
        if not colors:
            colors = ("red", "green", "blue")
    if not colors:
        raise ValueError("Color order must contain at least one color")
    unsupported = sorted(set(colors) - {"red", "green", "blue"})
    if unsupported:
        raise ValueError(
            "Unsupported colors in order: " + ", ".join(unsupported)
        )
    return colors


def build_launch_command(args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    mission = MISSIONS[args.mission]
    yaml_path = (
        Path(args.yaml).expanduser().resolve() if args.yaml else newest_yaml().resolve()
    )
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Supervisor YAML does not exist: {yaml_path}")
    if yaml_path.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError(f"Supervisor must be a YAML file: {yaml_path}")

    results_root = Path(args.results_root).expanduser().resolve()
    mission_results = results_root / str(mission["results"])
    mission_results.mkdir(parents=True, exist_ok=True)

    command = [
        "ros2",
        "launch",
        str(mission["package"]),
        "leo_gz.launch.py",
        f"metadata_yaml_path:={yaml_path}",
        f"results_dir:={mission_results}",
        f"total_robots:={args.total_robots}",
        f"run_duration:={args.run_duration}",
        f"headless:={'true' if args.headless else 'false'}",
        f"random_seed:={args.random_seed}",
    ]
    if args.mission == "patrolling":
        command.append(
            "target_color_order:=" + ",".join(color_order_for_run(args, yaml_path))
        )
    return command, yaml_path, mission_results


def sourced_environment() -> dict[str, str]:
    if not INSTALL_SETUP.is_file():
        raise FileNotFoundError(
            f"ROS workspace setup was not found: {INSTALL_SETUP}. Build the workspace first."
        )
    shell_command = (
        f"source {INSTALL_SETUP} >/dev/null 2>&1 && "
        f"{sys.executable} -c 'import os, json; print(json.dumps(dict(os.environ)))'"
    )
    completed = subprocess.run(
        ["bash", "-c", shell_command],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    environment = json.loads(completed.stdout)
    environment.update({key: value for key, value in os.environ.items() if key not in environment})
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Leo ROS package with an LLM-generated supervisor YAML."
    )
    parser.add_argument("mission", choices=sorted(MISSIONS))
    parser.add_argument(
        "--yaml",
        help="Generated supervisor YAML (default: newest resulting_automata/YAML/S_*.yaml)",
    )
    parser.add_argument("--total-robots", type=int, default=3)
    parser.add_argument("--run-duration", type=float, default=300.0)
    parser.add_argument("--random-seed", default="auto")
    parser.add_argument(
        "--color-order",
        help=(
            "Patrolling order as comma-separated colors. By default it is "
            "derived from the YAML's sibling .prompt.txt file."
        ),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--dry-run", action="store_true", help="Print paths and command without launching"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_robots < 1:
        print("error: --total-robots must be at least 1", file=sys.stderr)
        return 2
    if args.run_duration <= 0:
        print("error: --run-duration must be positive", file=sys.stderr)
        return 2
    try:
        command, yaml_path, results_dir = build_launch_command(args)
        print(f"Mission:    {args.mission}")
        print(f"Supervisor: {yaml_path}")
        print(f"Results:    {results_dir / yaml_path.stem / f'robots_{args.total_robots}'}")
        print("Command:    " + " ".join(command))
        if args.dry_run:
            return 0
        return subprocess.run(command, env=sourced_environment()).returncode
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
