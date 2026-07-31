#!/usr/bin/env python3
"""Evaluate exploration supervisor YAMLs with an identical list of seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from run_pipeline import (
    build_latest_exploration_feedback,
    latest_completed_exploration_run,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_leo_with_llm_yaml.py"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "llm"


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one seed")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        action="append",
        required=True,
        help="Exploration supervisor YAML to evaluate; repeat for every candidate",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("12345,23456,34567"),
        help="Identical comma-separated seeds used for every candidate",
    )
    parser.add_argument("--total-robots", type=int, default=3)
    parser.add_argument("--run-duration", type=float, default=300.0)
    parser.add_argument("--forward-probability", type=float, default=0.90)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Gazebo instead of running headless",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_robots < 1 or args.run_duration <= 0:
        raise SystemExit("total robots and run duration must be positive")
    if not 0.0 <= args.forward_probability <= 1.0:
        raise SystemExit("forward probability must be between 0 and 1")

    candidates = [Path(value).expanduser().resolve() for value in args.yaml]
    exploration_results = (
        Path(args.results_root).expanduser().resolve() / "results_exploration"
    )
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise SystemExit("Missing YAML files: " + ", ".join(map(str, missing)))

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(RUNNER),
                "exploration",
                "--yaml",
                str(candidate),
                "--total-robots",
                str(args.total_robots),
                "--run-duration",
                str(args.run_duration),
                "--random-seed",
                str(seed),
                "--forward-probability",
                str(args.forward_probability),
                "--results-root",
                str(Path(args.results_root).expanduser().resolve()),
            ]
            command.append("--no-headless" if args.headed else "--headless")
            print(f"Evaluating {candidate.stem} with seed {seed}", flush=True)
            previous_run = latest_completed_exploration_run(exploration_results)
            completed = subprocess.run(command, check=False)
            current_run = latest_completed_exploration_run(exploration_results)
            feedback = (
                build_latest_exploration_feedback(exploration_results)
                if current_run is not None and current_run != previous_run
                else None
            )
            evaluation = feedback.get("evaluation", {}) if feedback else {}
            source = feedback.get("source", {}) if feedback else {}
            rows.append(
                {
                    "candidate": candidate.stem,
                    "seed": seed,
                    "return_code": completed.returncode,
                    "coverage_percent": evaluation.get("total_coverage_percent"),
                    "duration_seconds": evaluation.get("run_duration_seconds"),
                    "run_id": source.get("run_id"),
                }
            )
            if completed.returncode:
                print(f"Run failed with code {completed.returncode}", file=sys.stderr)

    summary: dict[str, object] = {
        "seeds": list(args.seeds),
        "forward_probability": args.forward_probability,
        "runs": rows,
    }
    for candidate in candidates:
        coverages = [
            float(row["coverage_percent"])
            for row in rows
            if row["candidate"] == candidate.stem
            and row["return_code"] == 0
            and row["coverage_percent"] is not None
        ]
        if coverages:
            summary.setdefault("candidate_summary", {})[candidate.stem] = {
                "mean_coverage_percent": sum(coverages) / len(coverages),
                "minimum_coverage_percent": min(coverages),
                "maximum_coverage_percent": max(coverages),
                "completed_runs": len(coverages),
            }

    output = (
        exploration_results
        / f"same_seed_comparison_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison: {output}")
    return 0 if all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
