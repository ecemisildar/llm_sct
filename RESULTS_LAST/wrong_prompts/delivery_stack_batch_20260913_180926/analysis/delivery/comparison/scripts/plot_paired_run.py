#!/usr/bin/env python3
"""Plot a direct comparison of one baseline run and one LLM run."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_result(run: Path) -> tuple[bool, float]:
    with (run / "task_result.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    return row["success"].casefold() == "true", float(row["duration_s"])


def coverage_at(run: Path, cutoff: float) -> float:
    coverage = 0.0
    with (run / "coverage_timeseries.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["time_s"]) > cutoff:
                break
            coverage = float(row["coverage_pct"])
    return coverage


def coverage_series(run: Path, cutoff: float) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    with (run / "coverage_timeseries.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            elapsed = float(row["time_s"])
            if elapsed > cutoff:
                break
            times.append(elapsed)
            values.append(float(row["coverage_pct"]))
    return times, values


def collisions(run: Path) -> int:
    path = run / "bumps_global.csv"
    if not path.is_file():
        return 0
    with path.open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def completion_by_robot(run: Path, task: str) -> dict[str, float]:
    with (run / "task_progress.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if task == "delivery":
        output = {}
        for row in rows:
            delivery_times = []
            for color in ("red", "green", "blue"):
                value = row.get(f"{color}_delivered_s", "")
                if value:
                    delivery_times.append(float(value))
            if delivery_times:
                output[row["robot"]] = max(delivery_times)
        return output
    return {
        row["robot"]: float(row["completion_s"])
        for row in rows
        if row.get("completion_s")
    }


def normalized_event(name: str) -> str:
    name = name.removeprefix("EV_")
    name = name.removeprefix("task_")
    if name in {"search_red", "search_green", "search_blue"}:
        return "search_color"
    if name in {"approach_red", "approach_green", "approach_blue"}:
        return "approach_color"
    return name


def event_percentages(run: Path) -> dict[str, float]:
    counts: Counter[str] = Counter()
    event_paths = list(run.glob("selected_events_robot_*.csv"))
    if not event_paths:
        # Coverage and task logs are stored in S_<id>/run_<id>, but the
        # decision logger stores selected events in the matching
        # S_<id>/robots_<n>/run_<id> directory.
        event_paths = list(
            run.parent.glob(
                f"robots_*/{run.name}/selected_events_robot_*.csv"
            )
        )
    for path in event_paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                event = normalized_event(row.get("selected_event", "").strip())
                if event and event != "none":
                    counts[event] += 1
    total = sum(counts.values())
    return {event: 100.0 * count / total for event, count in counts.items()}


def label_bars(axis, bars, suffix="") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def detect_task(paths: tuple[Path, Path]) -> str:
    matches = {
        task
        for task in ("exploration", "patrolling", "delivery")
        if all(f"results_{task}" in path.parts for path in paths)
    }
    if len(matches) != 1:
        raise ValueError("Could not infer one common task from both run paths; use --task")
    return matches.pop()


def newest_completed_run_for_seed(root: Path, seed: int) -> Path:
    candidates: list[tuple[float, Path]] = []
    pattern = re.compile(rf"^\s*random_seed:\s*{seed}\s*$", re.MULTILINE)
    for status in root.rglob("SAVE_STATUS.txt"):
        text = status.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text) and "Saving OK" in text:
            candidates.append((status.stat().st_mtime, status.parent))
    if not candidates:
        raise FileNotFoundError(
            f"No completed run with seed {seed} was found under {root}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def plot(baseline: Path, llm: Path, output: Path, task: str) -> None:
    baseline_success, baseline_duration = read_result(baseline)
    llm_success, llm_duration = read_result(llm)
    equal_time = min(baseline_duration, llm_duration)
    runs = [baseline, llm]
    labels = ["Baseline", "LLM"]
    colors = ["#4C78A8", "#F58518"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    equal_coverage = [coverage_at(run, equal_time) for run in runs]
    width = 0.36

    if task == "exploration":
        for run, label, color in zip(runs, labels, colors):
            times, values = coverage_series(run, equal_time)
            axes[0, 0].plot(times, values, label=label, color=color, linewidth=2)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Coverage (%)")
        axes[0, 0].set_title("Coverage over time")
        axes[0, 0].legend()

        bars = axes[0, 1].bar(labels, equal_coverage, color=colors)
        label_bars(axes[0, 1], bars, "%")
        axes[0, 1].set_ylabel("Coverage (%)")
        axes[0, 1].set_title(f"Coverage at equal time ({equal_time:.1f} s)")

        bars = axes[1, 0].bar(labels, [collisions(run) for run in runs], color=colors)
        label_bars(axes[1, 0], bars)
        axes[1, 0].set_ylabel("Collision count")
        axes[1, 0].set_title(f"Collisions by {equal_time:.1f} s")
    else:
        bars = axes[0, 0].bar(labels, [baseline_duration, llm_duration], color=colors)
        for bar, duration, success in zip(
            bars,
            (baseline_duration, llm_duration),
            (baseline_success, llm_success),
        ):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                duration,
                f"{duration:.1f} s" if success else f"timeout ({duration:.1f} s)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[0, 0].set_ylabel("Seconds")
        axes[0, 0].set_title("Task completion time / timeout")

        bars = axes[0, 1].bar(labels, equal_coverage, color=colors)
        label_bars(axes[0, 1], bars, "%")
        axes[0, 1].set_ylabel("Coverage (%)")
        axes[0, 1].set_title(f"Coverage at equal time ({equal_time:.1f} s)")

        robot_times = [completion_by_robot(run, task) for run in runs]
        robots = sorted(set(robot_times[0]) | set(robot_times[1]))
        x = list(range(len(robots)))
        axes[1, 0].bar(
            [value - width / 2 for value in x],
            [robot_times[0].get(robot, 0.0) for robot in robots],
            width,
            label="Baseline",
            color=colors[0],
        )
        axes[1, 0].bar(
            [value + width / 2 for value in x],
            [robot_times[1].get(robot, 0.0) for robot in robots],
            width,
            label="LLM",
            color=colors[1],
        )
        axes[1, 0].set_xticks(
            x, [robot.replace("robot_", "Robot ") for robot in robots]
        )
        axes[1, 0].set_ylabel("Time (s)")
        axes[1, 0].set_title(
            "Per-robot completion time"
            if task == "patrolling"
            else "Time of each robot's last delivery"
        )
        axes[1, 0].legend()

    event_values = [event_percentages(run) for run in runs]
    events = sorted(
        set(event_values[0]) | set(event_values[1]),
        key=lambda event: -max(values.get(event, 0.0) for values in event_values),
    )
    x = list(range(len(events)))
    axes[1, 1].bar(
        [value - width / 2 for value in x],
        [event_values[0].get(event, 0.0) for event in events],
        width,
        label="Baseline",
        color=colors[0],
    )
    axes[1, 1].bar(
        [value + width / 2 for value in x],
        [event_values[1].get(event, 0.0) for event in events],
        width,
        label="LLM",
        color=colors[1],
    )
    event_labels = {
        "move_forward": "Move forward",
        "move_backward": "Move back",
        "search_color": "Search color",
        "approach_color": "Approach color",
        "full_rotate": "Full rotate",
        "rotate_clockwise": "Rotate CW",
        "rotate_counterclockwise": "Rotate CCW",
    }
    axes[1, 1].set_xticks(
        x,
        [event_labels.get(event, event.replace("_", " ")) for event in events],
        rotation=25,
        ha="right",
    )
    axes[1, 1].set_ylabel("Selected events (%)")
    axes[1, 1].set_title("Pooled event distribution")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.3)
    collision_text = f"Collisions: baseline {collisions(baseline)}, LLM {collisions(llm)}"
    if task == "exploration":
        subtitle = collision_text
    else:
        success_text = (
            f"Success: baseline {'yes' if baseline_success else 'no'}, "
            f"LLM {'yes' if llm_success else 'no'}"
        )
        subtitle = f"{success_text} — {collision_text}"
    fig.suptitle(f"Matched {task} comparison\n{subtitle}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_run", type=Path, nargs="?")
    parser.add_argument("llm_run", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("auto", "exploration", "patrolling", "delivery"),
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Automatically select the newest completed baseline and LLM runs with this seed",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "new_results",
    )
    args = parser.parse_args()
    if args.seed is not None:
        if args.baseline_run is not None or args.llm_run is not None:
            parser.error("Do not provide run paths together with --seed")
        if args.task == "auto":
            parser.error("--task is required when using --seed")
        task = args.task
        results = args.results_root.expanduser().resolve()
        baseline = newest_completed_run_for_seed(
            results / "baseline" / f"results_{task}", args.seed
        )
        llm = newest_completed_run_for_seed(
            results / "llm" / f"results_{task}", args.seed
        )
        print(f"Baseline run: {baseline}")
        print(f"LLM run:      {llm}")
    else:
        if args.baseline_run is None or args.llm_run is None:
            parser.error("Provide both run paths, or use --task with --seed")
        baseline = args.baseline_run.resolve()
        llm = args.llm_run.resolve()
        task = detect_task((baseline, llm)) if args.task == "auto" else args.task
    plot(baseline, llm, args.output.resolve(), task)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
