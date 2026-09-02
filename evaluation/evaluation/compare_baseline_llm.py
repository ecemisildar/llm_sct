#!/usr/bin/env python3
"""Compare baseline and LLM mission runs and generate publication-ready plots."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "new_results"


def run_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("task_result.csv")})


def seed_from_status(run_dir: Path) -> int:
    text = (run_dir / "SAVE_STATUS.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    match = re.search(r"^\s*random_seed:\s*(\d+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"No random_seed in {run_dir / 'SAVE_STATUS.txt'}")
    return int(match.group(1))


def yaml_name(run_dir: Path, condition: str) -> str:
    if condition == "baseline":
        return run_dir.parents[1].name
    # .../<yaml>/robots_3/run_<timestamp>
    return run_dir.parents[1].name


def coverage_at(run_dir: Path, cutoff: float | None) -> float:
    values: list[tuple[float, float]] = []
    with (run_dir / "coverage_timeseries.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                elapsed = float(row["time_s"])
                coverage = float(row["coverage_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            if cutoff is None or elapsed <= cutoff:
                values.append((elapsed, coverage))
    if not values:
        raise ValueError(f"No coverage sample at cutoff in {run_dir}")
    return values[-1][1]


def collision_count(run_dir: Path, cutoff: float | None) -> int:
    path = run_dir / "bumps_global.csv"
    if not path.is_file():
        return 0
    count = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if cutoff is None:
                count += 1
                continue
            try:
                elapsed = float(row.get("stamp_sec", "")) + (
                    float(row.get("stamp_nsec", "0")) * 1e-9
                )
            except (TypeError, ValueError):
                continue
            if elapsed <= cutoff:
                count += 1
    return count


def task_result(run_dir: Path) -> tuple[bool, float]:
    with (run_dir / "task_result.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream), {})
    return (
        str(row.get("success", "")).casefold() == "true",
        float(row.get("duration_s", 0) or 0),
    )


def collect(
    mission: str,
    condition: str,
    root: Path,
    cutoff: float | None,
) -> list[dict[str, object]]:
    rows = []
    for run_dir in run_dirs(root):
        success, duration = task_result(run_dir)
        rows.append(
            {
                "mission": mission,
                "condition": condition,
                "yaml": yaml_name(run_dir, condition),
                "seed": seed_from_status(run_dir),
                "run": run_dir.name,
                "path": str(run_dir),
                "cutoff_s": "" if cutoff is None else cutoff,
                "coverage_pct": coverage_at(run_dir, cutoff),
                "collisions": collision_count(run_dir, cutoff),
                "success": success,
                "duration_s": duration,
            }
        )
    return rows


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mission"]), str(row["condition"]))].append(row)
    for (mission, condition), group in sorted(groups.items()):
        coverage = [float(row["coverage_pct"]) for row in group]
        collisions = [float(row["collisions"]) for row in group]
        durations = [float(row["duration_s"]) for row in group]
        success_count = sum(bool(row["success"]) for row in group)
        cov = numeric_summary(coverage)
        col = numeric_summary(collisions)
        dur = numeric_summary(durations)
        output.append(
            {
                "mission": mission,
                "condition": condition,
                "runs": len(group),
                "yaml_count": len({str(row["yaml"]) for row in group}),
                "success_count": "" if mission == "exploration" else success_count,
                "success_rate_pct": (
                    "" if mission == "exploration"
                    else 100.0 * success_count / len(group)
                ),
                **{f"coverage_pct_{key}": value for key, value in cov.items()},
                **{f"collisions_{key}": value for key, value in col.items()},
                **{
                    f"duration_s_{key}": "" if mission == "exploration" else value
                    for key, value in dur.items()
                },
            }
        )
    return output


def per_yaml_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["condition"] != "llm":
            continue
        groups[(str(row["mission"]), str(row["yaml"]))].append(row)
    for (mission, yaml), group in sorted(groups.items()):
        output.append(
            {
                "mission": mission,
                "yaml": yaml,
                "runs": len(group),
                "success_rate_pct": (
                    "" if mission == "exploration"
                    else 100.0 * sum(bool(row["success"]) for row in group) / len(group)
                ),
                "coverage_pct_mean": statistics.mean(
                    float(row["coverage_pct"]) for row in group
                ),
                "collisions_mean": statistics.mean(
                    float(row["collisions"]) for row in group
                ),
                "duration_s_mean": (
                    "" if mission == "exploration"
                    else statistics.mean(float(row["duration_s"]) for row in group)
                ),
            }
        )
    return output


def normalized_event(name: str) -> str:
    """Map raw and task-prefixed forms of the same motion to one category."""
    name = name.removeprefix("EV_")
    if name.startswith("task_"):
        name = name.removeprefix("task_")
    return name


def selected_event_counts(
    run_dir: Path, cutoff: float | None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted(run_dir.glob("selected_events_robot_*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    elapsed = float(row.get("elapsed_s", ""))
                except (TypeError, ValueError):
                    continue
                if cutoff is not None and elapsed > cutoff:
                    continue
                event = normalized_event(str(row.get("selected_event", "")).strip())
                if event and event != "none":
                    counts[event] += 1
    return counts


def event_percentages(
    rows: list[dict[str, object]], exploration_cutoff: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_run = []
    run_counts: dict[tuple[str, str], Counter[str]] = {}
    for row in rows:
        mission = str(row["mission"])
        counts = selected_event_counts(
            Path(str(row["path"])),
            exploration_cutoff if mission == "exploration" else None,
        )
        total = sum(counts.values())
        key = (mission, str(row["condition"]))
        run_counts.setdefault(key, Counter()).update(counts)
        for event, count in sorted(counts.items()):
            per_run.append(
                {
                    "mission": mission,
                    "condition": row["condition"],
                    "yaml": row["yaml"],
                    "seed": row["seed"],
                    "run": row["run"],
                    "event": event,
                    "selected_count": count,
                    "run_percentage": 100.0 * count / total if total else 0.0,
                }
            )

    summary = []
    for (mission, condition), pooled in sorted(run_counts.items()):
        group_rows = [
            row for row in rows
            if row["mission"] == mission and row["condition"] == condition
        ]
        events = sorted(pooled)
        pooled_total = sum(pooled.values())
        percentages_by_run: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for item in per_run:
            if item["mission"] == mission and item["condition"] == condition:
                percentages_by_run[(str(item["yaml"]), str(item["run"]))][
                    str(item["event"])
                ] = float(item["run_percentage"])
        for event in events:
            percentages = [
                percentages_by_run[(str(row["yaml"]), str(row["run"]))].get(event, 0.0)
                for row in group_rows
            ]
            summary.append(
                {
                    "mission": mission,
                    "condition": condition,
                    "event": event,
                    "runs": len(group_rows),
                    "selected_count": pooled[event],
                    "pooled_percentage": 100.0 * pooled[event] / pooled_total,
                    "mean_run_percentage": statistics.mean(percentages),
                    "std_run_percentage": statistics.pstdev(percentages),
                    "min_run_percentage": min(percentages),
                    "max_run_percentage": max(percentages),
                }
            )
    return per_run, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, mission: str, field: str) -> list[list[float]]:
    return [
        [
            float(row[field])
            for row in rows
            if row["mission"] == mission and row["condition"] == condition
        ]
        for condition in ("baseline", "llm")
    ]


def boxplot(axis, values, ylabel: str, title: str) -> None:
    axis.boxplot(values, labels=["Baseline", "LLM"], showmeans=True)
    for index, group in enumerate(values, start=1):
        # Deterministic horizontal offsets keep all observations visible.
        offsets = [((item % 9) - 4) * 0.012 for item in range(len(group))]
        axis.scatter(
            [index + offset for offset in offsets],
            group,
            alpha=0.55,
            s=18,
            zorder=3,
        )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.35)


def plot_exploration(rows: list[dict[str, object]], output: Path, cutoff: float):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    boxplot(
        axes[0],
        grouped(rows, "exploration", "coverage_pct"),
        "Coverage (%)",
        f"Coverage at {cutoff:g} s",
    )
    boxplot(
        axes[1],
        grouped(rows, "exploration", "collisions"),
        "Collisions",
        f"Collisions by {cutoff:g} s",
    )
    fig.suptitle("Exploration: baseline vs LLM")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_patrolling(rows: list[dict[str, object]], output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    boxplot(
        axes[0, 0], grouped(rows, "patrolling", "duration_s"),
        "Completion time (s)", "Task completion time",
    )
    boxplot(
        axes[0, 1], grouped(rows, "patrolling", "coverage_pct"),
        "Coverage at completion (%)", "Coverage when task ended",
    )
    boxplot(
        axes[1, 0], grouped(rows, "patrolling", "collisions"),
        "Collisions", "Collisions before completion",
    )
    success = []
    for condition in ("baseline", "llm"):
        group = [
            row for row in rows
            if row["mission"] == "patrolling" and row["condition"] == condition
        ]
        success.append(100.0 * sum(bool(row["success"]) for row in group) / len(group))
    axes[1, 1].bar(["Baseline", "LLM"], success)
    axes[1, 1].set_ylabel("Success rate (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("Task success")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Patrolling: baseline vs LLM")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_event_percentages(
    summary: list[dict[str, object]], mission: str, output: Path, title: str
) -> None:
    mission_rows = [row for row in summary if row["mission"] == mission]
    events = sorted(
        {str(row["event"]) for row in mission_rows},
        key=lambda event: -max(
            float(row["mean_run_percentage"])
            for row in mission_rows if row["event"] == event
        ),
    )
    values = {}
    errors = {}
    for condition in ("baseline", "llm"):
        lookup = {
            str(row["event"]): row
            for row in mission_rows if row["condition"] == condition
        }
        values[condition] = [
            float(lookup[event]["mean_run_percentage"]) if event in lookup else 0.0
            for event in events
        ]
        errors[condition] = [
            float(lookup[event]["std_run_percentage"]) if event in lookup else 0.0
            for event in events
        ]

    positions = list(range(len(events)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(9, 1.2 * len(events)), 5.2))
    axis.bar(
        [position - width / 2 for position in positions],
        values["baseline"], width, yerr=errors["baseline"],
        capsize=3, label="Baseline",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        values["llm"], width, yerr=errors["llm"],
        capsize=3, label="LLM",
    )
    axis.set_xticks(positions, [event.replace("_", "\n") for event in events])
    axis.set_ylabel("Mean selected-event percentage per run (%)")
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and LLM exploration/patrolling results."
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--exploration-cutoff", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS / "comparison"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.exploration_cutoff <= 0:
        raise SystemExit("--exploration-cutoff must be positive")
    results = Path(args.results_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for mission, cutoff in (
        ("exploration", args.exploration_cutoff),
        ("patrolling", None),
    ):
        for condition in ("baseline", "llm"):
            root = results / condition / f"results_{mission}"
            mission_rows = collect(mission, condition, root, cutoff)
            if not mission_rows:
                raise SystemExit(f"No completed runs found under {root}")
            rows.extend(mission_rows)

    summary = summarize(rows)
    per_run_events, event_summary = event_percentages(
        rows, args.exploration_cutoff
    )
    write_csv(output / "per_run_metrics.csv", rows)
    write_csv(output / "comparison_summary.csv", summary)
    write_csv(output / "per_yaml_metrics.csv", per_yaml_summary(rows))
    write_csv(output / "per_run_event_percentages.csv", per_run_events)
    write_csv(output / "event_percentage_summary.csv", event_summary)
    plot_exploration(
        rows,
        output / f"exploration_comparison_{args.exploration_cutoff:g}s.png",
        args.exploration_cutoff,
    )
    plot_patrolling(rows, output / "patrolling_comparison.png")
    plot_event_percentages(
        event_summary,
        "exploration",
        output / f"exploration_event_percentages_{args.exploration_cutoff:g}s.png",
        f"Exploration selected events at {args.exploration_cutoff:g} s",
    )
    plot_event_percentages(
        event_summary,
        "patrolling",
        output / "patrolling_event_percentages.png",
        "Patrolling selected events until task completion",
    )

    for row in summary:
        message = (
            f"{row['mission']} {row['condition']}: n={row['runs']}, "
            f"coverage={row['coverage_pct_mean']:.3f}%, "
            f"collisions={row['collisions_mean']:.3f}"
        )
        if row["mission"] == "patrolling":
            message += (
                f", success={row['success_rate_pct']:.1f}%, "
                f"duration={row['duration_s_mean']:.3f}s"
            )
        print(message)
    print(f"Saved comparison outputs under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
