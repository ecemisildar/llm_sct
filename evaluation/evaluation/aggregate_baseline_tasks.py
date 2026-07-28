#!/usr/bin/env python3
"""Aggregate all baseline runs separately for each mission task."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_FIELDS = (
    "coverage_pct",
    "collisions",
    "duration_s",
    "completed_robots",
    "total_robots",
    "completed_targets",
    "total_targets",
    "progress_pct",
)


def last_coverage(path: Path) -> float:
    value = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                value = float(row["coverage_pct"])
            except (KeyError, TypeError, ValueError):
                continue
    if value is None:
        raise ValueError(f"No valid coverage values in {path}")
    return value


def collision_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as stream:
        return sum(1 for row in csv.DictReader(stream) if row)


def task_result(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "success": False,
            "duration_s": 0.0,
            "completed_robots": 0,
            "total_robots": 0,
            "completed_targets": 0,
            "total_targets": 0,
            "progress_pct": 0.0,
        }
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream), {})
    return {
        "success": str(row.get("success", "")).casefold() == "true",
        **{
            field: float(row.get(field, 0) or 0)
            for field in NUMERIC_FIELDS
            if field not in {"coverage_pct", "collisions"}
        },
    }


def load_run(run_dir: Path) -> dict[str, object]:
    coverage_path = run_dir / "coverage_timeseries.csv"
    if not coverage_path.is_file():
        raise ValueError(f"Missing {coverage_path}")
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "coverage_pct": last_coverage(coverage_path),
        "collisions": collision_count(run_dir / "bumps_global.csv"),
        **task_result(run_dir / "task_result.csv"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_selected_events(
    task_dir: Path,
    run_dirs: list[Path],
) -> list[dict[str, object]]:
    per_run_counts = []
    pooled_counts: Counter[str] = Counter()
    for run_dir in run_dirs:
        counts: Counter[str] = Counter()
        for path in sorted(run_dir.glob("selected_events_robot_*.csv")):
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    event = (
                        row.get("selected_event")
                        or row.get("event")
                        or ""
                    ).strip()
                    if event:
                        counts[event] += 1
        per_run_counts.append(counts)
        pooled_counts.update(counts)

    pooled_total = sum(pooled_counts.values())
    rows = []
    for event, count in pooled_counts.most_common():
        percentages = [
            100.0 * run_counts[event] / sum(run_counts.values())
            if run_counts else 0.0
            for run_counts in per_run_counts
        ]
        rows.append(
            {
                "event": event,
                "selected_count": count,
                "pooled_percentage": 100.0 * count / pooled_total,
                "mean_run_percentage": statistics.mean(percentages),
                "std_run_percentage": statistics.pstdev(percentages),
                "min_run_percentage": min(percentages),
                "max_run_percentage": max(percentages),
                "runs_present": sum(
                    run_counts[event] > 0 for run_counts in per_run_counts
                ),
                "total_runs": len(per_run_counts),
            }
        )
    if rows:
        write_csv(task_dir / "aggregate_event_percentages.csv", rows)
        labels = [str(row["event"]).removeprefix("EV_") for row in reversed(rows)]
        values = [float(row["pooled_percentage"]) for row in reversed(rows)]
        fig, axis = plt.subplots(
            figsize=(10, max(4.5, 0.38 * len(rows)))
        )
        axis.barh(labels, values)
        axis.set_xlabel("Selected event percentage (%)")
        axis.set_title(
            f"{task_dir.name.removeprefix('results_').capitalize()} "
            "selected events"
        )
        axis.grid(True, axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(task_dir / "aggregate_event_percentages.png", dpi=160)
        plt.close(fig)
    return rows


def aggregate_task_completion(
    task_dir: Path,
    run_dirs: list[Path],
    run_rows: list[dict[str, object]],
) -> None:
    rows = []
    run_result_by_name = {str(row["run"]): row for row in run_rows}
    for run_dir in run_dirs:
        progress_path = run_dir / "task_progress.csv"
        if not progress_path.is_file():
            continue
        run_result = run_result_by_name[run_dir.name]
        with progress_path.open(newline="") as stream:
            for progress in csv.DictReader(stream):
                complete = str(progress.get("complete", "")).casefold() == "true"
                try:
                    completion_s = float(progress.get("completion_s", ""))
                except (TypeError, ValueError):
                    completion_s = ""
                rows.append(
                    {
                        "run": run_dir.name,
                        "robot": progress.get("robot", ""),
                        "complete": complete,
                        "completion_s": completion_s,
                        "run_success": run_result["success"],
                        "run_duration_s": run_result["duration_s"],
                    }
                )
    if not rows:
        return
    write_csv(task_dir / "aggregate_task_completion.csv", rows)

    run_names = [run_dir.name for run_dir in run_dirs]
    run_positions = {name: index + 1 for index, name in enumerate(run_names)}
    robots = sorted({str(row["robot"]) for row in rows})
    fig, axis = plt.subplots(figsize=(12, 6))
    colors = {
        robot: plt.get_cmap("tab10")(index)
        for index, robot in enumerate(robots)
    }
    for robot in robots:
        completed = [
            row for row in rows
            if row["robot"] == robot and row["complete"]
        ]
        incomplete = [
            row for row in rows
            if row["robot"] == robot and not row["complete"]
        ]
        axis.scatter(
            [run_positions[str(row["run"])] for row in completed],
            [float(row["completion_s"]) for row in completed],
            label=f"{robot} completion",
            color=colors[robot],
            s=55,
        )
        axis.scatter(
            [run_positions[str(row["run"])] for row in incomplete],
            [float(row["run_duration_s"]) for row in incomplete],
            label=f"{robot} incomplete",
            color=colors[robot],
            marker="x",
            s=80,
            linewidths=2,
        )

    successful_runs = [
        row for row in run_rows if bool(row["success"])
    ]
    axis.scatter(
        [run_positions[str(row["run"])] for row in successful_runs],
        [float(row["duration_s"]) for row in successful_runs],
        label="All robots complete",
        color="black",
        marker="D",
        s=45,
    )
    axis.set_xticks(range(1, len(run_names) + 1))
    axis.set_xticklabels(
        [name.removeprefix("run_")[-6:] for name in run_names],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Run (HHMMSS)")
    axis.set_ylabel("Completion time (s)")
    axis.set_title(
        f"{task_dir.name.removeprefix('results_').capitalize()} "
        "task completion by robot"
    )
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(task_dir / "aggregate_task_completion.png", dpi=160)
    plt.close(fig)


def aggregate_task(task_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dirs = sorted(
        path
        for path in task_dir.rglob("run_*")
        if path.is_dir() and (path / "coverage_timeseries.csv").is_file()
    )
    rows = [load_run(path) for path in run_dirs]
    if not rows:
        return [], {}

    task = task_dir.name.removeprefix("results_")
    completion_applicable = task != "exploration"
    summary: dict[str, object] = {
        "task": task,
        "runs": len(rows),
        "successful_runs": (
            sum(bool(row["success"]) for row in rows)
            if completion_applicable else "N/A"
        ),
        "success_rate_pct": (
            100.0 * sum(bool(row["success"]) for row in rows) / len(rows)
            if completion_applicable else "N/A"
        ),
    }
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_std"] = statistics.pstdev(values)
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_max"] = max(values)
    write_csv(task_dir / "aggregate_runs.csv", rows)
    write_csv(task_dir / "aggregate_summary.csv", [summary])
    aggregate_selected_events(task_dir, run_dirs)
    if completion_applicable:
        aggregate_task_completion(task_dir, run_dirs, rows)
    return rows, summary


def save_plot(
    output_path: Path,
    task_rows: dict[str, list[dict[str, object]]],
) -> None:
    labels = list(task_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = (
        ("coverage_pct", "Final coverage (%)"),
        ("collisions", "Robot collisions"),
        ("duration_s", "Run duration (s)"),
    )
    colors = [plt.get_cmap("tab10")(index) for index in range(len(labels))]
    for axis, (field, title) in zip(axes, metrics):
        values_by_task = [
            [float(row[field]) for row in task_rows[label]]
            for label in labels
        ]
        means = [statistics.mean(values) for values in values_by_task]
        deviations = [
            statistics.pstdev(values) for values in values_by_task
        ]
        bars = axis.bar(
            labels,
            means,
            yerr=deviations,
            capsize=6,
            color=colors,
            alpha=0.85,
        )
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)
        for bar, mean in zip(bars, means):
            axis.annotate(
                f"{mean:.2f}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate current baseline result folders by task."
    )
    parser.add_argument(
        "--baseline-root",
        default=str(
            Path(__file__).resolve().parents[2] / "results" / "baseline"
        ),
    )
    args = parser.parse_args()
    baseline_root = Path(args.baseline_root).expanduser().resolve()

    all_summaries = []
    task_rows = {}
    for task_dir in sorted(baseline_root.glob("results_*")):
        rows, summary = aggregate_task(task_dir)
        task = task_dir.name.removeprefix("results_")
        if not rows:
            print(f"{task}: no runs found")
            continue
        task_rows[task] = rows
        all_summaries.append(summary)
        print(
            f"{task}: {len(rows)} runs, "
            f"coverage={summary['coverage_pct_mean']:.3f}%, "
            f"collisions={summary['collisions_mean']:.3f}, "
            f"duration={summary['duration_s_mean']:.3f}s"
        )

    if not all_summaries:
        raise SystemExit(f"No baseline runs found under {baseline_root}")
    write_csv(baseline_root / "aggregate_summary.csv", all_summaries)
    save_plot(baseline_root / "aggregate_summary.png", task_rows)
    print(f"Saved aggregate outputs under {baseline_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
