#!/usr/bin/env python3
"""Plot four aggregate panels for matched baseline/LLM seed pairs."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paired_run import (
    collisions,
    completion_by_robot,
    coverage_at,
    event_percentages,
    newest_completed_run_for_seed,
    read_result,
)


def paired_summary(axis, baseline, llm, ylabel, title):
    axis.boxplot(
        [baseline, llm],
        labels=["Baseline", "LLM"],
        showmeans=True,
        widths=0.45,
    )
    for baseline_value, llm_value in zip(baseline, llm):
        axis.plot([1, 2], [baseline_value, llm_value], color="0.72", alpha=0.7)
        axis.scatter(1, baseline_value, color="#4C78A8", zorder=3)
        axis.scatter(2, llm_value, color="#F58518", zorder=3)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("exploration", "patrolling", "delivery"), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "new_results",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = args.results_root.expanduser().resolve()
    pairs = []
    for seed in args.seeds:
        baseline = newest_completed_run_for_seed(
            results / "baseline" / f"results_{args.task}", seed
        )
        llm = newest_completed_run_for_seed(
            results / "llm" / f"results_{args.task}", seed
        )
        pairs.append((seed, baseline, llm))

    durations = {"baseline": [], "llm": []}
    equal_coverage = {"baseline": [], "llm": []}
    success = {"baseline": 0, "llm": 0}
    collision_totals = {"baseline": 0, "llm": 0}
    collision_runs = {"baseline": [], "llm": []}
    event_runs = {"baseline": [], "llm": []}
    robot_times = {"baseline": {}, "llm": {}}

    for seed, baseline, llm in pairs:
        run_paths = {"baseline": baseline, "llm": llm}
        results_by_condition = {
            condition: read_result(path) for condition, path in run_paths.items()
        }
        cutoff = min(value[1] for value in results_by_condition.values())
        for condition, path in run_paths.items():
            succeeded, duration = results_by_condition[condition]
            durations[condition].append(duration)
            equal_coverage[condition].append(coverage_at(path, cutoff))
            success[condition] += int(succeeded)
            run_collisions = collisions(path)
            collision_totals[condition] += run_collisions
            collision_runs[condition].append(run_collisions)
            event_runs[condition].append(event_percentages(path))
            robot_times[condition][seed] = completion_by_robot(path, args.task)

    if args.task == "exploration":
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        paired_summary(
            axes[0, 0],
            equal_coverage["baseline"],
            equal_coverage["llm"],
            "Coverage (%)",
            "Coverage at 300 seconds",
        )

        sample_times = list(range(0, 301, 10))
        for condition, pair_index, color, label in (
            ("baseline", 1, "#4C78A8", "Baseline"),
            ("llm", 2, "#F58518", "LLM"),
        ):
            run_paths = [pair[pair_index] for pair in pairs]
            samples = [
                [coverage_at(path, time) for time in sample_times]
                for path in run_paths
            ]
            means = [statistics.mean(run[index] for run in samples) for index in range(len(sample_times))]
            stds = [statistics.pstdev(run[index] for run in samples) for index in range(len(sample_times))]
            axes[0, 1].plot(sample_times, means, color=color, label=label, linewidth=2)
            axes[0, 1].fill_between(
                sample_times,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=color,
                alpha=0.18,
            )
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Coverage (%)")
        axes[0, 1].set_title("Mean coverage over time (±1 SD)")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="both", linestyle="--", alpha=0.3)

        paired_summary(
            axes[1, 0],
            collision_runs["baseline"],
            collision_runs["llm"],
            "Collision count",
            "Collisions by 300 seconds",
        )

        events = sorted(
            {
                event
                for condition in event_runs.values()
                for run in condition
                for event in run
            },
            key=lambda event: -max(
                statistics.mean(run.get(event, 0.0) for run in event_runs[condition])
                for condition in ("baseline", "llm")
            ),
        )
        event_labels = {
            "move_forward": "Move forward",
            "move_backward": "Move back",
            "full_rotate": "Full rotate",
            "rotate_clockwise": "Rotate CW",
            "rotate_counterclockwise": "Rotate CCW",
        }
        positions = list(range(len(events)))
        width = 0.38
        for offset, condition, color, label in (
            (-width / 2, "baseline", "#4C78A8", "Baseline"),
            (width / 2, "llm", "#F58518", "LLM"),
        ):
            means = [
                statistics.mean(run.get(event, 0.0) for run in event_runs[condition])
                for event in events
            ]
            errors = [
                statistics.pstdev(run.get(event, 0.0) for run in event_runs[condition])
                for event in events
            ]
            axes[1, 1].bar(
                [position + offset for position in positions],
                means,
                width,
                yerr=errors,
                capsize=3,
                label=label,
                color=color,
            )
        axes[1, 1].set_xticks(
            positions,
            [event_labels.get(event, event.replace("_", " ")) for event in events],
            rotation=25,
            ha="right",
        )
        axes[1, 1].set_ylabel("Mean selected events (%)")
        axes[1, 1].set_title("Event distribution across all seeds")
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

        baseline_mean = statistics.mean(equal_coverage["baseline"])
        llm_mean = statistics.mean(equal_coverage["llm"])
        figure.suptitle(
            f"Exploration: Baseline vs LLM, matched seeds {args.seeds[0]}–{args.seeds[-1]}\n"
            f"Mean coverage: baseline {baseline_mean:.1f}%, LLM {llm_mean:.1f}% — "
            f"Collisions: baseline {collision_totals['baseline']}, LLM {collision_totals['llm']}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {args.output.resolve()}")
        return 0

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    paired_summary(
        axes[0, 0],
        durations["baseline"], durations["llm"],
        "Seconds", "Task completion time / timeout",
    )
    paired_summary(
        axes[0, 1],
        equal_coverage["baseline"], equal_coverage["llm"],
        "Coverage (%)", "Coverage at paired equal time",
    )

    robots = sorted(
        {
            robot
            for condition in robot_times.values()
            for by_seed in condition.values()
            for robot in by_seed
        }
    )
    robot_values = {"baseline": [], "llm": []}
    for condition in ("baseline", "llm"):
        for index, seed in enumerate(args.seeds):
            for robot in robots:
                robot_values[condition].append(
                    robot_times[condition][seed].get(
                        robot, durations[condition][index]
                    )
                )
    paired_summary(
        axes[1, 0],
        robot_values["baseline"],
        robot_values["llm"],
        "Seconds",
        "Per-robot completion time / timeout",
    )

    events = sorted(
        {
            event
            for condition in event_runs.values()
            for run in condition
            for event in run
        },
        key=lambda event: -max(
            statistics.mean(run.get(event, 0.0) for run in event_runs[condition])
            for condition in ("baseline", "llm")
        ),
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
    positions = list(range(len(events)))
    width = 0.38
    for offset, condition, color in (
        (-width / 2, "baseline", "#4C78A8"),
        (width / 2, "llm", "#F58518"),
    ):
        means = [
            statistics.mean(run.get(event, 0.0) for run in event_runs[condition])
            for event in events
        ]
        errors = [
            statistics.pstdev(run.get(event, 0.0) for run in event_runs[condition])
            for event in events
        ]
        axes[1, 1].bar(
            [position + offset for position in positions],
            means,
            width,
            yerr=errors,
            capsize=3,
            label=condition.capitalize(),
            color=color,
        )
    axes[1, 1].set_xticks(
        positions,
        [event_labels.get(event, event.replace("_", " ")) for event in events],
        rotation=25,
        ha="right",
    )
    axes[1, 1].set_ylabel("Mean selected events (%)")
    axes[1, 1].set_title("Event distribution across all seeds")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

    figure.suptitle(
        f"{args.task.capitalize()} matched seeds {args.seeds[0]}–{args.seeds[-1]}\n"
        f"Successes: baseline {success['baseline']}/{len(args.seeds)}, "
        f"LLM {success['llm']}/{len(args.seeds)} — "
        f"Collisions: baseline {collision_totals['baseline']}, "
        f"LLM {collision_totals['llm']}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
