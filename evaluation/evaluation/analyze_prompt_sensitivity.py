#!/usr/bin/env python3
"""Aggregate prompt, generation, and seed sensitivity for one task."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paired_run import (
    collisions,
    coverage_at,
    event_percentages,
    newest_completed_run_for_seed,
    read_result,
)


def seed_for_run(run: Path) -> int | None:
    text = (run / "SAVE_STATUS.txt").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*random_seed:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    return statistics.mean(values), statistics.pstdev(values)


def prompt_texts_for(
    yaml_root: Path, prompts: list[int], generations: list[int]
) -> dict[int, str]:
    """Read and validate the user prompt saved beside each generated YAML."""
    result: dict[int, str] = {}
    for prompt in prompts:
        texts = {
            path.read_text(encoding="utf-8").strip()
            for generation in generations
            for path in (yaml_root / f"prompt_{prompt}" / f"generation_{generation}").glob(
                "S_*.prompt.txt"
            )
        }
        texts.discard("")
        if not texts:
            raise SystemExit(f"No saved prompt text found for prompt {prompt}")
        if len(texts) != 1:
            raise SystemExit(
                f"Prompt {prompt} has multiple texts in the selected generations: "
                f"{sorted(texts)!r}"
            )
        result[prompt] = texts.pop()
    return result


def summarize(group: list[dict[str, object]], prefix: dict[str, object]) -> dict[str, object]:
    durations = [float(row["duration_s"]) for row in group]
    successful = [float(row["duration_s"]) for row in group if bool(row["success"])]
    coverage = [float(row["coverage_pct"]) for row in group]
    successful_coverage = [
        float(row["coverage_pct"]) for row in group if bool(row["success"])
    ]
    collision_values = [float(row["collisions"]) for row in group]
    duration_mean, duration_sd = mean_sd(durations)
    successful_mean, successful_sd = mean_sd(successful)
    coverage_mean, coverage_sd = mean_sd(coverage)
    successful_coverage_mean, successful_coverage_sd = mean_sd(successful_coverage)
    collision_mean, collision_sd = mean_sd(collision_values)
    successes = sum(bool(row["success"]) for row in group)
    return {
        **prefix,
        "runs": len(group),
        "successes": successes,
        "success_rate_pct": 100.0 * successes / len(group),
        "duration_all_mean_s": duration_mean,
        "duration_all_sd_s": duration_sd,
        "duration_success_mean_s": successful_mean,
        "duration_success_sd_s": successful_sd,
        "coverage_mean_pct": coverage_mean,
        "coverage_sd_pct": coverage_sd,
        "coverage_success_mean_pct": successful_coverage_mean,
        "coverage_success_sd_pct": successful_coverage_sd,
        "collisions_total": int(sum(collision_values)),
        "collisions_mean": collision_mean,
        "collisions_sd": collision_sd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("exploration", "patrolling", "delivery"), required=True
    )
    parser.add_argument("--prompts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--generations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    yaml_root = root / "automata" / "resulting_automata" / "YAML" / args.task
    prompt_texts = prompt_texts_for(yaml_root, args.prompts, args.generations)
    write_csv(
        output / f"{args.task}_prompts.csv",
        [
            {"prompt": prompt, "prompt_text": prompt_texts[prompt]}
            for prompt in args.prompts
        ],
    )
    yaml_index: dict[str, tuple[int, int, Path]] = {}
    for yaml_path in yaml_root.glob("prompt_*/generation_*/S_*.yaml"):
        prompt = int(yaml_path.parents[1].name.removeprefix("prompt_"))
        generation = int(yaml_path.parent.name.removeprefix("generation_"))
        if prompt in args.prompts and generation in args.generations:
            yaml_index[yaml_path.stem] = (prompt, generation, yaml_path)

    newest: dict[tuple[str, int], Path] = {}
    result_root = root / "new_results" / "llm" / f"results_{args.task}"
    for result_file in result_root.glob("S_*/robots_*/run_*/task_result.csv"):
        run = result_file.parent
        status = run / "SAVE_STATUS.txt"
        if not status.is_file() or "Saving OK" not in status.read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        supervisor = run.parents[1].name
        seed = seed_for_run(run)
        if supervisor not in yaml_index or seed not in args.seeds:
            continue
        key = (supervisor, int(seed))
        if key not in newest or status.stat().st_mtime > (newest[key] / "SAVE_STATUS.txt").stat().st_mtime:
            newest[key] = run

    expected = len(args.prompts) * len(args.generations) * len(args.seeds)
    if len(newest) != expected:
        raise SystemExit(f"Found {len(newest)} completed cells; expected {expected}")

    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for (supervisor, seed), run in sorted(newest.items()):
        prompt, generation, yaml_path = yaml_index[supervisor]
        success, duration = read_result(run)
        row = {
            "task": args.task,
            "prompt": prompt,
            "prompt_text": prompt_texts[prompt],
            "generation": generation,
            "seed": seed,
            "supervisor": supervisor,
            "success": success,
            "duration_s": duration,
            "coverage_pct": coverage_at(run, duration),
            "collisions": collisions(run),
            "yaml": str(yaml_path),
            "run": str(run),
        }
        rows.append(row)
        for event, percentage in event_percentages(run).items():
            event_rows.append(
                {
                    "prompt": prompt,
                    "generation": generation,
                    "seed": seed,
                    "supervisor": supervisor,
                    "event": event,
                    "percentage": percentage,
                }
            )
    rows.sort(key=lambda row: (int(row["prompt"]), int(row["generation"]), int(row["seed"])))
    write_csv(output / f"{args.task}_prompt_sensitivity_runs.csv", rows)
    write_csv(output / f"{args.task}_prompt_sensitivity_events.csv", event_rows)

    generation_summary = []
    prompt_summary = []
    for prompt in args.prompts:
        prompt_group = [row for row in rows if row["prompt"] == prompt]
        prompt_summary.append(
            summarize(
                prompt_group,
                {"prompt": prompt, "prompt_text": prompt_texts[prompt]},
            )
        )
        for generation in args.generations:
            group = [
                row for row in prompt_group if row["generation"] == generation
            ]
            generation_summary.append(
                summarize(
                    group,
                    {
                        "prompt": prompt,
                        "prompt_text": prompt_texts[prompt],
                        "generation": generation,
                    },
                )
            )
    write_csv(output / f"{args.task}_prompt_summary.csv", prompt_summary)
    write_csv(output / f"{args.task}_generation_summary.csv", generation_summary)

    baseline_rows = []
    for seed in args.seeds:
        run = newest_completed_run_for_seed(
            root / "new_results" / "baseline" / f"results_{args.task}", seed
        )
        success, duration = read_result(run)
        baseline_rows.append(
            {
                "seed": seed,
                "success": success,
                "duration_s": duration,
                "coverage_pct": coverage_at(run, duration),
                "collisions": collisions(run),
                "run": str(run),
            }
        )
    baseline_summary = summarize(baseline_rows, {"prompt": "baseline"})
    write_csv(output / f"{args.task}_baseline_summary.csv", [baseline_summary])

    labels = ["Baseline", *[f"Prompt {prompt}" for prompt in args.prompts]]
    groups = [
        baseline_rows,
        *[[row for row in rows if row["prompt"] == prompt] for prompt in args.prompts],
    ]
    if args.task == "exploration":
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for axis, field, title, ylabel in (
            (axes[0, 0], "coverage_pct", "Coverage distribution", "Coverage (%)"),
            (axes[0, 1], "collisions", "Collisions", "Collision count"),
        ):
            values = [[float(row[field]) for row in group] for group in groups]
            axis.boxplot(values, labels=labels, showmeans=True)
            for index, group_values in enumerate(values, 1):
                axis.scatter([index] * len(group_values), group_values, s=15, alpha=0.45)
            axis.set_ylabel(ylabel)
            axis.set_title(title)

        baseline_coverage = {
            int(row["seed"]): float(row["coverage_pct"]) for row in baseline_rows
        }
        relative_coverage = [[0.0 for _ in baseline_rows]]
        relative_coverage.extend(
            [
                float(row["coverage_pct"]) - baseline_coverage[int(row["seed"])]
                for row in group
            ]
            for group in groups[1:]
        )
        axes[1, 0].boxplot(relative_coverage, labels=labels, showmeans=True)
        for index, group_values in enumerate(relative_coverage, 1):
            axes[1, 0].scatter(
                [index] * len(group_values), group_values, s=15, alpha=0.45
            )
        axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_ylabel("Coverage difference (percentage points)")
        axes[1, 0].set_title("Coverage relative to matched baseline")

        baseline_events = [event_percentages(Path(str(row["run"]))) for row in baseline_rows]
        prompt_event_groups: dict[int, list[dict[str, float]]] = {
            prompt: [] for prompt in args.prompts
        }
        by_run: dict[tuple[int, int, int, str], dict[str, float]] = defaultdict(dict)
        for item in event_rows:
            key = (
                int(item["prompt"]), int(item["generation"]),
                int(item["seed"]), str(item["supervisor"]),
            )
            by_run[key][str(item["event"])] = float(item["percentage"])
        for key, values in by_run.items():
            prompt_event_groups[key[0]].append(values)
        all_event_groups = [
            baseline_events,
            *[prompt_event_groups[prompt] for prompt in args.prompts],
        ]
        events = sorted(
            {event for group in all_event_groups for run in group for event in run},
            key=lambda event: -max(
                statistics.mean(run.get(event, 0.0) for run in group)
                for group in all_event_groups
            ),
        )
        positions = list(range(len(events)))
        width = 0.82 / len(labels)
        for index, (label, group) in enumerate(zip(labels, all_event_groups)):
            means = [
                statistics.mean(run.get(event, 0.0) for run in group)
                for event in events
            ]
            axes[1, 1].bar(
                [position - 0.41 + width / 2 + index * width for position in positions],
                means, width, label=label,
            )
        axes[1, 1].set_xticks(
            positions,
            [event.replace("_", " ") for event in events],
            rotation=28,
            ha="right",
        )
        axes[1, 1].set_ylabel("Mean selected events (%)")
        axes[1, 1].set_title("Event distribution")
        axes[1, 1].legend(fontsize=8)

        for axis in axes.flat:
            axis.grid(axis="y", linestyle="--", alpha=0.3)
        figure.suptitle(
            f"Exploration prompt sensitivity: {len(args.prompts)} prompts $\\times$ "
            f"{len(args.generations)} generations $\\times$ {len(args.seeds)} seeds"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(
            output / "exploration_prompt_sensitivity.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Analyzed {len(rows)} LLM runs and {len(baseline_rows)} baseline runs")
        print(f"Saved outputs under {output}")
        return 0

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    success_rates = [100.0 * sum(bool(row["success"]) for row in group) / len(group) for group in groups]
    axes[0, 0].bar(labels, success_rates, color=["#777777", *[plt.get_cmap("tab10")(i) for i in range(len(args.prompts))]])
    axes[0, 0].set_ylabel("Success rate (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_title("Task success")

    for axis, field, title, ylabel in (
        (axes[0, 1], "duration_s", "Completion time / timeout", "Seconds"),
        (axes[1, 0], "collisions", "Collisions", "Collision count"),
    ):
        values = [[float(row[field]) for row in group] for group in groups]
        axis.boxplot(values, labels=labels, showmeans=True)
        for index, group_values in enumerate(values, 1):
            axis.scatter([index] * len(group_values), group_values, s=15, alpha=0.45)
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    coverage_values = [
        [float(row["coverage_pct"]) for row in group]
        for group in groups
    ]
    axes[0, 2].boxplot(coverage_values, labels=labels, showmeans=True)
    for index, group_values in enumerate(coverage_values, 1):
        axes[0, 2].scatter(
            [index] * len(group_values), group_values, s=15, alpha=0.45
        )
    axes[0, 2].set_ylabel("Coverage (%)")
    axes[0, 2].set_title("Coverage distribution")

    baseline_coverage = {
        int(row["seed"]): float(row["coverage_pct"]) for row in baseline_rows
    }
    relative_coverage = [[0.0 for _ in baseline_rows]]
    relative_coverage.extend(
        [
            float(row["coverage_pct"]) - baseline_coverage[int(row["seed"])]
            for row in group
        ]
        for group in groups[1:]
    )
    axes[1, 1].boxplot(relative_coverage, labels=labels, showmeans=True)
    for index, group_values in enumerate(relative_coverage, 1):
        axes[1, 1].scatter(
            [index] * len(group_values), group_values, s=15, alpha=0.45
        )
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Coverage difference (percentage points)")
    axes[1, 1].set_title("Coverage relative to matched baseline")

    baseline_events = [event_percentages(Path(str(row["run"]))) for row in baseline_rows]
    prompt_event_groups: dict[int, list[dict[str, float]]] = {
        prompt: [] for prompt in args.prompts
    }
    by_run: dict[tuple[int, int, int, str], dict[str, float]] = defaultdict(dict)
    for item in event_rows:
        key = (
            int(item["prompt"]), int(item["generation"]),
            int(item["seed"]), str(item["supervisor"]),
        )
        by_run[key][str(item["event"])] = float(item["percentage"])
    for key, values in by_run.items():
        prompt_event_groups[key[0]].append(values)
    all_event_groups = [baseline_events, *[prompt_event_groups[prompt] for prompt in args.prompts]]
    events = sorted(
        {event for group in all_event_groups for run in group for event in run},
        key=lambda event: -max(
            statistics.mean(run.get(event, 0.0) for run in group)
            for group in all_event_groups
        ),
    )
    positions = list(range(len(events)))
    width = 0.82 / len(labels)
    for index, (label, group) in enumerate(zip(labels, all_event_groups)):
        means = [statistics.mean(run.get(event, 0.0) for run in group) for event in events]
        axes[1, 2].bar(
            [position - 0.41 + width / 2 + index * width for position in positions],
            means, width, label=label,
        )
    axes[1, 2].set_xticks(
        positions, [event.replace("_", " ") for event in events], rotation=28, ha="right"
    )
    axes[1, 2].set_ylabel("Mean selected events (%)")
    axes[1, 2].set_title("Event distribution")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.tick_params(axis="x", labelrotation=15)
    figure.suptitle(
        f"{args.task.capitalize()} prompt sensitivity: "
        f"{len(args.prompts)} prompts × {len(args.generations)} generations × "
        f"{len(args.seeds)} seeds"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / f"{args.task}_prompt_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Analyzed {len(rows)} LLM runs and {len(baseline_rows)} baseline runs")
    print(f"Saved outputs under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
