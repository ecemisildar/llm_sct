#!/usr/bin/env python3
"""Analyze one simulation seed across prompt and generation variants."""

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


def run_seed(run: Path) -> int | None:
    status = (run / "SAVE_STATUS.txt").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*random_seed:\s*(\d+)\s*$", status, re.MULTILINE)
    return int(match.group(1)) if match else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("exploration", "patrolling", "delivery"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    yaml_root = root / "automata" / "resulting_automata" / "YAML" / args.task
    yaml_index: dict[str, tuple[int, int, Path]] = {}
    for yaml_path in yaml_root.glob("prompt_*/generation_*/S_*.yaml"):
        prompt = int(yaml_path.parents[1].name.removeprefix("prompt_"))
        generation = int(yaml_path.parent.name.removeprefix("generation_"))
        yaml_index[yaml_path.stem] = (prompt, generation, yaml_path)

    result_root = root / "new_results" / "llm" / f"results_{args.task}"
    newest_by_supervisor: dict[str, Path] = {}
    for result_path in result_root.glob("S_*/robots_*/run_*/task_result.csv"):
        run = result_path.parent
        status = run / "SAVE_STATUS.txt"
        if not status.is_file() or "Saving OK" not in status.read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        if run_seed(run) != args.seed:
            continue
        supervisor = run.parents[1].name
        previous = newest_by_supervisor.get(supervisor)
        if previous is None or status.stat().st_mtime > (previous / "SAVE_STATUS.txt").stat().st_mtime:
            newest_by_supervisor[supervisor] = run

    rows: list[dict[str, object]] = []
    events_by_run: dict[str, dict[str, float]] = {}
    for supervisor, run in sorted(newest_by_supervisor.items()):
        if supervisor not in yaml_index:
            continue
        prompt, generation, yaml_path = yaml_index[supervisor]
        success, duration = read_result(run)
        rows.append(
            {
                "task": args.task,
                "seed": args.seed,
                "prompt": prompt,
                "generation": generation,
                "supervisor": supervisor,
                "success": success,
                "duration_s": duration,
                "coverage_pct": coverage_at(run, duration),
                "collisions": collisions(run),
                "yaml": str(yaml_path),
                "run": str(run),
            }
        )
        events_by_run[supervisor] = event_percentages(run)
    if not rows:
        raise SystemExit("No matching completed LLM runs found")

    rows.sort(key=lambda row: (int(row["prompt"]), int(row["generation"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output.with_suffix(".csv"), rows)

    baseline = newest_completed_run_for_seed(
        root / "new_results" / "baseline" / f"results_{args.task}", args.seed
    )
    baseline_success, baseline_duration = read_result(baseline)
    baseline_metrics = {
        "duration_s": baseline_duration,
        "coverage_pct": coverage_at(baseline, baseline_duration),
        "collisions": collisions(baseline),
    }
    prompts = sorted({int(row["prompt"]) for row in rows})
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    coverage_title = (
        "Coverage after 300 seconds"
        if args.task == "exploration"
        else "Coverage at run end"
    )
    metric_plots = [
        (axes[0, 1], "coverage_pct", "Coverage (%)", coverage_title),
        (axes[1, 0], "collisions", "Collision count", "Collisions"),
    ]
    if args.task == "exploration":
        axes[0, 0].axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            "Fixed-duration evaluation\n(300 seconds per run)",
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        metric_plots.insert(
            0,
            (axes[0, 0], "duration_s", "Seconds", "Completion time / timeout"),
        )

    for axis, field, ylabel, title in metric_plots:
        for prompt in prompts:
            group = [row for row in rows if row["prompt"] == prompt]
            x = [prompt + (int(row["generation"]) - 2) * 0.07 for row in group]
            y = [float(row[field]) for row in group]
            axis.scatter(x, y, s=55, color=colors((prompt - 1) % 10), zorder=3)
            if len(y) > 1:
                axis.errorbar(
                    prompt,
                    statistics.mean(y),
                    yerr=statistics.pstdev(y),
                    fmt="D",
                    color="black",
                    capsize=4,
                    markersize=5,
                )
        axis.axhline(
            float(baseline_metrics[field]),
            color="black",
            linestyle="--",
            label=f"Baseline seed {args.seed}",
        )
        axis.set_xticks(prompts, [f"Prompt {prompt}" for prompt in prompts])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend()

    events = sorted(
        {event for values in events_by_run.values() for event in values},
        key=lambda event: -statistics.mean(
            values.get(event, 0.0) for values in events_by_run.values()
        ),
    )
    positions = list(range(len(events)))
    width = 0.8 / max(1, len(prompts))
    for index, prompt in enumerate(prompts):
        group = [row for row in rows if row["prompt"] == prompt]
        means = [
            statistics.mean(
                events_by_run[str(row["supervisor"])].get(event, 0.0)
                for row in group
            )
            for event in events
        ]
        axes[1, 1].bar(
            [position - 0.4 + width / 2 + index * width for position in positions],
            means,
            width,
            label=f"Prompt {prompt}",
            color=colors((prompt - 1) % 10),
        )
    axes[1, 1].set_xticks(
        positions,
        [event.replace("_", " ") for event in events],
        rotation=25,
        ha="right",
    )
    axes[1, 1].set_ylabel("Mean selected events (%)")
    axes[1, 1].set_title("Event distribution by prompt")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

    if args.task == "exploration":
        subtitle = f"{len(rows)} LLM runs; coverage measured after 300 seconds"
    else:
        success_count = sum(bool(row["success"]) for row in rows)
        subtitle = (
            f"LLM successes {success_count}/{len(rows)}; baseline "
            f"{'success' if baseline_success else 'timeout'}"
        )
    figure.suptitle(
        f"{args.task.capitalize()} prompt/generation analysis — seed {args.seed}\n"
        f"{subtitle}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {args.output.with_suffix('.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
