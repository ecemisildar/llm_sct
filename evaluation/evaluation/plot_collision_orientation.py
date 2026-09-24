#!/usr/bin/env python3
"""Plot front-involved collision percentages by task, prompt, and controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FONT_SIZE = 20
COLORS = {"fixed_collision": "#7B2CBF", "llm_collision": "#00A896"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FONT_SIZE})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    prompts = list(range(1, 7))
    centers = list(range(1, 7))
    offsets = {"fixed_collision": -0.17, "llm_collision": 0.17}

    for axis, task in zip(axes, ("exploration", "patrolling")):
        task_rows = [row for row in rows if row["task"] == task]
        for controller in ("fixed_collision", "llm_collision"):
            front_other_values = []
            front_front_values = []
            front_counts = []
            totals = []
            for prompt in prompts:
                row = next(
                    item for item in task_rows
                    if item["controller"] == controller
                    and item["prompt"] == str(prompt)
                )
                front_other_values.append(float(row["front_other_pct"]))
                front_front_values.append(float(row["front_front_pct"]))
                front_counts.append(
                    int(row["front_other_count"]) + int(row["front_front_count"])
                )
                totals.append(int(row["total_collisions"]))
            positions = [center + offsets[controller] for center in centers]
            axis.bar(
                positions, front_other_values, width=0.30,
                color=COLORS[controller], alpha=0.7,
            )
            axis.bar(
                positions, front_front_values, width=0.30,
                bottom=front_other_values, color=COLORS[controller],
                alpha=0.7, hatch="///", edgecolor="#222222", linewidth=0.8,
            )
            for position, front_other, front_front, front_count, total in zip(
                positions, front_other_values, front_front_values,
                front_counts, totals,
            ):
                percentage = front_other + front_front
                axis.annotate(
                    f"({front_count}/{total})",
                    (position, percentage),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, rotation=90,
                )
        baseline = next(
            row for row in task_rows if row["controller"] == "baseline"
        )
        baseline_pct = (
            float(baseline["front_front_pct"])
            + float(baseline["front_other_pct"])
        )
        axis.axhline(
            baseline_pct, color="#333333", linestyle="--", linewidth=1.8
        )
        axis.set_xticks(centers, [f"P{prompt}" for prompt in prompts])
        axis.set_ylim(0, 105)
        axis.set_title(task.capitalize(), fontsize=FONT_SIZE)
        axis.set_ylabel("Front-involved collisions (%)", fontsize=FONT_SIZE)
        axis.tick_params(labelsize=FONT_SIZE)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLORS["fixed_collision"], alpha=0.7, label="Fixed collision"),
        Patch(facecolor=COLORS["llm_collision"], alpha=0.7, label="LLM collision"),
        Patch(
            facecolor="white", edgecolor="#222222", hatch="///",
            label="Front–front portion",
        ),
        Line2D(
            [0], [0], color="#333333", linestyle="--", linewidth=1.8,
            label="Baseline",
        ),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=2, frameon=False, fontsize=FONT_SIZE,
    )
    title = figure.suptitle(
        "Front-involved collision distribution by prompt", fontsize=FONT_SIZE
    )
    figure.tight_layout(rect=(0, 0.24, 1, 0.91))
    figure.savefig(
        args.output_dir / "collision_orientation_by_prompt.png",
        dpi=200, bbox_inches="tight",
    )
    title.set_visible(False)
    figure.tight_layout(rect=(0, 0.24, 1, 0.98))
    figure.savefig(
        args.output_dir / "collision_orientation_by_prompt_no_title.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
