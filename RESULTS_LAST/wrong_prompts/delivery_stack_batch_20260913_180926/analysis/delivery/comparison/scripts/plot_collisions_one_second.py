#!/usr/bin/env python3
"""Plot mean collisions per run after a uniform one-second pair debounce."""

from __future__ import annotations

import ast
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "final results"
OUTPUT = FINAL / "collision_analysis"
COLORS = {"fixed_collision": "#7B2CBF", "llm_collision": "#00A896"}
FONT_SIZE = 20
DEBOUNCE_S = 1.0


def collision_key(row: dict[str, str]) -> str:
    raw = row.get("key", "")
    if raw:
        try:
            value = ast.literal_eval(raw)
            if isinstance(value, tuple):
                return repr(tuple(sorted(str(item) for item in value)))
        except (SyntaxError, ValueError):
            pass
        return raw
    return "|".join(sorted((row.get("entity_a", ""), row.get("entity_b", ""))))


def count_with_debounce(path: Path) -> int:
    last: dict[str, float] = {}
    count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp = float(row["stamp_sec"]) + float(row.get("stamp_nsec", 0) or 0) * 1e-9
            key = collision_key(row)
            if key not in last or timestamp - last[key] >= DEBOUNCE_S:
                count += 1
                last[key] = timestamp
    return count


def run_index(task: str) -> dict[str, Path]:
    roots = [FINAL / task, ROOT / "new_results" / "llm" / task]
    result: dict[str, Path] = {}
    for root in roots:
        if root.exists():
            for path in root.glob("*_collision/seed_*/S_*/run_*/bumps_global.csv"):
                result[path.parent.name] = path
    return result


def prompt_runs(task: str) -> list[dict[str, object]]:
    index = run_index(task)
    source = FINAL / f"{task}_prompt_sensitivity" / f"{task}_prompt_sensitivity_runs.csv"
    rows: list[dict[str, object]] = []
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            run_name = Path(row["run"]).name
            path = index.get(run_name)
            if path is None:
                raise FileNotFoundError(f"Missing bump log for {task}/{run_name}")
            rows.append({
                "task": task,
                "prompt": int(row["prompt"]),
                "controller": row["collision_control"],
                "generation": int(row["generation"]),
                "seed": int(row["seed"]),
                "run": run_name,
                "collisions_1s": count_with_debounce(path),
            })
    return rows


def baseline_runs(task: str) -> list[int]:
    root = ROOT / "new_results" / "baseline" / f"results_{task}"
    values = []
    for seed_dir in sorted(root.glob("seed_*")):
        paths = sorted(seed_dir.glob("*/run_*/bumps_global.csv"))
        if paths:
            values.append(count_with_debounce(paths[-1]))
    return values


def mean_sd(values: list[int]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tasks = ("exploration", "patrolling")
    all_rows = [row for task in tasks for row in prompt_runs(task)]

    per_run_path = OUTPUT / "collisions_per_run_1s.csv"
    with per_run_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row in all_rows:
        grouped[(str(row["task"]), int(row["prompt"]), str(row["controller"]))].append(int(row["collisions_1s"]))

    summary_rows = []
    baselines = {}
    task_prompts = {
        task: sorted({int(row["prompt"]) for row in all_rows if row["task"] == task})
        for task in tasks
    }
    for task in tasks:
        baseline = baseline_runs(task)
        baselines[task] = mean_sd(baseline)
        for prompt in task_prompts[task]:
            for controller in ("fixed_collision", "llm_collision"):
                values = grouped[(task, prompt, controller)]
                mean, sd = mean_sd(values)
                summary_rows.append({"task": task, "prompt": prompt, "controller": controller,
                                     "runs": len(values), "mean_collisions_per_run": f"{mean:.4f}",
                                     "sd_collisions_per_run": f"{sd:.4f}", "total_collisions": sum(values)})
    summary_path = OUTPUT / "collisions_by_prompt_1s_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.rcParams.update({"font.size": FONT_SIZE})
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    offsets = {"fixed_collision": -0.18, "llm_collision": 0.18}
    for axis, task in zip(axes, tasks):
        centers = task_prompts[task]
        for controller in ("fixed_collision", "llm_collision"):
            means, errors = [], []
            for prompt in centers:
                means.append(float(next(row["mean_collisions_per_run"] for row in summary_rows
                                        if row["task"] == task and row["prompt"] == prompt and row["controller"] == controller)))
                errors.append(float(next(row["sd_collisions_per_run"] for row in summary_rows
                                         if row["task"] == task and row["prompt"] == prompt and row["controller"] == controller)))
            positions = [x + offsets[controller] for x in centers]
            bars = axis.bar(positions, means, width=0.32, color=COLORS[controller], alpha=0.75,
                            yerr=errors, capsize=3, error_kw={"linewidth": 1.2})
            for bar, value in zip(bars, means):
                axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                              fontsize=12, rotation=90)
        baseline_mean, _ = baselines[task]
        axis.axhline(baseline_mean, color="#333333", linestyle="--", linewidth=1.8)
        axis.set_xticks(list(centers), [f"P{x}" for x in centers])
        axis.set_ylim(bottom=0)
        axis.set_title(task.capitalize(), fontsize=FONT_SIZE)
        axis.set_ylabel("Collisions per run", fontsize=FONT_SIZE)
        axis.tick_params(labelsize=FONT_SIZE)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.legend(handles=[Patch(facecolor=COLORS["fixed_collision"], alpha=.75, label="Fixed"),
                           Patch(facecolor=COLORS["llm_collision"], alpha=.75, label="LLM"),
                           Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.8, label="Baseline mean")],
                  loc="lower center", ncol=3, frameon=False, fontsize=FONT_SIZE)
    title = figure.suptitle("Collision frequency with a 1-second debounce", fontsize=FONT_SIZE)
    figure.tight_layout(rect=(0, .17, 1, .92))
    figure.savefig(OUTPUT / "collisions_per_run_1s.png", dpi=200, bbox_inches="tight")
    title.set_visible(False)
    figure.tight_layout(rect=(0, .17, 1, .98))
    figure.savefig(OUTPUT / "collisions_per_run_1s_no_title.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
