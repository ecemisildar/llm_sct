"""Export prompt-sensitivity panels and legends as separate transparent PNGs."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "RESULTS_LAST" / "analysis"
COLORS = {
    "without_fixed_spec": "#00A896",
    "with_fixed_spec": "#7B2CBF",
}
ALPHA = 0.32
FONT_SIZE = 16


def read_rows(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["prompt"] = int(row["prompt"])
        row["generation"] = int(row["generation"])
    return rows


def baseline(path: Path):
    if not path.exists():
        return None
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


def setup(rows):
    prompts = sorted({row["prompt"] for row in rows})
    modes = [mode for mode in COLORS if any(row["collision_control"] == mode for row in rows)]
    centers = list(range(1, len(prompts) + 1))
    positions = [
        center + (mode_index - (len(modes) - 1) / 2) * 0.32
        for center in centers
        for mode_index in range(len(modes))
    ]
    groups = [
        [row for row in rows if row["prompt"] == prompt and row["collision_control"] == mode]
        for prompt in prompts
        for mode in modes
    ]
    colors = [COLORS[mode] for _prompt in prompts for mode in modes]
    return prompts, modes, centers, positions, groups, colors


def style_axis(axis, prompts, centers, ylabel, *, ylim=None, log_collisions=False):
    axis.set_xticks(centers, [f"P{prompt}" for prompt in prompts])
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="both", labelsize=FONT_SIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    if ylim is not None:
        axis.set_ylim(*ylim)
    else:
        axis.set_ylim(bottom=0)
    if log_collisions:
        axis.set_yscale("symlog", linthresh=1, linscale=0.5)
        axis.set_yticks([0, 1, 10], ["0", "1", "10"])
        axis.margins(y=0.12)


def save_panel(figure, output: Path):
    figure.tight_layout(pad=0.35)
    figure.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(figure)


def distribution_panel(rows, field, ylabel, output, *, baseline_value=None,
                       ylim=None, collision_scale=False, transform=None,
                       violations=False):
    prompts, _modes, centers, positions, groups, colors = setup(rows)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    values = []
    for group in groups:
        if transform:
            values.append([transform(row) for row in group if transform(row) is not None])
        else:
            values.append([float(row[field]) for row in group])
    boxes = axis.boxplot(values, positions=positions, widths=0.26,
                         showmeans=True, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(ALPHA)
    for position, group_values, color in zip(positions, values, colors):
        axis.scatter([position] * len(group_values), group_values,
                     s=14, alpha=ALPHA, color=color)
    if violations:
        for position, group in zip(positions, groups):
            bad = [transform(row) for row in group
                   if transform(row) is not None
                   and abs(float(row["red_boxes"]) - float(row["blue_boxes"])) > 1]
            axis.scatter([position] * len(bad), bad, marker="x", s=40,
                         color="#D62728", zorder=5)
        axis.axhline(50, color="#333333", linestyle="--", linewidth=1.5)
    if baseline_value is not None:
        axis.axhline(float(baseline_value), color="#333333",
                     linestyle="--", linewidth=1.5)
    style_axis(axis, prompts, centers, ylabel, ylim=ylim,
               log_collisions=collision_scale)
    save_panel(figure, output)


def bar_panel(rows, ylabel, output, *, baseline_value=None, best_median=False,
              figsize=(6.4, 4.8)):
    prompts, _modes, centers, positions, groups, colors = setup(rows)
    figure, axis = plt.subplots(figsize=figsize)
    if best_median:
        for position, group, color in zip(positions, groups, colors):
            by_generation = {}
            for row in group:
                by_generation.setdefault(row["generation"], []).append(row)
            ranked = sorted(
                ((generation, statistics.mean(float(row["task_success_pct"]) for row in generation_rows))
                 for generation, generation_rows in by_generation.items()),
                key=lambda item: (-round(item[1], 10), item[0]),
            )
            if not ranked:
                continue
            axis.bar(position, ranked[0][1], width=0.24, color=color,
                     alpha=ALPHA, edgecolor="#333333", linewidth=0.6)
            axis.scatter(
                position,
                ranked[len(ranked) // 2][1],
                marker="o",
                s=30,
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.0,
                zorder=5,
            )
    else:
        scores = [statistics.mean(float(row["task_success_pct"]) for row in group)
                  if group else 0 for group in groups]
        axis.bar(positions, scores, width=0.28, color=colors, alpha=ALPHA)
    if baseline_value is not None:
        axis.axhline(float(baseline_value), color="#333333",
                     linestyle="--", linewidth=1.5)
    style_axis(axis, prompts, centers, ylabel, ylim=(0, 105))
    if best_median:
        axis.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", linestyle="None",
                    markerfacecolor="white", markeredgecolor="#222222",
                    markersize=5.5, label="Median\ngeneration",
                )
            ],
            loc="lower right",
            bbox_to_anchor=(0.99, 1.02),
            frameon=True,
            fancybox=False,
            edgecolor="#333333",
            facecolor="white",
            framealpha=1.0,
            fontsize=11,
            handletextpad=0.5,
        )
    save_panel(figure, output)


def legend(output: Path, *, baseline_line=False, generations=False):
    handles = [
        Patch(facecolor=COLORS["without_fixed_spec"], alpha=ALPHA,
              label="LLM"),
        Patch(facecolor=COLORS["with_fixed_spec"], alpha=ALPHA,
              label=r"LLM + $Spec_{col}$"),
    ]
    if baseline_line:
        handles.append(Line2D([0], [0], color="#333333", linestyle="--",
                              linewidth=1.5, label="Baseline mean"))
    if generations:
        handles.extend([
            Patch(facecolor="white", edgecolor="#333333", label="Best generation"),
            Patch(facecolor="white", edgecolor="#333333", hatch="///",
                  label="Median generation"),
        ])
    figure = plt.figure(figsize=(max(4.5, 1.65 * len(handles)), 0.65))
    figure.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=12,
        handletextpad=1.25,
        columnspacing=2.0,
        borderaxespad=0.4,
    )
    figure.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(figure)


def export_exploration():
    folder = RESULTS / "exploration" / "with_fixed_spec"
    rows = read_rows(folder / "exploration_prompt_sensitivity_runs.csv")
    supplemental = sorted(
        (ROOT / "RESULTS_LAST" / "Stupid_prompts" / "p7_task_worlds").glob(
            "exploration_api4_*/results_strict_plain_*/analysis/exploration/"
            "exploration_prompt_sensitivity_runs.csv"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if supplemental:
        new_rows = read_rows(supplemental[-1])
        existing = {
            (row["prompt"], row["generation"], row["collision_control"], row["seed"])
            for row in rows
        }
        rows.extend(
            row for row in new_rows
            if (row["prompt"], row["generation"], row["collision_control"], row["seed"])
            not in existing
        )
    rows = [row for row in rows if row["prompt"] != 5]
    for row in rows:
        if row["prompt"] > 5:
            row["prompt"] -= 1
    base = baseline(folder / "exploration_baseline_summary.csv")
    output = folder / "separate_panels"
    output.mkdir(exist_ok=True)
    distribution_panel(rows, "coverage_pct", "Coverage (%)", output / "coverage.png",
                       baseline_value=base["coverage_mean_pct"], ylim=(0, 100))
    distribution_panel(rows, "collisions", "Collisions", output / "collisions.png",
                       baseline_value=base["collisions_mean"])
    legend(output / "legend.png", baseline_line=True)


def export_patrolling():
    folder = RESULTS / "patrolling" / "comparison"
    rows = read_rows(folder / "patrolling_prompt_sensitivity_runs.csv")
    supplemental = sorted(
        (ROOT / "RESULTS_LAST" / "Stupid_prompts" / "p7_task_worlds").glob(
            "patrolling_api4_*/results_strict_plain_*/analysis/patrolling/"
            "patrolling_prompt_sensitivity_runs.csv"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if supplemental:
        new_rows = read_rows(supplemental[-1])
        existing = {
            (row["prompt"], row["generation"], row["collision_control"], row["seed"])
            for row in rows
        }
        rows.extend(
            row for row in new_rows
            if (row["prompt"], row["generation"], row["collision_control"], row["seed"])
            not in existing
        )
    rows = [row for row in rows if row["prompt"] != 6]
    for row in rows:
        if row["prompt"] > 6:
            row["prompt"] -= 1
    base = baseline(folder / "patrolling_baseline_summary.csv")
    output = folder / "separate_panels"
    output.mkdir(exist_ok=True)
    bar_panel(rows, "Mean task progression (%)", output / "task_progression.png",
              baseline_value=base["task_success_mean_pct"], best_median=True,
              figsize=(8.6, 2.8))
    distribution_panel(rows, "duration_s", "Completion time (s)",
                       output / "completion_time.png",
                       baseline_value=base["duration_all_mean_s"])
    distribution_panel(rows, "collisions", "Collisions",
                       output / "collisions.png",
                       baseline_value=base["collisions_mean"])
    legend(output / "legend.png", baseline_line=True)


def red_percentage(row):
    total = float(row["red_boxes"]) + float(row["blue_boxes"])
    return 100.0 * float(row["red_boxes"]) / total if total else None


def export_delivery():
    folder = RESULTS / "delivery" / "comparison"
    rows = read_rows(folder / "delivery_prompt_sensitivity_figure_runs.csv")
    base = baseline(folder / "delivery_baseline_summary.csv")
    output = folder / "separate_panels"
    output.mkdir(exist_ok=True)
    distribution_panel(rows, "delivered_boxes", "Delivered boxes",
                       output / "delivered_boxes.png",
                       baseline_value=base["delivered_boxes_mean"], ylim=(0, 6.5))
    bar_panel(rows, "Mean task progression (%)", output / "task_progression.png",
              baseline_value=base["task_success_mean_pct"])
    distribution_panel(rows, "duration_s", "Completion time (s)",
                       output / "completion_time.png",
                       baseline_value=base["duration_all_mean_s"])
    distribution_panel(rows, "red_boxes", "Red boxes (% of zone deliveries)",
                       output / "zone_balance.png", ylim=(0, 105),
                       baseline_value=base["red_box_percentage_mean"],
                       transform=red_percentage, violations=True)
    distribution_panel(rows, "collisions", "Collisions (log scale)",
                       output / "collisions.png",
                       baseline_value=base["collisions_mean"], collision_scale=True)
    legend(output / "legend.png", baseline_line=True)


def main():
    plt.rcParams.update({"font.size": FONT_SIZE})
    export_exploration()
    export_patrolling()
    export_delivery()


if __name__ == "__main__":
    main()
