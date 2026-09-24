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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGURE_FONT_SIZE = 20
plt.rcParams.update({"font.size": FIGURE_FONT_SIZE})

from plot_paired_run import (
    coverage_at,
    event_percentages,
    newest_completed_run_for_seed,
    read_result,
)
from plot_collisions_one_second import count_with_debounce


def collisions(run: Path) -> int:
    """Count at most one collision per entity pair in each one-second window."""
    path = run / "bumps_global.csv"
    return count_with_debounce(path) if path.is_file() else 0


def seed_for_run(run: Path) -> int | None:
    text = (run / "SAVE_STATUS.txt").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*random_seed:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def selected_event_paths(run: Path) -> list[Path]:
    paths = list(run.glob("selected_events_robot_*.csv"))
    if paths:
        return paths
    return list(
        run.parent.glob(f"robots_*/{run.name}/selected_events_robot_*.csv")
    )


def task_success_pct(run: Path, task: str, success: bool) -> float:
    """Calculate partial credit from the task's recorded milestones."""
    if task == "patrolling":
        with (run / "task_progress.csv").open(newline="", encoding="utf-8") as stream:
            progress_rows = list(csv.DictReader(stream))
        completed_targets = sum(
            int(row.get("completed_colors", 0) or 0) for row in progress_rows
        )
        total_targets = sum(
            int(row.get("total_colors", 0) or 0) for row in progress_rows
        )
        return 100.0 * completed_targets / total_targets if total_targets else 0.0
    if task == "delivery":
        return 100.0 * delivery_counts(run)["delivered_boxes"] / delivery_counts(run)["total_targets"]
    return 100.0 if success else 0.0


def delivery_counts(run: Path) -> dict[str, int]:
    """Count zone announcements; generic drop_object precedes announcement."""
    counts: dict[str, int] = defaultdict(int)
    for path in selected_event_paths(run):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                counts[row.get("selected_event", "").strip()] += 1
    red = counts["EV_drop_zone_red"] + counts["EV_drop_zone_a"]
    blue = counts["EV_drop_zone_blue"] + counts["EV_drop_zone_b"]
    # Progress records the team count on each robot, so take its maximum.
    with (run / "task_progress.csv").open(newline="", encoding="utf-8") as stream:
        progress = list(csv.DictReader(stream))
    with (run / "task_result.csv").open(newline="", encoding="utf-8") as stream:
        result = next(csv.DictReader(stream))
    total = int(result["total_targets"])
    delivered = int(result["completed_targets"])
    if total <= 0 or sum(int(r["delivered_count"]) for r in progress) != delivered:
        raise ValueError(f"Inconsistent legacy delivery progress: {run}")
    return {
        "delivered_boxes": delivered, "total_targets": total, "red_boxes": red, "blue_boxes": blue,
        "zone_drop_events": red + blue,
        "delivery_count_matches_events": delivered == red + blue,
    }


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
            for path in yaml_root.glob(
                f"*/prompt_{prompt}/generation_{generation}/S_*.prompt.txt"
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
    task_scores = [float(row["task_success_pct"]) for row in group]
    task_score_mean, task_score_sd = mean_sd(task_scores)
    successes = sum(bool(row["success"]) for row in group)
    result = {
        **prefix,
        "runs": len(group),
        "successes": successes,
        "success_rate_pct": 100.0 * successes / len(group),
        "task_success_mean_pct": task_score_mean,
        "task_success_sd_pct": task_score_sd,
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
    if "delivered_boxes" in group[0]:
        result["delivered_boxes_mean"], result["delivered_boxes_sd"] = mean_sd(
            [float(row["delivered_boxes"]) for row in group]
        )
        result["red_boxes_mean"] = statistics.mean(float(row["red_boxes"]) for row in group)
        result["blue_boxes_mean"] = statistics.mean(float(row["blue_boxes"]) for row in group)
    return result


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
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help=(
            "Experiment directory containing YAML/<task> and <task> run folders; "
            "defaults to the repository's legacy result layout"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Analyze currently completed runs even when some cells are missing",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Generate LLM prompt-sensitivity outputs without baseline runs",
    )
    parser.add_argument(
        "--latest-per-cell",
        action="store_true",
        help=(
            "Use only the newest YAML for each prompt/generation/collision-control "
            "cell when multiple saved controllers exist"
        ),
    )
    parser.add_argument(
        "--collision-controls",
        nargs="+",
        choices=("without_fixed_spec", "with_fixed_spec", "llm_collision", "fixed_collision"),
        help="Analyze only the selected collision-control result directories",
    )
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    experiment_root = (
        args.experiment_root.expanduser().resolve()
        if args.experiment_root
        else None
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    yaml_root = (
        experiment_root / "YAML" / args.task
        if experiment_root
        else root / "automata" / "resulting_automata" / "YAML" / args.task
    )
    prompt_texts = prompt_texts_for(yaml_root, args.prompts, args.generations)
    write_csv(
        output / f"{args.task}_prompts.csv",
        [
            {"prompt": prompt, "prompt_text": prompt_texts[prompt]}
            for prompt in args.prompts
        ],
    )
    yaml_paths = list(
        yaml_root.glob("*/prompt_*/generation_*/S_*.yaml")
    )
    if args.collision_controls:
        yaml_paths = [
            path for path in yaml_paths
            if path.parents[2].name in args.collision_controls
        ]
    if args.latest_per_cell:
        latest: dict[tuple[int, int, str], Path] = {}
        for yaml_path in yaml_paths:
            cell = (
                int(yaml_path.parents[1].name.removeprefix("prompt_")),
                int(yaml_path.parent.name.removeprefix("generation_")),
                yaml_path.parents[2].name,
            )
            if cell not in latest or yaml_path.stat().st_mtime > latest[cell].stat().st_mtime:
                latest[cell] = yaml_path
        yaml_paths = list(latest.values())

    yaml_index: dict[tuple[str, str], tuple[int, int, str, Path]] = {}
    for yaml_path in yaml_paths:
        prompt = int(yaml_path.parents[1].name.removeprefix("prompt_"))
        generation = int(yaml_path.parent.name.removeprefix("generation_"))
        if prompt in args.prompts and generation in args.generations:
            mode = yaml_path.parents[2].name
            yaml_index[(mode, str(yaml_path.resolve()))] = (
                prompt, generation, mode, yaml_path
            )

    newest: dict[tuple[str, str, int], Path] = {}
    result_roots = (
        (experiment_root / args.task,)
        if experiment_root
        else (
            root / "new_results" / "llm" / args.task,
            root / "new_results" / "llm" / f"results_{args.task}",
        )
    )
    result_files = {
        path.resolve(): path
        for result_root in result_roots
        if result_root.is_dir()
        for path in result_root.rglob("task_result.csv")
        if path.parent.name.startswith("run_")
    }.values()
    for result_file in result_files:
        run = result_file.parent
        status = run / "SAVE_STATUS.txt"
        if not status.is_file() or "Saving OK" not in status.read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        supervisor = next(
            (parent.name for parent in run.parents if parent.name.startswith("S_")),
            "",
        )
        collision_control = (
            run.relative_to(experiment_root / args.task).parts[0]
            if experiment_root
            else next(
                (
                    parent.name
                    for parent in run.parents
                    if parent.name in {
                        "without_fixed_spec", "with_fixed_spec",
                        "llm_collision", "fixed_collision",
                    }
                ),
                "",
            )
        )
        seed = seed_for_run(run)
        status_text = status.read_text(encoding="utf-8", errors="replace")
        metadata = re.search(r"^metadata_yaml:\s*(.+)$", status_text, re.MULTILINE)
        if metadata:
            controller_key = (collision_control, str(Path(metadata.group(1).strip()).resolve()))
        else:
            candidates = [key for key in yaml_index if key[0] == collision_control
                          and Path(key[1]).stem == supervisor]
            if len(candidates) > 1:
                raise SystemExit(f"Ambiguous controller attribution without metadata: {run}")
            controller_key = candidates[0] if candidates else (collision_control, "")
        if controller_key not in yaml_index or seed not in args.seeds:
            continue
        key = (collision_control, controller_key[1], int(seed))
        if key not in newest or status.stat().st_mtime > (newest[key] / "SAVE_STATUS.txt").stat().st_mtime:
            newest[key] = run

    expected = len(yaml_index) * len(args.seeds)
    if len(newest) != expected:
        message = f"Found {len(newest)} completed cells; expected {expected}"
        if not args.allow_incomplete:
            raise SystemExit(message)
        print(message + "; analyzing available results only")
    if not newest:
        raise SystemExit("No completed runs found")

    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for (collision_control, controller_path, seed), run in sorted(newest.items()):
        supervisor = Path(controller_path).stem
        prompt, generation, collision_control, yaml_path = yaml_index[
            (collision_control, controller_path)
        ]
        success, duration = read_result(run)
        row = {
            "task": args.task,
            "prompt": prompt,
            "prompt_text": prompt_texts[prompt],
            "generation": generation,
            "collision_control": collision_control,
            "seed": seed,
            "supervisor": supervisor,
            "success": success,
            "task_success_pct": task_success_pct(run, args.task, success),
            "duration_s": duration,
            "coverage_pct": coverage_at(run, duration),
            "collisions": collisions(run),
            "yaml": str(yaml_path),
            "run": str(run),
        }
        if args.task == "delivery":
            row.update(delivery_counts(run))
        rows.append(row)
        for event, percentage in event_percentages(run).items():
            event_rows.append(
                {
                    "prompt": prompt,
                    "generation": generation,
                    "collision_control": collision_control,
                    "seed": seed,
                    "supervisor": supervisor,
                    "event": event,
                    "percentage": percentage,
                }
            )
    rows.sort(key=lambda row: (
        int(row["prompt"]), str(row["collision_control"]),
        int(row["generation"]), int(row["seed"]),
    ))
    write_csv(output / f"{args.task}_prompt_sensitivity_runs.csv", rows)
    write_csv(output / f"{args.task}_prompt_sensitivity_events.csv", event_rows)

    if args.task == "delivery":
        from delivery_timestep_analysis import analyze_timesteps
        analyze_timesteps(rows, output, root / "new_results" / "llm_run_logs")

    generation_summary = []
    prompt_summary = []
    preferred_modes = (
        "without_fixed_spec", "with_fixed_spec", "llm_collision", "fixed_collision"
    )
    present_modes = {str(row["collision_control"]) for row in rows}
    collision_controls = [mode for mode in preferred_modes if mode in present_modes]
    for prompt in args.prompts:
        for collision_control in collision_controls:
            prompt_group = [
                row for row in rows
                if row["prompt"] == prompt
                and row["collision_control"] == collision_control
            ]
            if not prompt_group:
                continue
            prompt_summary.append(
                summarize(
                    prompt_group,
                    {
                        "prompt": prompt,
                        "collision_control": collision_control,
                        "prompt_text": prompt_texts[prompt],
                    },
                )
            )
            for generation in args.generations:
                group = [
                    row for row in prompt_group if row["generation"] == generation
                ]
                if not group:
                    continue
                generation_summary.append(
                    summarize(
                        group,
                        {
                            "prompt": prompt,
                            "collision_control": collision_control,
                            "prompt_text": prompt_texts[prompt],
                            "generation": generation,
                        },
                    )
                )
    write_csv(output / f"{args.task}_prompt_summary.csv", prompt_summary)
    write_csv(output / f"{args.task}_generation_summary.csv", generation_summary)

    baseline_rows = []
    if not args.no_baseline:
        for seed in args.seeds:
            run = newest_completed_run_for_seed(
                root / "new_results" / "baseline" / f"results_{args.task}", seed
            )
            success, duration = read_result(run)
            baseline_rows.append(
                {
                    "seed": seed,
                    "success": success,
                    "task_success_pct": task_success_pct(
                        run, args.task, success
                    ),
                    "duration_s": duration,
                    "coverage_pct": coverage_at(run, duration),
                    "collisions": collisions(run),
                    "run": str(run),
                }
            )
        baseline_summary = summarize(baseline_rows, {"prompt": "baseline"})
        write_csv(output / f"{args.task}_baseline_summary.csv", [baseline_summary])

    labels = [
        f"P{prompt} {'With fixed spec' if mode in {'fixed_collision', 'with_fixed_spec'} else 'Without fixed spec'}"
        for prompt in args.prompts for mode in collision_controls
    ]
    groups = [
        [
            row for row in rows
            if row["prompt"] == prompt and row["collision_control"] == mode
        ]
        for prompt in args.prompts for mode in collision_controls
    ]
    prompt_labels = [f"P{prompt}" for prompt in args.prompts]
    prompt_centers = list(range(1, len(args.prompts) + 1))
    mode_spacing = 0.32
    group_positions = [
        prompt_index
        + (mode_index - (len(collision_controls) - 1) / 2) * mode_spacing
        for prompt_index in prompt_centers
        for mode_index in range(len(collision_controls))
    ]
    if baseline_rows:
        labels.insert(0, "Baseline")
        groups.insert(0, baseline_rows)
    if args.task == "exploration":
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes = axes.ravel()
        plot_labels = labels[1:] if baseline_rows else labels
        plot_groups = groups[1:] if baseline_rows else groups
        group_colors = []
        group_colors.extend(
            "#7B2CBF" if mode in {"fixed_collision", "with_fixed_spec"} else "#00A896"
            for _prompt in args.prompts
            for mode in collision_controls
        )
        for axis, field, title, ylabel in (
            (axes[0], "coverage_pct", "Coverage distribution", "Coverage (%)"),
            (
                axes[1], "collisions", "Collision distribution", "Collisions",
            ),
        ):
            values = [[float(row[field]) for row in group] for group in plot_groups]
            boxes = axis.boxplot(
                values, positions=group_positions, widths=0.26,
                showmeans=True, patch_artist=True
            )
            for box, color in zip(boxes["boxes"], group_colors):
                box.set_facecolor(color)
                box.set_alpha(0.55)
            for position, group_values, color in zip(
                group_positions, values, group_colors
            ):
                axis.scatter(
                    [position] * len(group_values), group_values,
                    s=15, alpha=0.55, color=color,
                )
            axis.set_xticks(prompt_centers, prompt_labels)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.set_ylim(bottom=0)
            axis.tick_params(
                axis="x", labelrotation=45, labelsize=FIGURE_FONT_SIZE
            )
            axis.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE)
            axis.xaxis.label.set_size(FIGURE_FONT_SIZE)
            axis.yaxis.label.set_size(FIGURE_FONT_SIZE)
            axis.title.set_size(FIGURE_FONT_SIZE)
            if baseline_rows:
                baseline_mean = statistics.mean(
                    float(row[field]) for row in baseline_rows
                )
                axis.axhline(
                    baseline_mean,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"Baseline mean ({baseline_mean:.1f})",
                )
        for axis in axes:
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        legend_handles = []
        if any(mode in {"fixed_collision", "with_fixed_spec"} for mode in collision_controls):
            legend_handles.append(
                Patch(facecolor="#7B2CBF", alpha=0.55, label="LLM + Spec_col")
            )
        if any(mode in {"llm_collision", "without_fixed_spec"} for mode in collision_controls):
            legend_handles.append(Patch(facecolor="#00A896", alpha=0.55, label="LLM"))
        if baseline_rows:
            legend_handles.append(
                Line2D(
                    [0], [0], color="#333333", linestyle="--", linewidth=1.8,
                    label="Baseline mean",
                )
            )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=FIGURE_FONT_SIZE,
        )
        title_artist = figure.suptitle(
            f"Exploration prompt sensitivity: {len(args.prompts)} prompts $\\times$ "
            f"{len(args.generations)} generations $\\times$ {len(args.seeds)} seeds",
            fontsize=FIGURE_FONT_SIZE,
        )
        figure.tight_layout(rect=(0, 0.12, 1, 0.93))
        figure.savefig(
            output / "exploration_prompt_sensitivity.png",
            dpi=200,
            bbox_inches="tight",
        )
        title_artist.set_visible(False)
        for axis in axes:
            axis.title.set_visible(False)
        figure.tight_layout(rect=(0, 0.12, 1, 0.98))
        figure.savefig(
            output / "exploration_prompt_sensitivity_no_title.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Analyzed {len(rows)} LLM runs" + (
            f" and {len(baseline_rows)} baseline runs" if baseline_rows else ""
        ))
        print(f"Saved outputs under {output}")
        return 0

    is_delivery = args.task == "delivery"
    if is_delivery:
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes.ravel()[-1].set_visible(False)
        all_axes = axes.ravel()[:5]
    else:
        figure, axes = plt.subplots(1, 4, figsize=(23, 5.5))
        all_axes = axes.ravel()
    panel_axes = all_axes[1:] if is_delivery else all_axes
    plot_labels = labels[1:] if baseline_rows else labels
    plot_groups = groups[1:] if baseline_rows else groups
    task_scores = [
        statistics.mean(float(row["task_success_pct"]) for row in group) if group else float("nan")
        for group in plot_groups
    ]
    group_colors = [
        "#7B2CBF" if mode in {"fixed_collision", "with_fixed_spec"} else "#00A896"
        for _prompt in args.prompts
        for mode in collision_controls
    ]
    panel_axes[0].bar(
        group_positions, task_scores, width=0.28,
        color=group_colors, alpha=0.7,
    )
    panel_axes[0].set_xticks(prompt_centers, prompt_labels)
    panel_axes[0].set_ylabel("Mean task completion (%)")
    panel_axes[0].set_ylim(0, 105)
    panel_axes[0].set_title("Task completion score")
    if baseline_rows:
        panel_axes[0].axhline(
            statistics.mean(
                float(row["task_success_pct"]) for row in baseline_rows
            ),
            color="#333333", linestyle="--", linewidth=1.8,
        )

    distributions = [
        (panel_axes[1], "duration_s", "Completion time / timeout", "Seconds"),
        (panel_axes[2], "coverage_pct", "Coverage distribution", "Coverage (%)"),
        (
            panel_axes[3], "collisions", "Collision distribution", "Collisions",
        ),
    ]
    if is_delivery:
        distributions.insert(0, (all_axes[0], "delivered_boxes", "Delivered boxes per run", "Delivered boxes (of 6)"))
    for axis, field, title, ylabel in distributions:
        values = [[float(row[field]) for row in group] for group in plot_groups]
        boxes = axis.boxplot(
            values, positions=group_positions, widths=0.26,
            showmeans=True, patch_artist=True
        )
        for box, color in zip(boxes["boxes"], group_colors):
            box.set_facecolor(color)
            box.set_alpha(0.55)
        for position, group_values, color in zip(
            group_positions, values, group_colors
        ):
            axis.scatter(
                [position] * len(group_values), group_values,
                s=15, alpha=0.55, color=color,
            )
        axis.set_xticks(prompt_centers, prompt_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if field == "collisions":
            # Logarithmic above one collision, with zero retained on the axis.
            axis.set_yscale("symlog", linthresh=1, linscale=0.5)
            maximum = max((value for group in values for value in group), default=1)
            ticks = [0, 1]
            power = 10
            while power <= max(10, maximum):
                ticks.append(power)
                power *= 10
            axis.set_yticks(ticks, [str(tick) for tick in ticks])
            axis.set_ylabel("Collisions (log scale)")
        if field == "delivered_boxes":
            axis.set_ylim(0, 6.5)
            axis.set_yticks(range(7))
        if baseline_rows:
            axis.axhline(
                statistics.mean(float(row[field]) for row in baseline_rows),
                color="#333333", linestyle="--", linewidth=1.8,
            )

    for axis in all_axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(bottom=0)
        axis.tick_params(
            axis="x", labelrotation=45, labelsize=FIGURE_FONT_SIZE
        )
        axis.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE)
        axis.xaxis.label.set_size(FIGURE_FONT_SIZE)
        axis.yaxis.label.set_size(FIGURE_FONT_SIZE)
        axis.title.set_size(FIGURE_FONT_SIZE)
    legend_handles = []
    if any(mode in {"fixed_collision", "with_fixed_spec"} for mode in collision_controls):
        legend_handles.append(
            Patch(facecolor="#7B2CBF", alpha=0.55, label="LLM + Spec_col")
        )
    if any(mode in {"llm_collision", "without_fixed_spec"} for mode in collision_controls):
        legend_handles.append(Patch(facecolor="#00A896", alpha=0.55, label="LLM"))
    if baseline_rows:
        legend_handles.append(
            Line2D(
                [0], [0], color="#333333", linestyle="--", linewidth=1.8,
                label="Baseline mean",
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FIGURE_FONT_SIZE,
    )
    title_artist = figure.suptitle(
        (f"{args.task.capitalize()} prompt sensitivity: {len(rows)} completed LLM runs"
         if args.allow_incomplete else
         f"{args.task.capitalize()} prompt sensitivity: "
        f"{len(args.prompts)} prompts × {len(args.generations)} generations × "
        f"{len(args.seeds)} seeds"),
        fontsize=FIGURE_FONT_SIZE,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.93))
    figure.savefig(
        output / f"{args.task}_prompt_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    title_artist.set_visible(False)
    for axis in all_axes:
        axis.title.set_visible(False)
    figure.tight_layout(rect=(0, 0.1, 1, 0.98))
    figure.savefig(
        output / f"{args.task}_prompt_sensitivity_no_title.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Analyzed {len(rows)} LLM runs" + (
        f" and {len(baseline_rows)} baseline runs" if baseline_rows else ""
    ))
    print(f"Saved outputs under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
