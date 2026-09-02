#!/usr/bin/env python3
"""Analyze completed LLM task runs without plotting dependencies."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter
from pathlib import Path


def status_value(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.*?)\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def read_last_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows[-1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    durations = [float(row["duration_s"]) for row in rows]
    successful = [
        float(row["duration_s"]) for row in rows if bool(row["success"])
    ]
    successes = len(successful)
    return {
        "runs": len(rows),
        "successes": successes,
        "success_rate_pct": 100.0 * successes / len(rows),
        "duration_all_mean_s": statistics.mean(durations),
        "duration_all_median_s": statistics.median(durations),
        "duration_success_mean_s": statistics.mean(successful) if successful else "",
        "duration_success_median_s": statistics.median(successful) if successful else "",
        "duration_min_s": min(durations),
        "duration_max_s": max(durations),
        "progress_mean_pct": statistics.mean(
            float(row["progress_pct"]) for row in rows
        ),
        "collisions_total": sum(int(row["collisions"]) for row in rows),
        "selected_events_total": sum(int(row["selected_events"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("delivery", "patrolling"), required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    yaml_root = root / "automata" / "resulting_automata" / "YAML" / args.task
    yaml_index: dict[str, tuple[int, int, Path]] = {}
    for yaml_path in yaml_root.glob("prompt_*/generation_*/S_*.yaml"):
        yaml_index[yaml_path.stem] = (
            int(yaml_path.parents[1].name.removeprefix("prompt_")),
            int(yaml_path.parent.name.removeprefix("generation_")),
            yaml_path,
        )

    newest: dict[str, tuple[float, Path]] = {}
    result_root = root / "new_results" / "llm" / f"results_{args.task}"
    for status_path in result_root.glob("S_*/robots_*/run_*/SAVE_STATUS.txt"):
        text = status_path.read_text(encoding="utf-8", errors="replace")
        if "Saving OK" not in text:
            continue
        if status_value(text, "random_seed") != str(args.seed):
            continue
        if status_value(text, "run_duration") != str(args.duration):
            continue
        supervisor = status_path.parents[2].name
        if supervisor not in yaml_index:
            continue
        modified = status_path.stat().st_mtime
        if supervisor not in newest or modified > newest[supervisor][0]:
            newest[supervisor] = (modified, status_path.parent)

    expected = len(yaml_index)
    if len(newest) != expected:
        raise SystemExit(f"Found {len(newest)} completed runs; expected {expected}")

    rows: list[dict[str, object]] = []
    event_totals: Counter[str] = Counter()
    for supervisor, (_, run) in newest.items():
        prompt, generation, yaml_path = yaml_index[supervisor]
        result = read_last_csv(run / "task_result.csv")
        collisions = 0
        bump_path = run / "bumps_global.csv"
        if bump_path.is_file():
            with bump_path.open(newline="", encoding="utf-8") as stream:
                collisions = sum(1 for _ in csv.DictReader(stream))
        selected = Counter()
        for event_path in run.glob("selected_events_robot_*.csv"):
            with event_path.open(newline="", encoding="utf-8") as stream:
                selected.update(
                    row["selected_event"] for row in csv.DictReader(stream)
                )
        event_totals.update(selected)
        rows.append(
            {
                "task": args.task,
                "seed": args.seed,
                "prompt": prompt,
                "generation": generation,
                "supervisor": supervisor,
                "success": result["success"].casefold() == "true",
                "duration_s": float(result["duration_s"]),
                "progress_pct": float(result["progress_pct"]),
                "collisions": collisions,
                "selected_events": sum(selected.values()),
                "yaml": str(yaml_path),
                "run": str(run),
            }
        )
    rows.sort(key=lambda row: (int(row["prompt"]), int(row["generation"])))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / f"{args.task}_runs.csv", rows)

    prompt_rows: list[dict[str, object]] = []
    for prompt in sorted({int(row["prompt"]) for row in rows}):
        group = [row for row in rows if row["prompt"] == prompt]
        prompt_rows.append({"prompt": prompt, **summarize(group)})
    write_csv(output / f"{args.task}_by_prompt.csv", prompt_rows)

    generation_rows: list[dict[str, object]] = []
    for generation in sorted({int(row["generation"]) for row in rows}):
        group = [row for row in rows if row["generation"] == generation]
        generation_rows.append({"generation": generation, **summarize(group)})
    write_csv(output / f"{args.task}_by_generation.csv", generation_rows)

    overall = summarize(rows)
    event_rows = [
        {
            "event": event,
            "count": count,
            "percentage": 100.0 * count / sum(event_totals.values()),
        }
        for event, count in event_totals.most_common()
    ]
    write_csv(output / f"{args.task}_events.csv", event_rows)

    report = [
        f"# {args.task.capitalize()} analysis — seed {args.seed}",
        "",
        f"- Runs: {overall['runs']}",
        f"- Successes: {overall['successes']}",
        f"- Success rate: {overall['success_rate_pct']:.1f}%",
        f"- Mean duration (all): {overall['duration_all_mean_s']:.3f} s",
        f"- Median duration (all): {overall['duration_all_median_s']:.3f} s",
        f"- Mean successful duration: {overall['duration_success_mean_s']:.3f} s"
        if overall["duration_success_mean_s"] != ""
        else "- Mean successful duration: N/A",
        f"- Mean progress: {overall['progress_mean_pct']:.3f}%",
        f"- Collisions: {overall['collisions_total']}",
        "",
        "## By prompt",
        "",
        "| Prompt | Successes | Rate | Mean successful duration |",
        "|---:|---:|---:|---:|",
    ]
    for row in prompt_rows:
        mean = row["duration_success_mean_s"]
        mean_text = f"{mean:.3f} s" if mean != "" else "N/A"
        report.append(
            f"| {row['prompt']} | {row['successes']}/{row['runs']} | "
            f"{row['success_rate_pct']:.1f}% | {mean_text} |"
        )
    report.extend(["", "## Individual runs", "", "| Prompt | Generation | Success | Duration | Progress |", "|---:|---:|:---:|---:|---:|"])
    for row in rows:
        report.append(
            f"| {row['prompt']} | {row['generation']} | "
            f"{'yes' if row['success'] else 'no'} | {row['duration_s']:.3f} s | "
            f"{row['progress_pct']:.3f}% |"
        )
    (output / f"{args.task}_analysis.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(output / f"{args.task}_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
