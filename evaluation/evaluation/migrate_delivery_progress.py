#!/usr/bin/env python3
"""Rebuild delivery task metrics from saved selected-event logs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


COLORS = ("red", "green", "blue")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def migrate(run_dir: Path) -> None:
    deliveries: dict[str, tuple[str, float]] = {}
    final_elapsed = 0.0
    for path in sorted(run_dir.glob("selected_events_robot_*.csv")):
        robot = path.stem.removeprefix("selected_events_")
        for row in rows(path):
            elapsed = float(row["elapsed_s"])
            final_elapsed = max(final_elapsed, elapsed)
            event = row["selected_event"].removeprefix("EV_publish_delivered_")
            if event in COLORS and event not in deliveries:
                deliveries[event] = (robot, elapsed)

    task_result = run_dir / "task_result.csv"
    task_progress = run_dir / "task_progress.csv"
    for path in (task_result, task_progress):
        backup = path.with_name(f"{path.stem}.legacy{path.suffix}")
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)

    success = len(deliveries) == len(COLORS)
    completion = max((value[1] for value in deliveries.values()), default=None)
    duration = completion if success else final_elapsed
    contributors = {value[0] for value in deliveries.values()}
    robot_count = int(run_dir.parent.name.removeprefix("robots_"))
    with task_result.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "success",
                "duration_s",
                "completed_robots",
                "total_robots",
                "completed_targets",
                "total_targets",
                "progress_pct",
            ]
        )
        writer.writerow(
            [
                str(success).lower(),
                f"{duration:.3f}",
                len(contributors),
                robot_count,
                len(deliveries),
                len(COLORS),
                f"{100.0 * len(deliveries) / len(COLORS):.3f}",
            ]
        )

    with task_progress.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "robot",
                "delivered_colors",
                "delivered_count",
                "team_total_colors",
                "contribution_pct",
                "team_complete",
                "completion_s",
                "red_delivered_s",
                "green_delivered_s",
                "blue_delivered_s",
            ]
        )
        for index in range(robot_count):
            robot = f"robot_{index}"
            colors = [
                color for color, value in deliveries.items()
                if value[0] == robot
            ]
            writer.writerow(
                [
                    robot,
                    "|".join(colors),
                    len(colors),
                    len(COLORS),
                    f"{100.0 * len(colors) / len(COLORS):.3f}",
                    str(success).lower(),
                    f"{completion:.3f}" if completion is not None and success else "",
                    *[
                        f"{deliveries[color][1]:.3f}"
                        if color in deliveries and deliveries[color][0] == robot
                        else ""
                        for color in COLORS
                    ],
                ]
            )
    print(
        f"{run_dir}: {len(deliveries)}/{len(COLORS)} team deliveries, "
        f"success={success}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    migrate(args.run_dir.resolve())


if __name__ == "__main__":
    main()
