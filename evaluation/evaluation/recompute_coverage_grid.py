#!/usr/bin/env python3
"""Recompute saved coverage artifacts from recorded robot paths."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from evaluation.coverage_counter import CoverageCounter


def blocked_cells(world: Path, grid_size: float) -> tuple[set[int], int]:
    counter = CoverageCounter.__new__(CoverageCounter)
    counter.env_min = -5
    counter.env_max = 5
    counter.grid_size = grid_size
    counter.num_cells_y = int(
        (counter.env_max - counter.env_min) / counter.grid_size
    )
    counter.cells = [
        (counter.env_min + ix * grid_size, counter.env_min + iy * grid_size)
        for ix in range(counter.num_cells_y)
        for iy in range(counter.num_cells_y)
    ]
    counter.world_sdf = world
    counter.obstacle_occupancy_threshold = 0.4
    counter.circle_obstacle_occupancy_threshold = 0.05
    return counter._compute_blocked_cells(), counter.num_cells_y


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def recompute_run(
    run_dir: Path,
    grid_size: float,
    blocked: set[int],
    cells_per_axis: int,
) -> None:
    paths_path = run_dir / "coverage_paths.csv"
    visited_path = run_dir / "coverage_visited_cells.csv"
    timeseries_path = run_dir / "coverage_timeseries.csv"
    if not paths_path.is_file() or not timeseries_path.is_file():
        return

    paths = sorted(read_rows(paths_path), key=lambda row: float(row["elapsed_s"]))
    sample_times = [
        float(row["time_s"]) for row in read_rows(timeseries_path)
    ]
    free_cells = cells_per_axis * cells_per_axis - len(blocked)
    visited: set[int] = set()
    visits: list[list[object]] = []
    coverage: list[list[object]] = []
    path_index = 0

    def consume(row: dict[str, str]) -> None:
        x = float(row["x"])
        y = float(row["y"])
        ix = math.floor((x + 5.0) / grid_size)
        iy = math.floor((y + 5.0) / grid_size)
        if not (0 <= ix < cells_per_axis and 0 <= iy < cells_per_axis):
            return
        index = ix * cells_per_axis + iy
        if index in blocked or index in visited:
            return
        visited.add(index)
        cell_min_x = -5.0 + ix * grid_size
        cell_min_y = -5.0 + iy * grid_size
        visits.append(
            [
                row["stamp_sec"],
                row["stamp_nsec"],
                row["robot"],
                index,
                f"{cell_min_x:.3f}",
                f"{cell_min_y:.3f}",
                f"{cell_min_x + grid_size / 2.0:.3f}",
                f"{cell_min_y + grid_size / 2.0:.3f}",
            ]
        )

    for sample_time in sample_times:
        while (
            path_index < len(paths)
            and float(paths[path_index]["elapsed_s"]) <= sample_time
        ):
            consume(paths[path_index])
            path_index += 1
        coverage.append(
            [f"{sample_time:.3f}", f"{100.0 * len(visited) / free_cells:.3f}"]
        )

    while path_index < len(paths):
        consume(paths[path_index])
        path_index += 1
    if coverage:
        coverage[-1][1] = f"{100.0 * len(visited) / free_cells:.3f}"

    for path in (visited_path, timeseries_path):
        backup = path.with_name(f"{path.stem}.grid_1m{path.suffix}")
        if path.is_file() and not backup.exists():
            shutil.copy2(path, backup)

    with visited_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stamp_sec",
                "stamp_nsec",
                "robot",
                "cell_index",
                "cell_min_x",
                "cell_min_y",
                "cell_center_x",
                "cell_center_y",
            ]
        )
        writer.writerows(visits)
    with timeseries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "coverage_pct"])
        writer.writerows(coverage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("world", type=Path)
    parser.add_argument("--grid-size", type=float, default=1.0)
    args = parser.parse_args()
    blocked, cells_per_axis = blocked_cells(args.world, args.grid_size)
    runs = sorted({path.parent for path in args.results_root.rglob("coverage_paths.csv")})
    for run_dir in runs:
        recompute_run(
            run_dir, args.grid_size, blocked, cells_per_axis
        )
    print(
        f"Updated {len(runs)} runs at {args.grid_size:g} m resolution "
        f"({cells_per_axis ** 2 - len(blocked)} free cells)."
    )


if __name__ == "__main__":
    main()
