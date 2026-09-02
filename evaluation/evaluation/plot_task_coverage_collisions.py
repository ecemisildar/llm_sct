#!/usr/bin/env python3
"""Create aggregate coverage maps and collision plots as standalone SVG files."""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_text(x: float, y: float, value: object, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {properties}>{esc(value)}</text>'


def coverage_svg(task: str, runs: list[dict[str, str]], output: Path) -> None:
    visits: Counter[tuple[float, float]] = Counter()
    cell_size = 1.0
    for row in runs:
        path = Path(row["run"]) / "coverage_visited_cells.csv"
        run_cells: set[tuple[float, float]] = set()
        with path.open(newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                minimum_x = float(item["cell_min_x"])
                minimum_y = float(item["cell_min_y"])
                center_x = float(item["cell_center_x"])
                cell_size = max(cell_size, 2.0 * (center_x - minimum_x))
                run_cells.add((minimum_x, minimum_y))
        visits.update(run_cells)
    if not visits:
        raise ValueError(f"No visited cells found for {task}")

    xs = [cell[0] for cell in visits]
    ys = [cell[1] for cell in visits]
    min_x, max_x = min(xs), max(xs) + cell_size
    min_y, max_y = min(ys), max(ys) + cell_size
    width, height, margin = 900.0, 720.0, 90.0
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    def px(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * plot_w

    def py(y: float) -> float:
        return margin + (max_y - y) / (max_y - min_y) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, f"{task.capitalize()} aggregate coverage — seed 1001", text_anchor="middle", font_size="22", font_family="sans-serif", font_weight="bold"),
        svg_text(width / 2, 58, f"Cell color = number of runs visiting the cell (n={len(runs)})", text_anchor="middle", font_size="14", font_family="sans-serif", fill="#444"),
    ]
    for (x, y), count in sorted(visits.items()):
        ratio = count / len(runs)
        red = int(242 - 205 * ratio)
        green = int(247 - 137 * ratio)
        blue = int(255 - 25 * ratio)
        elements.append(
            f'<rect x="{px(x):.2f}" y="{py(y + cell_size):.2f}" '
            f'width="{plot_w * cell_size / (max_x - min_x):.2f}" '
            f'height="{plot_h * cell_size / (max_y - min_y):.2f}" '
            f'fill="rgb({red},{green},{blue})" stroke="#d0d7de" stroke-width="0.7">'
            f'<title>cell ({x:.1f}, {y:.1f}): {count}/{len(runs)} runs</title></rect>'
        )
        elements.append(
            svg_text(px(x + cell_size / 2), py(y + cell_size / 2) + 4, count, text_anchor="middle", font_size="11", font_family="sans-serif", fill="#111")
        )
    elements.extend(
        [
            f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.5"/>',
            svg_text(width / 2, height - 22, "World X (m)", text_anchor="middle", font_size="14", font_family="sans-serif"),
            svg_text(22, height / 2, "World Y (m)", text_anchor="middle", font_size="14", font_family="sans-serif", transform=f"rotate(-90 22 {height/2:.1f})"),
            "</svg>",
        ]
    )
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def collisions_svg(task: str, runs: list[dict[str, str]], output: Path) -> None:
    runs = sorted(runs, key=lambda row: (int(row["prompt"]), int(row["generation"])))
    counts = [int(row["collisions"]) for row in runs]
    width, height = 1100.0, 620.0
    left, right, top, bottom = 80.0, 30.0, 80.0, 120.0
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(1, max(counts, default=0))
    slot = plot_w / len(runs)
    bar_w = slot * 0.68
    colors = ["#0969da", "#1a7f37", "#8250df", "#bf8700", "#cf222e"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 35, f"{task.capitalize()} collisions by prompt and generation — seed 1001", text_anchor="middle", font_size="22", font_family="sans-serif", font_weight="bold"),
        svg_text(width / 2, 59, f"Total collisions: {sum(counts)} across {len(runs)} runs", text_anchor="middle", font_size="14", font_family="sans-serif", fill="#444"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(maximum + 1):
        y = top + plot_h - tick / maximum * plot_h
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eaeef2"/>')
        elements.append(svg_text(left - 12, y + 4, tick, text_anchor="end", font_size="12", font_family="sans-serif"))
    for index, (row, count) in enumerate(zip(runs, counts)):
        x = left + index * slot + (slot - bar_w) / 2
        bar_h = count / maximum * plot_h
        y = top + plot_h - bar_h
        prompt = int(row["prompt"])
        generation = int(row["generation"])
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[prompt - 1]}"><title>Prompt {prompt}, generation {generation}: {count} collisions</title></rect>'
        )
        elements.append(svg_text(x + bar_w / 2, max(top + 14, y - 7), count, text_anchor="middle", font_size="12", font_family="sans-serif", font_weight="bold"))
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 22, f"P{prompt}G{generation}", text_anchor="middle", font_size="10", font_family="sans-serif"))
    elements.extend(
        [
            svg_text(22, top + plot_h / 2, "Collision count", text_anchor="middle", font_size="14", font_family="sans-serif", transform=f"rotate(-90 22 {top + plot_h/2:.1f})"),
            "</svg>",
        ]
    )
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("delivery", "patrolling"), required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.analysis_dir.resolve()
    with (directory / f"{args.task}_runs.csv").open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    coverage_svg(args.task, runs, directory / f"{args.task}_coverage_map.svg")
    collisions_svg(args.task, runs, directory / f"{args.task}_collisions.svg")
    print(directory / f"{args.task}_coverage_map.svg")
    print(directory / f"{args.task}_collisions.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
