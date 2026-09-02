#!/usr/bin/env python3
"""Generate three supervisor candidates for five prompts in each mission."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from llm_input import DEFAULT_MODEL, read_required_text
from run_pipeline import run_pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = SCRIPT_DIR / "prompts.txt"
MISSIONS = ("exploration", "patrolling", "delivery", "complex_task")


def parse_prompts(path: Path) -> dict[str, list[str]]:
    grouped = {mission: [] for mission in MISSIONS}
    current: str | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        heading = line.casefold().replace(" ", "_")
        if heading in grouped:
            current = heading
            continue
        match = re.fullmatch(r"P(\d+):\s*(.+)", line, flags=re.IGNORECASE)
        if match is None or current is None:
            raise ValueError(f"Invalid prompt line {line_number}: {raw_line!r}")
        expected = len(grouped[current]) + 1
        if int(match.group(1)) != expected:
            raise ValueError(
                f"Expected P{expected} for {current} on line {line_number}"
            )
        grouped[current].append(match.group(2).strip())
    for mission, prompts in grouped.items():
        if len(prompts) != 5:
            raise ValueError(f"Expected 5 {mission} prompts, found {len(prompts)}")
    return grouped


def already_generated(mission: str, prompt: int, generation: int, task: str) -> bool:
    output_dir = (
        SCRIPT_DIR.parent
        / "automata"
        / "resulting_automata"
        / "YAML"
        / mission
        / f"prompt_{prompt}"
        / f"generation_{generation}"
    )
    return any(
        yaml_path.with_suffix(".prompt.txt").is_file()
        and yaml_path.with_suffix(".prompt.txt").read_text(encoding="utf-8").strip()
        == task
        for yaml_path in output_dir.glob("S_*.yaml")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument(
        "--prompt-number",
        type=int,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Run only this prompt number instead of all five prompts",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mission", choices=MISSIONS, action="append", dest="missions",
        help="Run only this mission (repeat to select more than one)",
    )
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")

    prompts = parse_prompts(args.prompts.expanduser().resolve())
    selected_missions = tuple(args.missions or MISSIONS)
    prompt_count = 1 if args.prompt_number is not None else 5
    total = len(selected_missions) * prompt_count * args.generations
    completed = 0
    for mission in selected_missions:
        numbered_prompts = list(enumerate(prompts[mission], start=1))
        if args.prompt_number is not None:
            numbered_prompts = [numbered_prompts[args.prompt_number - 1]]
        for prompt_number, task in numbered_prompts:
            for generation_number in range(1, args.generations + 1):
                completed += 1
                label = (
                    f"[{completed}/{total}] {mission} prompt {prompt_number}, "
                    f"generation {generation_number}"
                )
                if args.resume and already_generated(
                    mission, prompt_number, generation_number, task
                ):
                    print(f"{label}: already complete", flush=True)
                    continue
                print(f"{label}: starting", flush=True)
                result = run_pipeline(
                    task=task,
                    model=args.model,
                    mission=mission,
                    prompt_number=prompt_number,
                    generation_number=generation_number,
                    status=lambda message, prefix=label: print(
                        f"{prefix}: {message}", flush=True
                    ),
                )
                print(f"{label}: complete: {result.yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
