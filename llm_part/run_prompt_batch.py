#!/usr/bin/env python3
"""Generate supervisor candidates for each prompt and collision-control mode."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from llm_input import DEFAULT_MODEL, read_required_text
from run_pipeline import run_pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_PATH = SCRIPT_DIR / "prompts.txt"
PROMPT_SECTIONS = ("exploration", "patrolling", "delivery", "payload_delivery")
MISSIONS = ("exploration", "patrolling", "delivery")
PROMPTS_PER_MISSION = 6
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "RESULTS_LAST"


def parse_prompts(
    path: Path, required_missions: tuple[str, ...] = MISSIONS
) -> dict[str, list[str]]:
    grouped = {mission: [] for mission in PROMPT_SECTIONS}
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
    for mission in required_missions:
        prompts = grouped[mission]
        if len(prompts) != PROMPTS_PER_MISSION:
            raise ValueError(
                f"Expected {PROMPTS_PER_MISSION} {mission} prompts, "
                f"found {len(prompts)}"
            )
    return grouped


def already_generated(
    mission: str,
    prompt: int,
    generation: int,
    task: str,
    collision_control: str,
    output_root: Path | None = None,
) -> bool:
    yaml_root = (
        output_root / "YAML"
        if output_root is not None
        else SCRIPT_DIR.parent / "automata" / "resulting_automata" / "YAML"
    )
    output_dir = (
        yaml_root
        / mission
        / ("with_fixed_spec" if collision_control == "fixed" else "without_fixed_spec")
        / f"prompt_{prompt}"
        / f"generation_{generation}"
    )
    return any(
        yaml_path.with_suffix(".prompt.txt").is_file()
        and yaml_path.with_suffix(".collision_control.txt").is_file()
        and yaml_path.with_suffix(".collision_control.txt").read_text(
            encoding="utf-8"
        ).strip() == collision_control
        and yaml_path.with_suffix(".prompt.txt").read_text(encoding="utf-8").strip()
        == task
        for yaml_path in output_dir.glob("S_*.yaml")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Candidates per prompt (default: 5; 30 supervisors per mission)",
    )
    parser.add_argument(
        "--prompt-number",
        type=int,
        choices=range(1, PROMPTS_PER_MISSION + 1),
        metavar="{1,2,3,4,5,6}",
        help=(
            "Run only this prompt number instead of all "
            f"{PROMPTS_PER_MISSION} prompts"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plant-dir", type=Path, help="Use XML plants from this directory")
    parser.add_argument("--fixed-spec", type=Path, action="append", help="Fixed specification XML (repeatable)")
    parser.add_argument("--collision-control", choices=("none", "fixed"), help="Synthesize only this variant")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Store all generated artifacts under this directory "
            f"(default: {DEFAULT_OUTPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--mission", choices=MISSIONS, action="append", dest="missions",
        help="Run only this mission (repeat to select more than one)",
    )
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")

    selected_missions = tuple(args.missions or ("exploration", "patrolling"))
    prompts = parse_prompts(
        args.prompts.expanduser().resolve(), selected_missions
    )
    output_root = args.output_root.expanduser().resolve()
    collision_controls = (args.collision_control,) if args.collision_control else ("none", "fixed")
    plant_paths = sorted(args.plant_dir.expanduser().resolve().glob("*.xml")) if args.plant_dir else None
    if plant_paths == []:
        parser.error("--plant-dir contains no XML plants")
    specification_paths = [path.expanduser().resolve() for path in args.fixed_spec] if args.fixed_spec else None
    prompt_count = 1 if args.prompt_number is not None else PROMPTS_PER_MISSION
    total = (
        len(selected_missions)
        * prompt_count
        * args.generations
    )
    completed = 0
    for mission in selected_missions:
        numbered_prompts = list(enumerate(prompts[mission], start=1))
        if args.prompt_number is not None:
            numbered_prompts = [numbered_prompts[args.prompt_number - 1]]
        for prompt_number, task in numbered_prompts:
            for generation_number in range(1, args.generations + 1):
                completed += 1
                generated_json_path = (
                    output_root
                    / "llm_outputs"
                    / mission
                    / ("without_fixed_spec" if "none" in collision_controls else "with_fixed_spec")
                    / f"prompt_{prompt_number}"
                    / f"generation_{generation_number}"
                    / "llm_output.json"
                )
                for collision_control in collision_controls:
                    variant = (
                        "with_fixed_spec"
                        if collision_control == "fixed"
                        else "without_fixed_spec"
                    )
                    label = (
                        f"[{completed}/{total}] {mission} prompt {prompt_number}, "
                        f"generation {generation_number}, {variant}"
                    )
                    if args.resume and already_generated(
                        mission,
                        prompt_number,
                        generation_number,
                        task,
                        collision_control,
                        output_root,
                    ):
                        print(f"{label}: already complete", flush=True)
                        continue
                    print(f"{label}: starting", flush=True)
                    reuse_json_path = None
                    if (collision_control == "fixed" and "none" in collision_controls) or (
                        args.resume
                        and generated_json_path.is_file()
                        and generated_json_path.with_suffix(".prompt.txt").is_file()
                        and generated_json_path.with_suffix(".prompt.txt").read_text(
                            encoding="utf-8"
                        ).strip() == task
                    ):
                        reuse_json_path = generated_json_path
                    if args.resume and reuse_json_path is None:
                        # Either variant uses the same API input; only synthesis
                        # adds or removes the fixed collision specification.
                        other_variant = (
                            "with_fixed_spec" if collision_control == "none"
                            else "without_fixed_spec"
                        )
                        other_dir = (
                            output_root / "YAML" / mission / other_variant
                            / f"prompt_{prompt_number}"
                            / f"generation_{generation_number}"
                        )
                        for candidate in sorted(
                            other_dir.glob("S_*.llm_output.json"),
                            key=lambda path: path.stat().st_mtime,
                            reverse=True,
                        ):
                            saved_prompt = candidate.with_name(
                                candidate.name.replace(".llm_output.json", ".prompt.txt")
                            )
                            if saved_prompt.is_file() and saved_prompt.read_text(
                                encoding="utf-8"
                            ).strip() == task:
                                reuse_json_path = candidate
                                break
                    result = run_pipeline(
                        task=task,
                        model=args.model,
                        mission=mission,
                        prompt_number=prompt_number,
                        generation_number=generation_number,
                        collision_control=collision_control,
                        output_root=output_root,
                        reuse_json_path=reuse_json_path,
                        plant_paths=plant_paths,
                        specification_paths=specification_paths,
                        status=lambda message, prefix=label: print(
                            f"{prefix}: {message}", flush=True
                        ),
                    )
                    print(f"{label}: complete: {result.yaml_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
