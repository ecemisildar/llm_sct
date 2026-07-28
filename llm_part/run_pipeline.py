#!/usr/bin/env python3
"""Run the complete task-to-runtime-YAML SCT generation pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import llm_json_to_xml
import nadzoru_sync
import supervisor_xml_to_yaml
from llm_input import (
    DEFAULT_API_KEY_PATH,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_PATH,
    all_events,
    generate_json,
    read_required_text,
)


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class PipelineResult:
    json_path: Path
    generated_xml: tuple[Path, ...]
    g_xml: Path
    k_xml: Path
    s_xml: Path
    yaml_path: Path
    payload: object


def select_profile(
    task: str, mission: str = "auto", exploration_mode: str = "auto"
) -> tuple[str, list[Path], list[Path]]:
    """Return mission name, fixed plants, and fixed specifications."""
    text = task.casefold()
    if mission == "auto":
        if "delivery" in text or "deliver" in text:
            mission = "delivery"
        elif any(
            keyword in text
            for keyword in (
                "explor",
                "sweep",
                "coverage",
                "cover ",
                "covering",
                "spiral",
                "roam",
                "mapping",
                "map the environment",
            )
        ):
            mission = "exploration"
        else:
            mission = "patrolling"

    baseline = nadzoru_sync.AUTOMATA_DIR / "baseline_automata"
    shared_motion_plant = baseline / "shared" / "motion_plant.xml"
    shared_obstacle_sensor = baseline / "shared" / "obstacle_sensor.xml"
    shared_collision_avoidance = baseline / "shared" / "collision_avoidance.xml"
    if mission == "exploration":
        return (
            mission,
            [shared_obstacle_sensor, shared_motion_plant],
            [shared_collision_avoidance],
        )
    if mission == "delivery":
        directory = baseline / "delivery"
        plants = [
            directory / name
            for name in (
                "obstacle_sensor.xml",
                "color_sensor.xml",
                "motion_plant.xml",
                "comm_plant.xml",
                "red_availability.xml",
                "green_availability.xml",
                "blue_availability.xml",
            )
        ]
        return (
            mission,
            plants,
            [directory / "collision_avoidance.xml"],
        )

    directory = baseline / "patrolling"
    plants = [
        shared_obstacle_sensor,
        directory / "color_sensor.xml",
        shared_motion_plant,
    ]
    return (
        mission,
        plants,
        [shared_collision_avoidance],
    )


def run_pipeline(
    task: str,
    model: str = DEFAULT_MODEL,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    api_key_path: Path = DEFAULT_API_KEY_PATH,
    status: StatusCallback | None = None,
    mission: str = "auto",
    exploration_mode: str = "auto",
) -> PipelineResult:
    report = status or (lambda _message: None)
    selected_mission, fixed_plants, fixed_specs = select_profile(
        task, mission, exploration_mode
    )
    report(
        f"Selected {selected_mission} fixed automata: "
        + ", ".join(path.stem for path in [*fixed_plants, *fixed_specs])
    )
    allowed_events = all_events()

    report("Requesting JSON automata from OpenAI…")
    payload, json_path = generate_json(
        task=task,
        prompt_path=prompt_path,
        api_key_path=api_key_path,
        model=model,
        context_files=[*fixed_plants, *fixed_specs],
        context_mission=selected_mission,
    )

    report("Converting generated JSON to Nadzoru XML…")
    plant_events: set[str] = set()
    for path in fixed_plants:
        plant_events.update(nadzoru_sync.xml_event_names(path))
    complete_event_alphabet = {
        event: controllable
        for event, controllable in allowed_events.items()
        if event in plant_events
    }
    report(
        "Completing the generated specifications with the fixed-plant event "
        "alphabet; unused actions will be disabled."
    )
    generated_xml = tuple(
        llm_json_to_xml.convert(
            json_path,
            llm_json_to_xml.DEFAULT_OUTPUT_DIR,
            llm_json_to_xml.DEFAULT_BASELINE_DIR,
            allowed_generated_events=set(allowed_events),
            complete_event_alphabet=complete_event_alphabet,
        )
    )

    report("Synchronizing plants and specifications with Nadzoru…")
    sync_args = nadzoru_sync.build_parser().parse_args([])
    # G uses the three tested patrolling plants. K adds the tested collision
    # specification and only this request's generated specifications. Old LLM
    # XML files remain on disk but cannot affect this run.
    sync_args.input_dir = []
    sync_args.input_file = [
        *(str(path) for path in [*fixed_plants, *fixed_specs]),
        *(str(path) for path in generated_xml),
    ]
    sync_args.plant = [path.stem for path in fixed_plants]
    sync_args.spec = [path.stem for path in fixed_specs]
    g_xml, k_xml, s_xml = nadzoru_sync.run(sync_args)

    report("Encoding the synthesized supervisor as runtime YAML…")
    yaml_path = supervisor_xml_to_yaml.DEFAULT_YAML_DIR / s_xml.with_suffix(".yaml").name
    supervisor_xml_to_yaml.convert(s_xml, yaml_path)
    source_prompt_path = json_path.with_suffix(".prompt.txt")
    yaml_prompt_path = yaml_path.with_suffix(".prompt.txt")
    if source_prompt_path.is_file():
        shutil.copy2(source_prompt_path, yaml_prompt_path)
        report(f"Saved matching UI prompt: {yaml_prompt_path}")
    report(f"Pipeline complete: {yaml_path}")

    return PipelineResult(
        json_path=json_path,
        generated_xml=generated_xml,
        g_xml=g_xml,
        k_xml=k_xml,
        s_xml=s_xml,
        yaml_path=yaml_path,
        payload=payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LLM automata and synthesize runtime supervisor YAML."
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Control task for the LLM")
    task_group.add_argument("--task-file", help="Text file containing the control task")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mission", choices=("auto", "exploration", "patrolling", "delivery"),
        default="auto",
    )
    parser.add_argument(
        "--exploration-mode",
        choices=("auto", "with_backward", "without_backward"), default="auto",
    )
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--api-key-file", default=str(DEFAULT_API_KEY_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task = (
            read_required_text(Path(args.task_file).expanduser().resolve(), "Task")
            if args.task_file
            else args.task
        )
        result = run_pipeline(
            task=task,
            model=args.model,
            prompt_path=Path(args.prompt_file).expanduser().resolve(),
            api_key_path=Path(args.api_key_file).expanduser().resolve(),
            mission=args.mission,
            exploration_mode=args.exploration_mode,
            status=print,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "json": str(result.json_path),
        "generated_xml": [str(path) for path in result.generated_xml],
        "G": str(result.g_xml),
        "K": str(result.k_xml),
        "S": str(result.s_xml),
        "yaml": str(result.yaml_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
