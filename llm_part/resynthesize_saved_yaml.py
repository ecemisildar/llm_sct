#!/usr/bin/env python3
"""Resynthesize saved LLM YAMLs after fixed-automata changes, without an API call."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import llm_json_to_xml
import nadzoru_sync
import supervisor_xml_to_yaml
from llm_input import events_available_in_automata, events_for_mission
from run_pipeline import select_profile


def resynthesize(yaml_path: Path, mission: str) -> None:
    yaml_path = yaml_path.expanduser().resolve()
    json_path = yaml_path.with_suffix(".llm_output.json")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML does not exist: {yaml_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Saved LLM output does not exist: {json_path}")

    _, fixed_plants, fixed_specs = select_profile(mission, mission=mission)
    allowed_events = events_available_in_automata(
        events_for_mission(mission), fixed_plants
    )
    plant_events: set[str] = set()
    for path in fixed_plants:
        plant_events.update(nadzoru_sync.xml_event_names(path))
    complete_event_alphabet = {
        event: controllable
        for event, controllable in allowed_events.items()
        if event in plant_events
    }

    with tempfile.TemporaryDirectory(prefix=f"resynthesize_{yaml_path.stem}_") as temp:
        temp_dir = Path(temp)
        generated_xml = llm_json_to_xml.convert(
            json_path,
            temp_dir / "generated",
            fixed_plants[0].parent,
            allowed_generated_events=set(allowed_events),
            complete_event_alphabet=complete_event_alphabet,
        )
        args = nadzoru_sync.build_parser().parse_args([])
        args.input_dir = []
        args.input_file = [
            *(str(path) for path in [*fixed_plants, *fixed_specs]),
            *(str(path) for path in generated_xml),
        ]
        args.plant = [path.stem for path in fixed_plants]
        args.spec = [path.stem for path in fixed_specs]
        args.output_dir = str(temp_dir / "result")
        _, _, supervisor_xml = nadzoru_sync.run(args)
        supervisor_xml_to_yaml.convert(supervisor_xml, yaml_path)
    print(f"Resynthesized {yaml_path} for {mission}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mission", choices=("exploration", "patrolling", "delivery", "payload_delivery", "complex_task")
    )
    parser.add_argument("yaml", type=Path, nargs="+")
    args = parser.parse_args()
    for yaml_path in args.yaml:
        resynthesize(yaml_path, args.mission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
