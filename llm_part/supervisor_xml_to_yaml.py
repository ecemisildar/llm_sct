#!/usr/bin/env python3
"""Convert the newest timestamped ``S_*.xml`` to runtime supervisor YAML."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR.parent / "automata" / "resulting_automata"
DEFAULT_SUPERVISOR_DIR = DEFAULT_RESULT_DIR / "S"
DEFAULT_YAML_DIR = DEFAULT_RESULT_DIR / "YAML"


def newest_supervisor_xml(directory: Path = DEFAULT_SUPERVISOR_DIR) -> Path:
    candidates = sorted(directory.glob("S_[0-9]*_[0-9]*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No timestamped S_*.xml file found in {directory}")
    return candidates[-1]


def required_attribute(element: ET.Element, attribute: str) -> str:
    value = element.get(attribute)
    if value is None:
        raise ValueError(f"<{element.tag}> is missing the '{attribute}' attribute")
    return value


def xml_to_supervisor_data(input_path: Path) -> dict[str, object]:
    try:
        data_element = ET.parse(input_path).getroot().find("data")
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML in {input_path}: {error}") from error
    if data_element is None:
        raise ValueError(f"{input_path} does not contain a <data> element")

    states = data_element.findall("state")
    events = data_element.findall("event")
    transitions = data_element.findall("transition")
    if not states:
        raise ValueError("The supervisor has no states")
    if not events:
        raise ValueError("The supervisor has no events")

    state_index: dict[str, int] = {}
    initial_states: list[int] = []
    for index, state in enumerate(states):
        state_id = required_attribute(state, "id")
        if state_id in state_index:
            raise ValueError(f"Duplicate state id: {state_id}")
        state_index[state_id] = index
        if required_attribute(state, "initial").casefold() == "true":
            initial_states.append(index)
    if len(initial_states) != 1:
        raise ValueError(f"Expected exactly one initial state, found {len(initial_states)}")
    if len(states) > 65536:
        raise ValueError("Runtime YAML format supports at most 65536 states")

    event_index: dict[str, int] = {}
    event_names: list[str] = []
    controllable: list[int] = []
    for index, event in enumerate(events):
        event_id = required_attribute(event, "id")
        if event_id in event_index:
            raise ValueError(f"Duplicate event id: {event_id}")
        event_index[event_id] = index
        event_names.append(f"EV_{required_attribute(event, 'name')}")
        controllable.append(
            1 if required_attribute(event, "controllable").casefold() == "true" else 0
        )

    outgoing: list[list[tuple[int, int]]] = [[] for _ in states]
    for transition in transitions:
        source_id = required_attribute(transition, "source")
        target_id = required_attribute(transition, "target")
        event_id = required_attribute(transition, "event")
        try:
            source = state_index[source_id]
            target = state_index[target_id]
            event = event_index[event_id]
        except KeyError as error:
            raise ValueError(
                f"Transition references unknown state/event id: {error.args[0]}"
            ) from error
        outgoing[source].append((event, target))

    supervisor_data: list[int | str] = []
    for state_transitions in outgoing:
        supervisor_data.append(len(state_transitions))
        for event, target in state_transitions:
            supervisor_data.extend((event_names[event], target // 256, target % 256))

    return {
        "num_events": len(events),
        "num_supervisors": 1,
        "events": event_names,
        "ev_controllable": controllable,
        "sup_events": [[1] * len(events)],
        "sup_init_state": initial_states,
        "sup_current_state": initial_states.copy(),
        "sup_data_pos": [0],
        "sup_data": supervisor_data,
    }


def convert(input_path: Path, output_path: Path) -> None:
    payload = xml_to_supervisor_data(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, default_flow_style=True)
    print(
        f"Saved {output_path} "
        f"({payload['num_events']} events, {len(payload['sup_data'])} data entries)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Nadzoru supervisor XML into SCT runtime YAML."
    )
    parser.add_argument(
        "--input",
        help="Input supervisor XML (default: newest S_<timestamp>.xml)",
    )
    parser.add_argument(
        "--output",
        help="Output YAML file (default: same name and timestamp as the input)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = (
            Path(args.input).expanduser().resolve()
            if args.input
            else newest_supervisor_xml().resolve()
        )
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else (DEFAULT_YAML_DIR / input_path.with_suffix(".yaml").name).resolve()
        )
        convert(input_path, output_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
