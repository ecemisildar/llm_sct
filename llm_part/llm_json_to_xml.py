#!/usr/bin/env python3
"""Convert the newest timestamped LLM JSON response to Nadzoru XML."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_DIR = SCRIPT_DIR / "llm_outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "automata" / "llm_generated_automata"
DEFAULT_BASELINE_DIR = SCRIPT_DIR.parent / "automata" / "baseline_automata"


def newest_json(directory: Path = DEFAULT_JSON_DIR) -> Path:
    candidates = sorted(directory.glob("llm_output_[0-9]*_[0-9]*.json"))
    if not candidates:
        raise FileNotFoundError(f"No llm_output_<timestamp>.json found in {directory}")
    return candidates[-1]


def string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"'{field}' must be a string or list")
    return [str(item) for item in value]


def parse_transition(value: Any) -> tuple[str, str, str]:
    if isinstance(value, dict):
        source = value.get("source", value.get("from", value.get("src")))
        event = value.get("event")
        target = value.get("target", value.get("to", value.get("dst")))
        if source is None or event is None or target is None:
            raise ValueError(f"Transition dictionary lacks source/event/target: {value}")
        return str(source), str(event), str(target)

    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(str(item) for item in value)  # type: ignore[return-value]

    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*\(?\s*['\"]?([^,'\"()]+)['\"]?\s*,\s*"
            r"['\"]?([^,'\"()]+)['\"]?\s*,\s*"
            r"['\"]?([^,'\"()]+)['\"]?\s*\)?\s*",
            value,
        )
        if match:
            return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]
    raise ValueError(f"Invalid transition; expected [source, event, target]: {value!r}")


def automaton_name(payload: dict[str, Any], fallback: str) -> str:
    value = (
        payload.get("name")
        or payload.get("automaton_name")
        or payload.get("supervisor_id")
        or payload.get("id")
        or fallback
    )
    return str(value).strip() or fallback


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot derive a filename from automaton name {name!r}")
    return cleaned


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def baseline_event_map(baseline_dir: Path) -> dict[str, bool]:
    """Return event controllability collected from all baseline XML files."""
    xml_paths = sorted(baseline_dir.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No baseline XML files found in {baseline_dir}")

    event_map: dict[str, bool] = {}
    event_sources: dict[str, Path] = {}
    for xml_path in xml_paths:
        try:
            data = ET.parse(xml_path).getroot().find("data")
        except ET.ParseError as error:
            raise ValueError(f"Invalid baseline XML in {xml_path}: {error}") from error
        if data is None:
            raise ValueError(f"Baseline XML has no <data> element: {xml_path}")
        for event in data.findall("event"):
            name = event.get("name")
            raw_controllable = event.get("controllable")
            if not name or raw_controllable is None:
                raise ValueError(f"Invalid event definition in baseline XML: {xml_path}")
            controllable = raw_controllable.casefold() == "true"
            if name in event_map and event_map[name] != controllable:
                raise ValueError(
                    f"Baseline event '{name}' has conflicting controllability in "
                    f"{event_sources[name]} and {xml_path}"
                )
            event_map[name] = controllable
            event_sources[name] = xml_path
    return event_map


def build_xml(
    payload: dict[str, Any], fallback_name: str, baseline_events: dict[str, bool]
) -> tuple[str, bytes]:
    raw_transitions = payload.get("transitions")
    if not isinstance(raw_transitions, list) or not raw_transitions:
        raise ValueError("Each automaton must contain a non-empty 'transitions' list")
    transitions = [parse_transition(item) for item in raw_transitions]

    raw_states = payload.get("states", [])
    state_names: list[str] = []
    state_options: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_states, list):
        raise ValueError("'states' must be a list")
    for state in raw_states:
        if isinstance(state, dict):
            name = state.get("name", state.get("id"))
            if name is None:
                raise ValueError(f"State dictionary has no name/id: {state}")
            name = str(name)
            state_names.append(name)
            state_options[name] = state
        else:
            state_names.append(str(state))
    for source, _, target in transitions:
        state_names.extend((source, target))
    state_names = ordered_unique(state_names)

    initial = string_list(
        payload.get("initial_states", payload.get("initial_state")), "initial_state"
    )
    if not initial:
        initial = [
            name
            for name, options in state_options.items()
            if bool(options.get("initial", False))
        ]
    if not initial:
        initial = state_names[:1]
    if len(initial) != 1:
        raise ValueError(f"Expected exactly one initial state, found: {initial}")

    marked_value = payload.get("marked_states", payload.get("marked"))
    marked = set(string_list(marked_value, "marked_states"))
    if not marked and marked_value is None:
        if any("marked" in options for options in state_options.values()):
            marked = {
                name
                for name, options in state_options.items()
                if bool(options.get("marked", False))
            }
        else:
            marked = set(state_names)

    unknown_states = (set(initial) | marked) - set(state_names)
    if unknown_states:
        raise ValueError(f"Initial/marked states not found in transitions/states: {unknown_states}")

    controllable = set(
        string_list(payload.get("controllable_events"), "controllable_events")
    )
    uncontrollable = set(
        string_list(payload.get("uncontrollable_events"), "uncontrollable_events")
    )
    overlap = controllable & uncontrollable
    if overlap:
        raise ValueError(f"Events are both controllable and uncontrollable: {sorted(overlap)}")
    event_names = ordered_unique(event for _, event, _ in transitions)
    declared_events = payload.get("events", [])
    if isinstance(declared_events, list):
        for event in declared_events:
            if isinstance(event, dict):
                name_value = event.get("name", event.get("id"))
                if name_value is None:
                    raise ValueError(f"Event dictionary has no name/id: {event}")
                name = str(name_value)
                if bool(event.get("controllable", True)):
                    controllable.add(name)
                else:
                    uncontrollable.add(name)
            else:
                name = str(event)
            if name not in event_names:
                event_names.append(name)
    elif declared_events:
        raise ValueError("'events' must be a list")

    overlap = controllable & uncontrollable
    if overlap:
        raise ValueError(f"Events are both controllable and uncontrollable: {sorted(overlap)}")
    explicit_event_properties = {
        **{name: True for name in controllable},
        **{name: False for name in uncontrollable},
    }
    for event, explicit_value in explicit_event_properties.items():
        if event in baseline_events and baseline_events[event] != explicit_value:
            raise ValueError(
                f"Event '{event}' conflicts with its baseline controllability "
                f"({baseline_events[event]})"
            )
    unknown_events = set(event_names) - set(baseline_events) - set(explicit_event_properties)
    if unknown_events:
        raise ValueError(
            "Events are absent from the baseline automata and must be explicitly listed "
            "under controllable_events or uncontrollable_events: "
            f"{sorted(unknown_events)}"
        )

    name = automaton_name(payload, fallback_name)
    root = ET.Element("model", version="0.0", type="FSA", id=name)
    data = ET.SubElement(root, "data")
    state_ids = {state: str(index) for index, state in enumerate(state_names)}
    event_ids = {event: str(index) for index, event in enumerate(event_names)}

    for index, state in enumerate(state_names):
        ET.SubElement(
            data,
            "state",
            id=str(index),
            name=state,
            initial=str(state in initial),
            marked=str(state in marked),
            x=str(150 + 180 * (index % 4)),
            y=str(150 + 100 * (index // 4)),
        )
    for index, event in enumerate(event_names):
        is_controllable = (
            explicit_event_properties[event]
            if event in explicit_event_properties
            else baseline_events[event]
        )
        ET.SubElement(
            data,
            "event",
            id=str(index),
            name=event,
            controllable=str(is_controllable),
            observable="True",
        )
    for source, event, target in transitions:
        ET.SubElement(
            data,
            "transition",
            source=state_ids[source],
            target=state_ids[target],
            event=event_ids[event],
        )

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return safe_filename(name), xml_body + b"\n"


def extract_automata(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        values = document
    elif isinstance(document, dict) and isinstance(document.get("automata"), list):
        values = document["automata"]
    elif isinstance(document, dict):
        values = [document]
    else:
        raise ValueError("The JSON root must be an automaton object, list, or {'automata': [...]}")
    if not values or not all(isinstance(item, dict) for item in values):
        raise ValueError("The automata collection must contain JSON objects")
    return values


def convert(json_path: Path, output_dir: Path, baseline_dir: Path) -> list[Path]:
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {json_path}: {error}") from error
    fallback = json_path.stem
    baseline_events = baseline_event_map(baseline_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    used_names: set[str] = set()
    for index, payload in enumerate(extract_automata(document), start=1):
        item_fallback = fallback if index == 1 else f"{fallback}_{index}"
        name, xml = build_xml(payload, item_fallback, baseline_events)
        filename = f"{name}.xml"
        if filename.casefold() in used_names:
            raise ValueError(f"Duplicate output automaton name: {filename}")
        used_names.add(filename.casefold())
        output = output_dir / filename
        output.write_bytes(xml)
        outputs.append(output)
        print(f"Saved {output}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LLM-generated JSON automata to Nadzoru XML."
    )
    parser.add_argument(
        "--json",
        help="Input JSON (default: newest llm_outputs/llm_output_<timestamp>.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"XML destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help=f"Baseline XML directory used for event definitions (default: {DEFAULT_BASELINE_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        json_path = Path(args.json).expanduser().resolve() if args.json else newest_json().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        baseline_dir = Path(args.baseline_dir).expanduser().resolve()
        convert(json_path, output_dir, baseline_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
