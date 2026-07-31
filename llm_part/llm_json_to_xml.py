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
ALLOWED_GENERATED_CONTROLLABLE_EVENTS = {
    "search_color",
    "approach_color",
    "task_move_forward",
    "search_red",
    "search_green",
    "search_blue",
    "approach_red",
    "approach_green",
    "approach_blue",
    "task_rotate_clockwise",
    "task_rotate_counterclockwise",
    "pub_going_red",
    "pub_going_green",
    "pub_going_blue",
    "skip_red",
    "skip_green",
    "skip_blue",
}


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
    if not isinstance(value, str):
        raise ValueError(
            "Every transition must be a JSON string formatted as "
            "'(\"state\", \"event\", \"next\")'; "
            f"received {type(value).__name__}: {value!r}"
        )
    match = re.fullmatch(
        r'\(\s*"([^"\r\n]+)"\s*,\s*"([^"\r\n]+)"\s*,\s*'
        r'"([^"\r\n]+)"\s*\)',
        value,
    )
    if not match:
        raise ValueError(
            "Transition must be a quoted tuple such as "
            f"'(\"state\", \"event\", \"next\")': {value!r}"
        )
    return match.group(1), match.group(2), match.group(3)


def automaton_name(payload: dict[str, Any], fallback: str) -> str:
    value = (
        payload.get("name")
        or payload.get("automaton_name")
        or payload.get("supervisor_id")
        or payload.get("id")
        or fallback
    )
    name = str(value).strip() or fallback
    automaton_type = str(payload.get("type", "specification")).casefold()
    if automaton_type in {"spec", "specification", "control_specification"}:
        if not name.casefold().endswith("_specification"):
            name += "_specification"
    elif automaton_type == "plant":
        if "spec" in name.casefold():
            raise ValueError(f"Plant name must not contain 'spec': {name!r}")
    else:
        raise ValueError(
            f"Unknown automaton type {automaton_type!r}; expected 'plant' or 'specification'"
        )
    return name


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


def validate_sct_rules(
    states: Sequence[str],
    transitions: Sequence[tuple[str, str, str]],
    baseline_events: dict[str, bool],
    require_uncontrollable_totality: bool,
    allowed_generated_events: set[str] | None = None,
) -> None:
    """Enforce the SCT constraints stated in ``input_prompt.txt``."""
    allowed_events = set(baseline_events)
    used_events = {event for _, event, _ in transitions}
    if allowed_generated_events is not None:
        outside_task = used_events - allowed_generated_events
        if outside_task:
            raise ValueError(
                "Generated specifications use events outside the authoritative "
                f"event list: {sorted(outside_task)}"
            )
    forbidden_events = {
        event
        for event in used_events
        if event in baseline_events
        and baseline_events[event]
        and event not in ALLOWED_GENERATED_CONTROLLABLE_EVENTS
    }
    if forbidden_events:
        raise ValueError(
            "Generated specifications use protected low-level controllable events. "
            "Use only authoritative high-level controllable events; forbidden events: "
            f"{sorted(forbidden_events)}"
        )
    unknown_events = used_events - allowed_events
    if unknown_events:
        raise ValueError(
            "Transitions introduce events that are absent from the baseline automata: "
            f"{sorted(unknown_events)}"
        )

    transition_map: dict[tuple[str, str], str] = {}
    outgoing_events: dict[str, set[str]] = {state: set() for state in states}
    for source, event, target in transitions:
        key = (source, event)
        if key in transition_map:
            previous = transition_map[key]
            raise ValueError(
                "Nondeterministic or duplicate transition for "
                f"(state={source!r}, event={event!r}): targets {previous!r} and {target!r}"
            )
        transition_map[key] = target
        outgoing_events[source].add(event)

    if require_uncontrollable_totality:
        uncontrollable_events = {
            event
            for event in used_events
            if not baseline_events[event]
        }
        for state in states:
            missing = uncontrollable_events - outgoing_events[state]
            if missing:
                raise ValueError(
                    f"Specification state {state!r} is missing outgoing transitions for "
                    f"uncontrollable events: {sorted(missing)}"
                )


def complete_specification_events(
    states: Sequence[str],
    transitions: Sequence[tuple[str, str, str]],
    self_loop_events: set[str],
) -> list[tuple[str, str, str]]:
    """Add missing inert transitions for events that must remain enabled."""
    completed = list(transitions)
    existing = {(source, event) for source, event, _ in transitions}
    for state in states:
        for event in sorted(self_loop_events):
            if (state, event) not in existing:
                completed.append((state, event, state))
    return completed



def build_xml(
    payload: dict[str, Any],
    fallback_name: str,
    baseline_events: dict[str, bool],
    allowed_generated_events: set[str] | None = None,
    complete_event_alphabet: dict[str, bool] | None = None,
    globally_selected_events: set[str] | None = None,
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
    declared_events = payload.get("events")
    if not isinstance(declared_events, list) or not declared_events:
        raise ValueError("'events' must be a non-empty list")
    originally_declared_events = {
        str(
            event.get("name", event.get("id"))
            if isinstance(event, dict)
            else event
        )
        for event in declared_events
    }
    automaton_type = str(payload.get("type", "specification")).casefold()
    if automaton_type not in {"spec", "specification", "control_specification"}:
        raise ValueError(
            f"Generated automata must be specifications; got type {automaton_type!r}. "
            "The pipeline supplies fixed, tested plant automata."
        )
    completion = complete_event_alphabet or {}
    selected = globally_selected_events or {
        event for _, event, _ in transitions
    }
    self_loop_events = {
        event
        for event, controllable in completion.items()
        if not controllable
        or (event in selected and event not in originally_declared_events)
    }
    transitions = complete_specification_events(
        state_names, transitions, self_loop_events
    )
    validate_sct_rules(
        state_names,
        transitions,
        baseline_events,
        require_uncontrollable_totality=True,
        allowed_generated_events=allowed_generated_events,
    )

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

    if "marked_states" not in payload:
        raise ValueError(
            "Each generated automaton must explicitly contain 'marked_states'; "
            "the LLM must decide which states represent acceptable conditions"
        )
    marked_value = payload["marked_states"]
    if not isinstance(marked_value, list):
        raise ValueError("'marked_states' must be a list")
    marked = set(string_list(marked_value, "marked_states"))

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
    if "events" not in payload:
        raise ValueError(
            "Each generated automaton must explicitly contain an 'events' "
            "list defining its local synchronization alphabet"
        )
    declared_event_names: list[str] = []
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
        declared_event_names.append(name)
        if name not in event_names:
            event_names.append(name)
    for name in sorted(completion):
        if name not in declared_event_names:
            declared_event_names.append(name)
        if name not in event_names:
            event_names.append(name)
    undeclared_transition_events = (
        set(event for _, event, _ in transitions) - set(declared_event_names)
    )
    if undeclared_transition_events:
        raise ValueError(
            "Transition events missing from the automaton's 'events' list: "
            f"{sorted(undeclared_transition_events)}"
        )

    overlap = controllable & uncontrollable
    if overlap:
        raise ValueError(f"Events are both controllable and uncontrollable: {sorted(overlap)}")
    explicit_event_properties = {
        **{name: True for name in controllable},
        **{name: False for name in uncontrollable},
    }
    undeclared_baseline_events = set(explicit_event_properties) - set(baseline_events)
    if undeclared_baseline_events:
        raise ValueError(
            "Event lists introduce events that are absent from the baseline automata: "
            f"{sorted(undeclared_baseline_events)}"
        )
    for event, explicit_value in explicit_event_properties.items():
        if event in baseline_events and baseline_events[event] != explicit_value:
            raise ValueError(
                f"Event '{event}' conflicts with its baseline controllability "
                f"({baseline_events[event]})"
            )
    unknown_events = set(event_names) - set(baseline_events)
    if unknown_events:
        raise ValueError(
            "Events are absent from the baseline automata: "
            f"{sorted(unknown_events)}"
        )
    if allowed_generated_events is not None:
        outside_task = set(event_names) - allowed_generated_events
        if outside_task:
            raise ValueError(
                "Generated specifications declare events outside the "
                f"authoritative event list: {sorted(outside_task)}"
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


def convert(
    json_path: Path,
    output_dir: Path,
    baseline_dir: Path,
    allowed_generated_events: set[str] | None = None,
    complete_event_alphabet: dict[str, bool] | None = None,
) -> list[Path]:
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {json_path}: {error}") from error
    automata = extract_automata(document)
    globally_selected_events = {
        event
        for payload in automata
        for _, event, _ in (
            parse_transition(item)
            for item in payload.get("transitions", [])
        )
    }
    fallback = json_path.stem
    baseline_events = baseline_event_map(baseline_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    used_names: set[str] = set()
    for index, payload in enumerate(automata, start=1):
        item_fallback = fallback if index == 1 else f"{fallback}_{index}"
        name, xml = build_xml(
            payload,
            item_fallback,
            baseline_events,
            allowed_generated_events=allowed_generated_events,
            complete_event_alphabet=complete_event_alphabet,
            globally_selected_events=globally_selected_events,
        )
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
