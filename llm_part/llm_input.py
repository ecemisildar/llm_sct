#!/usr/bin/env python3
"""Send ``input_prompt.txt`` to OpenAI and save a JSON response."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "llm_input.txt"
DEFAULT_API_KEY_PATH = SCRIPT_DIR / "api_key.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "llm_outputs"
OBSTACLE_EVENTS = {
    "obstacle_front": False,
    "obstacle_left": False,
    "obstacle_right": False,
    "path_clear": False,
}
COLOR_EVENTS = {
    f"{color}_{suffix}": False
    for color in ("red", "green", "blue")
    for suffix in ("visible", "not_visible", "reached", "not_reached")
}

DELIVERY_EVENTS = {
    **{
        event: True
        for event in (
            "search_object",
            "approach_object",
            "search_zone",
            "approach_zone",
            "pick_up_object",
            "drop_object",
            "claim_red",
            "claim_green",
            "claim_blue",
            "drop_zone_red",
            "drop_zone_green",
            "drop_zone_blue",
        )
    },
    **OBSTACLE_EVENTS,
    **COLOR_EVENTS,
    **{
        f"object_{observation}": False
        for observation in ("visible", "not_visible", "reached", "not_reached")
    },
    **{
        f"{prefix}_drop_zone_{color}": False
        for prefix in ("received", "recieve")
        for color in ("red", "green", "blue")
    },
    "search_color": True,
    "approach_color": True,
    **{
        f"{color}_{target}_{observation}": False
        for color in ("red", "green", "blue")
        for target in ("object", "zone")
        for observation in ("visible", "not_visible", "reached")
    },
    **{
        f"received_claim_{color}": False
        for color in ("red", "green", "blue")
    },
    "timeout": False,
}

ALL_EVENTS = {
    **{
        event: True
        for event in (
            "search_color",
            "approach_color",
            "move_forward",
            "move_backward",
            "rotate_clockwise",
            "rotate_counterclockwise",
            "u_turn",
        )
    },
    **OBSTACLE_EVENTS,
    **COLOR_EVENTS,
}

# One authoritative LLM vocabulary. Mission selection filters this vocabulary
# against the events actually provided by that mission's plants.
LLM_EVENTS = {**ALL_EVENTS, **DELIVERY_EVENTS}
LLM_EVENT_ALIASES = {}
LLM_EVENT_NAMES = {}

EVENT_MEANINGS = {
    "move_forward": "command the robot to drive forward.",
    "move_backward": "command the robot to drive backward.",
    "rotate_clockwise": "command an in-place clockwise rotation.",
    "rotate_counterclockwise": "command an in-place counterclockwise rotation.",
    "u_turn": "command the robot to turn around.",
    "obstacle_front": "An obstacle intersects the robot's forward safety region.",
    "obstacle_left": "An obstacle intersects the robot's forward-left safety region.",
    "obstacle_right": "An obstacle intersects the robot's forward-right safety region.",
    "path_clear": "sensor observation that the forward path is clear.",
    "search_color": "repeatable high-level action that searches for the requested color.",
    "approach_color": "repeatable high-level action that approaches the currently visible requested color.",
    "search_object": "repeatable high-level action that searches for a delivery object.",
    "approach_object": "repeatable high-level action that approaches a visible object.",
    "search_zone": "repeatable action that searches for a delivery zone.",
    "approach_zone": "repeatable action that approaches a visible delivery zone.",
    "pick_up_object": "command to pick up the reached delivery object.",
    "drop_object": "command to drop the carried object in the reached delivery zone.",
    "object_visible": "sensor observation that the delivery object is visible.",
    "object_not_visible": "sensor observation that the delivery object is no longer visible.",
    "object_reached": "sensor observation that the delivery object has been reached.",
    "object_not_reached": "sensor observation that the delivery object has not been reached.",
    "timeout": "timer observation used to advance or retry a waiting state.",
}


def event_meaning(event: str) -> str:
    """Return the concise runtime meaning of an authoritative event."""
    if event in EVENT_MEANINGS:
        return EVENT_MEANINGS[event]
    match = re.fullmatch(r"(red|green|blue)_(visible|not_visible|reached|not_reached)", event)
    if match:
        color, observation = match.groups()
        observation_meanings = {
            "visible": "is visible",
            "not_visible": "is no longer visible",
            "reached": "has been reached",
            "not_reached": "has not been reached",
        }
        return f"sensor observation that the {color} target {observation_meanings[observation]}."
    match = re.fullmatch(r"drop_zone_(red|green|blue)", event)
    if match:
        return f"command to claim the {match.group(1)} delivery-object assignment."
    match = re.fullmatch(r"(?:received|recieve)_drop_zone_(red|green|blue)", event)
    if match:
        return f"communication observation that another robot claimed {match.group(1)}."
    match = re.fullmatch(
        r"(red|green|blue)_(object|zone)_(visible|not_visible|reached)", event
    )
    if match:
        color, target, observation = match.groups()
        return (
            f"sensor observation that the {color} {target} is "
            f"{observation.replace('_', ' ')}."
        )
    match = re.fullmatch(r"claim_(red|green|blue)", event)
    if match:
        return f"command to claim the {match.group(1)} delivery assignment."
    match = re.fullmatch(r"received_claim_(red|green|blue)", event)
    if match:
        return f"observation that another robot claimed the {match.group(1)} assignment."
    raise ValueError(f"No event meaning is defined for authoritative event {event!r}")


def add_event_meanings(prompt: str, allowed_events: dict[str, bool]) -> str:
    """Append meanings only for events supplied to this LLM request."""
    meanings = "\n".join(
        f"- `{LLM_EVENT_NAMES.get(event, event)}`: {event_meaning(event)}"
        for event in sorted(allowed_events)
    )
    return f"{prompt}\n\nEVENT MEANINGS FOR THIS REQUEST\n{meanings}"


def all_events() -> dict[str, bool]:
    """Return the complete event alphabet available for every LLM request."""
    return dict(LLM_EVENTS)


def events_for_mission(mission: str) -> dict[str, bool]:
    """Select the relevant subset of the one global LLM vocabulary."""
    exploration_motion = {
        event: True
        for event in (
            "move_forward",
            "move_backward",
            "rotate_clockwise",
            "rotate_counterclockwise",
            "u_turn",
        )
    }
    if mission == "exploration":
        names = {**exploration_motion, **OBSTACLE_EVENTS}
    elif mission in {"delivery", "complex_task", "payload_delivery"}:
        names = {**DELIVERY_EVENTS, **exploration_motion}
    elif mission == "patrolling":
        names = {
            "search_color": True,
            "approach_color": True,
            **exploration_motion,
            **OBSTACLE_EVENTS,
            **COLOR_EVENTS,
        }
    else:
        names = all_events()
    return {event: LLM_EVENTS[event] for event in names}


def events_available_in_automata(
    allowed_events: dict[str, bool], xml_paths: Sequence[Path]
) -> dict[str, bool]:
    """Restrict the LLM vocabulary to events supplied by selected automata."""
    available: set[str] = set()
    for xml_path in xml_paths:
        data = ET.parse(xml_path).getroot().find("data")
        if data is None:
            raise ValueError(f"Baseline XML has no data element: {xml_path}")
        available.update(
            event.attrib["name"] for event in data.findall("event")
        )
    return {
        event: controllable
        for event, controllable in allowed_events.items()
        if event in available
    }


def events_from_automata(xml_paths: Sequence[Path]) -> dict[str, bool]:
    """Read the authoritative alphabet and controllability from selected plants."""
    events: dict[str, bool] = {}
    for xml_path in xml_paths:
        data = ET.parse(xml_path).getroot().find("data")
        if data is None:
            raise ValueError(f"Baseline XML has no data element: {xml_path}")
        for event in data.findall("event"):
            name = event.attrib["name"]
            controllable = event.get("controllable", "false").lower() == "true"
            if name in events and events[name] != controllable:
                raise ValueError(f"Conflicting plant controllability for {name}")
            event_meaning(name)
            events[name] = controllable
    if not events:
        raise ValueError("Selected plants contain no events")
    return events


def add_allowed_event_list(
    prompt: str, allowed_events: dict[str, bool]
) -> str:
    controllable = sorted(
        LLM_EVENT_NAMES.get(event, event)
        for event, is_controllable in allowed_events.items()
        if is_controllable
    )
    uncontrollable = sorted(
        LLM_EVENT_NAMES.get(event, event)
        for event, is_controllable in allowed_events.items()
        if not is_controllable
    )
    return (
        f"{prompt}\n\nAUTHORITATIVE EVENT LIST\n"
        "Use only events in this list. Events visible in fixed-automata JSON but "
        "absent here are forbidden in generated output.\n"
        "Select only the events relevant to the user's control objective.\n"
        f"Controllable events: {json.dumps(controllable)}\n"
        f"Uncontrollable observation events: {json.dumps(uncontrollable)}"
    )


TRANSITION_RE = re.compile(
    r'^\(\s*"([^"\r\n]+)"\s*,\s*"([^"\r\n]+)"\s*,\s*'
    r'"([^"\r\n]+)"\s*\)$'
)


def normalize_llm_event_aliases(payload: object) -> object:
    """Translate LLM-facing event aliases to fixed-automata identifiers."""
    if not isinstance(payload, dict):
        return payload
    automata = payload.get("automata")
    if isinstance(automata, list):
        for automaton in automata:
            if not isinstance(automaton, dict):
                continue
            events = automaton.get("events")
            if isinstance(events, list):
                automaton["events"] = [
                    LLM_EVENT_ALIASES.get(event, event)
                    if isinstance(event, str)
                    else event
                    for event in events
                ]
            transitions = automaton.get("transitions")
            if isinstance(transitions, list):
                normalized = []
                for transition in transitions:
                    if isinstance(transition, list) and len(transition) == 3:
                        source, event, target = transition
                        if all(isinstance(value, str) for value in transition):
                            normalized.append(
                                [source, LLM_EVENT_ALIASES.get(event, event), target]
                            )
                        else:
                            normalized.append(transition)
                        continue
                    match = (
                        TRANSITION_RE.fullmatch(transition)
                        if isinstance(transition, str)
                        else None
                    )
                    if match is None:
                        normalized.append(transition)
                        continue
                    source, event, target = match.groups()
                    event = LLM_EVENT_ALIASES.get(event, event)
                    normalized.append(f'("{source}", "{event}", "{target}")')
                automaton["transitions"] = normalized
    return payload


def delivery_semantic_errors(payload: object) -> list[str]:
    """Return runtime-contract violations in an LLM delivery specification."""
    if not isinstance(payload, dict) or not isinstance(payload.get("automata"), list):
        return ["The response must contain an automata array."]

    errors: list[str] = []
    parsed: list[tuple[str, list[tuple[str, str, str]]]] = []
    for index, automaton in enumerate(payload["automata"]):
        if not isinstance(automaton, dict):
            errors.append(f"automata[{index}] is not an object")
            continue
        name = str(automaton.get("name", f"automata[{index}]"))
        transitions: list[tuple[str, str, str]] = []
        for raw in automaton.get("transitions", []):
            if isinstance(raw, list) and len(raw) == 3:
                transitions.append(tuple(map(str, raw)))
                continue
            match = TRANSITION_RE.fullmatch(raw) if isinstance(raw, str) else None
            if match:
                transitions.append(match.groups())
        parsed.append((name, transitions))

    all_events = {event for _, transitions in parsed for _, event, _ in transitions}
    required_actions = {
        "search_object", "approach_object", "pick_up_object",
        "search_zone", "approach_zone", "drop_object",
    }
    missing_actions = required_actions - all_events
    if missing_actions:
        errors.append(f"Missing required delivery actions: {sorted(missing_actions)}")

    protocol_automata = [
        name
        for name, transitions in parsed
        if {event for _, event, _ in transitions} & required_actions
    ]
    if len(protocol_automata) != 1:
        errors.append(
            "Return exactly one unified delivery protocol specification containing "
            "all red, green, and blue branches; found protocol automata: "
            f"{protocol_automata}"
        )

    for name, transitions in parsed:
        local_events = {event for _, event, _ in transitions}
        # Auxiliary policy specifications may restrict only claims. Validate the
        # complete protocol automaton(s), identified by their delivery actions.
        if not (local_events & required_actions):
            continue
        incoming: dict[str, set[str]] = {}
        outgoing: dict[str, list[tuple[str, str]]] = {}
        for source, event, target in transitions:
            incoming.setdefault(target, set()).add(event)
            outgoing.setdefault(source, []).append((event, target))
            if event in {"search_object", "approach_object", "search_zone", "approach_zone"} and source != target:
                errors.append(
                    f"{name}: repeatable motion event {event!r} must self-loop, "
                    f"but has {source!r} -> {target!r}"
                )

        for color in ("red", "green", "blue"):
            claim = f"claim_{color}"
            claims = [(source, target) for source, event, target in transitions if event == claim]
            if not claims:
                errors.append(f"{name}: missing {claim} branch")
                continue
            for source, target in claims:
                source_events = {event for event, _ in outgoing.get(source, [])}
                if f"received_claim_{color}" not in source_events:
                    errors.append(
                        f"{name}: claim-attempt state {source!r} must handle "
                        f"received_claim_{color}"
                    )
                target_events = {event for event, _ in outgoing.get(target, [])}
                visible_before_claim = (
                    f"{color}_object_visible" in incoming.get(source, set())
                )
                required_next_action = (
                    "approach_object" if visible_before_claim else "search_object"
                )
                if required_next_action not in target_events:
                    errors.append(
                        f"{name}: state {target!r} after {claim} must enable "
                        f"{required_next_action}"
                    )

        for source, event, _target in transitions:
            prior = incoming.get(source, set())
            if event == "pick_up_object" and not any(
                item.endswith("_object_reached") for item in prior
            ):
                errors.append(
                    f"{name}: pick_up_object at {source!r} is not preceded by an "
                    "object_reached observation"
                )
            if event == "drop_object" and not any(
                item.endswith("_zone_reached") for item in prior
            ):
                errors.append(
                    f"{name}: drop_object at {source!r} is not preceded by a "
                    "zone_reached observation"
                )
    return list(dict.fromkeys(errors))


DEFAULT_MODEL = "gpt-5.6-sol"


def read_required_text(path: Path, description: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError(f"{description} file does not exist: {path}") from error
    except OSError as error:
        raise ValueError(f"Could not read {description} file {path}: {error}") from error
    if not value:
        raise ValueError(f"{description} file is empty: {path}")
    return value


def request_json(prompt: str, api_key: str, model: str) -> object:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI Python package is not installed. Install it with "
            "'python3 -m pip install openai'."
        ) from error

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
    except Exception as error:
        raise RuntimeError(f"OpenAI request failed: {error}") from error
    if not response.output_text:
        raise RuntimeError("The API returned no text output")
    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"The API response was not valid JSON: {error}") from error


def validate_compact_response(payload: object) -> None:
    """Validate the compact policy-only response shape."""
    if not isinstance(payload, dict):
        raise RuntimeError("The API response must be a JSON object")
    if set(payload) != {"automata"} or not isinstance(payload["automata"], list):
        raise RuntimeError("The API response must contain exactly an 'automata' array")
    if not payload["automata"]:
        raise RuntimeError("The 'automata' array must not be empty")
    required = {"name", "initial_state", "marked_states", "transitions"}
    for index, automaton in enumerate(payload["automata"]):
        if not isinstance(automaton, dict) or set(automaton) != required:
            raise RuntimeError(
                f"automata[{index}] must contain exactly: {sorted(required)}"
            )
        if not isinstance(automaton["name"], str) or not automaton["name"].strip():
            raise RuntimeError(f"automata[{index}].name must be a non-empty string")
        if not isinstance(automaton["initial_state"], str):
            raise RuntimeError(f"automata[{index}].initial_state must be a string")
        if not isinstance(automaton["marked_states"], list):
            raise RuntimeError(f"automata[{index}].marked_states must be an array")
        transitions = automaton["transitions"]
        if not isinstance(transitions, list) or not transitions:
            raise RuntimeError(f"automata[{index}].transitions must be non-empty")
        if any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) and value for value in item)
            for item in transitions
        ):
            raise RuntimeError(
                f"Every automata[{index}] transition must be [source, event, target]"
            )


def authoritative_event_errors(
    payload: object, allowed_events: dict[str, bool]
) -> list[str]:
    """Report event names that are not in the mission's fixed-automata alphabet."""
    if not isinstance(payload, dict) or not isinstance(payload.get("automata"), list):
        return ["The response must contain an 'automata' list."]

    allowed = set(allowed_events)
    errors: list[str] = []
    for index, automaton in enumerate(payload["automata"]):
        if not isinstance(automaton, dict):
            errors.append(f"Automaton {index + 1} must be a JSON object.")
            continue
        name = str(automaton.get("name") or f"automaton_{index + 1}")
        declared = automaton.get("events", [])
        declared_names: set[str] = set()
        if isinstance(declared, list):
            for event in declared:
                if isinstance(event, dict):
                    value = event.get("name", event.get("id"))
                    if value is not None:
                        declared_names.add(str(value))
                else:
                    declared_names.add(str(event))
            unknown_declared = sorted(
                declared_names - allowed
            )
            if unknown_declared:
                errors.append(
                    f"{name} declares events outside the authoritative event list: "
                    f"{unknown_declared}."
                )

        states = {
            str(state) for state in automaton.get("states", [])
        } if isinstance(automaton.get("states", []), list) else set()
        unknown_transition_events: set[str] = set()
        state_names_used_as_events: set[str] = set()
        used_transition_events: set[str] = set()
        transitions = automaton.get("transitions", [])
        if not isinstance(transitions, list):
            continue
        for transition in transitions:
            if isinstance(transition, list) and len(transition) == 3:
                source, event, target = map(str, transition)
            elif isinstance(transition, str):
                match = TRANSITION_RE.fullmatch(transition)
                if not match:
                    continue
                source, event, target = match.groups()
            else:
                continue
            used_transition_events.add(event)
            if event not in allowed:
                unknown_transition_events.add(event)
                if event in states:
                    state_names_used_as_events.add(event)
        if unknown_transition_events:
            detail = (
                " These names are states, not events: "
                f"{sorted(state_names_used_as_events)}."
                if state_names_used_as_events
                else ""
            )
            errors.append(
                f"{name} uses transition events outside the authoritative event "
                f"list: {sorted(unknown_transition_events)}.{detail} Use only exact "
                "event names from the provided list; do not invent an event to move "
                "between task phases."
            )
        unused_declared_controllables = sorted(
            event
            for event in declared_names - used_transition_events
            if allowed_events.get(event) is True
        )
        if unused_declared_controllables:
            errors.append(
                f"{name} declares controllable events but never uses them in "
                f"transitions: {unused_declared_controllables}. Remove each unused "
                "event from the automaton's events list or add a valid transition "
                "that implements it."
            )
    return errors


def default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"llm_output_{timestamp}.json"


def save_json(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Could not save JSON to {output_path}: {error}") from error


def save_task_prompt(task: str, output_path: Path) -> Path:
    """Save the exact user task beside its corresponding LLM JSON output."""
    prompt_path = output_path.with_suffix(".prompt.txt")
    try:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(task.rstrip() + "\n", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Could not save task prompt to {prompt_path}: {error}"
        ) from error
    return prompt_path


def save_llm_prompt(prompt: str, output_path: Path) -> Path:
    """Save the complete, exact input sent to the OpenAI Responses API."""
    prompt_path = output_path.with_suffix(".llm_prompt.txt")
    try:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Could not save complete LLM prompt to {prompt_path}: {error}"
        ) from error
    return prompt_path


def combine_prompt(
    base_prompt: str,
    task: str,
) -> str:
    """Combine the invariant instructions with the task-specific user objective."""
    task = task.strip()
    if not task:
        raise ValueError("The control task is empty")
    return f"{base_prompt}\n\nUSER INPUT\n{task}"


def automata_files_as_json(
    xml_paths: Sequence[Path], mission: str
) -> dict[str, object]:
    """Convert selected fixed Nadzoru XML files to JSON context."""
    automata: list[dict[str, object]] = []
    if not xml_paths:
        raise ValueError("No fixed XML automata were selected")

    for xml_path in xml_paths:
        try:
            data = ET.parse(xml_path).getroot().find("data")
        except (ET.ParseError, OSError) as error:
            raise ValueError(f"Could not read baseline XML {xml_path}: {error}") from error
        if data is None:
            raise ValueError(f"Baseline XML has no data element: {xml_path}")

        states_by_id = {
            state.attrib["id"]: state.attrib["name"] for state in data.findall("state")
        }
        events_by_id = {
            event.attrib["id"]: event.attrib["name"] for event in data.findall("event")
        }
        states = list(states_by_id.values())
        initial_states = [
            state.attrib["name"]
            for state in data.findall("state")
            if state.get("initial", "false").strip().casefold() == "true"
        ]
        marked_states = [
            state.attrib["name"]
            for state in data.findall("state")
            if state.get("marked", "false").strip().casefold() == "true"
        ]
        transitions = [
            {
                "source": states_by_id[transition.attrib["source"]],
                "event": events_by_id[transition.attrib["event"]],
                "target": states_by_id[transition.attrib["target"]],
            }
            for transition in data.findall("transition")
        ]
        stem = xml_path.stem.casefold()
        role = (
            "specification"
            if "collision_avoidance" in stem or stem.startswith("e")
            else "plant"
        )
        automata.append(
            {
                "name": xml_path.stem,
                "type": role,
                "states": states,
                "initial_state": initial_states[0] if initial_states else None,
                "marked_states": marked_states,
                "events": [
                    {
                        "name": event.attrib["name"],
                        "controllable": event.get("controllable", "false")
                        .strip()
                        .casefold()
                        == "true",
                    }
                    for event in data.findall("event")
                ],
                "transitions": transitions,
            }
        )
    plants = [item["name"] for item in automata if item["type"] == "plant"]
    specifications = [
        item["name"] for item in automata if item["type"] == "specification"
    ]
    return {
        "fixed_synthesis_roles": {
            "mission": mission,
            "G_plants": plants,
            "K_fixed_specifications": specifications,
            "llm_output_role": "specifications_only",
        },
        "fixed_automata": automata,
    }


def add_existing_automata_context(prompt: str, context: dict[str, object]) -> str:
    return (
        f"{prompt}\n\nFIXED AUTOMATA (JSON)\n"
        "These automata are already included during synchronization. Do not regenerate "
        "them. Use their exact states, events, and transitions to design compatible "
        "additional control specifications. When the control objective requires "
        "robot motion, explicitly connect relevant observations and mission states to "
        "the existing controllable movement events.\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}"
    )


def add_collision_control_context(prompt: str, collision_control: str) -> str:
    """State which layer owns collision avoidance for this generation."""
    if collision_control == "fixed":
        instruction = (
            "A fixed collision_avoidance specification will be added during "
            "synchronization and is not part of this API input. Use low-level "
            "movement events for the mission objective, but do not generate obstacle "
            "avoidance restrictions; the converter keeps missing motion transitions "
            "neutral and the fixed specification enforces safety."
        )
    elif collision_control in {"none", "llm"}:
        instruction = (
            "No fixed specification is included in this API input. Generate only "
            "the control specification required by the user input; synchronization "
            "will separately evaluate it with and without the fixed collision "
            "specification."
        )
    else:
        raise ValueError("collision_control must be 'fixed', 'none', or 'llm'")
    return f"{prompt}\n\nCOLLISION CONTROL MODE\n{collision_control}\n{instruction}"


def add_mission_protocol_context(prompt: str, mission: str) -> str:
    """Add runtime ordering rules that are not obvious from plant structure alone."""
    if mission != "delivery":
        return prompt
    return (
        prompt
        + "\n\nDELIVERY PROTOCOL\n"
        "Two claim orders are valid: claim an available color and then search for "
        "its object, or observe that color's object first and then claim it. After "
        "a claim, use search_object when the object has not yet been seen; when "
        "object_visible preceded the claim, use approach_object. Continue approaching "
        "until object_reached, "
        "and only then pick_up_object. Next, repeatedly search_zone until the "
        "matching color's zone_visible observation, approach_zone until "
        "zone_reached, and only then drop_object."
    )


def generate_json(
    task: str,
    context_files: Sequence[Path],
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    api_key_path: Path = DEFAULT_API_KEY_PATH,
    model: str = DEFAULT_MODEL,
    output_path: Path | None = None,
    context_mission: str = "patrolling",
    feedback: object | None = None,
    max_repair_attempts: int = 2,
    collision_control: str = "fixed",
    allowed_events: dict[str, bool] | None = None,
) -> tuple[object, Path]:
    if feedback is not None:
        raise ValueError(
            "Previous-run feedback is no longer part of the LLM input. "
            "Put any desired requirement in the user input instead."
        )
    base_prompt = read_required_text(prompt_path, "Prompt")
    api_key = read_required_text(api_key_path, "API key")
    context = automata_files_as_json(context_files, context_mission)
    if allowed_events is None:
        allowed_events = events_available_in_automata(
            events_for_mission(context_mission), context_files
        )
    # Keep the composition identical across repeated generations. The user task,
    # fixed-automata context, and mission-specific authoritative alphabet are the
    # only task inputs that vary.
    prompt = add_allowed_event_list(
        add_event_meanings(
            add_collision_control_context(
                add_existing_automata_context(combine_prompt(base_prompt, task), context),
                collision_control,
            ),
            allowed_events,
        ),
        allowed_events,
    )
    # Delivery-only protocol prompting is intentionally disabled so delivery is
    # evaluated with the same general instructions as the other missions.
    # prompt = add_mission_protocol_context(prompt, context_mission)
    destination = output_path or default_output_path(DEFAULT_OUTPUT_DIR)
    save_task_prompt(task, destination)
    save_llm_prompt(prompt, destination)
    for attempt in range(max_repair_attempts + 1):
        # Retries use the exact same input as the first request. Validation
        # details and prior candidates must not make the LLM prompt task/run specific.
        payload = normalize_llm_event_aliases(request_json(prompt, api_key, model))
        validate_compact_response(payload)
        semantic_errors = authoritative_event_errors(payload, allowed_events)
        # Delivery-only semantic validation is intentionally disabled for now.
        # if context_mission in {"delivery", "complex_task"}:
        #     semantic_errors.extend(delivery_semantic_errors(payload))
        if not semantic_errors:
            break
        if attempt == max_repair_attempts:
            raise RuntimeError(
                f"Generated {context_mission} specification failed semantic "
                "validation after "
                f"{max_repair_attempts + 1} attempts:\n- "
                + "\n- ".join(semantic_errors)
            )
    save_json(payload, destination)
    return payload, destination
