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
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "input_prompt.txt"
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
    for suffix in ("visible", "not_visible", "reached")
}

DELIVERY_EVENTS = {
    **{
        event: True
        for event in (
            "search_object",
            "approach_object",
            "search_zone",
            "approach_zone",
            "task_move_forward",
            "task_rotate_clockwise",
            "task_rotate_counterclockwise",
            "pick_up_object",
            "drop_object",
            "claim_red",
            "claim_green",
            "claim_blue",
        )
    },
    **OBSTACLE_EVENTS,
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
}

ALL_EVENTS = {
    **{
        event: True
        for event in (
            "search_color",
            "approach_color",
            "task_move_forward",
            "task_rotate_clockwise",
            "task_rotate_counterclockwise",
            *(
                f"{phase}_{color}"
                for phase in ("search", "approach")
                for color in ("red", "green", "blue")
            ),
            *(
                f"{action}_{color}"
                for action in ("pub_going", "skip")
                for color in ("red", "green", "blue")
            ),
        )
    },
    **OBSTACLE_EVENTS,
    **COLOR_EVENTS,
    **{
        f"received_going_{color}": False
        for color in ("red", "green", "blue")
    },
}

# One authoritative LLM vocabulary. Mission selection filters this vocabulary
# against the events actually provided by that mission's plants.
LLM_EVENTS = {**ALL_EVENTS, **DELIVERY_EVENTS}


def all_events() -> dict[str, bool]:
    """Return the complete event alphabet available for every LLM request."""
    return dict(LLM_EVENTS)


def events_for_mission(mission: str) -> dict[str, bool]:
    """Select the relevant subset of the one global LLM vocabulary."""
    task_motion = {
        event: True
        for event in (
            "task_move_forward",
            "task_rotate_clockwise",
            "task_rotate_counterclockwise",
        )
    }
    if mission == "exploration":
        names = {**task_motion, **OBSTACLE_EVENTS}
    elif mission == "delivery":
        names = dict(DELIVERY_EVENTS)
    elif mission == "patrolling":
        names = {
            **task_motion,
            **{
                event: ALL_EVENTS[event]
                for event in ALL_EVENTS
                if event.startswith(
                    ("search_", "approach_", "pub_going_", "skip_", "received_going_")
                )
            },
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


def add_allowed_event_list(
    prompt: str, allowed_events: dict[str, bool]
) -> str:
    controllable = sorted(
        event for event, is_controllable in allowed_events.items()
        if is_controllable
    )
    uncontrollable = sorted(
        event for event, is_controllable in allowed_events.items()
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


def add_mission_requirements(prompt: str, mission: str) -> str:
    """Add runtime constraints that cannot be inferred from event names alone."""
    if mission != "delivery":
        return prompt
    return (
        f"{prompt}\n\nDELIVERY RUNTIME REQUIREMENTS\n"
        "Each robot delivers exactly one box and is retired after dropping it. "
        "Generate the delivery task specification; the fixed automata describe "
        "plants and collision avoidance but do not contain the delivery solution.\n"
        "Output structure requirements:\n"
        "- Return exactly ONE unified specification automaton named "
        "delivery_task_specification. Do not return separate red, green, and blue "
        "specifications. The one automaton must branch by observed color.\n"
        "- Use observation, claim, pickup, and drop events to CHANGE protocol states. "
        "Never use a repeatable motion command to enter another state.\n"
        "- A repeatable motion command appears only as a SELF-LOOP in the state "
        "where that motion is active. For example, claim_red enters a red-object "
        "approach state, and approach_object self-loops in that approach state. "
        "Likewise, pick_up_object enters a red-zone search state, search_zone "
        "self-loops there, red_zone_visible enters a red-zone approach state, and "
        "approach_zone self-loops there.\n"
        "Runtime event contract:\n"
        "- search_object is repeatable rotation and must self-loop while searching "
        "for an unclaimed object.\n"
        "- A <color>_object_visible observation starts an attempt to claim that "
        "visible color. claim_<color> must not occur before this observation.\n"
        "- In the claim-attempt state, claim_<color> commits this robot to the "
        "color, while received_claim_<color> means another robot won and must "
        "return this robot to unclaimed object search.\n"
        "- approach_object is forward/steering motion. Enable it only after a "
        "successful claim and while pursuing that object, and make it a self-loop.\n"
        "- <color>_object_not_visible returns the claimed branch to an object-search "
        "state; <color>_object_visible returns it to approach. Both object search "
        "and approach must advance on <color>_object_reached because reached may "
        "arrive without visible.\n"
        "- pick_up_object is valid only after the claimed object is reached.\n"
        "- search_zone is repeatable rotation and must self-loop while searching "
        "for the matching zone. <color>_zone_visible enters zone approach.\n"
        "- approach_zone is forward/steering motion, is valid only while pursuing "
        "the matching visible zone, and must self-loop. <color>_zone_not_visible "
        "returns to zone search. Both zone states advance on <color>_zone_reached.\n"
        "- drop_object is valid only after the matching zone is reached.\n"
        "For every selectable color, provide a complete path through search, claim "
        "arbitration, object approach/reacquisition, pickup, matching-zone "
        "search/approach, and drop. Do not create a single-robot sequence that "
        "delivers multiple colors. Every listed controllable event must appear on "
        "at least one transition."
    )


TRANSITION_RE = re.compile(
    r'^\(\s*"([^"\r\n]+)"\s*,\s*"([^"\r\n]+)"\s*,\s*'
    r'"([^"\r\n]+)"\s*\)$'
)


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
                if f"{color}_object_visible" not in incoming.get(source, set()):
                    errors.append(
                        f"{name}: {claim} at {source!r} is not preceded by "
                        f"{color}_object_visible"
                    )
                source_events = {event for event, _ in outgoing.get(source, [])}
                if f"received_claim_{color}" not in source_events:
                    errors.append(
                        f"{name}: claim-attempt state {source!r} must handle "
                        f"received_claim_{color}"
                    )
                target_events = {event for event, _ in outgoing.get(target, [])}
                if "approach_object" not in target_events:
                    errors.append(
                        f"{name}: state {target!r} after {claim} must enable "
                        "approach_object"
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
DEFAULT_MODEL = "gpt-4.1"


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


def validate_response_explanation(payload: object) -> None:
    """Require the design rationale that accompanies generated automata."""
    if not isinstance(payload, dict):
        raise RuntimeError("The API response must be a JSON object")
    required_top_level = {"automata", "explanation"}
    if set(payload) != required_top_level:
        raise RuntimeError(
            "The API response must contain exactly the top-level fields "
            "'automata' and 'explanation'"
        )
    explanation = payload["explanation"]
    required_explanation = {
        "event_selection",
        "transition_design",
        "feedback_response",
    }
    if not isinstance(explanation, dict) or set(explanation) != required_explanation:
        raise RuntimeError(
            "The response explanation must contain exactly 'event_selection', "
            "'transition_design', and 'feedback_response'"
        )
    for field in sorted(required_explanation):
        value = explanation[field]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Explanation field '{field}' must be a non-empty string")


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
            if not isinstance(transition, str):
                continue
            match = re.fullmatch(
                r'\(\s*"([^"\r\n]+)"\s*,\s*"([^"\r\n]+)"\s*,\s*'
                r'"([^"\r\n]+)"\s*\)',
                transition,
            )
            if not match:
                continue
            event = match.group(2)
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


def combine_prompt(
    base_prompt: str,
    task: str,
    feedback: object | None = None,
) -> str:
    task = task.strip()
    if not task:
        raise ValueError("The control task is empty")
    sections = [f"CONTROL OBJECTIVE\n{task}"]
    if feedback is not None:
        sections.append(
            "PREVIOUS RUN FEEDBACK\n"
            "Use this evidence to revise the specification while preserving the "
            "control objective and fixed safety constraints.\n"
            f"{json.dumps(feedback, indent=2, ensure_ascii=False)}"
        )
    sections.append(base_prompt)
    return "\n\n".join(sections)


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
) -> tuple[object, Path]:
    base_prompt = read_required_text(prompt_path, "Prompt")
    api_key = read_required_text(api_key_path, "API key")
    context = automata_files_as_json(context_files, context_mission)
    allowed_events = events_available_in_automata(
        events_for_mission(context_mission), context_files
    )
    prompt = add_existing_automata_context(
        combine_prompt(base_prompt, task, feedback), context
    )
    prompt = add_mission_requirements(prompt, context_mission)
    prompt = add_allowed_event_list(prompt, allowed_events)
    destination = output_path or default_output_path(DEFAULT_OUTPUT_DIR)
    save_task_prompt(task, destination)
    if feedback is not None:
        save_json(feedback, destination.with_suffix(".feedback.json"))
    request_prompt = prompt
    for attempt in range(max_repair_attempts + 1):
        payload = request_json(request_prompt, api_key, model)
        validate_response_explanation(payload)
        semantic_errors = authoritative_event_errors(payload, allowed_events)
        if context_mission == "delivery":
            semantic_errors.extend(delivery_semantic_errors(payload))
        if not semantic_errors:
            break
        if attempt == max_repair_attempts:
            raise RuntimeError(
                f"Generated {context_mission} specification failed semantic "
                "validation after "
                f"{max_repair_attempts + 1} attempts:\n- "
                + "\n- ".join(semantic_errors)
            )
        request_prompt = (
            prompt
            + "\n\nSEMANTIC VALIDATION FAILURE\n"
            + "The previous candidate cannot be synthesized for execution. Repair "
            + "the automata while preserving the objective. Return the complete "
            + "replacement JSON, not a patch. Errors:\n- "
            + "\n- ".join(semantic_errors)
            + "\n\nPREVIOUS INVALID CANDIDATE\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )
    save_json(payload, destination)
    return payload, destination
