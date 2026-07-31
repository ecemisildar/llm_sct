#!/usr/bin/env python3
"""Send ``input_prompt.txt`` to OpenAI and save a JSON response."""

from __future__ import annotations

import json
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


def all_events() -> dict[str, bool]:
    """Return the complete event alphabet available for every LLM request."""
    return dict(ALL_EVENTS)


def events_for_mission(mission: str) -> dict[str, bool]:
    """Return the authoritative LLM event alphabet for a mission profile."""
    if mission == "exploration":
        return {
            "task_move_forward": True,
            "task_rotate_clockwise": True,
            "task_rotate_counterclockwise": True,
            **OBSTACLE_EVENTS,
        }
    return all_events()


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
) -> tuple[object, Path]:
    base_prompt = read_required_text(prompt_path, "Prompt")
    api_key = read_required_text(api_key_path, "API key")
    context = automata_files_as_json(context_files, context_mission)
    allowed_events = events_for_mission(context_mission)
    prompt = add_existing_automata_context(
        combine_prompt(base_prompt, task, feedback), context
    )
    prompt = add_allowed_event_list(prompt, allowed_events)
    destination = output_path or default_output_path(DEFAULT_OUTPUT_DIR)
    save_task_prompt(task, destination)
    if feedback is not None:
        save_json(feedback, destination.with_suffix(".feedback.json"))
    payload = request_json(prompt, api_key, model)
    validate_response_explanation(payload)
    save_json(payload, destination)
    return payload, destination
