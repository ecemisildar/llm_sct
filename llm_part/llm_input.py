#!/usr/bin/env python3
"""Send ``input_prompt.txt`` to OpenAI and save a JSON response."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "input_prompt.txt"
DEFAULT_API_KEY_PATH = SCRIPT_DIR / "api_key.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "llm_outputs"
DEFAULT_OBSTACLE_DIR = (
    SCRIPT_DIR.parent / "automata" / "baseline_automata" / "patrolling"
)
DEFAULT_OBSTACLE_CONTEXT_PATH = SCRIPT_DIR / "fixed_patrolling_automata.json"
FIXED_PATROLLING_AUTOMATA = {
    "obstacle_sensor",
    "color_sensor",
    "motion_plant",
    "collision_avoidance",
}
DEFAULT_MODEL = "gpt-4.1"
CONTROL_OBJECTIVE_PLACEHOLDER = (
    "Replace this paragraph with the physical behavior, mission, coordination, "
    "or safety requirements that the generated plant and specification automata "
    "must model or enforce."
)


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


def combine_prompt(base_prompt: str, task: str) -> str:
    task = task.strip()
    if not task:
        raise ValueError("The control task is empty")
    if CONTROL_OBJECTIVE_PLACEHOLDER in base_prompt:
        return base_prompt.replace(CONTROL_OBJECTIVE_PLACEHOLDER, task, 1)
    return f"CONTROL OBJECTIVE\n{task}\n\n{base_prompt}"


def obstacle_avoidance_as_json(xml_dir: Path) -> dict[str, object]:
    """Convert the existing obstacle-avoidance Nadzoru XML files to LLM context."""
    xml_paths = sorted(
        path for path in xml_dir.glob("*.xml")
        if path.stem in FIXED_PATROLLING_AUTOMATA
    )
    return automata_files_as_json(xml_paths, "patrolling")


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
        role = "specification" if "collision_avoidance" in xml_path.stem else "plant"
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
        f"{prompt}\n\nFIXED PATROLLING AUTOMATA (JSON)\n"
        "These automata are already included during synchronization. Do not regenerate "
        "them. Use their exact states, events, and transitions to design compatible "
        "additional plants and specifications. When the control objective requires "
        "robot motion, explicitly connect relevant observations and mission states to "
        "the existing controllable movement events.\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}"
    )


def generate_json(
    task: str,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    api_key_path: Path = DEFAULT_API_KEY_PATH,
    model: str = DEFAULT_MODEL,
    output_path: Path | None = None,
    obstacle_dir: Path = DEFAULT_OBSTACLE_DIR,
    obstacle_context_path: Path = DEFAULT_OBSTACLE_CONTEXT_PATH,
    context_files: Sequence[Path] | None = None,
    context_mission: str = "patrolling",
) -> tuple[object, Path]:
    base_prompt = read_required_text(prompt_path, "Prompt")
    api_key = read_required_text(api_key_path, "API key")
    context = (
        automata_files_as_json(context_files, context_mission)
        if context_files is not None
        else obstacle_avoidance_as_json(obstacle_dir)
    )
    save_json(context, obstacle_context_path)
    prompt = add_existing_automata_context(combine_prompt(base_prompt, task), context)
    payload = request_json(prompt, api_key, model)
    destination = output_path or default_output_path(DEFAULT_OUTPUT_DIR)
    save_json(payload, destination)
    return payload, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a prompt to OpenAI and save the JSON-formatted response."
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_PATH),
        help=f"Prompt text file (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--api-key-file",
        default=str(DEFAULT_API_KEY_PATH),
        help=f"File containing only the OpenAI API key (default: {DEFAULT_API_KEY_PATH})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: llm_outputs/llm_output_<timestamp>.json)",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="Control task to insert into the base prompt")
    task_group.add_argument("--task-file", help="Text file containing the control task")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    api_key_path = Path(args.api_key_file).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(DEFAULT_OUTPUT_DIR).resolve()
    )

    try:
        if args.task_file:
            task = read_required_text(
                Path(args.task_file).expanduser().resolve(), "Task"
            )
        elif args.task:
            task = args.task
        else:
            raise ValueError("Provide a control task with --task or --task-file")
        _, output_path = generate_json(
            task=task,
            prompt_path=prompt_path,
            api_key_path=api_key_path,
            model=args.model,
            output_path=output_path,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Saved JSON response to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
