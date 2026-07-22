#!/usr/bin/env python3
"""Run the complete task-to-runtime-YAML SCT generation pipeline."""

from __future__ import annotations

import argparse
import json
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


def run_pipeline(
    task: str,
    model: str = DEFAULT_MODEL,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    api_key_path: Path = DEFAULT_API_KEY_PATH,
    status: StatusCallback | None = None,
) -> PipelineResult:
    report = status or (lambda _message: None)

    report("Requesting JSON automata from OpenAI…")
    payload, json_path = generate_json(
        task=task,
        prompt_path=prompt_path,
        api_key_path=api_key_path,
        model=model,
    )

    report("Converting generated JSON to Nadzoru XML…")
    generated_xml = tuple(
        llm_json_to_xml.convert(
            json_path,
            llm_json_to_xml.DEFAULT_OUTPUT_DIR,
            llm_json_to_xml.DEFAULT_BASELINE_DIR,
        )
    )

    report("Synchronizing plants and specifications with Nadzoru…")
    sync_args = nadzoru_sync.build_parser().parse_args([])
    # Use baseline obstacle avoidance plus only this request's generated XMLs.
    # Old generated files remain available on disk but cannot affect this run.
    sync_args.input_dir = [
        str(nadzoru_sync.AUTOMATA_DIR / "baseline_automata" / "obstacle_avoidance")
    ]
    sync_args.input_file = [str(path) for path in generated_xml]
    g_xml, k_xml, s_xml = nadzoru_sync.run(sync_args)

    report("Encoding the synthesized supervisor as runtime YAML…")
    yaml_path = supervisor_xml_to_yaml.DEFAULT_YAML_DIR / s_xml.with_suffix(".yaml").name
    supervisor_xml_to_yaml.convert(s_xml, yaml_path)
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
