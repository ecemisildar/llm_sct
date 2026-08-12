#!/usr/bin/env python3
"""Run the complete task-to-runtime-YAML SCT generation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    events_for_mission,
    events_available_in_automata,
    generate_json,
    read_required_text,
    save_json,
)


StatusCallback = Callable[[str], None]
DEFAULT_EXPLORATION_FEEDBACK_PATH = (
    Path(__file__).resolve().parent / "exploration_feedback.json"
)
DEFAULT_EXPLORATION_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "llm"
    / "results_exploration"
)
DEFAULT_PATROLLING_FEEDBACK_PATH = (
    Path(__file__).resolve().parent / "patrolling_feedback.json"
)
DEFAULT_PATROLLING_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "llm"
    / "results_patrolling"
)
DEFAULT_DELIVERY_FEEDBACK_PATH = (
    Path(__file__).resolve().parent / "delivery_feedback.json"
)
DEFAULT_DELIVERY_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "llm"
    / "results_delivery"
)


@dataclass(frozen=True)
class PipelineResult:
    json_path: Path
    generated_xml: tuple[Path, ...]
    g_xml: Path
    k_xml: Path
    s_xml: Path
    yaml_path: Path
    payload: object


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def latest_completed_exploration_run(
    results_dir: Path = DEFAULT_EXPLORATION_RESULTS_DIR,
) -> Path | None:
    """Return the newest exploration run containing finalized coverage data."""
    candidates = [
        path
        for path in results_dir.glob("S_*/robots_*/run_*")
        if (path / "SAVE_STATUS.txt").is_file()
        and "Saving OK" in (path / "SAVE_STATUS.txt").read_text(encoding="utf-8")
        and _csv_rows(path / "coverage_timeseries.csv")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            (path / "coverage_timeseries.csv").stat().st_mtime,
            path.name,
        ),
    )


def latest_completed_patrolling_run(
    results_dir: Path = DEFAULT_PATROLLING_RESULTS_DIR,
) -> Path | None:
    """Return the newest finalized LLM patrolling run."""
    candidates = [
        path
        for path in results_dir.glob("S_*/robots_*/run_*")
        if (path / "SAVE_STATUS.txt").is_file()
        and "Saving OK" in (path / "SAVE_STATUS.txt").read_text(encoding="utf-8")
        and _csv_rows(path / "task_result.csv")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            (path / "task_result.csv").stat().st_mtime,
            path.name,
        ),
    )


def latest_completed_delivery_run(
    results_dir: Path = DEFAULT_DELIVERY_RESULTS_DIR,
) -> Path | None:
    """Return the newest finalized LLM delivery run."""
    candidates = [
        path
        for path in results_dir.glob("S_*/robots_*/run_*")
        if (path / "SAVE_STATUS.txt").is_file()
        and "Saving OK" in (path / "SAVE_STATUS.txt").read_text(encoding="utf-8")
        and _csv_rows(path / "task_result.csv")
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            (path / "task_result.csv").stat().st_mtime,
            path.name,
        ),
    )


def _saved_launch_value(status: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}:\s*(.*?)\s*$", status, re.MULTILINE)
    return match.group(1) if match else None


def _matching_previous_llm_output(
    supervisor_name: str,
    objective: str,
) -> object | None:
    yaml_dir = supervisor_xml_to_yaml.DEFAULT_YAML_DIR
    saved_outputs = list(yaml_dir.rglob(f"{supervisor_name}.llm_output.json"))
    if saved_outputs:
        return json.loads(saved_outputs[0].read_text(encoding="utf-8"))

    yaml_paths = list(yaml_dir.rglob(f"{supervisor_name}.yaml"))
    candidates = sorted(
        llm_json_to_xml.DEFAULT_JSON_DIR.glob("llm_output_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if yaml_paths:
        candidates = [
            path
            for path in candidates
            if path.stat().st_mtime <= yaml_path.stat().st_mtime
        ]
    for path in candidates:
        prompt_path = path.with_suffix(".prompt.txt")
        if (
            not objective
            or not prompt_path.is_file()
            or prompt_path.read_text(encoding="utf-8").strip() == objective
        ):
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_latest_exploration_feedback(
    results_dir: Path = DEFAULT_EXPLORATION_RESULTS_DIR,
) -> dict[str, object] | None:
    """Extract feedback from the newest completed LLM exploration run."""
    run_dir = latest_completed_exploration_run(results_dir)
    if run_dir is None:
        return None

    coverage_rows = _csv_rows(run_dir / "coverage_timeseries.csv")
    final_coverage = float(coverage_rows[-1]["coverage_pct"])
    task_rows = _csv_rows(run_dir / "task_result.csv")
    duration = (
        float(task_rows[-1]["duration_s"])
        if task_rows and task_rows[-1].get("duration_s")
        else float(coverage_rows[-1]["time_s"])
    )
    status = (run_dir / "SAVE_STATUS.txt").read_text(encoding="utf-8")
    objective_path = run_dir / "prompt.txt"
    objective = (
        objective_path.read_text(encoding="utf-8").strip()
        if objective_path.is_file()
        else ""
    )
    supervisor_name = run_dir.parent.parent.name
    robot_count_text = _saved_launch_value(status, "total_robots")
    random_seed = _saved_launch_value(status, "random_seed")
    bump_count = len(_csv_rows(run_dir / "bumps_global.csv"))

    event_statistics: dict[str, object] = {}
    for path in sorted(run_dir.glob("event_percentages_robot_*.csv")):
        rows = [
            row
            for row in _csv_rows(path)
            if row.get("selected_event") != "TOTAL"
        ]
        if not rows:
            continue
        robot = rows[0]["robot"]
        event_statistics[robot] = {
            row["selected_event"].removeprefix("EV_"): {
                "count": int(row["count"]),
                "percentage": float(row["percentage"]),
            }
            for row in rows
        }

    feedback: dict[str, object] = {
        "source": {
            "mission": "exploration",
            "supervisor": supervisor_name,
            "run_id": run_dir.name,
            "results_directory": str(run_dir),
            "objective": objective,
        },
        "evaluation": {
            "total_coverage_percent": final_coverage,
            "run_duration_seconds": duration,
            "robot_count": int(robot_count_text) if robot_count_text else None,
            "random_seed": random_seed,
            "recorded_bump_count": bump_count,
        },
        "selected_event_statistics": event_statistics,
        "refinement_request": (
            f"Revise the exploration specification to improve total coverage beyond "
            f"{final_coverage:g} percent while preserving fixed collision avoidance. "
            "Use the event statistics and previous design to justify the changes."
        ),
    }
    previous_output = _matching_previous_llm_output(supervisor_name, objective)
    if previous_output is not None:
        feedback["previous_llm_output"] = previous_output
    return feedback


def build_latest_patrolling_feedback(
    results_dir: Path = DEFAULT_PATROLLING_RESULTS_DIR,
) -> dict[str, object] | None:
    """Extract feedback from the newest completed LLM patrolling run."""
    run_dir = latest_completed_patrolling_run(results_dir)
    if run_dir is None:
        return None

    task = _csv_rows(run_dir / "task_result.csv")[-1]
    coverage_rows = _csv_rows(run_dir / "coverage_timeseries.csv")
    final_coverage = (
        float(coverage_rows[-1]["coverage_pct"]) if coverage_rows else None
    )
    status = (run_dir / "SAVE_STATUS.txt").read_text(encoding="utf-8")
    objective_path = run_dir / "prompt.txt"
    objective = (
        objective_path.read_text(encoding="utf-8").strip()
        if objective_path.is_file()
        else ""
    )
    supervisor_name = run_dir.parent.parent.name
    success = task.get("success", "").casefold() == "true"
    duration = float(task["duration_s"]) if task.get("duration_s") else None
    progress = float(task["progress_pct"]) if task.get("progress_pct") else None

    robot_progress: dict[str, object] = {}
    for row in _csv_rows(run_dir / "task_progress.csv"):
        robot = row.get("robot")
        if not robot:
            continue
        robot_progress[robot] = {
            "completed_goals": int(row["completed_colors"]),
            "total_goals": int(row["total_colors"]),
            "progress_percent": float(row["progress_pct"]),
            "complete": row["complete"].casefold() == "true",
            "completion_seconds": (
                float(row["completion_s"]) if row.get("completion_s") else None
            ),
            "red_reached_seconds": (
                float(row["red_reached_s"]) if row.get("red_reached_s") else None
            ),
            "green_reached_seconds": (
                float(row["green_reached_s"]) if row.get("green_reached_s") else None
            ),
            "blue_reached_seconds": (
                float(row["blue_reached_s"]) if row.get("blue_reached_s") else None
            ),
        }

    event_statistics: dict[str, object] = {}
    for path in sorted(run_dir.glob("event_percentages_robot_*.csv")):
        rows = [
            row
            for row in _csv_rows(path)
            if row.get("selected_event") != "TOTAL"
        ]
        if rows:
            event_statistics[rows[0]["robot"]] = {
                row["selected_event"].removeprefix("EV_"): {
                    "count": int(row["count"]),
                    "percentage": float(row["percentage"]),
                }
                for row in rows
            }

    feedback: dict[str, object] = {
        "source": {
            "mission": "patrolling",
            "supervisor": supervisor_name,
            "run_id": run_dir.name,
            "results_directory": str(run_dir),
            "objective": objective,
        },
        "evaluation": {
            "task_success": success,
            "task_duration_seconds": duration,
            "completed_robots": int(task["completed_robots"]),
            "total_robots": int(task["total_robots"]),
            "completed_goals": int(task["completed_targets"]),
            "total_goals": int(task["total_targets"]),
            "goal_progress_percent": progress,
            "total_coverage_percent": final_coverage,
            "random_seed": _saved_launch_value(status, "random_seed"),
            "recorded_bump_count": len(_csv_rows(run_dir / "bumps_global.csv")),
        },
        "per_robot_goal_progress": robot_progress,
        "selected_event_statistics": event_statistics,
        "refinement_request": (
            (
                "Revise the patrolling specification to preserve successful "
                f"completion of all goals while reducing completion time below "
                f"{duration:g} seconds."
            )
            if success and duration is not None
            else (
                "Revise the patrolling specification so every robot completes "
                f"all goals; the previous run reached {progress:g} percent progress."
            )
        )
        + " Preserve fixed collision avoidance and justify changes using the "
        "per-robot progress and event statistics.",
    }
    previous_output = _matching_previous_llm_output(supervisor_name, objective)
    if previous_output is not None:
        feedback["previous_llm_output"] = previous_output
    return feedback


def build_latest_delivery_feedback(
    results_dir: Path = DEFAULT_DELIVERY_RESULTS_DIR,
) -> dict[str, object] | None:
    """Extract feedback from the newest completed LLM delivery run."""
    run_dir = latest_completed_delivery_run(results_dir)
    if run_dir is None:
        return None

    task = _csv_rows(run_dir / "task_result.csv")[-1]
    coverage_rows = _csv_rows(run_dir / "coverage_timeseries.csv")
    status = (run_dir / "SAVE_STATUS.txt").read_text(encoding="utf-8")
    objective_path = run_dir / "prompt.txt"
    objective = (
        objective_path.read_text(encoding="utf-8").strip()
        if objective_path.is_file()
        else ""
    )
    supervisor_name = run_dir.parent.parent.name
    success = task.get("success", "").casefold() == "true"
    duration = float(task["duration_s"]) if task.get("duration_s") else None
    progress = float(task["progress_pct"]) if task.get("progress_pct") else 0.0

    robot_progress: dict[str, object] = {}
    for row in _csv_rows(run_dir / "task_progress.csv"):
        robot = row.get("robot")
        if not robot:
            continue
        robot_progress[robot] = {
            "delivered_colors": [
                color for color in row.get("delivered_colors", "").split("|")
                if color
            ],
            "delivered_count": int(row.get("delivered_count", 0) or 0),
            "contribution_percent": float(
                row.get("contribution_pct", 0) or 0
            ),
            "team_complete": row.get("team_complete", "").casefold() == "true",
            "red_delivered_seconds": (
                float(row["red_delivered_s"]) if row.get("red_delivered_s") else None
            ),
            "green_delivered_seconds": (
                float(row["green_delivered_s"])
                if row.get("green_delivered_s") else None
            ),
            "blue_delivered_seconds": (
                float(row["blue_delivered_s"])
                if row.get("blue_delivered_s") else None
            ),
        }

    event_statistics: dict[str, object] = {}
    for path in sorted(run_dir.glob("event_percentages_robot_*.csv")):
        rows = [
            row for row in _csv_rows(path)
            if row.get("selected_event") != "TOTAL"
        ]
        if rows:
            event_statistics[rows[0]["robot"]] = {
                row["selected_event"].removeprefix("EV_"): {
                    "count": int(row["count"]),
                    "percentage": float(row["percentage"]),
                }
                for row in rows
            }

    feedback: dict[str, object] = {
        "source": {
            "mission": "delivery",
            "supervisor": supervisor_name,
            "run_id": run_dir.name,
            "results_directory": str(run_dir),
            "objective": objective,
        },
        "evaluation": {
            "task_success": success,
            "task_duration_seconds": duration,
            "completed_robots": int(task["completed_robots"]),
            "total_robots": int(task["total_robots"]),
            "completed_boxes": int(task["completed_targets"]),
            "total_boxes": int(task["total_targets"]),
            "delivery_progress_percent": progress,
            "total_coverage_percent": (
                float(coverage_rows[-1]["coverage_pct"])
                if coverage_rows else None
            ),
            "random_seed": _saved_launch_value(status, "random_seed"),
            "recorded_bump_count": len(_csv_rows(run_dir / "bumps_global.csv")),
        },
        "per_robot_delivery_progress": robot_progress,
        "selected_event_statistics": event_statistics,
        "refinement_request": (
            (
                "Preserve successful one-box-per-robot delivery while reducing "
                f"team completion time below {duration:g} seconds."
            )
            if success and duration is not None
            else (
                "Revise the delivery specification so three robots claim unique "
                "colors and each completes claim, object search/approach, pickup, "
                f"zone search/approach, and drop. Previous progress was {progress:g}%."
            )
        )
        + " Preserve fixed collision avoidance and use per-robot event statistics "
        "to address stalled stages.",
    }
    previous_output = _matching_previous_llm_output(supervisor_name, objective)
    if previous_output is not None:
        feedback["previous_llm_output"] = previous_output
    return feedback


def select_profile(
    task: str, mission: str = "auto", exploration_mode: str = "auto"
) -> tuple[str, list[Path], list[Path]]:
    """Return mission name, fixed plants, and fixed specifications."""
    text = task.casefold()
    if mission == "auto":
        delivery_keywords = (
            "delivery",
            "deliver",
            "pick up",
            "pickup",
            "pick and drop",
            "drop off",
            "drop-off",
            "transport a box",
            "transport the box",
        )
        describes_box_to_zone = (
            ("box" in text or "boxes" in text)
            and ("zone" in text or "zones" in text)
        )
        if (
            any(keyword in text for keyword in delivery_keywords)
            or describes_box_to_zone
        ):
            mission = "delivery"
        elif any(
        # if any(
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

    llm_automata = nadzoru_sync.AUTOMATA_DIR / "llm_automata"
    shared_motion_plant = llm_automata / "shared" / "motion_plant.xml"
    shared_obstacle_sensor = llm_automata / "shared" / "obstacle_sensor.xml"
    shared_collision_avoidance = llm_automata / "shared" / "collision_avoidance.xml"
    shared_task_motion_safety = (
        llm_automata / "shared" / "task_motion_safety.xml"
    )
    if mission == "exploration":
        return (
            mission,
            [shared_obstacle_sensor, shared_motion_plant],
            [shared_collision_avoidance, shared_task_motion_safety],
        )
    if mission == "delivery":
        directory = llm_automata / "delivery"
        plants = [
            shared_obstacle_sensor,
            shared_motion_plant,
            directory / "task_motion_plant.xml",
            *(
                directory / name
                for name in (
                "pickup_and_drop_plant.xml",
                "color_sensor.xml",
                "communication_plant.xml",
                "red_availability.xml",
                "green_availability.xml",
                "blue_availability.xml",
                )
            ),
        ]
        return (
            mission,
            plants,
            [
                shared_collision_avoidance,
                shared_task_motion_safety,
                directory / "task_collision_avoidance.xml",
            ],
        )

    directory = llm_automata / "patrolling"
    plants = [
        shared_obstacle_sensor,
        directory / "color_sensor.xml",
        shared_motion_plant,
        directory / "task_motion_plant.xml",
    ]
    return (
        mission,
        plants,
        [
            shared_collision_avoidance,
            shared_task_motion_safety,
            directory / "task_collision_avoidance.xml",
        ],
    )


def run_pipeline(
    task: str,
    model: str = DEFAULT_MODEL,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    api_key_path: Path = DEFAULT_API_KEY_PATH,
    status: StatusCallback | None = None,
    mission: str = "auto",
    exploration_mode: str = "auto",
    feedback: object | None = None,
    auto_feedback: bool = True,
    prompt_number: int = 1,
    generation_number: int = 1,
) -> PipelineResult:
    if prompt_number < 1 or generation_number < 1:
        raise ValueError("Prompt and generation numbers must be positive integers")
    report = status or (lambda _message: None)
    selected_mission, fixed_plants, fixed_specs = select_profile(
        task, mission, exploration_mode
    )
    if auto_feedback and feedback is None and selected_mission in {
        "exploration", "patrolling", "delivery"
    }:
        if selected_mission == "exploration":
            feedback = build_latest_exploration_feedback()
            feedback_path = DEFAULT_EXPLORATION_FEEDBACK_PATH
        elif selected_mission == "patrolling":
            feedback = build_latest_patrolling_feedback()
            feedback_path = DEFAULT_PATROLLING_FEEDBACK_PATH
        else:
            feedback = build_latest_delivery_feedback()
            feedback_path = DEFAULT_DELIVERY_FEEDBACK_PATH
        if feedback is not None:
            save_json(feedback, feedback_path)
            source = feedback["source"]
            report(
                f"Using automatically extracted {selected_mission} feedback from "
                f"{source['supervisor']}/{source['run_id']}."
            )
        else:
            report(
                f"No completed {selected_mission} run was found; generating "
                "without previous-run feedback."
            )
    report(
        f"Selected {selected_mission} fixed automata: "
        + ", ".join(path.stem for path in [*fixed_plants, *fixed_specs])
    )
    allowed_events = events_available_in_automata(
        events_for_mission(selected_mission), fixed_plants
    )

    report("Requesting JSON automata from OpenAI…")
    payload, json_path = generate_json(
        task=task,
        prompt_path=prompt_path,
        api_key_path=api_key_path,
        model=model,
        context_files=[*fixed_plants, *fixed_specs],
        context_mission=selected_mission,
        feedback=feedback,
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
    # G uses the selected fixed plants. K adds the tested collision
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
    yaml_path = (
        supervisor_xml_to_yaml.DEFAULT_YAML_DIR
        / selected_mission
        / f"prompt_{prompt_number}"
        / f"generation_{generation_number}"
        / s_xml.with_suffix(".yaml").name
    )
    supervisor_xml_to_yaml.convert(s_xml, yaml_path)
    source_prompt_path = json_path.with_suffix(".prompt.txt")
    yaml_prompt_path = yaml_path.with_suffix(".prompt.txt")
    if source_prompt_path.is_file():
        shutil.copy2(source_prompt_path, yaml_prompt_path)
        report(f"Saved matching UI prompt: {yaml_prompt_path}")
    source_feedback_path = json_path.with_suffix(".feedback.json")
    yaml_feedback_path = yaml_path.with_suffix(".feedback.json")
    if source_feedback_path.is_file():
        shutil.copy2(source_feedback_path, yaml_feedback_path)
        report(f"Saved matching feedback: {yaml_feedback_path}")
    yaml_llm_output_path = yaml_path.with_suffix(".llm_output.json")
    shutil.copy2(json_path, yaml_llm_output_path)
    report(f"Saved matching LLM output: {yaml_llm_output_path}")
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
    parser.add_argument(
        "--feedback-file",
        help="Optional JSON file containing previous-run metrics and design feedback",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Generate independently without automatic or file-based feedback",
    )
    parser.add_argument("--prompt-number", type=int, default=1)
    parser.add_argument("--generation-number", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.no_feedback and args.feedback_file:
            raise ValueError("--no-feedback cannot be combined with --feedback-file")
        task = (
            read_required_text(Path(args.task_file).expanduser().resolve(), "Task")
            if args.task_file
            else args.task
        )
        feedback = (
            json.loads(
                read_required_text(
                    Path(args.feedback_file).expanduser().resolve(),
                    "Feedback",
                )
            )
            if args.feedback_file
            else None
        )
        result = run_pipeline(
            task=task,
            model=args.model,
            prompt_path=Path(args.prompt_file).expanduser().resolve(),
            api_key_path=Path(args.api_key_file).expanduser().resolve(),
            mission=args.mission,
            exploration_mode=args.exploration_mode,
            feedback=feedback,
            auto_feedback=not args.no_feedback,
            prompt_number=args.prompt_number,
            generation_number=args.generation_number,
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
