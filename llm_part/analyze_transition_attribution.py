#!/usr/bin/env python3
"""Attribute transition restrictions to fixed specs, generated specs, and SupC.

For every selected task/prompt/generation this script reconstructs the saved
LLM specification and builds four behavior layers::

    G       = Sync(fixed plants)
    K_fixed = Sync(G, fixed specifications)
    K_all   = Sync(G, fixed specifications, generated specifications)
    S       = SupC(G, K_all)

Restrictions are counted as controllable (source-state, event) occurrences,
using product-state projection rather than subtracting raw transition totals.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import Sequence

import llm_json_to_xml
import nadzoru_sync
from analyze_disabled_events import (
    Supervisor,
    disabled_transitions,
    load_supervisor,
    project_to_baseline,
    synthesize,
)
from llm_input import events_available_in_automata, events_for_mission
from run_pipeline import select_profile


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVED_ROOT = ROOT / "automata" / "resulting_automata" / "YAML"
DEFAULT_OUTPUT = ROOT / "new_results" / "transition_restriction_attribution.csv"


def restriction_summary(
    earlier: Supervisor, later: Supervisor
) -> dict[str, object]:
    rows = disabled_transitions(earlier, later)
    transitions = [transition for _, _, items in rows for transition in items]
    earlier_states = set(earlier.transitions)
    represented_states: set[str] = set()
    opportunities = 0
    for later_state in later.transitions:
        earlier_state = project_to_baseline(later_state, earlier_states)
        represented_states.add(earlier_state)
        opportunities += sum(
            transition.event in earlier.controllable_events
            for transition in earlier.transitions[earlier_state]
        )
    # A state removed entirely by the later stage contributes its controllable
    # choices once, matching disabled_transitions().
    for earlier_state in earlier_states - represented_states:
        opportunities += sum(
            transition.event in earlier.controllable_events
            for transition in earlier.transitions[earlier_state]
        )
    disabled = len(transitions)
    rate = 100.0 * disabled / opportunities if opportunities else 0.0
    if not 0.0 <= rate <= 100.0:
        raise AssertionError(
            f"Invalid restriction rate {rate:g}%: disabled={disabled}, "
            f"opportunities={opportunities}"
        )
    return {
        "disabled_occurrences": disabled,
        "controllable_opportunities": opportunities,
        "restriction_rate_pct": rate,
        "unique_disabled_events": len({item.event for item in transitions}),
        "disabled_events": "|".join(sorted({item.event for item in transitions})),
        "removed_states": sum(
            combined == "<state removed by synchronization>"
            for combined, _, _ in rows
        ),
    }


def automaton_size(supervisor: Supervisor) -> tuple[int, int, int]:
    transitions = [
        item
        for outgoing in supervisor.transitions.values()
        for item in outgoing
    ]
    controllable = sum(
        item.event in supervisor.controllable_events for item in transitions
    )
    return len(supervisor.transitions), len(transitions), controllable


def saved_json(
    saved_root: Path, task: str, prompt: int, generation: int
) -> Path:
    directory = (
        saved_root / task / f"prompt_{prompt}" / f"generation_{generation}"
    )
    matches = sorted(directory.glob("S_*.llm_output.json"))
    if len(matches) > 1:
        runs_csv = (
            ROOT / "new_results" / "comparison" / f"{task}_prompt_sensitivity"
            / f"{task}_prompt_sensitivity_runs.csv"
        )
        if runs_csv.is_file():
            with runs_csv.open(newline="", encoding="utf-8") as stream:
                supervisors = {
                    row["supervisor"]
                    for row in csv.DictReader(stream)
                    if int(row["prompt"]) == prompt
                    and int(row["generation"]) == generation
                }
            selected = [path for path in matches if path.stem.removesuffix(".llm_output") in supervisors]
            if len(selected) == 1:
                return selected[0]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one saved LLM JSON in {directory}, found {len(matches)}"
        )
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_task(
    task: str,
    prompts: Sequence[int],
    generations: Sequence[int],
    saved_root: Path,
    temporary_root: Path,
) -> list[dict[str, object]]:
    _, plants, fixed_specs = select_profile(task, mission=task)
    task_root = temporary_root / task
    baseline_outputs = synthesize(plants, fixed_specs, task_root / "fixed")
    g_path = baseline_outputs[0].with_name(
        baseline_outputs[0].name.replace("K_", "G_", 1)
    )
    g = load_supervisor(g_path)
    k_fixed = load_supervisor(baseline_outputs[0])
    fixed = restriction_summary(g, k_fixed)
    g_states, g_transitions, g_controllable = automaton_size(g)
    fixed_states, fixed_transitions, fixed_controllable = automaton_size(k_fixed)

    allowed_events = events_available_in_automata(events_for_mission(task), plants)
    plant_events: set[str] = set()
    for path in plants:
        plant_events.update(nadzoru_sync.xml_event_names(path))
    complete_alphabet = {
        event: controllable
        for event, controllable in allowed_events.items()
        if event in plant_events
    }

    rows: list[dict[str, object]] = []
    for prompt in prompts:
        for generation in generations:
            cell = task_root / f"prompt_{prompt}_generation_{generation}"
            generated = llm_json_to_xml.convert(
                saved_json(saved_root, task, prompt, generation),
                cell / "generated",
                plants[0].parent,
                allowed_generated_events=set(allowed_events),
                complete_event_alphabet=complete_alphabet,
            )
            combined_outputs = synthesize(
                plants, [*fixed_specs, *generated], cell / "synthesis"
            )
            k_all = load_supervisor(combined_outputs[0])
            supervisor = load_supervisor(combined_outputs[1])
            llm = restriction_summary(k_fixed, k_all)
            supc = restriction_summary(k_all, supervisor)
            k_states, k_transitions, k_controllable = automaton_size(k_all)
            s_states, s_transitions, s_controllable = automaton_size(supervisor)
            rows.append(
                {
                    "task": task,
                    "prompt": prompt,
                    "generation": generation,
                    "generated_specifications": len(generated),
                    "g_states": g_states,
                    "g_transitions": g_transitions,
                    "g_controllable_transitions": g_controllable,
                    "k_fixed_states": fixed_states,
                    "k_fixed_transitions": fixed_transitions,
                    "k_fixed_controllable_transitions": fixed_controllable,
                    "fixed_spec_disabled_occurrences": fixed["disabled_occurrences"],
                    "fixed_spec_controllable_opportunities": fixed["controllable_opportunities"],
                    "fixed_spec_restriction_rate_pct": fixed["restriction_rate_pct"],
                    "fixed_spec_unique_disabled_events": fixed["unique_disabled_events"],
                    "fixed_spec_disabled_events": fixed["disabled_events"],
                    "fixed_spec_removed_states": fixed["removed_states"],
                    "k_all_states": k_states,
                    "k_all_transitions": k_transitions,
                    "k_all_controllable_transitions": k_controllable,
                    "llm_spec_disabled_occurrences": llm["disabled_occurrences"],
                    "llm_spec_controllable_opportunities": llm["controllable_opportunities"],
                    "llm_spec_restriction_rate_pct": llm["restriction_rate_pct"],
                    "llm_spec_unique_disabled_events": llm["unique_disabled_events"],
                    "llm_spec_disabled_events": llm["disabled_events"],
                    "llm_spec_removed_states": llm["removed_states"],
                    "supervisor_states": s_states,
                    "supervisor_transitions": s_transitions,
                    "supervisor_controllable_transitions": s_controllable,
                    "supc_disabled_occurrences": supc["disabled_occurrences"],
                    "supc_controllable_opportunities": supc["controllable_opportunities"],
                    "supc_restriction_rate_pct": supc["restriction_rate_pct"],
                    "supc_unique_disabled_events": supc["unique_disabled_events"],
                    "supc_disabled_events": supc["disabled_events"],
                    "supc_removed_states": supc["removed_states"],
                }
            )
            print(
                f"{task} prompt={prompt} generation={generation}: "
                f"fixed={fixed['disabled_occurrences']}, "
                f"llm={llm['disabled_occurrences']}, "
                f"supc={supc['disabled_occurrences']}"
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("exploration", "patrolling", "delivery"),
        default=["exploration", "patrolling", "delivery"],
    )
    parser.add_argument("--prompts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--saved-root", type=Path, default=DEFAULT_SAVED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sct_attribution_") as temp:
        for task in args.tasks:
            rows.extend(
                analyze_task(
                    task,
                    args.prompts,
                    args.generations,
                    args.saved_root.expanduser().resolve(),
                    Path(temp),
                )
            )
    write_csv(args.output.expanduser().resolve(), rows)
    print(f"Saved {len(rows)} rows to {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
