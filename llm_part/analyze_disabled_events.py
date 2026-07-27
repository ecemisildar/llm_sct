#!/usr/bin/env python3
"""Report controllable events disabled by each generated specification in Sync.

For each generated XML, this script asks Nadzoru to build:

    K_baseline = Sync(G, *fixed_specifications)
    K_generated = Sync(G, *fixed_specifications, generated_specification)

It then compares the two synchronized products state by state. A controllable
transition present in ``K_baseline`` but absent from ``K_generated`` is reported
as disabled by that generated specification. Uncontrollable transitions are not
reported as disabled control choices.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import nadzoru_sync
from run_pipeline import select_profile


GENERATED_DIR = (
    Path(__file__).resolve().parent.parent
    / "automata"
    / "llm_generated_automata"
)


@dataclass(frozen=True)
class Transition:
    event: str
    target: str


@dataclass(frozen=True)
class Supervisor:
    initial_state: str
    transitions: dict[str, tuple[Transition, ...]]
    controllable_events: frozenset[str]


def newest_generated_xml(directory: Path = GENERATED_DIR) -> Path:
    candidates = list(directory.glob("*.xml"))
    if not candidates:
        raise FileNotFoundError(
            f"No generated specification XML files found in {directory}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_supervisor(path: Path) -> Supervisor:
    data = ET.parse(path).getroot().find("data")
    if data is None:
        raise ValueError(f"XML has no <data> element: {path}")

    states = {
        state.attrib["id"]: state.attrib["name"]
        for state in data.findall("state")
    }
    initial = [
        state.attrib["name"]
        for state in data.findall("state")
        if state.get("initial", "false").casefold() == "true"
    ]
    if len(initial) != 1:
        raise ValueError(
            f"Expected one initial state in {path}, found {initial}"
        )

    events = {
        event.attrib["id"]: (
            event.attrib["name"],
            event.get("controllable", "false").casefold() == "true",
        )
        for event in data.findall("event")
    }
    transitions: dict[str, list[Transition]] = {
        name: [] for name in states.values()
    }
    for transition in data.findall("transition"):
        event_name, _ = events[transition.attrib["event"]]
        transitions[states[transition.attrib["source"]]].append(
            Transition(
                event=event_name,
                target=states[transition.attrib["target"]],
            )
        )

    return Supervisor(
        initial_state=initial[0],
        transitions={
            state: tuple(outgoing)
            for state, outgoing in transitions.items()
        },
        controllable_events=frozenset(
            name for name, controllable in events.values() if controllable
        ),
    )


def synthesize(
    plants: Sequence[Path],
    specifications: Sequence[Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    args = SimpleNamespace(
        input_dir=[],
        input_file=[str(path) for path in [*plants, *specifications]],
        plant=[path.stem for path in plants],
        spec=[path.stem for path in specifications],
        output_dir=str(output_dir),
        nadzoru_root=str(nadzoru_sync.DEFAULT_NADZORU_ROOT),
    )
    _, synchronized_xml, supervisor_xml = nadzoru_sync.run(args)
    return synchronized_xml, supervisor_xml


def project_to_baseline(
    combined_state: str, baseline_states: set[str]
) -> str:
    """Find the baseline product state embedded in a combined state.

    Nadzoru may reorder automata during synchronization, so generated state
    components are not necessarily at the end of the combined state name.
    """
    components = combined_state.split(",")
    if not baseline_states:
        raise ValueError("The baseline supervisor has no states")
    baseline_widths = {len(state.split(",")) for state in baseline_states}
    matches = set()
    for width in baseline_widths:
        if width > len(components):
            continue
        for indices in combinations(range(len(components)), width):
            candidate = ",".join(components[index] for index in indices)
            if candidate in baseline_states:
                matches.add(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely project combined state {combined_state!r} onto "
            f"a baseline state; matches={sorted(matches)}"
        )
    return matches.pop()


def disabled_transitions(
    baseline: Supervisor,
    combined: Supervisor,
) -> list[tuple[str, str, tuple[Transition, ...]]]:
    rows = []
    represented_baseline_states = set()
    baseline_states = set(baseline.transitions)

    for combined_state, outgoing in combined.transitions.items():
        baseline_state = project_to_baseline(combined_state, baseline_states)
        represented_baseline_states.add(baseline_state)
        baseline_outgoing = baseline.transitions.get(baseline_state)
        if baseline_outgoing is None:
            raise ValueError(
                f"Projected state {baseline_state!r} is absent from the baseline "
                "supervisor. Product-state component ordering may have changed."
            )

        combined_events = {transition.event for transition in outgoing}
        disabled = tuple(
            transition
            for transition in baseline_outgoing
            if transition.event in baseline.controllable_events
            and transition.event not in combined_events
        )
        if disabled:
            rows.append((combined_state, baseline_state, disabled))

    for baseline_state, outgoing in baseline.transitions.items():
        if baseline_state in represented_baseline_states:
            continue
        disabled = tuple(
            transition
            for transition in outgoing
            if transition.event in baseline.controllable_events
        )
        if disabled:
            rows.append(("<state removed by synchronization>", baseline_state, disabled))

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        choices=("exploration", "patrolling", "delivery"),
        default="exploration",
    )
    parser.add_argument(
        "--exploration-mode",
        choices=("without_backward", "with_backward"),
        default="without_backward",
    )
    parser.add_argument(
        "--generated",
        action="append",
        type=Path,
        help="Generated specification XML (repeatable; default: newest generated XML)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generated = (
        [path.expanduser().resolve() for path in args.generated]
        if args.generated
        else [newest_generated_xml().resolve()]
    )
    missing = [path for path in generated if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Generated specification does not exist: "
            + ", ".join(str(path) for path in missing)
        )

    mission, plants, fixed_specs = select_profile(
        args.mission,
        mission=args.mission,
        exploration_mode=args.exploration_mode,
    )
    print(f"Mission: {mission}")
    print("Plants: " + ", ".join(path.stem for path in plants))
    print("Fixed specifications: " + ", ".join(path.stem for path in fixed_specs))

    found_disabled = False
    with tempfile.TemporaryDirectory(prefix="sct_transition_analysis_") as temp:
        root = Path(temp)
        baseline_outputs = synthesize(plants, fixed_specs, root / "baseline")
        baseline = load_supervisor(baseline_outputs[0])

        for index, specification in enumerate(generated, start=1):
            combined_outputs = synthesize(
                plants,
                [*fixed_specs, specification],
                root / f"generated_{index}",
            )
            combined = load_supervisor(combined_outputs[0])
            rows = disabled_transitions(baseline, combined)

            print(f"\nGenerated specification: {specification.stem}")
            print(
                "Compared synchronized K products: "
                f"baseline states={len(baseline.transitions)}, "
                f"generated states={len(combined.transitions)}"
            )
            if not rows:
                print("  No controllable events disabled during synchronization.")
                continue

            found_disabled = True
            for combined_state, baseline_state, transitions in rows:
                print(f"\n  synchronized state: {combined_state}")
                print(f"  baseline state:     {baseline_state}")
                for transition in transitions:
                    print(
                        f"    DISABLED {transition.event} -> {transition.target}"
                    )

    if not found_disabled:
        print("\nNo generated specification disabled a controllable event.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
