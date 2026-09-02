#!/usr/bin/env python3
"""Compose Nadzoru XML models and synthesize a supervisor.

By default XML files are read from the obstacle-avoidance baseline and from
``llm_generated_automata``.  Files whose stem contains a specification marker
(for example ``collision_avoidance`` or ``*_specification``) are specifications;
the remaining files are free-behaviour/plant models.

The equivalent Nadzoru operations are::

    G = Sync(*plants)
    K = Sync(G, *specifications)
    S = SupC(G, K)

Use ``--plant`` and/or ``--spec`` if a filename does not follow that convention.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
AUTOMATA_DIR = SCRIPT_DIR.parent / "automata"
DEFAULT_INPUT_DIRS = (
    AUTOMATA_DIR / "baseline_automata" / "exploration",
    AUTOMATA_DIR / "baseline_automata",
    AUTOMATA_DIR / "llm_generated_automata",
)
DEFAULT_OUTPUT_DIR = AUTOMATA_DIR / "resulting_automata"
DEFAULT_NADZORU_ROOT = Path.home() / "Documents/Nadzoru2"
SPEC_MARKERS = ("spec", "specification", "collision_avoidance", "target_approach")


def _normalise_names(values: Iterable[str]) -> set[str]:
    """Accept either ``name`` or ``name.xml`` in classification options."""
    return {Path(value).stem.casefold() for value in values}


def discover_xml(
    input_dirs: Sequence[Path], input_files: Sequence[Path] = ()
) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {directory}")
        files.extend(path for path in directory.glob("*.xml") if path.is_file())

    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(f"Input XML file does not exist: {path}")
        if path.suffix.casefold() != ".xml":
            raise ValueError(f"Input file is not XML: {path}")
        files.append(path)

    files.sort(key=lambda path: (path.name.casefold(), str(path.parent)))
    if not files:
        joined = ", ".join(str(path) for path in input_dirs)
        raise ValueError(f"No XML automata found in: {joined}")

    duplicate_names: dict[str, list[Path]] = {}
    for path in files:
        duplicate_names.setdefault(path.stem.casefold(), []).append(path)
    conflicts = {name: paths for name, paths in duplicate_names.items() if len(paths) > 1}
    if conflicts:
        details = "; ".join(
            f"{name}: {', '.join(str(path) for path in paths)}"
            for name, paths in conflicts.items()
        )
        raise ValueError(f"Duplicate automaton names across input directories: {details}")
    return files


def classify_xml(
    files: Sequence[Path], plant_names: Iterable[str], spec_names: Iterable[str]
) -> tuple[list[Path], list[Path]]:
    explicit_plants = _normalise_names(plant_names)
    explicit_specs = _normalise_names(spec_names)
    overlap = explicit_plants & explicit_specs
    if overlap:
        raise ValueError(f"Names classified as both plant and specification: {sorted(overlap)}")

    available = {path.stem.casefold() for path in files}
    unknown = (explicit_plants | explicit_specs) - available
    if unknown:
        raise ValueError(f"Classification names not found in the inputs: {sorted(unknown)}")

    plants: list[Path] = []
    specifications: list[Path] = []
    for path in files:
        stem = path.stem.casefold()
        if stem in explicit_plants:
            plants.append(path)
        elif stem in explicit_specs or any(marker in stem for marker in SPEC_MARKERS):
            specifications.append(path)
        else:
            plants.append(path)

    if not plants:
        raise ValueError("At least one plant/free-behaviour XML file is required")
    if not specifications:
        raise ValueError(
            "At least one specification XML file is required; use --spec NAME if its "
            "filename does not contain 'spec' or 'collision_avoidance'"
        )
    return plants, specifications


def xml_event_names(path: Path) -> set[str]:
    try:
        data = ET.parse(path).getroot().find("data")
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML in {path}: {error}") from error
    if data is None:
        raise ValueError(f"XML automaton has no <data> element: {path}")
    names = {event.get("name") for event in data.findall("event")}
    if None in names:
        raise ValueError(f"XML automaton contains an event without a name: {path}")
    return names  # type: ignore[return-value]


def validate_specification_event_coverage(
    plants: Sequence[Path], specifications: Sequence[Path]
) -> None:
    plant_events: set[str] = set()
    for path in plants:
        plant_events.update(xml_event_names(path))
    missing_by_specification: dict[str, list[str]] = {}
    for path in specifications:
        missing = xml_event_names(path) - plant_events
        if missing:
            missing_by_specification[path.stem] = sorted(missing)
    if missing_by_specification:
        details = "; ".join(
            f"{name}: {events}" for name, events in missing_by_specification.items()
        )
        raise ValueError(
            "Specifications use events that are absent from all plant automata. "
            f"Generate or provide plants containing these events: {details}"
        )


def import_automaton(nadzoru_root: Path | None):
    if nadzoru_root is not None:
        root = nadzoru_root.expanduser().resolve()
        if not (root / "machine" / "automaton.py").is_file():
            raise ImportError(f"Not a Nadzoru2 source directory: {root}")
        sys.path.insert(0, str(root))
    try:
        automaton_module = importlib.import_module("machine.automaton")
    except ImportError as error:
        raise ImportError(
            f"Could not import Nadzoru2 from {nadzoru_root}: {error}. "
            "Pass --nadzoru-root /path/to/Nadzoru2 if it is stored elsewhere."
        ) from error
    return automaton_module.Automaton


def load_automata(automaton_type, files: Sequence[Path]):
    loaded = []
    for path in files:
        automaton = automaton_type()
        automaton.load(str(path))
        loaded.append(automaton)
    return loaded


def synchronize(automaton_type, automata: Sequence[object]):
    # Nadzoru requires at least two operands. A lone plant is already G.
    if len(automata) == 1:
        return automata[0].copy()
    return automaton_type.synchronization(*automata)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dirs = [Path(path).expanduser().resolve() for path in args.input_dir]
    input_files = [
        Path(path).expanduser().resolve() for path in getattr(args, "input_file", [])
    ]
    files = discover_xml(input_dirs, input_files)
    plants, specifications = classify_xml(files, args.plant, args.spec)
    validate_specification_event_coverage(plants, specifications)
    Automaton = import_automaton(
        Path(args.nadzoru_root) if args.nadzoru_root is not None else None
    )

    print("Plants:         " + ", ".join(path.stem for path in plants))
    print("Specifications: " + ", ".join(path.stem for path in specifications))

    G = synchronize(Automaton, load_automata(Automaton, plants))
    specification_automata = load_automata(Automaton, specifications)
    K = Automaton.synchronization(G, *specification_automata)
    S = Automaton.sup_c(G, K)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_dir = output_dir / "G_K"
    supervisor_dir = output_dir / "S"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        intermediate_dir / f"G_{timestamp}.xml",
        intermediate_dir / f"K_{timestamp}.xml",
        supervisor_dir / f"S_{timestamp}.xml",
    )
    for automaton, output in zip((G, K, S), outputs):
        if not automaton.save(str(output)):
            raise OSError(f"Nadzoru failed to save {output}")
        print(f"Saved {output} ({len(automaton.states)} states)")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Directory containing XML inputs (repeatable; defaults to the baseline "
        "obstacle_avoidance and llm_generated_automata folders)",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help="Individual XML input (repeatable; useful for one pipeline run)",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--plant", action="append", default=[], metavar="NAME",
        help="Force an input filename/stem to be treated as a plant (repeatable)",
    )
    parser.add_argument(
        "--spec", action="append", default=[], metavar="NAME",
        help="Force an input filename/stem to be treated as a specification (repeatable)",
    )
    parser.add_argument(
        "--nadzoru-root",
        default=str(DEFAULT_NADZORU_ROOT),
        help=f"Nadzoru2 checkout containing machine/automaton.py "
        f"(default: {DEFAULT_NADZORU_ROOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_dir is None:
        args.input_dir = [str(path) for path in DEFAULT_INPUT_DIRS]
    try:
        run(args)
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
