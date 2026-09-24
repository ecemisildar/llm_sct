#!/usr/bin/env python3
"""Request delivery P1 five times and synthesize against delivery_stack."""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from llm_input import events_from_automata
from run_pipeline import run_pipeline
from run_prompt_batch import already_generated, parse_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-root", type=Path, help="Synthesize saved responses without API calls")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    plants = sorted((root / "automata/baseline_automata/delivery_stack").glob("G*.xml"))
    alphabet = events_from_automata(plants)
    task = parse_prompts(root / "llm_part/prompts.txt", ("delivery",))["delivery"][0]
    output = args.reuse_root.resolve() if args.reuse_root else root / (
        "RESULTS_DELIVERY_STACK_P1_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if args.reuse_root:
        if not output.is_dir():
            parser.error("Reuse directory does not exist")
    else:
        output.mkdir()
    tree = ET.parse(root / "automata/baseline_automata/delivery/collision_avoidance.xml")
    for event in tree.getroot().find("data").findall("event"):
        name = event.attrib["name"]
        event.set("name", {"search_zone": "search_color", "approach_zone": "approach_color"}.get(name, name))
        assert event.attrib["name"] in alphabet
        assert (event.attrib["controllable"].lower() == "true") == alphabet[event.attrib["name"]]
    collision = output / "collision_avoidance.xml"
    tree.write(collision, encoding="utf-8", xml_declaration=True)
    (output / "plant_events.json").write_text(json.dumps(alphabet, indent=2) + "\n")
    print(f"Output: {output}; plant events: {len(alphabet)}", flush=True)
    results = []
    for generation in range(1, 6):
        print(f"Generation {generation}/5", flush=True)
        try:
            candidate = output / f"candidate_{generation}"
            if args.reuse_root and already_generated("delivery", 1, generation, task, "fixed", candidate):
                yaml = next((candidate / f"YAML/delivery/with_fixed_spec/prompt_1/generation_{generation}").glob("S_*.yaml"))
                results.append({"generation": generation, "yaml": str(yaml), "json": str(
                    candidate / f"llm_outputs/delivery/with_fixed_spec/prompt_1/generation_{generation}/llm_output.json"
                )})
                print(f"Already synthesized: {yaml}", flush=True)
                (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                continue
            result = run_pipeline(
                task=task, mission="delivery", prompt_number=1,
                generation_number=generation, collision_control="fixed",
                output_root=output / f"candidate_{generation}",
                plant_paths=plants, specification_paths=[collision],
                max_repair_attempts=0, status=lambda message: print(message, flush=True),
                reuse_json_path=(
                    output / f"candidate_{generation}/llm_outputs/delivery/with_fixed_spec"
                    / f"prompt_1/generation_{generation}/llm_output.json"
                    if args.reuse_root else None
                ),
            )
            results.append({"generation": generation, "yaml": str(result.yaml_path), "json": str(result.json_path)})
        except Exception as error:
            print(f"Generation {generation} failed: {error}", flush=True)
            results.append({"generation": generation, "error": str(error)})
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any("error" in result for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
