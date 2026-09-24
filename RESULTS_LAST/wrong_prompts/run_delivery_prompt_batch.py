"""Generate five delivery responses and synthesize both collision variants."""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "llm_part"))
from run_pipeline import run_pipeline
from llm_input import events_from_automata

task = (ROOT / "delivery_prompt.txt").read_text().strip()
output = ROOT / ("delivery_stack_batch_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
output.mkdir()
plants = sorted((ROOT.parents[1] / "automata/baseline_automata/delivery_stack").glob("G*.xml"))
alphabet = events_from_automata(plants)
tree = ET.parse(ROOT.parents[1] / "automata/baseline_automata/delivery/collision_avoidance.xml")
for event in tree.getroot().find("data").findall("event"):
    name = event.attrib["name"]
    event.set("name", {"search_zone": "search_color", "approach_zone": "approach_color"}.get(name, name))
    assert event.attrib["name"] in alphabet
    assert (event.attrib["controllable"].lower() == "true") == alphabet[event.attrib["name"]]
collision = output / "collision_avoidance.xml"
tree.write(collision, encoding="utf-8", xml_declaration=True)
(output / "plant_events.json").write_text(json.dumps(alphabet, indent=2) + "\n")
(output / "prompt.txt").write_text(task + "\n")
summary = []
print(f"Output: {output}", flush=True)
for generation in range(1, 6):
    json_path = None
    for mode in ("none", "fixed"):
        print(f"Generation {generation}/5, collision control {mode}", flush=True)
        try:
            result = run_pipeline(
                task=task, mission="delivery", prompt_number=1,
                generation_number=generation, collision_control=mode,
                output_root=output / f"candidate_{generation}",
                reuse_json_path=json_path, max_repair_attempts=0,
                plant_paths=plants, specification_paths=[collision],
                status=lambda message: print(message, flush=True),
            )
            json_path = result.json_path
            summary.append({"generation": generation, "mode": mode,
                            "yaml": str(result.yaml_path), "json": str(json_path)})
        except Exception as error:
            summary.append({"generation": generation, "mode": mode, "error": str(error)})
            print(f"Failed: {error}", flush=True)
            if json_path is None:
                break
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
raise SystemExit(int(any("error" in row for row in summary)))
