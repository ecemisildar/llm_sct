"""Generate four more P7 exploration controllers and evaluate both SCT modes."""

import csv
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "llm_part"))

from run_pipeline import run_pipeline  # noqa: E402


PROMPT = "Move only forward"
PROMPT_NUMBER = 7
GENERATIONS = range(2, 6)
SEEDS = range(1001, 1011)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-batch", type=Path)
    args = parser.parse_args()
    batch = (
        args.reuse_batch.expanduser().resolve()
        if args.reuse_batch
        else HERE / ("exploration_api4_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    batch.mkdir(exist_ok=bool(args.reuse_batch))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = batch / (
        f"results_strict_plain_{run_stamp}" if args.reuse_batch else "results"
    )
    results.mkdir()
    print(f"Batch: {batch}", flush=True)

    controllers = []
    for generation in GENERATIONS:
        print(f"API generation {generation}/5", flush=True)
        saved_json = (
            batch / "llm_outputs" / "exploration" / "with_fixed_spec"
            / f"prompt_{PROMPT_NUMBER}" / f"generation_{generation}"
            / "llm_output.json"
        )
        fixed = run_pipeline(
            task=PROMPT,
            mission="exploration",
            prompt_number=PROMPT_NUMBER,
            generation_number=generation,
            collision_control="fixed",
            output_root=batch,
            reuse_json_path=saved_json if args.reuse_batch else None,
            status=lambda message: print(f"[G{generation} fixed] {message}", flush=True),
        )
        controllers.append({
            "generation": generation,
            "collision_control": "with_fixed_spec",
            "yaml": str(fixed.yaml_path),
            "json": str(fixed.json_path),
        })

        plain = run_pipeline(
            task=PROMPT,
            mission="exploration",
            prompt_number=PROMPT_NUMBER,
            generation_number=generation,
            collision_control="none",
            output_root=batch,
            reuse_json_path=fixed.json_path,
            complete_generated_alphabet=False,
            status=lambda message: print(f"[G{generation} plain] {message}", flush=True),
        )
        controllers.append({
            "generation": generation,
            "collision_control": "without_fixed_spec",
            "yaml": str(plain.yaml_path),
            "json": str(plain.json_path),
        })

    (batch / "controllers.json").write_text(
        json.dumps(controllers, indent=2) + "\n", encoding="utf-8"
    )

    # The batch may contain YAMLs from an interrupted attempt. Expose only the
    # eight controllers created in this invocation to the simulation runner.
    yaml_root = batch / f"YAML_strict_plain_{run_stamp}"
    for controller in controllers:
        source = Path(controller["yaml"])
        relative = source.relative_to(batch / "YAML")
        destination = yaml_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for sidecar in source.parent.glob(source.stem + ".*"):
            shutil.copy2(sidecar, destination.parent / sidecar.name)
    (results / "YAML").symlink_to(yaml_root, target_is_directory=True)

    runner = ROOT / "llm_part" / "run_parallel_llm_simulations.py"
    command = [
        sys.executable,
        str(runner),
        "--mission", "exploration",
        "--yaml-root", str(yaml_root),
        "--results-root", str(results),
        "--seeds", ",".join(str(seed) for seed in SEEDS),
        "--exploration-duration", "60",
        "--parallel", "6",
        "--domain-base", "181",
        "--partition-prefix", batch.name + "_exploration",
    ]
    print("Running 80 exploration jobs with six parallel workers.", flush=True)
    subprocess.run(command, check=True)

    statuses = list(results.glob("exploration/**/SAVE_STATUS.txt"))
    saved = [status for status in statuses if "Saving OK" in status.read_text()]
    if len(saved) != 80:
        raise RuntimeError(f"Expected 80 saved runs, found {len(saved)} of {len(statuses)}")
    for status in saved:
        run = status.parent
        world = (run / "sim_world.sdf").read_text()
        if "generic_delivery_box" in world:
            raise RuntimeError(f"Unexpected delivery world: {run}")
        with (run / "bumps_global.csv").open(newline="") as stream:
            if any("generic_delivery_box" in row.get("key", "") for row in csv.DictReader(stream)):
                raise RuntimeError(f"Delivery-box contact in exploration run: {run}")

    (batch / "VALIDATION.txt").write_text(
        "80/80 exploration runs saved; generations 2-5, with and without fixed "
        "collision specification, seeds 1001-1010, 60 s, six workers.\n",
        encoding="utf-8",
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/leo-prompt-replot-mpl")
    analysis = results / "analysis" / "exploration"
    subprocess.run([
        sys.executable,
        str(ROOT / "evaluation" / "evaluation" / "analyze_prompt_sensitivity.py"),
        "--task", "exploration",
        "--experiment-root", str(results),
        "--output-dir", str(analysis),
        "--prompts", "7",
        "--generations", "2", "3", "4", "5",
        "--seeds", *(str(seed) for seed in SEEDS),
        "--no-baseline",
    ], check=True)
    print(f"DONE: {batch}", flush=True)


if __name__ == "__main__":
    main()
