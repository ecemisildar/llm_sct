#!/usr/bin/env python3
"""Plot saved P2–P5, replacement P1, and completed P6 delivery runs."""
import csv
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/delivery-mpl")
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

base = Path(__file__).resolve().parent
root = base.parent
sys.path.insert(0, str(root / 'evaluation/evaluation'))
import analyze_prompt_sensitivity as analyzer


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

rows = [row for row in read(base / 'analysis/delivery/comparison/delivery_prompt_sensitivity_runs.csv')
        if int(row['prompt']) in (2, 3, 4, 5)]
updated = []
for yaml in sorted((base / 'YAML/delivery').glob('*/prompt_*/generation_*/*.yaml')):
    if not re.fullmatch(r'prompt_[1-6]', yaml.parents[1].name):
        continue
    prompt = int(yaml.parents[1].name.removeprefix('prompt_'))
    if prompt not in (1, 6):
        continue
    mode = yaml.parents[2].name
    newest = {}
    for status in (base / 'delivery' / mode).glob('seed_*/' + yaml.stem + '/run_*/SAVE_STATUS.txt'):
        content = status.read_text()
        match = re.search(r'^metadata_yaml:\s*(.+)$', content, re.M)
        if 'Saving OK' not in content or not match or Path(match.group(1).strip()).resolve() != yaml.resolve():
            continue
        seed = analyzer.seed_for_run(status.parent)
        if seed not in range(1001, 1011):
            continue
        if seed not in newest or status.stat().st_mtime > newest[seed].stat().st_mtime:
            newest[seed] = status
    for seed, status in newest.items():
        run = status.parent
        success, duration = analyzer.read_result(run)
        counts = analyzer.delivery_counts(run)
        counts["total_targets"] = 6
        row = dict(task="delivery", prompt=prompt, prompt_text=yaml.with_suffix(".prompt.txt").read_text().strip(),
                   generation=int(yaml.parent.name.removeprefix("generation_")), seed=seed,
                   supervisor=yaml.stem, yaml=str(yaml.resolve()), run=str(run.resolve()), success=success, collision_control=mode,
                   task_success_pct=100 * counts['delivered_boxes'] / counts['total_targets'],
                   duration_s=duration, coverage_pct=analyzer.coverage_at(run, duration),
                   collisions=analyzer.collisions(run))
        row.update(counts)
        updated.append(row)
for prompt in (1, 6):
    print(f'P{prompt}: {sum(r["prompt"] == prompt for r in updated)}/100 completed runs available')
rows.extend(updated)
for row in rows:
    row['prompt'] = int(row['prompt'])

# Use the current six-box delivery-stack baseline, one completed run per seed.
baseline_root = base / 'baseline/delivery_stack_simplified_10seeds/delivery/automata'
baseline_rows = []
for seed in range(1001, 1011):
    statuses = sorted(
        baseline_root.glob(f'seed_{seed}/supervisor/run_*/SAVE_STATUS.txt'),
        key=lambda path: path.stat().st_mtime,
    )
    statuses = [path for path in statuses if 'Saving OK' in path.read_text()]
    if not statuses:
        raise RuntimeError(f'No completed delivery baseline for seed {seed}')
    run = statuses[-1].parent
    success, duration = analyzer.read_result(run)
    counts = analyzer.delivery_counts(run)
    zone_total = counts['red_boxes'] + counts['blue_boxes']
    baseline_rows.append(dict(
        seed=seed,
        success=success,
        task_success_pct=100 * counts['delivered_boxes'] / 6,
        duration_s=duration,
        coverage_pct=analyzer.coverage_at(run, duration),
        collisions=analyzer.collisions(run),
        red_box_percentage=(100 * counts['red_boxes'] / zone_total if zone_total else 0),
        run=str(run.resolve()),
        **counts,
    ))
print(f'Baseline: {len(baseline_rows)}/10 completed runs available')
output = base / 'analysis/delivery/comparison'
baseline_summary = analyzer.summarize(baseline_rows, {'prompt': 'baseline'})
baseline_summary['red_box_percentage_mean'] = sum(
    row['red_box_percentage'] for row in baseline_rows
) / len(baseline_rows)
analyzer.write_csv(output / 'delivery_baseline_summary.csv', [baseline_summary])
fields = list(dict.fromkeys(key for row in rows for key in row))
with (output / 'delivery_prompt_sensitivity_figure_runs.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

namespace = dict(vars(analyzer))
namespace.update(rows=rows, args=SimpleNamespace(task='delivery', prompts=list(range(1, 7)),
                 generations=list(range(1, 6)), seeds=list(range(1001, 1011)), allow_incomplete=True),
                 collision_controls=['without_fixed_spec', 'with_fixed_spec'], baseline_rows=baseline_rows, output=output)
source = Path(analyzer.__file__).read_text()
start = source.index('    labels = [')
setup = source[start:source.index('    if args.task == "exploration":', start)]
start = source.index('    is_delivery = args.task == "delivery"')
plot = source[start:source.index('    print(f"Analyzed {len(rows)} LLM runs"', start)]
exec(textwrap.dedent(setup) + textwrap.dedent(plot), namespace)
print(output / 'delivery_prompt_sensitivity_no_title.png')
