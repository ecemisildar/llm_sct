#!/usr/bin/env python3
"""Export current completed-run controllable event distributions to LaTeX."""
import argparse
import csv
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ('exploration', 'patrolling', 'delivery')
MODES = ('without_fixed_spec', 'with_fixed_spec')
TABLE_MODES = ('baseline', *MODES)


def normalize(event):
    event = event.removeprefix('EV_').removeprefix('task_')
    return {'drop_zone_a': 'drop_zone_red', 'drop_zone_b': 'drop_zone_blue'}.get(event, event)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path, help='Use a saved figure snapshot for LLM runs')
    parser.add_argument('--baseline-root', type=Path, help='Defaults to new_results/baseline')
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    distributions = defaultdict(list)
    pooled = defaultdict(Counter)
    run_rows = []
    if args.run_manifest:
        with args.run_manifest.open(newline='') as stream:
            run_rows = list(csv.DictReader(stream))
    else:
        for task in TASKS:
            latest_yaml = {}
            for path in (root / 'YAML' / task).glob('*/prompt_*/generation_*/S_*.yaml'):
                cell = path.relative_to(root / 'YAML' / task).parts[:3]
                if cell[0] not in MODES or not re.fullmatch(r'prompt_[1-6]', cell[1]):
                    continue
                if cell not in latest_yaml or path.stat().st_mtime > latest_yaml[cell].stat().st_mtime:
                    latest_yaml[cell] = path
            index = {str(path.resolve()): path for path in latest_yaml.values()}
            newest = {}
            for status in (root / task).rglob('SAVE_STATUS.txt'):
                text = status.read_text(errors='replace')
                metadata = re.search(r'^metadata_yaml:\s*(.+)$', text, re.M)
                seed = re.search(r'^\s*random_seed:\s*(\d+)\s*$', text, re.M)
                if 'Saving OK' not in text or not metadata or not seed or not 1001 <= int(seed[1]) <= 1010:
                    continue
                controller = str(Path(metadata[1].strip()).resolve())
                if controller not in index or not (status.parent / 'task_result.csv').exists():
                    continue
                key = controller, int(seed[1])
                if key not in newest or status.stat().st_mtime > newest[key].stat().st_mtime:
                    newest[key] = status
            for (controller, seed), status in sorted(newest.items()):
                path = index[controller]
                run_rows.append(dict(task=task, collision_control=path.parents[2].name,
                    prompt=path.parents[1].name.removeprefix('prompt_'),
                    generation=path.parent.name.removeprefix('generation_'), seed=seed,
                    selected_events=0, run=str(status.parent), yaml=controller))
    baseline_root = args.baseline_root or root.parent / 'new_results/baseline'
    for task in TASKS:
        newest = {}
        for status in (baseline_root / f'results_{task}').rglob('SAVE_STATUS.txt'):
            text = status.read_text(errors='replace')
            seed = re.search(r'^\s*random_seed:\s*(\d+)\s*$', text, re.M)
            if 'Saving OK' not in text or not seed or not 1001 <= int(seed[1]) <= 1010:
                continue
            if not (status.parent / 'task_result.csv').exists():
                continue
            key = int(seed[1])
            if key not in newest or status.stat().st_mtime > newest[key].stat().st_mtime:
                newest[key] = status
        for seed, status in sorted(newest.items()):
            run_rows.append(dict(task=task, collision_control='baseline', prompt=0,
                generation=0, seed=seed, selected_events=0, run=str(status.parent.resolve()), yaml=''))
    for row in run_rows:
        run = Path(row['run'])
        paths = list(run.glob('selected_events_robot_*.csv'))
        if not paths:
            paths = list(run.parent.glob(f'robots_*/{run.name}/selected_events_robot_*.csv'))
        counts = Counter()
        for path in paths:
            with path.open(newline='') as stream:
                for event_row in csv.DictReader(stream):
                    event = normalize(event_row.get('selected_event', '').strip())
                    if event and event.lower() != 'none':
                        counts[event] += 1
        if not counts:
            raise ValueError(f'No controllable selections recorded: {run}')
        total = sum(counts.values())
        row['selected_events'] = total
        key = row['task'], row['collision_control']
        distributions[key].append({event: 100 * count / total for event, count in counts.items()})
        pooled[key].update(counts)
    summary = []
    for task in TASKS:
        events = sorted(set().union(*(pooled[(task, mode)] for mode in TABLE_MODES)))
        lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
                 rf'\caption{{Controllable event distribution for {task}. Values are mean $\pm$ population standard deviation in percent across completed runs (Baseline: $n={len(distributions[(task, "baseline")])}$; LLM: $n={len(distributions[(task, MODES[0])])}$; LLM + fixed specification: $n={len(distributions[(task, MODES[1])])}$). Each run pools its robots and receives equal weight; absent events count as zero.}}',
                 rf'\label{{tab:{task}-controllable-events}}', r'\begin{tabular}{lrrr}', r'\hline',
                 r'Controllable event & Baseline (\%) & LLM (\%) & LLM + fixed spec. (\%) \\', r'\hline']
        for event in events:
            cells = []
            for mode in TABLE_MODES:
                percentages = [d.get(event, 0.) for d in distributions[(task, mode)]]
                mean = statistics.mean(percentages) if percentages else 0.
                sd = statistics.pstdev(percentages) if percentages else 0.
                total = sum(pooled[(task, mode)].values())
                count = pooled[(task, mode)][event]
                summary.append({'task': task, 'collision_control': mode, 'event': event,
                                'runs': len(percentages), 'mean_pct': mean, 'sd_pct': sd,
                                'event_count': count, 'total_selected_events': total,
                                'pooled_pct': 100 * count / total if total else 0.})
                cells.append(rf'${mean:.2f} \pm {sd:.2f}$' if percentages else '--')
            label = ('EV_' + event).replace('_', r'\_')
            lines.append(r'\texttt{' + label + '} & ' + ' & '.join(cells) + r' \\')
        lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
        (output / f'{task}_controllable_event_distribution.tex').write_text('\n'.join(lines))
    combined = '% Generated from RESULTS_LAST; no extra LaTeX packages required.\n\n'
    combined += '\n'.join((output / f'{task}_controllable_event_distribution.tex').read_text() for task in TASKS)
    (output / 'controllable_event_distribution.tex').write_text(combined)
    write_csv(output / 'controllable_event_distribution.csv', summary)
    write_csv(output / 'controllable_event_distribution_runs.csv', run_rows)
    for task in TASKS:
        print(task, {mode: len(distributions[(task, mode)]) for mode in TABLE_MODES})
    (output / 'README.md').write_text('''# Controllable event distribution tables

`controllable_event_distribution.tex` contains one table per task. Individual task `.tex` files can also be included directly with `\\input{...}`. No extra packages are required.

Each cell is mean ± population standard deviation of the per-run event percentage. All robots are pooled within a run; runs are equally weighted. Events absent from a run contribute zero. Only selected controllable events are counted; broadcasts and detected uncontrollable events are excluded. The EV_ and task_ prefixes are normalized; delivery zone a/b aliases map to red/blue. Color-specific patrolling events remain separate.

Uses RESULTS_LAST, latest YAML per prompt/generation/setting, seeds 1001–1010, latest completed Saving OK run per exact metadata YAML path and seed. Baselines use the latest completed run per seed (1001–1010) from new_results/baseline. The delivery baseline is the older three-target task; current delivery controllers use six targets. Baseline and LLM event schemas can differ. The run manifest records the exact snapshot. The summary CSV also includes pooled counts and percentages (which weight runs by their number of selections).

Reproduce from src/llm_sct:

```bash
python3 evaluation/evaluation/export_controllable_event_tables.py --experiment-root RESULTS_LAST --output-dir RESULTS_LAST/analysis/controllable_event_distribution
```
''')


if __name__ == '__main__':
    main()
