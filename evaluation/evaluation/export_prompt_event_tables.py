#!/usr/bin/env python3
"""Export prompt-level event tables using the task event table run manifest."""
import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from export_controllable_event_tables import MODES, TASKS, normalize, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.run_manifest.open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    groups = defaultdict(list)
    pooled = defaultdict(Counter)
    for row in runs:
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
        total = sum(counts.values())
        if not total:
            raise ValueError(f'No controllable selections recorded: {run}')
        row['selected_events'] = total
        key = row['task'], row['collision_control'], int(row['prompt'])
        groups[key].append({event: 100 * n / total for event, n in counts.items()})
        pooled[key].update(counts)
    summary, tables = [], []
    for task in TASKS:
        prompts = sorted({int(r['prompt']) for r in runs if r['task'] == task and r['collision_control'] != 'baseline'})
        events = sorted(set().union(*(pool for key, pool in pooled.items() if key[0] == task)))
        task_tables = []
        for mode in MODES:
            setting = 'LLM + fixed specification' if mode == 'with_fixed_spec' else 'LLM'
            lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
                     rf'\caption{{Chosen controllable events by prompt for {task} ({setting}). Entries are mean percentages across completed runs, pooling robots within each run and weighting runs equally. Generations and seeds are combined within each prompt; absent events contribute zero.}}',
                     rf'\label{{tab:{task}-{mode.replace("_", "-")}-prompt-events}}',
                     r'\begin{tabular}{l' + 'r' * (len(prompts) + 1) + '}', r'\hline',
                     'Chosen event & Baseline (\\%) & ' + ' & '.join(f'P{p} (\\%)' for p in prompts) + r' \\',
                     'Runs ($n$) & ' + str(len(groups[(task, 'baseline', 0)])) + ' & ' + ' & '.join(str(len(groups[(task, mode, p)])) for p in prompts) + r' \\', r'\hline']
            for event in events:
                cells = []
                for column_mode, prompt in [('baseline', 0)] + [(mode, p) for p in prompts]:
                    key = task, column_mode, prompt
                    values = [d.get(event, 0.) for d in groups[key]]
                    mean = statistics.mean(values) if values else None
                    sd = statistics.pstdev(values) if values else None
                    count = pooled[key][event]
                    total = sum(pooled[key].values())
                    summary.append({'task': task, 'collision_control': column_mode, 'prompt': prompt,
                                    'event': event, 'runs': len(values),
                                    'mean_pct': mean, 'sd_pct': sd, 'event_count': count,
                                    'total_selected_events': total,
                                    'pooled_pct': 100 * count / total if total else None})
                    cells.append(f'{mean:.2f}' if mean is not None else '--')
                label = ('EV_' + event).replace('_', r'\_')
                lines.append(r'\texttt{' + label + '} & ' + ' & '.join(cells) + r' \\')
            lines += [r'\hline', r'\end{tabular}', r'\end{table}', '']
            table = '\n'.join(lines)
            (output / f'{task}_{mode}_prompt_events.tex').write_text(table)
            task_tables.append(table)
        (output / f'{task}_prompt_events.tex').write_text('\n'.join(task_tables))
        tables.extend(task_tables)
    (output / 'prompt_chosen_event_tables.tex').write_text('% Prompt event tables; no additional LaTeX packages required.\n\n' + '\n'.join(tables))
    unique_summary = {(r['task'], r['collision_control'], r['prompt'], r['event']): r for r in summary}
    write_csv(output / 'prompt_chosen_event_distribution.csv', list(unique_summary.values()))
    write_csv(output / 'prompt_chosen_event_runs.csv', runs)
    (output / 'README.md').write_text('''# Chosen controllable events by prompt

`prompt_chosen_event_tables.tex` includes two tables per task: LLM and LLM + fixed specification. Individual task files include both settings, and individual setting files are also available. Each table starts with a baseline column, followed by the prompt IDs recorded for each task and a run count row. The delivery baseline uses the older three-target task; current delivery controllers use six targets.

LaTeX cells show mean selected-event percentages across completed runs. Each run pools its robots; runs receive equal weight. All generations and seeds in a prompt are combined. Missing events contribute zero; missing run groups show --. Values can differ slightly from 100% after rounding.

The CSV also includes population standard deviation, event counts, and pooled percentages. The run manifest records the exact completed-run snapshot used by these tables. Only selected controllable events are included. Event names use the same normalization as the overall tables.

Reproduce from src/llm_sct:

```bash
python3 evaluation/evaluation/export_prompt_event_tables.py --run-manifest RESULTS_LAST/analysis/controllable_event_distribution/controllable_event_distribution_runs.csv --output-dir RESULTS_LAST/analysis/prompt_chosen_events
```
''')
    print(f'Generated six prompt tables from {len(runs)} runs under {output}')


if __name__ == '__main__':
    main()
