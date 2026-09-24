"""Export per-run controllable event count mean and population variance."""
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from export_controllable_event_tables import normalize, write_csv
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'RESULTS_LAST/analysis/prompt_event_mean_variance'
OUT.mkdir(exist_ok=True)
manifest = ROOT / 'RESULTS_LAST/analysis/prompt_chosen_events/prompt_chosen_event_runs.csv'
runs = list(csv.DictReader(manifest.open()))
runs = [r for r in runs if r['collision_control'] == 'baseline' or r['task'] == 'delivery']
for task, folder in [('exploration', 'with_fixed_spec'), ('patrolling', 'comparison')]:
    runs.extend(csv.DictReader((ROOT / f'RESULTS_LAST/analysis/{task}/{folder}/{task}_prompt_sensitivity_runs.csv').open()))
groups = defaultdict(list)
alphabets = defaultdict(set)
for row in runs:
    run = Path(row['run'])
    paths = list(run.glob('selected_events_robot_*.csv')) or list(run.parent.glob(f'robots_*/{run.name}/selected_events_robot_*.csv'))
    if not paths: raise ValueError(f'Missing selected events: {run}')
    counts = Counter()
    for path in paths:
        for record in csv.DictReader(path.open()):
            event = normalize(record.get('selected_event', '').strip())
            if event and event.lower() != 'none': counts[event] += 1
    prompt = 0 if row['collision_control'] == 'baseline' else int(row['prompt'])
    groups[row['task'], row['collision_control'], prompt].append(counts)
    alphabets[row['task']].update(counts)
summary, tables = [], []
for task in ['exploration', 'patrolling', 'delivery']:
    prompts = sorted({key[2] for key in groups if key[0] == task and key[1] != 'baseline'})
    for mode in ['without_fixed_spec', 'with_fixed_spec']:
        columns = [('baseline', 0)] + [(mode, p) for p in prompts]
        lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
                 rf'\caption{{Controllable event counts per run for {task}, '+mode.replace('_', r'\_')+r'. Entries are mean (population variance), pooling all robots in each run and combining generations and seeds. Missing events count as zero. Baseline uses ten runs.}}',
                 rf'\label{{tab:{task}-{mode}-event-variance}}',
                 r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l'+'r'*len(columns)+'}', r'\hline',
                 'Event & Baseline'+''.join(f' & P{p}' for p in prompts)+r' \\', r'\hline',
                 'Runs'+''.join(f' & {len(groups[task,m,p])}' for m,p in columns)+r' \\']
        for event in sorted(alphabets[task]):
            cells = [event.replace('_', r'\_')]
            for m,p in columns:
                values = [c[event] for c in groups[task,m,p]]
                mean = statistics.mean(values) if values else None
                variance = statistics.pvariance(values) if values else None
                summary.append(dict(task=task,collision_control=m,prompt=p,event=event,runs=len(values),mean_count=mean,variance_count=variance))
                cells.append(f'{mean:.2f} ({variance:.2f})' if values else '--')
            lines.append(' & '.join(cells)+r' \\')
        lines += [r'\hline', r'\end{tabular}}', r'\end{table*}']
        table='\n'.join(lines)+'\n'
        (OUT/f'{task}_{mode}_event_mean_variance.tex').write_text(table)
        tables.append(table)
unique = {(r['task'],r['collision_control'],r['prompt'],r['event']):r for r in summary}
write_csv(OUT/'event_mean_variance.csv',list(unique.values()))
(OUT/'event_mean_variance_tables.tex').write_text('% Requires \\usepackage{graphicx}\n'+ '\n'.join(tables))
(OUT/'README.md').write_text('Mean and population variance of selected controllable event COUNTS per run, pooling robots and weighting runs equally. Includes all currently saved prompt runs and ten baseline runs per task. P7 has one generation; P1–P6 combine available generations. These are counts, not event percentages. Different task durations affect counts. Parentheses show variance, not standard deviation. Header-only logs contribute zero counts.\n')
print(OUT)
for task in alphabets:
 print(task,{f'{m}:P{p}':len(v) for (t,m,p),v in groups.items() if t==task})
