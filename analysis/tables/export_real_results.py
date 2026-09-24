"""Export real-robot figure data as Markdown, CSV, and LaTeX tables."""
import csv
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent

def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

trials = read(ROOT / 'task_metrics/trial_summary.csv')
coverage = read(ROOT / 'coverage/trial_summary.csv')

def stats(rows, field):
    values = [float(row[field]) for row in rows]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

def display(pair):
    return f'{pair[0]:.2f} ± {pair[1]:.2f}'

def export(task, headers, formatted, numeric):
    with (OUT / f'{task}_real_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numeric[0]))
        writer.writeheader()
        writer.writerows(numeric)
    markdown = '| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join('---' for _ in headers) + ' |\n'
    markdown += '\n'.join('| ' + ' | '.join(row) + ' |' for row in formatted) + '\n'
    (OUT / f'{task}_real_results.md').write_text(markdown)
    def tex(value):
        return value.replace('%', r'\%').replace('±', r'$\pm$')
    latex = '\\begin{table}[htbp]\n\\centering\n'
    latex += '\\caption{' + task.capitalize() + ' real-robot results (mean $\\pm$ sample standard deviation).}\n'
    latex += '\\label{tab:real-' + task + '}\n'
    latex += '\\begin{tabular}{' + 'l' + 'r' * (len(headers) - 1) + '}\n\\hline\n'
    latex += ' & '.join(tex(h) for h in headers) + ' \\\\\n\\hline\n'
    latex += ''.join(' & '.join(tex(v) for v in row) + ' \\\\\n' for row in formatted)
    latex += '\\hline\n\\end{tabular}\n\\end{table}\n'
    (OUT / f'{task}_real_results.tex').write_text(latex)
    return markdown

formatted = []
numeric = []
for condition in ('baseline', 'llm'):
    selected = [r for r in trials if r['task'] == 'exploration' and r['condition'] == condition]
    selected_coverage = [r for r in coverage if r['task'] == 'exploration' and r['condition'] == condition]
    assert {r['video'] for r in selected} == {r['video'] for r in selected_coverage}
    cov = stats(selected_coverage, 'final_coverage_pct')
    col = stats(selected, 'total_collision_estimates')
    total = sum(int(r['total_collision_estimates']) for r in selected)
    label = 'Baseline' if condition == 'baseline' else 'LLM'
    formatted.append([label, str(len(selected)), display(cov), display(col), str(total)])
    numeric.append(dict(condition=condition, trials=len(selected), coverage_mean_pct=cov[0], coverage_sd_pct=cov[1],
                        collisions_mean=col[0], collisions_sd=col[1], collisions_total=total))
exploration = export('exploration', ['Controller', 'Trials', 'Final coverage (%)', 'Collisions/trial', 'Total collisions'], formatted, numeric)

formatted = []
numeric = []
for condition in ('baseline', 'llm'):
    selected = [r for r in trials if r['task'] == 'patrolling' and r['condition'] == condition]
    successful = [r for r in selected if r['success_under_5_min_rule'] == '1']
    progress = stats(selected, 'task_progression_pct')
    completion = stats(successful, 'recording_duration_completion_proxy_s')
    col = stats(selected, 'total_collision_estimates')
    total = sum(int(r['total_collision_estimates']) for r in selected)
    label = 'Baseline' if condition == 'baseline' else 'LLM'
    formatted.append([label, str(len(selected)), f'{len(successful)}/{len(selected)} ({100*len(successful)/len(selected):.0f}%)',
                      display(progress), display(completion), display(col), str(total)])
    numeric.append(dict(condition=condition, trials=len(selected), successful_trials=len(successful), success_rate_pct=100*len(successful)/len(selected),
                        task_progression_mean_pct=progress[0], task_progression_sd_pct=progress[1],
                        successful_completion_proxy_mean_s=completion[0], successful_completion_proxy_sd_s=completion[1],
                        collisions_mean=col[0], collisions_sd=col[1], collisions_total=total))
patrolling = export('patrolling', ['Controller', 'Trials', 'Successful trials', 'Task progression (%)', 'Completion proxy (s)', 'Collisions/trial', 'Total collisions'], formatted, numeric)
notes = ('Values are mean ± sample standard deviation across trials. Final coverage is measured over each complete recording, matching the figure.\n\n'
         'Patrolling success uses the existing recording-duration rule (<300 seconds); successful recordings receive all nine reaches. '
         'Completion proxy is recording duration for the eight successful trials per controller, not an exact last-arrival timestamp.\n\n'
         'Collision counts use the current user-approved marker-proximity metric: robot markers ≤0.60 m apart or a marker ≤0.10 m from an inner wall edge. '
         'Pillars, completed-robot parking, and handling are excluded as documented in task_metrics/proximity_collisions/README.md.\n\n'
         'Sources: task_metrics/trial_summary.csv and coverage/trial_summary.csv; identical to task_metrics/plot_combined_results.py. '
         'Reproduce: python3 analysis/tables/export_real_results.py from the leo_real_experiments package directory.\n')
(OUT / 'real_robot_results.md').write_text('# Real-robot exploration\n\n' + exploration + '\n# Real-robot patrolling\n\n' + patrolling + '\n' + notes)
print(exploration)
print(patrolling)
