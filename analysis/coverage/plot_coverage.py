#!/usr/bin/env python3
"""Render standalone figures with system Matplotlib (run with python3 -s)."""
import csv
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leo-real-coverage-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {'baseline': '#2878b5', 'llm': '#e47726'}
ROBOT_COLORS = ['#009e73', '#cc79a7', '#0072b2']


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def curve(directory):
    data = rows(directory / 'coverage_timeseries.csv')
    return [float(r['time_s']) for r in data], [float(r['coverage_pct']) for r in data]


def at(times, values, t):
    index = np.searchsorted(times, t, side='right') - 1
    return float(values[max(0, index)])


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def trial_plot(directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for row in rows(directory / 'coverage_visited_cells.csv'):
        ax.add_patch(Rectangle((float(row['cell_min_x']), float(row['cell_min_y'])), 1, 1,
                               facecolor='#d7ebd4', edgecolor='none'))
    trajectories = rows(directory / 'coverage_paths.csv')
    for marker in range(3):
        data = [r for r in trajectories if r['robot'] == f'robot_{marker}']
        x, y, last = [], [], None
        for row in data:
            t = float(row['elapsed_s'])
            if last is not None and t - last > .4 + 1e-6:
                x.append(float('nan')); y.append(float('nan'))
            x.append(float(row['x'])); y.append(float(row['y'])); last = t
        ax.plot(x, y, color=ROBOT_COLORS[marker], lw=.75, alpha=.85, label=f'Robot {marker}')
        if data:
            ax.scatter(float(data[0]['x']), float(data[0]['y']), color=ROBOT_COLORS[marker], s=25)
    ax.set(xlim=(0, 4), ylim=(0, 3), xlabel='x (m)', ylabel='y (m)', title='Visited 1 m cells and detected paths')
    ax.set_xticks(range(5)); ax.set_yticks(range(4)); ax.grid(color='#999999', lw=.6)
    ax.set_aspect('equal'); ax.legend(fontsize=8, loc='upper left')
    times, values = curve(directory)
    axes[1].step(times, values, where='post', color=COLORS[summary['condition']], lw=1.7)
    axes[1].set(xlim=(0, summary['duration_s']), ylim=(0, 105), xlabel='Video elapsed time (s)',
                ylabel='Team coverage (%)', title=f'Final coverage: {summary["final_coverage_pct"]:.2f}%')
    axes[1].grid(alpha=.25)
    detection = ', '.join(f'R{m} {summary["active_span_detection_pct"][str(m)]:.1f}%' for m in range(3))
    fig.suptitle(f'{summary["task"].capitalize()} · {directory.name}\nDetection within observed arena span: {detection}', fontsize=10)
    fig.tight_layout()
    save(fig, directory / 'coverage_map_and_curve')


def main():
    summaries = json.loads((ROOT / 'batch_manifest.json').read_text())
    trials = []
    common = {task: min(s['duration_s'] for s in summaries if s['task'] == task)
              for task in ('exploration', 'patrolling')}
    for summary in summaries:
        directory = ROOT / summary['task'] / Path(summary['video']).stem
        path_rows = rows(directory / 'coverage_paths.csv')
        observation_times = [float(r['time_s']) for r in rows(directory / 'tracking_observations.csv')]
        active_pct, internal_gap, spans = {}, {}, {}
        for marker in range(3):
            times_m = [float(r['elapsed_s']) for r in path_rows if r['robot'] == f'robot_{marker}']
            spans[str(marker)] = [times_m[0], times_m[-1]] if times_m else [None,None]
            denominator = sum(times_m[0]-1e-6 <= t <= times_m[-1]+1e-6 for t in observation_times) if times_m else 0
            active_pct[str(marker)] = 100.*len(times_m)/denominator if denominator else 0.
            internal_gap[str(marker)] = max((b-a for a,b in zip(times_m,times_m[1:])), default=0.)
        summary.update(active_span_detection_pct=active_pct, max_internal_missing_gap_s=internal_gap,
                       observed_arena_spans_s=spans)
        (directory / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        old_robot_rows = {r['robot']: r for r in rows(directory / 'robot_summary.csv')}
        with (directory / 'robot_summary.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['robot','detected_in_arena_samples','full_recording_in_arena_detection_pct',
                             'observed_span_detection_pct','first_in_arena_detection_s','last_in_arena_detection_s',
                             'longest_internal_inter_detection_interval_s','coverage_pct'])
            for marker in range(3):
                name = f'robot_{marker}'
                count = sum(r['robot']==name for r in path_rows)
                writer.writerow([name,count,summary['detection_pct'][str(marker)],active_pct[str(marker)],
                                 *spans[str(marker)],internal_gap[str(marker)],old_robot_rows[name]['coverage_pct']])
        trial_plot(directory, summary)
        times, values = curve(directory)
        trials.append(dict(task=summary['task'], condition=summary['condition'], video=Path(summary['video']).name,
                           duration_s=summary['duration_s'], final_coverage_pct=summary['final_coverage_pct'],
                           common_cutoff_s=common[summary['task']],
                           coverage_at_common_cutoff_pct=at(times, values, common[summary['task']]),
                           time_to_100_pct_s=summary['time_to_100_pct_s'],
                           robot_0_detection_pct=active_pct['0'], robot_1_detection_pct=active_pct['1'],
                           robot_2_detection_pct=active_pct['2'],
                           robot_0_full_recording_in_arena_detection_pct=summary['detection_pct']['0'],
                           robot_1_full_recording_in_arena_detection_pct=summary['detection_pct']['1'],
                           robot_2_full_recording_in_arena_detection_pct=summary['detection_pct']['2'],
                           max_internal_missing_gap_s=max(internal_gap.values()),
                           max_full_recording_missing_gap_s=max(summary['longest_missing_gap_s'].values()),
                           result_directory=str(directory)))
    with (ROOT / 'trial_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trials[0])); writer.writeheader(); writer.writerows(trials)
    aggregates = []
    report = ['# Real robot video coverage', '',
              'Coverage uses the simulation geometry directly: 1 m square cells, circular radius 0.36 m, '
              '0.2 s pose samples, and 0.5 s metric samples. Team coverage is the number of distinct touched '
              'cells divided by 12 floor cells, multiplied by 100. A touched cell counts in full; this is '
              'a grid visitation metric, not the exact fraction of floor swept by the robot.', '',
              'The 800 × 600 cropped arena represents approximately 4 × 3 m (12 m²). '
              'ArUco DICT_4X4_50 IDs 0, 1, 2 identify the robots; pillar markers are ignored. '
              'Missing markers are recovered from timestamp-aligned original frames and mapped through '
              'the saved crop homography into the same arena coordinates. '
              'Pixel marker centres are converted to metres with origin at the bottom left. '
              'The radius 0.36 m is retained from simulation for consistency and is not a measured physical footprint.', '',
              'Video time zero is the first frame. No task start/end event is inferred. '
              'The full recording is analyzed. Missing detections contribute no additional coverage and '
              'are not interpolated. Tops of robot markers approximate robot centres; no marker-height '
              'parallax correction is available. All 12 floor cells form the denominator; no surveyed '
              'obstacle mask is subtracted.', '',
              'Final values can reflect unequal recording durations. Each task also has a comparison '
              'at the shortest recording duration shared by all its trials.', '',
              '| Task | Condition | Trials | Common cutoff (s) | Coverage at cutoff, mean ± SD (%) | Final mean (%) |',
              '|---|---|---:|---:|---:|---:|']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for task_index, task in enumerate(('exploration', 'patrolling')):
        cutoff = common[task]
        ax_curve, ax_bar, ax_time = axes[task_index]
        for condition in ('baseline', 'llm'):
            group = [r for r in trials if r['task'] == task and r['condition'] == condition]
            data = [r['coverage_at_common_cutoff_pct'] for r in group]
            final = [r['final_coverage_pct'] for r in group]
            mean, sd = statistics.mean(data), statistics.stdev(data) if len(data) > 1 else 0.
            full_times = [r['time_to_100_pct_s'] for r in group if r['time_to_100_pct_s'] is not None]
            aggregates.append(dict(task=task, condition=condition, n=len(group), common_cutoff_s=cutoff,
                                   coverage_at_cutoff_mean_pct=mean, coverage_at_cutoff_sd_pct=sd,
                                   final_coverage_mean_pct=statistics.mean(final),
                                   trials_reaching_100_pct=len(full_times),
                                   mean_time_to_100_pct_s=statistics.mean(full_times) if full_times else None))
            report.append(f'| {task} | {condition} | {len(group)} | {cutoff:.2f} | {mean:.2f} ± {sd:.2f} | {statistics.mean(final):.2f} |')
            grid = np.arange(0, cutoff + 1e-8, .5)
            samples = []
            for row in group:
                times, values = curve(Path(row['result_directory']))
                samples.append([at(times, values, t) for t in grid])
            average = np.mean(samples, axis=0)
            spread = np.std(samples, axis=0, ddof=1) if len(group) > 1 else np.zeros_like(average)
            ax_curve.step(grid, average, where='post', color=COLORS[condition], label=f'{condition} (n={len(group)})')
            ax_curve.fill_between(grid, np.maximum(0, average-spread), np.minimum(100, average+spread),
                                  step='post', alpha=.15, color=COLORS[condition])
            index = ('baseline', 'llm').index(condition)
            ax_bar.bar(index, mean, yerr=sd, capsize=5, color=COLORS[condition], alpha=.8)
            ax_bar.scatter([index] * len(data), data, s=18, c='black', alpha=.5)
            if full_times:
                ax_time.scatter([index] * len(full_times), full_times, c=COLORS[condition], alpha=.7, s=30)
                ax_time.plot([index-.2, index+.2], [statistics.mean(full_times)]*2, color='black', lw=2)
            ax_time.text(index, -.14, f'{len(full_times)}/{len(group)} reached',
                         transform=ax_time.get_xaxis_transform(), ha='center', fontsize=9)
        ax_curve.set(title=f'{task.capitalize()}: mean coverage ± 1 SD', xlim=(0, cutoff), ylim=(0,105),
                      xlabel='Video elapsed time (s)', ylabel='Team coverage (%)')
        ax_curve.legend(fontsize=9); ax_curve.grid(alpha=.2)
        ax_bar.set(title=f'Coverage at common cutoff ({cutoff:.2f} s)', ylim=(0, 110), ylabel='Team coverage (%)')
        ax_bar.set_xticks([0, 1], ['Baseline', 'LLM']); ax_bar.grid(axis='y', alpha=.2)
        ax_time.set(title='Time to 100% (lines show mean)', xlim=(-.5,1.5), ylabel='Video elapsed time (s)')
        ax_time.set_xticks([0,1], ['Baseline','LLM']); ax_time.grid(axis='y', alpha=.2)
    fig.tight_layout(); save(fig, ROOT / 'baseline_vs_llm_coverage')
    with (ROOT / 'condition_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0])); writer.writeheader(); writer.writerows(aggregates)
    report.extend(['', '## Tracking quality', '',
                   'Detection quality below is calculated between each robot’s first and last in-arena '
                   'observation. Robots are sometimes physically removed during patrolling; their absence '
                   'after removal is excluded from this quality denominator. First/last spans and whole-recording '
                   'in-arena detection rates are retained in the per-trial JSON and summary CSV.', '',
                   '| Task | Video | Minimum active-span detection (%) | Longest internal interval (s) |',
                   '|---|---|---:|---:|'])
    for row in trials:
        minimum = min(row[f'robot_{m}_detection_pct'] for m in range(3))
        report.append(f'| {row["task"]} | {row["video"]} | {minimum:.1f} | {row["max_internal_missing_gap_s"]:.2f} |')
    report.extend(['', 'Missing detections can undercount visited cells; inspect `tracking_observations.csv` and '
                   '`detection_preview.jpg` in each trial folder. Coverage resolution is 8.33 percentage '
                   'points per cell, so saturation at 100% can hide differences between trajectories.', '',
                   'Files: `trial_summary.csv`, `condition_summary.csv`, `baseline_vs_llm_coverage.png`/`.svg`, '
                   'and per-trial CSVs and `coverage_map_and_curve.png`/`.svg`.', '',
                   'The filename `exp_lm` is grouped as LLM; `llm_best` is also LLM. '
                   'Original and cropped videos are read only.', '',
                   'Reproduce: `python3 analysis/coverage/analyze_coverage.py`, then '
                   '`python3 -s analysis/coverage/plot_coverage.py` from the experiment directory.', ''])
    (ROOT / 'REPORT.md').write_text('\n'.join(report))
    (ROOT / 'batch_manifest.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(f'Wrote {len(trials)} trial figures and baseline/LLM summaries.', flush=True)


if __name__ == '__main__':
    main()
