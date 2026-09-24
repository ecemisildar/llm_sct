"""Primitive-only P7 experiments in isolated, task-specific Gazebo worlds."""
import ast
import csv
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = ROOT / 'RESULTS_LAST'
sys.path.insert(0, str(ROOT / 'llm_part'))
import nadzoru_sync
import supervisor_xml_to_yaml

BATCH = HERE / ('batch_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
BATCH.mkdir()
RESULTS = BATCH / 'results'
RESULTS.mkdir()
(RESULTS / 'YAML').symlink_to(BATCH / 'YAML', target_is_directory=True)
print('Batch:', BATCH, flush=True)
summary = []
for task, primitive, prompt in [('exploration', 'move_forward', 'Move only forward'),
                                 ('patrolling', 'rotate_clockwise', 'Only rotate clockwise.')]:
    baseline = ROOT / 'automata/baseline_automata' / task
    plants = [baseline / 'motion_plant.xml', baseline / 'obstacle_sensor.xml']
    if task == 'patrolling': plants.append(baseline / 'color_sensor.xml')
    alphabet = {}
    for plant in plants:
        for event in ET.parse(plant).getroot().find('data').findall('event'):
            alphabet[event.get('name')] = event.get('controllable').lower() == 'true'
    folder = BATCH / 'models' / task
    folder.mkdir(parents=True)
    model = ET.Element('model', version='0.0', type='FSA', id='primitive_only_specification')
    data = ET.SubElement(model, 'data')
    ET.SubElement(data, 'state', id='0', name='active', initial='True', marked='True', x='0', y='0')
    for index, event in enumerate(sorted(e for e, control in alphabet.items() if control)):
        ET.SubElement(data, 'event', id=str(index), name=event, controllable='True', observable='True')
        if event == primitive:
            ET.SubElement(data, 'transition', source='0', target='0', event=str(index))
    primitive_spec = folder / 'primitive_only_specification.xml'
    ET.ElementTree(model).write(primitive_spec, encoding='utf-8', xml_declaration=True)
    for mode in ('without_fixed_spec', 'with_fixed_spec'):
        specs = [primitive_spec]
        if mode == 'with_fixed_spec': specs.append(baseline / 'collision_avoidance.xml')
        args = nadzoru_sync.build_parser().parse_args([])
        args.input_dir = []
        args.input_file = [str(p) for p in plants + specs]
        args.plant = [p.stem for p in plants]
        args.spec = [p.stem for p in specs]
        args.output_dir = str(folder / mode)
        g, k, s = nadzoru_sync.run(args)
        output = BATCH / 'YAML' / task / mode / 'prompt_7/generation_1' / f'S_p7_{task}_{mode}.yaml'
        output.parent.mkdir(parents=True)
        supervisor_xml_to_yaml.convert(s, output)
        output.with_suffix('.prompt.txt').write_text(prompt + '\n')
        output.with_suffix('.collision_control.txt').write_text('fixed\n' if mode == 'with_fixed_spec' else 'none\n')
        summary.append(dict(task=task, primitive=primitive, collision_control=mode, yaml=str(output)))
(BATCH / 'controllers.json').write_text(json.dumps(summary, indent=2) + '\n')
runner = ROOT / 'llm_part/run_parallel_llm_simulations.py'
command = [sys.executable, str(runner), '--mission', 'exploration', '--mission', 'patrolling',
           '--yaml-root', str(BATCH / 'YAML'), '--results-root', str(RESULTS),
           '--seeds', ','.join(str(s) for s in range(1001, 1011)),
           '--exploration-duration', '60', '--patrolling-duration', '300',
           '--parallel', '6', '--domain-base', '181',
           '--partition-prefix', BATCH.name + '_p7_task_worlds', '--color-order', 'red,green,blue']
print('Running 40 jobs with task-specific launch files and unique partitions.', flush=True)
subprocess.run(command, check=True)

# Verify all runs belong to their task environments before including their results.
statuses = list(RESULTS.glob('*/**/SAVE_STATUS.txt'))
assert len(statuses) == 40, len(statuses)
for status in statuses:
    run = status.parent
    task = run.relative_to(RESULTS).parts[0]
    assert 'Saving OK' in status.read_text(), status
    world = (run / 'sim_world.sdf').read_text()
    assert 'generic_delivery_box' not in world, f'Unexpected delivery world: {run}'
    bumps = list(csv.DictReader((run / 'bumps_global.csv').open()))
    assert not any('generic_delivery_box' in r.get('key', '') for r in bumps), f'Interference: {run}'
    expected = 'EV_move_forward' if task == 'exploration' else 'EV_rotate_clockwise'
    paths = list(run.glob('selected_events_robot_*.csv'))
    if not paths: paths = list(run.parent.glob(f'robots_*/{run.name}/selected_events_robot_*.csv'))
    assert paths, f'Missing selected-event logs: {run}'
    for path in paths:
        for row in csv.DictReader(path.open()):
            event = row.get('selected_event', '').strip()
            assert event in {expected, '', 'none'}, (path, event)
(BATCH / 'VALIDATION.txt').write_text('40/40 runs saved. Correct task-world files; no delivery-box contacts; selected controllable events restricted to each task primitive. Standard runtime recovery remains active.\n')

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leo-prompt-replot-mpl')
for task in ('exploration', 'patrolling'):
    analysis = RESULTS / 'analysis' / task
    command = [sys.executable, str(ROOT / 'evaluation/evaluation/analyze_prompt_sensitivity.py'),
               '--task', task, '--experiment-root', str(RESULTS), '--output-dir', str(analysis),
               '--prompts', '7', '--generations', '1', '--seeds'] + [str(s) for s in range(1001, 1011)] + ['--no-baseline']
    subprocess.run(command, check=True)
    main = EXPERIMENT / 'analysis' / task / ('with_fixed_spec' if task == 'exploration' else 'comparison')
    for suffix in ('prompts', 'prompt_summary', 'generation_summary', 'prompt_sensitivity_runs', 'prompt_sensitivity_events'):
        target = main / f'{task}_{suffix}.csv'
        with target.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            original = [r for r in reader if r['prompt'] != '7']
        additions = list(csv.DictReader((analysis / target.name).open()))
        with target.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(original + additions)

# Reuse the shared figure code with an accurate title for the uneven run counts.
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
source = ROOT / 'evaluation/evaluation/analyze_prompt_sensitivity.py'
parsed = ast.parse(source.read_text())
main_function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
start = next(i for i, n in enumerate(main_function.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'labels' for t in n.targets))
renderer = ast.FunctionDef(name='render_saved', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=main_function.body[start:], decorator_list=[])
helper = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'lowercase_plot_labels')
for node in ast.walk(renderer):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'suptitle':
        node.args[0] = ast.Constant('Prompt sensitivity: P1–P6 (5 generations × 10 seeds); P7 (1 generation × 10 seeds)')
module = ast.fix_missing_locations(ast.Module(body=[helper, renderer], type_ignores=[]))
plt.rcParams.update({'font.size': 20})
for task in ('exploration', 'patrolling'):
    output = EXPERIMENT / 'analysis' / task / ('with_fixed_spec' if task == 'exploration' else 'comparison')
    rows = list(csv.DictReader((output / f'{task}_prompt_sensitivity_runs.csv').open()))
    for row in rows: row['prompt'] = int(row['prompt'])
    assert len([r for r in rows if r['prompt'] == 7]) == 20
    baseline_summary = next(csv.DictReader((output / f'{task}_baseline_summary.csv').open()))
    baseline = [dict(coverage_pct=baseline_summary['coverage_mean_pct'], collisions=baseline_summary['collisions_mean'],
                     duration_s=baseline_summary['duration_all_mean_s'], task_success_pct=baseline_summary['task_success_mean_pct'])]
    namespace = dict(plt=plt, Patch=Patch, Line2D=Line2D, csv=csv, statistics=statistics,
                     FIGURE_FONT_SIZE=20, rows=rows, baseline_rows=baseline, output=output,
                     collision_controls=['without_fixed_spec', 'with_fixed_spec'],
                     args=SimpleNamespace(task=task, prompts=list(range(1, 8)), generations=list(range(1, 6)),
                                          seeds=list(range(1001, 1011)), allow_incomplete=True))
    exec(compile(module, str(source), 'exec'), namespace)
    namespace['render_saved']()
print('DONE: 40 isolated P7 runs validated; both main figures updated.', flush=True)
