"""Synchronize each prompt's best saved supervisor with target_approach."""
import csv
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / 'llm_part'))
import nadzoru_sync
import supervisor_xml_to_yaml

OUTPUT = BASE / 'best_patrolling_target_approach'
OUTPUT.mkdir(exist_ok=True)
SPEC = ROOT / 'automata/baseline_automata/patrolling/target_approach.xml'
Automaton = nadzoru_sync.import_automaton(nadzoru_sync.DEFAULT_NADZORU_ROOT)
rows = list(csv.DictReader((BASE / 'analysis/patrolling/comparison/patrolling_prompt_sensitivity_runs.csv').open()))
groups = defaultdict(list)
for row in rows:
    groups[(int(row['prompt']), row['yaml'])].append(row)
summary = []
for prompt in range(1, 7):
    candidates = []
    for (p, path), runs in groups.items():
        if p != prompt:
            continue
        successful = [r for r in runs if r['success'].lower() == 'true']
        progression = statistics.mean(float(r['task_success_pct']) for r in runs)
        success_rate = len(successful) / len(runs)
        completion = statistics.mean(float(r['duration_s']) for r in successful) if successful else float('inf')
        collisions = statistics.mean(float(r['collisions']) for r in runs)
        candidates.append(((-round(progression, 10), -success_rate, completion, collisions, int(runs[0]['generation'])), runs))
    score, runs = min(candidates, key=lambda item: item[0])
    row = runs[0]
    selected = Path(row['yaml'])
    folder = OUTPUT / f'prompt_{prompt}'
    folder.mkdir(exist_ok=True)
    result_root = BASE / 'resulting_automata/patrolling' / row['collision_control']
    saved_s = result_root / 'S' / (selected.stem + '.xml')
    saved_g = result_root / 'G_K' / (selected.stem.replace('S_', 'G_', 1) + '.xml')
    if saved_s.exists():
        # Confirm this XML is exactly the controller that was evaluated.
        original = yaml.safe_load(selected.read_text())
        decoded = supervisor_xml_to_yaml.xml_to_supervisor_data(saved_s)
        assert decoded == original, f'Saved XML/YAML mismatch: {selected}'
        supervisor = nadzoru_sync.load_automata(Automaton, [saved_s])[0]
        plant = nadzoru_sync.load_automata(Automaton, [saved_g])[0]
    else:
        # P7 runtime model: omitted controllables are disabled; sensor events are outside K.
        original = yaml.safe_load(selected.read_text())
        assert original['sup_data'] == [1, 'EV_rotate_clockwise', 0, 0]
        model = ET.Element('model', version='0.0', type='FSA', id='clockwise_only_specification')
        data = ET.SubElement(model, 'data')
        ET.SubElement(data, 'state', id='0', name='rotating', initial='True', marked='True', x='0', y='0')
        for i, (event, control) in enumerate(zip(original['events'], original['ev_controllable'])):
            if control:
                ET.SubElement(data, 'event', id=str(i), name=event.removeprefix('EV_'), controllable='True', observable='True')
                if event == 'EV_rotate_clockwise':
                    rotate_id = str(i)
        ET.SubElement(data, 'transition', source='0', target='0', event=rotate_id)
        saved_s = folder / 'source_clockwise_only.xml'
        ET.ElementTree(model).write(saved_s, encoding='utf-8', xml_declaration=True)
        plants = [ROOT / 'automata/baseline_automata/patrolling' / name for name in ('motion_plant.xml', 'obstacle_sensor.xml', 'color_sensor.xml')]
        supervisor = nadzoru_sync.load_automata(Automaton, [saved_s])[0]
        plant = nadzoru_sync.synchronize(Automaton, nadzoru_sync.load_automata(Automaton, plants))
    target = nadzoru_sync.load_automata(Automaton, [SPEC])[0]
    specification = Automaton.synchronization(plant, supervisor, target)
    synthesized = Automaton.sup_c(plant, specification)
    for name, automaton in [('G.xml', plant), ('K.xml', specification), ('S.xml', synthesized)]:
        assert automaton.save(str(folder / name))
    info = dict(prompt=prompt, generation=int(row['generation']), collision_control=row['collision_control'],
                source_yaml=str(selected), mean_task_progression_pct=-score[0], runs=len(runs),
                source_states=len(supervisor.states), synchronized_states=len(specification.states),
                resulting_states=len(synthesized.states), specification=str(SPEC))
    if synthesized.states:
        output_yaml = folder / f'S_prompt_{prompt}_target_approach.yaml'
        supervisor_xml_to_yaml.convert(folder / 'S.xml', output_yaml)
        info['result_yaml'] = str(output_yaml)
        info['status'] = 'generated'
        output_yaml.with_suffix('.prompt.txt').write_text(row['prompt_text'] + '\n')
        output_yaml.with_suffix('.collision_control.txt').write_text('fixed' if row['collision_control'] == 'with_fixed_spec' else 'none')
    else:
        info['result_yaml'] = None
        info['status'] = 'empty supervisor: no controllable nonblocking solution'
    summary.append(info)
    print(json.dumps(info), flush=True)
(OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUTPUT / 'README.md').write_text('Best patrolling supervisors synchronized with target_approach.xml.\n\n'
    'Selection: highest mean task progression, then full completion rate, successful completion time, and collisions. '
    'Selection compares both collision-control series for P1–P6; P7 is excluded.\n\n'
    'For each prompt: K = Sync(original plant, evaluated supervisor, target_approach); S = SupC(original plant, K). '
    'Original evaluated controllers and their collision constraints are preserved. No simulation was run. '
    'An empty supervisor has no valid runtime YAML. See summary.json for sources and state counts.\n')
