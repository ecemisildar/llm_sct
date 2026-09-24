"""Regenerate sensitivity PNGs from saved CSVs using the shared plotting code."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/leo-prompt-replot-mpl')
import ast
import csv
import statistics
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('analyze_prompt_sensitivity.py')
parsed=ast.parse(SOURCE.read_text())
main=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='main')
start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='labels' for t in n.targets))
plot_main=ast.FunctionDef(name='render_saved',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=main.body[start:],decorator_list=[])
helper=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='lowercase_plot_labels')
module=ast.fix_missing_locations(ast.Module(body=[helper,plot_main],type_ignores=[]))
plt.rcParams.update({'font.size':20})
for csv_path in sorted((ROOT/'RESULTS_LAST/analysis').glob('**/*prompt_sensitivity_runs.csv')):
    task=csv_path.name.split('_prompt_sensitivity')[0]
    if task not in {'exploration','patrolling'}:
        continue
    rows=list(csv.DictReader(csv_path.open()))
    for row in rows:
        row['prompt']=int(row['prompt'])
    output=csv_path.parent
    baseline_path=output/f'{task}_baseline_summary.csv'
    baseline=[]
    if baseline_path.exists():
        summary=next(csv.DictReader(baseline_path.open()))
        baseline=[{'coverage_pct':summary['coverage_mean_pct'],
                   'collisions':summary['collisions_mean'],
                   'duration_s':summary['duration_all_mean_s'],
                   'task_success_pct':summary['task_success_mean_pct']}]
    modes=[m for m in ['without_fixed_spec','with_fixed_spec','llm_collision','fixed_collision'] if any(r['collision_control']==m for r in rows)]
    args=SimpleNamespace(task=task,prompts=sorted({r['prompt'] for r in rows}),
                         generations=sorted({int(r['generation']) for r in rows}),
                         seeds=sorted({int(r['seed']) for r in rows}),allow_incomplete=(task=='patrolling'))
    namespace=dict(plt=plt,Patch=Patch,Line2D=Line2D,statistics=statistics,csv=csv,
                   FIGURE_FONT_SIZE=20,args=args,rows=rows,baseline_rows=baseline,
                   collision_controls=modes,output=output)
    exec(compile(module,str(SOURCE),'exec'),namespace)
    namespace['render_saved']()
