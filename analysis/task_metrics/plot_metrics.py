import os
os.environ['MPLCONFIGDIR'] = '/tmp/leo-real-metrics-mpl'
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root/'condition_summary.csv').open()))
fig, ax = plt.subplots(1,2,figsize=(10,4))
colors = ['#555555','#2672bb']
for i,condition in enumerate(('baseline','llm')):
    rs = [r for r in rows if r['condition']==condition]
    ax[0].bar([j+(i-.5)*.32 for j in range(2)], [float(r['mean_collision_estimates']) for r in rs],width=.32,label=condition,color=colors[i])
    ax[1].bar(i,float(rs[1]['mean_task_progression_points']),color=colors[i],width=.6)
ax[0].set_xticks([0,1],['Exploration','Patrolling'])
ax[0].set_ylabel('Mean marker proximity episodes / video')
ax[0].legend()
ax[0].set_title('Collisions: marker pair 0.60 m / wall 0.10 m')
ax[1].set_xticks([0,1],['Baseline','LLM'])
ax[1].set_ylim(0,3.2)
ax[1].set_ylabel('Mean ordered points per robot (maximum 3)')
ax[1].set_title('Patrolling: clips < 5 min credited all reaches')
fig.suptitle('Video-derived metrics: approximate 4 × 3 m arena')
fig.tight_layout()
for extension in ('png','svg'):
    fig.savefig(root/f'comparison.{extension}',dpi=180)
