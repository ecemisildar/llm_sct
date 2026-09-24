"""Real experiment plots styled after RESULTS_LAST/analysis prompt plots."""
import os
os.environ['MPLCONFIGDIR'] = '/tmp/leo-real-combined-mpl'
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'plots'
OUT.mkdir(exist_ok=True)
trials = list(csv.DictReader((ROOT/'trial_summary.csv').open()))
coverage = list(csv.DictReader((ROOT.parent/'coverage'/'trial_summary.csv').open()))
COLORS = ['#808080','#7B2CBF']
plt.rcParams.update({'font.size':20})

def grouped(rows,task,field,successful=False):
    return [[float(r[field]) for r in rows if r['task']==task and r['condition']==c and
             (not successful or r['success_under_5_min_rule']=='1')]
            for c in ('baseline','llm')]

def style(ax,title,ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1,2],['Baseline','LLM'])
    ax.tick_params(axis='x',labelrotation=45)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

def distribution(ax,values,title,ylabel,log=False,percent=False):
    boxes=ax.boxplot(values,positions=[1,2],widths=.26,showmeans=True,patch_artist=True)
    for box,color in zip(boxes['boxes'],COLORS):
        box.set_facecolor(color)
        box.set_alpha(.55)
    for i,(v,color) in enumerate(zip(values,COLORS),1):
        ax.scatter([i]*len(v),v,s=15,alpha=.55,color=color)
    ax.axhline(statistics.mean(values[0]),color='#333333',linestyle='--',linewidth=1.8)
    style(ax,title,ylabel)
    if percent:
        ax.set_ylim(0,105)
    if log:
        ax.set_yscale('symlog',linthresh=1,linscale=.5)
        maximum=max(max(v) for v in values)
        ticks=[0,1]
        power=10
        while power<=max(10,maximum):
            ticks.append(power)
            power*=10
        ax.set_yticks(ticks,[str(t) for t in ticks])
        ax.set_ylim(0,max(10,maximum)*1.15)

def save(fig,name,title,note):
    for axis in fig.axes:
        axis.set_title('')
        axis.set_xlabel(axis.get_xlabel().lower())
        axis.set_ylabel(axis.get_ylabel().lower())
        axis.set_xticks([1,2],['baseline','LLM'])
    handles=[Patch(facecolor=COLORS[0],alpha=.55,label='baseline'),
             Patch(facecolor=COLORS[1],alpha=.55,label='LLM'),
             Line2D([0],[0],color='#333333',linestyle='--',linewidth=1.8,label='baseline mean')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=20)
    fig.tight_layout(rect=(0,.15,1,1))
    fig.savefig(OUT/name,dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/name.replace('.png','_no_title.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def pending_collisions(ax):
    ax.set_title('Collisions')
    ax.text(.5,.5,'Proximity counts withdrawn\nPhysical contacts not validated',
            ha='center',va='center',transform=ax.transAxes,fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig,axes=plt.subplots(1,2,figsize=(13,5.5))
distribution(axes[0],grouped(coverage,'exploration','final_coverage_pct'),'Coverage distribution','Coverage (%)',percent=True)
distribution(axes[1],grouped(trials,'exploration','total_collision_estimates'),'Collision distribution','collision count')
save(fig,'exploration_collisions_coverage.png','Exploration: 10 baseline and 10 LLM trials',
     'Points: individual trials; triangles: means. Collision rule: robot markers within 0.60 m or marker within 0.10 m of a wall; completed robots excluded.')

fig,axes=plt.subplots(1,3,figsize=(18,5.5))
values=grouped(trials,'patrolling','task_progression_pct')
axes[0].bar([1,2],[statistics.mean(v) for v in values],width=.28,color=COLORS,alpha=.7)
axes[0].axhline(statistics.mean(values[0]),color='#333333',linestyle='--',linewidth=1.8)
style(axes[0],'Task progression','Mean task progression (%)')
axes[0].set_ylim(0,105)
distribution(axes[1],grouped(trials,'patrolling','recording_duration_completion_proxy_s',True),
             'Completion time proxy','completion time (s)')
distribution(axes[2],grouped(trials,'patrolling','total_collision_estimates'),'Collision distribution','collision count')
save(fig,'patrolling_completion_collisions_progression.png','Patrolling: 10 baseline and 10 LLM trials',
     'Clips < 300 s receive all reaches. Time: successful duration proxy (8 trials/condition). Collisions: marker pair <= 0.60 m / wall <= 0.10 m; completed robots excluded.')
print('Saved both requested PNGs and matching no-title versions in',OUT)
