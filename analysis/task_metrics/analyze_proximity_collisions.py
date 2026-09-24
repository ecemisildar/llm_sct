"""User-approved collision proxy: robot markers <=0.60 m; marker/wall <=0.10 m."""
import argparse,csv,json,math,statistics
import cv2
import numpy as np
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'proximity_collisions'

def write(path,rows,fields=None):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--robot-distance',type=float,default=.60);parser.add_argument('--wall-distance',type=float,default=.10);args=parser.parse_args()
 OUT.mkdir(exist_ok=True)
 calibration=json.loads((ROOT.parent/'cropping/arena_calibration.json').read_text())
 wall_calibration=json.loads((ROOT/'white_wall_calibration.json').read_text())
 pixels=cv2.perspectiveTransform(np.array([wall_calibration['source_inner_wall_corners_tl_tr_br_bl_px']],dtype=np.float64),np.array(calibration['source_to_cropped_homography']))[0]
 tl,tr,br,bl=[(q[0]/200,3-q[1]/200) for q in pixels]
 walls={'left':(tl,bl),'right':(tr,br),'top':(tl,tr),'bottom':(bl,br)}
 def wall_distance(p,a,b):
  v=(b[0]-a[0],b[1]-a[1]);u=max(0,min(1,((p[0]-a[0])*v[0]+(p[1]-a[1])*v[1])/(v[0]**2+v[1]**2)))
  q=(a[0]+u*v[0],a[1]+u*v[1]);return math.dist(p,q),q
 trials=list(csv.DictReader((ROOT.parent/'coverage/trial_summary.csv').open()))
 cutoffs={};sources={}
 def cutoff(video,robot,t,reason):
  key=(video,int(robot))
  if key not in cutoffs or t<cutoffs[key]:cutoffs[key]=t;sources[key]=reason
 for r in csv.DictReader((ROOT/'ordered_reach_events.csv').open()):
  if r['color']=='blue':cutoff(r['video'],r['robot'],float(r['elapsed_s']),'Observed ordered blue reach')
 for r in csv.DictReader((ROOT/'collision_candidate_reviews.csv').open()):
  if any(word in r['review_reason'].lower() for word in ('handling','removal','being handled')):
   cutoff(r['video'],r['robot'],max(0,float(r['elapsed_s'])-.6),'Human handling visible by candidate pre-frame; exclude from this time onward')
 # Original-frame review shows this robot already parked after its final
 # blue approach at 78.6 s, although its overhead center stayed outside 1 m.
 cutoff('2026-09-13 13-11-04_llm.mp4',0,78.6,'Reviewed final blue approach followed by completion parking; exclude robot at later close pass')
 events=[];results=[];exclusions=[]
 for trial in trials:
  common={k:trial[k] for k in ('task','condition','video')};frames=defaultdict(dict)
  for r in csv.DictReader((Path(trial['result_directory'])/'coverage_paths.csv').open()):frames[float(r['elapsed_s'])][int(r['robot'].split('_')[-1])]=(float(r['x']),float(r['y']))
  start={}
  # Ignore static placement before each robot's first clear movement.
  for i in range(3):
   samples=[(t,p[i]) for t,p in sorted(frames.items()) if i in p and 0<=p[i][0]<=4 and 0<=p[i][1]<=3]
   if samples:
    t0,p0=samples[0];start[i]=next((t for t,p in samples if math.dist(p,p0)>=.05),float('inf'))
  last={};counts=Counter();cutoff_counts=Counter();trial_events=[]
  for t,positions in sorted(frames.items()):
   positions={i:p for i,p in positions.items() if 0<=p[0]<=4 and 0<=p[1]<=3 and t>=start.get(i,float('inf')) and t<cutoffs.get((trial['video'],i),float('inf'))}
   near=[]
   for i,p in positions.items():
    for wall,(a,b) in walls.items():
     d,q=wall_distance(p,a,b)
     if d<=args.wall_distance:near.append(('wall',i,wall,d,.5,args.wall_distance,q))
    for j,q in positions.items():
     if j>i:
      d=math.dist(p,q)
      if d<=args.robot_distance:near.append(('robot',i,f'robot_{j}',d,.8,args.robot_distance,None))
   for kind,i,other,d,gap,threshold,wall_point in near:
    key=(kind,i,other)
    if key not in last or t-last[key]>gap+1e-8:
     e=dict(**common,elapsed_s=t,contact_type=kind,robot=i,other=other,distance_m=d,threshold_m=threshold,wall_nearest_x_m=wall_point[0] if wall_point else '',wall_nearest_y_m=wall_point[1] if wall_point else '')
     events.append(e);trial_events.append(e);counts[kind]+=1
     if t<=float(trial['common_cutoff_s']):cutoff_counts[kind]+=1
    last[key]=t
  results.append(dict(**common,duration_s=float(trial['duration_s']),robot_collision_estimates=counts['robot'],wall_collision_estimates=counts['wall'],pillar_collision_estimates=0,total_collision_estimates=sum(counts.values()),common_cutoff_s=float(trial['common_cutoff_s']),collisions_at_common_cutoff=sum(cutoff_counts.values()),collisions_per_minute=sum(counts.values())/float(trial['duration_s'])*60,collision_count_status='User-defined marker proximity episodes: 0.60 m pair / 0.10 m wall'))
  for i in range(3):
   exclusions.append(dict(**common,robot=i,first_clear_motion_s=start.get(i) if math.isfinite(start.get(i,float('inf'))) else '',exclude_at_s=cutoffs.get((trial['video'],i),''),exclude_reason=sources.get((trial['video'],i),'No verified completion/handling cutoff available')))
 groups=[]
 for task in ('exploration','patrolling'):
  for condition in ('baseline','llm'):
   rs=[r for r in results if r['task']==task and r['condition']==condition]
   groups.append(dict(task=task,condition=condition,trials=len(rs),total_collision_estimates=sum(r['total_collision_estimates'] for r in rs),mean_collision_estimates=statistics.mean(r['total_collision_estimates'] for r in rs),mean_robot_collision_estimates=statistics.mean(r['robot_collision_estimates'] for r in rs),mean_wall_collision_estimates=statistics.mean(r['wall_collision_estimates'] for r in rs),mean_pillar_collision_estimates=0,mean_collisions_at_common_cutoff=statistics.mean(r['collisions_at_common_cutoff'] for r in rs)))
 event_fields=['task','condition','video','elapsed_s','contact_type','robot','other','distance_m','threshold_m','wall_nearest_x_m','wall_nearest_y_m']
 write(OUT/'collision_events.csv',events,event_fields)
 write(OUT/'trial_summary.csv',results);write(OUT/'condition_summary.csv',groups);write(OUT/'robot_exclusions.csv',exclusions)
 (OUT/'method.json').write_text(json.dumps(dict(robot_marker_distance_m=args.robot_distance,marker_wall_distance_m=args.wall_distance,wall_reference='Distance to inner white-wall edges in ORIGINAL frames; see white_wall_calibration.json. Crop border is not the wall.',pillars_counted=False,sample_interval_s=.2,robot_episode_gap_s=.8,wall_episode_gap_s=.5,setup_policy='Start each robot after 0.05 m observed displacement from its first position',completion_policy='Exclude observed ordered blue completions, reviewed completion parking, and visible handling/removal. Cutoffs documented in robot_exclusions.csv.',limitations='Missing marker observations and unobserved completion times can affect counts; wall distances use approximate arena calibration. These are user-defined proximity episodes, not verified physical contacts.'),indent=2))
 # Publish new proxy totals into the combined metric tables, preserving task scores.
 combined=list(csv.DictReader((ROOT/'trial_summary.csv').open()));lookup={r['video']:r for r in results}
 for r in combined:
  for k,v in lookup[r['video']].items():
   if k not in ('task','condition','video','duration_s'):r[k]=v
 write(ROOT/'trial_summary.csv',combined)
 combined_groups=list(csv.DictReader((ROOT/'condition_summary.csv').open()));lookup={(r['task'],r['condition']):r for r in groups}
 for r in combined_groups:
  r.update(lookup[r['task'],r['condition']]);r['collision_count_status']='User-defined marker proximity episodes'
 write(ROOT/'condition_summary.csv',combined_groups)
 write(ROOT/'collision_events.csv',events,event_fields)
 method=json.loads((ROOT/'method.json').read_text())
 method['current_collision_policy']=json.loads((OUT/'method.json').read_text())
 method['collision_count_basis']='Current user-approved proximity definition; supersedes physical-contact-only counting for reported collision metrics.'
 (ROOT/'method.json').write_text(json.dumps(method,indent=2))
 print(json.dumps(groups,indent=2));print('TOTAL:',len(events))
if __name__=='__main__':main()
