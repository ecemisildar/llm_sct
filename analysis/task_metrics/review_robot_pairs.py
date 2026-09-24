"""Export broad robot-pair candidates for physical-contact review; never count distance alone."""
import csv,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'robot_pair_review';OUT.mkdir(exist_ok=True)
BASE=ROOT.parents[1]
H=np.linalg.inv(np.array(json.loads((BASE/'analysis/cropping/arena_calibration.json').read_text())['source_to_cropped_homography']))
rows=list(csv.DictReader((ROOT/'robot_pair_candidates.csv').open()))
for n,r in enumerate(rows,1):r['candidate_id']=n
for video in dict.fromkeys(r['video'] for r in rows):
 group=[r for r in rows if r['video']==video];task=group[0]['task']
 cap=cv2.VideoCapture(str(BASE/'videos'/task/video))
 paths=list(csv.DictReader((BASE/'analysis/coverage'/task/Path(video).stem/'coverage_paths.csv').open()))
 for r in group:
  t=float(r['closest_time_s']);ps=[p for p in paths if abs(float(p['elapsed_s'])-t)<.01 and int(p['robot'].split('_')[-1]) in (int(r['robot_i']),int(r['robot_j']))]
  qs=cv2.perspectiveTransform(np.array([[[float(p['x'])*200,(3-float(p['y']))*200] for p in ps]],dtype=np.float64),H)[0]
  center=np.mean(qs,axis=0);x=max(0,min(1280-360,round(center[0])-180));y=max(0,min(720-360,round(center[1])-180))
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,img=cap.read()
  if not ok:raise RuntimeError(r)
  patch=img[y:y+360,x:x+360].copy()
  for p,q in zip(ps,qs):
   point=(round(q[0])-x,round(q[1])-y);cv2.circle(patch,point,4,(0,255,255),1)
   cv2.putText(patch,p['robot'],(max(0,point[0]-30),max(15,point[1]-35)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
  patch=cv2.copyMakeBorder(patch,42,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
  cv2.putText(patch,f"#{r['candidate_id']:03d} {video}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1)
  cv2.putText(patch,f"R{r['robot_i']}/R{r['robot_j']} t={t:.1f}s d={float(r['min_marker_distance_m']):.3f}m",(4,33),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
  cv2.imwrite(str(OUT/f"candidate_{r['candidate_id']:03d}.png"),patch)
 cap.release()
for start in range(0,len(rows),12):
 tiles=[cv2.imread(str(OUT/f'candidate_{i+1:03d}.png')) for i in range(start,min(start+12,len(rows)))]
 while len(tiles)%3:tiles.append(np.full_like(tiles[0],255))
 cv2.imwrite(str(OUT/f'page_{start//12+1:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
print('Exported',len(rows),'robot-pair candidate frames from ORIGINAL videos.')
