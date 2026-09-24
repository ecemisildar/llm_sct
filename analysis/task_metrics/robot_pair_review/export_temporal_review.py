import csv,json
from pathlib import Path
import cv2,numpy as np
base=Path('/home/ecem/sct_ws/src/leo_real_experiments');root=base/'analysis/task_metrics';out=root/'robot_pair_review'
H=np.linalg.inv(np.array(json.loads((base/'analysis/cropping/arena_calibration.json').read_text())['source_to_cropped_homography']))
rows=list(csv.DictReader((root/'robot_pair_candidates.csv').open()))[:19]
strips=[]
for n,r in enumerate(rows,1):
 paths=list(csv.DictReader((base/'analysis/coverage'/r['task']/Path(r['video']).stem/'coverage_paths.csv').open()));t=float(r['closest_time_s']);ps=[p for p in paths if abs(float(p['elapsed_s'])-t)<.01 and int(p['robot'].split('_')[-1]) in (int(r['robot_i']),int(r['robot_j']))];qs=cv2.perspectiveTransform(np.array([[[float(p['x'])*200,(3-float(p['y']))*200] for p in ps]],dtype=np.float64),H)[0];q=np.mean(qs,axis=0);x=max(0,min(1280-360,round(q[0])-180));y=max(0,min(720-360,round(q[1])-180));cap=cv2.VideoCapture(str(base/'videos'/r['task']/r['video']));tiles=[]
 for dt in [-.6,0,.6]:
  tt=max(0,min(t+dt,max(float(p['elapsed_s']) for p in paths)));cap.set(cv2.CAP_PROP_POS_MSEC,tt*1000);ok,img=cap.read()
  if not ok:raise RuntimeError(r)
  patch=img[y:y+360,x:x+360].copy();cv2.putText(patch,f'{tt:.1f}s',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);tiles.append(patch)
 cap.release();strip=np.hstack(tiles);strip=cv2.copyMakeBorder(strip,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));cv2.putText(strip,f"#{n:03d} {r['video']} R{r['robot_i']}/R{r['robot_j']} d={float(r['min_marker_distance_m']):.3f}",(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);strips.append(strip);cv2.imwrite(str(out/f'temporal_{n:03d}.png'),strip)
for start in range(0,len(strips),3):cv2.imwrite(str(out/f'temporal_page_{start//3+1:02d}.jpg'),np.vstack(strips[start:start+3]))
print('Temporal review exported for the 19 closest passes.')
