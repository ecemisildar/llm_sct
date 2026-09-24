import csv,json
from pathlib import Path
import cv2,numpy as np
root=Path('/home/ecem/sct_ws/src/leo_real_experiments')
a=root/'analysis/task_metrics'
out=a/'collision_review';out.mkdir(exist_ok=True)
H=np.linalg.inv(np.array(json.loads((root/'analysis/cropping/arena_calibration.json').read_text())['source_to_cropped_homography']))
events=list(csv.DictReader((a/'collision_events.csv').open()))
rows=[]; current=None;cap=None;paths=None
for index,e in enumerate(events):
 if current!=e['video']:
  if cap:cap.release()
  current=e['video'];cap=cv2.VideoCapture(str(root/'videos'/e['task']/current))
  paths=list(csv.DictReader((root/'analysis/coverage'/e['task']/Path(current).stem/'coverage_paths.csv').open()))
 t=float(e['elapsed_s']);p=next(p for p in paths if p['robot']==f"robot_{e['robot']}" and abs(float(p['elapsed_s'])-t)<.01)
 q=cv2.perspectiveTransform(np.array([[[float(p['x'])*200,(3-float(p['y']))*200]]],dtype=np.float64),H)[0,0]
 tiles=[]
 for dt in (-.6,0,.6):
  cap.set(cv2.CAP_PROP_POS_MSEC,max(0,min(t+dt, max(float(p['elapsed_s']) for p in paths)))*1000);ok,img=cap.read()
  if not ok:raise RuntimeError(e)
  x,y=map(round,q);x=max(0,min(1280-300,x-150));y=max(0,min(720-300,y-150))
  patch=img[y:y+300,x:x+300].copy()
  cv2.circle(patch,(round(q[0])-x,round(q[1])-y),5,(0,255,255),1)
  patch=cv2.resize(patch,(240,240));patch=cv2.copyMakeBorder(patch,0,22,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
  cv2.putText(patch,f"{t+dt:.1f}s",(5,255),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);tiles.append(patch)
 row=np.hstack(tiles);row=cv2.copyMakeBorder(row,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
 label=f"#{index+1:02d} {e['video']} R{e['robot']} {e['other']} d={float(e['distance_m']):.3f}"
 cv2.putText(row,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);rows.append(row)
 if len(rows)==9 or index==len(events)-1:
  page=(index//9)+1;cv2.imwrite(str(out/f'candidate_review_{page:02d}.jpg'),np.vstack(rows));rows=[]
cap.release()
print('Saved 9 pages reviewing 81 candidates at t-0.6, t, t+0.6 from ORIGINAL videos.')
