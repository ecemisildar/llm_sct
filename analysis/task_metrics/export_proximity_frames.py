"""Annotated ORIGINAL frames for the current user-defined proximity events."""
import csv,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
OUT=ROOT/'proximity_collisions/event_review';OUT.mkdir(exist_ok=True)
H=np.linalg.inv(np.array(json.loads((BASE/'analysis/cropping/arena_calibration.json').read_text())['source_to_cropped_homography']))
events=list(csv.DictReader((ROOT/'proximity_collisions/collision_events.csv').open()))
def transform(p):
 q=cv2.perspectiveTransform(np.array([[p]],dtype=np.float64),H)[0,0]
 return tuple(round(v) for v in q)
sheets={}
for video in dict.fromkeys(e['video'] for e in events):
 group=[e for e in events if e['video']==video];task=group[0]['task']
 paths=list(csv.DictReader((BASE/'analysis/coverage'/task/Path(video).stem/'coverage_paths.csv').open()))
 cap=cv2.VideoCapture(str(BASE/'videos'/task/video));folder=OUT/Path(video).stem;folder.mkdir(exist_ok=True);tiles=[]
 for n,e in enumerate(group,1):
  t=float(e['elapsed_s']);positions={int(p['robot'].split('_')[-1]):(float(p['x'])*200,(3-float(p['y']))*200) for p in paths if abs(float(p['elapsed_s'])-t)<.01};i=int(e['robot']);p=positions[i];q=transform(p)
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,img=cap.read()
  if not ok:raise RuntimeError(e)
  cv2.circle(img,q,8,(0,255,255),2);cv2.putText(img,f'R{i}',(q[0]+12,q[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
  if e['contact_type']=='robot':
   j=int(e['other'].split('_')[-1]);end=transform(positions[j]);cv2.circle(img,end,8,(0,255,255),2);cv2.putText(img,f'R{j}',(end[0]+12,end[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
  else:
   end=transform((float(e['wall_nearest_x_m'])*200,(3-float(e['wall_nearest_y_m']))*200))
  cv2.line(img,q,end,(0,255,255),2)
  img=cv2.copyMakeBorder(img,0,75,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
  cv2.putText(img,f"COUNTED PROXIMITY: R{i} / {e['other']} at {t:.1f}s",(14,748),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
  cv2.putText(img,f"Distance {float(e['distance_m']):.3f} m <= {float(e['threshold_m']):.2f} m; {video}",(14,777),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
  name=f"event_{n:02d}_{e['contact_type']}_{t:.1f}s.png";cv2.imwrite(str(folder/name),img);tiles.append(cv2.resize(img,(640,398)))
 cap.release()
 if len(tiles)%2:tiles.append(np.full_like(tiles[0],255))
 cv2.imwrite(str(folder/'counted_events.png'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
print('Exported original-frame evidence for',len(events),'counted proximity episodes.')
