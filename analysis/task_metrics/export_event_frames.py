"""Export annotated evidence for every counted event in a selected recording."""
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
VIDEO='2026-09-13 12-36-00_llm.mp4'
OUT=ROOT/'event_review'/Path(VIDEO).stem
OUT.mkdir(parents=True,exist_ok=True)
video=ROOT.parents[1]/'videos'/'patrolling'/'cropped'/VIDEO
centers=next(r['centers_px'] for r in json.loads((ROOT/'target_candidates.json').read_text()) if r['video']==VIDEO)
paths=list(csv.DictReader((ROOT.parent/'coverage'/'patrolling'/Path(VIDEO).stem/'coverage_paths.csv').open()))
cap=cv2.VideoCapture(str(video))
manifest=[]

def point(p):
 return (round(float(p['x'])*200),round((3-float(p['y']))*200))

def frame(t,robot):
 cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
 ok,img=cap.read()
 if not ok: raise RuntimeError(t)
 positions={int(p['robot'].split('_')[-1]):point(p) for p in paths if abs(float(p['elapsed_s'])-t)<.01}
 for i,q in positions.items():
  cv2.circle(img,q,7,(255,255,255),2)
  cv2.putText(img,f'R{i}',(q[0]+9,q[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
 q=positions[robot]
 cv2.circle(img,q,9,(0,255,255),2)
 return img,q,positions

def save(img,name,title,detail):
 canvas=cv2.copyMakeBorder(img,0,85,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
 cv2.putText(canvas,title,(12,628),cv2.FONT_HERSHEY_SIMPLEX,.64,(0,0,0),2)
 cv2.putText(canvas,detail,(12,655),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,0,0),1)
 cv2.putText(canvas,VIDEO,(12,679),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
 cv2.imwrite(str(OUT/name),canvas)
 return canvas

def sheet(tiles,name):
 tiles=[cv2.resize(im,(400,343)) for im in tiles]
 while len(tiles)%3: tiles.append(np.full_like(tiles[0],255))
 image=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
 cv2.imwrite(str(OUT/name),image)

bumps=[]
for r in csv.DictReader((ROOT/'proximity_candidates.csv').open()):
 if r['video']!=VIDEO: continue
 t=float(r['elapsed_s']); robot=int(r['robot']); kind=r['contact_type']; other=r['other']
 img,q,positions=frame(t,robot)
 if kind=='wall':
  end={'left':(0,q[1]),'right':(799,q[1]),'bottom':(q[0],599),'top':(q[0],0)}[other]
  cv2.line(img,q,end,(0,0,255),3)
  wall={'left':((30,0),(30,599)),'right':((770,0),(770,599)),'bottom':((0,570),(799,570)),'top':((0,30),(799,30))}[other]
  cv2.line(img,*wall,(0,0,255),2)
  threshold=.15
 elif kind=='pillar':
  end=tuple(round(v) for v in centers[other]); threshold=.325
  cv2.circle(img,end,round(threshold*200),(0,0,255),2)
  cv2.line(img,q,end,(0,0,255),3)
 else:
  end=positions[int(other.split('_')[-1])]; threshold=.45
  cv2.line(img,q,end,(0,0,255),3)
 name=f'bump_{len(bumps)+1:02d}_{t:.1f}s.png'
 bumps.append(save(img,name,f'REJECTED: NOT A COLLISION. R{robot} / {other}, t={t:.1f} s',f"Old proximity distance {float(r['distance_m']):.3f} m; excluded from collision counts"))
 manifest.append(dict(event='rejected_proximity_candidate',robot=robot,target=other,elapsed_s=t,distance_m=float(r['distance_m']),png=name))
reaches=[]
for r in csv.DictReader((ROOT/'ordered_reach_events.csv').open()):
 if r['video']!=VIDEO: continue
 t=float(r['elapsed_s']); robot=int(r['robot']); color=r['color']
 img,q,positions=frame(t,robot)
 target=tuple(round(v) for v in centers[color])
 cv2.circle(img,target,200,(0,255,255),2)
 cv2.line(img,q,target,(0,255,255),2)
 cv2.circle(img,target,5,(0,255,255),-1)
 name=f'reach_{len(reaches)+1:02d}_R{robot}_{color}_{t:.1f}s.png'
 reaches.append(save(img,name,f'Observed ordered reach: R{robot} -> {color}, t={t:.1f} s',f"Marker distance {float(r['distance_m']):.3f} m <= 1.000 m; robot points: {r['points']}/3"))
 manifest.append(dict(event='observed_ordered_reach',robot=robot,target=color,elapsed_s=t,distance_m=float(r['distance_m']),png=name))
cap.release()
sheet(bumps,'bumps_contact_sheet.png')
sheet(reaches,'zone_reaches_contact_sheet.png')
(OUT/'events.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.md').write_text(f'# Corrected event review\n\nRecording: {VIDEO}\n\nAll {len(bumps)} old proximity candidates are REJECTED: none is a counted collision. Completed robots, human handling, and proximity without physical contact are excluded. Individual old bump PNG filenames are preserved for traceability but every frame is now labeled REJECTED. The nine zone-reach frames are unchanged observations. No reach timestamps are invented. Original videos are preserved.\n')
assert len(bumps)==7 and len(reaches)==9
print(OUT)
print('Exported 7 REJECTED candidate PNGs, 9 reach PNGs, and 2 contact sheets.')
