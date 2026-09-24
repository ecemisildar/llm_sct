import csv
import json
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parent
targets = {r['video']:r['centers_px'] for r in json.loads((root/'target_candidates.json').read_text())}
audit = list(csv.DictReader((root/'closest_approach_audit.csv').open()))
cases = sorted([r for r in audit if r['color']=='blue' and r['reached']=='0' and r['closest_time_s']],key=lambda r:float(r['closest_distance_while_expected_m']))
cases = [cases[i] for i in (0,7,15,23,31,38)]
tiles = []
for r in cases:
    video = root.parents[1]/'videos'/'patrolling'/'cropped'/r['video']
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC,float(r['closest_time_s'])*1000)
    ok,frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(str(video))
    p = targets[r['video']]['blue']
    cv2.circle(frame,tuple(round(v) for v in p),200,(255,255,0),2)
    paths = root.parent/'coverage'/'patrolling'/video.stem/'coverage_paths.csv'
    for row in csv.DictReader(paths.open()):
        if abs(float(row['elapsed_s'])-float(r['closest_time_s']))<.01:
            q = (round(float(row['x'])*200),round((3-float(row['y']))*200))
            cv2.circle(frame,q,6,(0,0,255),-1)
            cv2.putText(frame,row['robot'],q,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),2)
    tile = cv2.resize(frame,(480,360))
    tile = cv2.copyMakeBorder(tile,0,50,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    for y,text in [(379,r['video']), (401,f"robot {r['robot']}, t={r['closest_time_s']}s, distance={float(r['closest_distance_while_expected_m']):.3f}m")]:
        cv2.putText(tile,text,(5,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
    tiles.append(tile)
cv2.imwrite(str(root/'blue_closest_approach_review.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
