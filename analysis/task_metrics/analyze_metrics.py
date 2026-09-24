"""Reproducible geometric metrics from saved 5 Hz ArUco trajectories."""
import csv
import json
import math
import statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COVERAGE = ROOT.parent / 'coverage'
ORDER = ('red', 'green', 'blue')

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    targets = {r['video']: {c: (p[0]/200, 3-p[1]/200) for c, p in r['centers_px'].items()}
               for r in json.loads((ROOT/'target_candidates.json').read_text())}
    trials = list(csv.DictReader((COVERAGE/'trial_summary.csv').open()))
    results, contacts, reaches, audit, curves = [], [], [], [], []
    for trial in trials:
        video, task, condition = (trial[k] for k in ('video', 'task', 'condition'))
        common = dict(task=task, condition=condition, video=video)
        frames = defaultdict(dict)
        for r in csv.DictReader((Path(trial['result_directory'])/'coverage_paths.csv').open()):
            frames[float(r['elapsed_s'])][int(r['robot'].split('_')[-1])] = (float(r['x']), float(r['y']))
        centers = targets.get(video, {})
        state = [0]*3
        blue_times = [None]*3
        last_contact = {}
        counts = Counter()
        cutoff_counts = Counter()
        minima = {(i,c): (float('inf'), None) for i in range(3) for c in ORDER}
        for t, positions in sorted(frames.items()):
            # Removed robots outside the floor cannot create floor contact events.
            positions = {i:p for i,p in positions.items() if 0 <= p[0] <= 4 and 0 <= p[1] <= 3}
            if task == 'patrolling':
                for i,p in positions.items():
                    if state[i] < 3:
                        c = ORDER[state[i]]
                        distance = math.dist(p, centers[c])
                        if distance < minima[i,c][0]:
                            minima[i,c] = (distance,t)
                        if distance <= 1.0:
                            state[i] += 1
                            reaches.append(dict(**common, robot=i, color=c, elapsed_s=t, distance_m=distance, points=state[i]))
                            if state[i] == 3:
                                blue_times[i] = t
                curves.append(dict(**common, elapsed_s=t, robot_0_points=state[0], robot_1_points=state[1], robot_2_points=state[2], mean_points=sum(state)/3, progression_pct=sum(state)/9*100))
            near = []
            for i,p in positions.items():
                for wall,d in [('left',p[0]),('right',4-p[0]),('bottom',p[1]),('top',3-p[1])]:
                    if d <= .15:
                        near.append(('wall', f'robot_{i}:{wall}', i, wall, d, .5))
                for c,q in centers.items():
                    d = math.dist(p,q)
                    if d <= .325:
                        near.append(('pillar', f'robot_{i}:{c}', i, c, d, .5))
                for j,q in positions.items():
                    if j > i and math.dist(p,q) <= .45:
                        near.append(('robot', f'robot_{i}:robot_{j}', i, f'robot_{j}', math.dist(p,q), .8))
            for kind,key,i,other,d,cooldown in near:
                if key not in last_contact or t-last_contact[key] > cooldown+1e-8:
                    counts[kind] += 1
                    if t <= float(trial['common_cutoff_s']):
                        cutoff_counts[kind] += 1
                    contacts.append(dict(**common, elapsed_s=t, contact_type=kind, robot=i, other=other, distance_m=d))
                last_contact[key] = t
        completion = max(blue_times) if all(t is not None for t in blue_times) else None
        row = dict(**common, duration_s=float(trial['duration_s']), robot_collision_estimates=counts['robot'], wall_collision_estimates=counts['wall'], pillar_collision_estimates=counts['pillar'], total_collision_estimates=sum(counts.values()), common_cutoff_s=float(trial['common_cutoff_s']), collisions_at_common_cutoff=sum(cutoff_counts.values()), collisions_per_minute=sum(counts.values())/float(trial['duration_s'])*60, completion_time_s=completion, robots_completed=sum(t is not None for t in blue_times) if centers else '', robot_0_points=state[0] if centers else '', robot_1_points=state[1] if centers else '', robot_2_points=state[2] if centers else '', mean_task_progression_points=sum(state)/3 if centers else '', task_progression_pct=sum(state)/9*100 if centers else '')
        row['success_under_5_min_rule'] = int(float(trial['duration_s']) < 300) if centers else ''
        row['progression_basis'] = 'all reaches credited: recording < 300 s' if centers and row['success_under_5_min_rule'] else ('observed ordered 1 m reaches' if centers else '')
        row['observed_robot_0_points'] = state[0] if centers else ''
        row['observed_robot_1_points'] = state[1] if centers else ''
        row['observed_robot_2_points'] = state[2] if centers else ''
        row['observed_mean_task_progression_points'] = sum(state)/3 if centers else ''
        row['recording_duration_completion_proxy_s'] = float(trial['duration_s']) if centers and row['success_under_5_min_rule'] else ''
        if centers and row['success_under_5_min_rule']:
            row.update(robots_completed=3, robot_0_points=3, robot_1_points=3, robot_2_points=3, mean_task_progression_points=3.0, task_progression_pct=100.0)
        results.append(row)
        if centers:
            for i in range(3):
                for c in ORDER:
                    d,t = minima[i,c]
                    audit.append(dict(**common, robot=i, color=c, closest_distance_while_expected_m=d if math.isfinite(d) else '', closest_time_s=t, reached=int(any(e['video']==video and e['robot']==i and e['color']==c for e in reaches))))
    groups = []
    for task in ('exploration','patrolling'):
        for condition in ('baseline','llm'):
            rs = [r for r in results if r['task']==task and r['condition']==condition]
            ts = [r['completion_time_s'] for r in rs if r['completion_time_s'] is not None]
            groups.append(dict(task=task, condition=condition, trials=len(rs), total_collision_estimates=sum(r['total_collision_estimates'] for r in rs), mean_collision_estimates=statistics.mean(r['total_collision_estimates'] for r in rs), mean_robot_collision_estimates=statistics.mean(r['robot_collision_estimates'] for r in rs), mean_wall_collision_estimates=statistics.mean(r['wall_collision_estimates'] for r in rs), mean_pillar_collision_estimates=statistics.mean(r['pillar_collision_estimates'] for r in rs), mean_collisions_at_common_cutoff=statistics.mean(r['collisions_at_common_cutoff'] for r in rs), verified_completed_trials=len(ts) if task=='patrolling' else '', mean_completion_time_verified_s=statistics.mean(ts) if ts else '', mean_task_progression_points=statistics.mean(r['mean_task_progression_points'] for r in rs) if task=='patrolling' else '', mean_task_progression_pct=statistics.mean(r['task_progression_pct'] for r in rs) if task=='patrolling' else ''))
    for group in groups:
        rs = [r for r in results if r['task']==group['task'] and r['condition']==group['condition']]
        successes = [r for r in rs if r['success_under_5_min_rule']==1]
        group['successful_trials_under_5_min_rule'] = len(successes) if group['task']=='patrolling' else ''
        group['success_rate_pct'] = len(successes)/len(rs)*100 if group['task']=='patrolling' else ''
        group['mean_successful_recording_duration_proxy_s'] = statistics.mean(r['duration_s'] for r in successes) if successes else ''
    # Distance episodes are candidates only. Actual collisions require contact
    # review and an active, unfinished robot; unknown full counts stay blank.
    review_path = ROOT/'collision_candidate_reviews.csv'
    reviews = list(csv.DictReader(review_path.open())) if review_path.exists() else []
    review_lookup = {(r['video'],r['elapsed_s'],r['robot'],r['other']):r for r in reviews}
    completed = {(r['video'],r['robot']):r['elapsed_s'] for r in reaches if r['color']=='blue'}
    confirmed = []
    for event in contacts:
        key = (event['video'],str(event['elapsed_s']),str(event['robot']),event['other'])
        review = review_lookup.get(key)
        done = completed.get((event['video'],event['robot']))
        # Reach rows contain integer IDs. Also exclude a finished second robot.
        other_done = completed.get((event['video'],int(event['other'].split('_')[-1]))) if event['contact_type']=='robot' else None
        active = (done is None or event['elapsed_s'] < done) and (other_done is None or event['elapsed_s'] < other_done)
        if review and review['review_status']=='confirmed_contact' and active:
            confirmed.append(event)
    for row in results:
        for field in ('robot_collision_estimates','wall_collision_estimates','pillar_collision_estimates','total_collision_estimates','collisions_at_common_cutoff','collisions_per_minute'):
            row['invalid_proximity_'+field] = row[field]
            row[field] = None
        row['confirmed_contacts_in_reviewed_candidates'] = sum(e['video']==row['video'] for e in confirmed)
        row['uncertain_contact_candidates'] = sum(r['video']==row['video'] and r['review_status']=='uncertain_contact' for r in reviews)
        row['collision_count_status'] = 'Full recording contact count unvalidated; distance-only counts withdrawn'
    for group in groups:
        for field in ('total_collision_estimates','mean_collision_estimates','mean_robot_collision_estimates','mean_wall_collision_estimates','mean_pillar_collision_estimates','mean_collisions_at_common_cutoff'):
            group['invalid_proximity_'+field] = group[field]
            group[field] = None
        group['confirmed_contacts_in_reviewed_candidates'] = sum(e['task']==group['task'] and e['condition']==group['condition'] for e in confirmed)
        group['collision_count_status'] = 'Unvalidated; old proximity totals withdrawn'
    pair_review_path = ROOT/'robot_pair_reviews.csv'
    pair_reviews = list(csv.DictReader(pair_review_path.open())) if pair_review_path.exists() else []
    for row in results:
        rs = [r for r in pair_reviews if r['video']==row['video']]
        row['reviewed_robot_pair_encounters'] = len(rs)
        row['confirmed_robot_robot_contacts_in_review'] = sum(r['review_status']=='confirmed_contact' for r in rs)
    for group in groups:
        rs = [r for r in pair_reviews if r['task']==group['task'] and r['condition']==group['condition']]
        group['reviewed_robot_pair_encounters'] = len(rs)
        group['confirmed_robot_robot_contacts_in_review'] = sum(r['review_status']=='confirmed_contact' for r in rs)
    # Always replace even an empty event CSV, to prevent stale collision events.
    with (ROOT/'collision_events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','condition','video','elapsed_s','contact_type','robot','other','distance_m'])
        w.writeheader()
        w.writerows(confirmed)
    for name,rows in [('trial_summary',results),('condition_summary',groups),('proximity_candidates',contacts),('ordered_reach_events',reaches),('closest_approach_audit',audit),('observed_progression_timeseries',curves)]:
        write_csv(ROOT/f'{name}.csv',rows)
    (ROOT/'method.json').write_text(json.dumps(dict(zone_radius_m=1.0, order=ORDER, robot_pair_threshold_m=.45, wall_threshold_m=.15, pillar_threshold_m=.325, robot_contact_gap_s=.8, obstacle_contact_gap_s=.5, pose_interval_s=.2, arena_dimensions_m=[4,3], collision_scope='Whole recording; unique pair episodes; simultaneous wall and robot contacts count separately.', missing_data='No interpolation; a tracking gap can hide events or split a contact episode.', completion='Last of three robots to reach blue after red then green, measured from video start; blank if unverified.', pillar_centers='First-frame HSV component centroids, manually reviewed in target_calibration_sheet.jpg; approximate floor positions.', robot_position='ArUco center proxy; no camera-height or marker-to-body correction.'),indent=2))
    method = json.loads((ROOT/'method.json').read_text())
    method.update(success_rule='User instruction: patrolling recording duration strictly below 300 s implies success and all nine ordered reaches credited.', progression='Successful clips receive 3 points per robot; other clips retain observed ordered 1 m scores.', completion_proxy='Successful recording duration is provided separately as a proxy, not the measured last blue arrival. No timestamps are invented for credited reaches.')
    method.update(collision_policy='Actual physical contact only, confirmed by review. Completed robots, parking, human handling, setup and proximity alone are excluded. Distance thresholds generate candidates, never collision counts.', collision_validation=f"{sum(r['review_status']=='rejected' for r in reviews)} original candidates rejected; {sum(r['review_status']=='uncertain_contact' for r in reviews)} uncertain. Full-video collision totals are unvalidated and blank, not zero.", completed_robot_policy='Exclude either robot at or after its observed ordered blue reach; review must additionally exclude completion/parking missed by geometric tracking.')
    method['robot_robot_review'] = dict(encounters=len(pair_reviews), candidate_radius_m=.85, confirmed_contacts=sum(r['review_status']=='confirmed_contact' for r in pair_reviews), scope='Closest original frame per encounter; +/-0.6 s temporal frames for the 19 closest encounters. Not a full-frame audit.')
    (ROOT/'method.json').write_text(json.dumps(method,indent=2))
    print(json.dumps(groups,indent=2))

if __name__ == '__main__':
    main()
