# Revised video metrics

Per the user correction, patrolling recordings shorter than 300 seconds count as successful and receive all nine ordered reaches: red, green, blue for each of three robots. Longer recordings retain observed ordered 1 m circle scores as partial progression. Originals and cropped videos are preserved.

| Condition | Successes | Mean points/robot, all trials | Mean progression | Mean successful recording duration |
|---|---:|---:|---:|---:|
| Baseline | 8/10 (80%) | 2.833 / 3 | 94.44% | 152.04 s |
| LLM | 8/10 (80%) | 2.700 / 3 | 90.00% | 215.46 s |

Each successful trial receives 100% progression. Group means include partial scores for the four longer recordings: baseline 16-41-51 (323.90 s), baseline 17-10-48 (313.07 s), LLM 13-11-04 (325.53 s), and LLM 17-02-34 (311.07 s).

Recording duration is provided separately as a completion-time proxy; it is not an exact measurement of the last blue arrival. The duration rule establishes success without supplying arrival timestamps. The completion_time_s column retains measured last ordered blue entry where all three entries were detected. One such observation exists at 211.6 s in 2026-09-13 12-36-00_llm.mp4; this does not override the revised success classification. No timestamps are invented for credited reaches.

The observed_robot_*_points columns retain geometric scores. ordered_reach_events.csv and observed_progression_timeseries.csv contain observations only. The legacy progression_timeseries.csv also contains only the earlier geometric observations. Review images and closest_approach_audit.csv remain diagnostics.

| Task | Condition | Total estimated contacts | Mean/video |
|---|---|---:|---:|
| Exploration | Baseline | 19 | 1.9 |
| Exploration | LLM | 15 | 1.5 |
| Patrolling | Baseline | 17 | 1.7 |
| Patrolling | LLM | 30 | 3.0 |

Collision estimates are unchanged. These are video proximity episodes, not confirmed physical contacts. Robot separation threshold is 0.45 m; wall distance threshold is 0.15 m, matching the existing real tracker. Pillar distance threshold is 0.325 m, assuming robot radius 0.225 m plus pillar radius 0.10 m. Simulation episode debounce uses gaps greater than 0.8 s for robot pairs and 0.5 s for obstacles. Sustained proximity counts once; different simultaneous contacts count separately. No robot-pair episodes were detected. Patrolling has three pillar episodes per condition; all other events are wall proximity episodes.

Whole-recording counts include setup, retreat, parking, and removal while the marker remains in the floor arena. At the common coverage cutoff, mean contacts are exploration baseline 1.9 and LLM 1.5 at 64.567 s; patrolling baseline 1.1 and LLM 2.3 at 122.2 s. Per-minute rates are included in the trial table.

Inputs are saved 5 Hz ArUco-center trajectories in an approximately 4 by 3 m arena and visually reviewed first-frame HSV pillar centroids. Approximate scale, elevated markers, marker/body offsets, and missing detections introduce uncertainty. Positions are not interpolated; tracking gaps can hide events or split contact episodes. Outside-floor marker centers are excluded.

trial_summary.csv and condition_summary.csv contain revised scores and success rates. comparison.png and .svg use revised progression. method.json records both scoring rules. Reproduce using python3 analyze_metrics.py, python3 -s plot_metrics.py, and python3 make_review.py in this directory. The -s option selects compatible system plotting dependencies.
