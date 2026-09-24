# Real video metrics: current marker-proximity definition

# Current user-approved proximity collision metric

Thresholds: robot markers <= 0.60 m apart; robot marker <= 0.10 m from an inner white-wall edge. Pillars are excluded. Walls are measured against original-frame inner wall/floor edges in ../white_wall_calibration.json, rather than the inset crop border. Scale is approximately 4 x 3 m.

Robot pair episodes require a gap greater than 0.8 s before another count; wall episodes use 0.5 s. Static setup is excluded until 0.05 m marker displacement. Documented completed robots, completion parking, and human handling/removal are excluded; timestamps and sources are in robot_exclusions.csv. One additional original-frame review excludes robot 0 in 13-11-04 from its final blue approach/parking at 78.6 s.

Total: 30 proximity episodes (20 robot–robot, 10 marker–wall).

- exploration baseline: 13 episodes.
- exploration llm: 9 episodes.
- patrolling baseline: 2 episodes.
- patrolling llm: 6 episodes.

These are the user's marker-proximity collision metric, not confirmed touching. Missing markers and unobserved completion timestamps can affect counts. A video shorter than 300 s establishes success but cannot establish each robot's exact completion timestamp. Original recordings are preserved.

Event CSVs, per-video counts, condition means, exclusion cutoffs, and method JSON are here. Annotated ORIGINAL evidence frames are in event_review. The invalid_crop_border_event_preview folder is an archived preliminary preview using the crop border and must not be counted.

Reproduce using analyze_proximity_collisions.py, export_proximity_frames.py, then plot_combined_results.py and plot_metrics.py in the parent directory (use python3 -s for plotting). If analyze_metrics.py is run to rebuild progression, run analyze_proximity_collisions.py afterward to restore current collision metrics before plotting.

## Preserved patrolling results

Baseline and LLM each have 8/10 successes using the under-300-second duration rule. Successful clips receive all nine reaches; longer clips retain observed partial scores. Mean progression: baseline 94.44%, LLM 90.00%. Successful recording-duration completion proxies: baseline 152.04 s, LLM 215.46 s. Exact last blue arrival times are separate where detected.

## Historical physical-contact review

All 81 original wall/pillar candidates were rejected. No robot–robot physical contact was confirmed in 257 broad encounter frames, with before/after review for the 19 closest. These physical-contact observations are separate from the newly authorized marker-proximity metric. Archived invalid totals remain in invalid_proximity_v1 for traceability.
