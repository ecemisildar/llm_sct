# Real robot video coverage

Coverage uses the simulation geometry directly: 1 m square cells, circular radius 0.36 m, 0.2 s pose samples, and 0.5 s metric samples. Team coverage is the number of distinct touched cells divided by 12 floor cells, multiplied by 100. A touched cell counts in full; this is a grid visitation metric, not the exact fraction of floor swept by the robot.

The 800 × 600 cropped arena represents approximately 4 × 3 m (12 m²). ArUco DICT_4X4_50 IDs 0, 1, 2 identify the robots; pillar markers are ignored. Missing markers are recovered from timestamp-aligned original frames and mapped through the saved crop homography into the same arena coordinates. Pixel marker centres are converted to metres with origin at the bottom left. The radius 0.36 m is retained from simulation for consistency and is not a measured physical footprint.

Video time zero is the first frame. No task start/end event is inferred. The full recording is analyzed. Missing detections contribute no additional coverage and are not interpolated. Tops of robot markers approximate robot centres; no marker-height parallax correction is available. All 12 floor cells form the denominator; no surveyed obstacle mask is subtracted.

Final values can reflect unequal recording durations. Each task also has a comparison at the shortest recording duration shared by all its trials.

| Task | Condition | Trials | Common cutoff (s) | Coverage at cutoff, mean ± SD (%) | Final mean (%) |
|---|---|---:|---:|---:|---:|
| exploration | baseline | 10 | 64.57 | 100.00 ± 0.00 | 100.00 |
| exploration | llm | 10 | 64.57 | 97.50 ± 5.62 | 97.50 |
| patrolling | baseline | 10 | 122.20 | 97.50 ± 5.62 | 98.33 |
| patrolling | llm | 10 | 122.20 | 100.00 ± 0.00 | 100.00 |

## Tracking quality

Detection quality below is calculated between each robot’s first and last in-arena observation. Robots are sometimes physically removed during patrolling; their absence after removal is excluded from this quality denominator. First/last spans and whole-recording in-arena detection rates are retained in the per-trial JSON and summary CSV.

| Task | Video | Minimum active-span detection (%) | Longest internal interval (s) |
|---|---|---:|---:|
| exploration | 2026-09-13 17-58-52_exp_base.mp4 | 97.8 | 0.60 |
| exploration | 2026-09-13 18-01-11_exp_llm.mp4 | 96.7 | 0.40 |
| exploration | 2026-09-13 18-03-08_exp_base.mp4 | 98.8 | 0.40 |
| exploration | 2026-09-13 18-06-11_exp_llm.mp4 | 98.2 | 0.60 |
| exploration | 2026-09-13 18-08-41_exp_base.mp4 | 96.4 | 0.80 |
| exploration | 2026-09-13 18-11-12_exp_llm.mp4 | 96.1 | 0.80 |
| exploration | 2026-09-13 18-14-32_exp_base.mp4 | 95.6 | 1.00 |
| exploration | 2026-09-13 18-17-02_exp_llm.mp4 | 94.8 | 1.40 |
| exploration | 2026-09-13 18-21-51_exp_base.mp4 | 90.8 | 0.80 |
| exploration | 2026-09-13 18-24-07_exp_llm.mp4 | 93.6 | 1.00 |
| exploration | 2026-09-13 18-26-08_exp_base.mp4 | 96.9 | 0.80 |
| exploration | 2026-09-13 18-28-52_exp_llm.mp4 | 97.9 | 0.60 |
| exploration | 2026-09-13 18-31-50_exp_base.mp4 | 98.5 | 0.60 |
| exploration | 2026-09-13 18-34-45_exp_llm.mp4 | 98.2 | 0.40 |
| exploration | 2026-09-13 18-37-15_exp_base.mp4 | 98.2 | 0.40 |
| exploration | 2026-09-13 18-39-43_exp_llm.mp4 | 91.4 | 3.20 |
| exploration | 2026-09-13 18-42-20_exp_base.mp4 | 99.1 | 0.40 |
| exploration | 2026-09-13 18-44-36_exp_lm.mp4 | 96.3 | 0.60 |
| exploration | 2026-09-13 18-48-05_exp_base.mp4 | 97.4 | 1.40 |
| exploration | 2026-09-13 18-50-42_exp_llm.mp4 | 95.8 | 0.80 |
| patrolling | 2026-09-13 12-25-28_base.mp4 | 99.6 | 0.40 |
| patrolling | 2026-09-13 12-36-00_llm.mp4 | 74.3 | 5.20 |
| patrolling | 2026-09-13 12-41-54_base.mp4 | 93.8 | 1.40 |
| patrolling | 2026-09-13 12-45-21_llm_best.mp4 | 97.5 | 1.60 |
| patrolling | 2026-09-13 12-53-56_base.mp4 | 99.0 | 1.40 |
| patrolling | 2026-09-13 12-59-50_llm.mp4 | 96.8 | 2.40 |
| patrolling | 2026-09-13 13-06-27_base.mp4 | 96.0 | 2.80 |
| patrolling | 2026-09-13 13-11-04_llm.mp4 | 96.0 | 2.60 |
| patrolling | 2026-09-13 13-18-40_base.mp4 | 95.8 | 1.00 |
| patrolling | 2026-09-13 13-22-01_llm.mp4 | 92.1 | 3.00 |
| patrolling | 2026-09-13 16-31-31_base.mp4 | 98.8 | 1.00 |
| patrolling | 2026-09-13 16-35-23_llm.mp4 | 99.3 | 0.40 |
| patrolling | 2026-09-13 16-41-51_base.mp4 | 97.4 | 1.40 |
| patrolling | 2026-09-13 16-49-26_llm.mp4 | 95.7 | 2.00 |
| patrolling | 2026-09-13 16-55-09_base.mp4 | 92.7 | 1.60 |
| patrolling | 2026-09-13 17-02-34_llm.mp4 | 98.1 | 1.20 |
| patrolling | 2026-09-13 17-10-48_baseline.mp4 | 94.6 | 1.40 |
| patrolling | 2026-09-13 17-17-42_llm.mp4 | 82.1 | 3.80 |
| patrolling | 2026-09-13 17-23-51_base.mp4 | 95.3 | 1.20 |
| patrolling | 2026-09-13 17-30-52_llm.mp4 | 89.6 | 3.00 |

Missing detections can undercount visited cells; inspect `tracking_observations.csv` and `detection_preview.jpg` in each trial folder. Coverage resolution is 8.33 percentage points per cell, so saturation at 100% can hide differences between trajectories.

Files: `trial_summary.csv`, `condition_summary.csv`, `baseline_vs_llm_coverage.png`/`.svg`, and per-trial CSVs and `coverage_map_and_curve.png`/`.svg`.

The filename `exp_lm` is grouped as LLM; `llm_best` is also LLM. Original and cropped videos are read only.

Reproduce: `python3 analysis/coverage/analyze_coverage.py`, then `python3 -s analysis/coverage/plot_coverage.py` from the experiment directory.
