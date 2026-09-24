# Delivery prompt sensitivity — RESULTS_LAST

Snapshot rebuilt on 2026-09-13 from completed (`Saving OK`) runs: 6 prompts × 5 generations × 10 seeds × 2 settings. Includes 486 of 600 cells, using the latest saved YAML per cell and latest completed run per controller and seed. Older three-box baselines are excluded.

Controller attribution uses the exact `metadata_yaml` path in each run's SAVE_STATUS.txt. This repairs the previous timestamp-stem collisions; distinct prompt/generation controllers are no longer overwritten.

| Setting | Completed runs | Full successes | Mean delivered boxes | Verified balance violations |
|---|---:|---:|---:|---:|
| with_fixed_spec | 246 | 81 | 4.183 | 9 |
| without_fixed_spec | 240 | 0 | 1.508 | 1 |

## Delivery steps and zone balance

- `delivery_zone_timesteps.pdf`: every included run, grouped by prompt, setting, generation, and seed. The upper plot shows cumulative red/blue deliveries; the lower plot shows red minus blue with the permitted −1 to +1 band.
- `delivery_zone_timesteps.csv`: initial zero counts and one timestep per selected zone-drop event, with robot, zone, shared timestamp, cumulative team counts, difference, and a ±1 flag.
- `delivery_balance_per_run.csv`: final zone counts, maximum absolute difference, first violation time, and timing verification status for each run.
- `delivery_zone_robot_timesteps.csv`: all recorded zone deliveries with each robot's own elapsed clock and cumulative per-robot counts, including runs whose team ordering cannot be verified.

ROS selected-event log sequences are matched against the saved robot CSVs to recover shared Unix timestamps. A missing terminal log tail is accepted only when it contains no zone deliveries. Robot CSV elapsed clocks have different origins and are not sorted together. Plot/CSV shared elapsed time is measured from the earliest recovered robot clock origin. Counts change at delivery events and remain constant between them.

485 runs have verified delivery order: 475 remain within ±1 throughout and 10 violate the bound (9 with fixed spec, 1 without). Maximum observed absolute difference is 3. One run has unknown team ordering and is excluded from the pass/fail count:
- `/home/ecem/sct_ws/src/llm_sct/RESULTS_LAST/delivery/with_fixed_spec/seed_1002/S_20260912_233521/run_20260913_145311_613978930` — missing matching log for robot_2.

Delivered box totals use the maximum team progress repeated across robots in task_progress.csv. Zone counts represent selected zone-drop announcements; 11 runs have a discrepancy between team progress and zone-event totals, flagged in the CSVs. These data establish the balance of recorded announcements, not independent physical box removal or unique box identity.

## Reproduce

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/delivery-mpl python3 evaluation/evaluation/analyze_prompt_sensitivity.py \
  --task delivery --experiment-root RESULTS_LAST \
  --output-dir RESULTS_LAST/analysis/delivery/comparison \
  --prompts 1 2 3 4 5 6 --generations 1 2 3 4 5 \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 \
  --latest-per-cell --allow-incomplete --no-baseline
```

Run from `src/llm_sct`. ROS timing reconstruction reads `new_results/llm_run_logs`.

## Updated figure with replacement P1

The PNGs use 100 latest P1 runs from RESULTS_LAST/delivery, preserved P2–P5 data (386 runs), and 90 P6 runs (50 with fixed spec, 40 without), for 576 runs total. P6 generation 4 without fixed spec has no attributable saved runs for seeds 1001–1010. Exact resolved metadata_yaml paths distinguish generations sharing a filename stem. delivery_prompt_sensitivity_figure_runs.csv contains the plotted data; earlier summary CSVs are the historical snapshot.

Panels: delivered boxes, task progression, duration, red boxes as a percentage of all zone deliveries, and collisions. The blue count and coverage panels are omitted. The percentage is undefined for zero deliveries, which are excluded from that panel. A dashed line marks 50%; red crosses identify runs whose final red-blue count difference exceeds 1. This final-count figure does not establish balance at every timestep. Counts by zone come from recorded zone-drop announcements.

Reproduce from src/llm_sct: PYTHONNOUSERSITE=1 python3 RESULTS_LAST/replot_delivery_prompt1_replaced.py
