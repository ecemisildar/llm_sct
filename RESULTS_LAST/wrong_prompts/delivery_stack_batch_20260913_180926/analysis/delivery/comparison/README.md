# Delivery sensitivity analysis: delivery_stack_batch_20260913_180926

Separate analysis of the saved batch: one prompt, five generated controllers per setting, ten seeds, with and without fixed spec. All 100 expected completed cells are included. The previous RESULTS_LAST/analysis/delivery analysis is preserved.

Both settings have 50 runs, zero full successes, and 0% mean recorded task progress. This is generation and seed sensitivity for one prompt; comparisons between different prompts are not available in this batch.

This batch uses the legacy three-target delivery schema. The saved scripts adapt the existing analyzer to read completed_targets / total_targets from task_result.csv and verify completed_targets against the sum of per-robot delivered_count in task_progress.csv. These are recorded task targets; zone announcements are reported separately and do not establish unique physical box deliveries. The main six-box analyzer and original run data were not modified.

The CSVs contain run, generation, setting, seed, controller attribution, task progress, duration, coverage, collisions, and selected-event percentages. The sensitivity PNGs summarize both settings. Delivery timestep files report zone announcements and verification of shared timing; unknown timing is excluded from balance pass/fail counts.

To reproduce, run from src/llm_sct:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/delivery-stack-mpl python3 RESULTS_LAST/wrong_prompts/delivery_stack_batch_20260913_180926/analysis/delivery/comparison/scripts/analyze_prompt_sensitivity.py --repo-root . --task delivery --experiment-root RESULTS_LAST/wrong_prompts/delivery_stack_batch_20260913_180926 --output-dir RESULTS_LAST/wrong_prompts/delivery_stack_batch_20260913_180926/analysis/delivery/comparison --prompts 1 --generations 1 2 3 4 5 --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --latest-per-cell --allow-incomplete --no-baseline
```

No zone-drop announcements were recorded in these 100 runs. The balance reconstruction reports no violations, but this is vacuous because no deliveries occurred. All 100 runs have verified timing.
