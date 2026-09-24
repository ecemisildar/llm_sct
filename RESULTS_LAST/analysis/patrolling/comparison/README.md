# Patrolling comparison with current results

Completed runs from `RESULTS_LAST/patrolling`, using the latest YAML per prompt/generation, prompts 1–6, generations 1–5, and seeds 1001–1010. Baselines come from `new_results/baseline/results_patrolling` for the same seeds.

The set without fixed specifications is incomplete. Missing runs are excluded; an empty plot position means no completed data, not zero performance. These plots are a snapshot while simulations continue.

Dashed lines show baseline means. Task completion gives partial credit for the nine robot/color targets. Duration includes timeouts; successful completion time is reported separately in the summary CSVs. Collisions use a one-second debounce per entity pair.

| Setting | Completed runs | Full successes | Mean partial completion | Mean duration including timeouts | Collisions/run |
|---|---:|---:|---:|---:|---:|
| without_fixed_spec | 216 | 0/216 (0.0%) | 14.20% | 300.380 s | 0.523 |
| with_fixed_spec | 300 | 183/300 (61.0%) | 91.41% | 239.905 s | 0.360 |
| Baseline | 10 | 10/10 (100.0%) | 100.00% | 167.526 s | 0.100 |

Prompt 1 remains the best fixed-spec prompt by full success rate (80%). The available runs without fixed specs have no full completions. Unequal and incomplete sample counts limit comparisons for later prompts.

See `patrolling_prompt_summary.csv` for per-prompt counts and `patrolling_generation_summary.csv` for per-controller results.
