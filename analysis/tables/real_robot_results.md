# Real-robot exploration

| Controller | Trials | Final coverage (%) | Collisions/trial | Total collisions |
| --- | --- | --- | --- | --- |
| Baseline | 10 | 100.00 ± 0.00 | 1.30 ± 1.16 | 13 |
| LLM | 10 | 97.50 ± 5.62 | 0.90 ± 0.99 | 9 |

# Real-robot patrolling

| Controller | Trials | Successful trials | Task progression (%) | Completion proxy (s) | Collisions/trial | Total collisions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 10 | 8/10 (80%) | 94.44 ± 12.00 | 152.04 ± 31.14 | 0.20 ± 0.42 | 2 |
| LLM | 10 | 8/10 (80%) | 90.00 ± 22.50 | 215.46 ± 62.20 | 0.60 ± 0.70 | 6 |

Values are mean ± sample standard deviation across trials. Final coverage is measured over each complete recording, matching the figure.

Patrolling success uses the existing recording-duration rule (<300 seconds); successful recordings receive all nine reaches. Completion proxy is recording duration for the eight successful trials per controller, not an exact last-arrival timestamp.

Collision counts use the current user-approved marker-proximity metric: robot markers ≤0.60 m apart or a marker ≤0.10 m from an inner wall edge. Pillars, completed-robot parking, and handling are excluded as documented in task_metrics/proximity_collisions/README.md.

Sources: task_metrics/trial_summary.csv and coverage/trial_summary.csv; identical to task_metrics/plot_combined_results.py. Reproduce: python3 analysis/tables/export_real_results.py from the leo_real_experiments package directory.
