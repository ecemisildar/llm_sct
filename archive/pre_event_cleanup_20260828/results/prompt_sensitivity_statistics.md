# Prompt Sensitivity Statistics

All standard deviations below are population standard deviations. Each baseline
contains 5 runs, and each prompt contains 15 runs (3 generations x 5 seeds).

## Collision counts

| Task | Baseline (5 runs) | All LLM runs (75 runs) |
|---|---:|---:|
| Exploration | 0.400 +/- 0.490 | 0.520 +/- 0.957 |
| Patrolling | 0.800 +/- 1.166 | 0.053 +/- 0.278 |
| Delivery | 0.600 +/- 0.800 | 0.040 +/- 0.255 |

| Task | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|
| Exploration | 0.133 +/- 0.340 | 0.467 +/- 0.884 | 0.533 +/- 0.884 | 0.800 +/- 1.275 | 0.667 +/- 1.011 |
| Patrolling | 0.067 +/- 0.249 | 0.200 +/- 0.542 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Delivery | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.067 +/- 0.249 | 0.133 +/- 0.499 | 0.000 +/- 0.000 |

## Task success

Success is encoded as 1 and failure as 0. Therefore, the mean is the success
rate. Exploration is omitted because it is evaluated over a fixed time horizon
rather than with a binary completion outcome.

| Task | Baseline (5 runs) | All LLM runs (75 runs) |
|---|---:|---:|
| Patrolling | 0.800 +/- 0.400 (4/5) | 0.840 +/- 0.367 (63/75) |
| Delivery | 0.800 +/- 0.400 (4/5) | 1.000 +/- 0.000 (75/75) |

| Task | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|
| Patrolling | 0.933 +/- 0.249 (14/15) | 0.467 +/- 0.499 (7/15) | 0.933 +/- 0.249 (14/15) | 0.933 +/- 0.249 (14/15) | 0.933 +/- 0.249 (14/15) |
| Delivery | 1.000 +/- 0.000 (15/15) | 1.000 +/- 0.000 (15/15) | 1.000 +/- 0.000 (15/15) | 1.000 +/- 0.000 (15/15) | 1.000 +/- 0.000 (15/15) |

## Coverage

Coverage statistics include all runs and are expressed as percentages.

| Task | Baseline (5 runs) | All LLM runs (75 runs) |
|---|---:|---:|
| Exploration | 81.25 +/- 7.12% | 63.95 +/- 7.14% |
| Patrolling | 43.53 +/- 4.76% | 45.19 +/- 10.45% |
| Delivery | 36.20 +/- 5.71% | 33.48 +/- 5.65% |

| Task | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|
| Exploration | 63.92 +/- 5.88% | 60.17 +/- 6.74% | 62.83 +/- 8.31% | 66.08 +/- 6.92% | 66.75 +/- 5.49% |
| Patrolling | 40.16 +/- 6.74% | 54.35 +/- 14.23% | 42.20 +/- 7.12% | 44.78 +/- 4.96% | 44.47 +/- 10.18% |
| Delivery | 33.47 +/- 6.33% | 34.13 +/- 5.70% | 33.60 +/- 4.62% | 32.80 +/- 5.88% | 33.40 +/- 5.49% |

For delivery and patrolling, lower coverage can indicate a more direct route.
Exploration explicitly favors higher coverage.

## Successful completion time

Times are in seconds and include successful runs only. Failed timeout runs are
excluded.

| Task | Baseline | All successful LLM runs |
|---|---:|---:|
| Patrolling | 197.82 +/- 33.00 (n=4) | 224.86 +/- 101.82 (n=63) |
| Delivery | 125.79 +/- 49.63 (n=4) | 135.13 +/- 77.07 (n=75) |

| Task | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|
| Patrolling | 189.54 +/- 57.20 (n=14) | 189.05 +/- 54.24 (n=7) | 197.75 +/- 49.92 (n=14) | 303.34 +/- 140.43 (n=14) | 226.73 +/- 104.45 (n=14) |
| Delivery | 137.22 +/- 80.64 (n=15) | 142.23 +/- 83.74 (n=15) | 135.97 +/- 73.48 (n=15) | 130.66 +/- 88.78 (n=15) | 129.59 +/- 52.86 (n=15) |

## Run counts

| Task | Baseline | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exploration | 5 | 15 | 15 | 15 | 15 | 15 | 80 |
| Patrolling | 5 | 15 | 15 | 15 | 15 | 15 | 80 |
| Delivery | 5 | 15 | 15 | 15 | 15 | 15 | 80 |

## Prompts

| Task | Prompt | Prompt text |
|---|---:|---|
| Exploration | 1 | Explore the environment continuously to maximize area coverage while avoiding obstacles. |
| Exploration | 2 | Explore the area while maximizing the coverage |
| Exploration | 3 | Explore the area as much as possible |
| Exploration | 4 | Explore the environment in a team of 3 robots |
| Exploration | 5 | Maximize the covered area |
| Patrolling | 1 | Patrol the red, green, and blue zones in order. |
| Patrolling | 2 | Each robot should patrol red, green and blue zones respectively |
| Patrolling | 3 | Search and approach red, green and blue zones in order |
| Patrolling | 4 | Visit red, green and blue locations |
| Patrolling | 5 | First go to the red,then to the green and lastly to the blue zones |
| Delivery | 1 | Using three robots, find the red, green, and blue boxes, deliver each box to its matching colored zone, and complete all three deliveries. Each robot should deliver one box. |
| Delivery | 2 | There are 3 robots 3 boxes and 3 delivery zones in the environment. Choose a box and deliver it to its matching zone. Each robot should deliver one box. |
| Delivery | 3 | Choose a box, approach to it, pick it up and drop it to its matching delivery zone. The robot and the delivery zone should be same color. Each robot should deliver one box. |
| Delivery | 4 | Claim a box that is not claimed by another robot, approach to it and pick it up. Then search the delivery zone with the same color, approach it and drop the box there. Each robot should deliver only one box. |
| Delivery | 5 | Choose a box, pick it up and deliver it to its delivery zone, they have to be same color. To pick the box up, the robot has to search it and approach to it, then search the delivery zone and approach to it to drop the box. |

## Seed-by-seed comparisons

Each prompt/seed cell is the mean +/- population standard deviation across 3 generated supervisors. Each baseline/seed cell is the single matched baseline run.

### Collisions

| Task | Seed | Baseline | All LLM generations | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exploration | 1001 | 0 | 0.733 +/- 1.062 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 1.414 | 1.000 +/- 0.816 | 1.667 +/- 0.943 |
| Exploration | 1002 | 1 | 0.267 +/- 0.442 | 0.333 +/- 0.471 | 0.000 +/- 0.000 | 0.667 +/- 0.471 | 0.333 +/- 0.471 | 0.000 +/- 0.000 |
| Exploration | 1003 | 0 | 0.667 +/- 1.398 | 0.000 +/- 0.000 | 1.333 +/- 1.247 | 0.333 +/- 0.471 | 1.667 +/- 2.357 | 0.000 +/- 0.000 |
| Exploration | 1004 | 0 | 0.267 +/- 0.442 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.667 +/- 0.471 | 0.667 +/- 0.471 |
| Exploration | 1005 | 1 | 0.667 +/- 0.943 | 0.333 +/- 0.471 | 1.000 +/- 0.816 | 0.667 +/- 0.943 | 0.333 +/- 0.471 | 1.000 +/- 1.414 |
| Patrolling | 1001 | 0 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Patrolling | 1002 | 0 | 0.200 +/- 0.542 | 0.333 +/- 0.471 | 0.667 +/- 0.943 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Patrolling | 1003 | 0 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Patrolling | 1004 | 3 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Patrolling | 1005 | 1 | 0.067 +/- 0.249 | 0.000 +/- 0.000 | 0.333 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Delivery | 1001 | 2 | 0.133 +/- 0.499 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.667 +/- 0.943 | 0.000 +/- 0.000 |
| Delivery | 1002 | 0 | 0.067 +/- 0.249 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.333 +/- 0.471 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Delivery | 1003 | 0 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Delivery | 1004 | 0 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Delivery | 1005 | 1 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

### Coverage

| Task | Seed | Baseline | All LLM generations | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exploration | 1001 | 86.25% | 66.58 +/- 5.91% | 65.00 +/- 4.68% | 63.33 +/- 5.03% | 70.42 +/- 4.60% | 67.50 +/- 7.71% | 66.67 +/- 4.25% |
| Exploration | 1002 | 81.25% | 60.08 +/- 8.88% | 57.08 +/- 7.80% | 58.33 +/- 8.19% | 56.25 +/- 11.23% | 58.75 +/- 2.04% | 70.00 +/- 3.06% |
| Exploration | 1003 | 85.00% | 64.75 +/- 5.63% | 68.75 +/- 1.02% | 60.83 +/- 3.28% | 60.42 +/- 4.25% | 64.17 +/- 7.38% | 69.58 +/- 0.59% |
| Exploration | 1004 | 86.25% | 64.50 +/- 6.47% | 63.33 +/- 2.95% | 61.25 +/- 5.68% | 64.17 +/- 7.38% | 69.17 +/- 3.58% | 64.58 +/- 8.19% |
| Exploration | 1005 | 67.50% | 63.83 +/- 6.70% | 65.42 +/- 2.57% | 57.08 +/- 8.19% | 62.92 +/- 4.12% | 70.83 +/- 4.12% | 62.92 +/- 4.25% |
| Patrolling | 1001 | 45.88% | 52.08 +/- 10.35% | 48.63 +/- 5.29% | 67.06 +/- 10.91% | 52.55 +/- 4.44% | 46.67 +/- 3.37% | 45.49 +/- 6.95% |
| Patrolling | 1002 | 38.82% | 43.69 +/- 11.67% | 35.29 +/- 3.46% | 54.90 +/- 15.38% | 40.39 +/- 3.88% | 42.75 +/- 3.37% | 45.10 +/- 14.00% |
| Patrolling | 1003 | 37.65% | 41.80 +/- 10.17% | 35.69 +/- 3.64% | 58.04 +/- 11.46% | 38.04 +/- 3.37% | 40.39 +/- 3.37% | 36.86 +/- 2.93% |
| Patrolling | 1004 | 44.71% | 40.24 +/- 7.49% | 37.65 +/- 5.35% | 38.82 +/- 3.46% | 34.90 +/- 2.93% | 45.10 +/- 5.29% | 44.71 +/- 11.08% |
| Patrolling | 1005 | 50.59% | 48.16 +/- 6.95% | 43.53 +/- 3.46% | 52.94 +/- 9.75% | 45.10 +/- 3.09% | 49.02 +/- 4.00% | 50.20 +/- 7.08% |
| Delivery | 1001 | 43.00% | 36.87 +/- 2.00% | 36.67 +/- 2.05% | 37.67 +/- 0.94% | 35.00 +/- 1.41% | 38.33 +/- 1.70% | 36.67 +/- 1.89% |
| Delivery | 1002 | 29.00% | 26.53 +/- 1.67% | 26.67 +/- 0.94% | 27.00 +/- 1.41% | 27.00 +/- 0.00% | 27.67 +/- 1.25% | 24.33 +/- 1.70% |
| Delivery | 1003 | 30.00% | 30.27 +/- 5.08% | 28.67 +/- 6.85% | 30.33 +/- 1.70% | 31.00 +/- 0.82% | 26.67 +/- 6.24% | 34.67 +/- 2.05% |
| Delivery | 1004 | 41.00% | 39.73 +/- 1.81% | 39.33 +/- 1.89% | 41.33 +/- 1.70% | 40.33 +/- 1.25% | 38.00 +/- 0.82% | 39.67 +/- 1.25% |
| Delivery | 1005 | 38.00% | 34.00 +/- 3.78% | 36.00 +/- 4.97% | 34.33 +/- 4.92% | 34.67 +/- 1.89% | 33.33 +/- 2.62% | 31.67 +/- 1.25% |

### Task success

| Task | Seed | Baseline | All LLM generations | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patrolling | 1001 | 0 (failure) | 0.800 +/- 0.400 | 1.000 +/- 0.000 | 0.333 +/- 0.471 | 0.667 +/- 0.471 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Patrolling | 1002 | 1 (success) | 0.867 +/- 0.340 | 1.000 +/- 0.000 | 0.333 +/- 0.471 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Patrolling | 1003 | 1 (success) | 0.867 +/- 0.340 | 1.000 +/- 0.000 | 0.333 +/- 0.471 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Patrolling | 1004 | 1 (success) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Patrolling | 1005 | 1 (success) | 0.667 +/- 0.471 | 0.667 +/- 0.471 | 0.333 +/- 0.471 | 1.000 +/- 0.000 | 0.667 +/- 0.471 | 0.667 +/- 0.471 |
| Delivery | 1001 | 1 (success) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Delivery | 1002 | 1 (success) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Delivery | 1003 | 1 (success) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Delivery | 1004 | 1 (success) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Delivery | 1005 | 0 (failure) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |

### Successful completion time

| Task | Seed | Baseline | All LLM generations | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patrolling | 1001 | failure | 264.52 +/- 113.30 s | 254.39 +/- 76.68 s | 282.20 +/- 0.00 s | 211.22 +/- 9.63 s | 319.25 +/- 166.04 s | 249.55 +/- 111.87 s |
| Patrolling | 1002 | 202.84 s | 226.02 +/- 106.32 s | 160.22 +/- 21.56 s | 152.29 +/- 0.00 s | 203.81 +/- 25.04 s | 286.63 +/- 80.80 s | 277.98 +/- 168.40 s |
| Patrolling | 1003 | 147.25 s | 221.36 +/- 131.33 s | 176.63 +/- 51.22 s | 141.45 +/- 0.00 s | 205.02 +/- 59.92 s | 360.98 +/- 198.75 s | 169.43 +/- 49.41 s |
| Patrolling | 1004 | 201.40 s | 209.90 +/- 74.71 s | 170.75 +/- 19.74 s | 199.01 +/- 46.57 s | 167.32 +/- 53.64 s | 307.47 +/- 94.61 s | 204.97 +/- 22.61 s |
| Patrolling | 1005 | 239.78 s | 202.77 +/- 41.09 s | 183.76 +/- 21.27 s | 150.41 +/- 0.00 s | 205.89 +/- 57.18 s | 211.84 +/- 6.79 s | 234.21 +/- 14.92 s |
| Delivery | 1001 | 196.73 s | 196.94 +/- 77.57 s | 229.17 +/- 94.74 s | 164.20 +/- 75.33 s | 152.78 +/- 29.06 s | 244.01 +/- 90.50 s | 194.52 +/- 10.81 s |
| Delivery | 1002 | 56.57 s | 64.71 +/- 12.85 s | 69.31 +/- 10.26 s | 68.73 +/- 6.21 s | 73.51 +/- 20.34 s | 56.02 +/- 0.58 s | 55.99 +/- 1.15 s |
| Delivery | 1003 | 128.63 s | 120.87 +/- 54.82 s | 115.12 +/- 53.31 s | 87.70 +/- 5.86 s | 166.70 +/- 52.15 s | 103.04 +/- 74.60 s | 131.78 +/- 14.01 s |
| Delivery | 1004 | 121.23 s | 108.72 +/- 44.91 s | 80.03 +/- 8.17 s | 166.52 +/- 38.53 s | 103.45 +/- 11.01 s | 72.53 +/- 1.05 s | 121.06 +/- 52.53 s |
| Delivery | 1005 | failure | 184.43 +/- 79.89 s | 192.45 +/- 25.55 s | 224.00 +/- 108.14 s | 183.40 +/- 120.26 s | 177.69 +/- 29.59 s | 144.62 +/- 31.00 s |

