# Patrolling prompt and YAML analysis

This analysis covers all 300 runs under `patrolling/with_fixed_spec`: six prompts,
five generated YAML controllers per prompt, and ten random seeds per controller.

Ranking uses full-task completion as the primary metric. Mean completion time among
successful runs is the first tie-breaker, followed by one-second-debounced collision
count. Partial completion is the percentage of the nine robot/color targets reached.

## Best result

- Best prompt: **Prompt 1**
- Best generated controller: **Prompt 1, generation 4 — `S_20260912_105840.yaml`**
- Full completions: **10/10 (100%)**
- Mean completion time: **163.301 s** (population SD 39.254 s)
- Mean partial completion: **100%**
- Collisions: **1 total across 10 runs** (0.10/run, one-second pair debounce)

Prompt 1 text:

> Search for and visit the red, green, and blue locations in that order, while avoiding obstacles. If you reach a location out of sequence, continue your search for the correct location.

## Top controllers

| Rank | Prompt | Generation | YAML | Success | Mean successful time | Collisions |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 4 | `S_20260912_105840.yaml` | 10/10 | 163.301 s | 1 |
| 2 | 3 | 1 | `S_20260912_110432.yaml` | 10/10 | 185.405 s | 2 |
| 3 | 4 | 2 | `S_20260912_110647.yaml` | 10/10 | 235.810 s | 1 |
| 4 | 1 | 5 | `S_20260912_105914.yaml` | 9/10 | 171.614 s | 4 |
| 5 | 2 | 1 | `S_20260912_110004.yaml` | 9/10 | 162.110 s | 2 |

The fifth-ranked YAML is slightly faster when it succeeds, but its 90% completion
rate makes it less reliable than the three 100% controllers.

## Prompt-level ranking

Each prompt total contains 50 runs (five independently generated YAMLs by ten seeds).

| Rank | Prompt | Success | Mean partial completion | Mean successful time | Collisions/run |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 40/50 (80%) | 95.56% | 180.329 s | 0.26 |
| 2 | 2 | 39/50 (78%) | 94.67% | 189.756 s | 0.22 |
| 3 | 4 | 39/50 (78%) | 94.22% | 216.872 s | 0.54 |
| 4 | 3 | 34/50 (68%) | 93.11% | 193.631 s | 0.34 |
| 5 | 6 | 18/50 (36%) | 87.11% | 224.908 s | 0.44 |
| 6 | 5 | 13/50 (26%) | 83.78% | 239.682 s | 0.36 |

Prompt 1 is the practical winner because it has the highest completion rate and is
also faster than the other high-reliability prompts. The 80% versus 78% difference
between Prompts 1 and 2 is only one run and should not be interpreted as strong
statistical separation. The much larger gaps to Prompts 5 and 6 are operationally
meaningful in this dataset.
