# Controllable event distribution tables

`controllable_event_distribution.tex` contains one table per task. Individual task `.tex` files can also be included directly with `\input{...}`. No extra packages are required.

Each cell is mean ± population standard deviation of the per-run event percentage. All robots are pooled within a run; runs are equally weighted. Events absent from a run contribute zero. Only selected controllable events are counted; broadcasts and detected uncontrollable events are excluded. The EV_ and task_ prefixes are normalized; delivery zone a/b aliases map to red/blue. Color-specific patrolling events remain separate.

Uses RESULTS_LAST, latest YAML per prompt/generation/setting, seeds 1001–1010, latest completed Saving OK run per exact metadata YAML path and seed. Baselines use the latest completed run per seed (1001–1010) from new_results/baseline. The delivery baseline is the older three-target task; current delivery controllers use six targets. Baseline and LLM event schemas can differ. The run manifest records the exact snapshot. The summary CSV also includes pooled counts and percentages (which weight runs by their number of selections).

Reproduce from src/llm_sct:

```bash
python3 evaluation/evaluation/export_controllable_event_tables.py --experiment-root RESULTS_LAST --output-dir RESULTS_LAST/analysis/controllable_event_distribution
```
