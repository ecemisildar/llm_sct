# Chosen controllable events by prompt

`prompt_chosen_event_tables.tex` includes two tables per task: LLM and LLM + fixed specification. Individual task files include both settings, and individual setting files are also available. Each table starts with a baseline column, followed by the prompt IDs recorded for each task and a run count row. The delivery baseline uses the older three-target task; current delivery controllers use six targets.

LaTeX cells show mean selected-event percentages across completed runs. Each run pools its robots; runs receive equal weight. All generations and seeds in a prompt are combined. Missing events contribute zero; missing run groups show --. Values can differ slightly from 100% after rounding.

The CSV also includes population standard deviation, event counts, and pooled percentages. The run manifest records the exact completed-run snapshot used by these tables. Only selected controllable events are included. Event names use the same normalization as the overall tables.

Reproduce from src/llm_sct:

```bash
python3 evaluation/evaluation/export_prompt_event_tables.py --run-manifest RESULTS_LAST/analysis/controllable_event_distribution/controllable_event_distribution_runs.csv --output-dir RESULTS_LAST/analysis/prompt_chosen_events
```
