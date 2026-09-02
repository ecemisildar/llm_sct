# Pre-event-cleanup archive

Archived on 2026-08-28 before regenerating LLM specifications without the
color-specific `search_red`, `search_green`, `search_blue`, `approach_red`,
`approach_green`, and `approach_blue` events.

Contents:

- `results/`: previous LLM, comparison, mixed-mission, and summary results.
- `llm_outputs/`: previous raw LLM JSON and prompt records.
- `llm_generated_automata/`: previous generated specification XML files.
- `resulting_automata/`: previous synchronized automata and supervisor YAMLs.
- `legacy_baseline_layout/`: superseded shared and obstacle-avoidance baseline
  directories retained after the fixed models were consolidated by task.
- `legacy_task_collision_avoidance/`: superseded patrolling- and
  delivery-specific collision specifications, replaced by one canonical fixed
  collision-avoidance specification shared by all tasks.
- `redundant_delivery_xml/`: the one-state communication plant and overlapping
  claim-coordination specification removed during delivery-model cleanup.

The baseline simulation results remain in `../../results/baseline` because the
baseline event alphabets and trials are unchanged.
