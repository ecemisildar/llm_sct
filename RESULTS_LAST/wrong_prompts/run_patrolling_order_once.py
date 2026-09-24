"""Make exactly one request for an explicitly ordered patrolling task."""
import functools
import json
import sys
from datetime import datetime
from pathlib import Path
import openai

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'llm_part'))
import llm_input
from run_pipeline import run_pipeline

# Disable SDK transport retries as well as generation repair attempts.
openai.OpenAI = functools.partial(openai.OpenAI, max_retries=0)
original_request = llm_input.request_json
calls = 0
output = BASE / ('patrolling_blue_green_red_once_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
output.mkdir()
task = 'As a team, visit the blue, green, and red regions in this order.'
(output / 'prompt.txt').write_text(task + '\n')

def one_request(*args, **kwargs):
    global calls
    calls += 1
    assert calls == 1, 'Only one API call is authorized'
    payload = original_request(*args, **kwargs)
    (output / 'raw_response.json').write_text(json.dumps(payload, indent=2) + '\n')
    return payload

llm_input.request_json = one_request
print('Output:', output, flush=True)
result = run_pipeline(task=task, mission='patrolling', prompt_number=8, generation_number=1,
                      collision_control='fixed', output_root=output,
                      max_repair_attempts=0, auto_feedback=False,
                      status=lambda message: print(message, flush=True))
print('API calls:', calls, flush=True)
print('JSON:', result.json_path, flush=True)
print('YAML:', result.yaml_path, flush=True)
