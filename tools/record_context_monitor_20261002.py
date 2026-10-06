"""Record the live context-training decision without resetting current progress."""
import datetime as dt
import json
import os
from run_teacher_cycle_dlc_20261001 import OUT, write_state

state = json.loads((OUT / 'state.json').read_text())
assert state['active_run'] == 'lp_context_seed17_kl'
os.kill(state['pid'], 0)
review_pid = 106869
os.kill(review_pid, 0)
epochs = [json.loads(line) for line in (OUT / 'lp_context_seed17_kl/epochs.jsonl').read_text().splitlines()]
best = min(epochs, key=lambda e: e['metrics']['val_supervised_rmse'])
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), training_pid=state['pid'], cpu_review_waiter_pid=review_pid,
    completed_epochs=len(epochs), best_epoch=best, latest_epoch=epochs[-1],
    decision='Continue the already-running 30epoch bounded context pilot. Do not extend its training budget or launch another GPU job.',
    reason='Validation improved substantially early but fluctuates recently. Complete planned learning-rate decays before deciding; no evidence yet for a longer budget.',
    automatic_followup='CPU reviewer waits for summary.json up to60min/globaldeadline, then produces fixed-cohort metrics,24ranked gains/harms/failures and learning curve. It does not launch GPU work or change labels.',
    reviewer_log=str(OUT / 'lp_context_review.log'), reviewer_output=str(OUT / 'lp_context_seed17_kl/review'),
    next_action='At next heartbeat check training and CPU reviewer logs, visually inspect gain/harm/failure panels, then judge same-checkpoint contextual contribution separately from static-architecture comparison.',
    unchanged_guards='24pending groups andSEM01 excluded; lockedholdout untouched; no pseudo-GT release; original annotations/calibration unchanged.')
path = OUT / 'heartbeat_20261002_0425_decision.json'
path.write_text(json.dumps(report, indent=2))
write_state(latest_decision=str(path), cpu_review_pid=review_pid, cpu_review_log=report['reviewer_log'], next_action=report['next_action'])
print(json.dumps(report, indent=2))
