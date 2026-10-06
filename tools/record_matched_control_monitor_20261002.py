"""Persist monitoring evidence while leaving live training ownership unchanged."""
import datetime as dt
import json
import os
from run_teacher_cycle_dlc_20261001 import OUT, write_state

state = json.loads((OUT / 'state.json').read_text())
assert state['active_run'] == 'lp_contextoff_seed17_kl'
os.kill(state['pid'], 0)
os.kill(state['cpu_matched_review_pid'], 0)
epochs = [json.loads(line) for line in (OUT / 'lp_contextoff_seed17_kl/epochs.jsonl').read_text().splitlines()]
report = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), training_pid=state['pid'],
    completed_epochs=len(epochs), latest_epoch=epochs[-1], best_epoch=min(epochs, key=lambda e:e['metrics']['val_supervised_rmse']),
    decision='Continue fixed30epoch control; no additional GPUjob, no budget extension, no premature ON/OFF conclusion.',
    followup=dict(pid=state['cpu_matched_review_pid'], script='tools/compare_lp_context_ablation_20261002.py',
        assertions='Identical seed/backbone/input/loss/schedule/batch/split/labelindices; both must complete30epochs and7830steps.',
        intended_inputs='ON realneighbors vs OFF repeatedcenter; nativeconfidence-selection primary plus MF/SF separately.',
        outputs='Fixed288image metrics perkeypoint/camera/session, severeerrors, pairedgain/harm counts, descriptiveframe-clusterbootstrap,18rankedvisualcases.',
        warning='Bootstrap conditional on2reusedvideos, not animal/session generalization or independenttest.',
        timeout='45minutes or globaldeadline, whicheverfirst; STOP respected; noautomaticGPUlaunch.'),
    next_action='Nextheartbeat inspect control/checkpoints and matchedreport; visually review successes/failures before deciding whether temporalteacher should enter continuousclip/LOOgeometry diagnostics.',
    original_labels_modified=False, locked_holdout_opened=False, pseudo_gt_released=False)
p = OUT / 'heartbeat_20261002_0525_decision.json'
p.write_text(json.dumps(report, indent=2))
write_state(latest_decision=str(p), next_action=report['next_action'])
print(p, 'completedepochs', len(epochs))
