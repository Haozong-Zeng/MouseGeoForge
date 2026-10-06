"""Record bounded native-context teacher launch without changing its running stage."""
import datetime as dt
import json
import numpy as np
from run_teacher_cycle_dlc_20261001 import OUT, write_state

run = OUT / 'lp_context_seed17_kl'
m = json.loads((OUT / 'lp_context_dataset_v1/manifest.json').read_text())
smoke = json.loads((run / 'smoke.json').read_text())
assert smoke['passed'] and m['center_exact_checks'] == 1331
for name, key in [('lp_seed17_kl', 'uv'), ('lp_ensemble_17_31_73', 'uv_median'), ('dlc_extension_review', 'uv')]:
    with np.load(OUT / name / 'predictions_source_pixels.npz') as z:
        assert z[key].shape == (1336, 7, 2)
record = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), decision='Launch one native LP MHCRNN seed17 context teacher',
    dataset=dict(train_images=1043, validation_images=288, center_png_exact=1331, total_context_pngs=6461,
        pending_24_groups_excluded=True, SEM01_entire_group_excluded=True, locked_holdout_opened=False),
    smoke=smoke, limits=dict(epochs=30, wall_minutes=60, evaluation_reserve_minutes=5, global_deadline_utc='2026-10-02T07:21:52+00:00'),
    rationale='Posthoc temporal joint states increased accepted coverage without changing any common accepted predictions or correcting severe identity errors. Test learned image context instead of further rule tuning.',
    evaluation=['Same288developmentimages for all comparisons; not a new independent test',
        'SF,MF,nativeconfidence-selected heads and same-checkpoint repeated-center input',
        'Static seed17, static3seedmedian, DLC70 baseline on exact same subset',
        'Inspect source-pixel mean/P90 and severe paw errors; no strong3D or pseudo-GT release'],
    run=str(run), next_action='Inspect convergence and predictions at next heartbeat. Do not start another GPU job while this PID/lock is live.',
    originals_modified=False, pseudo_gt_released=False)
p = OUT / 'heartbeat_20261002_0400_decision.json'
p.write_text(json.dumps(record, indent=2))
updates = dict(latest_decision=str(p), project_config=str(run / 'settings.json'), checkpoint=None, result=None,
               batches_seen=None, session=None, pseudo_gt_released=False)
if not (run / 'epochs.jsonl').exists():
    updates['last_epoch_record'] = None
write_state(**updates)
print(p)
