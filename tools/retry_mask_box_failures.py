"""Preserve failed pre-protocol attempt; select actual stage12 training slots."""
import json
import subprocess
import sys
from pathlib import Path
import audit_mask_box_failures as audit

ROOT=audit.ROOT
audit.OUT=ROOT/'output/mask_box_failures_v1_actual_train'
schedule_path=ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json'
original_selected=audit.selected
original_write=audit.write_json
hashes={str(schedule_path):audit.fingerprint(schedule_path),str(Path(__file__).resolve()):audit.fingerprint(Path(__file__).resolve())}


def select_actual(spec):
    schedule=json.loads(schedule_path.read_text())['arms']['adaptive_temperature']
    keys={k for slot in schedule for k in slot}
    filtered={k:r for k,r in spec['records'].items() if r['split']!='train' or k in keys}
    assert all(k in spec['records'] and spec['records'][k]['split']=='train' for k in keys)
    return original_selected(dict(spec,records=filtered))


def write(path,value):
    if path.name=='protocol.json':
        value=dict(value,selection_input_sha256=hashes,
            retry_reason='Original record pool includes eight superseded ClimbWall scenes. Failed scene-count assertion before protocol/GPU. Retry uses only keys present in the actual stage12 training schedule:40 scenes, not all48 train-tagged scenes.')
    original_write(path,value)


if __name__=='__main__':
    if '--launch' in sys.argv:
        if (audit.OUT/'status.json').exists():raise RuntimeError('Already launched')
        audit.OUT.mkdir(parents=True,exist_ok=True)
        with (audit.OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(audit.OUT))))
    else:
        audit.selected=select_actual;audit.write_json=write;audit.run()
        assert all(audit.fingerprint(Path(p))==h for p,h in hashes.items())
        original_write(audit.OUT/'selection_verification.json',dict(passed=True,selection_inputs_unchanged=True,input_sha256=hashes))
