import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002'
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));spec=json.loads((base/'job_v056_quadruped.json').read_text())
spec.update(id='v057_tail_rings',blend=state['pose_bank']['quadruped'],script=spec['root']+'\\BlenderScripts\\probe_realism_tail_rings_v057_20261003.py',timeout_seconds=900)
(base/'job_v057.json').write_text(json.dumps(spec,indent=2))
print('Prepared v057 tail material-only A/F trial job.')
