import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));template=json.loads((base/'job_v059.json').read_text())
for tag in ['quadruped','headraise']:
    spec=dict(template);spec.update(id='v060_'+tag,blend=state['pose_bank'][tag],script=spec['root']+'\\BlenderScripts\\sync_realism_supports_'+tag+'_v060_20261003.py');(base/f'job_v060_{tag}.json').write_text(json.dumps(spec,indent=2))
print('Prepared two independent same-prop construction jobs.')
