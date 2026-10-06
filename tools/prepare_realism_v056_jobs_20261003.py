import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));template=json.loads((base/'job_v055.json').read_text())
for tag,script in [('quadruped','common'),('headraise','headraise'),('climb','climb')]:
    spec=dict(template);spec.update(id='v056_'+tag,blend=state['best_blend'] if tag=='climb' else state[tag+'_candidate_blend'],script=spec['root']+'\\BlenderScripts\\sync_realism_'+script+'_scene_v056_20261003.py',timeout_seconds=900)
    (base/f'job_v056_{tag}.json').write_text(json.dumps(spec,indent=2))
print('Prepared three versioned integration jobs; no worker started by this script.')
