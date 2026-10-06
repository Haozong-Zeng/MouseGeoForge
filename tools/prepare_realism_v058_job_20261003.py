import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));spec=json.loads((base/'job_v056_climb.json').read_text());spec.update(id='v058_backdrop_variation',blend=state['best_blend'],script=spec['root']+'\\BlenderScripts\\probe_realism_backdrop_variation_v058_20261003.py',timeout_seconds=900);(base/'job_v058.json').write_text(json.dumps(spec,indent=2));print('Prepared background-only trial.')
