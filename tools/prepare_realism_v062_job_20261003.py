import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));spec=json.loads((base/'job_v059.json').read_text());spec.update(id='v062_implant_caustics',blend=state['best_blend'],script=spec['root']+'\\BlenderScripts\\diagnose_realism_implant_caustics_v062_20261003.py',timeout_seconds=600);(base/'job_v062.json').write_text(json.dumps(spec,indent=2));print('Prepared one-view diagnostic without blend save.')
