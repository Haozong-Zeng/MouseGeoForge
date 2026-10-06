import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));spec=json.loads((base/'job_v062.json').read_text());spec.update(id='v063_implant_bevel',blend=state['best_blend'],script=spec['root']+'\\BlenderScripts\\probe_realism_implant_bevel_v063_20261003.py');(base/'job_v063.json').write_text(json.dumps(spec,indent=2));print('Prepared true mesh-bevel trial, shadow-caustic receiver remains on.')
