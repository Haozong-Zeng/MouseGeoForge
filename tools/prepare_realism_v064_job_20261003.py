import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));spec=json.loads((base/'job_v059.json').read_text());spec.update(id='v064_final_delivery',blend=state['best_blend'],script=spec['root']+'\\BlenderScripts\\finalize_realism_delivery_v064_20261003.py',timeout_seconds=1200);(base/'job_v064.json').write_text(json.dumps(spec,indent=2));print('Prepared three-copy save/reopen and twelve-view verification job.')
