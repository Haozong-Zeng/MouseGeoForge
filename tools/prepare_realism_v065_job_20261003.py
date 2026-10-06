import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'output/pymousestudio_realism_cycle_20261002';m=json.loads((base/'v064_final_delivery/manifest.json').read_text());assert m['complete'];spec=json.loads((base/'job_v064.json').read_text());spec.update(id='v065_repeatability',blend=m['poses']['climb']['delivery_blend'],script=spec['root']+'\\BlenderScripts\\check_realism_repeatability_v065_20261003.py',timeout_seconds=600);(base/'job_v065.json').write_text(json.dumps(spec,indent=2));print('Prepared unchanged-file repeat renders.')
