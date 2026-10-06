"""Read-only inventory and source job plan for approved surface migration."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/surface_keypoint_migration_v3';OUT.mkdir(exist_ok=True)
jobs=[];skipped=[]
for ds in sorted((ROOT/'dataset/BlenderPyMouse').iterdir()):
    if not ds.is_dir() or ds.name.startswith('_'):continue
    files=sorted(ds.glob('frame_*/keypoints.npz'))
    if not files:continue
    with np.load(files[0],allow_pickle=False) as z:names=z['names'].tolist()
    if len(names)!=23 or names[5] not in ('spine_center','dorsal_thorax_surface'):
        skipped.append(dict(dataset=ds.name,frames=len(files),reason='29-point forelimb/hand format; no affected anatomical channels'))
        continue
    sample=json.loads(files[0].with_suffix('.json').read_text())
    meta=json.loads((ds/'metadata.json').read_text())
    source=meta.get('source_blend',sample['source_blend'])
    jobs.append(dict(dataset=ds.name,source_blend=source,frames=len(files),
                     mesh_names=[i['mesh'] for i in sample['instances']] if 'instances' in sample else [sample['mesh']],
                     armature_names=[i['armature'] for i in sample['instances']] if 'instances' in sample else [sample['armature']]))
jobs.sort(key=lambda j:(j['dataset']!='PyMouseStudio-BrownFur-WalkAround-Core23',j['dataset']))
result=dict(jobs=jobs,skipped=skipped,frames=sum(j['frames'] for j in jobs))
(OUT/'inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
