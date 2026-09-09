"""Find actual frame-number gaps before completing renderable animation sequences."""
import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]
rows=[]
for root in sorted((BASE/'dataset/BlenderPyMouse').iterdir()):
    if not root.is_dir():
        continue
    frames=sorted(int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir() and p.name[6:].isdigit())
    if len(frames)<2:
        continue
    steps,counts=np.unique(np.diff(frames),return_counts=True)
    meta=json.loads((root/'metadata.json').read_text()) if (root/'metadata.json').exists() else {}
    record={'dataset':root.name,'count':len(frames),'range':[frames[0],frames[-1]],
        'steps':dict(zip(map(str,steps),map(int,counts))),
        'metadata_frame_step':meta.get('frame_step'),
        'temporal_mode':meta.get('temporal',{}).get('temporal_mode'),
        'source_blends':[str(p.relative_to(BASE)) for p in (BASE/'BlenderPyMouseProjects').rglob(root.name+'.blend')]}
    rows.append(record)
    print(json.dumps(record),flush=True)
out=BASE/'output/temporal_completion'
out.mkdir(parents=True,exist_ok=True)
(out/'source_audit.json').write_text(json.dumps(rows,indent=2))
