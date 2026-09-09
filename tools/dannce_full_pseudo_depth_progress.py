"""Read-only counts for 25fps full-length DANNCE mask/depth production."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/dannce_full_pseudo_depth'
state=json.loads((OUT/'status.json').read_text())
p=Path(f'/proc/{state.get("pid",0)}/cmdline')
state['process_alive']=p.exists() and b'generate_dannce_full_pseudo_depth.py' in p.read_bytes()
records=[]
for mouse in (1,2):
    session=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'
    root=session/'pseudo_depth_full_v1'
    chunks=[json.loads(p.read_text()) for p in root.glob('chunks/*/chunk_complete.json')]
    records.append({'mouse':mouse,'target_frames':4500,'validated_published_frames':sum(c['frame_count'] for c in chunks),
        'depth_frames_computed':len(list(root.glob('chunks/*/frame_metrics/*.json'))),
        'mask_images_written':len(list(root.glob('chunks/*/masks/*/*.png'))),
        'invalid_frames':sum(c['invalid_depth_frames'] for c in chunks)})
state['datasets']=records
print(json.dumps(state,indent=2))
