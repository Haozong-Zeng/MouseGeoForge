"""Read-only progress for the resumable temporal rendering queue."""
import json
import os
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/temporal_completion'
plan=json.loads((OUT/'plan.json').read_text())
state=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
try:
    os.kill(state['pid'],0)
    state['queue_process_alive']=True
except (KeyError,ProcessLookupError):
    state['queue_process_alive']=False
rows=[]
for job in plan['jobs']:
    for name in [job['dataset']]+([job['transferred']] if job['transferred'] else []):
        root=BASE/'dataset/BlenderPyMouse'/name
        frames={int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir()}
        added=len(frames-set(job['original_frames']))
        rows.append({'dataset':name,'available':len(frames),'target':job['end_frame'],
            'new_frames_published':added,'new_frames_target':len(job['missing_frames'])})
state['datasets']=rows
state['new_frames_published']=sum(r['new_frames_published'] for r in rows)
state['new_frames_target']=sum(r['new_frames_target'] for r in rows)
log=BASE/state.get('active_log','nonexistent')
if log.is_file():
    state['active_log_tail']=log.read_text(errors='replace').splitlines()[-8:]
print(json.dumps(state,indent=2,ensure_ascii=False))
