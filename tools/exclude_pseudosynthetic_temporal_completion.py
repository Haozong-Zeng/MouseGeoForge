"""Apply the user's explicit exclusion without stopping the current Blender batch."""
import os
import signal
import time
from datetime import datetime, timezone
from complete_subsampled_blender_datasets import OUT, DATA, WHITE, read
from core23_temporal_format import write_json_atomic

plan=read(OUT/'plan.json')
removed=[j for j in plan['jobs'] if 'PseudoSynthetic' in j['dataset']]
if not removed:
    print('PseudoSynthetic already excluded')
    raise SystemExit(0)
retained=[j for j in plan['jobs'] if 'PseudoSynthetic' not in j['dataset']]
assert [j['dataset'] for j in retained]==[WHITE]
state=read(OUT/'status.json')
assert state['dataset']==WHITE
updates=[]
for job in removed:
    for name in (job['dataset'],job['transferred']):
        root=DATA/name
        frames=sorted(int(p.name[6:]) for p in root.glob('frame_*') if p.is_dir())
        assert frames==job['original_frames'], f'Unexpected already-rendered frames: {name}'
        current=read(root/'metadata.json')
        original=read(OUT/'original'/name/'metadata.json')
        assert current['temporal_completion']['provenance']=='output/temporal_completion/plan.json'
        for key in ('frame_step','frame_numbering','body_mask_annotations'):
            if key in original:
                current[key]=original[key]
            else:
                current.pop(key,None)
        current.pop('temporal_completion',None)
        if 'body_masks' not in original.get('layout',{}):
            current.get('layout',{}).pop('body_masks',None)
        current['temporal']=dict(current.get('temporal',{}),temporal_mode='pose_set',
            temporal_eligible=False,evidence='user_confirmed_not_reliable_temporal')
        for key in ('export_complete','source_animation_fps'):
            current['temporal'].pop(key,None)
        current['render_completion_exclusion']={
            'reason':'User confirmed PseudoSynthetic is not reliable temporal data.',
            'cancelled_before_render':True,'existing_frames_preserved':True}
        updates.append((root/'metadata.json',current))

pid=state['pid']
process=__import__('pathlib').Path(f'/proc/{pid}')
cmd=(process/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
assert 'tools/complete_subsampled_blender_datasets.py' in cmd,cmd
assert '--wait-for-blender-pid' not in cmd,cmd
archive=OUT/'plan_before_pseudosynthetic_exclusion.json'
assert not archive.exists(),archive
write_json_atomic(archive,plan)
os.kill(pid,signal.SIGTERM)  # exact scheduler only; the Windows Blender child continues
for _ in range(50):
    if not process.exists() or (process/'stat').read_text().split()[2]=='Z':
        break
    time.sleep(.1)
else:
    raise RuntimeError('Old scheduler did not exit')
plan['jobs']=retained
plan['excluded_jobs']=removed
plan['exclusion_reason']='User confirmed PseudoSynthetic is not reliable temporal data.'
plan['scope']='Only complete WhiteFur WaterReaching FGcamB; no PseudoSynthetic render or style transfer.'
plan['revised_at_utc']=datetime.now(timezone.utc).isoformat()
write_json_atomic(OUT/'plan.json',plan)
for path,metadata in updates:
    write_json_atomic(path,metadata)
state.update(phase='scheduler_reconfigured',excluded_datasets=[p.parent.name for p,_ in updates],
             updated_at_utc=plan['revised_at_utc'])
write_json_atomic(OUT/'status.json',state)
print({'stopped_scheduler_pid':pid,'removed_datasets':len(updates),'retained':WHITE,
       'existing_frames_preserved':True},flush=True)
