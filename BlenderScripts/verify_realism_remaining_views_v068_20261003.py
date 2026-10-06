"""Render untouched delivered static poses from the three remaining side cameras."""
import bpy
import hashlib
import json
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[-1])
out = root / 'output/pymousestudio_realism_cycle_20261002/v068_remaining_views'
out.mkdir(exist_ok=True)
delivery = Path('C:/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
helper = delivery / 'render_views.py'
records = []
for pose in ['Climb', 'Quadruped', 'HeadRaise']:
    path = delivery / f'PyMouseStudio-RealCapture-{pose}-Static.blend'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(path))
    sys.argv = [str(helper), '--', '--views', 'CEG', '--output-dir', str(out / pose.lower())]
    runpy.run_path(str(helper), run_name='__main__')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    records.append(dict(pose=pose, frame=bpy.context.scene.frame_current, views=list('CEG'), blend_sha256=before, blend_unchanged=True))
    (out / 'manifest.json').write_text(json.dumps(dict(complete=len(records)==3, poses=records), indent=2))
print('REMAINING_VIEWS_COMPLETE')
