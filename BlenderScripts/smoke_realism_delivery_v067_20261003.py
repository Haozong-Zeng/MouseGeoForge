"""Exercise the actual portable helper from the copied Windows delivery."""
import bpy
import contextlib
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[-1])
out = root / 'output/pymousestudio_realism_cycle_20261002/v067_portable_helper'
out.mkdir(exist_ok=True)
delivery = Path('C:/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002/DELIVERY')
helper = delivery / 'render_views.py'
records = []
for pose in ['Climb', 'Quadruped', 'HeadRaise']:
    path = delivery / f'PyMouseStudio-RealCapture-{pose}-Static.blend'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(path))
    capture = io.StringIO()
    sys.argv = [str(helper), '--', '--dry-run']
    with contextlib.redirect_stdout(capture):
        runpy.run_path(str(helper), run_name='__main__')
    line = next(x for x in capture.getvalue().splitlines() if x.startswith('DELIVERY_DRY_RUN='))
    info = json.loads(line.split('=', 1)[1])
    assert info['devices'] and info['views'] == list('ABDF')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    records.append(dict(pose=pose, dry_run=info, blend_sha256_unchanged=True))

# One real invocation on the Windows copy; inherited packed dependencies only.
path = delivery / 'PyMouseStudio-RealCapture-HeadRaise-Static.blend'
before = hashlib.sha256(path.read_bytes()).hexdigest()
sys.argv = [str(helper), '--', '--views', 'F', '--output-dir', str(out / 'actual_render')]
runpy.run_path(str(helper), run_name='__main__')
assert (out / 'actual_render/camF.png').is_file()
assert hashlib.sha256(path.read_bytes()).hexdigest() == before
# Refuse to reuse an existing output directory, without changing its contents.
image_hash = hashlib.sha256((out / 'actual_render/camF.png').read_bytes()).hexdigest()
try:
    runpy.run_path(str(helper), run_name='__main__')
except FileExistsError:
    overwrite_refused = True
else:
    raise AssertionError('Helper failed to refuse existing directory')
assert hashlib.sha256((out / 'actual_render/camF.png').read_bytes()).hexdigest() == image_hash
(out / 'result.json').write_text(json.dumps(dict(complete=True, tests=records, windows_copy_rendered=True, blend_unchanged_after_render=True, overwrite_refused=overwrite_refused), indent=2))
print('PORTABLE_HELPER_TEST_COMPLETE')
