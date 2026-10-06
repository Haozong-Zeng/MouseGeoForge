import datetime as dt,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'output/pymousestudio_realism_cycle_20261002'
m=json.loads((out/'baseline_v000/manifest.json').read_text());assert len(m['renders'])==4
stats=[]
for r in m['renders']:
    p=out/'baseline_v000'/(r['camera']+'.png')
    with Image.open(p) as im:a=np.array(im.convert('RGB'));size=im.size
    assert size==(512,512)
    stats.append(dict(camera=r['camera'],rgb_min=int(a.min()),rgb_max=int(a.max()),rgb_mean=float(a.mean())))
plan=json.loads((out/'plan.json').read_text());assert hashlib.sha256(Path(plan['source_blend']).read_bytes()).hexdigest()==plan['source_sha256']
review=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),automation_id='pymousestudio',renders_verified=stats,
    inspected_images=['baseline_v000/Camera.007.png','baseline_v000/Camera.001.png','references/FR2/frame_000649/camA.png'],
    observations=['Bottom camera007 is completely black; side001 does show mouse and enclosure, so do not treat global rendering as broken. Diagnose bottom occlusion/near clipping/view-layer/lighting by ray and scene inspection before changing materials.',
        'Side001 is brown/dark smooth-looking mouse against pale beige/white enclosure, pink-red paws, weak visible fine fur. Real reference is grayscale with dark background and bright feet/tail, much stronger local reflections and curved clear boundary.',
        'Existing frame31 is a vertical rearing pose. It is not registered to real FR2f649; do not claim pixel correspondence. Need scale/camera/scene registration before pose error measurements.'],
    process_note='PowerShell initial wrapper marked failure despite four completed renders and clean Blender quit log, likely ExitCode retrieval ambiguity. Verified all4 output files/manifest; process has ended. Refresh added to future wrapper. This is output-complete, not proof of every render being useful.',
    next='First resolve bottom black frame with read-only ray/visibility audit, establish camera/world transform and black mouse material reference. Save v001 and render same four cameras; inspect shader/hair settings before blind parameter changes.',
    source_unchanged=True,deadline_utc=plan['deadline_utc'])
(out/'baseline_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
state=json.loads((out/'state.json').read_text(encoding='utf-8-sig'));state.update(stage='baseline_four_outputs_verified_bottom_black_needs_fix',active_pid=None,automation_id='pymousestudio',latest_review=str(out/'baseline_review.json'),next_action=review['next'],updated_utc=review['utc'])
(out/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
export=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-RealCapture-Replica-20261002');export.mkdir(exist_ok=True)
shutil.copytree(out/'baseline_v000',export/'baseline_v000',dirs_exist_ok=True)
for f in ['plan.json','baseline_review.json']:shutil.copy2(out/f,export/f)
print(json.dumps(review,ensure_ascii=False,indent=2))
