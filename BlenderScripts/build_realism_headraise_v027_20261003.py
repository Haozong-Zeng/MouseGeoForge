import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1])
OUT_TAG='v027_low_headraise'
BLEND_NAME='PyMouseStudio-RealCapture-Replica-v027-low-headraise.blend'
WALL_AWARE=True
POSE_FRAME=216
REFERENCE_SESSION='AZR2L2'
REFERENCE_FRAME=129
RANKING_FILE='pose_seed_fixedscale_all.json'
LOCAL_PITCHES=[('Neck2_M',10),('Head1_M',15)]
TAIL_EXTRA_CLEARANCE=.5
script=root/'BlenderScripts/build_realism_quadruped_20261003.py'
exec(compile(script.read_text(encoding='utf-8'),str(script),'exec'))
