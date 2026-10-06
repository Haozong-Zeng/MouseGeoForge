import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1])
OUT_TAG='v025_quadruped_tail'
BLEND_NAME='PyMouseStudio-RealCapture-Replica-v025-quadruped-tail.blend'
WALL_AWARE=True
script=root/'BlenderScripts/build_realism_quadruped_20261003.py'
exec(compile(script.read_text(encoding='utf-8'),str(script),'exec'))
