import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);script=root/'BlenderScripts/build_realism_camera_props_v030_20261003.py'
OUT_TAG='v030_camera_props_r1'
exec(compile(script.read_text(encoding='utf-8'),str(script),'exec'))
