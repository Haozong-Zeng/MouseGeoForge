import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);POSE_TAG='headraise'
exec(compile((root/'BlenderScripts/integrate_realism_pose_v041_20261003.py').read_text(encoding='utf-8'),'integrate_pose','exec'))
