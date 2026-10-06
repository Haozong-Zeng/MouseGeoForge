import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);POSE_TAG='headraise'
exec(compile((root/'BlenderScripts/sync_realism_pose_appearance_v046_20261003.py').read_text(encoding='utf-8'),'sync_pose_appearance','exec'))
