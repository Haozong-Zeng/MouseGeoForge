import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);POSE_TAG='headraise'
exec(compile((root/'BlenderScripts/export_realism_pose_payload_v041_20261003.py').read_text(encoding='utf-8'),'export_pose_payload','exec'))
