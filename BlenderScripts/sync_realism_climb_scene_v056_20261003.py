import sys
from pathlib import Path
POSE_TAG='climb';root=Path(sys.argv[sys.argv.index('--')+1])
exec(compile((root/'BlenderScripts/sync_realism_common_scene_v056_20261003.py').read_text(encoding='utf-8'),'common_scene_sync','exec'))
