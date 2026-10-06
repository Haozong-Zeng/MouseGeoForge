import runpy,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];scale=json.loads((root/'output/pymousestudio_realism_cycle_20261002/v004_upright_collision_guard/manifest.json').read_text())['scale']
runpy.run_path(str(root/'tools/rank_realism_pose_seeds_20261003.py'),run_name='__main__',init_globals={'FIXED_SCALE':scale,'OUTPUT_NAME':'pose_seed_fixedscale_ranking.json'})
