import sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1]);OUTPUT_RELATIVE='output/pymousestudio_realism_cycle_20261002/v053_sparse_dust/native128';RENDER_CASES=[('B',True),('D',True)]
exec(compile((root/'BlenderScripts/render_realism_native128_20261002.py').read_text(encoding='utf-8'),'render_native128','exec'))
