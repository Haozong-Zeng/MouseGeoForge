"""Start the 25fps DANNCE pseudo-depth worker independently of the terminal."""
from pathlib import Path
import json
import os
import subprocess
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/dannce_full_pseudo_depth'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'status.json').exists():
    state=json.loads((OUT/'status.json').read_text())
    p=Path(f'/proc/{state.get("pid",0)}/cmdline')
    if p.exists() and b'generate_dannce_full_pseudo_depth.py' in p.read_bytes():
        raise RuntimeError('DANNCE generator is already running')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TQDM_DISABLE='1')
with (OUT/'queue.log').open('ab',buffering=0) as log:
    process=subprocess.Popen([str(BASE/'.venv-vggt-omega-eval/bin/python'),'-u',
        str(BASE/'tools/generate_dannce_full_pseudo_depth.py')],cwd=BASE,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':process.pid,'log':str(OUT/'queue.log'),'launched':True}))
