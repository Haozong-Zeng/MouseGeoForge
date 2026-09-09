"""Launch the existing render queue detached from the interactive terminal (WSL)."""
from pathlib import Path
import json
import os
import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--wait-for-blender-pid',type=int,default=0)
args=parser.parse_args()

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/temporal_completion'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'status.json').exists():
    state=json.loads((OUT/'status.json').read_text())
    try:
        os.kill(state['pid'],0)
    except (KeyError,ProcessLookupError):
        pass
    else:
        raise RuntimeError('An existing queue/test process is still alive; do not launch a duplicate')
with (OUT/'queue.log').open('ab',buffering=0) as log:
    command=[str(BASE/'.venv-vggt-omega-eval/bin/python'),'-u',
        str(BASE/'tools/complete_subsampled_blender_datasets.py'),'--chunk-size','8']
    if args.wait_for_blender_pid:
        command += ['--wait-for-blender-pid',str(args.wait_for_blender_pid)]
    child=subprocess.Popen(command,
        cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'log':str(OUT/'queue.log'),'launched':True}),flush=True)
