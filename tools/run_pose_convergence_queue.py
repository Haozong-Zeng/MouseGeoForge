"""Launch a bounded, serial 256/512 convergence pair with fail-fast verification.

Linux/WSL only. A file lock prevents duplicates; existing outputs are never resumed
or overwritten. Failures require inspection, not automatic retry.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'output/pose_convergence_queue_v1'


def write_status(state):
    temp=QUEUE/'status.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(QUEUE/'status.json')


def run_stage(command, stage):
    with (QUEUE/f'{stage}.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write_status(dict(status='running',stage=stage,pid=os.getpid(),child_pid=child.pid))
        code=child.wait()
    if code:
        raise RuntimeError(f'{stage} exited {code}; see {QUEUE/stage}.log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    QUEUE.mkdir(parents=True,exist_ok=True)
    if args.launch:
        if (QUEUE/'status.json').exists():
            raise FileExistsError('Queue already initialized; inspect status, do not relaunch')
        # Exclusive log creation also protects two racing launch requests.
        with (QUEUE/'launcher.log').open('xb',buffering=0) as log:
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(launched=True,pid=child.pid,status_file=str(QUEUE/'status.json'))))
        return
    with (QUEUE/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (QUEUE/'status.json').exists():
            raise FileExistsError('Queue already attempted; no automatic retry')
        write_status(dict(status='preflight',pid=os.getpid()))
        try:
            from run_depth_pose_ablation import build_manifest
            configs=[]
            for side in (256,512):
                name=f'pose_convergence_{side}_15000_v1'
                if (ROOT/'output'/name).exists():
                    raise FileExistsError(f'Output already exists: {name}')
                config=json.loads((ROOT/'configs/experiments'/f'{name}.json').read_text())
                assert config['steps']==15000 and config['seeds']==[17] and config['modes']==['off']
                assert config['long_side']==side
                assert config['lr_schedule']==dict(type='hold_cosine',hold_steps=5000,min_learning_rate=1e-5)
                old_name='depth_pose_baseline_5000_v2' if side==256 else 'depth_pose_resolution_512_v1'
                old=json.loads((ROOT/'output'/old_name/'manifest.json').read_text())
                excluded={'purpose','notes','steps','lr_schedule'}
                assert {k:v for k,v in config.items() if k not in excluded}=={k:v for k,v in old['config'].items() if k not in excluded}
                current=build_manifest(config)
                for key in ('records','source_plan_sha256','grouping_sha256'):
                    assert current[key]==old[key],key
                configs.append(config)
            assert {k:v for k,v in configs[0].items() if k!='long_side'}=={k:v for k,v in configs[1].items() if k!='long_side'}
            # One GPU worker at a time, including independent reload validation.
            for side in (256,512):
                name=f'pose_convergence_{side}_15000_v1'
                output=f'output/{name}'
                run_stage(['tools/run_depth_pose_ablation.py','--config',f'configs/experiments/{name}.json','--output',output],f'train_{side}')
                run_stage(['tools/verify_depth_pose_ablation.py','--output',output,
                    '--visual-frame','40','--visual-frame','98','--visual-frame','246','--visual-frame','393'],f'verify_{side}')
            run_stage(['tools/compare_pose_resolution.py','--convergence'],'compare')
            write_status(dict(status='complete_awaiting_visual_review',pid=os.getpid()))
        except Exception as error:
            write_status(dict(status='failed',pid=os.getpid(),error=str(error)))
            raise


if __name__=='__main__': main()
