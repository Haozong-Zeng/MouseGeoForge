"""Bounded diverse400 768 queue: audit, train, reload, compare. No retries."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import run_pose_convergence_queue as queue
from pose_diverse_resolution import HIGH, PREFLIGHT

ROOT=queue.ROOT
queue.QUEUE=ROOT/'output/pose_diverse_resolution_768_queue_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    queue.QUEUE.mkdir(parents=True,exist_ok=True)
    if args.launch:
        if (queue.QUEUE/'status.json').exists(): raise FileExistsError('Queue already attempted')
        with (queue.QUEUE/'launcher.log').open('xb',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,launched=True,status_file=str(queue.QUEUE/'status.json'))))
        return
    with (queue.QUEUE/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (queue.QUEUE/'status.json').exists(): raise FileExistsError('No automatic retry')
        queue.write_status(dict(status='preflight',pid=os.getpid()))
        try:
            if (ROOT/'output'/HIGH).exists(): raise FileExistsError('Output exists')
            cfg=json.loads((ROOT/'configs/experiments'/f'{HIGH}.json').read_text())
            assert cfg['steps']==15000 and cfg['seeds']==[17] and cfg['modes']==['off'] and cfg['long_side']==768
            queue.run_stage(['tools/pose_diverse_resolution.py','audit'],'audit')
            geometry=json.loads((PREFLIGHT/'geometry_audit.json').read_text())
            assert geometry['passed'] and geometry['compared_scenes']==432
            queue.run_stage(['tools/run_depth_pose_ablation.py','--config',f'configs/experiments/{HIGH}.json',
                '--output',f'output/{HIGH}'],'train')
            queue.run_stage(['tools/verify_depth_pose_ablation.py','--output',f'output/{HIGH}',
                '--visual-frame','40','--visual-frame','98','--visual-frame','246','--visual-frame','393'],'verify')
            queue.run_stage(['tools/pose_diverse_resolution.py','compare'],'compare')
            queue.write_status(dict(status='complete_awaiting_visual_review',pid=os.getpid()))
        except Exception as error:
            queue.write_status(dict(status='failed',pid=os.getpid(),error=str(error))); raise


if __name__=='__main__': main()
