"""Isolate trainable scope using the exact completed balanced-view schedule.

One new 800-step arm, compared with the preserved all-heads balanced arm. No
production module is changed; only this runner selects a frozen-scope subclass.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import train_cylinder_view_balance as parent
from run_depth_pose_ablation import ROOT, DepthPoseModel, torch, fingerprint

OUTPUT = ROOT/'output/cylinder_pose_only_v1'
PARENT = ROOT/'output/cylinder_view_balance_v1'
CONFIG = ROOT/'configs/experiments/cylinder_pose_only_v1.json'


def freeze_except_pose(model):
    model.requires_grad_(False)
    model.instance_pose_head.requires_grad_(True)
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if not names or any(not name.startswith('instance_pose_head.') for name in names):
        raise AssertionError('Unexpected trainable parameter')
    return names


class PoseOnlyDepthPoseModel(DepthPoseModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        freeze_except_pose(self)

    def train(self,mode=True):
        super().train(mode)
        for name,module in self.named_children():
            if name!='instance_pose_head':
                module.eval()
        return self

    def _decode(self,*args,**kwargs):
        if any(p.requires_grad or p.grad is not None for name,p in self.named_parameters()
               if not name.startswith('instance_pose_head.')):
            raise AssertionError('Frozen parameter acquired a gradient')
        return super()._decode(*args,**kwargs)


def build_protocol(cfg):
    old=json.loads((PARENT/'protocol.json').read_text())
    if json.loads((PARENT/'status.json').read_text())['state']!='completed':
        raise ValueError('Parent must be complete')
    if not json.loads((PARENT/'delivery_verification.json').read_text())['passed']:
        raise ValueError('Parent must have passed delivery verification')
    current=parent.build_protocol(old['config'])
    if current!=old:
        raise ValueError('Parent protocol/source/code changed')
    for key in ('steps','seed','weight_decay','gradient_clip','long_side','manifest','checkpoint','acceptance'):
        if cfg[key]!=old['config'][key]:
            raise ValueError('Isolation control changed '+key)
    if cfg['arms']!=['pose_head_only'] or cfg['arm_learning_rates']['pose_head_only']!=old['config']['arm_learning_rates']['balanced_views']:
        raise ValueError('Unexpected arm/LR')
    spec=copy.deepcopy(old)
    spec['config']=cfg
    spec['arms']={'pose_head_only':old['arms']['balanced_views']}
    spec['scope']='Synthetic Cylinder trainable-scope isolation; exact V1 balanced schedule/initialization; no real data'
    spec['parent_protocol_sha256']=fingerprint(PARENT/'protocol.json')
    spec['parent_results_sha256']=fingerprint(PARENT/'results.json')
    spec['parent_candidate_sha256']=fingerprint(PARENT/'balanced_views/final_heads.pt')
    spec['code_sha256']['tools/train_cylinder_pose_only.py']=fingerprint(Path(__file__))
    spec['trainable_scope']='instance_pose_head.* only; encoder, shared_adapter, dense_head, depth_pose_bridge frozen'
    spec['gradient_note']='Same global clip threshold; changing trainable scope also changes the total gradient norm. This isolates update scope, not only depth-loss sign.'
    return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(OUTPUT/'run.log'))))
        return
    engine=parent.engine
    engine.build_protocol=build_protocol
    engine.load_batch=parent.oriented_batch
    engine.evaluate_scene=parent.physical_metrics
    engine.DepthPoseModel=PoseOnlyDepthPoseModel
    engine.run(OUTPUT,json.loads(CONFIG.read_text()),args.prepare_only)


if __name__=='__main__': main()
