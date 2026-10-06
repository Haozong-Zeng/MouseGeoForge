"""Exact V2 combination schedule with one added coarse-body supervision term."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import train_cylinder_view_balance as base
from run_depth_pose_ablation import ROOT,fingerprint,write_json
from mousegeoforge.coarse_pose_supervision import coarse_aux_pose_loss

PARENT=ROOT/'output/cylinder_combinations_v2'
OUTPUT=ROOT/'output/cylinder_coarse_aux_v1'
ARM='coarse_body_aux'
AUX_WEIGHT=2.5
LOSS_HISTORY=[]


def config():
    cfg=json.loads((PARENT/'protocol.json').read_text())['config']
    cfg.update(arms=[ARM],arm_learning_rates={ARM:3e-6},coarse_body_aux_weight=AUX_WEIGHT,
        coarse_policy='Same final-coordinate Hungarian matches; valid body16 xy including labeled occluded points; tail7:14 excluded; beta=.01')
    return cfg


def build_protocol(cfg):
    old=json.loads((PARENT/'protocol.json').read_text())
    if not json.loads((PARENT/'delivery_verification.json').read_text())['passed']:raise ValueError('Parent not verified')
    if cfg!=config():raise ValueError('Unexpected config')
    for p,h in old['code_sha256'].items():
        if fingerprint(ROOT/p)!=h:raise ValueError('Locked code changed: '+p)
    for scene,digest in {r['scene']:r['source_revision'] for r in old['records'].values()}.items():
        if base.scene_digest(ROOT/scene)!=digest:raise ValueError('Source changed: '+scene)
    if fingerprint(ROOT/cfg['checkpoint'])!=old['checkpoint_sha256']:raise ValueError('Initial weights changed')
    if fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')!=old['encoder_sha256']:raise ValueError('Encoder changed')
    if fingerprint(ROOT/cfg['manifest'])!=old['manifest_sha256']:raise ValueError('Manifest changed')
    spec=copy.deepcopy(old);spec['config']=cfg;spec['arms']={ARM:old['arms']['combination_coverage']}
    spec['parent_protocol_sha256']=fingerprint(PARENT/'protocol.json')
    spec['parent_results_sha256']=fingerprint(PARENT/'results.json')
    spec['parent_candidate_sha256']=fingerprint(PARENT/'combination_coverage/final_heads.pt')
    spec['scope']='Synthetic reused development; same V2 slots, initialization and trainable scope; added coarse-body auxiliary loss only'
    for p in ('mousegeoforge/coarse_pose_supervision.py','tools/train_cylinder_coarse_aux.py'):
        spec['code_sha256'][p]=fingerprint(ROOT/p)
    spec['loss_policy']='Final xy smooth-L1 weight5 unchanged + coarse body16 smooth-L1 weight2.5; one shared original final-xy Hungarian match; global gradient clip unchanged'
    return spec


def loss(predictions,targets,weights,**kwargs):
    total,logs=coarse_aux_pose_loss(predictions,targets,weights,aux_weight=AUX_WEIGHT,**kwargs)
    LOSS_HISTORY.append(dict(step=len(LOSS_HISTORY)+1,base_pose_loss=float(logs['loss_instance_base_pose'].detach()),
        coarse_body_loss=float(logs['loss_instance_coarse_body'].detach()),aux_weight=AUX_WEIGHT,
        supervised_body_points=logs['coarse_body_points'],combined_pose_loss=float(total.detach())))
    if len(LOSS_HISTORY)%25==0:write_json(OUTPUT/ARM/'coarse_loss_history.json',LOSS_HISTORY)
    return total,logs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--launch',action='store_true');args=p.parse_args()
    if args.launch:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid)));return
    base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
    base.engine.evaluate_scene=base.physical_metrics;base.engine.instance_pose_loss=loss
    base.engine.run(OUTPUT,config(),args.prepare_only)


if __name__=='__main__':main()
