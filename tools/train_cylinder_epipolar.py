"""Same source-mixture schedule, with one preflight-approved epipolar term."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import train_cylinder_view_balance as base
from run_depth_pose_ablation import ROOT,torch,fingerprint,write_json
from mousegeoforge.epipolar_supervision import epipolar_aux_pose_loss

PARENT=ROOT/'output/cylinder_source_mix_v1'
PREFLIGHT=ROOT/'output/cylinder_epipolar_preflight_v1'
OUTPUT=ROOT/'output/cylinder_epipolar_v1'
ARM='epipolar_body'
HISTORY=[]
BASELINE_CALLS=0


def config():
    cfg=json.loads((PARENT/'protocol.json').read_text())['config']
    cfg.update(arms=[ARM],arm_learning_rates={ARM:3e-6},epipolar_aux_weight=1.,
        epipolar_policy='Body16 signed Sampson / image diagonal, smooth-L1 beta .01, pair means then instance means; shared original matching and persistent IDs; mono differentiable zero')
    return cfg


def build_protocol(cfg):
    old=json.loads((PARENT/'protocol.json').read_text());pre=json.loads((PREFLIGHT/'protocol.json').read_text())
    report=json.loads((PREFLIGHT/'report.json').read_text())
    if not report['passed'] or not report['training_supported']:raise ValueError('Preflight does not authorize training')
    if not json.loads((PARENT/'delivery_verification.json').read_text())['passed']:raise ValueError('Unverified parent')
    if cfg!=config() or pre['parent_protocol_sha256']!=fingerprint(PARENT/'protocol.json'):raise ValueError('Changed proposal')
    for p,h in {**old['code_sha256'],**pre['code_sha256']}.items():
        if fingerprint(ROOT/p)!=h:raise ValueError('Locked code changed: '+p)
    for scene,h in old['source_sha256'].items():
        if base.scene_digest(ROOT/scene)!=h:raise ValueError('Source changed: '+scene)
    for p,h in ((ROOT/cfg['checkpoint'],old['checkpoint_sha256']),
                (ROOT/'vggt-omega/vggt_omega_1b_512.pt',old['encoder_sha256']),
                (ROOT/cfg['manifest'],old['manifest_sha256'])):
        if fingerprint(p)!=h:raise ValueError('Weights/manifest changed')
    spec=copy.deepcopy(old);spec['config']=cfg;spec['arms']={ARM:old['arms']['source_mix']}
    spec['parent_protocol_sha256']=fingerprint(PARENT/'protocol.json')
    spec['parent_results_sha256']=fingerprint(PARENT/'results.json')
    spec['preflight_protocol_sha256']=fingerprint(PREFLIGHT/'protocol.json')
    spec['preflight_report_sha256']=fingerprint(PREFLIGHT/'report.json')
    spec['code_sha256'].update(pre['code_sha256'])
    spec['code_sha256']['tools/train_cylinder_epipolar.py']=fingerprint(ROOT/'tools/train_cylinder_epipolar.py')
    spec['scope']='Same frozen source-mixture schedule and original initialization; added body16 epipolar loss only. Synthetic reused development, no real validation or end-to-end association.'
    spec['acceptance_policy']='Original 88 records/17 performance gates unchanged; substantive progress still requires multi-bottom mean NME reduction >=5% versus source_mix, with geometry/common-support and retention/depth tradeoffs reported.'
    return spec


def loss(predictions,targets,weights,**kwargs):
    total,logs=epipolar_aux_pose_loss(predictions,targets,weights,aux_weight=1.,**kwargs)
    HISTORY.append(dict(step=len(HISTORY)+1,base_pose_loss=float(logs['loss_instance_base_pose'].detach()),
        auxiliary_loss=float(logs['loss_instance_epipolar'].detach()),combined_pose_loss=float(total.detach()),
        instances=logs['epipolar_instances'],pairs=logs['epipolar_pairs'],points=logs['epipolar_points']))
    if len(HISTORY)%25==0:write_json(OUTPUT/ARM/'epipolar_loss_history.json',HISTORY)
    return total,logs


def checked_metrics(pred,batch,target):
    global BASELINE_CALLS
    metrics,assignment=base.physical_metrics(pred,batch,target)
    if BASELINE_CALLS<88:
        saved=torch.load(PARENT/'initial'/f'prediction_{BASELINE_CALLS:03d}.pt',weights_only=True,map_location='cpu')
        for k,value in saved['predictions'].items():torch.testing.assert_close(pred[k].detach().cpu(),value,atol=0,rtol=0)
        for k,value in saved['metrics'].items():
            if value is None:assert metrics[k] is None
            else:assert abs(metrics[k]-value)<1e-6,k
        BASELINE_CALLS+=1
        if BASELINE_CALLS==88:write_json(OUTPUT/'baseline_pretrain_verification.json',dict(passed=True,records=88,prediction_max_difference=0))
    return metrics,assignment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if not (OUTPUT/'status.json').exists() or json.loads((OUTPUT/'status.json').read_text())['state']!='prepared':raise RuntimeError('Prepare first, never restart existing run')
        with (OUTPUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid)));return
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
        base.engine.evaluate_scene=checked_metrics;base.engine.instance_pose_loss=loss
        base.engine.run(OUTPUT,config(),args.prepare_only)


if __name__=='__main__':main()
