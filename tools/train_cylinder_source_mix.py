"""Bounded source-mixture control using the verified frozen preflight schedule."""
import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys

import train_cylinder_view_balance as base
from run_depth_pose_ablation import ROOT, fingerprint, write_json, torch

PARENT = ROOT/'output/cylinder_combinations_v2'
PREFLIGHT = ROOT/'output/cylinder_source_mix_preflight_v1'
OUTPUT = ROOT/'output/cylinder_source_mix_v1'
ARM = 'source_mix'
BASELINE_CALLS = 0


def config():
    cfg = json.loads((PARENT/'protocol.json').read_text())['config']
    cfg.update(arms=[ARM], arm_learning_rates={ARM:3e-6},
        source_mix_policy='Replace eight ClimbWall poses with eight locked-train non-replay Cylinder Sample10 poses; exact physical-role and V/H/W slots, unchanged replay')
    return cfg


def validate_sequence(sequence, old):
    original = [old['records'][x[0]] for x in old['arms']['combination_coverage']]
    if len(sequence)!=800 or len(original)!=800:raise ValueError('Expected 800 slots')
    dev = [old['records'][e['key']] for e in old['evaluation']]
    heldout = {r['scene'] for r in dev}; heldout_sources={r['source_group'] for r in dev}
    images=tokens=changed=0
    for new,before,slot in zip(sequence,original,old['slots']):
        if new['split']!='train' or new['scene'] in heldout or new['source_group'] in heldout_sources:
            raise ValueError('Training/development collision')
        for k in ('image_hw','physical_roles'):
            if new[k]!=before[k]:raise ValueError('Slot mismatch: '+k)
        if len(new['views'])!=len(before['views']) or new['views']!=new['diagnostic_view_order']:
            raise ValueError('V/order mismatch')
        if slot['kind']!='cylinder' and new!=before:raise ValueError('Replay changed')
        if new['scene']==before['scene'] and new!=before:raise ValueError('Unchanged source altered')
        changed+=new['scene']!=before['scene']
        images+=len(new['views']); h,w=new['image_hw'];tokens+=len(new['views'])*(h//16)*(w//16)
        if not new['normalization_scale']>0:raise ValueError('Invalid fixed scale')
    if (changed,images,tokens,len({r['scene'] for r in sequence}))!=(200,1800,3540384,40):
        raise ValueError('Source/budget mismatch')
    return dict(changed_slots=changed,images=images,patch_tokens=tokens,unique_train_poses=40,replay_unchanged=True)


def build_protocol(cfg):
    old=json.loads((PARENT/'protocol.json').read_text())
    pre=json.loads((PREFLIGHT/'protocol.json').read_text())
    if cfg!=config():raise ValueError('Unexpected configuration')
    for folder,file in ((PARENT,'delivery_verification.json'),(PREFLIGHT,'report.json')):
        if not json.loads((folder/file).read_text())['passed']:raise ValueError('Missing verified prerequisite')
    if pre['parent_protocol_sha256']!=fingerprint(PARENT/'protocol.json'):raise ValueError('Parent changed')
    for p,h in {**old['code_sha256'],**pre['code_sha256']}.items():
        if fingerprint(ROOT/p)!=h:raise ValueError('Locked code changed: '+p)
    for p,h in pre['source_sha256'].items():
        if base.scene_digest(ROOT/p)!=h:raise ValueError('Source changed: '+p)
    for path,h in ((ROOT/cfg['checkpoint'],old['checkpoint_sha256']),
                   (ROOT/'vggt-omega/vggt_omega_1b_512.pt',old['encoder_sha256']),
                   (ROOT/cfg['manifest'],old['manifest_sha256'])):
        if fingerprint(path)!=h:raise ValueError('Weights/manifest changed')
    if pre['evaluation']!=old['evaluation']:raise ValueError('Evaluation changed')
    budget=validate_sequence(pre['training_sequence'],old)
    spec=copy.deepcopy(old);spec['config']=cfg;spec['arms']={ARM:[]}
    for rec in pre['training_sequence']:
        key=hashlib.sha256(json.dumps(rec,sort_keys=True).encode()).hexdigest()
        spec['records'][key]=rec;spec['arms'][ARM].append([key])
    spec['source_mix_budget']=budget
    spec['parent_protocol_sha256']=fingerprint(PARENT/'protocol.json')
    spec['parent_results_sha256']=fingerprint(PARENT/'results.json')
    spec['preflight_protocol_sha256']=fingerprint(PREFLIGHT/'protocol.json')
    spec['preflight_report_sha256']=fingerprint(PREFLIGHT/'report.json')
    spec['source_sha256']=pre['source_sha256']
    spec['code_sha256'].update(pre['code_sha256'])
    spec['code_sha256']['tools/train_cylinder_source_mix.py']=fingerprint(ROOT/'tools/train_cylinder_source_mix.py')
    spec['scope']=pre['scope']
    # The inherited V2 camera coverage described the old source pool, not this one.
    for key in ('sampling_coverage','previous_sampling_coverage'):spec.pop(key,None)
    spec['acceptance_policy']='Original 88 development records and 17 gates; multi-bottom mean NME relative improvement >=5% versus V2 defines substantive progress, with retention failures explicitly reported; no promotion'
    return spec


def checked_metrics(pred,batch,target):
    global BASELINE_CALLS
    metrics,assignment=base.physical_metrics(pred,batch,target)
    if BASELINE_CALLS<88:
        saved=torch.load(PARENT/'initial'/f'prediction_{BASELINE_CALLS:03d}.pt',weights_only=True,map_location='cpu')
        for key,value in saved['metrics'].items():
            if value is None:
                if metrics[key] is not None:raise ValueError('Initial missing metric mismatch')
            elif abs(metrics[key]-value)>1e-6:raise ValueError('Initial metric mismatch: '+key)
        for key,value in saved['predictions'].items():
            torch.testing.assert_close(pred[key].detach().cpu(),value,atol=0,rtol=0)
        BASELINE_CALLS+=1
        if BASELINE_CALLS==88:
            write_json(OUTPUT/'baseline_pretrain_verification.json',dict(passed=True,records=88,prediction_max_difference=0))
    return metrics,assignment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        if not (OUTPUT/'status.json').exists() or json.loads((OUTPUT/'status.json').read_text())['state']!='prepared':
            raise RuntimeError('Must prepare and inspect first; never restart existing run')
        with (OUTPUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(OUTPUT/'run.log'))));return
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
        base.engine.evaluate_scene=checked_metrics
        base.engine.run(OUTPUT,config(),a.prepare_only)


if __name__=='__main__':main()
