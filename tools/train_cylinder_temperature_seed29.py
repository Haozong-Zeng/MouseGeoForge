"""Independent training-order seed29, paired OFF/ON, exactly same input multiset."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import random
import subprocess
import sys
import train_cylinder_adaptive_temperature as first
from run_depth_pose_ablation import ROOT,torch,fingerprint,write_json,instance_pose_loss,tensor_state,state_hash
from mousegeoforge.adaptive_temperature import restore_adaptive,original_state_projection

OUTPUT=ROOT/'output/cylinder_temperature_seed29_v1'
ARMS=('temperature_off','temperature_on')
SPEC=None
CURRENT=None
OPTIMIZERS=0
EVAL_CALLS=0
HISTORY={a:[] for a in ARMS}


def config():
    cfg=first.config();cfg.update(seed=29,arms=list(ARMS),arm_learning_rates={a:3e-6 for a in ARMS},
        seed_policy='Fixed original pretrained weights; seed29 changes adaptation input order, not train/development membership or view/input multiset.')
    return cfg


def build_protocol(cfg):
    global SPEC
    assert cfg==config()
    assert json.loads((first.OUTPUT/'delivery_verification.json').read_text())['passed']
    assert json.loads((first.OUTPUT/'acceptance_800.json').read_text())['original_gates']['passed']
    original=first.build_protocol(first.config());sequence=original['arms'][first.ARM]
    order=list(range(800));random.Random(29).shuffle(order)
    assert sorted(order)==list(range(800)) and order!=list(range(800))
    spec=copy.deepcopy(original);spec['config']=cfg;spec['arms']={a:[sequence[i] for i in order] for a in ARMS}
    spec['slots']=[original['slots'][i] for i in order];spec['seed29_permutation']=order
    assert Counter(tuple(x) for x in spec['arms'][ARMS[0]])==Counter(tuple(x) for x in sequence)
    spec['seed17_protocol_sha256']=fingerprint(first.OUTPUT/'protocol.json')
    spec['seed17_acceptance_sha256']=fingerprint(first.OUTPUT/'acceptance_800.json')
    spec['code_sha256']['tools/train_cylinder_temperature_seed29.py']=fingerprint(ROOT/'tools/train_cylinder_temperature_seed29.py')
    spec['scope']='Seed29 paired OFF/ON, fixed original pretraining weights, permuted same800 input slots. Original88 development and17 gates unchanged. No independent dataset test or production promotion.'
    SPEC=spec;return spec


def optimizer(params,*,lr,**kwargs):
    global CURRENT,OPTIMIZERS
    if OPTIMIZERS>=2:raise RuntimeError('Unexpected extra optimizer')
    CURRENT=ARMS[OPTIMIZERS];OPTIMIZERS+=1;model=first.MODEL
    special=list(model.instance_pose_head.adaptive_temperature.parameters());ids={id(p) for p in special}
    enabled=CURRENT=='temperature_on';model.instance_pose_head.adaptive_temperature_enabled=enabled
    model.instance_pose_head.adaptive_temperature.requires_grad_(enabled)
    groups=[dict(params=[p for p in params if id(p) not in ids],lr=lr)]
    if enabled:groups.append(dict(params=special,lr=100*lr))
    return first.ADAMW(groups,lr=lr,**kwargs)


def loss(*args,**kwargs):
    result=instance_pose_loss(*args,**kwargs);scale=first.MODEL.instance_pose_head.last_temperature_scale
    row=dict(step=len(HISTORY[CURRENT])+1,mean=float(scale.mean()),min=float(scale.min()),max=float(scale.max()),pose_loss=float(result[0].detach()))
    if CURRENT=='temperature_off':assert row['min']==row['max']==1
    HISTORY[CURRENT].append(row)
    if len(HISTORY[CURRENT])%25==0:write_json(OUTPUT/CURRENT/'temperature_history.json',HISTORY[CURRENT])
    return result


def checked_metrics(pred,batch,target):
    global EVAL_CALLS
    metric,assignment=first.base.physical_metrics(pred,batch,target);idx=EVAL_CALLS%88
    if EVAL_CALLS<88:
        saved=torch.load(first.PARENT/'initial'/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
        for k,v in saved['predictions'].items():torch.testing.assert_close(pred[k].cpu(),v,atol=0,rtol=0)
        if EVAL_CALLS==87:
            old=torch.load(ROOT/config()['checkpoint'],weights_only=True,map_location='cpu')['model']
            assert state_hash(original_state_projection(tensor_state(first.MODEL)))==state_hash(old)
            write_json(OUTPUT/'baseline_pretrain_verification.json',dict(passed=True,records=88,prediction_max_difference=0,original_parameter_projection_equal=True))
    else:
        entry=SPEC['evaluation'][idx];record=SPEC['records'][entry['key']]
        path=OUTPUT/CURRENT/f'diagnostics_{idx:03d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(record=record,group=entry['group'],oracle_assignment=assignment,metrics=metric,
            predictions={k:v.cpu() for k,v in pred.items() if k.startswith('instance_') and 'mask' not in k},
            targets={k:v.cpu() for k,v in target.items() if k!='masks'},
            temperature_scale=first.MODEL.instance_pose_head.last_temperature_scale.cpu()),path)
    EVAL_CALLS+=1;return metric,assignment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        assert json.loads((OUTPUT/'status.json').read_text())['state']=='prepared'
        with (OUTPUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid)));return
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=first.base.engine;engine.build_protocol=build_protocol;engine.load_batch=first.base.oriented_batch
        engine.DepthPoseModel=first.TemperatureModel;engine.restore=restore_adaptive
        engine.torch.optim.AdamW=optimizer;engine.instance_pose_loss=loss;engine.evaluate_scene=checked_metrics
        engine.run(OUTPUT,config(),args.prepare_only)


if __name__=='__main__':main()
