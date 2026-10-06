"""Fixed-budget conditional temperature arm, isolated from locked old trainers."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import train_cylinder_view_balance as base
from run_depth_pose_ablation import ROOT,torch,DepthPoseModel,tensor_state,state_hash,fingerprint,write_json
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive,original_state_projection

PARENT=ROOT/'output/cylinder_source_mix_v1'
PREFLIGHT=ROOT/'output/cylinder_adaptive_temperature_preflight_v1_repeatable'
OUTPUT=ROOT/'output/cylinder_adaptive_temperature_v1'
ARM='adaptive_temperature'
MODEL=None
EVAL_CALLS=0
HISTORY=[]
ADAMW=torch.optim.AdamW
ORIGINAL_LOSS=base.engine.instance_pose_loss


class TemperatureModel(DepthPoseModel):
    @classmethod
    def from_spatial_model(cls,spatial,**kwargs):
        global MODEL
        MODEL=install_adaptive_temperature(super().from_spatial_model(spatial,**kwargs))
        return MODEL


def optimizer(params,*,lr,**kwargs):
    params=list(params);special=list(MODEL.instance_pose_head.adaptive_temperature.parameters());ids={id(p) for p in special}
    groups=[dict(params=[p for p in params if id(p) not in ids],lr=lr),dict(params=special,lr=100*lr)]
    assert sum(len(g['params']) for g in groups)==len(params)
    return ADAMW(groups,lr=lr,**kwargs)


def config():
    cfg=json.loads((PARENT/'protocol.json').read_text())['config']
    cfg.update(arms=[ARM],arm_learning_rates={ARM:3e-6},temperature_learning_rate=3e-4,
               temperature_range=[.5,8.],temperature_input='L2 normalized keypoint token',new_parameters=257)
    return cfg


def build_protocol(cfg):
    old=json.loads((PARENT/'protocol.json').read_text());pre=json.loads((PREFLIGHT/'protocol.json').read_text())
    report=json.loads((PREFLIGHT/'report.json').read_text())
    assert report['passed'] and report['training_supported']
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    assert cfg==config() and pre['parent_protocol_sha256']==fingerprint(PARENT/'protocol.json')
    for p,h in {**old['code_sha256'],**pre['code_sha256']}.items(): assert fingerprint(ROOT/p)==h
    for p,h in old['source_sha256'].items(): assert base.scene_digest(ROOT/p)==h
    assert fingerprint(ROOT/cfg['checkpoint'])==old['checkpoint_sha256']
    assert fingerprint(ROOT/cfg['manifest'])==old['manifest_sha256']
    assert fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')==old['encoder_sha256']
    spec=copy.deepcopy(old);spec['config']=cfg;spec['arms']={ARM:old['arms']['source_mix']}
    spec.update(parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),parent_results_sha256=fingerprint(PARENT/'results.json'),
        preflight_protocol_sha256=fingerprint(PREFLIGHT/'protocol.json'),preflight_report_sha256=fingerprint(PREFLIGHT/'report.json'))
    spec['code_sha256'].update(pre['code_sha256'])
    spec['code_sha256']['tools/train_cylinder_adaptive_temperature.py']=fingerprint(ROOT/'tools/train_cylinder_adaptive_temperature.py')
    spec['scope']='Same 800 source_mix steps and original tensors plus 257 zero temperature parameters. New branch lr3e-4, original lr3e-6. No changed loss/data/encoder/window. Reused synthetic development only.'
    spec['acceptance_policy']='Same 17 gates and 88 records; multi-bottom mean NME >=5% relative improvement versus same-budget OFF. No production promotion. Final method decision may follow predeclared convergence-based paired budget extension, maximum 3200 cumulative steps.'
    return spec


def loss(*args,**kwargs):
    result=ORIGINAL_LOSS(*args,**kwargs)
    scale=MODEL.instance_pose_head.last_temperature_scale
    HISTORY.append(dict(step=len(HISTORY)+1,scale_mean=float(scale.mean()),scale_min=float(scale.min()),scale_max=float(scale.max()),
                        scale_std=float(scale.std()),pose_loss=float(result[0].detach())))
    if len(HISTORY)%25==0:write_json(OUTPUT/ARM/'temperature_history.json',HISTORY)
    return result


def checked_metrics(pred,batch,target):
    global EVAL_CALLS
    metric,assignment=base.physical_metrics(pred,batch,target)
    index=EVAL_CALLS%88
    if EVAL_CALLS<88:
        saved=torch.load(PARENT/'initial'/f'prediction_{index:03d}.pt',weights_only=True,map_location='cpu')
        for k,v in saved['predictions'].items():torch.testing.assert_close(pred[k].cpu(),v,atol=0,rtol=0)
        assert (MODEL.instance_pose_head.last_temperature_scale==1).all()
        if EVAL_CALLS==87:
            old=torch.load(ROOT/config()['checkpoint'],weights_only=True,map_location='cpu')['model']
            assert state_hash(original_state_projection(tensor_state(MODEL)))==state_hash(old)
            write_json(OUTPUT/'baseline_pretrain_verification.json',dict(passed=True,records=88,prediction_max_difference=0,original_parameter_projection_equal=True,extra_zero_parameters=257))
    else:
        path=OUTPUT/ARM/f'temperature_{index:03d}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(scale=MODEL.instance_pose_head.last_temperature_scale.cpu(),oracle_assignment=assignment),path)
    EVAL_CALLS+=1
    return metric,assignment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        assert json.loads((OUTPUT/'status.json').read_text())['state']=='prepared'
        with (OUTPUT/'run.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=process.pid)));return
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
        base.engine.DepthPoseModel=TemperatureModel;base.engine.restore=restore_adaptive
        base.engine.torch.optim.AdamW=optimizer;base.engine.instance_pose_loss=loss;base.engine.evaluate_scene=checked_metrics
        base.engine.run(OUTPUT,config(),args.prepare_only)


if __name__=='__main__':main()
