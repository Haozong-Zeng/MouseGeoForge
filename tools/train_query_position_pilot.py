"""Same-data head-only query-position OFF/ON pilot, independent locked outputs."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
import train_dual_pose_diversity as diagnostics
import train_cylinder_view_balance as base
from train_cylinder_adaptive_temperature import TemperatureModel
from run_depth_pose_ablation import ROOT,torch,tensor_state,state_hash,write_json,fingerprint
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.query_position import install_query_position,query_position_context

OUT=ROOT/'output/query_position_seed17_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'
CHECKPOINT=PARENT/'adaptive_temperature/final_heads.pt'
PREFLIGHT=ROOT/'output/query_position_preflight_v2'
ARMS=('query_position_off','query_position_on')
ADAMW=torch.optim.AdamW
CLIP=torch.nn.utils.clip_grad_norm_
MODEL=None
SELECTED=[]
OPTIMIZERS=0
EVAL_CALLS=0
DEPTH_HASHES=[]
FROZEN_HASH=None


class QueryPositionModel(TemperatureModel):
    @classmethod
    def from_spatial_model(cls,spatial,**kwargs):
        global MODEL
        model=install_query_position(super().from_spatial_model(spatial,**kwargs))
        model.query_position_enabled=False;MODEL=model
        return model

    def forward(self,*args,**kwargs):
        with query_position_context(self,enabled=self.query_position_enabled):
            return super().forward(*args,**kwargs)

    def forward_cached(self,*args,**kwargs):
        with query_position_context(self,enabled=self.query_position_enabled):
            return super().forward_cached(*args,**kwargs)


def restore(model,state):
    merged=dict(state);key='instance_pose_head.query_position.offsets'
    if key not in merged:merged[key]=torch.zeros_like(model.state_dict()[key])
    restore_adaptive(model,merged)


class HeadAdamW(ADAMW):
    def zero_grad(self,set_to_none=True):
        # Autograd flags remain unchanged for dispatch parity. Clear also the
        # gradients of parameters not in this optimizer, without updating them.
        MODEL.zero_grad(set_to_none=set_to_none)


def optimizer(params,*,lr,**kwargs):
    global SELECTED,OPTIMIZERS
    all_requested=list(params);requested={id(p) for p in all_requested}
    enabled=OPTIMIZERS==1;assert OPTIMIZERS<2
    MODEL.query_position_enabled=enabled
    common=[];temperature=[];position=[]
    for n,p in MODEL.named_parameters():
        if not n.startswith('instance_pose_head.'):continue
        assert id(p) in requested
        if n.startswith('instance_pose_head.query_position.'):
            if enabled:position.append(p)
        elif n.startswith('instance_pose_head.adaptive_temperature.'):temperature.append(p)
        else:common.append(p)
    groups=[dict(params=common,lr=lr),dict(params=temperature,lr=3e-4)]
    if enabled:groups.append(dict(params=position,lr=3e-4))
    SELECTED=[p for g in groups for p in g['params']]
    ids={id(p) for p in SELECTED}
    assert len(ids)==len(SELECTED)
    assert all(n.startswith('instance_pose_head.') for n,p in MODEL.named_parameters() if id(p) in ids)
    write_json(OUT/f'{ARMS[OPTIMIZERS]}_optimizer_scope.json',dict(enabled=enabled,
        parameter_names=[n for n,p in MODEL.named_parameters() if id(p) in ids],
        parameter_count=sum(p.numel() for p in SELECTED),groups=[dict(lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups],
        unselected_parameters_may_have_gradients=True,clip_selected_only=True))
    OPTIMIZERS+=1
    return HeadAdamW(groups,lr=lr,**kwargs)


def selected_clip(params,max_norm,**kwargs):
    return CLIP(SELECTED,max_norm,**kwargs)


def config():
    cfg=copy.deepcopy(json.loads((PARENT/'protocol.json').read_text())['config'])
    cfg.update(arms=list(ARMS),arm_learning_rates={a:3e-6 for a in ARMS},steps=800,
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),max_run_hours=1.,
        trainable_scope='Both arms same instance_pose_head only; shared adapter and dense depth excluded from optimizer; ON additionally3072 zero query positions',
        checkpoint_policy='Fixed final800 per arm, fresh optimizer; no automatic extension or best-dev selection')
    return cfg


def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already prepared')
    report=json.loads((PREFLIGHT/'report.json').read_text())
    assert report['passed'] and report['zero_on_all_outputs_max_difference']==0
    assert report['common_gradients_relative_l2']<=report['gradient_relative_l2_limit']
    spec=copy.deepcopy(json.loads((PARENT/'protocol.json').read_text()))
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in spec['source_sha256'].items():assert base.scene_digest(ROOT/p)==h
    for p,h in report['code_sha256'].items():assert fingerprint(ROOT/p)==h
    seq=spec['arms']['adaptive_temperature'];assert len(seq)==800
    spec['config']=config();spec['arms']={a:copy.deepcopy(seq) for a in ARMS}
    spec['checkpoint_sha256']=fingerprint(CHECKPOINT)
    spec['preflight_sha256']=fingerprint(PREFLIGHT/'report.json')
    spec['code_sha256'].update({p:fingerprint(ROOT/p) for p in ('tools/train_query_position_pilot.py','tools/train_dual_pose_diversity.py',
        'mousegeoforge/query_position.py','vggt-omega/vggt_omega/models/heads/instance_pose_head.py')})
    spec['scope']='Second new training mechanism of ten-hour cycle: per-layer query-position embeddings only; paired common head-only scope. No spatial anchors, no persistent IDs, no masks/GT at inference.'
    spec['acceptance_policy']='Both800 fixed. Same historical88/17 gates. Seed29 only if ON all17 pass, dual detection-aware PCK gain>=.01, dual box-matched NME<=1.02xOFF, TP not lower, FP not higher, duplicate-mask view count not higher. Report frame296 separately. Failure ends new training for this cycle after prior diversity failure; no third mechanism or automatic extension.'
    spec['freeze_policy']='Original requires_grad flags preserved; model.zero_grad every step; optimizer and clipping selected head parameters only. Verify non-head parameters and all88 depth/depth_conf outputs exactly unchanged. Encoder gradients absent.'
    spec['gradient_preflight_note']='Original bitwise-backward preflight remains failed because OFF repeats are also nondeterministic; independent qualified v2 kept forward tolerance zero, allowed relative gradient L2<=1e-6. Actual v2 difference3.91e-8.'
    spec['production_promoted']=False
    write_json(OUT/'protocol.json',spec);write_json(OUT/'status.json',dict(state='prepared',total_steps=1600))


def build_protocol(cfg):
    spec=json.loads((OUT/'protocol.json').read_text());assert cfg==spec['config']==config()
    assert fingerprint(CHECKPOINT)==spec['checkpoint_sha256']
    assert fingerprint(PREFLIGHT/'report.json')==spec['preflight_sha256']
    assert fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')==spec['encoder_sha256']
    assert fingerprint(ROOT/cfg['manifest'])==spec['manifest_sha256']
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in spec['source_sha256'].items():assert base.scene_digest(ROOT/p)==h
    return spec


def checked_metrics(pred,batch,target):
    global EVAL_CALLS,FROZEN_HASH
    current=state_hash({k:v.detach().cpu() for k,v in pred.items() if k in ('depth','depth_conf')})
    if EVAL_CALLS<88:DEPTH_HASHES.append(current)
    else:assert current==DEPTH_HASHES[EVAL_CALLS%88]
    if EVAL_CALLS%88==0:
        frozen=state_hash({k:v for k,v in tensor_state(MODEL).items() if not k.startswith('instance_pose_head.')})
        if FROZEN_HASH is None:FROZEN_HASH=frozen
        else:assert frozen==FROZEN_HASH
    value=diagnostics.checked_metrics(pred,batch,target)
    EVAL_CALLS+=1
    if EVAL_CALLS%88==0:write_json(OUT/'depth_freeze_verification.partial.json',dict(evaluations=EVAL_CALLS,exact_depth_outputs=True,non_head_state_unchanged=True))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare();return
    if a.launch:
        assert json.loads((OUT/'status.json').read_text())['state']=='prepared'
        with (OUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,output=str(OUT))));return
    if time.time()+3600>=diagnostics.DEADLINE:raise RuntimeError('Insufficient cycle time')
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        diagnostics.OUT=OUT;diagnostics.ARMS=ARMS
        base.engine.build_protocol=build_protocol;base.engine.load_batch=base.oriented_batch
        base.engine.DepthPoseModel=QueryPositionModel;base.engine.restore=restore
        base.engine.torch.optim.AdamW=optimizer;base.engine.torch.nn.utils.clip_grad_norm_=selected_clip
        base.engine.evaluate_scene=checked_metrics;base.engine.instance_pose_loss=diagnostics.checked_loss
        base.engine.run(OUT,config(),False)
        build_protocol(config());assert EVAL_CALLS==264 and OPTIMIZERS==2
        assert all(len(x.multihead_attn._forward_pre_hooks)==0 for x in MODEL.instance_pose_head.instance_decoder.layers)
        write_json(OUT/'verification.json',dict(passed=True,evaluations=264,source_weights_code_unchanged=True,
            initial_prediction_max_difference=0,depth_output_max_difference=0,non_head_state_unchanged=True,
            optimizer_steps_per_arm=800,encoder_gradient_free=True,hooks_cleaned=True,
            protocol_sha256=fingerprint(OUT/'protocol.json'),production_promoted=False))


if __name__=='__main__':main()
