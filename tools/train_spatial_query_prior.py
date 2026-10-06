"""Paired head-only soft spatial attention prior, fixed1600 steps per arm."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
import train_query_position_pilot as pilot
from run_depth_pose_ablation import ROOT, torch, tensor_state, state_hash, fingerprint, write_json
from train_cylinder_adaptive_temperature import TemperatureModel
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.spatial_query_prior import install_spatial_query_prior, spatial_query_context

OUT=ROOT/'output/spatial_query_prior_seed17_v1'
PRIOR=ROOT/'output/query_position_seed17_v1'
PREFLIGHT=ROOT/'output/spatial_query_prior_preflight_v3'
ARMS=('spatial_prior_off','spatial_prior_on')
INITIAL_EXTRAS=None
PREFIX='instance_pose_head.spatial_query_prior.'


class SpatialModel(TemperatureModel):
    @classmethod
    def from_spatial_model(cls, spatial, **kwargs):
        global INITIAL_EXTRAS
        model=install_spatial_query_prior(super().from_spatial_model(spatial,**kwargs))
        INITIAL_EXTRAS={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(PREFIX)}
        model.spatial_prior_enabled=False;pilot.MODEL=model
        return model

    def forward(self,*args,**kwargs):
        with spatial_query_context(self,enabled=self.spatial_prior_enabled):
            return super().forward(*args,**kwargs)

    def forward_cached(self,*args,**kwargs):
        with spatial_query_context(self,enabled=self.spatial_prior_enabled):
            return super().forward_cached(*args,**kwargs)


def restore(model,state):
    extras={k for k in state if k.startswith(PREFIX)}
    if extras and extras!=set(INITIAL_EXTRAS):raise ValueError('Partial spatial state')
    restore_adaptive(model,dict(state) if extras else {**state,**INITIAL_EXTRAS})


def config():
    cfg=copy.deepcopy(json.loads((PRIOR/'protocol.json').read_text())['config'])
    cfg.update(arms=list(ARMS),arm_learning_rates={a:3e-6 for a in ARMS},steps=1600,max_run_hours=1.,
               trainable_scope='Same head-only parameters both arms; ON adds2108 spatial-prior parameters lr3e-4. No query-position vectors.',
               checkpoint_policy='Fresh original temperature warmstart, uninterrupted1600 two800 cycles. Save800/1600 optimizer state; fixed endpoint, no best-dev selection.')
    return cfg


def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already prepared')
    report=json.loads((PREFLIGHT/'report.json').read_text());assert report['passed']
    assert report['original_saved_prediction_max_difference']==report['depth_output_max_difference']==0
    spec=copy.deepcopy(json.loads((PRIOR/'protocol.json').read_text()))
    for p,h in {**spec['code_sha256'],**report['code_sha256']}.items():assert fingerprint(ROOT/p)==h
    seq=spec['arms']['query_position_off'];assert seq==spec['arms']['query_position_on'] and len(seq)==800
    assert spec['schedule']==list(range(800)) and all(len(x)==1 for x in seq)
    spec['arms']={a:copy.deepcopy(seq) for a in ARMS};spec['schedule']*=2;spec['slots']*=2
    spec['config']=config();spec['scope']='Independent SMCA-inspired soft spatial-prior mechanism, not fullSMCA/DAB; same data/matcher/loss/temperature and head-only optimizer both arms.'
    spec['preflight_sha256']=fingerprint(PREFLIGHT/'report.json')
    spec['parent800_protocol_sha256']=fingerprint(PRIOR/'protocol.json')
    spec['code_sha256'].update(report['code_sha256'])
    spec['code_sha256']['tools/train_spatial_query_prior.py']=fingerprint(ROOT/'tools/train_spatial_query_prior.py')
    spec['acceptance_policy']='Fixed1600 each. Original17 gates; dualPCK ON-OFF>=.01; dualNME<=1.02xOFF; dualTP not lower; all-viewFP and duplicate mask views not higher. Report initialON preflight separately, fixed296 and body/tail. Second seed only with practical retention benefit. Negative result prompts architecture/training diagnosis, not automatic budget doubling or globalstop.'
    spec['budget_policy']=dict(steps_per_arm=1600,images_per_arm=3600,patch_tokens_per_arm=7080768,new_train_records=0)
    spec['initialization_note']='ON nonzero soft prior from start; OFF exact original. Per-parameter lr for new2108 is3e-4; no zeroON equivalence claim. No GT-based anchor construction.'
    write_json(OUT/'protocol.json',spec);write_json(OUT/'status.json',dict(state='prepared',total_steps=3200))


def build_protocol(cfg):
    spec=json.loads((OUT/'protocol.json').read_text());assert cfg==config()==spec['config']
    assert fingerprint(PRIOR/'protocol.json')==spec['parent800_protocol_sha256']
    assert fingerprint(PREFLIGHT/'report.json')==spec['preflight_sha256']
    assert fingerprint(pilot.CHECKPOINT)==spec['checkpoint_sha256']
    assert fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')==spec['encoder_sha256']
    assert fingerprint(ROOT/cfg['manifest'])==spec['manifest_sha256']
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in spec['source_sha256'].items():assert pilot.base.scene_digest(ROOT/p)==h
    return spec


def optimizer(params,*,lr,**kwargs):
    requested={id(p) for p in params};enabled=pilot.OPTIMIZERS==1
    assert pilot.OPTIMIZERS<2
    model=pilot.MODEL;model.spatial_prior_enabled=enabled
    common=[];temperature=[];spatial=[]
    for name,param in model.named_parameters():
        if not name.startswith('instance_pose_head.'):continue
        assert id(param) in requested
        if name.startswith(PREFIX):
            if enabled:spatial.append(param)
        elif name.startswith('instance_pose_head.adaptive_temperature.'):temperature.append(param)
        else:common.append(param)
    groups=[dict(params=common,lr=lr),dict(params=temperature,lr=3e-4)]
    if enabled:groups.append(dict(params=spatial,lr=3e-4))
    pilot.SELECTED=[p for group in groups for p in group['params']]
    ids={id(p) for p in pilot.SELECTED};assert len(ids)==len(pilot.SELECTED)
    arm=ARMS[pilot.OPTIMIZERS];pilot.OPTIMIZERS+=1
    write_json(OUT/f'{arm}_optimizer_scope.json',dict(enabled=enabled,
        parameter_names=[n for n,p in model.named_parameters() if id(p) in ids],
        groups=[dict(lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups],
        unselected_parameters_may_have_gradients=True))
    opt=pilot.HeadAdamW(groups,lr=lr,**kwargs);base_step=opt.step;count=0
    def step(*a,**kw):
        nonlocal count
        result=base_step(*a,**kw);count+=1
        if count in (800,1600):
            state=tensor_state(model)
            torch.save(dict(model=state,optimizer=opt.state_dict(),steps=count,arm=arm,
                model_sha256=state_hash(state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                protocol_sha256=fingerprint(OUT/'protocol.json')),OUT/arm/f'training_state_{count:04d}.pt')
        return result
    opt.step=step
    return opt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    if args.launch:
        assert json.loads((OUT/'status.json').read_text())['state']=='prepared'
        with (OUT/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,output=str(OUT))));return
    if time.time()+3600>=pilot.diagnostics.DEADLINE:raise RuntimeError('Insufficient cycle time')
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pilot.OUT=OUT;pilot.diagnostics.OUT=OUT;pilot.diagnostics.ARMS=ARMS
        engine=pilot.base.engine;engine.build_protocol=build_protocol;engine.load_batch=pilot.base.oriented_batch
        engine.DepthPoseModel=SpatialModel;engine.restore=restore;engine.torch.optim.AdamW=optimizer
        engine.torch.nn.utils.clip_grad_norm_=pilot.selected_clip;engine.evaluate_scene=pilot.checked_metrics
        engine.instance_pose_loss=pilot.diagnostics.checked_loss
        engine.run(OUT,config(),False);build_protocol(config())
        assert pilot.EVAL_CALLS==264 and pilot.OPTIMIZERS==2
        assert not pilot.MODEL.instance_pose_head._forward_pre_hooks
        assert all(not x.multihead_attn._forward_pre_hooks for x in pilot.MODEL.instance_pose_head.instance_decoder.layers)
        for arm in ARMS:
            history=json.loads((OUT/arm/'history.json').read_text());assert len(history)==1600
            assert history[-1]['images_seen']==3600 and history[-1]['patch_tokens_seen']==7080768
            for n in (800,1600):assert (OUT/arm/f'training_state_{n:04d}.pt').exists()
        write_json(OUT/'verification.json',dict(passed=True,evaluations=264,optimizer_steps_per_arm=1600,
            source_weights_code_unchanged=True,initial_prediction_max_difference=0,depth_output_max_difference=0,
            non_head_state_unchanged=True,encoder_gradient_free=True,hooks_cleaned=True,
            protocol_sha256=fingerprint(OUT/'protocol.json'),production_promoted=False))


if __name__=='__main__':main()
