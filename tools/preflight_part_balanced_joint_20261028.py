"""Validate a cloud binding on real TRAIN batches; no Slurm or formal dispatch.

Factory API: build(config, arm, seed, device) -> dict(model, optimizer,
depth_batch, keypoint_batch, depth_loss, keypoint_loss, reference_depth,
reference_keypoints, identity). References are independent parent forward
callables taking RGB, and MUST load the same release encoder and saved heads.
They cannot simply call the new wrapper. identity contains the five SHA256
fields required by save_joint_checkpoint.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import torch
from mousegeoforge.joint_training import joint_train_step, validate_identity


def check_design(cfg):
    assert cfg['schema']=='part-balanced-depth-joint-v1'
    assert cfg['seeds']==[17,29,43]
    assert cfg['schedule']['depth_updates']==12000
    assert cfg['encoder']['tap_indices']==[5,11,17,23]
    assert not cfg['encoder']['final_token_cache_allowed']
    assert cfg['arms']=={'D0':{'joint_keypoints':False,'part_balance':False},
                         'D1':{'joint_keypoints':False,'part_balance':True},
                         'J0':{'joint_keypoints':True,'part_balance':False},
                         'J1':{'joint_keypoints':True,'part_balance':True}}
    part=cfg['part_balance']
    assert len(part['points'])==len(part['part_weights']) and all(x>0 for x in part['part_weights'])
    assert 0<part['strength'] and 1<=part['max_pixel_boost']<=4
    assert cfg['boundaries']['start_slurm_on_publish'] is False


def compare(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:raise TypeError('Parent/head outputs must be tensors or tensor containers')


def gradient_norm(loss,parameters):
    parameters=[p for p in parameters if p.requires_grad]
    if not parameters:raise ValueError('No trainable shared encoder parameters')
    grads=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
    return float(sum((g.detach().float().square().sum() for g in grads if g is not None),start=torch.zeros((),device=loss.device)).sqrt())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default=str(ROOT/'configs/experiments/part_balanced_depth_joint_20261028.json'))
    ap.add_argument('--check-design',action='store_true')
    ap.add_argument('--factory',help='Explicit cloud adapter module:function, no implicit dataset/checkpoint discovery')
    ap.add_argument('--arm',choices=['J0','J1'],default='J0');ap.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    ap.add_argument('--device',default='cpu');ap.add_argument('--output')
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());check_design(cfg)
    if args.check_design:
        print(json.dumps({'design_valid':True,'execution_enabled':cfg['execution_enabled'],'training_started':False}));return
    if not args.factory or not args.output:ap.error('--factory and a fresh --output are required for real-data preflight')
    target=Path(args.output)
    if target.exists():raise FileExistsError('Preflight evidence must not overwrite a prior attempt')
    target.parent.mkdir(parents=True,exist_ok=True)
    name,fn=args.factory.split(':',1)
    bundle=getattr(importlib.import_module(name),fn)(cfg,args.arm,args.seed,args.device)
    validate_identity(bundle['identity'])
    model=bundle['model'];model.eval()
    if tuple(model.tap_indices)!=tuple(cfg['encoder']['tap_indices']):
        raise ValueError('Head taps differ from the registered design')
    trainable_blocks=[i for i,b in enumerate(model.encoder.blocks) if any(p.requires_grad for p in b.parameters())]
    if trainable_blocks!=list(range(24)):
        raise ValueError('The joint all-block arm requires live parameters in every one of24 encoder blocks')
    for key in ['depth_batch','keypoint_batch']:
        if bundle[key].get('split')!='train':raise ValueError('Preflight may use only admitted TRAIN batches')
    with torch.no_grad():
        db,kb=bundle['depth_batch'],bundle['keypoint_batch']
        compare(model(db['images'],task='depth')['depth'],bundle['reference_depth'](db['images']))
        compare(model(kb['images'],task='keypoints')['keypoints'],bundle['reference_keypoints'](kb['images']))
    norms={}
    for task,batch,callback in [('depth',db,bundle['depth_loss']),('keypoints',kb,bundle['keypoint_loss'])]:
        loss=callback(model(batch['images'],task=task)[task],batch)
        norms[task]=gradient_norm(loss,model.encoder.parameters())
        if not 0<norms[task]<float('inf'):raise ValueError(f'No finite live encoder gradient for {task}')
    before=copy.deepcopy(model.state_dict())
    result=joint_train_step(model,bundle['optimizer'],db,kb,bundle['depth_loss'],bundle['keypoint_loss'],
        depth_weight=cfg['schedule']['joint_task_loss_weights']['depth'],
        keypoint_weight=cfg['schedule']['joint_task_loss_weights']['keypoints'])
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
    for prefix in ['encoder.','depth_head.','keypoint_head.']:
        if not any(k.startswith(prefix) for k in changed):raise ValueError(f'No parameter update in {prefix}')
    target.write_text(json.dumps(dict(passed=True,arm=args.arm,seed=args.seed,device=args.device,
        identity=bundle['identity'],parent_output_parity=True,encoder_task_gradient_norms=norms,
        trainable_encoder_blocks=trainable_blocks,
        discarded_one_update=result,changed_tensor_count=len(changed),saved_weights=False,
        formal_training_started=False,limitation='A binding preflight is not convergence, generalization or H100 multi-GPU validation.'),indent=2)+'\n')
    print(str(target))


if __name__=='__main__':main()
