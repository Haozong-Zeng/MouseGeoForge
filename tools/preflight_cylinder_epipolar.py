"""Training-only gradient preflight for the fixed weight-one epipolar proposal."""
from collections import defaultdict
import fcntl
import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_depth_pose_ablation import ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch,depth_loss,move_batch
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.epipolar_supervision import matched_epipolar_body_loss,epipolar_aux_pose_loss,upstream

OUT=ROOT/'output/cylinder_epipolar_preflight_v1'
PARENT=ROOT/'output/cylinder_source_mix_v1'


def select_records(spec):
    seq=[spec['records'][x[0]] for x in spec['arms']['source_mix']]
    special={r['scene']:r for r,s in zip(seq,spec['slots']) if s['kind']=='cylinder'}
    records=[]
    for name in sorted({r['dataset'] for r in special.values()}):
        poses=sorted([r for r in special.values() if r['dataset']==name],key=lambda r:r['frame'])
        for pose in (poses[0],poses[-1]):
            candidates=[r for r in seq if r['scene']==pose['scene']]
            records.append(next(r for r in candidates if r['physical_roles']==['bottom']))
            for n in (2,3,4):records.append(next(r for r in candidates if len(r['views'])==n))
    dual=sorted({r['scene']:r for r in seq if r['expected_instances']==2}.values(),key=lambda r:r['frame'])
    records.extend(dual[round(i*(len(dual)-1)/3)] for i in range(4))
    assert len(records)==20 and len({r['scene'] for r in records})==8
    assert all(r['split']=='train' for r in records)
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as gpu_lock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing preflight; do not overwrite')
        run()


def run():
    start=time.monotonic();spec=json.loads((PARENT/'protocol.json').read_text());records=select_records(spec)
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    assert json.loads((PARENT/'status.json').read_text())['state']=='completed'
    with (PARENT/'running.lock').open() as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths={'original':ROOT/spec['config']['checkpoint'],'mixture':PARENT/'source_mix/final_heads.pt'}
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    sources={r['scene']:r['source_revision'] for r in records}
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    code={p:fingerprint(ROOT/p) for p in ('tools/preflight_cylinder_epipolar.py','mousegeoforge/epipolar_supervision.py')}
    protocol=dict(records=records,checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in paths.items()},
        parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),code_sha256=code,source_sha256=sources,
        auxiliary_weight=1.,beta=.01,normalization='Signed Sampson pixel distance divided by image diagonal; per-pair body-point mean, then per-instance mean.',
        selection='First/last special-pool train pose per Cylinder dataset x bottom-mono + first actual V2/V3/V4 slots, plus four evenly spaced dual replay train poses.',
        gate='Both original and mixture checkpoints: median auxiliary/base total-gradient norm <=.25 separately for shared adapter and entire instance head over 16 multi-view records; all finite, every supervised record nonzero, mono exactly zero, GT loss<1e-9, upstream OFF exact parity, one shared matcher call.',
        no_optimizer_steps=True,scope='Training-only synthetic feasibility audit. No development-based tuning or accuracy claim.')
    write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in paths.items()}
    groups={'shared_adapter':list(model.shared_adapter.parameters()),'instance_head':list(model.instance_pose_head.parameters())}
    params=[p for ps in groups.values() for p in ps]
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    lossweights=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    rows=[];parity=0.;gt_max=0.;matcher=upstream.hungarian_match_instances
    for idx,r in enumerate(records):
        batch=move_batch(oriented_batch(r,num_views=len(r['views'])),'cuda');target=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,patch_start=model.aggregator(batch['images'])
        for tag,state in states.items():
            restore(model,state)
            pred=model.forward_cached(batch['images'],cached,patch_start,coupling='off')
            baseline,_=upstream.instance_pose_loss(pred,target,lossweights,image_size=tuple(r['image_hw']))
            with patch.object(upstream,'hungarian_match_instances',wraps=matcher) as calls:
                off,_=epipolar_aux_pose_loss(pred,target,lossweights,aux_weight=0.,image_size=tuple(r['image_hw']));assert calls.call_count==1
            difference=float((off-baseline).abs().detach());parity=max(parity,difference);assert difference==0
            del baseline,off
            with patch.object(upstream,'hungarian_match_instances',wraps=matcher) as calls:
                total,logs=epipolar_aux_pose_loss(pred,target,lossweights,aux_weight=1.,image_size=tuple(r['image_hw']));assert calls.call_count==1
            dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
            base=logs['loss_instance_base_pose']+dl;aux=logs['loss_instance_epipolar']
            ga=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(base,params,retain_graph=False,allow_unused=True)
            offset=0
            for group,ps in groups.items():
                a=[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(ga[offset:offset+len(ps)],ps)]
                b=[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(gb[offset:offset+len(ps)],ps)]
                norm_a=float(sum(x.square().sum() for x in a).sqrt());norm_b=float(sum(x.square().sum() for x in b).sqrt())
                assert np.isfinite(norm_a) and np.isfinite(norm_b) and norm_b>0
                cosine=float(sum((x*y).sum() for x,y in zip(a,b)))/(norm_a*norm_b) if norm_a>0 else None
                if len(r['views'])==1:assert norm_a==0
                else:assert logs['epipolar_points']>0 and norm_a>0
                rows.append(dict(checkpoint=tag,record=idx,scene=r['scene'],views=r['views'],roles=r['physical_roles'],group=group,
                    base_norm=norm_b,auxiliary_norm=norm_a,ratio=norm_a/norm_b,cosine=cosine,
                    base_total_loss=float(base.detach()),auxiliary_loss=float(aux.detach()),
                    supervised_pairs=logs['epipolar_pairs'],supervised_points=logs['epipolar_points']))
                offset+=len(ps)
            # The helper never consumes GT xy, but feeding GT as synthetic predicted
            # coordinates is an independent calibration/formula consistency check.
            matches=matcher(pred,target);gt_xy=pred['instance_keypoint_xy'].detach().clone()
            for v,(qs,ms) in enumerate(matches[0]):
                for q,m in zip(qs.tolist(),ms.tolist()):gt_xy[0,v,q]=target['keypoint_xy'][0,v,m]
            gt_loss,_=matched_epipolar_body_loss({'instance_keypoint_xy':gt_xy},target,matches,image_size=tuple(r['image_hw']))
            gt_max=max(gt_max,float(gt_loss));assert float(gt_loss)<1e-9
            assert state_hash(tensor_state(model))==state_hash(state)
            del pred,dl,base,aux,total,logs,ga,gb,a,b,gt_xy
        del cached,batch,target
        print('Checked',idx+1,'/20',flush=True)
        write_json(OUT/'status.json',dict(state='preflight',records=idx+1,total=20))
    summary=[]
    for tag in states:
        for group in groups:
            selected=[r for r in rows if r['checkpoint']==tag and r['group']==group and len(r['views'])>1]
            summary.append(dict(checkpoint=tag,group=group,records=len(selected),median_ratio=float(np.median([r['ratio'] for r in selected])),
                max_ratio=max(r['ratio'] for r in selected),median_cosine=float(np.median([r['cosine'] for r in selected]))))
    supported=all(r['median_ratio']<=.25 for r in summary)
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in paths.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    report=dict(passed=True,training_supported=supported,summary=summary,upstream_zero_weight_max_difference=parity,
        gt_loss_max=gt_max,matcher_calls_per_loss=1,weights_source_code_unchanged=True,optimizer_steps=0,
        encoder_forwards=20,head_forwards=40,records=20,elapsed_s=time.monotonic()-start)
    write_json(OUT/'parameter_gradients.json',rows);write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,group in zip(axes,groups):
        for tag in states:
            selected=[r for r in rows if r['checkpoint']==tag and r['group']==group and len(r['views'])>1]
            ax.plot([r['record'] for r in selected],[r['ratio'] for r in selected],marker='o',label=tag)
        ax.axhline(.25,color='red',ls='--',label='median gate = .25');ax.set(title=group,xlabel='fixed training record',ylabel='auxiliary / base gradient norm');ax.legend(fontsize=8)
    fig.suptitle('Weight-one epipolar feasibility | per-record ratios, gate applies to median | no updates')
    fig.savefig(OUT/'gradient_budget.png',dpi=145);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
