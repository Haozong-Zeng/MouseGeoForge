"""Frozen training-only audit of local-grid indirect coordinate gradients."""
from collections import defaultdict
import fcntl
import json
from types import MethodType,SimpleNamespace
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,
    fingerprint,write_json,instance_targets_from_batch,depth_loss,move_batch)
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.coarse_pose_supervision import coarse_aux_pose_loss,upstream
from mousegeoforge.grid_gradient_audit import select_records,point_gradient,cosine

OUT=ROOT/'output/cylinder_grid_gradient_audit_v1'
PARENT=ROOT/'output/cylinder_combinations_v2'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'running.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing audit; do not overwrite')
        run()


def run():
    start=time.monotonic();spec=json.loads((PARENT/'protocol.json').read_text());records=select_records(spec)
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    paths={'original':ROOT/spec['config']['checkpoint'],'adapted':PARENT/'combination_coverage/final_heads.pt'}
    for path,h in spec['code_sha256'].items():assert fingerprint(ROOT/path)==h
    sources={r['scene']:r['source_revision'] for r in records}
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    code={p:fingerprint(ROOT/p) for p in ('tools/audit_cylinder_grid_gradients.py','mousegeoforge/grid_gradient_audit.py','mousegeoforge/coarse_pose_supervision.py')}
    protocol=dict(records=records,checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in paths.items()},
        parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),code_sha256=code,source_sha256=sources,
        scope='Training-only, 4 Cylinder poses x 5 subsets plus 4 dual poses; both original and adapted checkpoints; no updates.',
        comparison='Full grid gradient vs detach grid only; same frozen RGB features, all forward outputs and Hungarian matches.',
        gradients='Coordinate loss(weight5) and complete depth+pose loss; coarse tensor, coarse MLP, shared adapter; unclipped gradients.',
        decision_rule='Support further detach-only control only if >=10% body points have harmful full-gradient reversal in >=3 of 4 Cylinder poses at BOTH checkpoints. Diagnostic threshold, not statistical significance.',
        units='Gradients and GT directions in normalized UV (not pixel metric); body16 summary excludes tail; all23 retain training loss.')
    write_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in paths.items()}
    groups={'coarse_mlp':list(model.instance_pose_head.coarse_keypoint_xy.parameters()),'shared_adapter':list(model.shared_adapter.parameters())}
    params=[p for ps in groups.values() for p in ps];sizes=[len(x) for x in groups.values()]
    capture={};original_local=model.instance_pose_head._local_refinement
    def wrapped(self,pixels,tokens,coarse):
        capture['coarse']=coarse
        return original_local(pixels,tokens,coarse)
    model.instance_pose_head._local_refinement=MethodType(wrapped,model.instance_pose_head)
    lossweights=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    rows=[];parameter_rows=[];parity=0.;coarse_total_difference=0.
    for idx,r in enumerate(records):
        batch=move_batch(oriented_batch(r,num_views=len(r['views'])),'cuda');target=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,patch_start=model.aggregator(batch['images'])
        for tag,state in states.items():
            restore(model,state);modes={}
            for detach in (False,True):
                model.instance_pose_head.detach_local_sampling_grid=detach
                pred=model.forward_cached(batch['images'],cached,patch_start,coupling='off');coarse=capture.pop('coarse')
                pl,logs=coarse_aux_pose_loss(pred,target,lossweights,aux_weight=0.,image_size=tuple(r['image_hw']))
                dl=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
                coordinate=5*logs['loss_instance_keypoint_l1'];total=pl+dl
                gs=[]
                for term in (coordinate,total):
                    grads=torch.autograd.grad(term,[coarse]+params,retain_graph=term is coordinate,allow_unused=True)
                    gs.append([torch.zeros_like(p).cpu() if g is None else g.detach().cpu() for g,p in zip(grads,[coarse]+params)])
                modes[detach]=dict(pred={k:v.detach().cpu() for k,v in pred.items()},coarse=coarse.detach().cpu(),gradients=gs,loss=float(total.detach()))
                del pred,coarse,pl,dl,total,coordinate,logs,grads,gs
            a,b=modes[False],modes[True]
            for k in a['pred']:
                diff=float((a['pred'][k]-b['pred'][k]).abs().max());parity=max(parity,diff);assert diff==0
            assert a['loss']==b['loss']
            matches=upstream.hungarian_match_instances(a['pred'],{k:v.cpu() for k,v in target.items()})
            for mode in (a,b):
                diff=float((mode['gradients'][0][0]-mode['gradients'][1][0]).abs().max())
                coarse_total_difference=max(coarse_total_difference,diff);assert diff<1e-7
            for v,(qs,ms) in enumerate(matches[0]):
                for q,m in zip(qs.tolist(),ms.tolist()):
                    for k in list(range(7))+list(range(14,23)):
                        if not target['keypoint_valid'][0,v,m,k]:continue
                        c=a['coarse'][v,q,k].numpy();gt=target['keypoint_xy'][0,v,m,k].cpu().numpy()
                        stats=point_gradient(a['gradients'][0][0][v,q,k].numpy(),b['gradients'][0][0][v,q,k].numpy(),gt-c)
                        rows.append(dict(checkpoint=tag,record=idx,scene=r['scene'],camera=r['views'][v],role=r['physical_roles'][v],
                            instance=m,keypoint=k,kind='dual' if r['expected_instances']==2 else 'cylinder',
                            inside_window=bool((np.abs(gt-c)<=.1).all()),**stats))
            for term_index,term_name in enumerate(('coordinate','total')):
                offset=1
                for group,n in zip(groups,sizes):
                    full=torch.cat([g.flatten() for g in a['gradients'][term_index][offset:offset+n]]).numpy()
                    direct=torch.cat([g.flatten() for g in b['gradients'][term_index][offset:offset+n]]).numpy()
                    parameter_rows.append(dict(checkpoint=tag,record=idx,group=group,term=term_name,
                        full_direct_cos=cosine(full,direct),indirect_direct_cos=cosine(full-direct,direct),
                        full_norm=float(np.linalg.norm(full)),direct_norm=float(np.linalg.norm(direct)),
                        indirect_over_direct=float(np.linalg.norm(full-direct)/max(np.linalg.norm(direct),1e-12))))
                    offset+=n
            assert state_hash(tensor_state(model))==state_hash(state)
            del modes,a,b
        del cached,batch,target
        print('Completed',idx+1,'/24',flush=True)
        write_json(OUT/'status.json',dict(state='auditing',records=idx+1,total=24,elapsed_s=time.monotonic()-start))
    model.instance_pose_head.detach_local_sampling_grid=False
    grouped=defaultdict(list)
    for row in rows:grouped[row['checkpoint'],row['kind'],row['role']].append(row)
    summary=[dict(checkpoint=k[0],kind=k[1],role=k[2],points=len(v),
        harmful_reversal_fraction=float(np.mean([r['harmful_reversal'] for r in v])),
        indirect_away_fraction=float(np.mean([r['indirect_away'] for r in v])),
        full_direct_cos_mean=float(np.mean([r['full_direct_cos'] for r in v if r['full_direct_cos'] is not None]))) for k,v in sorted(grouped.items())]
    pose_support={}
    for tag in states:
        bypose=defaultdict(list)
        for row in rows:
            if row['checkpoint']==tag and row['kind']=='cylinder':bypose[row['scene']].append(row['harmful_reversal'])
        pose_support[tag]={scene:float(np.mean(v)) for scene,v in bypose.items()}
    supported=all(sum(x>=.1 for x in v.values())>=3 for v in pose_support.values())
    write_json(OUT/'points.json',rows);write_json(OUT/'parameter_gradients.json',parameter_rows)
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in paths.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    report=dict(passed=True,summary=summary,pose_support=pose_support,detach_training_supported=supported,
        forward_max_difference=parity,coarse_coordinate_total_gradient_max_difference=coarse_total_difference,
        no_updates=True,weights_source_code_unchanged=True,records=24,encoder_forwards=24,head_forwards=96,
        protocol_sha256=fingerprint(OUT/'protocol.json'),elapsed_s=time.monotonic()-start)
    write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=[(k,r) for k,r in [('cylinder','bottom'),('cylinder','top'),('cylinder','side'),('dual','top'),('dual','side')] if any(s['kind']==k and s['role']==r for s in summary)]
    for i,(tag,color) in enumerate((('original','#708090'),('adapted','#218d89'))):
        selected=[next(s for s in summary if s['checkpoint']==tag and (s['kind'],s['role'])==label) for label in labels]
        for ax,key in zip(axes,('harmful_reversal_fraction','indirect_away_fraction')):
            ax.bar(np.arange(len(labels))+(i-.5)*.35,[s[key] for s in selected],.35,label=tag,color=color)
    for ax,title in zip(axes,('Full descent away, detached descent toward GT','Indirect grid descent away from GT')):
        ax.set_title(title,fontsize=10);ax.set_xticks(range(len(labels)),[' / '.join(x) for x in labels],rotation=20,ha='right');ax.set_ylim(0,1);ax.legend()
    fig.suptitle('Frozen training-only gradient diagnosis | normalized UV | no optimizer steps')
    fig.savefig(OUT/'gradient_routes.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(state='failed',error=repr(e)));raise
