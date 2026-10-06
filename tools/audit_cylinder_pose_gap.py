"""Frozen train-pose/development gap at exactly matched camera subsets/orders."""
from collections import defaultdict
import fcntl
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,
    fingerprint,write_json,instance_targets_from_batch)
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.localization_audit import refinement_statistics,summarize
from mousegeoforge.pose_gap_audit import GROUPS,build_training_cohort
from audit_cylinder_localization import part

OUT=ROOT/'output/cylinder_pose_gap_v1'
PARENT=ROOT/'output/cylinder_combinations_v2'
AUX=ROOT/'output/cylinder_coarse_aux_v1'
AUDIT=ROOT/'output/cylinder_localization_audit_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'running.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Audit already exists; do not overwrite')
        run()


def run():
    start=time.monotonic();spec=json.loads((PARENT/'protocol.json').read_text());cohort=build_training_cohort(spec)
    assert len(cohort)==88 and len({e['record']['scene'] for e in cohort})==24
    for folder in (PARENT,AUX):assert json.loads((folder/'delivery_verification.json').read_text())['passed']
    sources={r['scene']:r['source_revision'] for r in spec['records'].values()}
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    paths={'original':ROOT/spec['config']['checkpoint'],'adapted':PARENT/'combination_coverage/final_heads.pt','auxiliary':AUX/'coarse_body_aux/final_heads.pt'}
    reused=[AUDIT/'points.json',AUX/'localization_points.json',PARENT/'results.json',AUX/'results.json']
    code={p:fingerprint(ROOT/p) for p in ('tools/audit_cylinder_pose_gap.py','mousegeoforge/pose_gap_audit.py','mousegeoforge/localization_audit.py')}
    protocol=dict(cohort=cohort,checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in paths.items()},
        source_sha256=sources,code_sha256=code,reused_sha256={str(p.relative_to(ROOT)):fingerprint(p) for p in reused},
        parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),
        scope='16 existing Cylinder train poses x5 exact dev camera orders +8 dual train pairs. Read-only; no new training or real data.',
        caution='Train poses can have camera subsets/orders not visited during the 800-step adaptation. Exposure counts retained. Train/dev differ in action or source session, not only train membership.',
        aggregation='Valid body16 point-weighted NME/window statistics; original Core23 detection-aware PCK is scene-macro.',
        matching='GT box and objectness only; fixed coarse/final query mapping per checkpoint, not end-to-end association.')
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
    names=json.loads((AUDIT/'verification.json').read_text())['keypoint_names']
    rows=[];metrics=[];projection=[];camera_diffs=[]
    prototypes={}
    for idx,e in enumerate(spec['evaluation']):
        if e['group'] in GROUPS and e['group'] not in prototypes:
            prototypes[e['group']]=torch.load(PARENT/'initial'/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
    for idx,e in enumerate(cohort):
        r=e['record'];batch=oriented_batch(r,num_views=len(r['views']));target=instance_targets_from_batch(batch);images=batch['images'].cuda()
        proto=prototypes[e['group']]
        assert r['views']==proto['record']['views'] and r['physical_roles']==proto['record']['physical_roles']
        # Different homecage sessions can have different camera extrinsics; only
        # require physical camera parity for the verified identical Cylinder rig.
        if e['group']!='old_dual_pair':
            for key in ('intrinsics','extrinsics'):
                difference=float((target[key]-proto['targets'][key]).abs().max());camera_diffs.append(difference);assert difference<1e-5
        xyz=target['keypoint_world_xyz'][0];homo=torch.cat((xyz,torch.ones_like(xyz[...,:1])),dim=-1)
        uvw=torch.einsum('vij,mkj->vmki',target['intrinsics'][0]@target['extrinsics'][0,:,:3],homo)
        expected=target['keypoint_xy'][0]*torch.tensor(r['image_hw'][::-1])
        projection.extend((uvw[...,:2]/uvw[...,2:3]-expected).norm(dim=-1)[target['keypoint_valid'][0]].tolist())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):cached,patch_start=model.aggregator(images)
        for tag,state in states.items():
            restore(model,state)
            with torch.inference_mode():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,patch_start,coupling='off').items()}
            metric,assignment=evaluate_scene(pred,batch,target)
            metrics.append(dict(checkpoint=tag,split='train_pose',group=e['group'],scene=r['scene'],metrics=metric))
            folder=OUT/tag;folder.mkdir(exist_ok=True)
            torch.save(dict(record=r,group=e['group'],oracle_assignment=assignment,
                predictions={k:v for k,v in pred.items() if k.startswith('instance_') and 'mask' not in k},
                targets={k:v for k,v in target.items() if k!='masks'}),folder/f'prediction_{idx:03d}.pt')
            for v,(qs,ms) in enumerate(assignment):
                for q,m in zip(qs,ms):
                    for k in list(range(7))+list(range(14,23)):
                        if not target['keypoint_valid'][0,v,m,k]:continue
                        s=refinement_statistics(pred['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),pred['instance_keypoint_xy'][0,v,q,k].numpy(),
                            target['keypoint_xy'][0,v,m,k].numpy(),pred['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),r['image_hw'],target['boxes_cxcywh'][0,v,m,2:].numpy())
                        rows.append(dict(checkpoint=tag,split='train_pose',group=e['group'],record=idx,scene=r['scene'],camera=r['views'][v],role=r['physical_roles'][v],
                            instance=m,keypoint=k,part=part(names[k],k),exact_order_visits=e['exact_order_visits'],unordered_subset_visits=e['unordered_subset_visits'],**s))
            del pred
        del cached,images,batch,target
        if (idx+1)%16==0:
            write_json(OUT/'status.json',dict(state='auditing',records=idx+1,total=88,elapsed_s=time.monotonic()-start));print('Completed',idx+1,'/88',flush=True)
    assert max(projection)<.01
    for tag,state in states.items():restore(model,state);assert state_hash(tensor_state(model))==state_hash(state)
    oldpoints=json.loads((AUDIT/'points.json').read_text())
    for tag,source_tag in (('original','original'),('adapted','new_sampling')):
        rows.extend(dict(r,checkpoint=tag,split='development') for r in oldpoints if r['checkpoint']==source_tag and r['group'] in GROUPS and r['part']!='tail')
    rows.extend(dict(r,checkpoint='auxiliary',split='development') for r in json.loads((AUX/'localization_points.json').read_text()) if r['group'] in GROUPS)
    pr=json.loads((PARENT/'results.json').read_text());ar=json.loads((AUX/'results.json').read_text())
    for tag,ev in (('original',pr['baseline']),('adapted',pr['arms']['combination_coverage']),('auxiliary',ar['arms']['coarse_body_aux'])):
        metrics.extend(dict(r,checkpoint=tag,split='development') for r in ev['records'] if r['group'] in GROUPS)
    grouped=defaultdict(list)
    for r in rows:
        if r['group'] in GROUPS[1:5] and r['role']!='bottom':continue
        for subset in ('body',r['part']):grouped[r['checkpoint'],r['split'],r['group'],subset].append(r)
    summary=[dict(checkpoint=k[0],split=k[1],group=k[2],subset=k[3],scenes=len({r['scene'] for r in v}),**summarize(v)) for k,v in sorted(grouped.items())]
    exposure=[]
    for g in GROUPS:
        entries=[e for e in cohort if e['group']==g]
        exposure.append(dict(group=g,train_pose_records=len(entries),exact_seen=sum(e['exact_order_visits']>0 for e in entries),subset_seen=sum(e['unordered_subset_visits']>0 for e in entries)))
    pck=[]
    for tag in states:
        for split in ('train_pose','development'):
            for g in ('old_cylinder_side_pair','old_dual_pair'):
                subset=[r['metrics'] for r in metrics if (r['checkpoint'],r['split'],r['group'])==(tag,split,g)]
                pck.append(dict(checkpoint=tag,split=split,group=g,scenes=len(subset),pck=float(np.mean([r['detection_aware_pck_0.1'] for r in subset]))))
    write_json(OUT/'points.json',rows);write_json(OUT/'summary.json',summary);write_json(OUT/'metrics.json',metrics)
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in paths.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    assert all(fingerprint(ROOT/p)==h for p,h in protocol['reused_sha256'].items())
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    report=dict(passed=True,weights_source_code_reused_outputs_unchanged=True,weights_updated=False,
        encoder_forwards=88,head_forwards=264,train_poses=24,training_records=88,
        gt_projection_max_px=max(projection),cylinder_camera_train_dev_max_difference=max(camera_diffs),
        exposure=exposure,original_pair_pck=pck,elapsed_s=time.monotonic()-start)
    write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    gs=list(GROUPS[1:5])+['old_dual_pair']
    for tag,color in (('original','#708090'),('adapted','#218d89'),('auxiliary','#b27649')):
        for split,style in (('train_pose','--'),('development','-')):
            data=[next(r for r in summary if (r['checkpoint'],r['split'],r['group'],r['subset'])==(tag,split,g,'body')) for g in gs]
            for ax,key in zip(axes,('coarse_nme','refined_nme','inside_window')):
                ax.plot(range(5),[r[key] for r in data],ls=style,marker='o',color=color,label=f'{tag} / {split}')
    for ax,title in zip(axes,('Coarse body NME','Refined body NME','GT inside local window')):
        ax.set_title(title);ax.set_xticks(range(5),gs,rotation=25,ha='right')
    axes[0].legend(fontsize=6);fig.suptitle('Same camera orders | train poses vs held-out action/session | solid=development, dashed=train poses')
    fig.savefig(OUT/'pose_gap.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
    print(json.dumps([r for r in summary if r['subset']=='body'],indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(state='failed',error=repr(e)));raise
