"""Cylinder full-task gradients versus dual-body task; no optimizer updates."""
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
from run_depth_pose_ablation import (ROOT, torch, VGGTOmega, DepthPoseModel, restore, tensor_state,
    state_hash, fingerprint, write_json, instance_targets_from_batch, depth_loss, move_batch)
from train_cylinder_view_balance import oriented_batch, scene_digest
from mousegeoforge.task_gradient_audit import matched_task_terms, compare_gradient_sets, select_task_records, upstream

OUT = ROOT / 'output/cylinder_task_gradients_v1'
PARENT = ROOT / 'output/cylinder_source_mix_v1'


def run():
    start = time.monotonic(); spec = json.loads((PARENT/'protocol.json').read_text())
    assert json.loads((PARENT/'status.json').read_text())['state'] == 'completed'
    assert json.loads((PARENT/'delivery_verification.json').read_text())['passed']
    with (PARENT/'running.lock').open() as lock: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    entries = select_task_records(spec); sources = {e['record']['scene']: e['record']['source_revision'] for e in entries}
    code = dict(spec['code_sha256'])
    for p in ('tools/audit_cylinder_task_gradients.py', 'mousegeoforge/task_gradient_audit.py'):
        code[p] = fingerprint(ROOT/p)
    for p,h in code.items(): assert fingerprint(ROOT/p) == h
    assert all(scene_digest(ROOT/p) == h for p,h in sources.items())
    paths = {'original': ROOT/spec['config']['checkpoint'], 'mixed': PARENT/'source_mix/final_heads.pt'}
    protocol = dict(entries=entries, source_sha256=sources, code_sha256=code,
        checkpoints={t:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for t,p in paths.items()},
        parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),
        decision='Support only if SAME parameter group has mean Cylinder-total versus dual-body cosine < -.1 at BOTH checkpoints AND negative pair fraction >= .5 at BOTH. No threshold tuning.',
        losses='Original total depth+pose; body and tail are contributions to weight5 all-valid-Core23 mean, sharing original Hungarian once. Full=pose+depth. No loss/optimizer changes.',
        scope='8 dual replay and 4+4 Cylinder existing TRAIN poses; exact existing side-bottom order, original/mixed train-mode heads; frozen eval encoder. First-order raw gradients, not Adam trajectory or causal proof.',
        optimizer_steps=0)
    write_json(OUT/'protocol.json', protocol)
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    encoder = ROOT/'vggt-omega/vggt_omega_1b_512.pt'; assert fingerprint(encoder)==spec['encoder_sha256']
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().train(); del spatial
    states={t:torch.load(p,weights_only=True,map_location='cpu')['model'] for t,p in paths.items()}
    hashes={t:state_hash(s) for t,s in states.items()}
    groups={'shared_adapter':list(model.shared_adapter.parameters()),'instance_head':list(model.instance_pose_head.parameters())}
    params=[p for ps in groups.values() for p in ps]
    lw=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    args=SimpleNamespace(semantic_label_weights='',body_depth_balance=.5,body_depth_fraction=.5,depth_gradient_weight=0.)
    vectors=defaultdict(list); norms=[]; loss_rows=[]; off_difference=0.; decomposition_difference=0.
    for idx,e in enumerate(entries):
        r=e['record']; batch=move_batch(oriented_batch(r,num_views=2),'cuda'); targets=instance_targets_from_batch(batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): cached,patch_start=model.aggregator(batch['images'])
        for tag,state in states.items():
            restore(model,state); pred=model.forward_cached(batch['images'],cached,patch_start,coupling='off')
            with patch.object(upstream,'hungarian_match_instances',wraps=upstream.hungarian_match_instances) as matcher:
                pose,logs,parts=matched_task_terms(pred,targets,lw,image_size=r['image_hw']); assert matcher.call_count==1
            baseline,_=upstream.instance_pose_loss(pred,targets,lw,image_size=r['image_hw'])
            off_difference=max(off_difference,float((pose-baseline).abs().detach())); assert torch.equal(pose,baseline)
            delta=float((parts['body']+parts['tail']-5*logs['loss_instance_keypoint_l1']).abs().detach())
            decomposition_difference=max(decomposition_difference,delta); assert delta<1e-7
            depth=depth_loss(pred['depth'],batch['depths'],batch['depth_masks'],batch['body_masks'],batch['semantic_masks'],args)['loss_depth']
            terms={'total':pose+depth,'body':parts['body'],'depth':depth}
            for term,value in terms.items():
                gradients=torch.autograd.grad(value,params,retain_graph=term!='depth',allow_unused=True)
                offset=0
                for group,ps in groups.items():
                    vector=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten().cpu()
                                      for p,g in zip(ps,gradients[offset:offset+len(ps)])])
                    assert torch.isfinite(vector).all()
                    if term=='depth' and group=='instance_head': assert not vector.any()
                    vectors[tag,e['kind'],group,term].append(vector)
                    norms.append(dict(checkpoint=tag,record=idx,kind=e['kind'],group=group,term=term,norm=float(vector.norm())))
                    offset+=len(ps)
            loss_rows.append(dict(checkpoint=tag,record=idx,kind=e['kind'],total=float((pose+depth).detach()),pose=float(pose.detach()),
                body=float(parts['body'].detach()),tail=float(parts['tail'].detach()),depth=float(depth.detach()),valid_points=parts['valid_points']))
            assert state_hash(tensor_state(model))==hashes[tag]
            del pred,pose,logs,parts,depth,terms,gradients,baseline
        del cached,batch,targets
        print(f'Gradients {idx+1}/16',flush=True)
        write_json(OUT/'status.json',dict(state='auditing',records=idx+1,total=16))
    summary=[]
    for tag in states:
        for group in groups:
            dual=vectors[tag,'dual',group,'body']
            for term in ('total','body','depth'):
                summary.append(dict(checkpoint=tag,group=group,cylinder_term=term,
                    **compare_gradient_sets(vectors[tag,'cylinder',group,term],dual)))
            # Pose-only gradient follows linear subtraction; no extra matching/backward.
            pose_vectors=[a-b for a,b in zip(vectors[tag,'cylinder',group,'total'],vectors[tag,'cylinder',group,'depth'])]
            summary.append(dict(checkpoint=tag,group=group,cylinder_term='pose',**compare_gradient_sets(pose_vectors,dual)))
    supported=[]
    for group in groups:
        rows=[r for r in summary if r['group']==group and r['cylinder_term']=='total']
        if all(r['mean_gradient_cosine'] is not None and r['mean_gradient_cosine']<-.1 and r['negative_pair_fraction']>=.5 for r in rows): supported.append(group)
    for p,h in code.items(): assert fingerprint(ROOT/p)==h
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    assert all(fingerprint(p)==protocol['checkpoints'][t]['sha256'] for t,p in paths.items())
    report=dict(passed=True,conflict_mechanism_supported=bool(supported),supported_groups=supported,
        zero_weight_max_difference=off_difference,body_tail_decomposition_max_difference=decomposition_difference,
        depth_to_instance_head_gradient_zero=True,original_matches_per_loss=1,encoder_forwards=16,head_forwards=32,
        optimizer_steps=0,source_code_weights_unchanged=True,elapsed_s=time.monotonic()-start,summary=summary)
    for name,value in [('report',report),('gradient_norms',norms),('losses',loss_rows)]: write_json(OUT/(name+'.json'),value)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,group in zip(axes,groups):
        for tag,offset,color in [('original',-.18,'#698fa9'),('mixed',.18,'#c38a4c')]:
            values=[next(r['mean_gradient_cosine'] for r in summary if (r['checkpoint'],r['group'],r['cylinder_term'])==(tag,group,term)) for term in ('total','pose','body','depth')]
            ax.bar(np.arange(4)+offset,[v if v is not None else 0 for v in values],width=.36,label=tag,color=color)
        ax.axhline(0,color='black',lw=.7);ax.axhline(-.1,color='red',ls=':',lw=.8)
        ax.set_ylim(-1,1);ax.set_xticks(range(4),['total','pose','body','depth']);ax.set_title(group);ax.legend()
    fig.suptitle('Cylinder mean gradient vs dual body | head depth is absent, not cosine zero')
    fig.savefig(OUT/'gradient_conflict.png',dpi=150);plt.close(fig)
    write_json(OUT/'status.json',dict(state='completed'))
    print(json.dumps({k:v for k,v in report.items() if k!='summary'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='pair_cosines'} for r in summary],indent=2))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as glock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists(): raise RuntimeError('Do not overwrite existing audit')
        try:run()
        except BaseException as exc:
            write_json(OUT/'status.json',dict(state='failed',error=repr(exc),optimizer_steps=0));raise
