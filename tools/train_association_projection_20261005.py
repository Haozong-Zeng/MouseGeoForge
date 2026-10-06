"""Matched TRAIN-only RGB/gray metric learning, fixed budget and inner holdout."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.association_projection import AssociationProjection,scene_contrastive_loss
from diagnose_cross_query_affinity_20261005 import auc
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'association_projection_pilot_v1'


def pack(crops,evidence,role):
    rows=sorted([x for x in crops if x['role']==role],key=lambda x:x['group']);lookup={x['group']:x for x in evidence if x['role']==role}
    x={arm:[] for arm in ['rgb','gray']};labels=[];views=[];queries=[]
    for row in rows:
        e=lookup[row['group']];assert row['scene']==e['scene'] and len(e['nodes'])==8
        for arm in x:
            feature=torch.stack([row['descriptors'][arm+'_cls'][n['view_index'],n['query']] for n in e['nodes']]);assert torch.isfinite(feature).all();x[arm].append(feature)
        labels.append(torch.tensor([-1 if y is None else y for y in e['identity_diagnostic']]))
        views.append(torch.tensor([n['view'] for n in e['nodes']]));queries.append(torch.tensor([n['query'] for n in e['nodes']]))
    return dict(features={k:torch.stack(v) for k,v in x.items()},labels=torch.stack(labels),views=torch.stack(views),queries=torch.stack(queries),scenes=[x['scene'] for x in rows])


@torch.no_grad()
def evaluate(embeddings,data,indices):
    z=F.normalize(embeddings.detach().float().cpu(),dim=-1);labels=data['labels'];views=data['views'];queries=data['queries'];pairs=[];rankings=[]
    for index in indices:
        matrix=(z[index]@z[index].T).numpy();ids=labels[index].tolist();vs=views[index].tolist();qs=queries[index].tolist();best={}
        for a in range(len(ids)):
            for view in sorted(set(vs)):
                candidates=[b for b in range(len(ids)) if vs[b]==view and view!=vs[a]]
                if not candidates:continue
                maximum=max(matrix[a,b] for b in candidates);ties=[b for b in candidates if abs(matrix[a,b]-maximum)<1e-8]
                if len(ties)==1:best[a,view]=ties[0]
            for b in range(a+1,len(ids)):
                if vs[a]!=vs[b] and ids[a]>=0 and ids[b]>=0:pairs.append(dict(score=float(matrix[a,b]),same=ids[a]==ids[b],same_query=qs[a]==qs[b]))
        for (a,view),b in best.items():
            known=ids[a]>=0 and ids[b]>=0
            rankings.append(dict(known=known,same=ids[a]==ids[b] if known else False,mutual=best.get((b,vs[a]))==a and a<b,same_query=qs[a]==qs[b]))
    def rankstats(items):
        known=[x for x in items if x['known']]
        return dict(proposed=len(items),known=len(known),unknown=len(items)-len(known),correct=sum(x['same'] for x in known),
            precision_known=float(np.mean([x['same'] for x in known])) if known else None,same_query_fraction=float(np.mean([x['same_query'] for x in items])) if items else None)
    loss,log=scene_contrastive_loss(z[indices],labels[indices],views[indices])
    return dict(scenes=len(indices),known_nodes=int((labels[indices]>=0).sum()),unknown_nodes=int((labels[indices]<0).sum()),contrastive_loss=float(loss),
        active_anchors=int(log['active_anchors']),known_pairs=len(pairs),same_identity_auc=auc([x['score'] for x in pairs],[x['same'] for x in pairs]),
        same_query_auc=auc([x['score'] for x in pairs],[x['same_query'] for x in pairs]),
        positive_mean=float(np.mean([x['score'] for x in pairs if x['same']])),negative_mean=float(np.mean([x['score'] for x in pairs if not x['same']])),
        top1=rankstats(rankings),mutual=rankstats([x for x in rankings if x['mutual']]))


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());cutoff=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert dt.datetime.now(dt.timezone.utc)<cutoff
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cropfile=CYCLE/'predicted_crop_encoder_v1/descriptors.pt';evidencefile=CYCLE/'cross_query_affinity_diagnostic_v1/evidence.pt'
    crops=torch.load(cropfile,map_location='cpu',weights_only=False);evidence=torch.load(evidencefile,map_location='cpu',weights_only=False)
    data={role:pack(crops,evidence,role) for role in ['train','validation']};assert len(data['train']['scenes'])==21 and len(data['validation']['scenes'])==16
    assert set(data['train']['scenes']).isdisjoint(data['validation']['scenes'])
    order=np.random.default_rng(2718).permutation(21);cal=sorted(order[:5].tolist());fit=sorted(order[5:].tolist());indices=dict(fit=fit,calibration=cal)
    protocol=dict(seeds=[17,29,43],steps=800,batch_scenes=8,projection=[1024,64],parameters=65536,arms=['rgb','gray'],readout='crop CLS',
        learning_rate=[.001,.0001],weight_decay=.01,temperature=.1,clip_norm=1,optimizer='AdamW',fit_indices=fit,calibration_indices=cal,
        fit_scenes=[data['train']['scenes'][i] for i in fit],calibration_scenes=[data['train']['scenes'][i] for i in cal],
        source_sha256=dict(crops=fingerprint(cropfile),evidence=fingerprint(evidencefile)),
        scope='Association projection ONLY. Within-scene contrastive labels use previously predicted box IoU>=.5; unknown nodes enter forward but never negative loss. No identities/count/views enter model forward. RGB/gray identical initial weights/schedules per seed. Fit16, internal-calibration5 TRAINscenes from same action sourceA; stage-specific holdout, base pose model already saw TRAIN. Outer development sourceB disjoint. No validation until all six fixed800step fits complete; no step/model selection on it. No fusion/production promotion.')
    write_json(OUT/'protocol.json',protocol);torch.save(data,OUT/'inputs.pt')
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='association_projection_training',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    features={k:v.cuda() for k,v in data['train']['features'].items()};labels=data['train']['labels'].cuda();views=data['train']['views'].cuda();curves=[];runs=[];started=time.monotonic()
    for seed in protocol['seeds']:
        torch.manual_seed(seed);initial=AssociationProjection();initial_state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()};torch.save(initial_state,OUT/f'seed{seed}_initial.pt')
        schedule=np.random.default_rng(seed).choice(fit,size=(800,8),replace=True);np.save(OUT/f'seed{seed}_schedule.npy',schedule)
        for arm in protocol['arms']:
            assert dt.datetime.now(dt.timezone.utc)<cutoff
            model=AssociationProjection().cuda();model.load_state_dict(initial_state);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=800,eta_min=.0001);t0=time.monotonic();torch.cuda.reset_peak_memory_stats();sample_losses=[]
            with torch.no_grad():
                z=model(features[arm],enabled=True)
                for split,ids in indices.items():curves.append(dict(seed=seed,arm=arm,step=0,split=split,metrics=evaluate(z,data['train'],ids)))
            for step,batch in enumerate(schedule,1):
                if step%100==1:assert dt.datetime.now(dt.timezone.utc)<cutoff
                idx=torch.as_tensor(batch,device='cuda');opt.zero_grad(set_to_none=True)
                z=model(features[arm][idx],enabled=True);loss,log=scene_contrastive_loss(z,labels[idx],views[idx]);assert int(log['active_scenes'])>0 and torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();scheduler.step();sample_losses.append(float(loss.detach()))
                if step%100==0:
                    checkpoint=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,step=step)
                    torch.save(checkpoint,OUT/f'{arm}_seed{seed}_step{step:04d}.pt')
                    with torch.no_grad():
                        z=model(features[arm],enabled=True)
                        for split,ids in indices.items():curves.append(dict(seed=seed,arm=arm,step=step,split=split,metrics=evaluate(z,data['train'],ids)))
                    write_json(OUT/'curves.json',curves);write_json(OUT/'status.json',dict(stage='training',pid=os.getpid(),seed=seed,arm=arm,step=step))
            torch.cuda.synchronize();runs.append(dict(seed=seed,arm=arm,seconds=time.monotonic()-t0,peak_gpu_allocated=torch.cuda.max_memory_allocated(),last100_sample_loss=float(np.mean(sample_losses[-100:])),previous100_sample_loss=float(np.mean(sample_losses[-200:-100]))))
            print('TRAINED',seed,arm,runs[-1],flush=True);del model,opt,scheduler
    # All fixed-budget training is now over. Only now evaluate development.
    results=[];embeddings={}
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            model=AssociationProjection().cuda().eval()
            for phase in ['raw','initial','trained']:
                if phase=='initial':model.load_state_dict(torch.load(OUT/f'seed{seed}_initial.pt',map_location='cpu',weights_only=True))
                if phase=='trained':model.load_state_dict(torch.load(OUT/f'{arm}_seed{seed}_step0800.pt',map_location='cpu',weights_only=False)['state'])
                for role in ['train','validation']:
                    with torch.no_grad():z=model(data[role]['features'][arm].cuda(),enabled=phase!='raw').cpu()
                    embeddings[f'{arm}_seed{seed}_{phase}_{role}']=z
                    for split,ids in (indices.items() if role=='train' else [('validation',list(range(16)))]):
                        results.append(dict(seed=seed,arm=arm,phase=phase,split=split,metrics=evaluate(z,data[role],ids)))
            del model
    torch.save(embeddings,OUT/'embeddings.pt');write_json(OUT/'results.json',dict(results=results,runs=runs,elapsed_seconds=time.monotonic()-started,
        protocol_sha256=fingerprint(OUT/'protocol.json'),inputs_sha256=fingerprint(OUT/'inputs.pt'),module_sha256=fingerprint(ROOT/'mousegeoforge/models/association_projection.py'),script_sha256=fingerprint(Path(__file__)),production_changed=False,
        claim='Association-only metric pilot, no threshold/fusion deployment. Action-source-held-out synthetic development shares geometry/material assets; gray is chroma ablation not same-color validation.'))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='association_projection_pilot_complete',active_job=None,active_pid=None,next='Inspect fit/calibration curves and all3seed RGB/gray association results, visualize pair outcomes; no VAL threshold or deployment.');state['completed'].append('association_projection_rgb_gray_800x3x2');write_json(CYCLE/'state.json',state)
    print(json.dumps([r for r in results if r['phase']=='trained' and r['split']=='validation'],indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
