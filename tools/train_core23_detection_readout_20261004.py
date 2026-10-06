"""Frozen-token box/objectness-only adaptation, fixed1600steps x3seeds."""
import argparse,collections,datetime as dt,fcntl,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout
from mousegeoforge.detection_readout_loss import detection_readout_loss,match_detection,synthetic_detection_retention
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';SOURCE=CYCLE/'detection_features_v1';OUT=CYCLE/'detection_readout_pilot_v1'


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--retention',action='store_true');parser.add_argument('--context',choices=['roi','global']);args=parser.parse_args()
    if args.context:args.retention=True
    if args.retention:OUT=CYCLE/'detection_readout_retention_v1'
    if args.context:OUT=CYCLE/f'detection_context_{args.context}_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();manifest=json.loads((SOURCE/'manifest.json').read_text());OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='training_detection_only_readout',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    rows=[]
    for r in manifest['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    data={k:torch.stack([r[k] for r in rows]).cuda() for k in ['instance_tokens','boxes','box_valid','instance_valid','original_boxes','original_objectness']}
    data['valid']=data['box_valid']&data['instance_valid'];complete=torch.tensor([r['metadata']['detection_complete'] for r in rows],device='cuda',dtype=torch.bool)
    meta=[r['metadata'] for r in rows];sets=collections.defaultdict(list);buckets={k:collections.defaultdict(list) for k in ['real','synthetic']}
    for i,m in enumerate(meta):
        r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3']
        if not m['detection_complete']:continue
        kind='real' if r['kind']=='real' else 'synthetic';sets[kind+'_'+r['role']].append(i)
        if r['role']=='train':buckets[kind][r.get('session') or r['scene'].split('/')[2]].append(i)
    assert all(sets[k+'_'+role] for k in buckets for role in ['train','validation'])
    synthetic_train=torch.tensor([m['record']['kind']!='real' and m['record']['role']=='train' and m['detection_complete'] for m in meta],device='cuda',dtype=torch.bool)
    if args.context:
        roi_manifest=json.loads((CYCLE/'detection_roi_features_v1/manifest.json').read_text());assert fingerprint(ROOT/roi_manifest['path'])==roi_manifest['sha256'];roi_features=torch.load(ROOT/roi_manifest['path'],map_location='cpu',weights_only=False);assert roi_features['metadata']==meta
        for key in ['roi','global_memory']:data[key]=roi_features[key].float().cuda()
    def forward(model,ids):
        if args.context:return model(data['instance_tokens'][ids],data['roi'][ids],data['global_memory'][ids],data['original_boxes'][ids],data['original_objectness'][ids])
        return model(data['instance_tokens'][ids])
    @torch.no_grad()
    def predict(model):
        out=collections.defaultdict(list)
        for start in range(0,len(rows),128):
            for k,v in forward(model,slice(start,start+128)).items():out[k].append(v)
        return {k:torch.cat(v) for k,v in out.items()}
    @torch.no_grad()
    def evaluate(p):
        result={};assignment=[]
        for group,ids in sets.items():
            pp={k:v[ids] for k,v in p.items()};matches=match_detection(pp,data['boxes'][ids],data['valid'][ids]);tp=gt=pred=duplicates=0;ious=[];card=[]
            for j,i in enumerate(ids):
                qs,ms=matches[j];scores=pp['instance_objectness_logits'][j].sigmoid();selected=scores>=.5;pred+=int(selected.sum());gt+=len(ms);card.append(abs(int(selected.sum())-len(ms)))
                matrix=box_iou(pp['instance_boxes_cxcywh'][j],data['boxes'][i,ms])
                for q,m in zip(qs.tolist(),ms.tolist()):
                    col=ms.tolist().index(m);iou=float(matrix[q,col]);ious.append(iou);tp+=int(iou>=.5 and bool(selected[q]));assignment.append([i,q,m])
                if len(ms):duplicates+=int(((matrix[selected]>=.5).sum(0)-1).clamp_min(0).sum())
            result[group]=dict(images=len(ids),tp=tp,gt=gt,predictions=pred,precision=tp/max(pred,1),recall=tp/max(gt,1),matched_iou_mean=float(np.mean(ious)),matched_iou_median=float(np.median(ious)),count_absolute_error=float(np.mean(card)),duplicate_boxes=duplicates)
        return result,assignment
    initial=SOURCE/'initial_detector.pt';assert fingerprint(initial)==manifest['initial_detector_sha256']
    base={'instance_boxes_cxcywh':data['original_boxes'],'instance_objectness_logits':data['original_objectness']}
    baseline,base_assignment=evaluate(base);write_json(OUT/'baseline.json',dict(metrics=baseline,assignment=base_assignment))
    write_json(OUT/'protocol.json',dict(steps=1600,seeds=[17,29,43],batch='8real+8synthetic source-uniform',lr='cosine3e-4to1e-5',weight_decay=.001,
        loss='Original-detector Hungarian cost objectness+5L1; weighted BCE positive1/noobject.1 +5boxL1 +2GIoU. No keypoint/mask/encoder/decoder update.',
        retention='weight1*(all-query soft BCE+5L1 original confidence>=.5 boxes), only synthetic TRAIN; teacher detached. Fixed before results; no validation retention.' if args.retention else 'off',
        context=args.context,context_protocol='Matched ROI/global residual heads, original frozen box/objectness predictions plus zero-initialized residual;5x5spatial ROI uses2x originalpredictedbox. Global repeats whole-image mean. Parameters identical. No GT crops; not a full DAB-DETR reproduction.' if args.context else None,
        context_feature_sha256=roi_manifest['sha256'] if args.context else None,
        model_source_sha256=fingerprint(ROOT/'mousegeoforge/models'/('spatial_detection_readout.py' if args.context else 'instance_detection_readout.py')),
        loss_source_sha256=fingerprint(ROOT/'mousegeoforge/detection_readout_loss.py'),
        data='Only complete mask-backed views. Original source split unchanged; no fabricated negatives from missing masks. GT boxes audited against masks pixel-edge convention.',
        evaluation='Scores>=.5 and matchedIoU>=.5 noNMS; not AP. Include IoU/count error/duplicates and assignment changes. Downstream pose is separate pending audit.',
        initial_sha256=fingerprint(initial),source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),script_sha256=fingerprint(Path(__file__)),counts={k:len(v) for k,v in sets.items()}))
    write_json(OUT/'observations.json',meta);results=[];allpred={}
    for seed in [17,29,43]:
        torch.manual_seed(seed)
        if args.context:
            model=SpatialDetectionReadout(args.context).cuda();torch.save(model.state_dict(),OUT/f'seed{seed}_initial.pt')
            if args.context=='roi':
                control_init=CYCLE/f'detection_context_global_v1/seed{seed}_initial.pt'
                if control_init.exists():
                    for k,v in torch.load(control_init,map_location='cuda',weights_only=True).items():torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)
        else:model=InstanceDetectionReadout().cuda();model.load_state_dict(torch.load(initial,map_location='cuda',weights_only=True),strict=True)
        initial_pred=predict(model)
        for key in base:torch.testing.assert_close(initial_pred[key],base[key],atol=2e-5,rtol=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001);rng=random.Random(seed+5000);history=[dict(step=0,metrics=baseline)]
        schedule=[]
        matched_schedule=json.loads((CYCLE/f'detection_readout_pilot_v1/seed{seed}_schedule.json').read_text()) if args.retention else None
        for step in range(1,1601):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            ids=[rng.choice(buckets[k][rng.choice(sorted(buckets[k]))]) for k in ['real','synthetic'] for _ in range(8)];schedule.append(ids)
            if matched_schedule is not None:assert ids==matched_schedule[step-1]
            lr=1e-5+(3e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/1600))
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True);p=forward(model,ids);loss,logs=detection_readout_loss(p,data['boxes'][ids],data['valid'][ids],complete[ids])
            if args.retention:
                retention,retlogs=synthetic_detection_retention(p,data['original_boxes'][ids],data['original_objectness'][ids],synthetic_train[ids]);loss=loss+retention;logs.update(retlogs)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
            if step%200==0:
                prediction=predict(model);metrics,assignment=evaluate(prediction);row=dict(step=step,loss=float(loss.detach()),loss_components={k:float(v.detach()) for k,v in logs.items()},metrics=metrics);history.append(row);write_json(OUT/f'seed{seed}_history.json',history)
                torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,seed=seed),OUT/f'seed{seed}_latest.pt');status('training',seed=seed,step=step);print('DET',seed,step,metrics['real_validation'],metrics['synthetic_validation'],flush=True)
        results.append(dict(seed=seed,**history[-1]));write_json(OUT/'partial_results.json',results);write_json(OUT/f'seed{seed}_schedule.json',schedule);write_json(OUT/f'seed{seed}_assignment.json',assignment)
        for k,v in prediction.items():allpred[f'seed{seed}__{k}']=v.cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**allpred);write_json(OUT/'results.json',dict(results=results,no_pose_evaluation_yet=True,no_production_promotion=True));status('complete')
    state.update(stage='detection_readout_results_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=f'Review {OUT.name} metrics vsbaseline and matched pilot. Recompute all-query spatial pose using predicted adaptedboxes, compare conditional and detection-gated coverage, switches/duplicates. No pose gain claim from detection gain alone.');state['completed'].append(OUT.name+'_3seeds1600steps');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
