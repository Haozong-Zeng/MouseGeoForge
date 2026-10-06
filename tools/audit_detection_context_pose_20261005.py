"""Frozen mixed800 pose with old/new detectors, all queries before association."""
import argparse,collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';DET=CYCLE/'detection_view_context_control_v1';OUT=CYCLE/'detection_context_pose_audit_v1'

def main():
    global DET,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--order-control',action='store_true');parser.add_argument('--full-context-control',action='store_true');args=parser.parse_args();assert not(args.order_control and args.full_context_control)
    contexts=['paired','single','reversed'] if args.order_control else ['paired','single'];arms=['fixed','random_ref'] if args.order_control else ['paired','mixed'];pose_dir=CYCLE/('view_order_control_v1' if args.order_control else 'view_context_control_v1');pose_arm='random_ref' if args.order_control else 'mixed'
    if args.order_control:DET=CYCLE/'detection_reference_control_v1';OUT=CYCLE/'detection_reference_pose_audit_v1'
    if args.full_context_control:
        DET=CYCLE/'detection_full_context_control_v1';OUT=CYCLE/'detection_full_context_pose_audit_v1';contexts=['shared','single','full'];arms=['paired','full'];pose_dir=CYCLE/'full_context_pose_control_v1';pose_arm='full'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and (DET/'results.json').exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='auditing_detector_context_frozen_pose',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state);status('loading')
    rows=[];source=CYCLE/'detection_features_v1/manifest.json'
    for rec in json.loads(source.read_text())['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    with np.load(pose_dir/'predictions.npz') as a:indices=a['image_indices'].tolist();gold={s:{c:a[f'{pose_arm}_seed{s}_{c}'].copy() for c in contexts} for s in [17,29,43]}
    meta=[rows[i]['metadata'] for i in indices];assert all(m['record']['role']=='validation' and m['record'].get('session') not in ['GR2','UR2','UR3'] for m in meta)
    data={k:torch.stack([rows[i][k] for i in indices]).cuda() for k in ['boxes','box_valid','instance_valid','gt_xy','point_valid','pixels']};valid=data['box_valid']&data['instance_valid']
    for j,m in enumerate(meta):
        if m['expected_instances']>1:data['point_valid'][j,:,7:14]=False
    n=len(indices);tokens={'paired':torch.stack([rows[i]['instance_tokens'] for i in indices]).cuda(),'single':torch.zeros(n,6,256,device='cuda')};memories={c:[None]*n for c in tokens};lookup={(m['record']['key'],m['view_index']):j for j,m in enumerate(meta)}
    for rec in json.loads((CYCLE/'native_decoder_memory_v1/manifest.json').read_text())['records']:
        if rec['record']['role']!='validation':continue
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)
        for m in raw['metadata']:
            key=(m['record']['key'],m['view_index'])
            if key in lookup:memories['paired'][lookup[key]]=raw['memory'][m['view_index']].cuda()
    lookup_index={i:j for j,i in enumerate(indices)}
    for rec in json.loads((CYCLE/'single_view_validation_context_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);j=lookup_index[rec['image_index']];assert raw['metadata']==meta[j];memories['single'][j]=raw['memory'].cuda();tokens['single'][j]=raw['instance_tokens'].cuda()
    if args.order_control:
        memories['reversed']=[None]*n;tokens['reversed']=torch.zeros(n,6,256,device='cuda')
        for rec in json.loads((CYCLE/'reversed_view_validation_context_v1/manifest.json').read_text())['records']:
            assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);j=lookup_index[rec['image_index']];assert raw['metadata']==meta[j];memories['reversed'][j]=raw['memory'].cuda();tokens['reversed'][j]=raw['instance_tokens'].cuda()
    if args.full_context_control:
        memories.pop('paired');tokens.pop('paired')
        for c,prefix in [('shared','shared'),('full','fullshared')]:
            memories[c]=[None]*n;tokens[c]=torch.zeros(n,6,256,device='cuda')
            for rec in json.loads((CYCLE/f'{prefix}_view_validation_context_v1/manifest.json').read_text())['records']:
                assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);j=lookup_index[rec['image_index']];assert raw['metadata']==meta[j];memories[c][j]=raw['memory'].cuda();tokens[c][j]=raw['instance_tokens'].cuda()
    assert all(m is not None for items in memories.values() for m in items)
    original_single_manifest=json.loads((CYCLE/'single_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/original_single_manifest['golden_path'])==original_single_manifest['golden_sha256'];single_gold=torch.load(ROOT/original_single_manifest['golden_path'],map_location='cuda',weights_only=False)
    old={s:{} for s in [17,29,43]};new={}
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:
        for s in old:
            old[s]['paired']={k:torch.tensor(a[f'seed{s}__{k}'][indices],device='cuda') for k in single_gold[s]};old[s]['single']={k:v[indices] for k,v in single_gold[s].items()}
    with np.load(DET/'predictions.npz') as a:
        for s in old:
            for c in contexts:
                for arm in arms:new[(s,c,arm)]={k:torch.tensor(a[f'{arm}_seed{s}_{c}__{k}'][indices],device='cuda') for k in single_gold[s]}
    if args.order_control:
        rev=json.loads((CYCLE/'reversed_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/rev['golden_path'])==rev['golden_sha256'];rg=torch.load(ROOT/rev['golden_path'],map_location='cuda',weights_only=False)
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            for s in old:
                for c in ['paired','single']:old[s][c]={k:torch.tensor(a[f'mixed_seed{s}_{c}__{k}'][indices],device='cuda') for k in single_gold[s]}
                old[s]['reversed']={k:v[indices] for k,v in rg[s].items()}
    if args.full_context_control:
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            for s in old:old[s]['single']={k:torch.tensor(a[f'mixed_seed{s}_single__{k}'][indices],device='cuda') for k in single_gold[s]}
        for c,prefix in [('shared','shared'),('full','fullshared')]:
            cm=json.loads((CYCLE/f'{prefix}_detection_context_v1/manifest.json').read_text());assert fingerprint(ROOT/cm['golden_path'])==cm['golden_sha256'];cg=torch.load(ROOT/cm['golden_path'],map_location='cuda',weights_only=False)
            for s in old:old[s][c]={k:v[indices] for k,v in cg[s].items()}
    @torch.no_grad()
    def predict(model,context,boxes):
        parts=[None]*n;groups=collections.defaultdict(list)
        for j,memory in enumerate(memories[context]):groups[tuple(memory.shape[-2:])].append(j)
        for ids in groups.values():
            for start in range(0,len(ids),8):
                group=ids[start:start+8];p=model(tokens[context][group],torch.stack([memories[context][j] for j in group]),boxes[group])['xy']
                for k,j in enumerate(group):parts[j]=p[k]
        return torch.stack(parts)
    def evaluate(xy,det,old_matches):
        matches=match_detection(det,data['boxes'],valid);accum=collections.defaultdict(lambda:dict(error=[],same_query_error=[],points=0,pck16=0,pck32=0));detmetrics=collections.defaultdict(lambda:dict(images=0,tp=0,gt=0,predictions=0,query_switches=0))
        for i,m in enumerate(meta):
            kind='real' if m['record']['kind']=='real' else 'synthetic';cam=f'cam{chr(65+int(m["view"]))}';subtype='dual' if m['expected_instances']>1 else 'single';qs,ms=matches[i];oldq={int(m):int(q) for q,m in zip(*old_matches[i])};scores=det['instance_objectness_logits'][i].sigmoid();tp=0;switches=0
            for q,target in zip(qs.tolist(),ms.tolist()):
                hit=bool(scores[q]>=.5 and box_iou(det['instance_boxes_cxcywh'][i,q:q+1],data['boxes'][i,target:target+1])[0,0]>=.5);tp+=hit;switches+=int(q!=oldq[target])
                good=data['point_valid'][i,target].cpu();err=((xy[i,q]-data['gt_xy'][i,target])*data['pixels'][i]).norm(dim=-1).cpu();same=((xy[i,oldq[target]]-data['gt_xy'][i,target])*data['pixels'][i]).norm(dim=-1).cpu()
                reliable=good&torch.tensor([k in [0,14,15,16,17] if kind=='real' else k not in [5,6] for k in range(23)]);groups={'all':good,'reliable':reliable,subtype:good}
                if kind=='real':groups[cam+'/feet']=good&torch.tensor([k in [14,15,16,17] for k in range(23)])
                for group,mask in groups.items():
                    a=accum[kind+'/'+group];values=err[mask].tolist();a['error']+=values;a['same_query_error']+=same[mask].tolist();a['points']+=len(values)
                    if hit:a['pck16']+=sum(v<=16 for v in values);a['pck32']+=sum(v<=32 for v in values)
            for group in [kind,kind+'/'+subtype]+([kind+'/'+cam] if kind=='real' else []):
                a=detmetrics[group];a['images']+=1;a['tp']+=tp;a['gt']+=len(ms);a['predictions']+=int((scores>=.5).sum());a['query_switches']+=switches
        output={}
        for group,a in accum.items():
            for name in ['error','same_query_error']:
                values=a.pop(name);a[name+'_mean']=float(np.mean(values)) if values else None
            a['gated_pck16']=a['pck16']/max(a['points'],1);a['gated_pck32']=a['pck32']/max(a['points'],1);output[group]=a
        for a in detmetrics.values():a.update(precision=a['tp']/max(a['predictions'],1),recall=a['tp']/max(a['gt'],1))
        return output,dict(detmetrics)
    write_json(OUT/'protocol.json',dict(source_sha256=fingerprint(source),detector_predictions_sha256=fingerprint(DET/'predictions.npz'),pose='Frozen view_context_control_v1/mixed_seed{17,29,43}_latest.pt',cases='112 validation images, original paired and separatelyreencoded single contexts. All6queries predicted first, GTboxes/objectness association only in evaluation. Detectorarms baseline/paired/mixed. New predictedboxes provide readoutprior; never GTboxes.',decomposition='error_mean uses currentdetector assignment; same_query_error_mean uses OLDdetector association with newpredictedbox prior, to separate queryswitch from prior effect. Neither is end-to-end detection. GatedPCK counts missed detections but not false-positive-point penalty; detector precision separately.',script_sha256=fingerprint(Path(__file__)),production_changed=False))
    results=[];saved={};max_delta=0.
    if args.order_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(pose='Frozen view_order_control_v1/random_ref800 pose, same all detectorarms',cases='112validation images in single/paired/reversed contexts. Baseline mixed800detector vs matched fixed/random_ref detector continuation. All138posequeries before GT association.',pose_predictions_sha256=fingerprint(pose_dir/'predictions.npz'));write_json(OUT/'protocol.json',protocol)
    if args.full_context_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(pose='Frozen full_context_pose_control_v1/full800 pose, sameall detectorarms.',cases='112validation images eachsharedpair/single/eligiblefull context. Baseline mixed800detector vs paired/full context800continuation; originalvieworder no referenceaugmentation. Full138queries. Syntheticfullcache2views only.',pose_predictions_sha256=fingerprint(pose_dir/'predictions.npz'));write_json(OUT/'protocol.json',protocol)
    for seed in [17,29,43]:
        cp_path=pose_dir/f'{pose_arm}_seed{seed}_latest.pt';cp=torch.load(cp_path,map_location='cuda',weights_only=False);assert cp['step']==800 and cp['seed']==seed and cp['arm']==pose_arm;model=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);model.load_state_dict(cp['state'],strict=True)
        for context in contexts:
            old_matches=match_detection(old[seed][context],data['boxes'],valid)
            for arm in ['baseline']+arms:
                assert dt.datetime.now(dt.timezone.utc)<deadline
                det=old[seed][context] if arm=='baseline' else new[(seed,context,arm)]
                with torch.no_grad():xy=predict(model,context,det['instance_boxes_cxcywh'])
                if arm=='baseline':
                    delta=float((xy.cpu()-torch.tensor(gold[seed][context])).abs().max());assert delta<2e-5;max_delta=max(max_delta,delta)
                metrics,detmetrics=evaluate(xy,det,old_matches);row=dict(seed=seed,context=context,arm=arm,metrics=metrics,detector=detmetrics,pose_checkpoint_sha256=fingerprint(cp_path));results.append(row);saved[f'{arm}_seed{seed}_{context}']=xy.cpu().numpy();write_json(OUT/'partial_results.json',results);status('auditing',seed=seed,context=context,arm=arm);print('DETPOSE',seed,context,arm,metrics['real/reliable'],detmetrics['real'],flush=True)
    np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(indices),**saved);write_json(OUT/'results.json',dict(results=results,max_baseline_golden_delta=max_delta,no_production_promotion=True));status('complete')
    state.update(stage='detection_context_frozen_pose_audit_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Review detector3seeds and frozenpose downstream metrics jointly, including queryswitches, real/syntheticdual recall and rawconfidence overlays. If no stable improvement, stop similar detector probes rather than more steps. Any promising candidate requires RGB1/2/allintegration parity and visualreview, no automaticpromotion.');state['completed'].append('detection_context_frozen_pose_audit');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
