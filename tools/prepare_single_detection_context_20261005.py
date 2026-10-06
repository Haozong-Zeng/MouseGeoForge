"""Reproduce single-context ROI inputs and fixed detector goldens, no GT crops."""
import argparse,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from torch.nn import functional as F
from mousegeoforge.models.instance_detection_readout import InstanceDetectionReadout
from mousegeoforge.models.spatial_detection_readout import SpatialDetectionReadout,sample_predicted_roi
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'single_detection_context_v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--reversed',action='store_true');parser.add_argument('--context',choices=['single','reversed','shared','fullshared'],default='single');args=parser.parse_args()
    if args.reversed:
        assert args.context in ['single','reversed'];args.context='reversed'
    args.reversed=args.context!='single' # Legacy path selects mixed800 goldens for all newer contexts.
    OUT=CYCLE/f'{args.context}_detection_context_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='preparing_single_detector_context',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');source=CYCLE/'detection_features_v1/manifest.json';manifest=json.loads(source.read_text());rows=[]
    for rec in manifest['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    meta=[o['metadata'] for o in rows];n=len(rows);assert n==1350
    base_path=CYCLE/'detection_features_v1/initial_detector.pt';assert fingerprint(base_path)==manifest['initial_detector_sha256']
    base=InstanceDetectionReadout().cuda().eval().requires_grad_(False);base.load_state_dict(torch.load(base_path,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad():
        replay=base(torch.stack([o['instance_tokens'] for o in rows]).cuda())
        for k,original in [('instance_boxes_cxcywh','original_boxes'),('instance_objectness_logits','original_objectness')]:torch.testing.assert_close(replay[k].cpu(),torch.stack([o[original] for o in rows]),atol=2e-5,rtol=0)
    heads={};head_hash={}
    for seed in [17,29,43]:
        path=CYCLE/f'detection_context_roi_v1/seed{seed}_latest.pt';cp=torch.load(path,map_location='cuda',weights_only=False);assert cp['step']==1600 and cp['seed']==seed
        if args.reversed:
            path=CYCLE/f'detection_view_context_control_v1/mixed_seed{seed}_latest.pt';cp=torch.load(path,map_location='cuda',weights_only=False);assert cp['step']==800 and cp['seed']==seed and cp['arm']=='mixed'
        heads[seed]=SpatialDetectionReadout().cuda().eval().requires_grad_(False);heads[seed].load_state_dict(cp['state'],strict=True);head_hash[seed]=fingerprint(path)
    data=dict(instance_tokens=torch.zeros(n,6,256),roi=torch.zeros(n,6,25,256,dtype=torch.float16),global_memory=torch.zeros(n,256,dtype=torch.float16),original_boxes=torch.zeros(n,6,4),original_objectness=torch.zeros(n,6),available=torch.zeros(n,dtype=torch.bool),metadata=meta)
    max_delta={s:0. for s in heads};gold={s:dict(instance_boxes_cxcywh=torch.zeros(n,6,4),instance_objectness_logits=torch.zeros(n,6)) for s in heads};sources={};done=0
    for role in ['train','validation']:
        path=CYCLE/f'{args.context}_view_{role}_context_v1/manifest.json';single=json.loads(path.read_text());assert single['role']==role and single['source_sha256']==fingerprint(source);sources[role]=fingerprint(path)
        for rec in single['records']:
            assert dt.datetime.now(dt.timezone.utc)<deadline
            i=rec['image_index'];assert meta[i]['record']['role']==role and meta[i]['record'].get('session') not in ['GR2','UR2','UR3'] and not data['available'][i]
            assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==meta[i]
            tokens=raw['instance_tokens'][None].cuda()
            with torch.no_grad():
                p=base(tokens)
                if args.reversed:
                    torch.testing.assert_close(p['instance_boxes_cxcywh'][0].cpu(),raw['original_boxes'],atol=2e-5,rtol=0)
                    torch.testing.assert_close(p['instance_objectness_logits'][0].cpu(),raw['original_objectness'],atol=2e-5,rtol=0)
                    # ROI sampling must use the exact stored PREDICTED boxes.
                    # Recomputing a7-view batched MLP as a singleton changes low
                    # bits and can cross an fp16 ROI rounding boundary. These
                    # are original model outputs, never annotation/GT boxes.
                    p={'instance_boxes_cxcywh':raw['original_boxes'][None].cuda(),'instance_objectness_logits':raw['original_objectness'][None].cuda()}
                memory=F.interpolate(raw['memory'][None].cuda().half().float(),(48,48),mode='bilinear',align_corners=False);roi,glob=sample_predicted_roi(memory,p['instance_boxes_cxcywh'])
                for seed,head in heads.items():
                    current=head(tokens,roi.half().float(),glob.half().float(),p['instance_boxes_cxcywh'],p['instance_objectness_logits'])
                    for k,v in current.items():
                        delta=float((v[0].cpu()-raw['adapted'][seed][k]).abs().max());assert delta<2e-5,(i,seed,k,delta);max_delta[seed]=max(max_delta[seed],delta);gold[seed][k][i]=v[0].cpu()
            data['instance_tokens'][i]=raw['instance_tokens'];data['roi'][i]=roi[0].half().cpu();data['global_memory'][i]=glob[0].half().cpu();data['original_boxes'][i]=p['instance_boxes_cxcywh'][0].cpu();data['original_objectness'][i]=p['instance_objectness_logits'][0].cpu();data['available'][i]=True;done+=1
            if done%100==0:status('preparing',completed=done,total=1335)
    assert done==1335
    path=OUT/'features.pt';torch.save(data,path);goldpath=OUT/'golden.pt';torch.save(gold,goldpath)
    write_json(OUT/'manifest.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),golden_path=goldpath.relative_to(ROOT).as_posix(),golden_sha256=fingerprint(goldpath),images=done,rows=n,source_sha256=fingerprint(source),single_sources=sources,initial_detector_sha256=fingerprint(base_path),roi_initial_sha256=head_hash,max_golden_delta=max_delta,script_sha256=fingerprint(Path(__file__)),contract='Same single-view tokens/native FP32 memories. Original box/objectness replay from immutable original head; noGT boxes used in feature extraction. Native memory half roundtrip then48x48, prediction-only ROI, then half roundtrip. Incomplete rows unavailable, NOT negatives. Role-separated source manifests.'))
    if args.reversed:
        manifest_path=OUT/'manifest.json';record=json.loads(manifest_path.read_text());record.update(context='reversed original2view',golden_detector='detection_view_context_control_v1/mixed800',contract='Actual reversed2view original-special-token memories; same1335eligibleimages, noGTcrop, immutable original box/objectness replay and mixed800ROI goldens checked. Native FP32->half->48x48->ROIhalf input path. Incomplete rows unavailable, not negatives.');write_json(manifest_path,record)
        if args.context in ['shared','fullshared']:
            record.update(context=args.context,contract='Actual '+args.context+' encoder memories, same1335eligibleimages, noGTcrop. Sharedfirstspecialtokens were used onlyin2Dcache; originalgeometry separate. Immutableoriginal box/objectness replay and mixed800ROI3seedgoldens checked. NativeFP32->half->48x48->ROIhalf; incomplete rows unavailable notnegatives.');write_json(manifest_path,record)
    status('complete',images=done,max_golden_delta=max_delta);state.update(stage='single_detector_context_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Single ROI input goldens passed. Train matched paired vs mixed detection800steps3seeds using same accepted image schedules as view_context_control_v1, freeze pose/encoder/instance decoder. Inspect coverage/duplicates/syntheticdual, then downstream pose using predictednewboxes. No promotion.');state['completed'].append('single_detector_context_exact_input_audit');write_json(CYCLE/'state.json',state)
    if args.reversed:
        state.update(stage='reversed_detector_context_ready',next='Run matched detector reference-order control800steps3seeds. Pose fixed; original encoder tokens unchanged. See plan.');state['completed'].append('reversed_detector_context_exact_input_audit');write_json(CYCLE/'state.json',state)
        if args.context in ['shared','fullshared']:
            state.update(stage=args.context+'_detector_context_ready',next='Follow full-context detector plan; shared/fullshared ROIinputs verified againstsamefrozenmixed800goldens. No traininglaunched byextractor.');state['completed'][-1]=args.context+'_detector_context_exact_input_audit';write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
