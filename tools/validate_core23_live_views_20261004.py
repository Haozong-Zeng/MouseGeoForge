"""Full-query default-off integration parity and 1/2/all-view development audit."""
import argparse,collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from torch import nn
from vggt_omega.models import VGGTOmega
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models.spatial_readout_pose import SpatialReadoutDepthPoseModel
from mousegeoforge.models.view_token_probe import shared_first_view_tokens
from mousegeoforge.models.independent_view_pose_probe import independent_view_pose_probe,POSE_KEYS
from mousegeoforge.models.shared_token_pose_probe import shared_token_pose_probe
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from training.finetune_blendedmvs import load_scene
from run_surface_curve_pilot_20261003 import load_item
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json,move_batch
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OLD=ROOT/'output/surface_curve_cycle_20261003'
OUT=CYCLE/'live_view_audit_v2';READOUT=CYCLE/'readout_post4800_control_v1/spatial_seed17_latest.pt'


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p90=float(np.quantile(values,.9))) if values else dict(n=0)


def main():
    global OUT,READOUT
    parser=argparse.ArgumentParser();parser.add_argument('--detector-roi',action='store_true');parser.add_argument('--decoder-adapted',choices=['uniform','balanced']);parser.add_argument('--context-control',choices=['paired','mixed']);parser.add_argument('--detector-context',choices=['paired','mixed']);parser.add_argument('--permutation-audit',action='store_true');parser.add_argument('--shared-first-tokens',action='store_true');parser.add_argument('--reference-control',choices=['fixed','random_ref']);parser.add_argument('--reference-detector',choices=['fixed','random_ref']);parser.add_argument('--independent-pose',action='store_true');parser.add_argument('--shared-pose-control',choices=['original','shared']);parser.add_argument('--full-context-control',choices=['paired','full']);parser.add_argument('--full-context-detector',choices=['paired','full']);args=parser.parse_args()
    if args.full_context_detector:
        assert args.full_context_control in [None,'full'];args.full_context_control='full'
    if args.full_context_control:
        assert not args.shared_pose_control;args.shared_pose_control=args.full_context_control
    if args.shared_pose_control:
        assert not any([args.shared_first_tokens,args.reference_control,args.reference_detector,args.independent_pose]);assert args.detector_context in [None,'mixed'];args.detector_context='mixed';args.permutation_audit=True
    if args.independent_pose:
        assert args.reference_detector in [None,'random_ref'];args.reference_detector='random_ref'
    if args.reference_detector:
        assert args.reference_control in [None,'random_ref'];args.reference_control='random_ref'
    if args.reference_control:
        assert not args.shared_first_tokens and args.detector_context in [None,'mixed'];args.detector_context='mixed';args.permutation_audit=True
    if args.shared_first_tokens:assert args.permutation_audit,'Probe requires the fixed order-audit protocol'
    if args.permutation_audit:assert args.detector_context=='mixed','Order audit fixed to mixed pose+mixed detector candidate'
    if args.detector_context:
        assert args.context_control in [None,'mixed'];args.context_control='mixed'
    assert not (args.context_control and args.decoder_adapted)
    if args.context_control:args.decoder_adapted='context_'+args.context_control
    if args.decoder_adapted:args.detector_roi=True
    if args.detector_roi:OUT=CYCLE/'live_view_roi_detector_audit_v1'
    if args.decoder_adapted:
        arm='balanced' if args.decoder_adapted=='balanced' else 'unfrozen'
        decoder_dir=CYCLE/('bottom_balanced_extended2400_v1' if arm=='balanced' else 'decoder_unfrozen_extended2400_v1')
        READOUT=decoder_dir/f'{arm}_seed17_latest.pt'
        OUT=CYCLE/f'live_view_decoder_{args.decoder_adapted}2400_audit_v1'
        if args.context_control:
            arm=args.context_control;decoder_dir=CYCLE/'view_context_control_v1';READOUT=decoder_dir/f'{arm}_seed17_latest.pt';OUT=CYCLE/f'live_view_context_{arm}800_audit_v1'
        if args.detector_context:OUT=CYCLE/f'live_view_detector_context_{args.detector_context}800_audit_v1'
        if args.permutation_audit:OUT=CYCLE/'live_view_order_mixed800_audit_v1'
        if args.shared_first_tokens:OUT=CYCLE/'live_view_shared_first_tokens_audit_v1'
        if args.reference_control:
            arm=args.reference_control;decoder_dir=CYCLE/'view_order_control_v1';READOUT=decoder_dir/f'{arm}_seed17_latest.pt';OUT=CYCLE/f'live_view_reference_{arm}800_audit_v1'
        if args.reference_detector:OUT=CYCLE/f'live_view_reference_detector_{args.reference_detector}800_audit_v1'
        if args.independent_pose:OUT=CYCLE/'live_view_independent_pose_audit_v1'
        if args.shared_pose_control:
            arm=args.shared_pose_control;decoder_dir=CYCLE/'shared_token_pose_control_v1';READOUT=decoder_dir/f'{arm}_seed17_latest.pt';OUT=CYCLE/f'live_view_shared_pose_{arm}800_audit_v1'
        if args.full_context_control:
            arm=args.full_context_control;decoder_dir=CYCLE/'full_context_pose_control_v1';READOUT=decoder_dir/f'{arm}_seed17_latest.pt';OUT=CYCLE/f'live_view_full_context_{arm}800_audit_v1'
    if args.full_context_detector:OUT=CYCLE/f'live_view_full_detector_{args.full_context_detector}800_audit_v1'
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'predictions').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=True
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='validating_full_query_live_view_integration',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    initial=OLD/'training_v1/pilot/initial_heads.pt';manifest=json.loads((CYCLE/'features_v1/manifest.json').read_text())
    assert fingerprint(initial)==manifest['initial_sha256']
    cp=torch.load(READOUT,map_location='cpu',weights_only=False)
    assert cp['seed']==17
    if args.decoder_adapted:assert cp['step']==(800 if args.context_control else 2400) and cp['arm']==arm
    else:assert cp['step']==6400 and cp['mode']=='spatial'
    model=SpatialReadoutDepthPoseModel(nn.Identity(),DenseHead(),MultiInstanceMouseHead(num_keypoints=23)).cuda().eval().requires_grad_(False)
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False);assert not result.unexpected_keys and all(k.startswith(('spatial_readout.','spatial_detector.')) for k in result.missing_keys)
    if args.decoder_adapted:
        consumed=set()
        for prefix,module in [('keypoint_decoder',model.instance_pose_head.keypoint_decoder),('keypoint_queries',model.instance_pose_head.keypoint_queries),('coarse_keypoint_xy',model.instance_pose_head.coarse_keypoint_xy),('spatial_readout',model.spatial_readout)]:
            selected={k[len(prefix)+1:]:v for k,v in cp['state'].items() if k.startswith(prefix+'.')}
            module.load_state_dict(selected,strict=True);consumed.update(prefix+'.'+k for k in selected)
        assert consumed==set(cp['state'])
    else:model.spatial_readout.load_state_dict(cp['state'],strict=True)
    detector_path=CYCLE/'detection_context_roi_v1/seed17_latest.pt';detector_cases=pose_cases=0;max_detector=0.
    if args.detector_context:detector_path=CYCLE/f'detection_view_context_control_v1/{args.detector_context}_seed17_latest.pt'
    if args.reference_detector:detector_path=CYCLE/f'detection_reference_control_v1/{args.reference_detector}_seed17_latest.pt'
    if args.detector_roi:
        detector_cp=torch.load(detector_path,map_location='cpu',weights_only=False);assert detector_cp['seed']==17 and detector_cp['step']==(800 if args.detector_context else 1600)
        if args.detector_context:assert detector_cp['arm']==(args.reference_detector or args.detector_context)
        model.spatial_detector.load_state_dict(detector_cp['state'],strict=True);model.spatial_detector_enabled=True
        golden_rows=[]
        for r in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:
            assert fingerprint(ROOT/r['path'])==r['sha256'];golden_rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
        lookup={(o['metadata']['record']['key'],o['metadata']['view_index']):i for i,o in enumerate(golden_rows)}
        with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:detector_golden={k:a['seed17__'+k].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
        with np.load(CYCLE/'detection_pose_context_roi_audit_v1/predictions.npz') as a:pose_golden={int(i):a['seed17'][j,1].copy() for j,i in enumerate(a['image_indices'])}
        if args.decoder_adapted:
            with np.load(decoder_dir/'predictions.npz') as a:
                pose_golden={int(i):a[f'{arm}_seed17'+('_paired' if args.context_control else '')][j].copy() for j,i in enumerate(a['image_indices'])}
                if args.context_control:single_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):a[f'{arm}_seed17_single'][j].copy() for j,i in enumerate(a['image_indices'])}
        if args.detector_context:
            with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:detector_golden={k:a[f'{args.detector_context}_seed17_paired__{k}'].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
            if not args.reference_control and not args.shared_pose_control:
                with np.load(CYCLE/'detection_context_pose_audit_v1/predictions.npz') as a:
                    pose_golden={int(i):a[f'{args.detector_context}_seed17_paired'][j].copy() for j,i in enumerate(a['image_indices'])}
                    single_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):a[f'{args.detector_context}_seed17_single'][j].copy() for j,i in enumerate(a['image_indices'])}
        if args.reference_detector:
            with np.load(CYCLE/'detection_reference_control_v1/predictions.npz') as a:detector_golden={k:a[f'{args.reference_detector}_seed17_paired__{k}'].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
            with np.load(CYCLE/'detection_reference_pose_audit_v1/predictions.npz') as a:
                pose_golden={int(i):a[f'{args.reference_detector}_seed17_paired'][j].copy() for j,i in enumerate(a['image_indices'])}
                single_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):a[f'{args.reference_detector}_seed17_single'][j].copy() for j,i in enumerate(a['image_indices'])}
        if args.shared_pose_control:
            with np.load(decoder_dir/'predictions.npz') as a:shared_pose_golden={(golden_rows[int(i)]['metadata']['record']['key'],int(golden_rows[int(i)]['metadata']['view'])):a[f'{arm}_seed17_shared'][j].copy() for j,i in enumerate(a['image_indices'])}
        if args.full_context_control:
            full_records={r['image_index']:r for r in json.loads((CYCLE/'fullshared_view_validation_context_v1/manifest.json').read_text())['records']}
            with np.load(decoder_dir/'predictions.npz') as a:full_pose_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):dict(xy=a[f'{arm}_seed17_full'][j].copy(),views=full_records[int(i)]['record']['views']) for j,i in enumerate(a['image_indices'])}
    with np.load(CYCLE/'readout_post4800_control_v1/predictions.npz') as a:reference=a['spatial_seed17'].copy()
    index=0;max_new=0.;max_original=0.;off_checks=0
    for n,r in enumerate(manifest['records']):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        saved=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);assert fingerprint(ROOT/r['path'])==r['sha256']
        item=torch.load(OLD/f'training_v1/pilot/cache/{r["key"]}.pt',map_location='cpu',weights_only=False)
        images=item['batch']['images'].cuda();cached=[None if x is None else x.cuda() for x in item['cached']]
        model.spatial_pose_enabled=True
        with torch.no_grad():pred=model.forward_cached(images,cached,item['patch_start'],coupling='off')
        if args.detector_roi:
            for v in range(images.shape[1]):
                key=(r['key'],v)
                if key not in lookup:continue
                i=lookup[key]
                for name in detector_golden:
                    delta=float((pred[name][0,v].cpu()-torch.tensor(detector_golden[name][i])).abs().max());assert delta<2e-5,(key,name,delta);max_detector=max(max_detector,delta)
                detector_cases+=1
                if i in pose_golden:
                    delta=float((pred['instance_keypoint_xy'][0,v].cpu()-torch.tensor(pose_golden[i])).abs().max());assert delta<2e-5,(key,delta);max_new=max(max_new,delta);pose_cases+=1
            if n in [0,132,150,180]:
                model.spatial_pose_enabled=model.spatial_detector_enabled=False
                with torch.no_grad():off=model.forward_cached(images,cached,item['patch_start'],coupling='off')
                for key,value in off.items():
                    reference_key={'instance_keypoint_xy':'instance_keypoint_original_xy','instance_boxes_cxcywh':'instance_boxes_original_cxcywh','instance_objectness_logits':'instance_objectness_original_logits'}.get(key,key)
                    torch.testing.assert_close(pred[reference_key],value,atol=0,rtol=0)
                off_checks+=1;model.spatial_detector_enabled=True
            index+=len(saved['observations']);status('parity',completed=n+1,total=len(manifest['records']));continue
        if n in [0,132,150,180]:
            model.spatial_pose_enabled=False
            with torch.no_grad():off=model.forward_cached(images,cached,item['patch_start'],coupling='off')
            assert set(pred)==set(off)|{'instance_keypoint_original_xy'}
            for k,v in off.items():torch.testing.assert_close(pred['instance_keypoint_original_xy' if k=='instance_keypoint_xy' else k],v,atol=0,rtol=0)
            off_checks+=1
        for o in saved['observations']:
            m=o['metadata'];v,q=m['view_index'],m['query']
            delta=float((pred['instance_keypoint_xy'][0,v,q].cpu()-torch.tensor(reference[index])).abs().max());assert delta<2e-5,(index,delta);max_new=max(max_new,delta)
            delta=float((pred['instance_keypoint_original_xy'][0,v,q].cpu()-o['original_xy']).abs().max());assert delta<2e-4;max_original=max(max_original,delta);index+=1
        status('parity',completed=n+1,total=len(manifest['records']))
    assert index==393
    if args.detector_roi:assert detector_cases>100 and pose_cases>50
    write_json(OUT/'parity.json',dict(observations=index,max_new_delta=max_new,max_original_delta=max_original,default_off_exact_cases=off_checks,all6queries_forward=True,detector_cases=detector_cases,pose_cases=pose_cases,max_detector_delta=max_detector,detector_sha256=fingerprint(detector_path) if args.detector_roi else None,readout_sha256=fingerprint(READOUT),initial_sha256=fingerprint(initial),module_sha256=fingerprint(ROOT/'mousegeoforge/models/spatial_readout_pose.py')))
    # The original-token gate above deliberately uses the baseline mixed800 detector.
    # Only now switch to the new detector, whose goldens are actual single/shared/full contexts.
    candidate_detection_goldens={};candidate_detector_cases=0;max_candidate_detector_delta=0.
    if args.full_context_detector:
        detector_path=CYCLE/f'detection_full_context_control_v1/{args.full_context_detector}_seed17_latest.pt'
        detector_cp=torch.load(detector_path,map_location='cpu',weights_only=False)
        assert detector_cp['seed']==17 and detector_cp['step']==800 and detector_cp['arm']==args.full_context_detector
        model.spatial_detector.load_state_dict(detector_cp['state'],strict=True)
        with np.load(CYCLE/'detection_full_context_pose_audit_v1/predictions.npz') as a:
            ids=a['image_indices'];prefix=f'{args.full_context_detector}_seed17_'
            single_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):a[prefix+'single'][j].copy() for j,i in enumerate(ids)}
            shared_pose_golden={(golden_rows[int(i)]['metadata']['record']['key'],int(golden_rows[int(i)]['metadata']['view'])):a[prefix+'shared'][j].copy() for j,i in enumerate(ids)}
            full_pose_golden={(golden_rows[int(i)]['metadata']['record']['scene'],int(golden_rows[int(i)]['metadata']['view'])):dict(xy=a[prefix+'full'][j].copy(),views=full_records[int(i)]['record']['views']) for j,i in enumerate(ids)}
        with np.load(CYCLE/'detection_full_context_control_v1/predictions.npz') as a:
            for context in ['single','shared','full']:
                for i in ids:
                    m=golden_rows[int(i)]['metadata'];key=(context,m['record']['key'] if context=='shared' else m['record']['scene'],int(m['view']))
                    candidate_detection_goldens[key]={name:a[f'{args.full_context_detector}_seed17_{context}__{name}'][int(i)].copy() for name in ['instance_boxes_cxcywh','instance_objectness_logits']}
    del cached,images,pred,item,saved
    status('loading_encoder')
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';raw=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
    selected={k[len('aggregator.'):]:v for k,v in raw.items() if k.startswith('aggregator.')};spatial.aggregator.load_state_dict(selected,strict=True);del raw,selected
    model.aggregator=spatial.aggregator.cuda().eval().requires_grad_(False);del spatial
    # Original development scenes only. No locked sessions or new split choices.
    real={};synthetic=collections.defaultdict(dict)
    for rec in manifest['records']:
        r=rec['record']
        if r['role']!='validation':continue
        if r['kind']=='real':real.setdefault(r['scene'],r)
        else:
            group=r.get('group','synthetic');synthetic[group].setdefault(r['scene'],r)
    chosen=list(real.values())+[r for group in sorted(synthetic) for r in list(synthetic[group].values())[:2]]
    tasks=[]
    for r in chosen:
        assert r.get('session') not in ('GR2','UR2','UR3')
        available=sorted(load_scene(ROOT/r['scene']).frames);anchor=r['views'][0]
        variants=[('single',[anchor]),('paired',r['views']),('all',available)]
        if args.permutation_audit:variants.extend([('paired_reversed',list(reversed(r['views']))),('all_reversed',list(reversed(available)))])
        for variant,views in variants:
            tasks.append(dict(record=dict(r,views=views),variant=variant,anchor_view=anchor))
    write_json(OUT/'protocol.json',dict(tasks=tasks,readout_sha256=fingerprint(READOUT),encoder_sha256=fingerprint(encoder),
        inference='All6queries from RGB only. Optional ROI residual detector then frozen spatial pose with adapted predicted box prior. No GT selecting queries/crops. GT Hungarian assignment only AFTER inference for metrics.' if args.detector_roi else 'All6queries from RGB only. Frozen detector outputs unchanged. No GT selecting readout queries. GT box/objectness Hungarian assignment only AFTER inference for metrics.',
        detector_sha256=fingerprint(detector_path) if args.detector_roi else None,
        detector_metric='Objectness>=.5 and matchedboxIoU>=.5 without NMS. Count coverage/precision separately; not AP. Keypoint metrics remain association-assisted and must not be called end-to-end pose accuracy.',
        views='Compare same anchor view under1/2/all available contexts, plus all-view aggregate. Original development scenes, not external test.',
        numeric='bf16encoder cuDNNTF32false; fp32headTF32true; matmulTF32false; readoutfp16memoryroundtrip then48x48. Full-model wrapper disabled by default.',
        production_changed=False))
    if args.decoder_adapted:
        protocol=json.loads((OUT/'protocol.json').read_text())
        protocol.update(decoder_adapted=args.decoder_adapted,decoder_steps=cp['step'],seed=17,
            inference='All6queries from RGB, fixed ROI detector, adapted full138-query keypoint decoder and spatial readout. No GT crops or GT counts. GT assignment only for evaluation.',
            legacy_comparator='Local-refined outputs now use the adapted decoder, NOT the immutable original head. Default-off parity compares both modes with the SAME adapted weights. Compare previous audit files for original-decoder results.',
            selection='Seed17 fixed in earlier live audits; not selected from the three seeds by validation performance.')
        write_json(OUT/'protocol.json',protocol)
    if args.detector_context:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(detector_context=args.detector_context,detector_steps=800,pose='Frozen mixed-context800 pose, not refit to new detector assignments. Readout uses new PREDICTEDboxes.',inference='All6queries from RGB, context-adapted ROI detector and frozen mixed800pose. No GTcrop/count/query selection; GT assignment only for metrics.');write_json(OUT/'protocol.json',protocol)
    if args.permutation_audit:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(order_audit='Same RGB sets, reversed input order for paired/all; outputs compared by physical camera view IDs, not array positions. Same fixed anchor chosen BEFORE permutation. First-frame versus other-frame tokens in local aggregator may cause reference-order sensitivity; this is a hypothesis, not a measured conclusion yet.',aggregator_source_sha256=fingerprint(ROOT/'vggt-omega/vggt_omega/models/aggregator.py'));write_json(OUT/'protocol.json',protocol)
    if args.shared_first_tokens:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(shared_token_probe='Runtime-only camera/register other-frame slots copied from first-frame slot during each forward, restored afterward. No gradient/training/checkpoint writes. Single-view is unchanged control. Accuracy and order-symmetry are separate endpoints; no promotion.',probe_source_sha256=fingerprint(ROOT/'mousegeoforge/models/view_token_probe.py'));write_json(OUT/'protocol.json',protocol)
    if args.reference_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(reference_control=args.reference_control,pose='Adapted '+args.reference_control+' reference-order pose800; identical frozenmixed800detector. Originalencoder specialtokens untouched. Same60 original/reversed tasks. No GTorderselection.');write_json(OUT/'protocol.json',protocol)
    if args.reference_detector:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(reference_detector=args.reference_detector,detector_checkpoint_sha256=fingerprint(detector_path),pose='Frozen random_ref800 pose, '+args.reference_detector+' reference-order detector800. Original pretrained special tokens, same60tasks. No GTforward.');write_json(OUT/'protocol.json',protocol)
    if args.independent_pose:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(independent_pose='Inference-only six2Dfield diagnostic. Same random_ref800 pose+detector checkpoints, independent V1 pass per physicalcamera. Original multiview geometry output preserved by identity. Auxiliary pose/curve fields not reconciled and MUST NOT be used as joint-training output. No crossviewidentity association. Extra sequential encoder passes: not a compute-matched architecture comparison.',probe_sha256=fingerprint(ROOT/'mousegeoforge/models/independent_view_pose_probe.py'));write_json(OUT/'protocol.json',protocol)
    if args.shared_pose_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(shared_pose_control=args.shared_pose_control,pose='Shared-token2D branch with '+arm+'800 trainedpose andsamefrozenmixed800ROI detector. Original multiviewgeometry retainedbyidentity fromoriginaltokenpass; replaceonlysix2Dfields. Auxpose/curvefields unreconciled, notjointtrainingoutput. Extraencoderpass, notcompute-matched. Singleview noextra pass. No GTforward orcrossviewidentity claim.',probe_sha256=fingerprint(ROOT/'mousegeoforge/models/shared_token_pose_probe.py'));write_json(OUT/'protocol.json',protocol)
    if args.full_context_control:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(full_context_control=args.full_context_control,pose_training='Matched paired/full800 fromshared800pose. Both50percent actualsingle; treatment25percent actualeligiblefullcontexts. Same mixed800ROI frozen. Real5/7fullRGB goldens compared onlywhenorderedcontext matchesfullcache. Synthetic4/8fullRGB untrained stress-only.');write_json(OUT/'protocol.json',protocol)
    if args.full_context_detector:
        protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(full_context_detector=args.full_context_detector,detector_checkpoint_sha256=fingerprint(detector_path),pose='Frozen full-context full800 pose for baseline/paired/full detector comparisons. Original-token cached gate uses baseline mixed800 detector, then candidate detector loaded. Actual RGB single/shared/full pose AND detector outputs must match candidate cache goldens. Original geometry retained; auxiliary fields unreconciled. No production change.',parity_scope='parity.json is the baseline-detector integration gate; results.json includes candidate RGB parity.',pose_training='Same frozen full800 pose; only detector checkpoint differs from live_view_full_context_full800_audit_v1. Synthetic4/8view remains untrained stress.');write_json(OUT/'protocol.json',protocol)
    reports=[];model.spatial_pose_enabled=True;single_cases=0;max_single_delta=0.
    token_snapshot={name:getattr(model.aggregator,name).detach().clone() for name in ['camera_token','register_token']};single_full_control=False;shared_cases=0;max_shared_delta=0.;full_cases=0;max_full_delta=0.
    for n,task in enumerate(tasks):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        r=task['record'];item=load_item(r);images=item['batch']['images'].cuda()
        if args.shared_first_tokens and n==0:
            with torch.no_grad():single_control=model(images,coupling='off')
        with torch.no_grad(),shared_first_view_tokens(model.aggregator,enabled=args.shared_first_tokens):pred=model(images,coupling='off')
        if args.shared_first_tokens:
            for name,original in token_snapshot.items():torch.testing.assert_close(getattr(model.aggregator,name),original,atol=0,rtol=0)
            if n==0:
                assert task['variant']=='single' and set(pred)==set(single_control)
                for key in pred:torch.testing.assert_close(pred[key],single_control[key],atol=0,rtol=0)
                single_full_control=True;del single_control
        if args.independent_pose:
            multiview=pred
            with torch.no_grad():pred=independent_view_pose_probe(images,multiview,lambda x:model(x,coupling='off'))
            for key in multiview:
                if key not in POSE_KEYS:assert pred[key] is multiview[key]
            if images.shape[1]==1:
                for key in multiview:torch.testing.assert_close(pred[key],multiview[key],atol=0,rtol=0)
        if args.shared_pose_control:
            multiview=pred;pred=shared_token_pose_probe(images,multiview,model)
            for key in multiview:
                if key not in POSE_KEYS:assert pred[key] is multiview[key]
            for name,value in token_snapshot.items():torch.testing.assert_close(getattr(model.aggregator,name),value,atol=0,rtol=0)
            if task['variant']=='single':
                for key in multiview:torch.testing.assert_close(pred[key],multiview[key],atol=0,rtol=0)
            if task['variant']=='paired':
                for v,view in enumerate(r['views']):
                    key=(r['key'],int(view));assert key in shared_pose_golden
                    delta=float((pred['instance_keypoint_xy'][0,v].cpu()-torch.tensor(shared_pose_golden[key])).abs().max());assert delta<2e-5,(key,delta);shared_cases+=1;max_shared_delta=max(max_shared_delta,delta)
            if args.full_context_control and task['variant']=='all':
                for v,view in enumerate(r['views']):
                    key=(r['scene'],int(view));gold=full_pose_golden.get(key)
                    if gold is None or gold['views']!=r['views']:continue
                    delta=float((pred['instance_keypoint_xy'][0,v].cpu()-torch.tensor(gold['xy'])).abs().max());assert delta<2e-5,(key,delta);full_cases+=1;max_full_delta=max(max_full_delta,delta)
        if args.context_control and task['variant']=='single':
            key=(r['scene'],int(r['views'][0]));assert key in single_golden
            delta=float((pred['instance_keypoint_xy'][0,0].cpu()-torch.tensor(single_golden[key])).abs().max());assert delta<2e-5,(key,delta);single_cases+=1;max_single_delta=max(max_single_delta,delta)
        if args.full_context_detector and task['variant'] in ['single','paired','all']:
            context={'single':'single','paired':'shared','all':'full'}[task['variant']]
            for v,view in enumerate(r['views']):
                if context=='full':
                    full_gold=full_pose_golden.get((r['scene'],int(view)))
                    if full_gold is None or full_gold['views']!=r['views']:continue
                key=(context,r['key'] if context=='shared' else r['scene'],int(view));gold=candidate_detection_goldens[key]
                for name,value in gold.items():
                    delta=float((pred[name][0,v].cpu()-torch.tensor(value)).abs().max());assert delta<2e-5,(key,name,delta);max_candidate_detector_delta=max(max_candidate_detector_delta,delta)
                candidate_detector_cases+=1
        t=instance_targets_from_batch(move_batch(item['batch'],'cuda'));mt=dict(t);mt['keypoint_valid']=torch.zeros_like(t['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0]
        h,w=images.shape[-2:];scale=images.new_tensor([w,h]);metrics=collections.defaultdict(list);tp=0;gt_count=0;point_count=detected_points=pck16=pck32=0;prediction_count=int((pred['instance_objectness_logits'].sigmoid()>=.5).sum())
        for v,(qs,ms) in enumerate(matches):
            for q,m in zip(qs.tolist(),ms.tolist()):
                gt_count+=1;iou=box_iou(pred['instance_boxes_cxcywh'][0,v,q:q+1],t['boxes_cxcywh'][0,v,m:m+1])[0,0];detected=bool((iou>=.5)&(pred['instance_objectness_logits'][0,v,q].sigmoid()>=.5));tp+=detected
                valid=t['keypoint_valid'][0,v,m].clone()
                if len(ms)>1:valid[7:14]=False
                if r['kind']=='real':
                    select=torch.zeros_like(valid);select[[0,14,15,16,17]]=True;valid&=select
                point_count+=int(valid.sum())
                spatial_error=((pred['instance_keypoint_xy'][0,v,q]-t['keypoint_xy'][0,v,m])*scale).norm(dim=-1)
                if detected:
                    detected_points+=int(valid.sum());pck16+=int(((spatial_error<=16)&valid).sum());pck32+=int(((spatial_error<=32)&valid).sum())
                for name,key in [('legacy_on_adapted_query' if args.detector_roi else 'original','instance_keypoint_original_xy'),('spatial','instance_keypoint_xy')]:
                    error=((pred[key][0,v,q]-t['keypoint_xy'][0,v,m])*scale).norm(dim=-1)
                    metrics[name+'/all'].extend(error[valid].tolist())
                    if r['views'][v]==task['anchor_view']:metrics[name+'/anchor'].extend(error[valid].tolist())
                    if detected:metrics[name+'/detected_only'].extend(error[valid].tolist())
        compact={k:pred[k].detach().cpu().numpy() for k in ['instance_keypoint_xy','instance_keypoint_original_xy','instance_boxes_cxcywh','instance_objectness_logits']}
        compact.update(gt_xy=t['keypoint_xy'].cpu().numpy(),valid=t['keypoint_valid'].cpu().numpy(),gt_boxes=t['boxes_cxcywh'].cpu().numpy(),box_valid=t['box_valid'].cpu().numpy())
        if args.detector_roi:
            for k in ['instance_boxes_original_cxcywh','instance_objectness_original_logits']:compact[k]=pred[k].detach().cpu().numpy()
        path=OUT/'predictions'/f'{n:03d}.npz';np.savez_compressed(path,**compact)
        row=dict(task=task,metrics={k:stats(v) for k,v in metrics.items()},detector=dict(tp=tp,gt=gt_count,predictions=prediction_count,recall=tp/max(gt_count,1),precision=tp/max(prediction_count,1)),gated_pose=dict(total_points=point_count,detected_points=detected_points,pck16_count=pck16,pck32_count=pck32,pck16=pck16/max(point_count,1),pck32=pck32/max(point_count,1)),path=path.relative_to(ROOT).as_posix(),source_hashes=item['source_hashes'])
        reports.append(row);write_json(OUT/'partial_results.json',reports);status('live_views',completed=n+1,total=len(tasks));print('LIVE_VIEWS',n+1,len(tasks),r['kind'],task['variant'],row['detector'],flush=True)
    if args.full_context_detector:
        assert candidate_detector_cases==84
        write_json(OUT/'candidate_detector_parity.json',dict(cases=candidate_detector_cases,max_delta=max_candidate_detector_delta,detector_sha256=fingerprint(detector_path),pose_sha256=fingerprint(READOUT),single_cases=12,shared_cases=24,full_real_cases=48))
    if args.shared_pose_control:assert shared_cases==24
    if args.full_context_control:assert full_cases==48
    write_json(OUT/'results.json',dict(reports=reports,parity_passed=True,single_rgb_parity_cases=single_cases,max_single_rgb_delta=max_single_delta,shared_token_probe=args.shared_first_tokens,single_full_output_exact=single_full_control,independent_pose=args.independent_pose,shared_pose_control=args.shared_pose_control,shared_rgb_parity_cases=shared_cases,max_shared_rgb_delta=max_shared_delta,full_context_control=args.full_context_control,full_rgb_parity_cases=full_cases,max_full_rgb_delta=max_full_delta,geometry_preserved_by_identity=bool(args.independent_pose or args.shared_pose_control),visual_review_pending=True,no_production_promotion=True));status('complete')
    state.update(stage='live_view_integration_audit_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next=f'Review {OUT.name} parity and detector coverage versus conditional/gated point metrics, same anchor under1/2/allcontexts. Visualize raw predictions, not GT-selected queries only. ROI detector and pose defaultoff; inspect variable-view regression before further training. Do not promote based on cached paired data.')
    state['completed'].append(OUT.name+'_integration_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
