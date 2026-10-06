"""Predeclared calibration/test blocks unseen by the fixed 768 pose checkpoint.

Both blocks come from the same previously held-out CRIM13 session, NOT independent
animals/sessions. Excludes all previously inspected validation frames and +/-100
source frames around them. No 2D model, detection or depth weights are optimized.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from diagnose_occlusion_pose import ROOT,torch,infer,fingerprint,write_json
from mousegeoforge.uncertainty_calibration import fit_temperature,apply_temperature,uncertainty_metrics
from vggt_omega.utils.robust_pose_geometry import triangulate_robust

DUAL='Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1'


def prepare_protocol():
    run=ROOT/'output/pose_diverse_core23_768_v1'
    old=json.loads((run/'manifest.json').read_text())
    plan=json.loads((ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json').read_text())
    old_val=[r for r in old['records'] if r['split']=='val' and r['expected_instances']==2]
    train_scenes={r['scene'] for r in old['records'] if r['split']=='train'}
    excluded={r['scene'] for r in old['records']}
    candidate=[]
    for k in plan['keyframes']:
        session=Path(k['source_image'].replace('\\','/')).parent.name
        if session!='021309_A29_Block16_BCma1_t': continue
        scene=f'dataset/BlenderPyMouse/{DUAL}/frame_{k["scene_frame"]:06d}'
        if scene in excluded: continue
        if min(abs(k['source_image_frame']-r['source_frame']) for r in old_val)<=100: continue
        # Require the strict source/annotation verification from stage 1.
        if not (ROOT/scene/'keypoint_occlusion.npz').exists(): continue
        candidate.append(dict(old_val[0],scene=scene,frame=k['scene_frame'],
            source_frame=k['source_image_frame'],annotation_sha256=fingerprint(ROOT/scene/'keypoints.npz')))
    candidate.sort(key=lambda r:r['source_frame'])
    if len(candidate)<96: raise ValueError(f'Need 96 unused, verified frames; found {len(candidate)}')
    calibration=[dict(r,split='calibration') for r in candidate[:48]]
    test=[dict(r,split='test') for r in candidate[-48:]]
    gap=test[0]['source_frame']-calibration[-1]['source_frame']
    if gap<500: raise ValueError(f'Insufficient source-frame gap: {gap}')
    records=calibration+test
    assert len({r['scene'] for r in records})==96
    assert not {r['scene'] for r in records}&train_scenes
    return dict(records=records,candidate_frames=len(candidate),source_frame_gap=gap,
        head_checkpoint_sha256=fingerprint(run/'seed17_off/final_heads.pt'),
        prior_manifest_sha256=fingerprint(run/'manifest.json'),
        plan_sha256=fingerprint(ROOT/'dataset/Homecage/LightningPose-CRIM13-Diverse400.plan.json'),
        source_session='021309_A29_Block16_BCma1_t',calibration_scenes=48,test_scenes=48,
        calibration_test_session_independent=False,calibration_test_frame_disjoint=True,
        pose_training_session_disjoint=True,previous_validation_exclusion_margin_source_frames=100,
        primary_variant='temperature_original_sigma',primary_policy='fit one temperature on calibration block; no test tuning',
        calibration_gate=dict(coverage95_min=.90,coverage95_max=.99,visibility_group_min=.85,
                              visibility_group_max=.995,nll_must_improve=True),
        geometry_gate=dict(max_paired_mean_error_ratio=1.01,max_paired_p95_error_ratio=1.05,
                           max_coverage_drop=.03),
        limitation='Same synthetic assets and held-out session across calibration/test; no independent real-animal generalization claim.')


def points(data, head):
    residual=[]; original=[]; fitted=[]; visible=[]; frames=[]; indices=[]
    for i,item in enumerate(data):
        t,p=item['targets'],item['predictions']
        for v,(qs,ms) in enumerate(item['oracle_assignment']):
            for q,m in zip(qs,ms):
                valid=t['keypoint_valid'][0,v,m]
                residual.append((p['instance_keypoint_xy'][0,v,q]-t['keypoint_xy'][0,v,m])[valid])
                original.append(p['instance_keypoint_sigma'][0,v,q][valid])
                with torch.no_grad(): fitted.append(1e-3+torch.nn.functional.softplus(head(item['sigma_tokens'][v,q][valid]).squeeze(-1)))
                visible.append(t['keypoint_visible'][0,v,m][valid])
                frames.extend([i]*int(valid.sum()))
                indices.extend((i,v,q,k) for k in torch.where(valid)[0].tolist())
    return dict(residual=torch.cat(residual),original=torch.cat(original),fitted=torch.cat(fitted),
                visible=torch.cat(visible),frames=torch.tensor(frames),indices=indices)


def metrics(x, sigma):
    result=uncertainty_metrics(x['residual'],sigma)
    result['by_visibility']={name:uncertainty_metrics(x['residual'][mask],sigma[mask])
        for name,mask in (('visible',x['visible']),('occluded',~x['visible'])) if mask.any()}
    # Bootstrap whole scene summaries, not 4,416 correlated keypoints as iid data.
    z2=(x['residual']/sigma[:,None]).square().sum(-1)
    rates=torch.stack([(z2[x['frames']==i]<=5.991465).float().mean() for i in x['frames'].unique()])
    generator=torch.Generator().manual_seed(17)
    sampled=rates[torch.randint(len(rates),(1000,len(rates)),generator=generator)].mean(-1)
    result['scene_bootstrap_coverage95_interval']=torch.quantile(sampled,torch.tensor([.025,.975])).tolist()
    result['bootstrap_limitation']='Frames within temporal blocks can still correlate; interval is descriptive, not an iid guarantee.'
    return result


def geometry(test, temperatures):
    rows=[]
    for i,item in enumerate(test):
        p,t=item['predictions'],item['targets']
        h,w=item['record']['image_hw']; scale=torch.tensor([w,h])
        proj=t['intrinsics'][0]@t['extrinsics'][0,:,:3]
        oracle=[{m:q for q,m in zip(qs,ms)} for qs,ms in item['oracle_assignment']]
        native=float(item['scale'].flatten()[0])
        for m in range(2):
            qs=[o[m] for o in oracle]
            for k in range(23):
                uv=torch.stack([p['instance_keypoint_xy'][0,v,q,k] for v,q in enumerate(qs)])*scale
                sigma=torch.stack([p['instance_keypoint_sigma'][0,v,q,k] for v,q in enumerate(qs)])[:,None]*scale
                r=dict(scene=item['record']['scene'],instance=m,keypoint=k)
                for variant,s in (('uniform',None),('temperature_original_sigma',sigma*temperatures['original'])):
                    out=triangulate_robust(uv,proj,t['keypoint_valid'][0,:,m,k],sigma_px=s,
                        uncertainty_calibrated=s is not None)
                    r[variant]=float((out.xyz-t['keypoint_world_xyz'][0,m,k]).norm())*native if out.supported else None
                rows.append(r)
    summaries={}
    for key in ('uniform','temperature_original_sigma'):
        errors=torch.tensor([r[key] for r in rows if r[key] is not None])
        summaries[key]=dict(points=len(rows),supported=len(errors),coverage=len(errors)/len(rows),
            mean_mm=float(errors.mean()) if len(errors) else None,
            p95_mm=float(torch.quantile(errors,.95)) if len(errors) else None)
    paired=[r for r in rows if all(r[k] is not None for k in summaries)]
    summaries['paired_points']=len(paired)
    for key in ('uniform','temperature_original_sigma'):
        errors=torch.tensor([r[key] for r in paired])
        summaries[key]['paired_mean_mm']=float(errors.mean()) if len(errors) else None
        summaries[key]['paired_p95_mm']=float(torch.quantile(errors,.95)) if len(errors) else None
    a,b=summaries['uniform'],summaries['temperature_original_sigma']
    passed=bool(paired) and (b['paired_mean_mm']<=a['paired_mean_mm']*1.01 and b['paired_p95_mm']<=a['paired_p95_mm']*1.05
                            and b['coverage']>=a['coverage']-.03)
    return dict(summary=summaries,gate_passed=passed,rows=rows,
                limitation='GT box-assisted instance correspondence; two-view geometry; not end-to-end detection/association.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/occlusion_pose_v2/calibration')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    protocol=prepare_protocol()
    path=args.output/'protocol.json'; args.output.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert json.loads(path.read_text())==protocol,'Protocol changed; use a new output directory'
    write_json(path,protocol)
    print({k:v for k,v in protocol.items() if k!='records'},flush=True)
    if args.prepare_only: return
    if not args.report_only: infer(ROOT/'output/pose_diverse_core23_768_v1',args.output,protocol['records'])
    data=[torch.load(args.output/'predictions'/f'{i:03d}.pt',weights_only=True) for i in range(96)]
    for record, item in zip(protocol['records'], data):
        if item['record'] != record:
            raise ValueError('Prediction cache does not match locked calibration protocol')
        if fingerprint(ROOT/record['scene']/'keypoints.npz') != record['annotation_sha256']:
            raise ValueError('Original annotation changed after protocol lock')
    head=torch.nn.Linear(data[0]['sigma_tokens'].shape[-1],1)
    saved=torch.load(ROOT/'output/occlusion_pose_v1/sigma_only_pilot.pt',weights_only=True)
    head.load_state_dict(saved['state_dict'])
    cal=points(data[:48],head); test=points(data[48:],head)
    temperatures={k:fit_temperature(cal['residual'],cal[k]) for k in ('original','fitted')}
    constant=fit_temperature(cal['residual'],torch.ones_like(cal['original']))
    report=dict(temperatures=temperatures,constant_sigma=constant,protocol_sha256=fingerprint(path),
                checkpoint_sha256=protocol['head_checkpoint_sha256'],coordinate_model_unchanged=True,
                secondary_sigma_checkpoint_sha256=fingerprint(ROOT/'output/occlusion_pose_v1/sigma_only_pilot.pt'),
                prediction_cache_records_verified=True,
                variants={},scope='synthetic dual-mouse homecage, fixed rig, same held-out source session')
    for split,x in (('calibration',cal),('test',test)):
        variants=dict(original_sigma=x['original'],temperature_original_sigma=apply_temperature(x['original'],temperatures['original']),
                      failed_train_sigma=x['fitted'],temperature_failed_train_sigma=apply_temperature(x['fitted'],temperatures['fitted']),
                      constant=torch.full_like(x['original'],constant))
        report['variants'][split]={k:metrics(x,s) for k,s in variants.items()}
    primary=report['variants']['test']['temperature_original_sigma']
    baseline=report['variants']['test']['original_sigma']
    gate=protocol['calibration_gate']
    report['calibration_gate_passed']=(gate['coverage95_min']<=primary['coverage_95']<=gate['coverage95_max']
        and primary['nll']<baseline['nll']
        and all(gate['visibility_group_min']<=v['coverage_95']<=gate['visibility_group_max'] for v in primary['by_visibility'].values()))
    g=geometry(data[48:],temperatures)
    write_json(args.output/'geometry_test.json',g)
    report['geometry_gate_passed']=g['gate_passed']
    report['fusion_gate_passed']=bool(report['calibration_gate_passed'] and g['gate_passed'])
    report['production_checkpoint_modified']=False
    report['production_fusion_enabled']=False
    write_json(args.output/'calibration_report.json',report)
    print('CALIBRATION COMPLETE',report['calibration_gate_passed'],'geometry',g['gate_passed'],flush=True)


if __name__=='__main__': main()
