"""Frozen 768 checkpoint audit + isolated sigma-only pilot (no pose/depth updates).

Exports small prediction caches, stratified 2D/3D metrics and a held-out calibration
gate. GT-assisted association is explicitly separate from prediction-only pairing.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from run_depth_pose_ablation import (ROOT, torch, VGGTOmega, DepthPoseModel, load_batch,
    fingerprint, restore, write_json, instance_targets_from_batch)
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.pose_association import associate_two_views
from training.instance_pose_losses import hungarian_match_instances, keypoint_uncertainty_loss
from vggt_omega.utils.robust_pose_geometry import triangulate_robust
import numpy as np


def infer(run, destination, records, *, batch_loader=None):
    batch_loader = load_batch if batch_loader is None else batch_loader
    base = ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    spatial = VGGTOmega(enable_camera=False, enable_depth=True, enable_multi_instance_pose=True, num_keypoints=23)
    original = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    selected = {k:v for k,v in original.items() if k.startswith(('aggregator.', 'dense_head.'))}
    result = spatial.load_state_dict(selected, strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys)
    del original, selected
    model = DepthPoseModel.from_spatial_model(spatial).cuda().eval()
    checkpoint = torch.load(run/'seed17_off/final_heads.pt', map_location='cpu', weights_only=True)
    assert checkpoint['checkpoint_sha256'] == fingerprint(base)
    restore(model, checkpoint['model'])
    expected = {r['scene']: r['metrics'] for r in json.loads((run/'seed17_off/validation.json').read_text())['records']}
    capture = {}
    handle = model.instance_pose_head.keypoint_log_sigma.register_forward_pre_hook(
        lambda module, args: capture.update(tokens=args[0].detach().cpu()))
    for i, r in enumerate(records):
        path = destination/'predictions'/f'{i:03d}.pt'
        if path.exists():
            saved = torch.load(path, weights_only=True)
            assert saved['record'] == r
            continue
        batch = batch_loader(r)
        target = instance_targets_from_batch(batch)
        with torch.no_grad():
            images = batch['images'].cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cached, start = model.aggregator(images)
            output = model.forward_cached(images, cached, start, coupling='off')
            cpu = {k:v.cpu() for k,v in output.items()}
            metrics, assignment = evaluate_scene(cpu, batch, target)
        max_difference = 0.
        if r['split'] == 'val':
            max_difference = max(abs(v-expected[r['scene']][k]) for k,v in metrics.items() if v is not None)
            assert max_difference < 1e-6, (r['scene'], max_difference)
        keep = ('instance_objectness_logits', 'instance_boxes_cxcywh', 'instance_keypoint_xy',
                'instance_keypoint_visibility_logits', 'instance_keypoint_sigma', 'instance_reid_embedding')
        targets = {k:v for k,v in target.items() if k not in ('masks',)}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(record=r, predictions={k:cpu[k] for k in keep}, targets=targets,
                        sigma_tokens=capture['tokens'], scale=batch['scale'],
                        actual_view_ids=batch['view_ids'], oracle_assignment=assignment,
                        baseline_2d=metrics, baseline_parity_max_error=max_difference), path)
        print(f'Frozen inference {i+1}/{len(records)}: {r["dataset"]} {r["frame"]}', flush=True)
        del output, cached, images
    handle.remove()
    torch.save(model.instance_pose_head.keypoint_log_sigma.cpu().state_dict(), destination/'initial_sigma.pt')
    del model, spatial
    torch.cuda.empty_cache()


def calibrated_metrics(rows, head):
    errors, scales, old_scales, groups = [], [], [], []
    for item in rows:
        pred, target = item['predictions'], item['targets']
        for v, (qs, ms) in enumerate(item['oracle_assignment']):
            for q,m in zip(qs, ms):
                valid = target['keypoint_valid'][0,v,m]
                error = (pred['instance_keypoint_xy'][0,v,q]-target['keypoint_xy'][0,v,m])[valid]
                tokens = item['sigma_tokens'][v,q][valid]
                scales.append((1e-3+torch.nn.functional.softplus(head(tokens).squeeze(-1))).detach())
                old_scales.append(pred['instance_keypoint_sigma'][0,v,q][valid])
                errors.append(error)
                vis = target['keypoint_visible'][0,v,m][valid]
                known = target['keypoint_visibility_valid'][0,v,m][valid]
                groups.extend(['visible' if a else 'occluded' if b else 'unknown'
                               for a,b in zip(vis.tolist(), known.tolist())])
    error, scale, old = torch.cat(errors), torch.cat(scales), torch.cat(old_scales)
    def summary(s):
        z2 = (error/s[:,None]).square().sum(-1)
        order = s.argsort()
        result = dict(points=len(s), gaussian_nll=float((.5*z2+2*s.log()).mean()),
                      coverage_68=float((z2<=2.278869).float().mean()),
                      coverage_95=float((z2<=5.991465).float().mean()),
                      risk_coverage={str(f):float(error[order[:max(1,int(len(order)*f))]].norm(dim=-1).mean())
                                     for f in (.25,.5,.75,1.)})
        result['by_visibility'] = {g:dict(points=int((mask:=torch.tensor([x==g for x in groups])).sum()),
            coverage_95=float((z2[mask]<=5.991465).float().mean())) for g in sorted(set(groups))}
        return result
    return dict(initial=summary(old), fitted=summary(scale))


def fit_scale(data, destination):
    head = torch.nn.Linear(data[0]['sigma_tokens'].shape[-1], 1)
    head.load_state_dict(torch.load(destination/'initial_sigma.pt', weights_only=True))
    tokens, xy, gt = [], [], []
    train = [x for x in data if x['record']['split']=='train']
    val = [x for x in data if x['record']['split']=='val']
    for item in train:
        for v, (qs, ms) in enumerate(hungarian_match_instances(item['predictions'], item['targets'])[0]):
            for q,m in zip(qs.tolist(), ms.tolist()):
                valid = item['targets']['keypoint_valid'][0,v,m]
                tokens.append(item['sigma_tokens'][v,q,valid])
                xy.append(item['predictions']['instance_keypoint_xy'][0,v,q,valid])
                gt.append(item['targets']['keypoint_xy'][0,v,m,valid])
    tokens, xy, gt = torch.cat(tokens), torch.cat(xy), torch.cat(gt)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    history = []
    for step in range(300):
        optimizer.zero_grad()
        sigma = 1e-3+torch.nn.functional.softplus(head(tokens).squeeze(-1))
        loss = keypoint_uncertainty_loss(xy, sigma, gt, torch.ones(len(xy),dtype=torch.bool))
        loss.backward(); optimizer.step()
        if step%50 == 0 or step==299:
            history.append(dict(step=step+1,nll=float(loss.detach())))
    report = dict(train=calibrated_metrics(train,head), validation=calibrated_metrics(val,head),
                  history=history, optimized_parameters=sum(p.numel() for p in head.parameters()),
                  fixed_pose_depth_encoder=True, train_scenes=len(train), val_scenes=len(val),
                  steps=300, learning_rate=.003, seed=17, validation_not_used_for_fitting=True)
    fitted = report['validation']['fitted']
    risks = list(fitted['risk_coverage'].values())
    report['fusion_gate_passed'] = (.90 <= fitted['coverage_95'] <= .99
        and fitted['gaussian_nll'] < report['validation']['initial']['gaussian_nll']
        and risks[0] < risks[-1]
        and all(.85 <= r['coverage_95'] <= .995 for r in fitted['by_visibility'].values() if r['points']>=30))
    report['gate_policy'] = '95% coverage 90-99%; subgroup 85-99.5%; NLL improves; low-uncertainty quartile has lower error. Pilot gate only, not broad generalization.'
    torch.save(dict(state_dict=head.state_dict(), report=report), destination/'sigma_only_pilot.pt')
    write_json(destination/'uncertainty_calibration.json', report)
    return head, report


def summarize(values):
    accepted = [r for r in values if r['error_native'] is not None]
    errors = [r['error_native'] for r in accepted]
    return dict(points=len(values), supported=len(accepted), coverage=len(accepted)/max(len(values),1),
                mean_error_native=float(np.mean(errors)) if errors else None,
                median_error_native=float(np.median(errors)) if errors else None,
                p95_error_native=float(np.percentile(errors,95)) if errors else None)


def geometry_report(data, head, calibration, destination):
    rows, associations, point2d = [], [], []
    for item in data:
        if item['record']['split'] != 'val': continue
        pred, target = item['predictions'], item['targets']
        hw = item['record']['image_hw']; scale2d = torch.tensor([hw[1],hw[0]])
        projections = target['intrinsics'][0]@target['extrinsics'][0,:,:3]
        pairs = associate_two_views(pred, projections, hw)
        pair_map = {a:b for a,b,_ in pairs}
        oracle = [{m:q for q,m in zip(qs,ms)} for qs,ms in item['oracle_assignment']]
        native_scale = float(item['scale'].flatten()[0])
        cause_path=ROOT/item['record']['scene']/'keypoint_occlusion.npz'
        causes=None
        if cause_path.exists():
            with np.load(cause_path,allow_pickle=False) as ca:
                lookup={int(v):i for i,v in enumerate(ca['view_ids'])}
                view_order=[int(v) for v in item['actual_view_ids'][0].split(',')]
                causes=ca['occlusion_cause'][[lookup[v] for v in view_order]]
        def cause_name(v,m,k):
            if target['keypoint_visibility_valid'][0,v,m,k] and target['keypoint_visible'][0,v,m,k]:
                return 'visible'
            if causes is None: return 'unknown'
            return {-1:'unknown',0:'visible',1:'self',2:'other_mouse',3:'mixed'}[int(causes[v,m,k])]
        for m in range(item['record']['expected_instances']):
            qs = [x.get(m) for x in oracle]
            if any(q is None for q in qs): continue
            predicted_pair = pair_map.get(qs[0])
            associations.append(dict(scene=item['record']['scene'], instance=m,
                linked=predicted_pair is not None, correct=predicted_pair==qs[1]))
            visible = target['keypoint_visible'][0,:,m]
            known = target['keypoint_visibility_valid'][0,:,m]
            for v,q in enumerate(qs):
                for k in range(23):
                    if not target['keypoint_valid'][0,v,m,k]: continue
                    error=float(((pred['instance_keypoint_xy'][0,v,q,k]-target['keypoint_xy'][0,v,m,k])*scale2d).norm())
                    diag=float((target['boxes_cxcywh'][0,v,m,2:]*scale2d).norm())
                    point2d.append(dict(dataset=item['record']['dataset'],scene=item['record']['scene'],
                        instance=m,view=v,keypoint=k,cause=cause_name(v,m,k),error_px=error,nme=error/max(diag,1.)))
            for k in range(23):
                if not target['world_valid'][0,m,k]: continue
                vis_count = int(visible[:,k].sum())
                group = ('unknown' if not known[:,k].all() else
                         'all_visible' if vis_count==2 else 'one_visible' if vis_count==1 else 'all_occluded')
                methods = ('linear_oracle_association', 'robust_oracle_association',
                           'pred_visibility_oracle_association', 'gt_visibility_diagnostic',
                           'sigma_oracle_association_experimental', 'robust_predicted_association')
                for mode in methods:
                    queries = qs.copy()
                    if mode == 'robust_predicted_association': queries[1] = predicted_pair
                    error, reason = None, 'unmatched_instance'
                    if queries[1] is not None:
                        uv = torch.stack([pred['instance_keypoint_xy'][0,v,q,k] for v,q in enumerate(queries)])*scale2d
                        valid = torch.isfinite(uv).all(-1) & (uv>=0).all(-1) & (uv<scale2d).all(-1)
                        # GT label validity is only for reference association diagnostics.
                        if mode != 'robust_predicted_association': valid &= target['keypoint_valid'][0,:,m,k]
                        if mode.startswith('pred_visibility'):
                            valid &= torch.stack([pred['instance_keypoint_visibility_logits'][0,v,q,k] for v,q in enumerate(queries)]) > 0
                        if mode.startswith('gt_visibility'): valid &= visible[:,k] & known[:,k]
                        sigma = None
                        if mode.startswith('sigma'):
                            sigma = torch.stack([1e-3+torch.nn.functional.softplus(head(item['sigma_tokens'][v,q,k]).squeeze())
                                                 for v,q in enumerate(queries)]).detach()[:,None]*scale2d
                        linear = mode.startswith('linear')
                        r = triangulate_robust(uv, projections, valid, sigma_px=sigma,
                            uncertainty_calibrated=sigma is not None, robust=not linear,
                            reprojection_threshold_px=1e9 if linear else 8., min_ray_angle_deg=0. if linear else 1.)
                        reason = r.reason
                        if r.supported:
                            error = float((r.xyz-target['keypoint_world_xyz'][0,m,k]).norm())*native_scale
                    rows.append(dict(scene=item['record']['scene'], dataset=item['record']['dataset'],
                        instance=m, keypoint=k, visibility=group, method=mode,
                        causes=[cause_name(v,m,k) for v in range(2)],
                        error_native=error, reason=reason))
        print(f'Geometry audit: {item["record"]["dataset"]} {item["record"]["frame"]}', flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['method'],row['visibility'])].append(row)
        groups[(row['dataset'],row['method'],'all')].append(row)
    report = dict(metrics=[dict(dataset=d,method=m,visibility=v,**summarize(x)) for (d,m,v),x in groups.items()],
        association=dict(instances=len(associations), linked=sum(r['linked'] for r in associations),
                         correct=sum(r['correct'] for r in associations)),
        sigma_pilot_gate_passed=calibration['fusion_gate_passed'],
        baseline_parity_max_error=max(x['baseline_parity_max_error'] for x in data),
        units='Original dataset world-coordinate units, not pooled across datasets; no per-frame scale alignment.',
        limitations=['Two views do not provide robust outlier redundancy.',
          'GT box-assisted association modes are diagnostics, not end-to-end inference.',
          'Sigma experimental mode is reported even if the deployment calibration gate fails.',
          'Coverage must accompany error; rejected points are not counted as correct.',
          'No temporal, depth-surface equality, tail-curve or encoder training in this pilot.'])
    baseline={(r['scene'],r['instance'],r['keypoint']):r['error_native'] for r in rows if r['method'].startswith('linear')}
    paired={}
    for (d,m,v),items in groups.items():
        if v!='all': continue
        pairs=[(r['error_native'],baseline[(r['scene'],r['instance'],r['keypoint'])])
               for r in items if r['error_native'] is not None and baseline[(r['scene'],r['instance'],r['keypoint'])] is not None]
        paired[d+'/'+m]=dict(points=len(pairs),mean_method=float(np.mean([a for a,b in pairs])) if pairs else None,
                             mean_linear_same_points=float(np.mean([b for a,b in pairs])) if pairs else None)
    report['paired_on_same_accepted_points']=paired
    report['two_d_by_occlusion']=[]
    for d in sorted({r['dataset'] for r in point2d}):
        for c in sorted({r['cause'] for r in point2d}):
            subset=[r for r in point2d if r['dataset']==d and r['cause']==c]
            if subset:
                report['two_d_by_occlusion'].append(dict(dataset=d,cause=c,points=len(subset),
                    mean_nme=float(np.mean([r['nme'] for r in subset])),
                    mean_error_px=float(np.mean([r['error_px'] for r in subset]))))
    write_json(destination/'geometry_report.json', report)
    write_json(destination/'geometry_points.json', rows)
    write_json(destination/'points_2d.json',point2d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'output/pose_diverse_core23_768_v1')
    p.add_argument('--output',type=Path,default=ROOT/'output/occlusion_pose_v1')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest=json.loads((args.run/'manifest.json').read_text())
    groups=defaultdict(list)
    for r in manifest['records']:
        if r['split']=='train': groups[r['source_group']].append(r)
    train=[]
    for group in sorted(groups):
        rs=groups[group]
        train.extend(rs[i] for i in np.linspace(0,len(rs)-1,min(8,len(rs)),dtype=int))
    records=train+[r for r in manifest['records'] if r['split']=='val']
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(records=records, heads_sha256=fingerprint(args.run/'seed17_off/final_heads.pt'),
                  seed=17, sigma_steps=300, sigma_lr=.003, inference_score_threshold=.5,
                  epipolar_threshold_px=32., reprojection_threshold_px=8.)
    path=args.output/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol, 'Refusing to mix prediction caches'
    write_json(path,protocol)
    if not args.report_only: infer(args.run,args.output,records)
    data=[torch.load(args.output/'predictions'/f'{i:03d}.pt',weights_only=True) for i in range(len(records))]
    if args.report_only:
        saved=torch.load(args.output/'sigma_only_pilot.pt',weights_only=True)
        head=torch.nn.Linear(data[0]['sigma_tokens'].shape[-1],1)
        head.load_state_dict(saved['state_dict']); calibration=saved['report']
    else:
        head,calibration=fit_scale(data,args.output)
    geometry_report(data,head,calibration,args.output)
    print('OCCLUSION AUDIT COMPLETE; sigma gate:',calibration['fusion_gate_passed'],flush=True)


if __name__=='__main__': main()
