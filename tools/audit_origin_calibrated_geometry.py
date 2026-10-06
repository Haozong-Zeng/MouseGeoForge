"""CPU-only, GT-associated triangulation diagnostic of saved 2D predictions."""
import fcntl,json,os,time
from collections import Counter,defaultdict
import numpy as np
import torch
from run_depth_pose_ablation import ROOT,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from vggt_omega.utils.robust_pose_geometry import triangulate_robust,project

SOURCE=ROOT/'output/origin_independent_joint_v1'
OUT=ROOT/'output/origin_calibrated_geometry_v1'
BODY=list(range(7))+list(range(14,23))
ARMS=['frozen','base_origin','anchor_origin']
MODES=['independent','joint']


def main():
    torch.set_num_threads(2)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
        began=time.monotonic()
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        def cutoff():
            if time.monotonic()-began>600 or time.time()>1789233567:raise TimeoutError('Geometry diagnostic cutoff')
        status('preflight')
        try:
            assert json.loads((SOURCE/'verification.json').read_text())['passed']
            protocol=json.loads((SOURCE/'protocol.json').read_text())
            assert all(fingerprint(ROOT/p)==h for p,h in {**protocol['guards'],**protocol['code_sha256']}.items())
            saved=json.loads((SOURCE/'synthetic_records.json').read_text())
            lookup={(r['index'],r['view'],r['instance'],r['mode'],r['arm']):r for r in saved}
            assert len(lookup)==len(saved)
            paths=[SOURCE/'synthetic_records.json',SOURCE/'protocol.json',ROOT/'tools/audit_origin_calibrated_geometry.py',ROOT/'vggt-omega/vggt_omega/utils/robust_pose_geometry.py']
            hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in paths}
            write_json(OUT/'protocol.json',dict(hashes=hashes,training_steps=0,encoder_calls=0,selected=protocol['selected'],
                preflight='GT projection maximum1px; supported GT triangulation maximum normalized3D error.005. Stop before predicted solves if failed.',
                triangulation=dict(reprojection_threshold_px=8.,huber_delta_px=3.,min_ray_angle_deg=1.),
                association='GT-mask-assisted per-instance/view from stage33; NOT deployable identity association. All same valid body/world GT set, no objectness gates.',
                units='Preserve each source native units; normalize additionally by GT body3D bbox diagonal; no cross-unit pooled native errors.',
                warning='Coverage and own/common-support errors reported separately. Two views have no arbitrary outlier redundancy. Single-view3D unsupported.',
                heldout_accessed=False,real_image_reads=0,production_promoted=False))
            inputs=[];gt_rows=[];collisions=[];projection_errors=[]
            for index,e in enumerate(protocol['selected']):
                cutoff();rec=protocol['records'][e['key']];batch=oriented_batch(rec,num_views=len(rec['views']));t=instance_targets_from_batch(batch)
                h,w=batch['images'].shape[-2:];scale=float(batch['scale'][0]);assert abs(scale-rec['normalization_scale'])<1e-4
                p=t['intrinsics'][0].double()@t['extrinsics'][0,:,:3].double()
                assert t['keypoint_world_xyz'].ndim==4 and t['keypoint_world_xyz'].shape[-2:]==(23,3)
                instances=torch.where(t['instance_valid'][0].any(0))[0].tolist()
                for mode in MODES:
                    for v,view in enumerate(rec['views']):
                        qs=[lookup[(index,view,m,mode,'frozen')]['query'] for m in instances if (index,view,m,mode,'frozen') in lookup]
                        collisions.append(dict(index=index,mode=mode,view=view,targets=len(qs),unique_queries=len(set(qs)),collision=len(qs)!=len(set(qs))))
                for m in instances:
                    xyz=t['keypoint_world_xyz'][0,m].double();world_valid=t['world_valid'][0,m].bool()
                    body_ids=[k for k in BODY if world_valid[k] and torch.isfinite(xyz[k]).all()]
                    if not body_ids:continue
                    diag=float((xyz[body_ids].amax(0)-xyz[body_ids].amin(0)).norm());assert diag>1e-9
                    gt=t['keypoint_xy'][0,:,m].double()*torch.tensor([w,h],dtype=torch.float64)
                    for k in body_ids:
                        valid=t['keypoint_valid'][0,:,m,k].bool()&t['instance_valid'][0,:,m].bool()&t['camera_available'][0].bool()
                        projected,z=project(p,xyz[k]);err=(projected-gt[:,k]).norm(dim=-1)
                        projection_errors.extend(err[valid].tolist())
                        result=triangulate_robust(gt[:,k],p,valid,huber_delta_px=3.)
                        gt_rows.append(dict(index=index,group=e['group'],instance=m,keypoint=k,valid_views=int(valid.sum()),supported=result.supported,reason=result.reason,
                            max_projection_px=float(err[valid].max()) if valid.any() else None,
                            normalized_error=float((result.xyz-xyz[k]).norm())/diag if result.supported else None))
                        inputs.append(dict(index=index,group=e['group'],record=rec,instance=m,keypoint=k,projection=p,valid=valid,xyz=xyz[k],diag=diag,scale=scale))
                status('preflight',records=index+1,total=32)
            supported_gt=[r['normalized_error'] for r in gt_rows if r['supported']]
            preflight=dict(max_projection_px=max(projection_errors),max_supported_gt_normalized_error=max(supported_gt),
                gt_support=sum(r['supported'] for r in gt_rows),gt_points=len(gt_rows),reasons=dict(Counter(r['reason'] for r in gt_rows)),
                association_collision_views=sum(r['collision'] for r in collisions))
            write_json(OUT/'preflight.json',preflight);write_json(OUT/'gt_records.json',gt_rows);write_json(OUT/'association_collisions.json',collisions)
            assert preflight['max_projection_px']<=1.,'GT projection self-check failed'
            assert preflight['max_supported_gt_normalized_error']<=.005,'GT triangulation self-check failed'
            rows=[]
            for i,item in enumerate(inputs):
                cutoff();rec=item['record'];m=item['instance'];k=item['keypoint'];vcount=len(rec['views'])
                for mode in MODES:
                    for arm in ARMS:
                        xy=torch.full((vcount,2),float('nan'),dtype=torch.float64)
                        for v,view in enumerate(rec['views']):
                            if item['valid'][v]:
                                r=lookup[(item['index'],view,m,mode,arm)]
                                assert k in r['body_indices'];xy[v]=torch.tensor(r['xy'][r['body_indices'].index(k)],dtype=torch.float64)
                        result=triangulate_robust(xy,item['projection'],item['valid'],huber_delta_px=3.)
                        error=float((result.xyz-item['xyz']).norm()) if result.supported else None
                        rows.append(dict(index=item['index'],group=item['group'],instance=m,keypoint=k,mode=mode,arm=arm,views=vcount,
                            valid_views=int(item['valid'].sum()),supported=result.supported,reason=result.reason,units=rec['depth_units'],
                            normalized_error=error/item['diag'] if error is not None else None,error_native=error*item['scale'] if error is not None else None,
                            xyz_native=(result.xyz*item['scale']).tolist() if result.supported else None,
                            gt_xyz_native=(item['xyz']*item['scale']).tolist(),inliers=result.inliers.tolist(),ray_angle_deg=result.max_ray_angle_deg,
                            reprojection_px=[float(x) if torch.isfinite(x) else None for x in result.reprojection_px]))
                if (i+1)%32==0:status('triangulating',points=i+1,total=len(inputs))
            by_point=defaultdict(list)
            for r in rows:by_point[(r['index'],r['instance'],r['keypoint'])].append(r)
            common={key for key,rr in by_point.items() if len(rr)==6 and all(r['supported'] for r in rr)}
            summary={}
            for group in dict.fromkeys(r['group'] for r in rows):
                summary[group]={}
                for mode in MODES:
                    for arm in ARMS:
                        rr=[r for r in rows if (r['group'],r['mode'],r['arm'])==(group,mode,arm)]
                        own=[r for r in rr if r['supported']];paired=[r for r in own if (r['index'],r['instance'],r['keypoint']) in common]
                        assert len({r['units'] for r in rr})==1
                        def metrics(xx):
                            return dict(n=len(xx),mean_native=float(np.mean([r['error_native'] for r in xx])) if xx else None,
                                mean_normalized=float(np.mean([r['normalized_error'] for r in xx])) if xx else None)
                        summary[group][mode+'/'+arm]=dict(total=len(rr),coverage=len(own)/len(rr),units=rr[0]['units'],own_support=metrics(own),common_support=metrics(paired),reasons=dict(Counter(r['reason'] for r in rr)))
            assert all(fingerprint(ROOT/p)==h for p,h in hashes.items())
            write_json(OUT/'records.json',rows);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,encoder_calls=0,training_steps=0,source_unchanged=True,gt_self_check_passed=True,
                points=len(inputs),prediction_solves=len(rows),common_supported_points=len(common),heldout_accessed=False,production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()
