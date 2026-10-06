"""Same locked oracle crops, six frozen readouts; no per-point best selection."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor
from mousegeoforge.roi_scale_diagnostic import warp_rgb_masks,inverse_points,mask_query_iou
from mousegeoforge.affine_augmentation import transform_xy
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.frozen_spatial_readouts import ARMS,frozen_readouts

OUT=ROOT/'output/roi_readout_stages_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
ZOOM=ROOT/'output/oracle_roi_scale_diagnostic_v1'
CENTER=ROOT/'output/oracle_roi_center_diagnostic_v1'


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            parent=json.loads((PARENT/'protocol.json').read_text());assert json.loads((PARENT/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/p)==h for p,h in parent['code_sha256'].items())
            plan=json.loads((ZOOM/'protocol.json').read_text());references={}
            for name,directory in [('zoom',ZOOM),('center',CENTER)]:
                assert json.loads((directory/'verification.json').read_text())['passed']
                references[name]={(row['index'],row['view'],row['instance']):row for row in json.loads((directory/'records.json').read_text())}
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';candidate=PARENT/'scale_sigma_step800.pt'
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate,ZOOM/'protocol.json',ZOOM/'records.json',CENTER/'records.json')}
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            assert all(guards[p]==h for p,h in plan['weights'].items())
            code=['tools/audit_roi_readout_stages.py','mousegeoforge/frozen_spatial_readouts.py','mousegeoforge/roi_scale_diagnostic.py','mousegeoforge/deterministic_local_sampling.py']
            codehash={p:fingerprint(ROOT/p) for p in code}
            write_json(OUT/'protocol.json',dict(selected=plan['selected'],readouts=ARMS,geometries=['original','center','zoom'],guards=guards,code_sha256=codehash,
                training_steps=0,real_accessed=False,warning='GT assisted crops/mask query; same points and original box denominator, not detection PCK. CF are untrained readouts; never choose per-point best.',
                input='V1 each physical image; exact saved affine; separately re-encoded; fixed all24 source records/56 targets',
                replay='Original and both transformed trained_final body xy and matching must exactly replay stage22; full-head deterministic local replay tolerance1e-6'))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False);modelhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})
            count=0;replay=[]
            def infer(rgb):
                nonlocal count
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('ROI readout cutoff')
                h,w=rgb.shape[:2];x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None,None].cuda()
                coordinates,pred,error=frozen_readouts(model,x);replay.append(error)
                masks=(F.interpolate(pred['instance_mask_logits'][0,0][:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0).numpy()
                count+=1;return {a:xy[0].numpy()*[w,h] for a,xy in coordinates.items()},masks
            results=[];seen=set()
            for index,e in enumerate(plan['selected']):
                record=plan['records'][e['key']];batch=oriented_batch(record,num_views=len(record['views']));target=instance_targets_from_batch(batch)
                for v in range(batch['images'].shape[1]):
                    rgb=batch['images'][0,v].permute(1,2,0).numpy();h,w=rgb.shape[:2];original=infer(rgb)
                    masks=target['masks'][0,v].numpy().astype(bool)
                    for m in np.flatnonzero(target['instance_valid'][0,v].numpy()):
                        key=(index,record['views'][v],int(m));prior=references['zoom'][key]
                        ids=prior['body_indices'];gt=target['keypoint_xy'][0,v,m].numpy()*[w,h]
                        np.testing.assert_array_equal(gt[ids],prior['gt_body_xy'])
                        for geometry in ('original','center','zoom'):
                            ref=prior if geometry=='original' else references[geometry][key]
                            affine=np.eye(3) if geometry=='original' else np.array(ref['affine'])
                            if geometry=='original':im=rgb;current=original;current_masks=masks
                            else:
                                im,current_masks,_=warp_rgb_masks(rgb,masks,affine);current=infer(im)
                            q,ious=mask_query_iou(current[1],current_masks[m]);qname='original_query' if geometry=='original' else 'roi_query'
                            assert q==ref[qname]
                            np.testing.assert_array_equal(ious,ref['original_mask_ious' if geometry=='original' else 'roi_mask_ious'])
                            mapped={a:inverse_points(xy[q],affine) for a,xy in current[0].items()}
                            np.testing.assert_array_equal(mapped['trained_final'][ids],ref['original_body_xy' if geometry=='original' else 'roi_body_xy_in_original'])
                            for arm,xy in mapped.items():
                                errors=np.linalg.norm(xy[ids]-gt[ids],axis=-1)/ref['reference_box_diag_px']
                                results.append(dict(index=index,group=e['group'],physical_role=record['physical_roles'][v],view=record['views'][v],instance=int(m),
                                    geometry=geometry,arm=arm,query=q,body_indices=ids,errors=errors.tolist(),body_nme=float(errors.mean()),body_pck=float((errors<=.1).mean()),
                                    xy_original=xy[ids].tolist(),gt_original=gt[ids].tolist()))
                            if e['group'] not in seen:
                                cols=[];truth=transform_xy(gt,affine)
                                for arm in ARMS:
                                    canvas=cv2.cvtColor(np.uint8(np.clip(im*255,0,255)),cv2.COLOR_RGB2BGR)
                                    for k in ids:
                                        cv2.circle(canvas,tuple(np.rint(truth[k]).astype(int)),5,(0,255,0),1)
                                        cv2.circle(canvas,tuple(np.rint(current[0][arm][q,k]).astype(int)),3,(0,255,255),-1)
                                    cv2.putText(canvas,arm,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
                                    cols.append(cv2.resize(canvas,(384,round(h*384/w))))
                                cv2.imwrite(str(OUT/f'review/{e["group"]}_v{v}_m{m}_{geometry}.png'),np.concatenate(cols,1))
                seen.add(e['group']);status('evaluating',records=index+1,total=24,encoder_calls=count)
            summary={}
            for group,role in sorted({(r['group'],r['physical_role']) for r in results}):
                groupkey=group+'/'+role;summary[groupkey]={}
                for geometry in ('original','center','zoom'):
                    summary[groupkey][geometry]={}
                    for arm in ARMS:
                        rows=[r for r in results if (r['group'],r['physical_role'],r['geometry'],r['arm'])==(group,role,geometry,arm)]
                        err=np.concatenate([r['errors'] for r in rows]);summary[groupkey][geometry][arm]=dict(n=len(rows),points=len(err),body_nme=float(err.mean()),body_pck=float((err<=.1).mean()))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==modelhash
            assert all(p.grad is None for p in model.parameters())
            assert all(fingerprint(ROOT/p)==h for p,h in {**guards,**codehash}.items())
            write_json(OUT/'records.json',results);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,weights_unchanged=True,real_accessed=False,production_promoted=False,
                selected_records=24,encoder_calls=count,all_three_geometries_prior_final_replay_exact=True,full_head_local_max_error=max(replay)))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()
