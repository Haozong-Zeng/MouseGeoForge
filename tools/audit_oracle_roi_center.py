"""Frozen original vs GT-assisted recentering only; not deployable accuracy."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.roi_scale_diagnostic import BODY,body_roi_affine,warp_rgb_masks,inverse_points,mask_query_iou
from mousegeoforge.affine_augmentation import transform_xy
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/oracle_roi_center_diagnostic_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'


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
            spec=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
            selected=[]
            for group in ('old_dual_pair','old_cylinder_side_pair','bottom_only'):
                rows=[e for e in spec['evaluation'] if e['group']==group]
                selected.extend(rows[round(i*(len(rows)-1)/7)] for i in range(8))
            assert len(selected)==24
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            candidate=PARENT/'scale_sigma_step800.pt'
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate)}
            code=['tools/audit_oracle_roi_center.py','mousegeoforge/roi_scale_diagnostic.py','mousegeoforge/affine_augmentation.py']
            codehash={p:fingerprint(ROOT/p) for p in code}
            write_json(OUT/'protocol.json',dict(selected=selected,records={e['key']:spec['records'][e['key']] for e in selected},
                weights=guards,code_sha256=codehash,model='Frozen scale_sigma800, no additional gate',training_steps=0,real_accessed=False,
                inputs='Each physical view independently, V=1; same source raster HW for original/ROI, RGB re-encoded separately',
                roi='GT valid non-tail body bounding extent, margin1.5, unit scale, recentering only; center canvas; black padding; all queries retained',
                matching='Each target selects query by largest GT-mask IoU in both arms, no GT keypoint error selection; independent target diagnostic, not one-to-one detection',
                metric='Inverse-mapped predicted body points, common original GT box diagonal and original valid body set; no objectness/IoU threshold filtering',
                caveats=['GT crop and mask matching are oracle-assisted; not formal detection-aware PCK or deployable pipeline.',
                         'ROI changes scale, position and background context, not a pure resolution intervention.',
                         'All original targets retained, including poor masks; other mice can remain or be partially cropped.',
                         'Diagnostic reuses development sources; no independent final test.',
                         'No geometric fusion: K updated for consistency but not supplied to this encoder; E unchanged.']))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False)
            modelhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})
            count=0
            def infer(rgb):
                nonlocal count
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('ROI diagnostic cutoff')
                h,w=rgb.shape[:2];x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None,None].cuda()
                with torch.inference_mode():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cache,start=model.aggregator(x)
                    with spatial_anchor_context(model,enabled=True):pred=model.forward_cached(x,cache,start,coupling='off')
                    masks=(F.interpolate(pred['instance_mask_logits'][0,0][:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0).cpu().numpy()
                    xy=pred['instance_keypoint_xy'][0,0].cpu().numpy()*[w,h]
                    score=pred['instance_objectness_logits'][0,0].sigmoid().cpu().numpy()
                count+=1;return xy,masks,score
            reference=json.loads((ROOT/'output/oracle_roi_scale_diagnostic_v1/records.json').read_text())
            reference={(row['index'],row['view'],row['instance']):row for row in reference}
            reference_hash=fingerprint(ROOT/'output/oracle_roi_scale_diagnostic_v1/records.json')
            results=[];seen=set()
            for index,e in enumerate(selected):
                r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));target=instance_targets_from_batch(batch)
                for v in range(batch['images'].shape[1]):
                    rgb=batch['images'][0,v].permute(1,2,0).numpy();h,w=rgb.shape[:2]
                    assert max(h,w)==768
                    masks=target['masks'][0,v].numpy().astype(bool);valid=target['instance_valid'][0,v].numpy()
                    original=infer(rgb);original_queries=[]
                    for m in np.flatnonzero(valid):
                        assert target['mask_valid'][0,v,m] and masks[m].any()
                        gt=target['keypoint_xy'][0,v,m].numpy()*[w,h];pv=target['keypoint_valid'][0,v,m].numpy()
                        active=np.zeros(23,bool);active[BODY]=True;active&=pv
                        assert np.isfinite(gt[active]).all()
                        affine,meta=body_roi_affine(gt,pv,(h,w),max_scale=1.);roi,roi_masks,padding=warp_rgb_masks(rgb,masks,affine)
                        gt_roi=transform_xy(gt,affine)
                        assert ((gt_roi[active]>=0).all() and (gt_roi[active]<[w,h]).all())
                        intrinsic=batch['intrinsics'][0,v].numpy();new_k=affine@intrinsic
                        np.testing.assert_allclose(np.linalg.inv(affine)@new_k,intrinsic,atol=1e-9)
                        zoomed=infer(roi)
                        q0,iou0=mask_query_iou(original[1],masks[m]);q1,iou1=mask_query_iou(zoomed[1],roi_masks[m]);original_queries.append(q0)
                        mapped=inverse_points(zoomed[0][q1],affine)
                        diag=max(1,float(np.linalg.norm(target['boxes_cxcywh'][0,v,m,2:].numpy()*[w,h])))
                        row=dict(index=index,group=e['group'],key=e['key'],view=r['views'][v],physical_role=r['physical_roles'][v],instance=int(m),
                            affine=affine.tolist(),intrinsic_original=intrinsic.tolist(),intrinsic_roi=new_k.tolist(),extrinsic_unchanged=batch['extrinsics'][0,v].tolist(),
                            **meta,padding_fraction=padding,gt_mask_retained_ratio=float(roi_masks[m].sum()/max(1,masks[m].sum()*meta['scale']**2)),
                            original_positive_queries=int((original[2]>=.5).sum()),roi_positive_queries=int((zoomed[2]>=.5).sum()),
                            original_query=q0,roi_query=q1,original_mask_ious=iou0.tolist(),roi_mask_ious=iou1.tolist(),
                            original_objectness=float(original[2][q0]),roi_objectness=float(zoomed[2][q1]),body_indices=np.flatnonzero(active).tolist(),reference_box_diag_px=diag,
                            gt_body_xy=gt[active].tolist(),original_body_xy=original[0][q0][active].tolist(),roi_body_xy_in_original=mapped[active].tolist())
                        for arm,xy in [('original',original[0][q0]),('roi',mapped)]:
                            err=np.linalg.norm(xy[active]-gt[active],axis=-1)/diag
                            row[arm]=dict(body_nme=float(err.mean()),body_pck=float((err<=.1).mean()),errors=err.tolist())
                        ref=reference[(index,r['views'][v],int(m))]
                        assert row['scale']==1., 'Expected strict recentering, no scale change'
                        np.testing.assert_array_equal(row['original_body_xy'],ref['original_body_xy'])
                        assert row['original_query']==ref['original_query'] and row['original_mask_ious']==ref['original_mask_ious']
                        results.append(row)
                        # Fixed first record of each group; every view/instance of it, no error-based selection.
                        if e['group'] not in seen:
                            cols=[]
                            for name,im,truth,predxy in [('original',rgb,gt,original[0][q0]),('oracle centered only',roi,gt_roi,zoomed[0][q1])]:
                                canvas=cv2.cvtColor(np.uint8(np.clip(im*255,0,255)),cv2.COLOR_RGB2BGR)
                                for k in np.flatnonzero(active):
                                    g=tuple(np.rint(truth[k]).astype(int));p=tuple(np.rint(predxy[k]).astype(int))
                                    cv2.circle(canvas,g,5,(0,255,0),1);cv2.circle(canvas,p,3,(0,255,255),-1)
                                    cv2.putText(canvas,str(k),p,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
                                cv2.putText(canvas,name+' GT green / prediction yellow',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                                cols.append(canvas)
                            cv2.imwrite(str(OUT/f'review/{e["group"]}_v{v}_instance{m}.png'),np.concatenate(cols,1))
                    for row in results:
                        if row['index']==index and row['view']==r['views'][v]:row['original_gt_targets_sharing_query']=len(original_queries)!=len(set(original_queries))
                seen.add(e['group']);status('evaluating',records=index+1,total=24,instances=len(results),encoder_calls=count)
            summary={}
            for group in dict.fromkeys(row['group'] for row in results):
                rows=[row for row in results if row['group']==group]
                entry=dict(instance_views=len(rows),scale_mean=float(np.mean([row['scale'] for row in rows])),
                    original_query_collision_rows=sum(row['original_gt_targets_sharing_query'] for row in rows),
                    improved_instances=sum(row['roi']['body_nme']<row['original']['body_nme'] for row in rows),
                    low_roi_mask_iou_instances=sum(max(row['roi_mask_ious'])<.5 for row in rows))
                for arm in ('original','roi'):
                    errors=np.concatenate([row[arm]['errors'] for row in rows]);entry[arm]=dict(body_nme=float(errors.mean()),body_pck=float((errors<=.1).mean()),points=len(errors))
                summary[group]=entry
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==modelhash
            assert all(p.grad is None for p in model.parameters())
            assert all(fingerprint(ROOT/p)==h for p,h in {**guards,**codehash}.items())
            write_json(OUT/'records.json',results);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,weights_unchanged=True,real_accessed=False,production_promoted=False,
                selected_records=24,instance_views=len(results),encoder_calls=count,all_original_valid_body_points_retained=True,original_reference_replay_exact=True,reference_records_sha256=reference_hash))
            status('completed');print(json.dumps(summary,indent=2),flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()


