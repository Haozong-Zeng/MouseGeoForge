"""Frozen GT-mask identity audit. Never changes pose evaluation assignment."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.per_view_affine_batch import augment_per_view
from mousegeoforge.pilot_metrics import evaluate_scene

OUT=ROOT/'output/wide_instance_assignment_audit_v1'
PILOT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
PLAN=ROOT/'output/wide_position_library_v1/plan.json'


def iou_matrix(left,right):
    l=left.reshape(len(left),-1).float();r=right.reshape(len(right),-1).float()
    inter=l@r.T;return inter/(l.sum(-1)[:,None]+r.sum(-1)[None]-inter).clamp_min(1)


def assign(iou):
    rows,cols=linear_sum_assignment(-iou.numpy())
    return {int(r):int(c) for r,c in zip(rows,cols)}


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            p=json.loads((PILOT/'protocol.json').read_text());plan=json.loads(PLAN.read_text())
            assert json.loads((PILOT/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/path)==s for path,s in p['code_sha256'].items())
            assert plan['parent_protocol_sha256']==fingerprint(PILOT/'protocol.json')
            assert json.loads((PLAN.parent/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/path)==h for path,h in plan['code_sha256'].items())
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            assert fingerprint(base)==p['base_sha256'] and fingerprint(encoder)==p['encoder_sha256']
            write_json(OUT/'protocol.json',dict(pilot_protocol_sha256=fingerprint(PILOT/'protocol.json'),plan_sha256=fingerprint(PLAN),
                training_steps=0,synthetic_train_scenes=32,variants=plan['variants'],real_images_accessed=False,
                box_assignment='Unchanged evaluate_scene box L1*5 minus objectness Hungarian',
                mask_assignment='Maximum sum GT instance/predicted binary mask IoU; diagnostic only, no keypoint coordinates',
                strong_conflict='mask optimum IoU>=0.5 and >=0.2 higher than box-assigned IoU',
                cross_affine='All-query one-to-one mask IoU between affine warped original prediction and new prediction',
                caveat='GT masks help inspect synthetic training target identity, not an alternate reported pose metric. High overlap/duplicate predictions can be ambiguous.'))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del weights,spatial
            model.requires_grad_(False);before=tensor_mapping_sha256(model.state_dict())
            rows=[];instance_rows=[];examples=0;n=0
            for key in p['train_keys']:
                record=p['records'][key];original_batch=oriented_batch(record,num_views=len(record['views']));identity=None
                for planned in [r for r in plan['records'] if r['key']==key and r['accepted']]:
                    variant=planned['variant']
                    if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('Audit cutoff')
                    batch,info=augment_per_view(original_batch,planned['affines'])
                    with torch.inference_mode():pred={k:v.cpu() for k,v in model(batch['images'].cuda(),coupling='off').items()}
                    targets=instance_targets_from_batch(batch);_,matches=evaluate_scene(pred,batch,targets)
                    h,w=batch['images'].shape[-2:];current=[]
                    for v,(queries,instances) in enumerate(matches):
                        masks=F.interpolate(pred['instance_mask_logits'][0,v][:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0
                        gt=batch['masks'][0,v,instances];ious=iou_matrix(gt,masks);maskassignment=assign(ious)
                        boxmap=dict(zip(instances,queries));maskmap={instances[i]:q for i,q in maskassignment.items()}
                        current.append(dict(masks=masks,boxmap=boxmap,maskmap=maskmap))
                        if identity is None:
                            track={q:q for q in range(len(masks))};track_iou=torch.eye(len(masks))
                        else:
                            oldmasks=identity[v]['masks']
                            warped=torch.from_numpy(np.stack([cv2.warpAffine(m.numpy().astype(np.uint8),np.array(info['affines'][v])[:2],(w,h),flags=cv2.INTER_NEAREST)>0 for m in oldmasks]))
                            track_iou=iou_matrix(warped,masks);track=assign(track_iou)
                        row=dict(key=key,scene=record['scene'],variant=variant['name'],view_index=v,physical_role=record['physical_roles'][v],
                                 instance_count=len(instances),gt_instance_indices=instances,box_queries=queries,mask_queries=[maskmap[m] for m in instances],
                                 gt_pred_mask_iou=ious.tolist(),query_track_iou=track_iou.tolist(),query_track=track,
                                 objectness=pred['instance_objectness_logits'][0,v].sigmoid().tolist())
                        rows.append(row);conflict=False
                        for i,m in enumerate(instances):
                            qb=boxmap[m];qm=maskmap[m];ib=float(ious[i,qb]);im=float(ious[i,qm])
                            strong=im>=.5 and im-ib>=.2;conflict|=strong
                            oldqb=qb if identity is None else identity[v]['boxmap'][m]
                            oldqm=qm if identity is None else identity[v]['maskmap'][m]
                            instance_rows.append(dict(key=key,variant=variant['name'],view_index=v,instance_count=len(instances),instance=m,
                                box_query=qb,mask_query=qm,box_mask_disagree=qb!=qm,box_assigned_iou=ib,mask_assigned_iou=im,strong_conflict=strong,
                                box_identity_track_switch=track[oldqb]!=qb,mask_identity_track_switch=track[oldqm]!=qm,
                                box_track_support=float(track_iou[oldqb,track[oldqb]]),mask_track_support=float(track_iou[oldqm,track[oldqm]])))
                        if examples<8 and (conflict or (len(instances)==2 and examples<2)):
                            rgb=np.uint8(np.clip(batch['images'][0,v].permute(1,2,0).numpy()*255,0,255));panels=[]
                            for q,mask in enumerate(masks):
                                canvas=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                                canvas[mask.numpy()]=np.uint8(.65*canvas[mask.numpy()]+.35*np.array([0,255,255]))
                                for i,m in enumerate(instances):
                                    contours,_=cv2.findContours(gt[i].numpy().astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(canvas,contours,-1,[(0,255,0),(255,100,0)][i],2)
                                cv2.rectangle(canvas,(0,0),(w,52),(20,20,20),-1)
                                cv2.putText(canvas,f'q{q} box:{boxmap} mask:{maskmap}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                                cv2.putText(canvas,variant['name']+' GT outlines; predicted yellow fill',(5,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                                panels.append(cv2.resize(canvas,(384,round(h*384/w))))
                            cv2.imwrite(str(OUT/f'review/example_{examples:02d}.png'),np.concatenate(panels,1));examples+=1
                    if identity is None:identity=current
                    n+=1;status('auditing',scene_variants=n,total=160)
            summary={}
            for count in (1,2):
                for variant_group in ('identity','augmented'):
                    selected=[r for r in instance_rows if r['instance_count']==count and ((r['variant']=='identity')==(variant_group=='identity'))]
                    summary[f'{count}mouse_{variant_group}']=dict(n=len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in
                        ('box_mask_disagree','strong_conflict','box_assigned_iou','mask_assigned_iou','box_identity_track_switch','mask_identity_track_switch','box_track_support','mask_track_support')})
            assert tensor_mapping_sha256(model.state_dict())==before
            assert fingerprint(base)==p['base_sha256'] and fingerprint(encoder)==p['encoder_sha256']
            write_json(OUT/'view_records.json',rows);write_json(OUT/'instance_records.json',instance_rows);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,variants=n,view_records=len(rows),instance_records=len(instance_rows),training_steps=0,
                 weights_unchanged=True,real_images_accessed=False,eval_matching_unchanged=True))
            status('completed');print(json.dumps(summary,indent=2),flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()


