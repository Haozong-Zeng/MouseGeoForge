"""Frozen real-development affine equivariance, not manual-label accuracy."""
import fcntl,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/spatial_anchor_augmentation_affine_probe_v1'
PILOT=ROOT/'output/spatial_anchor_augmentation_deterministic_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'
BODY=list(range(7))+list(range(14,23))


def transform_points(points,affine):
    return np.asarray(points)@affine[:,:2].T+affine[:,2]


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            assert json.loads((PILOT/'verification.json').read_text())['passed']
            protocol=json.loads((PILOT/'protocol.json').read_text())
            assert all(fingerprint(ROOT/p)==h for p,h in protocol['code_sha256'].items())
            real=json.loads((REAL/'protocol.json').read_text())
            assert not set(real['sources'])&{'GR2','UR2','UR3'}
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt'
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            candidate=PILOT/'augmentation_on_step800.pt'
            candidate_off=PILOT/'augmentation_off_step800.pt'
            assert fingerprint(base)==protocol['base_sha256'] and fingerprint(encoder)==protocol['encoder_sha256']
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate,candidate_off)}
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del weights,spatial
            model.requires_grad_(False);base_hash=tensor_mapping_sha256(model.state_dict())
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False)
            states={a:torch.load(p,map_location='cpu',weights_only=True)['branch'] for a,p in [('off',candidate_off),('on',candidate)]}
            def infer(rgb):
                if time.monotonic()-began>900 or time.time()>1789233567:raise TimeoutError('Probe limit')
                x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None,None].cuda().float()/255
                with torch.inference_mode():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(x)
                    model.instance_pose_head.spatial_anchor.load_state_dict(states['off'])
                    with spatial_anchor_context(model,enabled=True):original=model.forward_cached(x,cached,start,coupling='off')
                    model.instance_pose_head.spatial_anchor.load_state_dict(states['on'])
                    with spatial_anchor_context(model,enabled=True):new=model.forward_cached(x,cached,start,coupling='off')
                    for k in ('depth','instance_boxes_cxcywh','instance_mask_logits','instance_objectness_logits'):
                        torch.testing.assert_close(original[k],new[k],atol=0,rtol=0)
                    keys=('instance_keypoint_xy','instance_keypoint_coarse_xy','instance_mask_logits','instance_objectness_logits')
                    return {a:{k:p[k][0,0].cpu().numpy() for k in keys} for a,p in [('augmentation_off',original),('augmentation_on',new)]}
            rows=[];readouts={};input_hashes={};n=0
            transforms={'left64':(1.,-64,0),'right64':(1.,64,0),'up64':(1.,0,-64),'down64':(1.,0,64),'scale090':(.9,0,0),'scale110':(1.1,0,0)}
            write_json(OUT/'protocol.json',dict(sources=real['sources'],conditions=['bottom'],transforms=transforms,
                interpolation='cv2 INTER_LINEAR RGB, INTER_NEAREST binary masks; black canvas; scaling about pixel-center canvas center',
                source_images='Previously once-rectified network input PNGs; no further undistortion',
                baseline='deterministic augmentation OFF800',candidate='deterministic augmentation ON800',weight_sha256=guards,code_sha256=fingerprint(Path(__file__)),
                primary='body mean residual from affine-transformed ORIGINAL prediction, not GT',
                matching='best IoU with affine-transformed original top-score prediction mask; no NMS or forced output count',
                heldout_accessed=False,training_steps=0,caveats=['Boundary/context changes are included.','Equivariance is not semantic accuracy.','9 reused diagnostic timepoints, 3 AZL3 adjacent.','Scaling distances reported in transformed network pixels.']))
            for session,frames in real['sources'].items():
                for frame in frames:
                    path=REAL/f'review/{session}_{frame:06d}_A_input.png';input_hashes[str(path.relative_to(ROOT))]=fingerprint(path)
                    rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];assert (h,w)==(768,704)
                    old=infer(rgb);q=int(old['augmentation_off']['instance_objectness_logits'].argmax())
                    mask=cv2.resize(old['augmentation_off']['instance_mask_logits'][q],(w,h))>0
                    panels=[]
                    for name,(scale,dx,dy) in transforms.items():
                        affine=np.float32([[scale,0,(1-scale)*(w-1)/2+dx],[0,scale,(1-scale)*(h-1)/2+dy]])
                        warped=cv2.warpAffine(rgb,affine,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
                        expectedmask=cv2.warpAffine(mask.astype(np.uint8),affine,(w,h),flags=cv2.INTER_NEAREST)>0
                        pred=infer(warped)
                        masks=[cv2.resize(m,(w,h))>0 for m in pred['augmentation_off']['instance_mask_logits']]
                        ious=[np.count_nonzero(m&expectedmask)/max(1,np.count_nonzero(m|expectedmask)) for m in masks]
                        qn=int(np.argmax(ious));cols=[]
                        for arm in ('augmentation_off','augmentation_on'):
                            originalxy=old[arm]['instance_keypoint_xy'][q]*[w,h]
                            expected=transform_points(originalxy,affine);observed=pred[arm]['instance_keypoint_xy'][qn]*[w,h]
                            expectedcoarse=transform_points(old[arm]['instance_keypoint_coarse_xy'][q]*[w,h],affine)
                            coarse=pred[arm]['instance_keypoint_coarse_xy'][qn]*[w,h]
                            inside=(expected[:,0]>=0)&(expected[:,0]<w)&(expected[:,1]>=0)&(expected[:,1]<h)
                            validbody=[k for k in BODY if inside[k]];validtail=[k for k in range(7,14) if inside[k]]
                            errors=np.linalg.norm(observed-expected,axis=-1)
                            row=dict(session=session,frame=frame,transform=name,arm=arm,affine=affine.tolist(),original_query=q,matched_query=qn,
                                matched_is_top_score=qn==int(pred[arm]['instance_objectness_logits'].argmax()),mask_equivariance_iou=float(ious[qn]),
                                mask_retained_area_ratio=float(expectedmask.sum()/max(1,mask.sum()*scale**2)),
                                body_valid_points=len(validbody),tail_valid_points=len(validtail),body_residual_px=float(errors[validbody].mean()),
                                tail_residual_px=float(errors[validtail].mean()) if validtail else None,
                                body_coarse_residual_px=float(np.linalg.norm(coarse-expectedcoarse,axis=-1)[validbody].mean()),
                                body_spread_px=float(np.linalg.norm(observed[BODY]-observed[BODY].mean(0),axis=-1).mean()),per_point_residual_px=errors.tolist())
                            rows.append(row)
                            readouts[f'{session}_{frame:06d}_{name}_{arm}']=dict(expected=expected.tolist(),observed=observed.tolist(),inside=inside.tolist())
                            if frame==780 and session in ('FR2','AZL3'):
                                canvas=cv2.cvtColor(warped,cv2.COLOR_RGB2BGR)
                                for k in BODY:
                                    p=tuple(np.rint(expected[k]).astype(int));o=tuple(np.rint(observed[k]).astype(int))
                                    cv2.line(canvas,p,o,(140,140,140),1);cv2.circle(canvas,p,5,(255,255,0),1);cv2.circle(canvas,o,3,(0,255,255),-1)
                                cv2.rectangle(canvas,(0,0),(w,58),(20,20,20),-1)
                                cv2.putText(canvas,f'{name} {arm} residual {row["body_residual_px"]:.1f}px',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                                cv2.putText(canvas,'cyan=expected prediction; yellow=new; NO GT',(8,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                                cols.append(cv2.resize(canvas,(352,384)))
                        if cols:panels.append(np.concatenate(cols,axis=1))
                        n+=1;status('probing',transform_groups=n,total=54)
                    if panels:cv2.imwrite(str(OUT/f'review/{session}_{frame:06d}_affine.png'),np.concatenate(panels,axis=0))
            summary={}
            for arm in ('augmentation_off','augmentation_on'):
                summary[arm]={}
                for name in [*transforms,'translations','scales','all']:
                    selected=[r for r in rows if r['arm']==arm and (r['transform']==name or name=='all' or (name=='translations' and '64' in r['transform']) or (name=='scales' and 'scale' in r['transform']))]
                    summary[arm][name]={k:float(np.mean([r[k] for r in selected if r[k] is not None])) for k in ('body_residual_px','tail_residual_px','body_coarse_residual_px','mask_equivariance_iou','mask_retained_area_ratio')}
                    summary[arm][name]['n']=len(selected)
            assert tensor_mapping_sha256({k:v for k,v in model.state_dict().items() if not k.startswith('instance_pose_head.spatial_anchor.')})==base_hash
            assert all(fingerprint(ROOT/p)==s for p,s in guards.items())
            assert all(fingerprint(ROOT/p)==s for p,s in input_hashes.items())
            write_json(OUT/'records.json',rows);write_json(OUT/'readouts.json',readouts);write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,unchanged_base=True,source_images_sha256=input_hashes,
                weight_files_unchanged=True,depth_boxes_masks_scores_exactly_unchanged=True,heldout_accessed=False,production_promoted=False))
            status('completed');print(json.dumps(summary,indent=2),flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()


