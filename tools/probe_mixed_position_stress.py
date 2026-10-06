"""Locked synthetic recentering and real affine stress for F/control/mixed."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor
from mousegeoforge.instance_coordinate_residual import InstanceCoordinateResidual
from mousegeoforge.frozen_coordinate_readouts import coordinate_readouts
from mousegeoforge.roi_scale_diagnostic import warp_rgb_masks,inverse_points,mask_query_iou,BODY
from mousegeoforge.affine_augmentation import transform_xy
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/mixed_position_stress_v1'
HEADS=ROOT/'output/mixed_position_residual_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            hp=json.loads((HEADS/'protocol.json').read_text());assert json.loads((HEADS/'verification.json').read_text())['passed']
            assert fingerprint(HEADS/'selection.json')==hp['selection_sha256']
            assert all(fingerprint(ROOT/p)==h for p,h in {**hp['hashes'],**hp['code_sha256']}.items())
            parent=json.loads((PARENT/'protocol.json').read_text());assert all(fingerprint(ROOT/p)==h for p,h in parent['code_sha256'].items())
            plan=json.loads((ROOT/'output/oracle_roi_scale_diagnostic_v1/protocol.json').read_text())
            center_path=ROOT/'output/oracle_roi_center_diagnostic_v1/records.json'
            reference={(r['index'],r['view'],r['instance']):r for r in json.loads(center_path.read_text())}
            real_ref_path=ROOT/'output/spatial_anchor_scale_affine_probe_v1/readouts.json';real_ref=json.loads(real_ref_path.read_text())
            real=json.loads((REAL/'protocol.json').read_text());assert not set(real['sources'])&{'GR2','UR2','UR3'}
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';candidate=PARENT/'scale_sigma_step800.pt'
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate,HEADS/'control_step400.pt',HEADS/'mixed_step400.pt',center_path,real_ref_path)}
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            code=['tools/probe_mixed_position_stress.py','mousegeoforge/frozen_coordinate_readouts.py','mousegeoforge/instance_coordinate_residual.py']
            codehash={p:fingerprint(ROOT/p) for p in code}
            write_json(OUT/'protocol.json',dict(selected=plan['selected'],real_sources=real['sources'],guards=guards,code_sha256=codehash,training_steps=0,
                models=['frozen','control','mixed'],synthetic='Same24/56 GT targets original/recenter saved affine; GT-mask query, fixed body set and original box denominator. V1.',
                real='Same9 A PNGs, uint8 OpenCV affine before GPU /255 normalization; original highest score slot, matched by warped original mask IoU. No GT; no production count change.',
                replay='Synthetic F final and query exact; real F expected/observed coordinates exact replay old scale-sigma probe.',
                warning='Synthetic oracle diagnostic, not formal detection PCK; real affine consistency not accuracy; no per-point best selection.',heldout_accessed=False,production_promoted=False))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False);modelhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})
            heads={}
            for arm in ('control','mixed'):
                decoder=InstanceCoordinateResidual(relative=True).cuda();decoder.load_state_dict(torch.load(HEADS/f'{arm}_step400.pt',map_location='cpu',weights_only=True)['state'])
                decoder.eval().requires_grad_(False);heads[arm]=decoder
            count=0;replays=[];synthetic=[];realrows=[];imagehashes={};seen=set()
            def infer(rgb,real_input=False):
                nonlocal count
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('Position probe cutoff')
                h,w=rgb.shape[:2];x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None,None].cuda().float()
                if real_input:x=x/255
                outputs,pred,error=coordinate_readouts(model,heads,x);replays.append(error);count+=1
                logits=pred['instance_mask_logits'][0,0]
                masks=np.stack([cv2.resize(m.numpy(),(w,h))>0 for m in logits]) if real_input else (F.interpolate(logits[:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0).numpy()
                return {a:(xy[0].numpy()*[w,h],coarse[0].numpy()*[w,h]) for a,(xy,coarse) in outputs.items()},masks,pred['instance_objectness_logits'][0,0].sigmoid().numpy()
            for index,e in enumerate(plan['selected']):
                rec=plan['records'][e['key']];batch=oriented_batch(rec,num_views=len(rec['views']));target=instance_targets_from_batch(batch)
                for v in range(batch['images'].shape[1]):
                    rgb=batch['images'][0,v].permute(1,2,0).numpy();h,w=rgb.shape[:2];original=infer(rgb);masks=target['masks'][0,v].numpy().astype(bool)
                    for m in np.flatnonzero(target['instance_valid'][0,v].numpy()):
                        old=reference[(index,rec['views'][v],int(m))];ids=old['body_indices'];gt=np.array(old['gt_body_xy'])
                        for geometry in ('original','center'):
                            a=np.eye(3) if geometry=='original' else np.array(old['affine'])
                            if geometry=='original':im=rgb;current=original;ms=masks;name='original'
                            else:im,ms,_=warp_rgb_masks(rgb,masks,a);current=infer(im);name='roi'
                            q,iou=mask_query_iou(current[1],ms[m]);assert q==old[name+'_query']
                            mapped={arm:inverse_points(xy[q],a) for arm,(xy,_) in current[0].items()}
                            np.testing.assert_array_equal(mapped['frozen'][ids],old['original_body_xy' if geometry=='original' else 'roi_body_xy_in_original'])
                            for arm,xy in mapped.items():
                                errors=np.linalg.norm(xy[ids]-gt,axis=-1)/old['reference_box_diag_px']
                                synthetic.append(dict(index=index,group=e['group'],physical_role=rec['physical_roles'][v],view=rec['views'][v],instance=int(m),geometry=geometry,arm=arm,
                                    errors=errors.tolist(),body_nme=float(errors.mean()),body_pck=float((errors<=.1).mean())))
                            if e['group'] not in seen:
                                panels=[];truth=transform_xy(gt,a)
                                for arm,(xy,_) in current[0].items():
                                    canvas=cv2.cvtColor(np.uint8(np.clip(im*255,0,255)),cv2.COLOR_RGB2BGR)
                                    for pt in truth:cv2.circle(canvas,tuple(np.rint(pt).astype(int)),5,(0,255,0),1)
                                    for k in ids:cv2.circle(canvas,tuple(np.rint(xy[q,k]).astype(int)),3,(0,255,255),-1)
                                    cv2.putText(canvas,arm,(6,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(cv2.resize(canvas,(384,round(h*384/w))))
                                cv2.imwrite(str(OUT/f'review/synthetic_{e["group"]}_v{v}_m{m}_{geometry}.png'),np.concatenate(panels,1))
                seen.add(e['group']);status('synthetic',records=index+1,total=24,encoder_calls=count)
            transforms={'left64':(1.,-64,0),'right64':(1.,64,0),'up64':(1.,0,-64),'down64':(1.,0,64),'scale090':(.9,0,0),'scale110':(1.1,0,0)}
            for session,frames in real['sources'].items():
                for frame in frames:
                    path=REAL/f'review/{session}_{frame:06d}_A_input.png';imagehashes[str(path.relative_to(ROOT))]=fingerprint(path)
                    rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];assert (h,w)==(768,704)
                    original=infer(rgb,True);q=int(original[2].argmax());panels=[]
                    for name,(scale,dx,dy) in transforms.items():
                        a=np.float32([[scale,0,(1-scale)*(w-1)/2+dx],[0,scale,(1-scale)*(h-1)/2+dy]])
                        warped=cv2.warpAffine(rgb,a,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
                        expectedmask=cv2.warpAffine(original[1][q].astype(np.uint8),a,(w,h),flags=cv2.INTER_NEAREST)>0
                        pred=infer(warped,True);qn,iou=mask_query_iou(pred[1],expectedmask);cols=[]
                        for arm,(xy,coarse) in pred[0].items():
                            expected=transform_xy(original[0][arm][0][q],a);observed=xy[qn]
                            if arm=='frozen':
                                old=real_ref[f'{session}_{frame:06d}_{name}_scale_sigma']
                                np.testing.assert_array_equal(expected,old['expected']);np.testing.assert_array_equal(observed,old['observed'])
                            inside=(expected[:,0]>=0)&(expected[:,0]<w)&(expected[:,1]>=0)&(expected[:,1]<h)
                            body=[k for k in BODY if inside[k]];tail=[k for k in range(7,14) if inside[k]];errors=np.linalg.norm(observed-expected,axis=-1)
                            realrows.append(dict(session=session,frame=frame,transform=name,arm=arm,body_residual_px=float(errors[body].mean()),tail_residual_px=float(errors[tail].mean()) if tail else None,
                                body_points=len(body),tail_points=len(tail),mask_iou=float(iou[qn]),original_query=q,matched_query=qn,expected=expected.tolist(),observed=observed.tolist()))
                            if session in ('AZL3','FR2') and frame==780:
                                canvas=cv2.cvtColor(warped,cv2.COLOR_RGB2BGR)
                                for k in BODY:
                                    ep=tuple(np.rint(expected[k]).astype(int));op=tuple(np.rint(observed[k]).astype(int))
                                    cv2.line(canvas,ep,op,(140,140,140),1);cv2.circle(canvas,ep,5,(255,255,0),1);cv2.circle(canvas,op,3,(0,255,255),-1)
                                cv2.putText(canvas,name+' '+arm+' NO GT',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);cols.append(cv2.resize(canvas,(352,384)))
                        if cols:panels.append(np.concatenate(cols,1))
                    if panels:cv2.imwrite(str(OUT/f'review/real_{session}_{frame:06d}_affine.png'),np.concatenate(panels,0))
                    status('real',source=f'{session}_{frame}',encoder_calls=count)
            summary={'synthetic':{},'real':{}}
            for group,role in sorted({(r['group'],r['physical_role']) for r in synthetic}):
                entry=summary['synthetic'].setdefault(group+'/'+role,{})
                for geom in ('original','center'):
                    entry[geom]={}
                    for arm in ('frozen','control','mixed'):
                        rows=[r for r in synthetic if (r['group'],r['physical_role'],r['geometry'],r['arm'])==(group,role,geom,arm)]
                        errors=np.concatenate([r['errors'] for r in rows]);entry[geom][arm]=dict(body_nme=float(errors.mean()),body_pck=float((errors<=.1).mean()),n=len(rows))
            for arm in ('frozen','control','mixed'):
                summary['real'][arm]={}
                for name in ('translations','scales'):
                    rows=[r for r in realrows if r['arm']==arm and ('64' in r['transform'])==(name=='translations')]
                    summary['real'][arm][name]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ('body_residual_px','tail_residual_px','mask_iou')}
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==modelhash
            assert all(fingerprint(ROOT/p)==h for p,h in {**guards,**codehash,**imagehashes}.items())
            write_json(OUT/'summary.json',summary);write_json(OUT/'synthetic_records.json',synthetic);write_json(OUT/'real_records.json',realrows)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,weights_unchanged=True,prior_frozen_synthetic_and_real_exact=True,source_images_sha256=imagehashes,
                encoder_calls=count,full_head_replay_max=max(replays),heldout_accessed=False,production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()



