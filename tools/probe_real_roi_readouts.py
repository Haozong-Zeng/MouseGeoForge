"""GT-free predicted-mask ROI visual diagnostic, not real pose accuracy."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor
from mousegeoforge.roi_scale_diagnostic import warp_rgb_masks,inverse_points,mask_query_iou,BODY
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.frozen_spatial_readouts import ARMS,frozen_readouts

OUT=ROOT/'output/real_roi_readouts_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'


def proposal(mask,max_scale):
    opened=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    fallback=not opened.any()
    if fallback:opened=mask.astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(opened,8)
    if n<2:raise ValueError('No nonempty predicted component')
    chosen=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));core=labels==chosen
    yy,xx=np.where(core);low=np.array([xx.min(),yy.min()]);high=np.array([xx.max(),yy.max()]);center=(low+high)/2
    h,w=mask.shape;extent=np.maximum(high-low,1);scale=min(max_scale,(w-1)/(1.5*extent[0]),(h-1)/(1.5*extent[1])) if max_scale>1 else 1.
    shift=np.array([(w-1)/2,(h-1)/2])-scale*center;a=np.array([[scale,0,shift[0]],[0,scale,shift[1]],[0,0,1.]])
    return a,dict(scale=float(scale),opened_component_area=int(core.sum()),original_area=int(mask.sum()),fallback_unopened=fallback,extent=extent.tolist())


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
            real=json.loads((REAL/'protocol.json').read_text());assert not set(real['sources'])&{'GR2','UR2','UR3'}
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';candidate=PARENT/'scale_sigma_step800.pt'
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate)}
            code=['tools/probe_real_roi_readouts.py','mousegeoforge/frozen_spatial_readouts.py','mousegeoforge/roi_scale_diagnostic.py']
            codehash={p:fingerprint(ROOT/p) for p in code}
            write_json(OUT/'protocol.json',dict(sources=real['sources'],camera='A',input_hw=[768,704],readouts=ARMS,guards=guards,code_sha256=codehash,
                proposal='Every original query with objectness>=.5 and nonempty mask; no NMS/count clamp. Elliptical15 opening + largest component for extent only; margin1.5; max3 zoom or unit-scale center.',
                matching='ROI query greatest IoU against warped ORIGINAL full predicted mask, not GT/DLC; duplicate original proposals retained.',
                warning='No manual GT or pseudo pose/depth loss. Morphology for ROI extent only, not a corrected segmentation. CF untrained; shifts are consistency, not accuracy.',
                preprocessing='Existing once-undistorted input PNGs, no second undistortion; source size unchanged; actual RGB encoding each affine.',
                training_steps=0,heldout_accessed=False,production_promoted=False))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False);modelhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})
            count=0;replay=[];imagehashes={};rows=[];all_original={}
            def infer(rgb):
                nonlocal count
                if time.monotonic()-began>900 or time.time()>1789233567:raise TimeoutError('Real ROI cutoff')
                h,w=rgb.shape[:2];x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None,None].cuda()
                xy,pred,error=frozen_readouts(model,x);replay.append(error);count+=1
                masks=(F.interpolate(pred['instance_mask_logits'][0,0][:,None],size=(h,w),mode='bilinear',align_corners=False)[:,0]>0).numpy()
                return {a:v[0].numpy()*[w,h] for a,v in xy.items()},masks,pred['instance_objectness_logits'][0,0].sigmoid().numpy()
            for session,frames in real['sources'].items():
                for frame in frames:
                    path=REAL/f'review/{session}_{frame:06d}_A_input.png';imagehashes[str(path.relative_to(ROOT))]=fingerprint(path)
                    rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB).astype(np.float32)/255;assert rgb.shape==(768,704,3)
                    original=infer(rgb);name=f'{session}_{frame:06d}';all_original[name]=dict(scores=original[2].tolist(),xy={a:v.tolist() for a,v in original[0].items()})
                    positive=np.flatnonzero(original[2]>=.5)
                    for q in positive:
                        if not original[1][q].any():
                            rows.append(dict(name=name,query=int(q),skipped_roi='empty original predicted mask',original_positives=len(positive)));continue
                        for geometry,cap in [('original',1),('center',1),('zoom',3)]:
                            if geometry=='original':affine=np.eye(3);meta={};im=rgb;pred=original;qn=int(q);ious=None;padding=0
                            else:
                                affine,meta=proposal(original[1][q],cap);im,expected,padding=warp_rgb_masks(rgb,original[1],affine)
                                pred=infer(im);qn,ious=mask_query_iou(pred[1],expected[q])
                            mapped={a:inverse_points(xy[qn],affine) for a,xy in pred[0].items()}
                            rows.append(dict(name=name,query=int(q),matched_query=qn,geometry=geometry,affine=affine.tolist(),**meta,padding_fraction=padding,
                                original_positives=len(positive),transformed_positives=int((pred[2]>=.5).sum()),mask_iou_vector=None if ious is None else ious.tolist(),
                                body_shift_px={a:float(np.linalg.norm(xy[BODY]-original[0][a][q,BODY],axis=-1).mean()) for a,xy in mapped.items()},
                                xy_original={a:xy.tolist() for a,xy in mapped.items()}))
                            # Every positive original query is exported, including duplicates.
                            cols=[]
                            for a in ARMS:
                                canvas=cv2.cvtColor(np.uint8(np.clip(im*255,0,255)),cv2.COLOR_RGB2BGR)
                                for k in BODY:
                                    pt=tuple(np.rint(pred[0][a][qn,k]).astype(int));cv2.circle(canvas,pt,3,(0,255,255),-1)
                                    cv2.putText(canvas,str(k),pt,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
                                cv2.putText(canvas,a,(6,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
                                cols.append(cv2.resize(canvas,(352,384)))
                            cv2.imwrite(str(OUT/f'review/{name}_q{q}_{geometry}.png'),np.concatenate(cols,1))
                    status('evaluating',source=name,encoder_calls=count,rows=len(rows))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==modelhash
            assert all(p.grad is None for p in model.parameters())
            assert all(fingerprint(ROOT/p)==h for p,h in {**guards,**codehash,**imagehashes}.items())
            write_json(OUT/'records.json',rows);write_json(OUT/'original_all_queries.json',all_original)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,weights_unchanged=True,source_images_sha256=imagehashes,heldout_accessed=False,
                encoder_calls=count,source_frames=len(all_original),full_head_replay_max=max(replay),production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()
