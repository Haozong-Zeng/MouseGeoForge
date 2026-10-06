"""Frozen real-cylinder development diagnostics; no GT accuracy or training."""
import copy,json,time
from collections import defaultdict
import cv2,numpy as np,torch
from scipy.stats import spearmanr
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,fingerprint,write_json
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.raw_validation import RawVideoValidation
from vggt_omega.utils.pose_enc import encoding_to_camera
from add_dannce_real_multiview_pseudo_depth import read_pfm

OUT=ROOT/'output/real_cylinder_bottom_frozen_v1'
CONDITIONS={'bottom':['A'],'side':['D'],'side_bottom':['D','A'],'sides':['D','E'],
            'sides_bottom':['D','E','A'],'bottom_first':['A','D','E']}
SOURCES={'AZL3':[780,796,811],'CR2':[0,780,1559],'FR2':[0,780,1559]}
BODY=list(range(7))+list(range(14,23))


def resize_image(im):
    # cv2 half-pixel convention and exact integer resize, centered patch padding.
    h,w=im.shape[:2];nh=768;nw=round(w*nh/h);pad=(-nw)%16;left=pad//2
    resized=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA)
    out=cv2.copyMakeBorder(resized,0,0,left,pad-left,cv2.BORDER_CONSTANT,value=(0,0,0))
    sx=nw/w;sy=nh/h
    transform=np.array([[sx,0,left+(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    return out,transform,(nw,nh,left,pad-left)


def resize_field(im,geometry):
    nw,nh,left,right=geometry
    a=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(a,0,0,left,right,cv2.BORDER_CONSTANT,value=0)


def rotation_error(pred,truth):
    return float(np.degrees(np.arccos(np.clip((np.trace(pred@truth.T)-1)/2,-1,1))))


def safe_rank(a,b,mask):
    if mask.sum()<50 or np.ptp(a[mask])<1e-6 or np.ptp(b[mask])<1e-6:return None
    result=float(spearmanr(a[mask],b[mask]).statistic)
    return result if np.isfinite(result) else None


def main():
    starttime=time.time();torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'review').mkdir();(OUT/'predictions').mkdir()
    base=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    paths={'base768':ROOT/'output/pose_diverse_core23_768_v1/seed17_off/final_heads.pt',
           'cylinder800':ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt'}
    hashes={n:fingerprint(p) for n,p in paths.items()};hashes['encoder']=fingerprint(base)
    protocol=dict(sources=SOURCES,conditions=CONDITIONS,head_paths={n:str(p.relative_to(ROOT)) for n,p in paths.items()},hashes=hashes,
        weights_updated=False,holdout_accessed=False,manual_accuracy_metrics=False,pseudo_depth_accuracy=False,
        source_rgb='original decode + one lens remap under user-provided unrectified assumption',
        native_wh=[1080,1200],network_hw=[768,704],content_wh=[691,768],padding_lr=[6,7],
        threshold=.5,top_slot='highest objectness only; no GT/mask matching',
        limits=['DLC and supplied masks are predictions, not GT','World calibration/synchronization/refraction unverified',
                'AZL3 three nearby frames are not independent poses','Two historical checkpoints differ in training, not a pure temperature ablation',
                'Known cameras used for diagnosis only, not supplied to encoder'])
    write_json(OUT/'protocol.json',protocol)
    spatial=VGGTOmega(enable_multi_instance_pose=True,num_keypoints=23)
    loaded=spatial.load_state_dict(torch.load(base,map_location='cpu',weights_only=True,mmap=True),strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    original=copy.deepcopy(spatial.dense_head).cuda().eval();camera=spatial.camera_head.cuda().eval()
    model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval().requires_grad_(False)
    states={n:torch.load(p,map_location='cpu',weights_only=True)['model'] for n,p in paths.items()}
    reader=RawVideoValidation(ROOT/'dataset/PyMouseStudioRaw/_validation_v2/sequences.jsonl',ROOT)
    rows=[];comparisons={};forward_count=0
    for session,frames in SOURCES.items():
        for f in frames:
            sample=reader.read(f'pymousestudio_raw/{session}',f,cameras=['A','D','E'],undistort=True)
            images={};transforms={};fields={};pseudo={}
            for j,n in enumerate(sample['camera_names']):
                im,T,geo=resize_image(sample['images'][j]);images[n]=im;transforms[n]=T
                fields[n]=resize_field(sample['foreground_mask_uint8'][j],geo)>127 if sample['mask_valid'][j] else None
                p=ROOT/f'dataset/PyMouseStudio/sessions/{session}/reconstruction_unified_guarded_v1/frame_{f:06d}/rendered_depth_maps/{"ADE".index(n) if n=="A" else (3 if n=="D" else 4):08d}.pfm'
                if p.exists():
                    pd=resize_field(read_pfm(p),geo);pseudo[n]=pd
                    if fields[n] is None:fields[n]=pd>0
                cv2.imwrite(str(OUT/f'review/{session}_{f:06d}_{n}_input.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            for condition,names in CONDITIONS.items():
                tensor=torch.from_numpy(np.stack([images[n] for n in names])).permute(0,3,1,2).unsqueeze(0).cuda().float()/255
                with torch.inference_mode():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cached,tokenstart=model.aggregator(tensor)
                    pretrained,_=original(cached,tensor,tokenstart)
                    pose=camera(cached,patch_token_start=tokenstart);ep,_=encoding_to_camera(pose,tensor.shape[-2:])
                    pretrained=pretrained[0,...,0].cpu().numpy();ep=ep[0].cpu().numpy()
                    outputs={}
                    for name,state in states.items():
                        restore_adaptive(model,state)
                        outputs[name]={k:v.cpu() for k,v in model.forward_cached(tensor,cached,tokenstart).items()
                            if k in ['depth','instance_objectness_logits','instance_boxes_cxcywh','instance_keypoint_xy','instance_keypoint_visibility_logits','instance_mask_logits']}
                forward_count+=1;plotrows=[]
                for j,n in enumerate(names):
                    im=images[n];h,w=im.shape[:2];fg=fields[n];erow=[]
                    baseindex=sample['camera_names'].index(names[0]);index=sample['camera_names'].index(n)
                    gtR=sample['world_to_camera_mm'][index,:3,:3]@sample['world_to_camera_mm'][baseindex,:3,:3].T
                    predR=ep[j,:3,:3]@ep[0,:3,:3].T
                    for modelname,output in outputs.items():
                        scores=output['instance_objectness_logits'][0,j].sigmoid().numpy().reshape(-1);q=int(scores.argmax())
                        points=output['instance_keypoint_xy'][0,j,q].numpy()*[w,h]
                        box=output['instance_boxes_cxcywh'][0,j,q].numpy()*[w,h,w,h]
                        logits=output['instance_mask_logits'][0,j,q].numpy()
                        predmask=cv2.resize(logits,(w,h),interpolation=cv2.INTER_LINEAR)>0
                        depth=output['depth'][0,j,...,0].numpy()
                        top=dict(session=session,frame=f,condition=condition,camera=n,model=modelname,top_score=float(scores[q]),
                            count_above_half=int((scores>=.5).sum()),top_query=q,points_px=points.tolist(),box_cxcywh_px=box.tolist(),
                            original_relative_rotation_error_deg=rotation_error(predR,gtR) if j else None,
                            predicted_mask_pixels=int(predmask.sum()),source_mask_kind='pseudo_hull' if n=='A' and fg is not None else 'unverified_supplied' if fg is not None else None)
                        if fg is not None:
                            inter=np.count_nonzero(fg&predmask);union=np.count_nonzero(fg|predmask)
                            top['mask_pseudo_iou']=float(inter/max(1,union))
                            xy=np.rint(points[BODY]).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
                            top['body_points_in_pseudo_foreground']=float(np.mean(fg[xy[:,1],xy[:,0]]))
                        if n in pseudo:
                            support=cv2.erode(np.uint8(pseudo[n]>0),np.ones((7,7),np.uint8))>0
                            top['pretrained_depth_rank_vs_pseudo']=safe_rank(pretrained[j],pseudo[n],support)
                            top['trained_depth_rank_vs_pseudo']=safe_rank(depth,pseudo[n],support)
                        rows.append(top);comparisons[session,f,condition,n,modelname]=top
                        canvas=cv2.cvtColor(im,cv2.COLOR_RGB2BGR);cx,cy,bw,bh=box
                        cv2.rectangle(canvas,(round(cx-bw/2),round(cy-bh/2)),(round(cx+bw/2),round(cy+bh/2)),(255,0,255),2)
                        for k in BODY:
                            x,y=np.rint(points[k]).astype(int);cv2.circle(canvas,(x,y),4,(0,255,255),-1)
                            cv2.putText(canvas,str(k),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
                        cv2.rectangle(canvas,(0,0),(w,50),(30,30,30),-1)
                        cv2.putText(canvas,f'{modelname} {n} score={scores[q]:.3f} N={int((scores>=.5).sum())}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                        erow.append(cv2.resize(canvas,(352,384)))
                    d=pretrained[j];valid=(fg if fg is not None else np.ones_like(d,bool))&np.isfinite(d)
                    lo,hi=np.percentile(d[valid],[2,98]);colored=cv2.applyColorMap(np.uint8(np.clip((d-lo)/max(1e-6,hi-lo)*255,0,255)),cv2.COLORMAP_TURBO)
                    cv2.putText(colored,f'Pretrained depth {n} relative',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                    erow.append(cv2.resize(colored,(352,384)));plotrows.append(np.concatenate(erow,1))
                cv2.imwrite(str(OUT/f'review/{session}_{f:06d}_{condition}.png'),np.concatenate(plotrows,0))
                torch.save(dict(camera_order=names,pretrained_depth=torch.from_numpy(pretrained),camera_extrinsics=torch.from_numpy(ep),outputs=outputs),OUT/f'predictions/{session}_{f:06d}_{condition}.pt')
                write_json(OUT/'progress.json',dict(forwards=forward_count,total=54,session=session,frame=f,condition=condition))
                print('REAL',forward_count,session,f,condition,flush=True)
                del cached,tensor,outputs,pose
    changes=[]
    for session,frames in SOURCES.items():
        for f in frames:
            for modelname in paths:
                for c1,c2,n in [('bottom','side_bottom','A'),('bottom','sides_bottom','A'),('side','sides_bottom','D'),('sides','sides_bottom','D'),('sides_bottom','bottom_first','A'),('sides_bottom','bottom_first','D')]:
                    a=comparisons[session,f,c1,n,modelname];b=comparisons[session,f,c2,n,modelname]
                    changes.append(dict(session=session,frame=f,model=modelname,camera=n,from_condition=c1,to_condition=c2,
                        top_slot_body_mean_displacement_px=float(np.linalg.norm(np.array(a['points_px'])[BODY]-np.array(b['points_px'])[BODY],axis=-1).mean()),
                        score_before=a['top_score'],score_after=b['top_score'],note='Context sensitivity, not accuracy; top slot may switch'))
    grouped=defaultdict(list)
    for row in rows:grouped[row['model'],row['condition'],row['camera']].append(row)
    summary=[]
    for (m,c,n),rr in grouped.items():
        summary.append(dict(model=m,condition=c,camera=n,images=len(rr),mean_top_score=float(np.mean([r['top_score'] for r in rr])),
            zero_detections=sum(r['count_above_half']==0 for r in rr),multiple_detections=sum(r['count_above_half']>1 for r in rr),
            mean_relative_rotation_error_deg=float(np.mean([r['original_relative_rotation_error_deg'] for r in rr])) if rr[0]['original_relative_rotation_error_deg'] is not None else None))
    write_json(OUT/'per_view.json',rows);write_json(OUT/'context_changes.json',changes)
    assert forward_count==54 and all(fingerprint(p)==hashes[n] for n,p in paths.items()) and fingerprint(base)==hashes['encoder']
    write_json(OUT/'summary.json',dict(state='complete',forwards=forward_count,head_view_records=len(rows),summary=summary,
        elapsed_seconds=time.time()-starttime,weights_unchanged=True,holdout_accessed=False,train_allowed=False))
    print('DONE',time.time()-starttime,flush=True)


if __name__=='__main__':main()
