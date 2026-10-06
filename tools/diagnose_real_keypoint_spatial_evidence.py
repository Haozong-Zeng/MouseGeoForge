"""Frozen keypoint coarse/local/global evidence diagnosis on real development."""
import fcntl
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,CORE23
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.spatial_heatmap_probe import spatial_expectation

SOURCE=ROOT/'output/real_cylinder_bottom_frozen_v1'
OUT=ROOT/'output/real_keypoint_spatial_evidence_v1'
DEADLINE=1789233567  # 2026-09-12 17:19:27 UTC
BODY=list(range(7))+list(range(14,23))


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kwargs):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kwargs))
        status('preparing')
        hooks=[]
        try:
            protocol=json.loads((SOURCE/'protocol.json').read_text())
            conditions={k:protocol['conditions'][k] for k in ('bottom','side','sides_bottom')}
            write_json(OUT/'protocol.json',dict(sources=protocol['sources'],conditions=conditions,
                checkpoint=protocol['head_paths']['cylinder800'],source_protocol_sha256=fingerprint(SOURCE/'protocol.json'),
                code_sha256=fingerprint(Path(__file__)),steps=0,holdout_accessed=False,
                warning='Global reuse of local matching is untrained. Mask-prior expectation is diagnostic and can collapse joints; not GT or amodal decoding.',
                temperature='Existing per-keypoint temperature, unchanged; no sweep or tuning'))
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(ROOT/protocol['head_paths']['cylinder800'],map_location='cpu',weights_only=True)['model'])
            del weights,spatial
            digest=tensor_mapping_sha256(model.state_dict())
            captured={}
            def hook(name):
                def save(module,args,result):captured[name]=result.detach()
                return save
            hooks=[model.instance_pose_head.local_query_projection.register_forward_hook(hook('queries')),
                   model.instance_pose_head.local_pixel_projection.register_forward_hook(hook('pixels'))]
            rows=[];saved={};count=0
            items=[(s,f,c,names,0) for s,frames in protocol['sources'].items() for f in frames for c,names in conditions.items()]
            items += [('AZL3',f,'bottom_shift',['A'],64) for f in protocol['sources']['AZL3']]
            for session,frame,condition,cameras,shift in items:
                if time.time()>=DEADLINE:raise TimeoutError('Authorized window ended')
                rgb=[cv2.cvtColor(cv2.imread(str(SOURCE/f'review/{session}_{frame:06d}_{cam}_input.png')),cv2.COLOR_BGR2RGB) for cam in cameras]
                if shift:rgb=[cv2.warpAffine(im,np.float32([[1,0,shift],[0,1,0]]),(im.shape[1],im.shape[0])) for im in rgb]
                x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2)[None].cuda().float()/255
                with torch.inference_mode():
                    pred=model(x,coupling='off')
                    pixel=F.normalize(captured['pixels'],dim=1,eps=1e-6)
                    query=F.normalize(captured['queries'],dim=-1,eps=1e-6)
                    scale=model.instance_pose_head.last_temperature_scale
                    for j,cam in enumerate(cameras):
                        h,w=rgb[j].shape[:2]
                        q=int(pred['instance_objectness_logits'][0,j].argmax())
                        coarse=pred['instance_keypoint_coarse_xy'][0,j,q]
                        refined=pred['instance_keypoint_xy'][0,j,q]
                        mask=pred['instance_mask_logits'][0,j,q]
                        global_logits=torch.einsum('kd,dhw->khw',query[j,q],pixel[j])*scale[j,q,:,None,None]
                        global_xy=spatial_expectation(coarse,global_logits,enabled=True)
                        mask_xy=spatial_expectation(coarse,global_logits,enabled=True,mask_logits=mask)
                        gh,gw=mask.shape
                        peak=global_logits.flatten(-2).argmax(-1)
                        peak_xy=torch.stack(((peak%gw+.5)/gw,(peak//gw+.5)/gh),-1)
                        local=pred['instance_keypoint_local_heatmap_logits'][0,j,q]
                        prob=local.flatten(-2).softmax(-1)
                        entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/np.log(prob.shape[-1])
                        radius=model.instance_pose_head.local_window_radius
                        binary=mask>0
                        coarse_inside=F.grid_sample(binary.float()[None,None],(coarse*2-1)[None,None],mode='nearest',align_corners=False)[0,0,0]>0
                        axis=torch.linspace(-radius,radius,local.shape[-1],device=x.device)
                        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
                        grid=coarse[:,None,None]+torch.stack((xx,yy),-1)
                        sampled=F.grid_sample(binary.float()[None,None],(grid*2-1).reshape(1,23*len(axis),len(axis),2),mode='nearest',align_corners=False).reshape(23,-1)
                        span=lambda points:float((points[BODY].std(dim=0)*points.new_tensor([w,h])).norm())
                        row=dict(session=session,frame=frame,condition=condition,camera=cam,query=q,
                            local_body_entropy=float(entropy[BODY].mean()),local_body_peak_probability=float(prob[BODY].max(-1).values.mean()),
                            coarse_to_refined_body_px=float(((coarse-refined)[BODY]*coarse.new_tensor([w,h])).norm(dim=-1).mean()),
                            coarse_body_inside_predicted_mask=float(coarse_inside[BODY].float().mean()),
                            body_windows_with_any_predicted_mask=float((sampled[BODY].sum(-1)>0).float().mean()),
                            local_sampling_foreground_fraction=float(sampled[BODY].mean()),
                            temperature_body_mean=float(scale[j,q,BODY].mean()),
                            refined_body_spread_px=span(refined),global_body_spread_px=span(global_xy),mask_prior_body_spread_px=span(mask_xy))
                        rows.append(row)
                        data={name:value.cpu() for name,value in dict(coarse=coarse,refined=refined,global_xy=global_xy,mask_prior_xy=mask_xy,peak_xy=peak_xy,mask_logits=mask).items()}
                        data['query']=q;saved[f'{session}_{frame:06d}_{condition}_{cam}']=data
                        panels=[]
                        for label,coords,color in [('original',refined,(0,255,255)),('global expectation',global_xy,(255,255,0)),('mask prior: diagnostic',mask_xy,(255,0,255)),('global peak: diagnostic',peak_xy,(0,160,255))]:
                            canvas=cv2.cvtColor(rgb[j],cv2.COLOR_RGB2BGR)
                            for k in BODY:
                                xy=tuple(np.rint(coords[k].cpu().numpy()*[w,h]).astype(int))
                                cv2.circle(canvas,xy,4,color,-1)
                            cv2.rectangle(canvas,(0,0),(w,35),(20,20,20),-1)
                            cv2.putText(canvas,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                            panels.append(cv2.resize(canvas,(352,384)))
                        cv2.imwrite(str(OUT/f'review/{session}_{frame:06d}_{condition}_{cam}.png'),np.concatenate(panels,1))
                        if condition=='bottom' and session=='FR2' and frame==780:
                            hp=[]
                            for k in (0,14,15,16,17,22):
                                z=global_logits[k].cpu().numpy();lo,hi=np.percentile(z,[2,98])
                                heat=cv2.applyColorMap(np.uint8(np.clip((z-lo)/max(hi-lo,1e-6),0,1)*255),cv2.COLORMAP_TURBO)
                                heat=cv2.resize(heat,(w,h));base=cv2.cvtColor(rgb[j],cv2.COLOR_RGB2BGR)
                                heat=cv2.addWeighted(base,.5,heat,.5,0)
                                cv2.putText(heat,CORE23[k],(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                                hp.append(cv2.resize(heat,(352,384)))
                            cv2.imwrite(str(OUT/'review/FR2_000780_global_heatmaps.png'),np.concatenate(hp,1))
                count+=1;status('running',forwards=count,total=len(items));captured.clear()
                del pred,pixel,query,x
            translation=[]
            for frame in protocol['sources']['AZL3']:
                a=saved[f'AZL3_{frame:06d}_bottom_A'];b=saved[f'AZL3_{frame:06d}_bottom_shift_A']
                # Single mouse top slot only: report switch explicitly, do not use GT matching.
                r=dict(frame=frame,query_before=a['query'],query_after=b['query'])
                for key in ('coarse','refined','global_xy','mask_prior_xy','peak_xy'):
                    delta=(b[key]-a[key])[BODY]*torch.tensor([704,768])
                    r[key+'_residual_px']=float((delta-torch.tensor([64,0])).norm(dim=-1).mean())
                translation.append(r)
            assert tensor_mapping_sha256(model.state_dict())==digest
            assert all(p.grad is None for p in model.parameters())
            grouped=[]
            for condition,cams in conditions.items():
                for cam in cams:
                    selected=[r for r in rows if r['condition']==condition and r['camera']==cam]
                    keys=[k for k in selected[0] if k not in ('session','frame','condition','camera','query')]
                    grouped.append(dict(condition=condition,camera=cam,n=len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
            torch.save(saved,OUT/'readouts.pt');write_json(OUT/'per_view.json',rows)
            write_json(OUT/'summary.json',dict(groups=grouped,translation=translation,weights_unchanged=True,manual_accuracy=False))
            status('completed',forwards=count,total=len(items));print(json.dumps(dict(groups=grouped,translation=translation)),flush=True)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            for handle in hooks:handle.remove()


if __name__=='__main__':main()
