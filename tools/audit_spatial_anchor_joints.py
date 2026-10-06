"""Full frozen 88-record retention plus real-development visual audit."""
from functools import partial
import copy
import fcntl,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.pilot_metrics import evaluate_scene
from mousegeoforge.keypoint_breakdown import keypoint_rows,summarize_joint_rows
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor,spatial_anchor_context
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

OUT=ROOT/'output/spatial_anchor_joint_audit_v1'
PILOT=ROOT/'output/spatial_anchor_resume800_v1'
REAL=ROOT/'output/real_cylinder_bottom_frozen_v1'
BODY=list(range(7))+list(range(14,23))


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
        began=time.monotonic()
        def status(state,**kwargs):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kwargs))
        status('preparing')
        try:
            assert json.loads((PILOT/'verification.json').read_text())['passed']
            pilot=json.loads((PILOT/'protocol.json').read_text())
            assert all(fingerprint(ROOT/p)==h for p,h in pilot['code_sha256'].items())
            spec=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
            real=json.loads((REAL/'protocol.json').read_text())
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt'
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            assert fingerprint(base)==pilot['base_sha256'] and fingerprint(encoder)==pilot['encoder_sha256']
            states={a:torch.load(PILOT/f'{a}_step800.pt',map_location='cpu',weights_only=True)['branch'] for a in ('mask_prior_off','mask_prior_on')}
            states['mask_prior_on_body_only']=copy.deepcopy(states['mask_prior_on'])
            states['mask_prior_on_body_only']['gate'][7:14]=0
            write_json(OUT/'protocol.json',dict(pilot_sha256=fingerprint(PILOT/'protocol.json'),end_step=800,training_steps=0,
                synthetic_records=88,real_sources={},real_conditions=[],
                heldout_real_accessed=False,warning='Synthetic development reused; no real manual accuracy. All final800 arms, no checkpoint selection.'))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del weights,spatial
            model.requires_grad_(False);base_hash=tensor_mapping_sha256(model.state_dict())
            install_spatial_anchor(model);model.eval();branch=model.instance_pose_head.spatial_anchor
            original_forward=branch.forward
            def infer(image):
                if time.monotonic()-began>900 or time.time()>1789233567:raise TimeoutError('Audit limit')
                with torch.no_grad():
                    with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(image)
                    outputs={'baseline':{k:v.cpu() for k,v in model.forward_cached(image,cached,start,coupling='off').items()}}
                    try:
                        for arm,state in states.items():
                            branch.load_state_dict(state)
                            branch.forward=partial(original_forward,use_mask_prior=arm.startswith('mask_prior_on'))
                            with spatial_anchor_context(model,enabled=True):pred=model.forward_cached(image,cached,start,coupling='off')
                            outputs[arm]={k:v.cpu() for k,v in pred.items()}
                            for k in ('depth','instance_boxes_cxcywh','instance_mask_logits','instance_objectness_logits'):
                                torch.testing.assert_close(outputs[arm][k],outputs['baseline'][k],atol=0,rtol=0)
                    finally:branch.forward=original_forward
                for k in ('instance_keypoint_xy','instance_keypoint_coarse_xy'):
                    torch.testing.assert_close(outputs['mask_prior_on_body_only'][k][...,BODY,:],outputs['mask_prior_on'][k][...,BODY,:],atol=0,rtol=0)
                    torch.testing.assert_close(outputs['mask_prior_on_body_only'][k][...,7:14,:],outputs['baseline'][k][...,7:14,:],atol=0,rtol=0)
                return outputs
            def visualize(rgb,outputs,path,targets=None,matches=None):
                panels=[]
                for v,im in enumerate(rgb):
                    cols=[];h,w=im.shape[:2]
                    for arm,pred in outputs.items():
                        canvas=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                        q=int(pred['instance_objectness_logits'][0,v].argmax())
                        if targets is not None:
                            for m in torch.where(targets['instance_valid'][0,v])[0]:
                                for k in BODY:
                                    if targets['keypoint_valid'][0,v,m,k]:
                                        xy=tuple(np.rint(targets['keypoint_xy'][0,v,m,k].numpy()*[w,h]).astype(int))
                                        cv2.circle(canvas,xy,5,(0,255,0),1)
                        for k in BODY:
                            xy=tuple(np.rint(pred['instance_keypoint_xy'][0,v,q,k].numpy()*[w,h]).astype(int))
                            cv2.circle(canvas,xy,3,(0,255,255),-1)
                        cv2.rectangle(canvas,(0,0),(w,35),(20,20,20),-1)
                        cv2.putText(canvas,arm+' top slot',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        cols.append(cv2.resize(canvas,(352,384)))
                    panels.append(np.concatenate(cols,1))
                cv2.imwrite(str(path),np.concatenate(panels,0))
            results={a:[] for a in ['baseline',*states]};seen=set();joints={a:[] for a in results}
            for i,e in enumerate(spec['evaluation']):
                record=spec['records'][e['key']];batch=oriented_batch(record,num_views=len(record['views']))
                target=instance_targets_from_batch(batch);outputs=infer(batch['images'].cuda())
                for arm,pred in outputs.items():
                    metric,matches=evaluate_scene(pred,batch,target)
                    detail=keypoint_rows(pred,batch,target,matches)
                    joints[arm].extend(dict(index=i,group=e['group'],**r) for r in detail)
                    results[arm].append(dict(index=i,group=e['group'],record=record,metrics=metric))
                if e['group'] not in seen:
                    rgb=np.uint8(np.clip(batch['images'][0].permute(0,2,3,1).numpy()*255,0,255))
                    visualize(rgb,outputs,OUT/f'review/synthetic_{i:03d}_{e["group"]}.png',target)
                    seen.add(e['group'])
                status('synthetic_audit',records=i+1,total=88)
                del outputs
            write_json(OUT/'synthetic_results.json',results)
            real_readouts={}
            summary={}
            for arm,rows in results.items():
                summary[arm]={}
                for group in dict.fromkeys(r['group'] for r in rows):
                    metrics=[r['metrics'] for r in rows if r['group']==group]
                    summary[arm][group]={k:float(np.mean([m[k] for m in metrics if m[k] is not None])) if any(m[k] is not None for m in metrics) else None for k in metrics[0]}
            assert tensor_mapping_sha256({k:v for k,v in model.state_dict().items() if not k.startswith('instance_pose_head.spatial_anchor.')})==base_hash
            torch.save(real_readouts,OUT/'real_readouts.pt');write_json(OUT/'summary.json',summary)
            write_json(OUT/'verification.json',dict(passed=True,synthetic_records=88,real_groups=0,body_coordinates_exactly_preserved=True,tail_control_exact_baseline=True,unchanged_base=True,
                depth_boxes_masks_scores_exactly_unchanged=True,training_steps=0,production_promoted=False))
            write_json(OUT/'joint_records.json',joints)
            segments={'body':BODY,'tail':list(range(7,14)),'head':list(range(5)),'torso':[5,6,22],
                'forepaws':[14,15],'hindpaws':[16,17],'proximal_limbs':[18,19,20,21]}
            breakdown={}
            for arm,rows in joints.items():
                breakdown[arm]={}
                for group in dict.fromkeys(r['group'] for r in rows):
                    group_rows=[r for r in rows if r['group']==group]
                    breakdown[arm][group]={name:summarize_joint_rows([r for r in group_rows if r['keypoint'] in ids]) for name,ids in segments.items()}
                    breakdown[arm][group]['per_joint']={str(k):summarize_joint_rows([r for r in group_rows if r['keypoint']==k]) for k in range(23)}
                    for visible in (True,False):
                        breakdown[arm][group]['visible' if visible else 'occluded']=summarize_joint_rows([r for r in group_rows if r['visible'] is visible])
            write_json(OUT/'joint_summary.json',breakdown)
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()
