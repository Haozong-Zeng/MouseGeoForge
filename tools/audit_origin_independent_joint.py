"""Same rasters, independent V1 vs joint V1/V2/V3, frozen readout diagnostic."""
import fcntl,json,os,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,VGGTOmega,DepthPoseModel,write_json,fingerprint,instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import install_adaptive_temperature,restore_adaptive
from mousegeoforge.spatial_anchor_experiment import install_spatial_anchor
from mousegeoforge.roi_scale_diagnostic import mask_query_iou,BODY
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256
from mousegeoforge.frozen_coordinate_readouts import coordinate_readouts
from mousegeoforge.origin_coordinate_residual import OriginCoordinateResidual

OUT=ROOT/'output/origin_independent_joint_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
OLD=ROOT/'output/independent_joint_readouts_v1'
HEADS=ROOT/'output/origin_coordinate_residual_v1'
ARMS=['frozen','base_origin','anchor_origin']
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
            parent=json.loads((PARENT/'protocol.json').read_text());assert json.loads((PARENT/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/p)==h for p,h in parent['code_sha256'].items())
            crop_plan=json.loads((ROOT/'output/oracle_roi_scale_diagnostic_v1/protocol.json').read_text())
            full=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text())
            selected=crop_plan['selected']+[e for e in full['evaluation'] if e['group']=='sides_bottom'];assert len(selected)==32
            reference={(r['index'],r['view'],r['instance'],r['mode']):r for r in json.loads((OLD/'synthetic_records.json').read_text()) if r['arm']=='trained_final'}
            stress_path=ROOT/'output/origin_coordinate_stress_v1/synthetic_records.json'
            stress={(r['index'],r['view'],r['instance'],r['arm']):r for r in json.loads(stress_path.read_text()) if r['geometry']=='original'}
            real_reference={(r['session'],r['frame'],r['camera'],r['independent_query']):r for r in json.loads((OLD/'real_records.json').read_text())}
            hp=json.loads((HEADS/'protocol.json').read_text());assert json.loads((HEADS/'verification.json').read_text())['passed']
            assert all(fingerprint(ROOT/p)==h for p,h in {**hp['hashes'],**hp['code_sha256']}.items())
            assert fingerprint(HEADS/'selection.json')==hp['selection_sha256']
            real=json.loads((REAL/'protocol.json').read_text());assert not set(real['sources'])&{'GR2','UR2','UR3'}
            base=ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt';encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';candidate=PARENT/'scale_sigma_step800.pt'
            assert fingerprint(base)==parent['base_sha256'] and fingerprint(encoder)==parent['encoder_sha256']
            guards={str(p.relative_to(ROOT)):fingerprint(p) for p in (base,encoder,candidate,OLD/'synthetic_records.json',OLD/'real_records.json',stress_path,HEADS/'base_origin_step400.pt',HEADS/'anchor_origin_step400.pt')}
            code=['tools/audit_origin_independent_joint.py','mousegeoforge/frozen_coordinate_readouts.py','mousegeoforge/origin_coordinate_residual.py','mousegeoforge/roi_scale_diagnostic.py']
            codehash={p:fingerprint(ROOT/p) for p in code}
            write_json(OUT/'protocol.json',dict(selected=selected,records={e['key']:full['records'][e['key']] for e in selected},guards=guards,code_sha256=codehash,readouts=ARMS,
                modes=['independent','joint'],training_steps=0,synthetic_records=32,real_sources=real['sources'],real_joint_order=['D','E','A'],heldout_accessed=False,
                pairing='Same pixel tensor slice, no crop or resize change; GT-mask max-IoU per target in each mode, never GT point query selection; scores/mask changes recorded.',
                metric='All valid body points, same GT box diagonal; GT-assisted diagnostic PCK, not formal detection PCK. Three frozen trained readouts; different origin baselines. No optimization.',
                real_matching='Every independent positive query, matched by maximum predicted-mask IoU in same physical joint view; duplicates retained, no GT.',
                replay='F both modes exactly replay stage24; independent24 all three body errors exactly replay stage32; one-view modes exact; real F coordinates replay stage24; PNG once rectified only.'))
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,map_location='cpu',weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model=install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model,torch.load(base,map_location='cpu',weights_only=True)['model']);del spatial,weights
            install_spatial_anchor(model);model.instance_pose_head.spatial_anchor.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True)['branch'])
            model.eval().requires_grad_(False);modelhash=tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})
            heads={}
            for arm in ('base_origin','anchor_origin'):
                checkpoint=torch.load(HEADS/f'{arm}_step400.pt',map_location='cpu',weights_only=True)
                assert checkpoint['anchor_origin']==(arm=='anchor_origin')
                decoder=OriginCoordinateResidual(anchor_origin=arm=='anchor_origin').cuda();decoder.load_state_dict(checkpoint['state']);decoder.eval().requires_grad_(False);heads[arm]=decoder
            headhash={a:tensor_mapping_sha256(h.state_dict()) for a,h in heads.items()}
            count=0;replay=[];rows=[];realrows=[];imagehashes={};seen=set()
            def infer(images):
                nonlocal count
                if time.monotonic()-began>1200 or time.time()>1789233567:raise TimeoutError('Independent/joint cutoff')
                assert images.shape[0]==1
                h,w=images.shape[-2:];outputs,pred,error=coordinate_readouts(model,heads,images.cuda());coordinates={a:xy for a,(xy,coarse) in outputs.items()};replay.append(error);count+=1
                logits=pred['instance_mask_logits'][0];v,q=logits.shape[:2]
                masks=(F.interpolate(logits.reshape(v*q,1,*logits.shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)>0).numpy().reshape(v,q,h,w)
                return {a:xy.numpy()*[w,h] for a,xy in coordinates.items()},masks,pred['instance_objectness_logits'][0].sigmoid().numpy()
            def visualize(rgb,outputs,queries,path,gt=None,ids=BODY):
                # Two rows, 3 fixed readouts, not per-image best selection.
                panels=[]
                for mode,(xy,v,q) in outputs.items():
                    cols=[]
                    for arm in ARMS:
                        canvas=cv2.cvtColor(np.uint8(np.clip(rgb*255,0,255)),cv2.COLOR_RGB2BGR)
                        if gt is not None:
                            for k in ids:cv2.circle(canvas,tuple(np.rint(gt[k]).astype(int)),5,(0,255,0),1)
                        for k in ids:cv2.circle(canvas,tuple(np.rint(xy[arm][v,q,k]).astype(int)),3,(0,255,255),-1)
                        cv2.putText(canvas,mode+' '+arm,(5,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        h,w=canvas.shape[:2];cols.append(cv2.resize(canvas,(384,round(h*384/w))))
                    panels.append(np.concatenate(cols,1))
                cv2.imwrite(str(path),np.concatenate(panels,0))
            for index,e in enumerate(selected):
                rec=full['records'][e['key']];batch=oriented_batch(rec,num_views=len(rec['views']));target=instance_targets_from_batch(batch)
                joint=infer(batch['images']);h,w=batch['images'].shape[-2:]
                for v in range(batch['images'].shape[1]):
                    single=infer(batch['images'][:,v:v+1].contiguous())
                    if len(rec['views'])==1:
                        for a in ARMS:np.testing.assert_array_equal(single[0][a],joint[0][a])
                        np.testing.assert_array_equal(single[1],joint[1]);np.testing.assert_array_equal(single[2],joint[2])
                    mask=target['masks'][0,v].numpy().astype(bool)
                    for m in np.flatnonzero(target['instance_valid'][0,v].numpy()):
                        gt=target['keypoint_xy'][0,v,m].numpy()*[w,h];valid=target['keypoint_valid'][0,v,m].numpy()
                        ids=[k for k in BODY if valid[k]];diag=max(1,float(np.linalg.norm(target['boxes_cxcywh'][0,v,m,2:].numpy()*[w,h])))
                        visual={}
                        for mode,pred,i in [('independent',single,0),('joint',joint,v)]:
                            q,iou=mask_query_iou(pred[1][i],mask[m]);visual[mode]=(pred[0],i,q)
                            for arm in ARMS:
                                xy=pred[0][arm][i,q];errors=np.linalg.norm(xy[ids]-gt[ids],axis=-1)/diag
                                if arm=='frozen':
                                    old=reference[(index,rec['views'][v],int(m),mode)];assert q==old['query']
                                    np.testing.assert_array_equal(xy[ids],old['xy'])
                                if mode=='independent' and index<24:
                                    oldstress=stress[(index,rec['views'][v],int(m),arm)]
                                    np.testing.assert_array_equal(errors,oldstress['errors'])
                                rows.append(dict(index=index,group=e['group'],physical_role=rec['physical_roles'][v],view=rec['views'][v],instance=int(m),mode=mode,arm=arm,query=q,
                                    matched_mask_iou=float(iou[q]),objectness=float(pred[2][i,q]),positive_queries=int((pred[2][i]>=.5).sum()),
                                    body_indices=ids,xy=xy[ids].tolist(),errors=errors.tolist(),body_nme=float(errors.mean()),body_pck=float((errors<=.1).mean())))
                        if e['group'] not in seen:visualize(batch['images'][0,v].permute(1,2,0).numpy(),visual,None,OUT/f'review/synthetic_{e["group"]}_v{v}_m{m}.png',gt,ids)
                seen.add(e['group']);status('synthetic',records=index+1,total=32,encoder_calls=count)
            for session,frames in real['sources'].items():
                for frame in frames:
                    images=[]
                    for camera in ('D','E','A'):
                        path=REAL/f'review/{session}_{frame:06d}_{camera}_input.png';imagehashes[str(path.relative_to(ROOT))]=fingerprint(path)
                        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB).astype(np.float32)/255;assert rgb.shape==(768,704,3);images.append(rgb)
                    x=torch.from_numpy(np.stack(images)).permute(0,3,1,2)[None];joint=infer(x)
                    for v,camera in enumerate(('D','E','A')):
                        single=infer(x[:,v:v+1].contiguous())
                        for q in np.flatnonzero(single[2][0]>=.5):
                            qn,iou=mask_query_iou(joint[1][v],single[1][0,q])
                            oldreal=real_reference[(session,frame,camera,int(q))];assert qn==oldreal['joint_query']
                            np.testing.assert_array_equal(single[0]['frozen'][0,q],oldreal['independent_xy']['trained_final'])
                            np.testing.assert_array_equal(joint[0]['frozen'][v,qn],oldreal['joint_xy']['trained_final'])
                            realrows.append(dict(session=session,frame=frame,camera=camera,independent_query=int(q),joint_query=qn,predicted_mask_iou=float(iou[qn]),
                                independent_positive_queries=int((single[2][0]>=.5).sum()),joint_positive_queries=int((joint[2][v]>=.5).sum()),
                                body_shift_px={a:float(np.linalg.norm(single[0][a][0,q,BODY]-joint[0][a][v,qn,BODY],axis=-1).mean()) for a in ARMS},
                                independent_xy={a:single[0][a][0,q].tolist() for a in ARMS},joint_xy={a:joint[0][a][v,qn].tolist() for a in ARMS}))
                            visualize(images[v],{'independent':(single[0],0,int(q)),'joint':(joint[0],v,qn)},None,OUT/f'review/real_{session}_{frame:06d}_{camera}_q{q}.png')
                    status('real',source=f'{session}_{frame}',encoder_calls=count)
            summary={}
            for group,role in sorted({(r['group'],r['physical_role']) for r in rows}):
                summary[group+'/'+role]={}
                for mode in ('independent','joint'):
                    summary[group+'/'+role][mode]={}
                    for arm in ARMS:
                        subset=[r for r in rows if (r['group'],r['physical_role'],r['mode'],r['arm'])==(group,role,mode,arm)]
                        err=np.concatenate([r['errors'] for r in subset]);summary[group+'/'+role][mode][arm]=dict(body_nme=float(err.mean()),body_pck=float((err<=.1).mean()),n=len(subset),
                            mask_iou=float(np.mean([r['matched_mask_iou'] for r in subset])),score=float(np.mean([r['objectness'] for r in subset])))
            assert tensor_mapping_sha256({k:v.reshape(-1) for k,v in model.state_dict().items()})==modelhash
            assert all(p.grad is None for p in model.parameters())
            assert all(tensor_mapping_sha256(h.state_dict())==headhash[a] for a,h in heads.items())
            assert all(fingerprint(ROOT/p)==h for p,h in {**guards,**codehash,**imagehashes}.items())
            write_json(OUT/'summary.json',summary);write_json(OUT/'synthetic_records.json',rows);write_json(OUT/'real_records.json',realrows)
            write_json(OUT/'verification.json',dict(passed=True,training_steps=0,weights_unchanged=True,source_images_sha256=imagehashes,heldout_accessed=False,encoder_calls=count,
                full_head_replay_max=max(replay),prior_24_single_view_body_errors_exact=True,prior_32_frozen_both_modes_coordinates_exact=True,prior_real_frozen_coordinates_exact=True,single_view_joint_exact=True,production_promoted=False))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()


