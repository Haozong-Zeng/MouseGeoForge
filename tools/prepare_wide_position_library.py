"""CPU preparation: same source scenes/scales/5 variants, wider per-view positions."""
import json
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT,write_json,fingerprint
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.affine_augmentation import centered_affine,transform_xy
from mousegeoforge.per_view_affine_batch import augment_per_view
from mousegeoforge.affine_batch_experiment import augment_batch

OUT=ROOT/'output/wide_position_library_v1'
PARENT=ROOT/'output/spatial_anchor_scale_heatmap_v1'
VARIANTS=[('identity',1.,.5,.5),('left_up',.95,.1,.1),('right_down',.95,.9,.9),('left_down',1.05,.1,.9),('right_up',1.05,.9,.1)]


def safe_affines(batch,scale,fx,fy):
    h,w=batch['images'].shape[-2:];matrices=[];metadata=[]
    for v in range(batch['images'].shape[1]):
        selected=batch['instance_valid'][0,v].numpy();mask=batch['masks'][0,v].numpy()[selected].any(0)
        yy,xx=np.where(mask);assert len(xx)>0
        points=batch['instance_keypoint_xy'][0,v].numpy()*[w,h]
        valid=batch['instance_keypoint_valid'][0,v].numpy()&selected[:,None]
        points=points[valid];assert np.isfinite(points).all()
        boxes=batch['boxes_cxcywh'][0,v].numpy()[selected]*[w,h,w,h]
        corners=np.concatenate([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],0)
        support=np.concatenate([points,corners,np.array([[xx.min(),yy.min()],[xx.max(),yy.max()]])],0)
        a=centered_affine(h,w,scale=scale);transformed=transform_xy(support,a)
        lower=np.array([8.,8.])-transformed.min(0);upper=np.array([w-9.,h-9.])-transformed.max(0)
        if (upper<lower).any():raise ValueError(f'No safe interval view{v} scale{scale}: {lower} {upper}')
        # Exact integer translations; do not change the existing .95/1.05 scale choices.
        lo=np.ceil(lower);hi=np.floor(upper);assert (hi>=lo).all()
        delta=np.rint(lo+np.array([fx,fy])*(hi-lo));a[:2,2]+=delta
        matrices.append(a);metadata.append(dict(dx=float(delta[0]),dy=float(delta[1]),interval=[lo.tolist(),hi.tolist()]))
    return np.array(matrices),metadata


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);OUT.mkdir(exist_ok=False);(OUT/'review').mkdir()
    try:
        parent=json.loads((PARENT/'protocol.json').read_text());rows=[]
        unchanged=('extrinsics','actual_instance_count','instance_valid','instance_ids','instance_keypoint_label_valid','instance_keypoint_occlusion_cause',
                   'instance_keypoint_world_xyz','instance_world_valid','instance_surface_valid','keypoint_joint_world_xyz','keypoint_surface_world_xyz',
                   'keypoint_valid','surface_world_xyz_instances','detection_annotation_complete')
        for index,key in enumerate(parent['train_keys']):
            record=parent['records'][key];batch=oriented_batch(record,num_views=len(record['views']));h,w=batch['images'].shape[-2:]
            before={k:v.clone() for k,v in batch.items() if torch.is_tensor(v)}
            for name,scale,fx,fy in VARIANTS:
                if name=='identity':affines=np.stack([np.eye(3)]*batch['images'].shape[1]);meta=[]
                else:affines,meta=safe_affines(batch,scale,fx,fy)
                out,info=augment_per_view(batch,affines)
                for k in unchanged:torch.testing.assert_close(out[k],batch[k],atol=0,rtol=0)
                assert torch.equal(out['instance_keypoint_valid'],batch['instance_keypoint_valid'])
                if name=='identity':
                    for k in before:torch.testing.assert_close(out[k],batch[k],atol=0,rtol=0)
                    # Uniform affine adapter parity, all raster/coordinate fields.
                    same=np.stack([centered_affine(h,w,scale=.95,dx=-48,dy=-32)]*batch['images'].shape[1])
                    old,_=augment_batch(batch,scale=.95,dx=-48,dy=-32);new,_=augment_per_view(batch,same)
                    for k in before:torch.testing.assert_close(old[k],new[k],atol=0,rtol=0)
                for v,a in enumerate(affines):
                    xyz=batch['instance_keypoint_world_xyz'][0].double().numpy();e=batch['extrinsics'][0,v].double().numpy()
                    camera=xyz@e[:3,:3].T+e[:3,3];p=camera@batch['intrinsics'][0,v].double().numpy().T;p2=camera@out['intrinsics'][0,v].double().numpy().T
                    uv=p[...,:2]/p[...,2:];uv2=p2[...,:2]/p2[...,2:];finite=np.isfinite(uv).all(-1)&np.isfinite(uv2).all(-1)&(np.abs(camera[...,2])>1e-4)
                    np.testing.assert_allclose(uv2[finite],transform_xy(uv,a)[finite],atol=1e-3,rtol=1e-5)
                rows.append(dict(key=key,variant=dict(name=name,scale=scale),affines=affines.tolist(),placement=meta,accepted=True,minimum_retained=info['minimum_retained'],expected_instances=record['expected_instances']))
                if index in (0,16) and name!='identity':
                    panels=[]
                    for v in range(out['images'].shape[1]):
                        rgb=np.uint8(np.clip(out['images'][0,v].permute(1,2,0).numpy()*255,0,255));canvas=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                        for m in torch.where(out['instance_valid'][0,v])[0]:
                            for k in torch.where(out['instance_keypoint_valid'][0,v,m])[0]:
                                pt=tuple(np.rint(out['instance_keypoint_xy'][0,v,m,k].numpy()*[w,h]).astype(int));cv2.circle(canvas,pt,3,(0,255,0),1)
                        panels.append(cv2.resize(canvas,(384,round(h*384/w))))
                    cv2.imwrite(str(OUT/f'review/scene{index}_{name}.png'),np.concatenate(panels,1))
            for k,v in before.items():torch.testing.assert_close(batch[k],v,atol=0,rtol=0)
            write_json(OUT/'status.json',dict(state='preparing',scenes=index+1,total=32))
        assert len(rows)==160
        write_json(OUT/'plan.json',dict(records=rows,variants=VARIANTS,parent_protocol_sha256=fingerprint(PARENT/'protocol.json'),
            warning='Per-view image-plane augmentation; physical E/world/depth units unchanged, K individually transformed. No temporal/camera-motion claim.',
            code_sha256={p:fingerprint(ROOT/p) for p in ['tools/prepare_wide_position_library.py','mousegeoforge/per_view_affine_batch.py','mousegeoforge/affine_augmentation.py']}))
        write_json(OUT/'verification.json',dict(passed=True,scenes=32,variants=160,all_original_valid_points_retained=True,uniform_adapter_parity_exact=True,
            world_ids_extrinsics_unchanged=True,projection_commutes=True,source_batch_unchanged=True,minimum_mask_retained=min(r['minimum_retained'] for r in rows)))
        write_json(OUT/'status.json',dict(state='completed',scenes=32,variants=160));print('COMPLETED',flush=True)
    except BaseException as e:write_json(OUT/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':main()
