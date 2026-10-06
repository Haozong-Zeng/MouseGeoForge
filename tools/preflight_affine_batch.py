"""Validate real loader fields and fixed augmentation acceptance before GPU work."""
import json
import numpy as np
import torch
from run_depth_pose_ablation import ROOT,write_json,fingerprint
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.affine_batch_experiment import augment_batch,UnsafeCrop

OUT=ROOT/'output/spatial_anchor_affine_batch_preflight_v1_retry1'
VARIANTS=[dict(name='identity',scale=1.,dx=0.,dy=0.),dict(name='left_up',scale=.95,dx=-48.,dy=-32.),
          dict(name='right_down',scale=.95,dx=48.,dy=32.),dict(name='left_down',scale=1.05,dx=-48.,dy=32.),
          dict(name='right_up',scale=1.05,dx=48.,dy=-32.)]


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'output/spatial_anchor_diverse32_v1/protocol.json').read_text());rows=[]
    unchanged=('extrinsics','actual_instance_count','instance_valid','instance_ids','instance_keypoint_label_valid',
               'instance_keypoint_occlusion_cause','instance_keypoint_world_xyz','instance_world_valid',
               'instance_surface_valid','keypoint_joint_world_xyz','keypoint_surface_world_xyz','keypoint_valid',
               'surface_world_xyz_instances','detection_annotation_complete')
    for i,key in enumerate(plan['train_keys']):
        record=plan['records'][key];batch=oriented_batch(record,num_views=len(record['views']))
        original={k:v.clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        for variant in VARIANTS:
            row=dict(key=key,variant=variant,expected_instances=record['expected_instances'])
            try:out,info=augment_batch(batch,**{k:v for k,v in variant.items() if k!='name'})
            except UnsafeCrop as e:rows.append(dict(row,accepted=False,reason=str(e)));continue
            for k in unchanged:torch.testing.assert_close(out[k],batch[k],atol=0,rtol=0)
            assert out['instance_keypoint_valid'][out['instance_valid']].any(-1).all()
            h,w=out['images'].shape[-2:];a=np.array(info['affine'])
            for v in range(out['images'].shape[1]):
                xyz=batch['instance_keypoint_world_xyz'][0].double().numpy()
                e=batch['extrinsics'][0,v].double().numpy();camera=xyz@e[:3,:3].T+e[:3,3]
                p=camera@batch['intrinsics'][0,v].double().numpy().T;projected=p[...,:2]/p[...,2:]
                p2=camera@out['intrinsics'][0,v].double().numpy().T;projected2=p2[...,:2]/p2[...,2:]
                finite=np.isfinite(projected).all(-1)&np.isfinite(projected2).all(-1)&(np.abs(camera[...,2])>1e-4)
                expected=projected@a[:2,:2].T+a[:2,2]
                np.testing.assert_allclose(projected2[finite],expected[finite],atol=1e-3,rtol=1e-5)
            if variant['name']=='identity':
                for k in original:torch.testing.assert_close(out[k],batch[k],atol=0,rtol=0)
            rows.append(dict(row,accepted=True,**info))
        for k in original:torch.testing.assert_close(batch[k],original[k],atol=0,rtol=0)
        print('PREFLIGHT',i+1,flush=True)
    write_json(OUT/'plan.json',dict(variants=VARIANTS,records=rows,parent_protocol_sha256=fingerprint(ROOT/'output/spatial_anchor_diverse32_v1/protocol.json'),
        code_sha256={p:fingerprint(ROOT/p) for p in ['mousegeoforge/affine_augmentation.py','mousegeoforge/affine_batch_experiment.py','tools/preflight_affine_batch.py']}))
    write_json(OUT/'verification.json',dict(passed=True,scenes=32,accepted=sum(r['accepted'] for r in rows),rejected=sum(not r['accepted'] for r in rows),
        identity_exact=True,source_batch_unchanged=True,world_labels_and_extrinsics_unchanged=True,projection_commutes=True))


if __name__=='__main__':main()
