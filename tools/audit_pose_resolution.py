"""Compare the registered 512 input geometry with the completed 256 baseline."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import ROOT,torch,build_manifest,load_batch,write_json,depth_alignment_audit


def main():
    torch.set_num_threads(2)
    old=json.loads((ROOT/'output/depth_pose_baseline_5000_v2/manifest.json').read_text())
    new=build_manifest(json.loads((ROOT/'configs/experiments/depth_pose_resolution_512_v1.json').read_text()))
    ignored={'purpose','notes','long_side'}
    assert {k:v for k,v in old['config'].items() if k not in ignored}=={k:v for k,v in new['config'].items() if k not in ignored}
    assert new['source_plan_sha256']==old['source_plan_sha256']
    rows=[]
    for low,high in zip(old['records'],new['records'],strict=True):
        assert {k:v for k,v in low.items() if k!='image_hw'}=={k:v for k,v in high.items() if k!='image_hw'}
        a,b=load_batch(low),load_batch(high)
        assert a['view_ids']==b['view_ids']
        assert b['images'].shape[-2:]==tuple(high['image_hw'])
        assert b['depths'].shape[-2:]==tuple(high['image_hw'])
        for key in ('extrinsics','scale','instance_keypoint_xy','instance_keypoint_world_xyz',
                    'instance_keypoint_valid','instance_ids'):
            torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=0)
        scaled=a['intrinsics'].clone(); scaled[...,:2,:]*=2
        torch.testing.assert_close(scaled,b['intrinsics'],atol=1e-5,rtol=0)
        h,w=high['image_hw']; errors=[]
        for v in range(2):
            e=b['extrinsics'][0,v]; k=b['intrinsics'][0,v]
            world=b['instance_keypoint_world_xyz'][0]
            camera=world@e[:3,:3].T+e[:3,3]; p=camera@k.T
            xy=p[...,:2]/p[...,2:]
            gt=b['instance_keypoint_xy'][0,v]*torch.tensor([w,h])
            valid=b['instance_keypoint_valid'][0,v]
            errors.extend((xy-gt).norm(dim=-1)[valid].tolist())
        audit=depth_alignment_audit(b)
        assert max(errors)<.01
        assert audit['surface_anchor_absrel_median']<.15
        rows.append(dict(scene=high['scene'],image_hw=high['image_hw'],actual_view_order=b['view_ids'],
            joint_projection_max_px=max(errors),**audit))
        if len(rows)%16==0: print(f'Audited {len(rows)}/96 geometry pairs',flush=True)
    write_json(ROOT/'output/depth_pose_resolution_512_v1/resolution_audit.json',dict(passed=True,
        compared_scenes=len(rows),only_config_change='long_side256_to512',
        unchanged_world_scale=True,unchanged_normalized_uv=True,intrinsics_scaled_twice=True,records=rows))
    print('RESOLUTION AUDIT PASS',flush=True)


if __name__=='__main__': main()
