"""Immutable 768-letterbox real pilot derivative, preserving partial-label flags."""
import json,hashlib,sys,datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import torch
from torch.utils.data import default_collate
from mousegeoforge.surface_curve_supervision import back_tangents
from training.finetune_blendedmvs import BlendedMVSDataset,load_scene
from training.instance_pose_losses import instance_targets_from_batch
BASE=ROOT/'output/surface_curve_cycle_20261003/training_v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def letterbox(p,mask=False):
    with Image.open(p) as im:
        assert im.size==(1080,1200),(p,im.size)
        im=im.convert('L' if mask else 'RGB').resize((691,768),Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
    canvas=Image.new(im.mode,(704,768),0);canvas.paste(im,(6,0));return canvas
def main():
    torch.set_num_threads(2)
    cfg=json.loads((BASE/'protocol.json').read_text());out=BASE/'real_derived'
    assert not out.exists(),'Immutable derivation already exists';out.mkdir()
    affine=np.array(cfg['real_affine']);records=[];checks=[]
    allowed=np.zeros(23,bool);allowed[np.array(cfg['real_supervised_ids_1based'])-1]=True
    for record in cfg['real']:
        assert record['session'] not in ('GR2','UR2','UR3')
        src=ROOT/record['normalized'];assert sha(src)==record['normalized_sha256']
        with np.load(src) as a:original=dict(a)
        assert not bool(original['excluded_group'])
        observed=original['xy_observed'].astype(bool)&np.isfinite(original['uv']).all(-1)
        if 'loss_eligible' in original:observed &= original['loss_eligible'].astype(bool)
        uv=original['uv'].astype(np.float32).copy();uv=uv@affine[:2,:2].T+affine[:2,2]
        inside=observed&(uv[...,0]>=0)&(uv[...,0]<704)&(uv[...,1]>=0)&(uv[...,1]<768)
        tangent,tangent_valid=back_tangents(torch.tensor(uv,dtype=torch.float32),torch.tensor(inside))
        cameras=[str(c) for c in original['camera_names']];view_ids=np.array([ord(c[-1])-ord('A') for c in cameras])
        scene=out/f"{record['session']}_frame_{record['frame']:06d}"
        for sub in ('blended_images','masks','cams'):(scene/sub).mkdir(parents=True)
        cal=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration'/record['session']/'rectified_calibration.json'
        calibration=json.loads(cal.read_text());viewmeta=[]
        for row in record['views']:
            camera=row['camera'];vid=ord(camera[-1])-ord('A');assert vid in view_ids
            assert sha(ROOT/row['annotation'])==row['annotation_sha256']
            image=ROOT/row['source_image'];letterbox(image).save(scene/'blended_images'/f'{vid:08d}.png')
            if row['mask_valid']:
                assert sha(ROOT/row['mask'])==row['mask_sha256']
                letterbox(ROOT/row['mask'],True).save(scene/'masks'/f'{vid:08d}.png')
            c=calibration[camera];assert c['size_wh']==[1080,1200] and not np.any(c['distortion'])
            K=affine@np.asarray(c['K']);E=np.asarray(c['world_to_camera_mm'])
            lines=['extrinsic']+[' '.join(map(str,r)) for r in E]+['','intrinsic']+[' '.join(map(str,r)) for r in K]+['','1 1 192 193']
            (scene/'cams'/f'{vid:08d}_cam.txt').write_text('\n'.join(lines)+'\n')
            viewmeta.append(dict(camera=camera,view_id=vid,source_image=row['source_image'],source_sha256=sha(image),mask_valid=row['mask_valid'],mask_source_sha256=row['mask_sha256']))
        assert len(viewmeta)==len(cameras)
        n=len(cameras);unknown3=np.full((1,23,3),np.nan,np.float32)
        np.savez_compressed(scene/'keypoints.npz',names=original['names'],semantic_names=original['semantic_names'],
            view_ids=view_ids,instance_ids=np.array([0]),uv=uv.astype(np.float32),in_frame=inside,
            joint_world_xyz=unknown3,surface_world_xyz=unknown3.copy(),surface_visible=np.zeros((n,1,23),bool),
            xy_label_valid=inside&allowed[None,None,:],visibility_valid=np.zeros((n,1,23),bool),
            world_valid=np.zeros((1,23),bool),surface_valid=np.zeros((1,23),bool))
        np.savez_compressed(scene/'surface_aux.npz',view_ids=view_ids,uv=uv,observed=inside,
            tangent=tangent.numpy(),tangent_valid=tangent_valid.numpy())
        meta=dict(session=record['session'],source_frame=record['frame'],role=record['role'],real=True,
            source_normalized=record['normalized'],source_normalized_sha256=sha(src),calibration_sha256=sha(cal),
            image_affine=affine.tolist(),image_wh=[704,768],coordinate_units='mm',detection_annotation_complete=True,
            visibility='unknown; coordinate absence is not occlusion',depth='unavailable; do not supervise',
            curve='unavailable; two ambiguous manual back points are not a full curve',views=viewmeta)
        save(scene/'scene_meta.json',meta)
        ds=BlendedMVSDataset([load_scene(scene)],num_views=n,image_width=704,image_height=768,samples_per_scene=1,seed=17,
            use_masked_images=False,pfm_flip_y=True,canonicalize_camera=False,scale_mode='median_baseline',top_srcs=6,load_keypoints=True)
        ds.max_instances=2;sample=ds[0];batch=default_collate([sample]);target=instance_targets_from_batch(batch)
        assert sample['camera_available'].all() and not sample['depth_available'].any()
        assert not sample['instance_keypoint_visibility_valid'].any() and not sample['instance_world_valid'].any()
        assert sample['actual_instance_count']==1 and not sample['instance_valid'][:,1].any()
        assert not sample['box_valid'][~sample['mask_valid']].any(),'Sparse points must not produce valid boxes'
        assert not target['keypoint_valid'][...,~torch.tensor(allowed)].any()
        order=[int(x) for x in sample['view_ids'].split(',')]
        for i,vid in enumerate(order):
            oi=view_ids.tolist().index(vid);expected=np.nan_to_num(uv[oi,0]/[704,768])
            torch.testing.assert_close(sample['instance_keypoint_xy'][i,0],torch.tensor(expected,dtype=torch.float32),atol=1e-6,rtol=1e-6)
        for k in ('images','intrinsics','extrinsics','instance_keypoint_xy'):assert torch.isfinite(sample[k]).all(),k
        checks.append(dict(scene=scene.relative_to(ROOT).as_posix(),views=n,observed_back=int(inside[...,5:7].sum()),known_back_tangents=int(tangent_valid[...,5:7].sum()),valid_masks=int(sample['mask_valid'].sum()),valid_points=int(target['keypoint_valid'].sum()),passed=True))
        records.append(dict(scene=scene.relative_to(ROOT).as_posix(),role=record['role'],session=record['session'],frame=record['frame'],views=view_ids.tolist(),normalization_scale=float(sample['scale'])))
        print(record['session'],record['frame'],'passed',flush=True)
    report=dict(passed=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(BASE/'protocol.json'),records=records,checks=checks,
        totals=dict(groups=len(records),images=sum(x['views'] for x in checks),observed_back=sum(x['observed_back'] for x in checks),known_back_tangents=sum(x['known_back_tangents'] for x in checks)))
    save(BASE/'real_derived_report.json',report);print(json.dumps(report['totals']),flush=True)
if __name__=='__main__':main()
