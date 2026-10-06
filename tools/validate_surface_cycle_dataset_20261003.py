"""Independent mesh ray-depth check and actual multi-instance dataloader smoke."""
import json,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/surface_curve_cycle_20261003/render_staging'
sys.path.insert(0,str(ROOT/'vggt-omega'))
from training.finetune_blendedmvs import read_pfm,load_scene,BlendedMVSDataset
from training.instance_pose_losses import instance_targets_from_batch
import torch
from torch.utils.data import DataLoader
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rays_mesh_z(rays,verts,faces):
    tri=verts[faces];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;tvec=-a;qvec=np.cross(tvec,e1);num=np.einsum('ij,ij->i',e2,qvec);out=[]
    for d in rays:
        p=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,p);safe=np.where(abs(det)>1e-10,det,1)
        u=np.einsum('ij,ij->i',tvec,p)/safe;v=qvec@d/safe;t=num/safe
        valid=(abs(det)>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
        out.append(float(t[valid].min()) if valid.any() else np.nan)
    return np.array(out)
def main():
    manifests=json.loads((BASE/'manifest.json').read_text());rows=[];rng=np.random.default_rng(1703)
    for pose in manifests['poses']:
        f=BASE/f"frame_{pose['frame_index']:06d}"
        with np.load(f/'geometry.npz') as a:verts=a['vertices'];faces=a['triangles']
        with np.load(f/'keypoints.npz') as a:kp=dict(a)
        with np.load(f/'surface_curve.npz') as a:curve=dict(a)
        assert curve['world_xyz'].shape==(1,65,3) and curve['uv'].shape==(7,1,65,2)
        assert not curve['visible'][0].any(),'Bottom view cannot directly see the dorsal surface'
        assert kp['uv'].shape==(7,23,2) and kp['visibility_valid'].shape==(7,23)
        for v in pose['views']:
            i=v['view_id'];K=np.array(v['intrinsic']);E=np.array(v['extrinsic']);mask=np.asarray(Image.open(f/'masks'/f'{i:08d}.png'))>0
            z=read_pfm(f/'rendered_depth_maps'/f'{i:08d}.pfm',flip_y=True);assert np.isfinite(z).all() and(z>=0).all()
            eroded=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0;yx=np.argwhere(eroded);sel=yx[rng.choice(len(yx),min(128,len(yx)),replace=False)]
            rays=np.c_[(sel[:,1]+.5-K[0,2])/K[0,0],(sel[:,0]+.5-K[1,2])/K[1,1],np.ones(len(sel))]
            cz=rays_mesh_z(rays,verts@E[:3,:3].T+E[:3,3],faces);observed=z[sel[:,0],sel[:,1]];valid=np.isfinite(cz)
            err=abs(observed[valid]-cz[valid]);raw_counter=abs(observed[valid]/np.linalg.norm(rays[valid],axis=-1)-cz[valid]);flip_err=abs(z[1199-sel[valid,0],sel[valid,1]]-cz[valid])
            row=dict(pose=pose['pose'],camera=v['camera'],rays=len(sel),hit_fraction=float(valid.mean()),depth_median_abs_mm=float(np.median(err)),depth_p95_abs_mm=float(np.quantile(err,.95)),incorrect_ray_norm_division_median_error_mm=float(np.median(raw_counter)),wrong_vertical_flip_median_error_mm=float(np.median(flip_err)))
            rows.append(row);print(row,flush=True)
    gates=dict(depth_passed=all(r['hit_fraction']>.95 and r['depth_median_abs_mm']<.5 and r['depth_p95_abs_mm']<2 for r in rows))
    save(BASE/'independent_geometry_qa.json',dict(gates=gates,views=rows,method='128 interior skin-mask pixels/view; independent vectorized Moller-Trumbore nearest mesh intersection with pixel-center pinhole rays, mm'))
    assert gates['depth_passed'],'Do not publish incorrect camera depth'
    scenes=[load_scene(BASE/f"frame_{p['frame_index']:06d}") for p in manifests['poses']];smokes=[]
    for nv in [1,2,7]:
        ds=BlendedMVSDataset(scenes=scenes,num_views=nv,image_width=691,image_height=768,samples_per_scene=1,seed=17,use_masked_images=False,pfm_flip_y=True,canonicalize_camera=False,scale_mode='raw',top_srcs=6,load_keypoints=True)
        batch=next(iter(DataLoader(ds,batch_size=3,shuffle=False)));target=instance_targets_from_batch(batch)
        assert ds.max_instances==1 and batch['actual_instance_count'].tolist()==[1]*3
        assert target['keypoint_xy'].shape==(3,nv,1,23,2)
        for k in ['images','depths','intrinsics','extrinsics','instance_keypoint_xy']:assert torch.isfinite(batch[k]).all(),k
        assert batch['masks'].any() and batch['instance_valid'].all()
        smokes.append(dict(views=nv,batch_size=3,input_wh=[691,768],keypoint_shape=list(target['keypoint_xy'].shape),all_finite=True))
    save(BASE/'loader_qa.json',dict(passed=True,tests=smokes,depth_units='mm',pfm_flip_y=True))
    print('INDEPENDENT_GEOMETRY_AND_LOADER_PASSED',flush=True)
if __name__=='__main__':main()
