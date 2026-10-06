"""Native bundled Blender Python: convert EXR passes and stage BlenderPyMouse files."""
import json,hashlib,shutil,sys
from pathlib import Path
import numpy as np
import OpenEXR,Imath
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/surface_curve_cycle_20261003/render_staging'
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pfm(p,x):
    with p.open('wb') as f:f.write(f'Pf\n{x.shape[1]} {x.shape[0]}\n-1.0\n'.encode());np.flipud(x).astype('<f4').tofile(f)
def exr(p):
    f=OpenEXR.InputFile(str(p));h=f.header();w=h['dataWindow'].max.x-h['dataWindow'].min.x+1;hh=h['dataWindow'].max.y-h['dataWindow'].min.y+1
    def channel(s):
        keys=[k for k in h['channels'] if k.endswith(s)];assert len(keys)==1
        return np.frombuffer(f.channel(keys[0],Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(hh,w).copy()
    z,idx=channel('.Depth.Z'),channel('.Object Index.X');f.close();return z,idx
def project(x,K,E):
    c=x@E[:3,:3].T+E[:3,3];q=c@K.T;return q[...,:2]/q[...,2:],c[...,2]
def visible(x,K,E,depth,mask):
    uv,z=project(x,K,E);shape=z.shape;pts=uv.reshape(-1,2);zz=z.ravel();v=np.zeros(len(pts),bool)
    for i,((u,y),d) in enumerate(zip(pts,zz)):
        if d<=0 or not(0<=u<1080 and 0<=y<1200):continue
        ix,iy=int(u),int(y);y0,y1=max(0,iy-1),min(1200,iy+2);x0,x1=max(0,ix-1),min(1080,ix+2)
        depths=depth[y0:y1,x0:x1];ms=mask[y0:y1,x0:x1];v[i]=np.any(ms & (abs(depths-d)<.75))
    return v.reshape(shape)
def main():
    r=json.loads((BASE/'manifest.json').read_text());passmeta=json.loads((BASE/'geometry_pass_manifest.json').read_text());assert passmeta['complete']
    old=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-BrownFur-ClimbWall-Core23/frame_000001/keypoints.npz'
    with np.load(old) as a:names=a['names'].copy();groups=a['groups'].copy()
    summary=[]
    for pose in r['poses']:
        f=BASE/f"frame_{pose['frame_index']:06d}"
        if (f/'keypoints.npz').exists():
            assert '--replace-staged-after-depth-test' in sys.argv
            assert json.loads((BASE/'independent_geometry_qa_v1_failed.json').read_text())['gates']['depth_passed'] is False
        with np.load(f/'geometry.npz') as a:g=dict(a)
        with np.load(f/'visibility_support.npz') as a:sup=dict(a)
        for d in ['blended_images','rendered_depth_maps','masks','cams','annotations']: (f/d).mkdir(exist_ok=True)
        arrays=[];views=[];checks=[];curves=[]
        for v in pose['views']:
            i=v['view_id'];stem=f'{i:08d}';K=np.array(v['intrinsic']);E=np.array(v['extrinsic']);raw,idx=exr(f/'geometry_passes'/f'{stem}.exr');assert raw.shape==(1200,1080)
            yy,xx=np.mgrid[:1200,:1080];ray=np.stack([(xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones_like(xx)],-1)
            # Blender5.1 output in this scene was independently verified against
            # mesh ray intersections: the Z pass already is camera-axis Z.
            # Dividing by ray norm produced a systematic 1.2-4.6mm bias.
            z=np.where(np.isfinite(raw)&(raw>0)&(raw<1e6),raw,0).astype(np.float32)
            mask=(np.abs(idx-1)<.1)&(z>0);assert mask.sum()>5000,(pose['pose'],v['camera'],mask.sum())
            im=Image.open(BASE/v['rgb']).convert('RGB');im.save(f/'blended_images'/f'{stem}.jpg',quality=98,subsampling=0)
            Image.fromarray(mask.astype(np.uint8)*255).save(f/'masks'/f'{stem}.png');pfm(f/'rendered_depth_maps'/f'{stem}.pfm',z)
            dmin,dmax=float(z[z>0].min()),float(z.max());txt='extrinsic\n'+'\n'.join(' '.join(f'{q:.10g}' for q in row) for row in E)+'\n\nintrinsic\n'+'\n'.join(' '.join(f'{q:.10g}' for q in row) for row in K)+f'\n\n{dmin} {(dmax-dmin)/191} 192 {dmax}\n';(f/'cams'/f'{stem}_cam.txt').write_text(txt)
            uv,jz=project(g['joint_world_xyz'],K,E);inside=(jz>0)&(uv[:,0]>=0)&(uv[:,0]<1080)&(uv[:,1]>=0)&(uv[:,1]<1200)
            support_vis=visible(sup['support_world_xyz'],K,E,z,mask);vv=support_vis.any(-1)
            actual=visible(g['joint_world_xyz'],K,E,z,mask);vv[[5,6,22]]=actual[[5,6,22]];vv &= inside
            cu,cz=project(g['curve_world_xyz'],K,E);cv=visible(g['curve_world_xyz'],K,E,z,mask)
            # Pinhole projected material points may be hidden; never train a visible line through the belly.
            curves.append(dict(uv=cu,visible=cv,camera_z=cz))
            arrays.append(dict(uv=uv,in_frame=inside,visible=vv,K=K,E=E,camera_xyz=g['joint_world_xyz']@E[:3,:3].T+E[:3,3],support_count=support_vis.sum(-1)))
            bounds=np.where(mask);q=np.quantile(z[mask],[.01,.5,.99]);checks.append(dict(camera=v['camera'],mask_pixels=int(mask.sum()),mouse_depth_mm=q.tolist(),scene_depth_mm=[dmin,dmax],curve_visible_samples=int(cv.sum()),back_visible=[bool(vv[5]),bool(vv[6])]))
            # Keep full-scene depth preview separate from image-sized mouse/curve overlay.
            disp=np.zeros_like(z,np.uint8);valid=z>0;lo,hi=np.quantile(z[valid],[.01,.99]);disp[valid]=np.clip(255*(1-(z[valid]-lo)/max(hi-lo,1e-6)),0,255).astype(np.uint8);Image.fromarray(disp).save(f/'rendered_depth_maps'/f'{stem}_preview.png')
            overlay=im.copy();paint=ImageDraw.Draw(overlay)
            for a,b,va,vb in zip(cu[:-1],cu[1:],cv[:-1],cv[1:]):
                if va and vb:paint.line([tuple(a),tuple(b)],fill='#ffad22',width=4)
            edge=mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1));rgb=np.array(overlay);rgb[edge]=[30,255,80];Image.fromarray(rgb).save(f/'annotations'/f'{stem}_mask_curve.png')
            views.append(dict(view_id=i,camera=v['camera'],image=f'blended_images/{stem}.jpg',mask=f'masks/{stem}.png',depth_pfm=f'rendered_depth_maps/{stem}.pfm',cam=f'cams/{stem}_cam.txt',intrinsic=K.tolist(),extrinsic=E.tolist(),depth_available=True))
        pair=['7']
        for i in range(7):pair.extend([str(i),'6 '+' '.join(f'{j} 1' for j in range(7) if j!=i)])
        (f/'cams/pair.txt').write_text('\n'.join(pair)+'\n')
        surface=sup['support_world_xyz'].mean(1);surface[[5,6,22]]=g['joint_world_xyz'][[5,6,22]]
        # Keep established Core23 names for loader compatibility; separate semantic aliases.
        uv=np.stack([a['uv'] for a in arrays]).astype(np.float32);inf=np.stack([a['in_frame'] for a in arrays]);vis=np.stack([a['visible'] for a in arrays])
        np.savez_compressed(f/'keypoints.npz',format=np.asarray('freemoving_core23_surface_v3'),definition_version=np.asarray('pymousestudio_manual_surface_core23_v3'),
            names=names,semantic_names=g['names'],bones=g['bones'],groups=groups,frame=np.asarray(pose['frame_index']),source_frame=np.asarray(pose['source_frame']),
            view_ids=np.arange(7,dtype=np.int32),camera_names=np.array([v['camera'] for v in pose['views']]),
            joint_world_xyz=g['joint_world_xyz'].astype(np.float32),surface_world_xyz=surface.astype(np.float32),
            support_vertex_ids=sup['support_vertex_ids'],support_world_xyz=sup['support_world_xyz'].astype(np.float32),
            uv=uv,in_frame=inf,surface_visible=vis,xy_label_valid=np.isfinite(uv).all(-1),visibility_valid=np.ones_like(inf),
            world_valid=np.ones(23,bool),surface_valid=np.ones(23,bool),intrinsics=np.stack([a['K'] for a in arrays]).astype(np.float32),extrinsics=np.stack([a['E'] for a in arrays]).astype(np.float32),
            joint_camera_xyz=np.stack([a['camera_xyz'] for a in arrays]).astype(np.float32),support_visible_count=np.stack([a['support_count'] for a in arrays]).astype(np.int16),depth_available=np.ones(7,bool))
        np.savez_compressed(f/'surface_curve.npz',schema=np.asarray('dorsal_material_midline_v1'),world_xyz=g['curve_world_xyz'][None].astype(np.float32),
            uv=np.stack([a['uv'] for a in curves])[:,None].astype(np.float32),visible=np.stack([a['visible'] for a in curves])[:,None],
            camera_z=np.stack([a['camera_z'] for a in curves])[:,None].astype(np.float32),triangle_vertex_ids=g['curve_triangle_ids'],barycentric_weights=g['curve_barycentric_weights'],
            instance_ids=np.array([0]),view_ids=np.arange(7),unit=np.asarray('mm'),core23_anchor_ids=np.array([6,7]),
            endpoint_semantics=np.array(['neck_level_dorsal_surface_NOT_bone05','tail_root_level_dorsal_surface_NOT_bone08']))
        save(f/'scene_meta.json',dict(frame=pose['frame_index'],source_frame=pose['source_frame'],pose=pose['pose'],views=views,
             depth_unit='mm',depth_type='camera_z',temporal_reliable=False,keypoint_format='freemoving_core23_surface_v3',keypoint_names=names.tolist(),
             source_blend=pose['blend'],source_blend_sha256=pose['blend_sha256'],geometry_pass_policy=passmeta['policy'],mask_definition=passmeta['mask_definition']))
        summary.append(dict(pose=pose['pose'],frame=pose['frame_index'],checks=checks,keypoints_sha256=sha(f/'keypoints.npz')))
    save(BASE/'conversion_report.json',dict(complete=True,loader_qa_pending=True,geometry_depth_validation_pending=True,poses=summary,
        depth_conversion='Native Blender5.1 Z pass already camera-axis Z verified by independent mesh-ray test; no ray-norm division; numerical mm as Cylinder; invalid0',
        caveats=['Geometry depth excludes hair and RGB refraction; mask is biological skin, not all visible fur.','Curve endpoints are virtual dorsal surface points, not original internal neck/tail-root joints.','Visibility raster tolerance .75mm is provisional until independent geometry QA.']))
    print('CONVERTED',len(summary),'poses; pending QA')
if __name__=='__main__':main()
