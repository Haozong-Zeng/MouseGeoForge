"""Validate staged/live labels against the pre-migration source and real loader."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vggt-omega'))
BASE=ROOT/'output/surface_keypoint_migration_v3'
LIVE=ROOT/'dataset/BlenderPyMouse'
IDS=[5,6,22];KEEP=[k for k in range(23) if k not in IDS]


def main(live=False):
    manifest=json.loads((BASE/'manifest.json').read_text())
    root=LIVE if live else BASE/'candidate'
    backup=zipfile.ZipFile(manifest['backup']) if live else None
    def old_bytes(rel):return backup.read(str(rel)) if backup else (LIVE/rel).read_bytes()
    frames=0;max_projection_error=0.;min_separation=float('inf');json_points=0
    for ds in manifest['datasets']:
        for f in sorted((root/ds['dataset']).glob('frame_*/keypoints.npz')):
            rel=f.relative_to(root)
            with np.load(f) as z:a=dict(z)
            with np.load(io.BytesIO(old_bytes(rel))) as z:old=dict(z)
            dual=a['joint_world_xyz'].ndim==3
            for k in ('intrinsics','extrinsics','names','bones','frame','view_ids','instance_ids'):
                if k in old:assert np.array_equal(a[k],old[k]),(f,k)
            for k in ('joint_world_xyz','surface_world_xyz','support_world_xyz','support_vertex_ids'):
                assert np.array_equal(a[k][:,KEEP] if dual else a[k][KEEP],old[k][:,KEEP] if dual else old[k][KEEP]),(f,k)
            for k in ('uv','joint_camera_xyz','in_frame','surface_visible','support_visible_count'):
                assert np.array_equal(a[k][:,:,KEEP] if dual else a[k][:,KEEP],old[k][:,:,KEEP] if dual else old[k][:,KEEP]),(f,k)
            world=a['joint_world_xyz'][:,IDS] if dual else a['joint_world_xyz'][None,IDS]
            min_separation=min(min_separation,float(np.linalg.norm(world[:,0]-world[:,2],axis=-1).min()))
            E=a['extrinsics'].astype(float);K=a['intrinsics'].astype(float)
            cam=np.einsum('vij,mkj->vmki',E[:,:3,:3],world)+E[:,None,None,:3,3]
            p=np.einsum('vij,vmkj->vmki',K,cam);uv=p[...,:2]/p[...,2:]
            expected=a['uv'][:,:,IDS] if dual else a['uv'][:,None,IDS]
            err=float(np.abs(uv-expected).max());max_projection_error=max(max_projection_error,err)
            assert err<.005,(f,err)
            def check_json(before,after,view=None,m=0):
                nonlocal json_points
                if 'view_id' in before:view=list(a['view_ids']).index(int(before['view_id']))
                if dual and 'instance_id' in before:m=list(a['instance_ids']).index(before['instance_id'])
                if 'id' in before and 'uv' in before:
                    k=int(before['id'])
                    if k not in IDS:assert before==after,(f,view,m,k,'untouched JSON point changed')
                    else:
                        target=a['uv'][view,m,k] if dual else a['uv'][view,k]
                        assert np.array_equal(np.asarray(after['uv'],np.float32),target)
                        assert after['definition_version']=='pymousestudio_manual_surface_core23_v3'
                        json_points+=1
                for key in ('joint_world_xyz','surface_world_xyz','support_vertex_ids'):
                    if key in before and len(before[key])==23:
                        for k in KEEP:assert before[key][k]==after[key][k]
                for key in ('views','instances','keypoints'):
                    for b,c in zip(before.get(key,[]),after.get(key,[])):check_json(b,c,view,m)
            for p in [f.with_suffix('.json'),f.with_name('scene_meta.json'),*sorted((f.parent/'annotations').glob('*.json'))]:
                check_json(json.loads(old_bytes(p.relative_to(root))),json.loads(p.read_text()))
            frames+=1
        print('GEOMETRY_JSON_OK',ds['dataset'],flush=True)
    from training.finetune_blendedmvs import BlendedMVSDataset,load_scene
    import torch
    results=[]
    for ds in manifest['datasets']:
        paths=[sorted((LIVE/ds['dataset']).glob('frame_*'))[0]]
        paths += [LIVE/ds['dataset']/f"frame_{x['frame']:06d}" for x in ds['source_mismatch_frames']]
        for path in paths:
            scene=load_scene(path)
            if not live:scene=replace(scene,root=root/path.relative_to(LIVE))
            for vcount in sorted(set([1,min(2,len(scene.frames)),len(scene.frames)])):
                dataset=BlendedMVSDataset(scenes=[scene],num_views=vcount,image_width=768,image_height=432,
                    samples_per_scene=1,seed=1,use_masked_images=False,pfm_flip_y=False,
                    canonicalize_camera=False,scale_mode='raw',top_srcs=8,load_keypoints=True)
                sample=dataset[0]
                assert len(dataset.keypoint_names)==23
                assert torch.isfinite(sample['instance_keypoint_xy']).all()
                with np.load(scene.root/'keypoints.npz') as a:
                    valid=a['world_valid'];valid=valid[None] if valid.ndim==1 else valid
                    assert np.array_equal(sample['instance_world_valid'].numpy(),valid)
                    order=[int(v) for v in sample['view_ids'].split(',')]
                    vind=[a['view_ids'].tolist().index(v) for v in order]
                    vis=a['visibility_valid'][vind];vis=vis[:,None] if vis.ndim==2 else vis
                    assert np.array_equal(sample['instance_keypoint_visibility_valid'].numpy(),vis)
                    xy=a['xy_label_valid'][vind];xy=xy[:,None] if xy.ndim==2 else xy
                    assert np.array_equal(sample['instance_keypoint_label_valid'].numpy(),xy)
                results.append(dict(dataset=ds['dataset'],frame=path.name,views=vcount,status='passed'))
        print('DATALOADER_OK',ds['dataset'],flush=True)
    # Existing test module is executable directly; pytest is not installed here.
    import training.test_blendedmvs_multi_instance_loader as tests
    test_count=0
    for name in dir(tests):
        if name.startswith('test_'):
            with tempfile.TemporaryDirectory(prefix='surface-core23-test-') as temp:getattr(tests,name)(Path(temp))
            test_count+=1
    result=dict(status='passed',mode='live' if live else 'candidate',frames=frames,
                changed_json_points_checked=json_points,max_projection_error_px=max_projection_error,
                min_dorsal_ventral_distance_native_units=min_separation,loader_checks=results,
                existing_loader_tests_passed=test_count)
    (BASE/('validation_live.json' if live else 'validation_candidate.json')).write_text(json.dumps(result,indent=2))
    print('VALIDATED',frames,'frames;',len(results),'loader checks;',test_count,'tests',flush=True)
    if backup:backup.close()


def visualize():
    root=BASE/'candidate'
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,18)
    rows=[('WalkAround',31,[0,6,7]),('WashFace',31,[0,6,7]),('ClimbWall',31,[0,6,7]),('dual',1,[0,1,3])]
    canvas=Image.new('RGB',(1440,4*400+60),'#17202b');d=ImageDraw.Draw(canvas)
    d.text((18,12),'06 orange = dorsal thorax | 07 green = dorsal lumbar | 23 blue = ventral abdomen',font=font,fill='white')
    colors=['#ffac36','#66ee6b','#41c8ff']
    for row,(action,frame,views) in enumerate(rows):
        name=f'PyMouseStudio-BrownFur-{action}-Core23' if action!='dual' else 'Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose400-DepthHeightV1'
        folder=Path(name)/f'frame_{frame:06d}'
        with np.load(root/folder/'keypoints.npz') as z:a=dict(z)
        dual=a['uv'].ndim==4
        for col,vid in enumerate(views):
            vi=a['view_ids'].tolist().index(vid)
            im=Image.open(LIVE/folder/'blended_images'/f'{vid:08d}.jpg').convert('RGB')
            uv=a['uv'][vi] if dual else a['uv'][vi,None]
            # Tight common body crop, including all labeled body points, excludes tail.
            body=uv[:,:15].reshape(-1,2);lo=body.min(0)-40;hi=body.max(0)+40
            x0=max(0,int(lo[0]));y0=max(0,int(lo[1]));x1=min(im.width,int(hi[0]));y1=min(im.height,int(hi[1]))
            crop=im.crop((x0,y0,x1,y1));scale=min(465/crop.width,350/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
            ox=col*480+(480-crop.width)//2;oy=60+row*400+35
            canvas.paste(crop,(ox,oy));d.text((col*480+12,60+row*400),f'{action} frame {frame}, view {vid}',font=font,fill='white')
            for m in range(len(uv)):
                for j,k in enumerate(IDS):
                    x=ox+(float(uv[m,k,0])-x0)*scale;y=oy+(float(uv[m,k,1])-y0)*scale
                    visible=bool(a['surface_visible'][vi,m,k] if dual else a['surface_visible'][vi,k])
                    d.ellipse((x-5,y-5,x+5,y+5),outline=colors[j],fill=colors[j] if visible else None,width=2)
                    tx=x+16+(j%2)*20;ty=y-25+j*16
                    d.line((x,y,tx,ty+9),fill=colors[j],width=1)
                    label=f'{k+1:02d}' + (f' m{m}' if dual else '')
                    d.text((tx,ty),label,font=font,fill=colors[j],stroke_width=2,stroke_fill='black')
    canvas.save(BASE/'surface_keypoint_qa.jpg',quality=95)
    print(BASE/'surface_keypoint_qa.jpg')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--visualize',action='store_true');args=parser.parse_args()
    visualize() if args.visualize else main(args.live)
