"""CPU RGB-colored pointcloud review from fixed seed17 predictions; no model calls."""
from pathlib import Path
import datetime as dt,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image,ImageDraw
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';OLD=ROOT/'output/calibrated_mouse_10h_20261025';DEST=OUT/'pointcloud_review_v1'
VERTEX=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('source_view','u1')])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unproject(depth,mask,k,e,rgb):
    valid=mask&np.isfinite(depth)&(depth>0);y,x=np.where(valid);xy=np.c_[x,y]
    uv=np.c_[x,y,np.ones(len(x),dtype=np.int64)]
    camera=(uv@np.linalg.inv(k).T)*depth[y,x,None]
    world=(camera-e[:3,3])@e[:3,:3]
    return world,rgb[y,x],xy,depth[y,x]
def project(world,k,e):
    camera=world@e[:3,:3].T+e[:3,3];uv=camera@k.T
    return uv[:,:2]/uv[:,2,None],camera[:,2]
def write_ply(path,world,rgb,view):
    assert len(world)==len(rgb)==len(view) and np.isfinite(world).all()
    assert rgb.dtype==np.uint8 and view.dtype==np.uint8
    records=np.empty(len(world),dtype=VERTEX)
    for i,key in enumerate(['x','y','z']):records[key]=world[:,i]
    for i,key in enumerate(['red','green','blue']):records[key]=rgb[:,i]
    records['source_view']=view
    header='ply\nformat binary_little_endian 1.0\ncomment coordinates world millimeters; camera_Z input; no scale alignment\ncomment synchronized visible predicted surfaces, concatenated not watertight\nelement vertex '+str(len(world))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar source_view\nend_header\n'
    with Path(path).open('wb') as f:f.write(header.encode('ascii'));records.tofile(f)
    return records

def read_ply(path):
    with Path(path).open('rb') as f:
        lines=[]
        while True:
            line=f.readline();assert line
            lines.append(line.decode('ascii').strip())
            if lines[-1]=='end_header':break
        count=int(next(x for x in lines if x.startswith('element vertex ')).split()[-1]);records=np.fromfile(f,dtype=VERTEX)
    assert len(records)==count;return records

def palette(depth,valid,low,high):
    knots=np.array([[30,50,190],[0,205,220],[245,230,30],[210,30,30]],float)
    safe=np.where(np.isfinite(depth),depth,low);level=np.clip((safe.astype(float)-low)/max(1e-9,high-low),0,1)*3;index=np.minimum(level.astype(int),2);fraction=level-index
    color=knots[index]*(1-fraction[...,None])+knots[index+1]*fraction[...,None];color[~valid]=0
    return color.astype(np.uint8)

def main():
    assert not DEST.exists(),'Preserve existing review attempt'
    manifest_path=OLD/'data_manifest.json';manifest=json.loads(manifest_path.read_text());scenes=manifest['scenes'];indices=[next(i for i,s in enumerate(scenes) if s['source']==source and s['split']=='dev') for source in ['climb','moseq']]
    assert indices==[0,360]
    DEST.mkdir();inputs={str(manifest_path.relative_to(ROOT)):sha(manifest_path)};clouds={};records=[];panels=[]
    for index in indices:
        scene=scenes[index];meta_path=ROOT/scene['metadata'];assert sha(meta_path)==scene['metadata_sha256'];inputs[scene['metadata']]=sha(meta_path);meta=json.loads(meta_path.read_text())
        saved={};original_ply={}
        for mode in ['full','roi']:
            directory=OUT/f'roi_pipeline_s17_{mode}_v1';evaluation=directory/'evaluation.json';ev=json.loads(evaluation.read_text());assert ev['calibration_fixed'] and not ev['oracle_mask_inference'] and not ev['gt_scale_alignment']
            assert ev['full_scene_failure_denominator']==len(ev['scenes'])==144
            inputs[str(evaluation.relative_to(ROOT))]=sha(evaluation)
            folder=directory/f'scene{index:06d}';camera_json=folder/'cameras.json';cams=json.loads(camera_json.read_text());assert cams['depth_units']=='millimeter' and cams['depth_semantics']=='camera_z'
            inputs[str(camera_json.relative_to(ROOT))]=sha(camera_json)
            original_ply[mode]=folder/'predicted_world_mm.ply';inputs[str(original_ply[mode].relative_to(ROOT))]=sha(original_ply[mode]);saved[mode]=[]
            for view in range(3):
                path=folder/f'view{view}.npz';inputs[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path) as z:saved[mode].append({key:z[key].copy() for key in ['depth_mm','mask','K','E']})
        all_data={m:dict(world=[],rgb=[],xy=[],view=[],z=[]) for m in ['full','roi']}
        for view,source in enumerate(scene['views']):
            image_path=ROOT/source['image'];assert sha(image_path)==source['image_sha256'];inputs[source['image']]=sha(image_path);rgb=np.array(Image.open(image_path).convert('RGB'))
            a,b=saved['full'][view],saved['roi'][view];assert np.array_equal(a['mask'],b['mask']) and np.array_equal(a['K'],b['K']) and np.array_equal(a['E'],b['E'])
            assert a['depth_mm'].shape==a['mask'].shape==rgb.shape[:2]
            assert np.max(np.abs(a['K'].astype(float)-np.array(meta['views'][view]['intrinsic'])))<1e-4
            assert np.max(np.abs(a['E'].astype(float)-np.array(meta['views'][view]['extrinsic'])))<1e-4
            for mode in ['full','roi']:
                d=saved[mode][view];world,colors,xy,z=unproject(d['depth_mm'],d['mask'],d['K'],d['E'],rgb);uv,zback=project(world,d['K'],d['E']);pixel_error=float(np.max(np.abs(uv-xy))) if len(xy) else 0.
                assert pixel_error<.001
                all_data[mode]['world'].append(world);all_data[mode]['rgb'].append(colors);all_data[mode]['xy'].append(xy);all_data[mode]['z'].append(z);all_data[mode]['view'].append(np.full(len(world),view,np.uint8))
                records.append(dict(scene=index,source=scene['source'],view=view,mode=mode,predicted_mask_pixels=int(d['mask'].sum()),valid_points=len(world),invalid_predicted_mask_pixels=int(d['mask'].sum())-len(world),native_reprojection_max_abs_pixel=pixel_error,cameraZ_backprojection_max_abs_mm=float(np.max(np.abs(zback-z))) if len(z) else 0.))
            mask=a['mask'];y,x=np.where(mask);assert len(x)
            # Display crop is predicted-mask-only, never used for inference or export filtering.
            x0,x1=max(0,int(x.min())-16),min(rgb.shape[1],int(x.max())+17);y0,y1=max(0,int(y.min())-16),min(rgb.shape[0],int(y.max())+17)
            pool=np.concatenate([d['depth_mm'][d['mask']&np.isfinite(d['depth_mm'])&(d['depth_mm']>0)] for d in [a,b]]);low,high=float(pool.min()),float(pool.max())
            mask_overlay=rgb.copy();mask_overlay[mask]=(.6*rgb[mask]+np.array([0,220,0])*.4).astype(np.uint8)
            pictures=[rgb,mask_overlay[y0:y1,x0:x1]]
            for d in [a,b]:pictures.append(palette(d['depth_mm'],d['mask']&np.isfinite(d['depth_mm'])&(d['depth_mm']>0),low,high)[y0:y1,x0:x1])
            panels.append(dict(label=f'{scene["source"]} scene{index} view{view} camera-Z [{low:.1f},{high:.1f}] mm',pictures=pictures))
        for mode,parts in all_data.items():
            combined={key:np.concatenate(values) for key,values in parts.items()};clouds[(index,mode)]=combined['world'],combined['rgb']
            source_xyz=np.loadtxt(original_ply[mode],skiprows=7);assert source_xyz.shape==combined['world'].shape
            assert np.max(np.abs(source_xyz-combined['world']))<=1e-6
            name=f'{scene["source"]}_scene{index:06d}_{mode}';path=DEST/(name+'_rgb_world_mm.ply');write_ply(path,combined['world'],combined['rgb'],combined['view']);decoded=read_ply(path)
            xyz=np.c_[decoded['x'],decoded['y'],decoded['z']];assert np.max(np.abs(xyz-combined['world']))<5e-5
            np.testing.assert_array_equal(np.c_[decoded['red'],decoded['green'],decoded['blue']],combined['rgb']);np.testing.assert_array_equal(decoded['source_view'],combined['view'])
            np.savez_compressed(DEST/(name+'_pixel_receipt.npz'),**combined,K=np.stack([d['K'] for d in saved[mode]]),E=np.stack([d['E'] for d in saved[mode]]))
    panel=Image.new('RGB',(1200,len(panels)*310),'white');draw=ImageDraw.Draw(panel)
    for i,row in enumerate(panels):
        for j,(picture,title) in enumerate(zip(row['pictures'],['native RGB','predicted-mask crop','FULL camera-Z','ROI camera-Z'])):
            image=Image.fromarray(picture);image.thumbnail((296,272));panel.paste(image,(j*300+(300-image.width)//2,i*310+34+(272-image.height)//2));draw.text((j*300+3,i*310+17),title,fill='black')
        draw.text((3,i*310+2),row['label'],fill='black')
    panel.save(DEST/'native_rgbd_review.png')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(13,10))
    for row,index in enumerate(indices):
        both=np.concatenate([clouds[(index,m)][0] for m in ['full','roi']]);lo,hi=both.min(0),both.max(0);center=(lo+hi)/2;radius=max(hi-lo)*.55
        for col,mode in enumerate(['full','roi']):
            ax=figure.add_subplot(2,2,row*2+col+1,projection='3d');points,color=clouds[(index,mode)];ax.scatter(points[:,0],points[:,1],points[:,2],c=color/255,s=1,depthshade=False)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='world X (mm)',ylabel='world Y (mm)',zlabel='world Z (mm)',title=f'{scenes[index]["source"]} {mode.upper()} | {len(points)} points')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-50)
    figure.suptitle('Fixed seed17 predicted-mask visible points | RGB colors | equal mm axes | no alignment');figure.tight_layout();figure.savefig(DEST/'colored_pointcloud_review.png',dpi=140);plt.close(figure)
    result=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),selection='First admitted DEV scene in manifest for each source, before viewing predicted geometry',scene_indices=indices,seed=17,modes=['full','roi'],source_sha256=sha(__file__),input_sha256=inputs,views=records,export_vertex_count=sum(r['valid_points'] for r in records),
        geometry='Existing pipeline world-mm formula and source point order exactly replayed to1e-6mm; binaryfloat32 export<=5e-5mm; all valid predicted points retained',confidence_filtering=False,GT_inputs_used=False,RGB_from_native_same_pixel=True,model_calls=0,new_dataset_admission=False,
        color_scale='Per view shared FULL/ROI min/max positive prediction camera-Z, visualization only; depth values/metrics unchanged',
        limitations=['Visible points concatenated across same synthetic synchronized instant; overlapping samples retained, not watertight mesh or complete mouse shape.','Fixed original M18000 predicted masks; no GT crop/scale fitting or model promotion.','Synthetic opaque skin supervision excludes fur/glass; independent real-mm accuracy unverified.','Two first-index examples are review artifacts, not a replacement for144scene432view metrics.'])
    result['output_sha256']={p.name:sha(p) for p in sorted(DEST.iterdir()) if p.is_file()}
    (DEST/'completion.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_pointcloud_review_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['scene_indices','export_vertex_count','model_calls']}))
if __name__=='__main__':main()
