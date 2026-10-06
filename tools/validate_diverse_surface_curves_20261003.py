"""Independent projection/hash QA plus inspectable overlays for candidate sidecars."""
import json,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
from PIL import Image,ImageDraw
from training.finetune_blendedmvs import load_scene
BASE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_gt_v1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    data=json.loads((BASE/'manifest.json').read_text());reports=[];accepted=[];folder=BASE/'visuals';folder.mkdir(exist_ok=True)
    historical=set()
    if BASE.name=='diverse_curve_gt_v2':historical=set(json.loads((BASE.parent/'diverse_curve_gt_v1/qa.json').read_text())['rejected_scenes'])
    for row in data['records']:
        scene=ROOT/row['scene'];path=scene/'keypoints.npz';assert sha(path)==row['source_annotation_sha256']
        with np.load(path) as a:kp=dict(a)
        with np.load(ROOT/row['sidecar']) as a:curve=dict(a)
        m=row['instances'];assert curve['world_xyz'].shape==(m,65,3) and curve['uv'].shape==(len(curve['view_ids']),m,65,2)
        assert not curve['visibility_valid'].any()
        residual=[]
        for v,(E,K) in enumerate(zip(kp['extrinsics'].astype(float),kp['intrinsics'].astype(float))):
            p=np.einsum('ij,mnj->mni',E[:3,:3],curve['world_xyz'].astype(float))+E[:3,3];q=np.einsum('ij,mnj->mni',K,p)
            uv=q[...,:2]/q[...,2:3];residual.append(float(abs(uv-curve['uv'][v]).max()))
        assert max(residual)<.01,(row['scene'],max(residual))
        assert curve['position_valid'].all()==row['accepted_for_training']
        eligible=row['accepted_for_training'] and row['scene'] not in historical
        reports.append(dict(scene=row['scene'],role=row['role'],accepted=eligible,current_geometry_passed=row['accepted_for_training'],historical_mismatch_exclusion=row['scene'] in historical,projection_max_px=max(residual),source_support_error=row['max_support_error']))
        if eligible:accepted.append(row)
    # Choose visualization examples deterministically by source and split, not errors.
    buckets=collections.defaultdict(list)
    for row in accepted:buckets[(row['dataset'],row['role'])].append(row)
    visuals=[]
    for (name,role),rows in sorted(buckets.items()):
        row=sorted(rows,key=lambda x:x['scene'])[len(rows)//2];scene=load_scene(ROOT/row['scene'])
        with np.load(ROOT/row['sidecar']) as a:cur=dict(a)
        with np.load(ROOT/row['scene']/'keypoints.npz') as a:kp=dict(a)
        vids=[i for i in [0,2] if i in cur['view_ids']];panels=[]
        for vid in vids:
            v=cur['view_ids'].tolist().index(vid);image=Image.open(scene.frames[vid].image_path).convert('RGB');w,h=image.size
            p=cur['uv'][v];allxy=p.reshape(-1,2);lo=np.maximum(allxy.min(0)-[.12*w,.12*h],0);hi=np.minimum(allxy.max(0)+[.12*w,.12*h],[w,h])
            if np.any(hi-lo<5):lo=np.zeros(2);hi=np.array([w,h])
            draw=ImageDraw.Draw(image)
            for inst,line in enumerate(p):
                color=['#ffdd33','#33ddff'][inst%2];draw.line([tuple(x) for x in line],fill=color,width=max(2,round(w/350)))
                xy=kp['uv'][v] if row['instances']==1 else kp['uv'][v,inst]
                for k in [5,6]:
                    if np.isfinite(xy[k]).all():
                        x,y=xy[k];r=max(3,w/200);draw.ellipse((x-r,y-r,x+r,y+r),fill='#50ff60')
            crop=image.crop((*lo,*hi));crop.thumbnail((600,400));panel=Image.new('RGB',(600,425),(24,24,28));panel.paste(crop,(0,25));ImageDraw.Draw(panel).text((5,5),f'{role} view{vid} curve=amodal (visibility UNKNOWN)',fill='white');panels.append(panel)
        canvas=Image.new('RGB',(600*len(panels),425),(24,24,28))
        for i,p in enumerate(panels):canvas.paste(p,(600*i,0))
        filename=f'{len(visuals):02d}_{role}.jpg';canvas.save(folder/filename,quality=95);visuals.append(dict(file='visuals/'+filename,scene=row['scene']))
    payload=dict(numeric_passed=True,visual_review_pending=True,records=reports,accepted=len(accepted),rejected=len(reports)-len(accepted),max_projection_error_px=max(x['projection_max_px'] for x in reports),visuals=visuals,
        rejected_scenes=[r['scene'] for r in reports if not r['accepted']],effective_accepted_scenes=[r['scene'] for r in accepted],source_datasets_unchanged=True,not_published_into_dataset=True)
    (BASE/'qa.json').write_text(json.dumps(payload,indent=2))
    html=['<!doctype html><meta charset="utf-8"><title>Diverse curve candidate QA</title><h1>Mesh-derived candidate curves</h1><p>Yellow/cyan: independent mouse surface curves. Green: existing06/07anchors. Curves include occluded portions; visibility UNKNOWN, no visibility loss. Failed source-geometry rows are excluded.</p>']
    for r in visuals:html.append(f'<h2>{r["scene"]}</h2><img style="max-width:100%" src="{r["file"]}">')
    (BASE/'index.html').write_text('\n'.join(html));print(json.dumps({k:payload[k] for k in ['accepted','rejected','max_projection_error_px','rejected_scenes']}),flush=True)
if __name__=='__main__':main()
