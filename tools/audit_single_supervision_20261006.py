"""Read-only selected synthetic-single provenance and visible-point/mask audit."""
import sys,json,collections,shutil,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from preflight_mixed_local_backbone_20261005 import sha
C=ROOT/'output/local_grouping_cycle_20261005';INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1';OUT=C/'single_supervision_geometry_audit_v1';KP=[0,14,15,16,17]
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt') if r['metadata']['kind']=='single'};assert len(rows)==124
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};groups=collections.defaultdict(list)
    for i,r in rows.items():groups[(r['metadata']['scene'],tuple(manifest[i]['image_hw']))].append(i)
    details=[];hashes={};images={}
    for (scene,hw),ids in groups.items():
        views=sorted({rows[i]['metadata']['view'] for i in ids});item=load_item(dict(scene=scene,views=views,image_hw=list(hw),kind='single'));t=instance_targets_from_batch(item['batch']);hashes.update(item['source_hashes'])
        for i in ids:
            row=rows[i];v=views.index(row['metadata']['view']);h,w=hw;known=t['keypoint_valid'][0,v,:,KP]&t['instance_valid'][0,v,:,None]
            eligible=known&t['keypoint_visibility_valid'][0,v,:,KP]&t['keypoint_visible'][0,v,:,KP]
            assert torch.equal(eligible,row['eligible']) and torch.allclose(t['keypoint_xy'][0,v,:,KP][known],row['target'][known],atol=1e-6,rtol=0)
            rgb=(item['batch']['images'][0,v].permute(1,2,0)*255).round().byte().numpy();assert np.array_equal(rgb,np.asarray(Image.open(ROOT/manifest[i]['image'])));assert sha(ROOT/manifest[i]['image'])==manifest[i]['sha256']
            points=[];im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
            for g in t['instance_valid'][0,v].nonzero().flatten().tolist():
                assert t['mask_valid'][0,v,g];mask=t['masks'][0,v,g].bool().numpy();distance=distance_transform_edt(~mask)
                for k in eligible[g].nonzero().flatten().tolist():
                    xy=(row['target'][g,k]*torch.tensor([w,h])).tolist();x,y=xy;ix,iy=int(np.clip(round(x),0,w-1)),int(np.clip(round(y),0,h-1))
                    points.append(dict(keypoint=KP[k]+1,xy=xy,mask_distance_px=float(distance[iy,ix])))
                    color='yellow' if distance[iy,ix]<=16 else 'red';d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);d.text((x+5,y),str(KP[k]+1),fill=color)
            d.text((5,5),f'index{i} source={row["metadata"]["source"]} view={row["metadata"]["view"]} eligible={len(points)}',fill='yellow')
            details.append(dict(index=i,metadata=row['metadata'],points=points,max_mask_distance=max(p['mask_distance_px'] for p in points)));images[i]=im
    assert all(sha(ROOT/p)==s for p,s in hashes.items())
    summary={}
    for stage in ['fit','calibration']:
        subset=[r for r in details if r['metadata']['stage']==stage];pts=[p for r in subset for p in r['points']]
        summary[stage]=dict(images=len(subset),points=len(pts),outside16=sum(p['mask_distance_px']>16 for p in pts),outside8=sum(p['mask_distance_px']>8 for p in pts),max_distance=max(p['mask_distance_px'] for p in pts))
    selected=list(dict.fromkeys([194]+[r['index'] for r in sorted(details,key=lambda r:r['max_mask_distance'],reverse=True)[:15]]))
    for i in selected:images[i].save(OUT/f'gt_{i}.jpg',quality=93)
    domain={}
    for arm in ['independent','within_instance']:
        saved=json.loads((C/'proposal_grouping_continue1920_v1'/f'{arm}_step1920_details.json').read_text());by=collections.defaultdict(list)
        for r in saved:
            if r['metadata']['stage']=='calibration' and r['metadata']['kind']=='single':by[(r['metadata']['source'],r['metadata']['view'])].append(r)
        domain[arm]=[]
        for (source,view),ss in sorted(by.items()):
            pp=[p for s in ss for p in s['metrics']['points']]
            domain[arm].append(dict(source=source,view=view,images=len(ss),tp=sum(s['metrics']['tp64'] for s in ss),fp=sum(s['metrics']['fp64'] for s in ss),fn=sum(s['metrics']['fn64'] for s in ss),points=len(pp),joint_pck16=float(np.mean([p['joint64_error']<=16 for p in pp])) if pp else None))
    result=dict(summary=summary,details=details,source_hashes=hashes,source_rgb_point_visibility_parity=True,sources_unchanged=True,domain_metrics=domain,caveat='Visible projected points can sit slightly outside rendered mask due surface/fur/antialiasing. Distance is diagnostic, not automatic invalidation. Only existing124FIT/CAL selectedsingle images, noDEV/locked, no model update.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>合成单鼠可见点与掩码只读核验</h1><p>只画训练实际使用的有效可见点。背景器材上的模型预测不等于GT。未修改任何标签或筛选规则。</p><pre>'+json.dumps(dict(summary=summary,domain_metrics=domain),indent=2)+'</pre>'+''.join(f'<img style="max-width:768px" src="gt_{i}.jpg">' for i in selected),encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps(dict(stage='complete',utc=dt.datetime.now(dt.timezone.utc).isoformat())))
    shutil.copytree(OUT,Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name)
    print(json.dumps(dict(summary=summary,domain_metrics=domain),indent=2))
if __name__=='__main__':main()
