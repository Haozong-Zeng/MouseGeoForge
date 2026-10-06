"""Read-only reconstruction audit; outputs diagnostic media, not training data."""
from pathlib import Path
import sys,json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from add_dannce_real_multiview_pseudo_depth import project,read_pfm,rasterize_depth
from export_pymousestudio_rectified import write
from repair_pymousestudio_limb_hull import component_metrics
cv2.setNumThreads(1)


def overlay(rgb,mask,color):
    im=rgb.copy();im[mask]=(.55*im[mask]+.45*np.array(color)).astype(np.uint8)
    return im


def main():
    s=ROOT/'dataset/PyMouseStudio/sessions/AZL3';base=s/'reconstruction_limb_guarded_v1'
    out=ROOT/'output/qa/pymousestudio_camF_20260912';out.mkdir(parents=True,exist_ok=True)
    rows=[];tiles=[];writer=None
    for f in range(780,812):
        rgb=cv2.imread(str(s/f'pilot_rgb/{f:06d}/camF.png'))
        src=cv2.imread(str(s/f'pilot_input_masks/{f:06d}/camF.png'),0)>0
        depth=read_pfm(base/f'frame_{f:06d}/rendered_depth_maps/00000005.pfm');mask=depth>0
        missing=src&~mask
        colored=cv2.applyColorMap(np.uint8(np.clip((depth-190)/65*255,0,255)),cv2.COLORMAP_TURBO)
        colored[~mask]=255
        panels=[rgb,overlay(rgb,src,(255,180,0)),overlay(rgb,mask,(0,255,70)),
                overlay(rgb,missing,(0,0,255)),colored]
        titles=['RGB','Input mask (not GT)','Current mask','Input minus current','Depth 190-255mm']
        # Fixed crop encloses the animal in all 32 frames; never interpolate masks for analysis.
        views=[]
        for im,title in zip(panels,titles):
            crop=im[200:1100,220:960].copy()
            crop=cv2.resize(crop,(370,450))
            cv2.rectangle(crop,(0,0),(369,35),(35,35,35),-1)
            cv2.putText(crop,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
            views.append(crop)
        montage=np.concatenate(views,1)
        cv2.putText(montage,f'camF source frame {f} | diagnostic, NOT ground truth',(15,442),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        cv2.imwrite(str(out/f'camF_{f:06d}.png'),montage)
        if writer is None:writer=cv2.VideoWriter(str(out/'camF_audit_8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(1850,450))
        writer.write(montage)
        tile=overlay(rgb,missing,(0,0,255))[200:1100,220:960]
        tile=cv2.resize(tile,(222,270));cv2.putText(tile,str(f),(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        tiles.append(tile)
        a=cv2.imread(str(base/f'frame_{f:06d}/limb_added_pixels/00000005.png'),0)>0
        seam_diffs=[]
        for axis in [0,1]:
            p=[slice(None),slice(None)];q=p.copy();p[axis]=slice(None,-1);q[axis]=slice(1,None)
            p,q=tuple(p),tuple(q);valid=mask[p]&mask[q]&(a[p]^a[q])
            seam_diffs.extend(np.abs(depth[p]-depth[q])[valid].tolist())
        rows.append(dict(frame=f,input_pixels=int(src.sum()),missing_pixels=int(missing.sum()),
            coverage_of_unverified_input=float((mask&src).sum()/src.sum()),components=component_metrics(mask),
            limb_seam_neighbor_diff_mm_median=float(np.median(seam_diffs)) if seam_diffs else None,
            limb_seam_neighbor_diff_mm_p95=float(np.percentile(seam_diffs,95)) if seam_diffs else None))
    writer.release()
    for page in range(2):
        group=tiles[page*16:(page+1)*16]
        cv2.imwrite(str(out/f'camF_missing_contact_{page+1}.png'),np.concatenate([np.concatenate(group[i:i+4],1) for i in range(0,16,4)],0))
    # Counterfactual constraint probes; these outputs are never published as repaired geometry.
    cfg=json.loads((s/'reconstruction_bottom_refined_v1/config.json').read_text())
    cal=json.loads((s/'calibration.json').read_text());K={n:np.array(cal[n]['K']) for n in 'ABCDEFG'};E={n:np.array(cal[n]['world_to_camera_mm']) for n in 'ABCDEFG'}
    low,high=np.array(cfg['bounds_mm']);axes=[np.arange(low[i],high[i]+.5) for i in range(3)]
    pts=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    cache={}
    for n in 'ABCDEFG':
        uv,z=project(pts,K[n],E[n]);x,y=np.rint(uv).astype(int).T
        cache[n]=(np.clip(y,0,1199)*1080+np.clip(x,0,1079),(z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200))
    probes=[]
    for f in [780,802,803,811]:
        masks={n:cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'),0) for n in 'BCDEFG'}
        olda=cv2.imread(str(s/f'reconstruction_bottom_refined_v1/frame_{f:06d}/bottom_sam2_mask.png'),0)
        fulla=cv2.imread(str(s/f'tail_dlc_video_masks_v1/masks/{f:06d}.png'),0)
        hit={n:(masks[n].ravel()[cache[n][0]]>0)&cache[n][1] for n in 'BCDEFG'}
        votes=sum(h.astype(np.uint8) for h in hit.values())
        ahit=(olda.ravel()[cache['A'][0]]>0)&cache['A'][1]
        fullhit=(fulla.ravel()[cache['A'][0]]>0)&cache['A'][1]
        current=read_pfm(base/f'frame_{f:06d}/rendered_depth_maps/00000005.pfm')>0
        rgb=cv2.imread(str(s/f'pilot_rgb/{f:06d}/camF.png'));miss=(masks['F']>0)&~current
        # Image-left body strip, avoids most tail and feet; explicit diagnostic ROI.
        roi=np.zeros_like(miss);roi[350:920,260:420]=True;target=miss&roi
        comparisons=[]
        for name,sel in [('5side_oldA',(votes>=5)&ahit),('5side_fullA',(votes>=5)&fullhit),
                         ('5side_noA',votes>=5),('4side_fullA',(votes>=4)&fullhit)]:
            rendered=rasterize_depth(pts[sel],masks['F'],K['F'],E['F'],1.)
            candidate=rendered>0
            probe=dict(frame=f,variant=name,missing_body_strip_pixels=int(target.sum()),
                recovered_missing_body_strip_pixels=int((candidate&target).sum()),
                warning='Counterfactual image support only, not accuracy; no published data changes.')
            if name=='5side_fullA':
                original=read_pfm(base/f'frame_{f:06d}/rendered_depth_maps/00000005.pfm')
                additions=cv2.imread(str(base/f'frame_{f:06d}/limb_added_pixels/00000005.png'),0)>0
                seam=[[],[]]
                for axis in [0,1]:
                    p=[slice(None),slice(None)];q=p.copy();p[axis]=slice(None,-1);q[axis]=slice(1,None)
                    p,q=tuple(p),tuple(q)
                    valid=current[p]&current[q]&candidate[p]&candidate[q]&(additions[p]^additions[q])
                    seam[0].extend(abs(original[p]-original[q])[valid].tolist())
                    seam[1].extend(abs(rendered[p]-rendered[q])[valid].tolist())
                probe['same_seam_pixel_pairs']=len(seam[0])
                if seam[0]:
                    probe['current_seam_p95_mm']=float(np.percentile(seam[0],95))
                    probe['unified_fullA_render_seam_p95_mm']=float(np.percentile(seam[1],95))
            probes.append(probe)
            im=overlay(rgb,candidate,(0,255,70))[200:1100,220:960]
            im=cv2.resize(im,(370,450));cv2.putText(im,name,(8,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
            comparisons.append(im)
        cv2.imwrite(str(out/f'constraint_probe_{f:06d}.png'),np.concatenate(comparisons,1))
    write(out/'report.json',dict(frames=rows,constraint_probes=probes,production_files_changed=False,train_allowed=False,
        note='Mask coverage is not independent accuracy. Seam jumps may include true occlusion boundaries.'))
    print(json.dumps(dict(frames=32,coverage_min=min(r['coverage_of_unverified_input'] for r in rows),
         coverage_max=max(r['coverage_of_unverified_input'] for r in rows),probes=probes),indent=2))


if __name__=='__main__':main()
