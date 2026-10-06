"""Scientific audit figures, never model predictions or corrected annotations."""
import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_pymousestudio_manual_multiview import projection,stats

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OUT=ROOT/'output/pymousestudio_manual_multiview_20260930'
ACTIVE=[0,5,6,7,13,14,15,16,17,22]
FONT='/mnt/c/Windows/Fonts/msyh.ttc'

def main():
    ingest=json.loads((BASE/'ingestion_manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    observations=json.loads((OUT/'observation_metrics.json').read_text());points=json.loads((OUT/'point_metrics.json').read_text())
    font=lambda n:ImageFont.truetype(FONT,n)
    fig,ax=plt.subplots(figsize=(12,10),layout='constrained')
    names=[next(p['name'] for p in points if p['id']==k) for k in range(1,24)]
    ls=[summary['keypoints'][str(k)]['all_ls_observations_3plus']['median'] for k in range(1,24)]
    loo=[summary['keypoints'][str(k)]['loo_observations']['median'] for k in range(1,24)]
    ax.barh(np.arange(23),loo,color=['#267fba' if k in ACTIVE else '#8d979f' for k in range(23)],label='Leave-one-view-out median')
    ax.scatter(ls,np.arange(23),color='#e87924',marker='D',label='All-view LS median',zorder=3)
    ax.set_yticks(np.arange(23),[f'{k+1:02d} {n}' for k,n in enumerate(names)]);ax.invert_yaxis();ax.set_xlabel('Reprojection error (pixels, original 1080 x 1200 images)')
    ax.set_title('Manual label multiview consistency | 300 groups | fixed calibration\nBlue: 10 approved active slots; gray: other provisional slots');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'keypoint_consistency.png',dpi=140);plt.close(fig)
    cams=list('ABCDEFG');matrix=np.full((7,7),np.nan)
    for i,a in enumerate(cams):
        for j,b in enumerate(cams):
            if i!=j:matrix[i,j]=summary['active_epipolar_pairs']['-'.join(sorted([a,b]))]['median']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    vals=[]
    for c in cams:
        vals.append(stats([r['loo_reprojection_px'] for r in observations if r['camera']==c and r['active'] and r['loo_reprojection_px'] is not None])['median'])
    axes[0].bar(cams,vals,color=['#ce633f' if c in 'AF' else '#267fba' for c in cams]);axes[0].set_ylabel('LOO median error (px)');axes[0].set_title('Active landmarks, by camera')
    heat=axes[1].imshow(matrix,vmin=0,vmax=30,cmap='YlOrRd');axes[1].set_xticks(range(7),cams);axes[1].set_yticks(range(7),cams);axes[1].set_title('Active landmark epipolar median (px)')
    for i in range(7):
        for j in range(7):
            if i!=j:axes[1].text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='black')
    fig.colorbar(heat,ax=axes[1]);fig.savefig(OUT/'camera_consistency.png',dpi=150);plt.close(fig)
    # Fixed first-frame sample from every session plus one best/one worst dev group.
    grouped=defaultdict(list)
    for r in observations:
        if r['split']=='real_development' and r['active'] and r['loo_reprojection_px'] is not None:grouped[(r['session'],r['frame'])].append(r['loo_reprojection_px'])
    ranked=sorted((float(np.median(v)),k) for k,v in grouped.items() if len(v)>=30)
    chosen=[(s,0) for s in ingest['sessions']]+[ranked[0][1],ranked[-1][1]]
    galleries=[]
    for session,frame in chosen:
        g=next(g for g in ingest['frame_groups'] if g['session']==session and g['frame']==frame);split=g['split'];cams=g['cameras']
        npz=BASE/'geometry_diagnostics_NOT_GT'/split/session/f'frame_{frame:06d}.npz'
        with np.load(npz) as z:a=dict(z)
        stem=f'{session}_frame_{frame:06d}';folder=OUT/'examples'/stem;folder.mkdir(parents=True,exist_ok=True)
        sheet=Image.new('RGB',(1500,80+math.ceil(len(cams)/3)*540),'#17212c');d=ImageDraw.Draw(sheet)
        d.text((15,10),f'{session} 源帧{frame} | 绿圈=人工标注，洋红×=全视角LS，蓝×=留一视角重投影',font=font(22),fill='white')
        d.text((15,44),'仅展示10个已确认槽位；未改原标注。误差为像素；3D仅诊断，不是真值。',font=font(20),fill='#bed0df')
        for vi,c in enumerate(cams):
            src=BASE/'real_images'/split/session/f'frame_{frame:06d}'/f'cam{c}.png';raw=Image.open(src).convert('RGB')
            label=json.loads(src.with_suffix('.json').read_text());coords=np.array([xy for s in label['shapes'] for xy in s['points']],float)
            lo=np.maximum(coords.min(0)-35,0);hi=np.minimum(coords.max(0)+35,[1080,1200]);box=tuple([int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])])
            overlay=raw.copy();od=ImageDraw.Draw(overlay);proj=np.full((23,2),np.nan)
            for k in ACTIVE:
                if np.isfinite(a['xyz_all_ls_mm'][k]).all():proj[k]=projection(a['xyz_all_ls_mm'][k],a['K'][vi:vi+1],a['world_to_camera_mm'][vi:vi+1])[0][0]
                uv=a['uv_observed'][vi,k]
                if not np.isfinite(uv).all():continue
                x,y=uv;od.ellipse((x-5,y-5,x+5,y+5),outline='#77ff55',width=2);od.text((x+7,y-17),str(k+1),font=font(19),fill='#77ff55',stroke_width=1,stroke_fill='black')
                for p,color in [(proj[k],'#ff59d0'),(a['uv_leave_one_view_out'][vi,k],'#47cfff')]:
                    if np.isfinite(p).all() and -1080<p[0]<2160 and -1200<p[1]<2400:
                        u,v=p;od.line((x,y,u,v),fill=color,width=1);od.line((u-5,v-5,u+5,v+5),fill=color,width=2);od.line((u-5,v+5,u+5,v-5),fill=color,width=2)
            left=raw.crop(box);right=overlay.crop(box)
            detail=Image.new('RGB',(left.width*2,left.height+46),'#17212c');detail.paste(left,(0,46));detail.paste(right,(left.width,46))
            ImageDraw.Draw(detail).text((10,8),f'{session} frame {frame} cam{c} 原图 / 标注与重投影',font=font(22),fill='white');detail.save(folder/f'cam{c}_comparison.jpg',quality=95)
            panel=right.copy();panel.thumbnail((485,455));ox=(vi%3)*500;oy=80+(vi//3)*540
            err=a['error_leave_one_view_out_px'][vi,ACTIVE];valid=err[np.isfinite(err)]
            d.text((ox+10,oy+4),f'cam{c} | LOO中位 {np.median(valid):.1f}px' if len(valid) else f'cam{c} | 无LOO结果',font=font(23),fill='white')
            sheet.paste(panel,(ox+(500-panel.width)//2,oy+45))
        sheet.save(folder/'multiview_reprojection.jpg',quality=95)
        # Two static angles in original calibration coordinates, equal metric scaling.
        fig=plt.figure(figsize=(12,5),layout='constrained');valid=[k for k in ACTIVE if np.isfinite(a['xyz_all_ls_mm'][k]).all()]
        xyz=a['xyz_all_ls_mm'][valid];center=np.median(xyz,axis=0);span=max(np.ptp(xyz,axis=0).max(),20)/2+10
        for i,azim in enumerate([30,135]):
            ax=fig.add_subplot(1,2,i+1,projection='3d');ax.scatter(*xyz.T,c=valid,cmap='tab20',s=40)
            for k,x in zip(valid,xyz):ax.text(*x,str(k+1),fontsize=9)
            for dim,setter in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):setter(center[dim]-span,center[dim]+span)
            ax.set_box_aspect((1,1,1));ax.view_init(25,azim);ax.set_xlabel('X mm');ax.set_ylabel('Y mm');ax.set_zlabel('Z mm');ax.set_title(f'Calibration coordinates | azimuth {azim}')
        fig.suptitle(f'{session} frame {frame}: all-view LS active points (NOT ground truth)');fig.savefig(folder/'triangulated_3d.png',dpi=140);plt.close(fig)
        galleries.append(dict(session=session,frame=frame,split=split,folder=f'examples/{stem}'))
        print('VISUALIZED',stem,flush=True)
    (OUT/'visualization_manifest.json').write_text(json.dumps(galleries,indent=2))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>人工标注多视角一致性检查</title><body style="background:#17212c;color:white;font:18px sans-serif"><h1>1950张人工标注 多视角一致性审计</h1><p>原标注未改；绿色人工点、洋红全视角拟合、蓝色留一视角重投影。3D是诊断输出，不是已验收真值。留出集维持隔离。</p><h2>整体统计</h2><img style="max-width:100%" src="keypoint_consistency.png"><img style="max-width:100%" src="camera_consistency.png">'+''.join(f'<h2>{g["session"]} 源帧{g["frame"]}</h2><a href="{g["folder"]}/multiview_reprojection.jpg"><img style="max-width:100%" src="{g["folder"]}/multiview_reprojection.jpg"></a><img style="max-width:100%" src="{g["folder"]}/triangulated_3d.png"><p>'+ ' · '.join(f'<a style="color:#80d6ff" href="{g["folder"]}/cam{c}_comparison.jpg">cam{c}原图对照</a>' for c in next(x['cameras'] for x in ingest['frame_groups'] if x['session']==g['session'] and x['frame']==g['frame']))+'</p>' for g in galleries))

if __name__=='__main__':main()
