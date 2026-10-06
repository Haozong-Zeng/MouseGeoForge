"""Render actual bottom anchors and side predictions; all ranking is evaluation-only."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state

def main():
    _,_,check=budget(300);src=OUT/'epipolar_pilot_v1';dest=src/'review';dest.mkdir(exist_ok=False)
    rows=json.loads((src/'observations.json').read_text());summary=json.loads((src/'summary.json').read_text())
    m=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        data={(m['entries'][i]['session'],m['entries'][i]['frame'],m['entries'][i]['camera']):dict(entry=m['entries'][i],uv=z['uv_selected'][p,3:].copy(),gt=z['gt'][p,3:].copy()) for p,i in enumerate(z['manifest_indices'])}
    chosen=[]
    for category,name,direction,count in [('conservative_gain','manual_conservative',1,4),('conservative_harm','manual_conservative',-1,3),('argmin_harm','manual_argmin',-1,4)]:
        candidates=sorted(rows,key=lambda r:direction*(r['raw_error']-r['methods'][name]['direct_error']),reverse=True);used=set()
        for r in candidates:
            delta=r['raw_error']-r['methods'][name]['direct_error'];key=(r['session'],r['frame'],r['camera'])
            if direction*delta<=1 or key in used:continue
            used.add(key);chosen.append((category,r,delta))
            if len(used)>=count:break
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',18);large=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',23);files=[]
    for n,(category,r,delta) in enumerate(chosen,1):
        check();key=(r['session'],r['frame'],r['camera']);side=data[key];anchor=data[(r['session'],r['frame'],'A')]
        sheet=Image.new('RGB',(2160,700),'#202020');d=ImageDraw.Draw(sheet)
        d.text((8,8),f"{category}: {r['session']} frame{r['frame']} cam{r['camera']} point{r['id']} | error reduction {delta:.1f}px",font=large,fill='white')
        d.text((8,40),'Green square = human (bottom is INPUT, side is EVALUATION only); pink crosses = model including unknown points.',font=font,fill='white')
        panels=[('camA HUMAN INPUT',anchor,None),('SIDE raw predictions',side,'raw'),('SIDE conservative',side,'manual_conservative'),('SIDE argmin',side,'manual_argmin')]
        local={x['id']:x for x in rows if (x['session'],x['frame'],x['camera'])==key}
        group=json.loads((src/'groups.json').read_text());g=next(g for g in group if (g['session'],g['frame'])==key[:2])
        for column,(title,item,method) in enumerate(panels):
            im=Image.open(ROOT/item['entry']['source']).convert('RGB').resize((540,600),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(im)
            pts=item['uv'].copy()
            if method not in [None,'raw']:
                for pair in g['metadata'][method][r['camera']]:
                    if pair['swapped']:
                        a,b=pair['pair'];pts[[a,b]]=item['uv'][[b,a]]
            for j,kp in enumerate([15,16,17,18]):
                if np.isfinite(item['gt'][j]).all():
                    x,y=item['gt'][j]*.5;draw.rectangle((x-4,y-4,x+4,y+4),outline='#64ff6b',width=2)
                    draw.text((x+5,y-18),str(kp),font=font,fill='#64ff6b',stroke_width=1,stroke_fill='black')
                if method is not None:
                    u,v=pts[j]*.5;draw.line((u-4,v-4,u+4,v+4),fill='#ff75d5',width=2);draw.line((u-4,v+4,u+4,v-4),fill='#ff75d5',width=2)
                    draw.text((u+5,v+5),str(kp),font=font,fill='#ff75d5',stroke_width=1,stroke_fill='black')
            sheet.paste(im,(column*540,100));d.text((column*540+8,74),title,font=font,fill='white')
        filename=f"{n:02d}_{category}_{r['session']}_{r['frame']:06d}_cam{r['camera']}.jpg"
        sheet.save(dest/filename,quality=95);files.append(dict(file=filename,category=category,session=r['session'],frame=r['frame'],camera=r['camera'],id=r['id'],error_reduction_px=delta))
    shutil.copy2(src/'bottom_manual_audit.jpg',dest/'bottom_manual_audit.jpg')
    shutil.copy2(src/'summary.json',dest/'summary.json');(dest/'visual_manifest.json').write_text(json.dumps(files,indent=2))
    page='<html><meta charset="utf-8"><h1>底视参照：身份重分配初测</h1><p>只改候选ID，不改原标签。底视人工点是输入，侧面人工点只评估。先看底视人工标记约定，再对照改善与退化。</p><img width="1200" src="bottom_manual_audit.jpg">'
    page+='<table border="1"><tr><th>方法</th><th>侧面平均</th><th>侧面P90</th><th>换点数</th><th>改善&gt;5px</th><th>损害&gt;5px</th><th>共同LOO平均</th></tr>'
    for name,v in summary['methods'].items():page+=f'<tr><td>{name}</td><td>{v["side2d"]["mean"]:.2f}</td><td>{v["side2d"]["p90"]:.2f}</td><td>{v["swaps_on_labeled_points"]}</td><td>{v["swap_improved_gt5"]}</td><td>{v["swap_harmed_gt5"]}</td><td>{v["common_loo"]["mean"]:.2f}</td></tr>'
    page+='</table>'+''.join(f'<h2>{f["file"]}</h2><img width="1400" src="{f["file"]}">' for f in files)+'</html>'
    (dest/'index.html').write_text(page)
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002/epipolar_pilot_v1');assert not export.exists();shutil.copytree(dest,export)
    write_state(latest_visual_directory=str(export),latest_result=str(src/'summary.json'),next_action='Inspect strongest conservative gains, all conservative harms and argmin failures. Decide multi-view/depth-aware assignment, not threshold sweep.')
    print(json.dumps(files,indent=2))

if __name__=='__main__':main()
