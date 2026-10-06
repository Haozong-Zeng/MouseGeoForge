"""Actual-image diagnostic gallery; GT used only to rank review examples."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bottom_anchor_common_20261002 import ROOT,OUT,PRIOR,budget,write_state


def main():
    _,_,check=budget(300)
    src=OUT/'multiview_consensus_v1';dest=src/'review';dest.mkdir(exist_ok=False)
    rows=json.loads((src/'observations.json').read_text());summary=json.loads((src/'summary.json').read_text())
    groups={(g['session'],g['frame']):g for g in json.loads((src/'groups.json').read_text())}
    manifest=json.loads((PRIOR/'lp_context_dataset_v1/manifest.json').read_text())
    with np.load(PRIOR/'lp_context_seed17_kl/predictions_source_pixels.npz') as z:
        data={(manifest['entries'][int(i)]['session'],manifest['entries'][int(i)]['frame'],manifest['entries'][int(i)]['camera']):
              dict(entry=manifest['entries'][int(i)],pred=z['uv_selected'][p,3:].copy(),gt=z['gt'][p,3:].copy()) for p,i in enumerate(z['manifest_indices'])}
    selections=[]
    for category,name,direction in [('manual_gain','manual_flexible',1),('manual_harm','manual_flexible',-1),('predicted_harm','predicted_flexible',-1)]:
        eligible=[r for r in rows if r['methods'][name]['direct_error'] is not None]
        ranked=sorted(eligible,key=lambda r:direction*(r['raw_error']-r['methods'][name]['direct_error']),reverse=True)
        used=set()
        for r in ranked:
            key=(r['session'],r['frame'],r['camera'])
            if key in used or direction*(r['raw_error']-r['methods'][name]['direct_error'])<1:continue
            used.add(key);selections.append((category,r))
            if len(used)>=2:break
    for category,eligible,sortkey in [
        ('manual_LOO_failure',[r for r in rows if r['methods']['manual_flexible']['loo_error'] is not None],lambda r:r['methods']['manual_flexible']['loo_error']),
        ('unknown_raw_failure',[r for r in rows if r['methods']['manual_flexible']['direct_error'] is None],lambda r:r['raw_error'])]:
        used=set()
        for r in sorted(eligible,key=sortkey,reverse=True):
            key=(r['session'],r['frame'],r['camera'])
            if key in used:continue
            used.add(key);selections.append((category,r))
            if len(used)>=2:break
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',17)
    files=[]
    for n,(category,r) in enumerate(selections,1):
        check();key=(r['session'],r['frame'],r['camera']);bottom=data[(*key[:2],'A')];side=data[key]
        group=groups[key[:2]];vi=group['cameras'].index(r['camera']);local={x['id']:x for x in rows if (x['session'],x['frame'],x['camera'])==key}
        sheet=Image.new('RGB',(2160,735),'#202020');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f"{category}: {r['session']} frame {r['frame']} cam{r['camera']} focus kp{r['id']}",font=font,fill='white')
        draw.text((8,29),'Green square=manual; pink cross=candidate; yellow circle=TRUE held-view-out projection. Missing pink=UNKNOWN, not invisible.',font=font,fill='white')
        draw.text((8,52),'Bottom manual=INPUT; side manual=EVALUATION ONLY. LOO excludes this side before every selection. No original label changes.',font=font,fill='white')
        for col,method in enumerate([None,'raw','manual_flexible','predicted_flexible']):
            item=bottom if method is None else side
            im=Image.open(ROOT/item['entry']['source']).convert('RGB').resize((540,600),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
            for j,kp in enumerate([15,16,17,18]):
                if np.isfinite(item['gt'][j]).all():
                    x,y=item['gt'][j]/2;d.rectangle((x-4,y-4,x+4,y+4),outline='#60ff60',width=2)
                    d.text((x+5,y-20),str(kp),font=font,fill='#60ff60',stroke_width=1,stroke_fill='black')
                point=item['pred'][j] if method in [None,'raw'] else None
                if method not in [None,'raw']:
                    result=group['full'][method][j//2]
                    mapped=result['mapping'][vi][j%2] if result['accepted'] else -1
                    if mapped>=0:point=item['pred'][2*(j//2)+mapped]
                if point is not None and np.isfinite(point).all():
                    x,y=point/2;d.line((x-4,y-4,x+4,y+4),fill='#ff75d5',width=2);d.line((x-4,y+4,x+4,y-4),fill='#ff75d5',width=2)
                    d.text((x+5,y+5),str(kp),font=font,fill='#ff75d5',stroke_width=1,stroke_fill='black')
                if method not in [None,'raw'] and kp in local:
                    p=local[kp]['methods'][method]['loo_uv']
                    if p is not None:
                        x,y=np.array(p)/2;d.ellipse((x-5,y-5,x+5,y+5),outline='#ffe75d',width=2)
                        d.text((x-20,y+5),str(kp),font=font,fill='#ffe75d',stroke_width=1,stroke_fill='black')
            title='camA manual + predicted INPUTS' if method is None else method
            draw.text((col*540+6,79),title,font=font,fill='white');sheet.paste(im,(col*540,105))
            if method not in [None,'raw']:
                v=r['methods'][method];direct=v['direct_error'];loo=v['loo_error']
                label=f"focus direct={direct:.1f}px" if direct is not None else 'focus direct=UNKNOWN'
                label+=f" LOO={loo:.1f}px" if loo is not None else ' LOO=UNKNOWN'
                draw.text((col*540+6,710),label,font=font,fill='white')
        file=f"{n:02d}_{category}_{r['session']}_{r['frame']:06d}_cam{r['camera']}.jpg"
        sheet.save(dest/file,quality=95);files.append(dict(file=file,category=category,session=r['session'],frame=r['frame'],camera=r['camera'],id=r['id'],methods=r['methods']))
    page='<html><meta charset="utf-8"><h1>底视锚点 + 多侧视候选共识：诊断结果</h1><p>绿色人工点，粉色候选，黄色真正留一视角投影。缺失候选表示不确定，非不可见。人工底视为输入，侧视人工点仅评估；不修改原标签。</p><p>约23%留一视角覆盖，不能把筛选后平均值理解为整体改善。人工/预测底视分别报告，同一批保留点对照未证明几何改善。</p>'
    page+='<pre>'+json.dumps(summary['paired'],indent=2)+'</pre>'
    page+=''.join(f'<h2>{f["file"]}</h2><img width="1500" src="{f["file"]}">' for f in files)+'</html>'
    (dest/'index.html').write_text(page);(dest/'visual_manifest.json').write_text(json.dumps(files,indent=2));shutil.copy2(src/'summary.json',dest/'summary.json')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002/multiview_consensus_v1')
    assert not export.exists();shutil.copytree(dest,export)
    write_state(latest_visual_directory=str(export),next_action='Visually inspect manual gains, harms, predicted-anchor failures, held-view failures and rejected cases; decide model-candidate ambiguity audit before any propagation.')
    print(json.dumps([dict(file=f['file'],id=f['id']) for f in files],indent=2))


if __name__=='__main__':main()
