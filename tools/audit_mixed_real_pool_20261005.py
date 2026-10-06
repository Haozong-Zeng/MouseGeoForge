"""Post-hoc paired real CAL diagnostics; no inference, label edits or selection for training."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';RUN=CYCLE/'mixed_real_pool_control_v1';INPUT=CYCLE/'mixed_local_rgb_inputs_v1';OUT=CYCLE/'mixed_real_pool_breakdown_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
IDS=[1,15,16,17,18];COLORS=['#ff4422','#00bbff','#ee44dd','#99ff33','#ffbb00']
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),pck8=float((a<=8).mean()),pck16=float((a<=16).mean()),pck32=float((a<=32).mean()),over64=int((a>64).sum())) if len(a) else dict(n=0)
def main():
    global RUN,OUT,DEST
    rotation4000='--rotation4000' in sys.argv
    rotation='--rotation' in sys.argv or rotation4000
    control_name='native4000' if rotation4000 else ('native1600' if rotation else 'same1600');treatment_name='rotated4000' if rotation4000 else ('rotated1600' if rotation else 'expanded1600')
    if rotation:
        RUN=CYCLE/'mixed_rotation_control_v1';OUT=CYCLE/'mixed_rotation_breakdown_v1';DEST=DEST.parent/OUT.name
    if rotation4000:
        RUN=CYCLE/'mixed_rotation4000_v1';OUT=CYCLE/'mixed_rotation4000_breakdown_v1';DEST=DEST.parent/OUT.name
    assert (RUN/'results.json').exists();OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt') if r['metadata']['kind']=='real' and r['metadata']['stage']=='calibration'};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    preds={'baseline800':read(CYCLE/'mixed_local_backbone800_v1/train_official_backbone_predictions.pt'),control_name:read(RUN/('native_rgb_predictions.pt' if rotation else 'same_pool_predictions.pt')),treatment_name:read(RUN/('quarter_turn_predictions.pt' if rotation else 'expanded_real_predictions.pt'))};points=[]
    for arm,pp in preds.items():
        for i,r in rows.items():
            assert int(r['instance_valid'].sum())==1
            p=pp[i];xy=p['xy']*r['pixels'];gt=r['target']*r['pixels']
            for q,k in r['eligible'].nonzero().tolist():
                dist=torch.linalg.vector_norm(xy[k]-gt[q,k],dim=-1);dist=dist.masked_fill(~p['available'][k],float('inf'));opposite={1:2,2:1,3:4,4:3}.get(k);swap=None
                if opposite is not None and r['eligible'][q,opposite] and float(torch.linalg.vector_norm(gt[q,k]-gt[q,opposite]))>10:
                    other=float(torch.linalg.vector_norm(xy[k,0]-gt[q,opposite]));swap=bool(other<=32 and other+5<float(dist[0]))
                points.append(dict(arm=arm,index=i,scene=str(r['metadata'].get('scene',i)),bottom=bool(r['metadata']['physical_bottom']),point=IDS[k],top1=float(dist[0]),nearest6=float(dist.min()),opposite_closer_flag=swap))
    summary={}
    for arm in preds:
        summary[arm]={}
        for view,b in [('all',None),('bottom',True),('other',False)]:
            summary[arm][view]={}
            for name,k in [('all',None),*[(str(k),k) for k in IDS]]:
                rr=[p for p in points if p['arm']==arm and (b is None or p['bottom']==b) and (k is None or p['point']==k)];flags=[p['opposite_closer_flag'] for p in rr if p['opposite_closer_flag'] is not None]
                summary[arm][view][name]=dict(top1=stats([p['top1'] for p in rr]),nearest6=stats([p['nearest6'] for p in rr]),opposite_closer=dict(n=len(flags),flagged=sum(flags)))
    deltas=[]
    for i,r in rows.items():
        vals={arm:np.mean([p['top1'] for p in points if p['arm']==arm and p['index']==i]) for arm in preds};deltas.append(dict(index=i,bottom=bool(r['metadata']['physical_bottom']),scene=str(r['metadata'].get('scene',i)),means=vals,delta=float(vals[treatment_name]-vals[control_name])))
    # Paired scene resampling is descriptive only: internal CAL has repeatedly informed this cycle.
    clusters={s:[p for p in points if p['scene']==s] for s in {p['scene'] for p in points}};rng=np.random.default_rng(17);boot=[];keys=sorted(clusters)
    for _ in range(1000):
        rr=[p for s in rng.choice(keys,len(keys),replace=True) for p in clusters[s]];boot.append(np.mean([p['top1'] for p in rr if p['arm']==treatment_name])-np.mean([p['top1'] for p in rr if p['arm']==control_name]))
    result=dict(summary=summary,points=points,image_deltas=deltas,descriptive_paired_scene_bootstrap=dict(n_scenes=len(keys),resamples=1000,seed=17,delta_ci95=np.percentile(boot,[2.5,97.5]).tolist()),caveats='Input pixels; known coordinates not visibility labels. Opposite-foot flag is proximity diagnostic, not proof of anatomical swap. Top6 optimistic candidate coverage, not final pose. Reused internal CAL; no independent significance claim.')
    result['arm_definitions']={'baseline800':'original mixed800',control_name:'native_rgb' if rotation else 'same_pool',treatment_name:'quarter_turn' if rotation else 'expanded_real'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');selected=[]
    for b in [True,False]:
        dd=sorted([z for z in deltas if z['bottom']==b],key=lambda z:z['delta'])
        for label,z in [('most_improved',dd[0]),('most_regressed',dd[-1])]:
            i=z['index'];r=rows[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');tile=Image.new('RGB',(1520,1800),'white');d=ImageDraw.Draw(tile)
            for slot,arm in enumerate(['known_GT',*preds]):
                p=im.copy();draw=ImageDraw.Draw(p)
                if arm=='known_GT':
                    for q,k in r['eligible'].nonzero().tolist():
                        x,y=(r['target'][q,k]*r['pixels']).tolist();draw.line((x-7,y,x+7,y),fill=COLORS[k],width=3);draw.line((x,y-7,x,y+7),fill=COLORS[k],width=3);draw.text((x+8,y),str(IDS[k]),fill=COLORS[k],font=font(15))
                else:
                    a=preds[arm][i]
                    for k in range(5):
                        for rank in reversed(range(6)):
                            if not a['available'][k,rank]:continue
                            x,y=(a['xy'][k,rank]*r['pixels']).tolist();radius=7 if rank==0 else 2;draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=COLORS[k],width=2)
                            if rank==0:draw.text((x+8,y),str(IDS[k]),fill=COLORS[k],font=font(15))
                row,col=divmod(slot,2);p.thumbnail((744,800));tile.paste(p,(col*760+(760-p.width)//2,row*890+60));d.text((col*760+10,row*890+10),result['arm_definitions'].get(arm,arm)+f' index{i}',fill='black',font=font(24))
            name=f'{"bottom" if b else "other"}_{label}_{i}.jpg';tile.save(OUT/name,quality=95);selected.append(dict(**z,file=name,selection=label))
    page=['<!doctype html><meta charset="utf-8"><title>真实足部拆分诊断</title><style>body{font:17px system-ui;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%}</style><h1>真实内部校准：底视与左右足</h1><p>同32图131个已知坐标；768长边输入像素。大圆Top1，小圆其余固定候选；颜色表示部位而非实例。这里没有新增GT或修改标签。</p><table><tr><th>方法</th><th>视角</th><th>点ID</th><th>N</th><th>Top1均值</th><th>中位数</th><th>PCK16</th><th>最近Top6</th><th>对侧更近/可检查</th></tr>']
    if rotation:page.insert(1,f'<p>{control_name}＝未增强，{treatment_name}＝旋转增强，并非扩大数据池。两组共享原800步祖先与配对样本、学习率和训练预算；每条轨迹保留自己的优化器状态。图像选择按旋转减未增强的误差差值。</p>')
    for arm,views in summary.items():
        for view,kk in views.items():
            for k,z in kk.items():
                a=z['top1'];f=z['opposite_closer'];page.append('<tr>'+''.join(f'<td>{v}</td>' for v in [arm,view,k,a['n'],f"{a.get('mean',0):.2f}",f"{a.get('median',0):.2f}",f"{a.get('pck16',0):.1%}",f"{z['nearest6'].get('mean',0):.2f}",f"{f['flagged']}/{f['n']}"])+'</tr>')
    page.append('</table><p>对侧更近：双方标注已知且间距&gt;10px，预测距对侧≤32px且比距本侧至少近5px；只作为疑似左右混淆诊断。图例ID1鼻尖、15左前、16右前、17左后、18右后。下面按实验组减对照组的误差挑选最大改善/退化，因此不代表平均效果。校准集反复使用，不作外部泛化宣称。</p>')
    for z in selected:page.append(f"<h2>{z['selection']} / {'bottom' if z['bottom'] else 'other'} / Δ{z['delta']:.2f}px</h2><img src='{z['file']}'>")
    page.append('<a href="results.json">完整逐点结果</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');(OUT/'selected.json').write_text(json.dumps(selected,indent=2))
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps({a:{v:z['all'] for v,z in vv.items()} for a,vv in summary.items()},indent=2));print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
