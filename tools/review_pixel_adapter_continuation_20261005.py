"""Report endpoint of bounded continuation, with fixed pre-existing visual cases."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';BASE=CYCLE/'pixel_adapter_matched_pilot_v1';SOLO=CYCLE/'pixel_adapter_only_pilot_v1';CONT=CYCLE/'pixel_adapter_only_continuation_v1'
OUT=CYCLE/'pixel_adapter_continuation_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    assert json.loads((CONT/'status.json').read_text())['stage']=='complete';OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    results=json.loads((CONT/'results.json').read_text());summary={'baseline':json.loads((BASE/'results.json').read_text())['summary']['baseline'],'step400':json.loads((SOLO/'results.json').read_text())['summary'],'step1600':results['summary']}
    curves=json.loads((SOLO/'curves.json').read_text())+json.loads((CONT/'curves.json').read_text())[1:]
    canvas=Image.new('RGB',(1440,950),'white');d=ImageDraw.Draw(canvas)
    for n,kind in enumerate(['real','single','dual','real_bottom_feet']):
        x=70+n%2*710;y=65+n//2*425;w=580;h=300
        upper=max(10,np.ceil(max(r[p][kind]['mean'] for r in curves for p in ['fit','calibration'])/10)*10)
        d.text((x,y-33),kind+' mean input px',font=font(21),fill='black')
        for j in range(6):
            yy=y+h-h*j/5;d.line((x,yy,x+w,yy),fill='#ddd');d.text((x-48,yy-8),f'{upper*j/5:.0f}',font=font(15),fill='black')
        for phase,color in [('fit','#266abf'),('calibration','#bd337e')]:
            pts=[(x+r['step']/1600*w,y+h-r[phase][kind]['mean']/upper*h) for r in curves];d.line(pts,fill=color,width=3)
            for xx,yy in pts:d.ellipse((xx-3,yy-3,xx+3,yy+3),fill=color)
        yy=y+h-summary['baseline']['calibration'][kind]['mean']/upper*h;d.line((x,yy,x+w,yy),fill='#222',width=1)
        for step in [0,400,800,1200,1600]:d.text((x+step/1600*w-10,y+h+10),str(step),font=font(15),fill='black')
    d.text((35,905),'Blue: fixed72 TRAIN monitor | Purple:220 internal calibration | Black: original calibration',font=font(21),fill='black');canvas.save(OUT/'curves.png')
    data=read(BASE/'predictions.pt');pred={'baseline':data['predictions']['baseline'],'step400':read(SOLO/'predictions.pt')['predictions'],'step1600':read(CONT/'predictions.pt')['predictions']};rows=prepare_rows()
    chosen=json.loads((CYCLE/'pixel_adapter_four_arm_review_v1/visual_examples.json').read_text())
    for e in chosen:
        i=e['index'];q=e['query'];kp=e['point']-1;k=KP.index(kp);r=rows[i];m=r['metadata'];item=load_item(r['record']);vi=r['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
        image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=image.size
        tile=Image.new('RGB',(1830,800),'white');td=ImageDraw.Draw(tile);e['errors']={}
        for ci,(name,p) in enumerate(pred.items()):
            panel=image.copy();di=ImageDraw.Draw(panel);a,b=(p[i,q,kp]*torch.tensor([w,h])).tolist();c,f=(data['gt'][i,q,k]*torch.tensor([w,h])).tolist()
            err=float(((p[i,q,kp]-data['gt'][i,q,k])*data['pixels'][i]).norm());e['errors'][name]=err
            di.line((a,b,c,f),fill='yellow',width=2);di.ellipse((a-8,b-8,a+8,b+8),outline='red',width=4);di.line((c-10,f,c+10,f),fill='#00ff66',width=4);di.line((c,f-10,c,f+10),fill='#00ff66',width=4)
            panel.thumbnail((594,650));tile.paste(panel,(ci*610+(610-panel.width)//2,130));td.text((ci*610+12,12),name,font=font(24),fill='black');td.text((ci*610+12,48),f'error {err:.2f} input px',font=font(22),fill='black');td.text((ci*610+12,83),f'{e["category"]} view{m["view"]} q{q} point{kp+1}',font=font(20),fill='black')
        tile.save(OUT/e['file'],quality=94)
    write_json(OUT/'visual_examples.json',chosen);write_json(OUT/'results.json',dict(summary=summary,stop_reason=results['stop_reason'],step=results['step'],incremental_train_seconds=results['incremental_train_seconds'],peak_gpu_bytes=results['peak_gpu_bytes']))
    page=['<!doctype html><meta charset="utf-8"><title>图像适配1600步收敛检查</title><style>body{font:17px system-ui;max-width:1800px;margin:25px auto;padding:15px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
      '<h1>400→1600步：收益很小，停止继续堆步数</h1><p>冻结姿态头/原骨干，只优化末层图像LoRA；从400恢复AdamW，用1e-5低学习率续训1200步。未触发内部校验早停，到预定1600上限停止，原冻结权重哈希不变。不是与400步模型同计算预算的架构比较。</p>',
      '<p>开发整体均值22.267（原）→22.272（400）→22.173（1600），主要来自双鼠小幅改善；真实31.985→32.124→32.232、单鼠19.225→19.307→19.329仍稍差。底脚43.465→42.332→42.707，PCK32仍为0.5，且只有24已知点。没有整体可靠的真实改善，不晋升、不继续自动延长这个配置。</p>',
      '<p>内部校验整体24.790→24.831→24.675；最后1200步仅改善0.157px。固定fit监视21.209→20.981。底视校验31.130→30.354→30.274，改善已很小。单种子，重复使用的开发验证，不是外部测试。标签/源数据不变。</p>']
    for phase in ['calibration','validation']:
        page.append(f'<h2>{phase}</h2><table><tr><th>范围</th><th>方法</th><th>known</th><th>mean↓</th><th>P95↓</th><th>PCK32↑</th></tr>')
        for kind in ['all','real','single','dual','real_bottom_feet']:
            for name,s in summary.items():
                v=s[phase][kind];page.append(f'<tr><td>{kind}</td><td>{name}</td><td>{v["known"]}</td><td>{v["mean"]:.3f}</td><td>{v["p95"]:.3f}</td><td>{v["pck32"]:.3f}</td></tr>')
        page.append('</table>')
    page+=['<h2>曲线</h2><p>单位为模型输入像素；没有根据开发集选择中间checkpoint。</p><img src="curves.png">',
      '<h2>固定在400步时选择的案例</h2><p>沿用上一报告8个极值案例，未为了1600效果重新挑选。红圈预测、绿十字评价GT。不是发生频率估计。</p>']
    for e in chosen:page.append(f'<h3>{e["category"]} {e["selection"]} at400</h3><img src="{e["file"]}">')
    page+=['<h2>转向数据覆盖检查</h2><p>当前合成fit不是完全没有底视：Cylinder的21个姿态提供21张底视＋21张侧视，另22个Homecage姿态提供44图。对应TRAIN源已有200帧，圆筒每帧8视角、Homecage每帧4视角。下一项可先固定这些原43个姿态，只扩大训练视角，保持真实/双鼠和采样预算，隔离数据视角覆盖因素。不能把相邻帧或新角度当作新独立视频源。</p>',
      '<p>写实SurfaceCurve的21图保留排除：至少climb姿态与校验climb来源有关，其他111/216帧的原动作分段尚需细查；外观/位姿种子明确用过FR2/AZR2L2开发参考。不能直接恢复到训练并声称独立。此处只审计、不移动或修改数据。</p>',
      '<p><a href="results.json">结果</a> | <a href="visual_examples.json">图例索引</a> | <a href="coverage.json">物理相机与源覆盖审计</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');shutil.copy2(CYCLE/'pixel_training_coverage_audit_v1/coverage.json',OUT/'coverage.json')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(DEST/'index.html')
if __name__=='__main__':main()

