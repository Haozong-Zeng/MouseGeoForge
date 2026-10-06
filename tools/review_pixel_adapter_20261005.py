"""Read-only metrics and actual RGB overlays for fixed400 pixel-adaptation pilot."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';PILOT=CYCLE/'pixel_adapter_matched_pilot_v1';OUT=CYCLE/'pixel_adapter_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def chart(curves):
    canvas=Image.new('RGB',(1440,950),'white');d=ImageDraw.Draw(canvas)
    colors={('head_only','fit'):'#206bc4',('head_only','calibration'):'#d24629',('head_plus_lora','fit'):'#139559',('head_plus_lora','calibration'):'#9b38a4'}
    for n,kind in enumerate(['real','single','dual','real_bottom_feet']):
        x=70+(n%2)*710;y=60+(n//2)*425;w=580;h=300
        values=[r[s][kind]['mean'] for r in curves for s in ['fit','calibration'] if r[s][kind]['mean'] is not None];upper=max(10,np.ceil(max(values)/10)*10)
        d.text((x,y-32),kind+' mean input-pixel error',font=font(21),fill='black')
        for j in range(6):
            v=upper*j/5;yy=y+h-h*j/5;d.line((x,yy,x+w,yy),fill='#dddddd');d.text((x-48,yy-8),f'{v:.0f}',font=font(15),fill='black')
        for (arm,stage),color in colors.items():
            points=[(x+r['step']/400*w,y+h-r[stage][kind]['mean']/upper*h) for r in curves if r['arm']==arm and r[stage][kind]['mean'] is not None]
            if len(points)>1:d.line(points,fill=color,width=3)
            for xx,yy in points:d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=color)
        for step in [0,100,200,300,400]:d.text((x+step/400*w-8,y+h+10),str(step),font=font(15),fill='black')
    for j,((arm,stage),color) in enumerate(colors.items()):d.text((35+j*352,905),arm+'/'+stage,font=font(17),fill=color)
    canvas.save(OUT/'localization_curves.png')
def main():
    assert json.loads((PILOT/'status.json').read_text())['stage']=='complete'
    OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    results=json.loads((PILOT/'results.json').read_text());curves=json.loads((PILOT/'curves.json').read_text());protocol=json.loads((PILOT/'protocol.json').read_text())
    data=torch.load(PILOT/'predictions.pt',map_location='cpu',weights_only=False);rows=prepare_rows();pred=data['predictions'];errors={name:((p[:,:,KP]-data['gt'])*data['pixels'][:,None,None]).norm(dim=-1) for name,p in pred.items()}
    chart(curves);chosen=[]
    for label,predicate in [('real_bottom',lambda m:m['physical_bottom']),('real_side',lambda m:m['kind']=='real' and not m['physical_bottom']),('dual',lambda m:m['kind']=='dual'),('single',lambda m:m['kind']=='single')]:
        candidates=[(float(errors['head_plus_lora'][i,q,k]-errors['head_only'][i,q,k]),i,q,k) for i,m in enumerate(data['metadata']) if m['stage']=='validation' and predicate(m) for q in range(6) for k in range(1,5) if data['valid'][i,q,k]]
        for mode,entry in [('best',min(candidates)),('worst',max(candidates))]:
            delta,i,q,k=entry;r=rows[i];m=r['metadata'];item=load_item(r['record']);vi=r['record']['views'].index(m['view']);rgb=item['batch']['images'][0,vi]
            image=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=image.size
            tile=Image.new('RGB',(1830,800),'white');draw=ImageDraw.Draw(tile)
            for ci,name in enumerate(['baseline','head_only','head_plus_lora']):
                panel=image.copy();d=ImageDraw.Draw(panel);gt=data['gt'][i,q,k];xy=pred[name][i,q,KP[k]]
                a,b=float(xy[0])*w,float(xy[1])*h;c,f=float(gt[0])*w,float(gt[1])*h
                d.line((a,b,c,f),fill='yellow',width=2);d.ellipse((a-8,b-8,a+8,b+8),outline='red',width=4);d.line((c-10,f,c+10,f),fill='#00ff66',width=4);d.line((c,f-10,c,f+10),fill='#00ff66',width=4)
                panel.thumbnail((594,650));xx=ci*610+(610-panel.width)//2;tile.paste(panel,(xx,130))
                draw.text((ci*610+10,12),name,font=font(23),fill='black');draw.text((ci*610+10,48),f'error {errors[name][i,q,k]:.2f} input px',font=font(22),fill='black')
                draw.text((ci*610+10,84),f'{label} view{m["view"]} q{q} point{KP[k]+1}',font=font(20),fill='black')
            file=f'{label}_{mode}.jpg';tile.save(OUT/file,quality=94)
            chosen.append(dict(category=label,selection=mode,delta_lora_minus_head=delta,index=i,query=q,k_index=k,point=KP[k]+1,metadata=m,errors={n:float(e[i,q,k]) for n,e in errors.items()},file=file))
    write_json(OUT/'visual_examples.json',chosen)
    page=['<!doctype html><meta charset="utf-8"><title>实际RGB骨干适配400步</title><style>body{font:17px system-ui;max-width:1800px;margin:25px auto;padding:15px}img{max-width:100%}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>',
      '<h1>实际 RGB 图像骨干适配：同图像预算400步对照</h1><p>原模型、继续训练姿态头、姿态头＋最后图像骨干层低秩适配。只有一个训练种子，尚不能宣称普遍有效或收敛。固定原模型预测框（不是GT框）；条件定位指标，不是检测AP或3D效果。所有6实例、138关键点先前向，匹配/标注仅在loss与评价使用。</p>',
      '<p>相同初始化、400步、每步2真实/1单鼠合成/1双鼠合成；AdamW初始1e-4余弦降至约1e-5。鼻尖和四脚归一化XY平方误差；缺失标签不作为不可见或错误。姿态头1,116,928可训练参数，LoRA额外32,768参数，原骨干权重冻结且哈希不变。图像更新数相同，但参数和计算成本不相同。全部两组训练完成后才统计开发验证。原标注、尾巴、腹部、正式默认模型不变。</p>',
      '<p>source-group fit/cal/开发划分继承置信度阶段；真实校验FR3，单鼠校验整段爬墙，双鼠训练源内5场景。基础姿态模型之前见过TRAIN，所以只是本次适配阶段留出。双鼠开发源视频不同但资产/场地相同。真实开发FR2/AZR2L2反复使用，不是独立测试。</p>',
      '<h2>判定：不替换原模型，不直接加长本配置</h2><p>LoRA相对只训练头的整体均值仅改善0.124px（22.048→21.924），尚无种子重复支持。相对原模型，真实31.985→34.252px、单鼠19.225→20.971px均恶化；双鼠20.425→18.629px改善。不能用双鼠占多数的整体均值掩盖真实回退。真实内部校验底脚31.130→41.723px，开发底脚43.465→45.896px（24点）。训练总前反向时间约1.92倍、峰值显存4.05→9.28GB。下一步只补冻结姿态头的LoRA-only400对照，分解头部更新与像素适配，不宣称当前方案成功。</p>',
      '<p>视觉检查：底视可见脚仍被预测在腹部，侧面仍存在足点/地面附近混淆。图例best仅指LoRA相对head-only最有利，可能两者都远差于原模型：例如底脚64.41→98.59→96.68px，侧面89.26→180.82→175.59px。它不是绝对成功样例。</p>',
      '<h2>开发验证：已知点条件误差</h2><table><tr><th>范围</th><th>方法</th><th>已知/未知</th><th>均值px↓</th><th>P95px↓</th><th>PCK32↑</th></tr>']
    for kind in ['all','real','single','dual','real_bottom_feet']:
        for name in ['baseline','head_only','head_plus_lora']:
            r=results['summary'][name]['validation'][kind]
            page.append(f'<tr><td>{kind}</td><td>{name}</td><td>{r["known"]}/{r["unknown"]}</td><td>{r["mean"]:.2f}</td><td>{r["p95"]:.2f}</td><td>{r["pck32"]:.3f}</td></tr>')
    page+=['</table><p>单位为768级网络输入像素，非原始像素/毫米。只评价原预测score≥0.5且框IoU≥0.5的有效已标点；同一固定子集。底视脚仅24个已知点，波动很大。</p>',
      '<h2>训练/内部校验曲线</h2><p>固定72图fit监视，不是全训练均值；全部220图内部校验。校验底视脚数量与开发不同。曲线没有用开发集挑epoch。</p><img src="localization_curves.png">',
      '<h2>相对头部训练的最好/最坏足点变化</h2><p>事后在各范围按LoRA减head误差选极值，不代表频率。红圈=各方法预测，绿十字=同一个人工/合成GT，仅作评价。图为显示缩放，标题误差仍为模型输入像素。</p>']
    for e in chosen:page.append(f'<h3>{e["category"]} {e["selection"]}: LoRA-head {e["delta_lora_minus_head"]:+.2f}px</h3><img src="{e["file"]}">')
    page+=['<h2>计算成本</h2><pre>'+html.escape(json.dumps(results['performance'],indent=2))+'</pre>',
      '<p>训练计时含同步的前向/反向/更新，不含初始RGB读盘和校验；总计时包含内部校验但不含初始共享读盘。单次测量，不是稳定吞吐基准。</p><p><a href="results.json">结果</a> | <a href="protocol.json">方案及固定采样计划</a> | <a href="visual_examples.json">图例索引</a></p>']
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for name in ['results.json','protocol.json','curves.json','weight_update_audit.json']:shutil.copy2(PILOT/name,OUT/name)
    for path in OUT.iterdir():
        if path.is_file():shutil.copy2(path,DEST/path.name)
    print(DEST/'index.html')
if __name__=='__main__':main()
