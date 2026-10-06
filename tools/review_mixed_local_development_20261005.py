"""Final fixed endpoint development report, including every real bottom image."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
C=ROOT/'output/singleview_fusion_cycle_20261005';RUN=C/'mixed_local_development_eval_v1';INPUT=C/'mixed_local_development_inputs_v1';OUT=C/'mixed_local_development_review_v1';DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def main():
    result=json.loads((RUN/'results.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text());rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};arms=['baseline800','native4000','rotated4000'];pred={a:read(RUN/(a+'_predictions.pt')) for a in arms};selected=protocol['visual_indices']+[i for i,r in rows.items() if r['metadata']['kind']=='real' and r['metadata']['physical_bottom'] and i not in protocol['visual_indices']];OUT.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True);colors=['#ff4422','#00bbff','#ee44dd','#99ff33','#ffbb00'];ids=[1,15,16,17,18];examples=[]
    for i in selected:
        r=rows[i];im=Image.open(ROOT/manifest[i]['image']).convert('RGB');tile=Image.new('RGB',(1600,1820),'white');draw=ImageDraw.Draw(tile)
        for slot,arm in enumerate(['eligible_GT',*arms]):
            p=im.copy();d=ImageDraw.Draw(p)
            if arm=='eligible_GT':
                for q,k in r['eligible'].nonzero().tolist():
                    x,y=(r['target'][q,k]*r['pixels']).tolist();d.line((x-6,y,x+6,y),fill=colors[k],width=3);d.line((x,y-6,x,y+6),fill=colors[k],width=3);d.text((x+7,y),str(ids[k]),font=font(15),fill=colors[k])
            else:
                a=pred[arm][i]
                for k in range(5):
                    for rank in reversed(range(6)):
                        if not a['available'][k,rank]:continue
                        x,y=(a['xy'][k,rank]*r['pixels']).tolist();radius=6 if rank==0 else 2;d.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colors[k],width=2)
                        if rank==0:d.text((x+7,y),str(ids[k]),font=font(15),fill=colors[k])
            row,col=divmod(slot,2);p.thumbnail((780,810));tile.paste(p,(col*800+(800-p.width)//2,row*880+55));draw.text((col*800+10,row*880+10),arm+f' index{i}',font=font(24),fill='black')
        label_note = 'Real: known coordinates; visibility unknown.' if r['metadata']['kind']=='real' else 'Synthetic GT: explicitly visible points only.'
        draw.text((10,1770),label_note+' Big=Top1; small=Top2-6. Colors=point type, NOT instance.',font=font(20),fill='black');name=f'{r["metadata"]["kind"]}_{i}.jpg';tile.save(OUT/name,quality=95);examples.append(dict(index=i,file=name,metadata=r['metadata'],selection='predeclared first example' if i in protocol['visual_indices'] else 'all remaining real bottom views'))
    page=['<!doctype html><meta charset="utf-8"><title>固定终点开发验证</title><style>body{font:17px system-ui;margin:28px;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}img{max-width:100%}</style><h1>144图固定终点开发验证：无后续调参</h1><p>原800步、无增强4000步、旋转增强4000步；全图768长边，前向无GT框、数量、点或数据域。48张真实图来自FR2/AZR2L2，32张合成单鼠来自WashFace，64张双鼠来自动作视频B。均未用于当前混合训练或FR3内部校准，但此前工作使用过这些开发来源，不能称为独立外部测试。</p><p>终点在验证前固定，不挑中间最优步；本轮不再根据这些结果训练。真实只评价已知坐标、可见性未知；合成主要表格用明确可见点。未知标注不是不可见或背景。真实有1张图五类点均缺失，保留图像但不计入定位误差。</p><h2>结果边界</h2><p>真实Top1平均误差52.76→35.79→32.16px；底视四足100.87→84.42→75.07px，仍不可靠。旋转组双鼠候选覆盖较好，但没有实例分配，不能当最终多鼠姿态精度。合成单鼠Top1 48.84→42.37→48.16px，没有一致增益。未晋升checkpoint，也没有更新数据集或标签。</p><table><tr><th>模型</th><th>组</th><th>已知点N</th><th>Top1误差↓</th><th>PCK16↑</th><th>最近Top6↓</th><th>一对一Top6↓</th></tr>']
    for arm in arms:
        for group,z in result['summary'][arm].items():
            a=z['eligible'];fmt=lambda x:'—' if x is None else f'{x:.3f}';page.append('<tr>'+''.join(f'<td>{v}</td>' for v in [arm,group,a['n'],fmt(a['top1_single_mean']),fmt(a['top1_single_pck16']),fmt(a['nearest_top6_mean']),fmt(a['unique_top6_mean'])])+'</tr>')
    page.append('</table><p>单位是输入像素，不是原始像素或毫米。Top6最近候选是用GT事后选最近点的乐观覆盖率；一对一也使用GT评价，并不构成部署时实例关联。仅单鼠列出Top1。固定6候选可能有同一峰周围多个点，不是六只脚。</p><h2>原图复核</h2><p>前6图为预定各组首例，另包含开发集全部8张真实底视，不按误差选择。十字为采用的GT；圆为预测。颜色ID1鼻、15左前、16右前、17左后、18右后。预测中没有标注的部位不能据此判为错误或正确。</p>')
    for z in examples:page.append(f'<h3>{z["metadata"]["kind"]} / {z["metadata"]["source"]} / index{z["index"]}</h3><img src="{z["file"]}">')
    page.append('<a href="results.json">完整分组结果</a> | <a href="protocol.json">冻结协议与哈希</a> | <a href="examples.json">图像来源</a>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');(OUT/'examples.json').write_text(json.dumps(examples,indent=2)+'\n');shutil.copy2(RUN/'results.json',OUT/'results.json');shutil.copy2(RUN/'protocol.json',OUT/'protocol.json')
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    print(json.dumps(examples,indent=2))
if __name__=='__main__':main()
