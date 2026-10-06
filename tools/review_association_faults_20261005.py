"""Show a predeclared-mode counterexample; do not present injected faults as real failures."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'dual_allscene_association_audit_v1';OUT=CYCLE/'association_logit_fault_audit_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
KP=[0,14,15,16,17]


def main():
    r=json.loads((OUT/'results.json').read_text());groups=torch.load(SOURCE/'groups.pt',map_location='cpu',weights_only=False);faults=torch.load(OUT/'predictions.pt',map_location='cpu',weights_only=False)
    # First mixed scene for each fixed intervention, selected posthoc for illustration only.
    examples=[next(x for x in r['scenes'] if x['fault']==mode and x['mixed_identity_pairs']>0) for mode in r['summary']];gallery=[]
    for example in examples:
        gi=example['group'];mode=example['fault'];g=groups[gi];f=next(x for x in faults if x['group']==gi and x['fault']==mode);p=g['predictions']
        for vi,view in enumerate(g['record']['views']):
            rgb=Image.open(SOURCE/g['images'][vi]).convert('RGB');w,h=rgb.size;im=Image.new('RGB',(3*w,h+55),'white');draw=ImageDraw.Draw(im)
            keep=f['keep_queries'][vi];identity=example['identities'][vi]['reliable_identity']
            for col,arm in enumerate(['baseline','local','composite']):
                im.paste(rgb,(col*w,55));draw.text((col*w+8,8),f'{arm} | keep q{keep} | posthoc identity {identity}',fill='black');draw.text((col*w+8,28),f'FAULT INJECTION: {mode} | view {view}',fill='black')
                for q in np.flatnonzero(p['objectness_logits'][vi].sigmoid().numpy()>=.5):
                    retained=q==keep;color=('#00ee99' if q==4 else '#ee00cc') if retained else '#999999'
                    cx,cy,bw,bh=p['boxes'][vi,q].numpy()*[w,h,w,h];draw.rectangle((col*w+cx-bw/2,55+cy-bh/2,col*w+cx+bw/2,55+cy+bh/2),outline=color,width=3 if retained else 1)
                    draw.text((col*w+cx-bw/2,55+cy-bh/2),f'q{q} '+('kept' if retained else 'suppressed'),fill=color)
                    if retained:
                        for k in KP:
                            x,y=f['outputs'][arm][vi,q,k].numpy()*[w,h]+[col*w,55]
                            draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+4,y),str(k+1),fill=color)
            name=f'{mode}_g{gi:02d}_view{view}.jpg';im.save(OUT/name,quality=92);gallery.append(name)
    page=['<!doctype html><meta charset="utf-8"><title>跨鼠融合故障注入</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>故障注入：唯一预测不等于同一只鼠</h1><p>这不是自然漏检率，也不是遮挡图片测试。原RGB/框/概率/关键点不变，仅在复制的模型输出上保留一个原有高置信度query，其他objectness logit设为-20。选择规则只使用预测，不使用GT身份或数量；GT框关联仅在推理后解释对应动物。</p><table><tr><th>固定规则</th><th>有混合身份的场景/16</th><th>不同身份/可靠视角对</th><th>融合可用点</th><th>相对局部读出改变点</th></tr>']
    for mode,s in r['summary'].items():page.append(f'<tr><td>{mode}</td><td>{s["scenes_with_mixed_reliable_identities"]}</td><td>{s["mixed_identity_pairs"]}/{s["reliable_identity_pairs"]}</td><td>{s["fusion_eligible"]}</td><td>{s["changed_vs_local"]}</td></tr>')
    page.append('</table><p>可靠身份诊断要求事后框关联IoU≥0.5；不可靠视角记为未知。固定原始关联与修改logit后重新关联在本次完全一致。每种规则320个可用点都被门控放行，说明当前门控只确认视角/检测可用，并没有验证同鼠身份。不能把单鼠试验直接部署到未知多鼠。</p><p>首个场景的highest_confidence模式：四个视角都保留q4，事后动物身份依次1、0、1、0。query编号显然不是跨相机身份。灰框是人为压低置信度的原始预测，实际图像中鼠并没有消失。下面展示每种规则的第一个混合身份反例，非随机准确率样本。</p><p>无训练、无GPU编码、原始缓存哈希未变。正式模型默认关闭，未更新数据标注。</p><p><a href="results.json">完整指标/固定关联与操作关联</a> | <a href="protocol.json">协议</a></p>')
    for name in gallery:page.append(f'<h2>{name}</h2><img src="{name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in gallery+['index.html','results.json','protocol.json']:shutil.copy2(OUT/name,DEST/name)
    print(DEST/'index.html')


if __name__=='__main__':main()
