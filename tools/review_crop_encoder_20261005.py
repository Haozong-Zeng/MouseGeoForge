"""Show actual auxiliary crops and separate RGB/gray readouts."""
import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'predicted_crop_encoder_v1';OUT=CYCLE/'predicted_crop_affinity_diagnostic_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name


def main():
    r=json.loads((OUT/'results.json').read_text());runtime=json.loads((SOURCE/'results.json').read_text());canvas=Image.new('RGB',(4*224,2*254),'white');draw=ImageDraw.Draw(canvas)
    for view in range(4):
        for row,mode in enumerate(['rgb','gray']):
            path=SOURCE/f'crops/validation_view{view}_q4_{mode}.png';im=Image.open(path).convert('RGB');canvas.paste(im,(view*224,row*254+30));draw.text((view*224+6,row*254+8),f'view{view} q4 {mode}',fill='black')
    canvas.save(OUT/'actual_crop_inputs.jpg',quality=95)
    page=['<!doctype html><meta charset="utf-8"><title>预测框裁剪编码</title><style>body{font:17px system-ui;margin:30px}img{max-width:100%}td,th{padding:8px;border:1px solid #aaa}</style><h1>先裁剪后编码：query偏向减轻，但仍不足以可靠关联</h1><p>148张既有图像，296个预测框，RGB与灰度共592个主要裁剪输入；另有5个输入用于重复/单张一致性检查，共150次批次前向。只运行已有Omega内的冻结图像骨干。主检测/姿态分辨率及结果不变；辅助裁剪224×224，保持宽高比，固定均值填充。没有GT框、GT数量或训练。</p><table><tr><th>划分</th><th>固定读出</th><th>身份AUC</th><th>同query AUC</th><th>互选正确/已知</th><th>未知/全部</th></tr>']
    for role,rows in r['results'].items():
        for name,s in rows.items():
            m=s['mutual'];page.append(f'<tr><td>{role}</td><td>{name}</td><td>{s["same_identity_auc"]:.3f}</td><td>{s["same_query_auc"]:.3f}</td><td>{m["correct"]}/{m["known"]}</td><td>{m["unknown"]}/{m["proposed"]}</td></tr>')
    page.append('</table><p>RGB CLS在开发集已知互选中55/79正确，但41/120条互选身份未知；灰度CLS为42/74，另有35/109未知。全配对AUC仍只有0.559与0.524，不能把有限互选收益当成可靠身份验证。相比整图ROI池化，裁剪编码减轻同query偏向；不证明所有差异都由上下文造成。</p><p>灰度只去掉色度，不去掉毛色亮度差异，因此不是同毛色双鼠实验。训练/开发仍来自同一合成项目的不同帧，不是外部视频。</p><h2>实际送入骨干的首个开发场景裁剪</h2><p>各相机q4未必是同一只鼠，部分框仍含两只鼠。下图就是编码输入（RGB/灰度、均值填充）；没有用标签净化裁剪。</p><img src="actual_crop_inputs.jpg"><h2>计算与数值</h2>')
    compute=dict(wall_seconds=runtime['elapsed_seconds'],main_crop_examples=runtime['crop_examples'],extra_repeat_crop_examples=5,encoder_batch_calls=runtime['encoder_batch_calls'],tests=runtime['tests'])
    page.append('<pre>'+json.dumps(compute,indent=2)+'</pre><p>批次逆序完全一致；单张与批处理存在bf16数值差异（最大0.00805，余弦≥0.999985），已如实保留，不宣称严格逐元素相同。</p><p>下一步停止原始余弦特征替换：可做一个仅TRAIN监督的低维关联投影对照，预留TRAIN内校准场景，RGB/灰度同初始化与预算；开发集不用于梯度或选步数。暂不融合、不晋升模型。</p><p><a href="results.json">完整关联指标</a> | <a href="compute.json">计算记录</a></p>')
    (OUT/'compute.json').write_text(json.dumps(compute,indent=2),encoding='utf-8');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(exist_ok=True)
    for name in ['index.html','results.json','compute.json','actual_crop_inputs.jpg']:shutil.copy2(OUT/name,DEST/name)
    print(DEST/'index.html')


if __name__=='__main__':main()
