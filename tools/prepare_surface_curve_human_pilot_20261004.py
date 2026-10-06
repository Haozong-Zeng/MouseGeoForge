"""Prepare a blinded two-annotator pilot, without editing original labels."""
import argparse,collections,hashlib,html,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.surface_curve_review import parse_annotation
OUT=ROOT/'output/surface_curve_human_pilot_20261004'
SOURCE=ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json'
IMAGES=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/real_images'
SYNTHETIC=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def guide(annotator):
    return f'''<!doctype html><meta charset="utf-8"><title>背侧体表线独立试标说明</title>
<style>body{{font:18px/1.7 sans-serif;max-width:980px;margin:30px auto;padding:20px}}td,th{{padding:10px;border:1px solid #999}}table{{border-collapse:collapse}}code{{background:#eee}}</style>
<h1>背侧体表线独立试标说明</h1><p>标注者 {annotator}。这是新的曲线试标，不是修改旧 Core23 点。请独立完成再与另一位标注者复核；不要查看另一人的结果，不使用模型自动描线。先做前两组确认工具操作，再完成其余组。</p>
<p>图片已经矫正畸变，原生 1080×1200。不要裁剪、缩放保存、旋转或再次矫正。打开 images 中每组的 camX.png；旁边同名 JSON 是空白 Labelme 模板，不是已标好的结果。</p>
<h2>描什么</h2><p>只描能从图像中辨认的背侧皮肤表面中线：颈背过渡向尾根方向。它不是内部椎骨位置、身体 mask 中轴，也不必等于侧面的外轮廓。可以借助同步其他视角理解姿态，但不能在本图不可见处推断补线。</p>
<table><tr><th>类别名</th><th>形状</th><th>使用条件</th></tr>
<tr><td>dorsal_surface</td><td>linestrip 开放折线</td><td>确认是可见背侧体表中线；从头端向尾端描</td></tr>
<tr><td>dorsal_surface_uncertain</td><td>linestrip 开放折线</td><td>疑似中线但无法确认，单独存放，不进入强监督</td></tr>
<tr><td>silhouette_only</td><td>linestrip 开放折线</td><td>只能辨认剪影，不确认是共享体表中线</td></tr>
<tr><td>neck_surface_endpoint</td><td>point</td><td>真正颈背过渡端点可见且可确认时才点；不是内部颈骨05</td></tr>
<tr><td>tail_root_surface_endpoint</td><td>point</td><td>真正尾根背侧表面端点可见且可确认时才点；不猜皮下骨点</td></tr></table>
<p>仅一只鼠，group_id 留空或填 0。每个遮挡断点两侧画成两条独立折线，不跨过不可见区；不要画闭合多边形。曲线端点缺失就留空，不把局部片段边缘假装成身体端点。不要给本次图片补旧编号06或07，不要求它们位于任何等分位置。</p>
<h2>可见状态与完成标志</h2><p>每张检查结束后将图像 flags 中 reviewed 勾为 true，否则当作未完成。完全没有可确认可见中线时，勾 no_visible_dorsal_surface；若只能辨认剪影，画 silhouette_only 并勾 only_silhouette_identifiable。既无中线也无可靠剪影时可以没有 shapes，但仍须明确勾选无可见中线。模板空白不等于不可见。</p>
<p>不确定端点宁可不标。若软件允许 shape flags，可设置 uncertain=true；它不会用作多视角端点锚点。底视通常看不见背侧中线，不能把腹部中线当背部；仍应实际检查，不按相机名批量判定。</p>
<h2>参考图</h2><p>references/synthetic 有三姿态的七视角。绿色表示网格几何判定可见的背侧线，灰点表示隐藏的投影，只用于讲解，不能照抄隐藏部分。即使网格可见，毛发和低对比度仍可能让真实图无法辨认；实际不确定时如实标 uncertain 或不标。仅看得到剪影不代表看到了同一物质线。</p>
<p>合成线两端是模板的虚拟颈背和尾根水平表面点，并非可直接辨认的骨骼点。不要把参考图中绿线的终止位置机械地搬到真实鼠；真实端点只有在可确认其定义和位置时才标，不能确认就不标。可见片段仍可在没有端点标记的情况下提交。</p>
<p><a href="references/index.html">查看合成定义参考</a> · <a href="overview.html">查看本批同步组概览</a></p>
<h2>交回什么</h2><p>保持 images 下组名、camX.png 和 camX.json 文件名不变。返回本文件夹或 images 中全部 JSON，注明标注者 {annotator}；无需重新压缩或修改原始 PNG。不要混入另一人的结果。复核时才同时查看两套折线。</p>
<p>我们会检查坐标、可见片段、双人差异及可靠端点的留一视角重投影。不因曲线平滑就认定正确，不把各相机独立等弧长位置直接三角化。曲线含义无法统一也属于有效试标结果，请记录在 notes.txt 中。</p>
<p>格式依据：<a href="https://github.com/wkentaro/labelme/blob/main/examples/primitives/primitives.json">Labelme 官方开放折线与点格式</a>。不需要安装新插件或上传图片。</p>'''


def preview():
    assert not OUT.exists();OUT.mkdir()
    manifest=json.loads(SOURCE.read_text());eligible={(g['session'],g['frame']):g for g in manifest['frame_groups'] if g['eligible_for_development']}
    source=torch.load(ROOT/'output/surface_curve_cycle_20261003/expanded_real_point_features_v1/real_inputs.pt',map_location='cpu',weights_only=False)
    val=sorted({(m['session'],m['frame']) for m in source['metadata'] if m['role']=='validation'})
    buckets=collections.defaultdict(list)
    for key,g in eligible.items():
        if g['role']=='train':buckets[g['session']].append(key)
    selected=[sorted(v)[len(v)//2] for k,v in sorted(buckets.items())]
    selected.append(sorted(buckets['FR3'])[0]);assert len(selected)==8
    selected+=val;assert len(selected)==16 and len(set(selected))==16
    rows=[]
    for n,key in enumerate(selected,1):
        g=eligible[key];assert g['session'] not in ('GR2','UR2','UR3')
        rows.append(dict(id=f'C{n:02d}_{g["session"]}_f{g["frame"]:06d}',**g,
                         selection='TRAIN session-stratified midpoint plus FR3 earliest' if g['role']=='train' else 'existing frozen8group validation including known ambiguous and ordinary cases'))
    save(OUT/'selection.json',dict(manifest_sha256=sha(SOURCE),groups=rows,policy='No excluded/locked groups. Train/validation roles preserved. Not random prevalence sample. Human packets omit model and old keypoint overlays.'))
    for page in range(4):
        canvas=Image.new('RGB',(864,540),'#17171b');draw=ImageDraw.Draw(canvas)
        for slot,g in enumerate(rows[page*4:page*4+4]):
            x=(slot%2)*432;y=(slot//2)*270;draw.text((x+4,y+3),g['id']+' '+g['role'],fill='white')
            cameras=[c for c in ['F','G'] if c in g['cameras']]
            if len(cameras)<2:cameras=g['cameras'][-2:]
            for j,c in enumerate(cameras):
                with Image.open(IMAGES/g['split']/g['session']/f'frame_{g["frame"]:06d}'/f'cam{c}.png') as im:
                    im=im.convert('RGB');im.thumbnail((216,240));canvas.paste(im,(x+j*216,y+27))
                draw.text((x+j*216+3,y+15),'cam'+c,fill='#ffde55')
        canvas.save(OUT/f'selection_{page+1}.jpg',quality=95)
    print(json.dumps(dict(groups=len(rows),images=sum(len(g['cameras']) for g in rows),selection=rows),indent=2))


def package():
    selection=json.loads((OUT/'selection.json').read_text());assert (OUT/'selection_visual_qa.json').exists()
    assert sha(SOURCE)==selection['manifest_sha256']
    for annotator in ('A','B'):
        kit=OUT/f'Annotator_{annotator}';assert not kit.exists();kit.mkdir();files=[];overview=[]
        for g in selection['groups']:
            overview.append(f'<h2>{g["id"]}</h2><div>')
            for camera in g['cameras']:
                name=f'cam{camera}.png';src=IMAGES/g['split']/g['session']/f'frame_{g["frame"]:06d}'/name
                rel=Path('images')/g['id']/name;dst=kit/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                with Image.open(dst) as im:w,h=im.size
                assert (w,h)==(1080,1200) and sha(src)==sha(dst)
                a=dict(version='5.0.1',flags=dict(reviewed=False,no_visible_dorsal_surface=False,only_silhouette_identifiable=False),
                       shapes=[],imagePath=name,imageData=None,imageHeight=h,imageWidth=w)
                save(dst.with_suffix('.json'),a);assert parse_annotation(a,width=w,height=h,image_name=name)['valid']
                files.append(dict(path=rel.as_posix(),annotation=rel.with_suffix('.json').as_posix(),sha256=sha(dst),source=str(src.relative_to(ROOT)),
                                  group=g['id'],session=g['session'],frame=g['frame'],camera=camera,role=g['role'],width=w,height=h))
                overview.append(f'<a href="{rel.as_posix()}"><img width="180" src="{rel.as_posix()}">cam{camera}</a>')
            overview.append('</div>')
        references=[]
        for frame in (1,2,3):
            scene=SYNTHETIC/f'frame_{frame:06d}';z=np.load(scene/'surface_curve.npz')
            for v in range(7):
                with Image.open(scene/'blended_images'/f'{v:08d}.jpg') as im:im=im.convert('RGB')
                draw=ImageDraw.Draw(im);uv=z['uv'][v,0];visible=z['visible'][v,0]
                for j in range(len(uv)-1):
                    if visible[j] and visible[j+1]:draw.line([tuple(uv[j]),tuple(uv[j+1])],fill='#35ff55',width=5)
                for j,p in enumerate(uv):
                    if not visible[j]:x,y=p;draw.ellipse((x-2,y-2,x+2,y+2),fill='#bbbbbb')
                draw.rectangle((0,0,1080,40),fill='#14141c');draw.text((10,12),'GREEN = mesh-visible surface; GREY dots = hidden, do NOT annotate hidden line.',fill='white')
                rel=Path('references/synthetic')/f'pose{frame}_cam{chr(65+v)}.jpg';(kit/rel).parent.mkdir(parents=True,exist_ok=True);im.save(kit/rel,quality=95)
                references.append(f'<h2>Pose {frame} cam{chr(65+v)}</h2><img width="540" src="synthetic/{rel.name}">')
        (kit/'references/index.html').write_text('<!doctype html><meta charset="utf-8"><h1>合成体表线定义参考</h1><p>绿色是网格可见线，灰点是隐藏投影。隐藏段不描；网格可见也不保证RGB足够清晰。</p>'+''.join(references),encoding='utf-8')
        (kit/'overview.html').write_text('<!doctype html><meta charset="utf-8"><h1>同步视角概览</h1><p>点击图片看原始分辨率，不在缩略图标注。</p>'+''.join(overview),encoding='utf-8')
        (kit/'START_HERE.html').write_text(guide(annotator),encoding='utf-8')
        save(kit/'manifest.json',dict(schema='visible_dorsal_surface_fragments_v1',annotator=annotator,images=len(files),groups=16,files=files,
             note='Blank templates pending human work, no predictions or previous keypoint overlays. Preserveoriginalroles; no pseudoGT.',source_manifest_sha256=sha(SOURCE)))
        zip_path=OUT/f'PyMouseStudio-SurfaceCurve-Pilot16-Annotator{annotator}.zip'
        with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in sorted(kit.rglob('*')):
                if path.is_file():z.write(path,Path(kit.name)/path.relative_to(kit))
        with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
        assert zip_path.stat().st_size<1_000_000_000
        save(OUT/f'package_{annotator}.json',dict(path=str(zip_path.relative_to(ROOT)),bytes=zip_path.stat().st_size,sha256=sha(zip_path),images=len(files),groups=16,zip_crc_passed=True))
    save(OUT/'state.json',dict(stage='awaiting_two_independent_human_annotations',training_started=False,automation_started=False,
        next='Receive A/B annotations, run audit_surface_curve_human_returns_20261004.py; no real curve training until reviewed. Do not alter original labels or locked holdout.'))
    print('Packaged A/B, pending genuine human labels.')


def refresh_guides():
    # Only regenerate our untouched blank kits, never returned human annotation.
    for annotator in ('A','B'):
        kit=OUT/f'Annotator_{annotator}';record=json.loads((OUT/f'package_{annotator}.json').read_text())
        zip_path=ROOT/record['path'];assert sha(zip_path)==record['sha256']
        manifest=json.loads((kit/'manifest.json').read_text())
        for spec in manifest['files']:
            a=json.loads((kit/spec['annotation']).read_text());assert not a['shapes'] and not a['flags']['reviewed']
            assert sha(kit/spec['path'])==spec['sha256']
        (kit/'START_HERE.html').write_text(guide(annotator),encoding='utf-8')
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted(kit.rglob('*')):
                if p.is_file():z.write(p,Path(kit.name)/p.relative_to(kit))
        with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
        record.update(bytes=zip_path.stat().st_size,sha256=sha(zip_path));save(OUT/f'package_{annotator}.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--package',action='store_true');group.add_argument('--refresh-guides',action='store_true');args=parser.parse_args()
    refresh_guides() if args.refresh_guides else package() if args.package else preview()
