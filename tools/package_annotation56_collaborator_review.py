"""Curated, offline visualization-only handoff; preserve all returned coordinates."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/annotation_review_20260914_real_images_56'
NAME='PyMouseStudio-56-Annotation-CollaboratorReview-v1'
PACKAGE=ROOT/'output/annotation_packages'/NAME
DOWNLOAD=Path('/mnt/c/Users/55395/Downloads')
FRAMES=[0,64,129,194,259,324,389,454]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archive=PACKAGE.with_suffix('.zip');delivery=DOWNLOAD/archive.name
    assert not PACKAGE.exists() and not archive.exists() and not delivery.exists(),'Never overwrite a delivery.'
    PACKAGE.mkdir(parents=True)
    for frame in FRAMES:
        stem=f'frame_{frame:06d}';(PACKAGE/stem).mkdir()
        shutil.copy2(SOURCE/(stem+'_overview.jpg'),PACKAGE/(stem+'_overview.jpg'))
        for cam in 'ABCDEFG':
            for suffix in ('detail.jpg','review.jpg','overlay.png','mask.png'):
                filename=f'cam{cam}_{suffix}'
                shutil.copy2(SOURCE/stem/filename,PACKAGE/stem/filename)
        body=''.join(f'<h2>cam{c}</h2><p><a href="cam{c}_overlay.png">完整标注图</a> · <a href="cam{c}_review.jpg">原图 / 标注 / mask</a> · <a href="cam{c}_mask.png">分割mask</a></p><a href="cam{c}_detail.jpg"><img src="cam{c}_detail.jpg" style="max-width:100%"></a>' for c in 'ABCDEFG')
        (PACKAGE/stem/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>源帧 {frame} · 七视角</title><body style="background:#14202b;color:white;font:17px sans-serif"><p><a href="../index.html" style="color:#8cd4ff">返回首页</a></p><h1>AZL3 · 源帧 {frame}</h1><p>左侧原图，右侧原始标注；局部仅裁切放大，不改变坐标。边缘尾尖按负责人确认保留。</p>'+body,encoding='utf-8')
    # Reuse the previously verified browser UI, excluding superseded tail warnings.
    gallery=(DOWNLOAD/'PyMouseStudio-56-Annotation-Review/00_打开检查.html').read_text(encoding='utf-8')
    old='<p><a href="tail_tip_boundary_flags.jpg"'
    start=gallery.index(old);end=gallery.index('</p>',start)+4
    gallery=gallery[:start]+'<p>负责人已确认接受本批尾尖标注：不因靠近或位于图像边缘而判无效、屏蔽或要求返修。图像和坐标未改。</p><p><a href="all_frames.html">连续浏览全部八组</a> · <a href="README_先读我.md">文件说明</a></p>'+gallery[end:]
    gallery=gallery.replace('尾巴截断位置不等于真实尾尖。','边缘位置不作为本批尾尖有效性的硬性阈值。')
    gallery=gallery.replace('<title>PyMouseStudio · 56张回标可视化</title>','<title>PyMouseStudio · 56张标注 · 合作者检查包</title>')
    (PACKAGE/'index.html').write_text(gallery,encoding='utf-8')
    (PACKAGE/'all_frames.html').write_text('<!doctype html><meta charset="utf-8"><title>全部56张标注</title><body style="background:#14202b;color:white;font:18px sans-serif"><a href="index.html">返回可切换首页</a><h1>8个时刻 × 7个视角</h1>'+''.join(f'<h2><a href="frame_{f:06d}/index.html">源帧 {f}：逐视角原图对照</a></h2><a href="frame_{f:06d}_overview.jpg"><img style="max-width:100%" src="frame_{f:06d}_overview.jpg"></a>' for f in FRAMES),encoding='utf-8')
    readme='''# PyMouseStudio 56张标注：合作者检查包

## 打开方式

请先完整解压ZIP，再用Chrome、Edge或其他浏览器打开 `index.html`，不要只从压缩包内拖出HTML。无网络、无需安装软件。

首页可切换源帧、A–G视角及显示类型；点击“打开原尺寸图片”查看细节。`all_frames.html` 可连续查看所有概览。

## 本包内容

- AZL3，同一只鼠，源帧0、64、129、194、259、324、389、454，各camA–G，合计56张。
- 原始图像尺寸1080×1200；这8个时刻为抽样，不是连续逐帧。图中使用原始回标坐标，没有模型预测或自动修正。
- 每帧的 `camX_detail.jpg`：原图 / 标注局部对照；仅裁切放大。
- `camX_review.jpg`：原图 / 标注 / mask 三栏全图；顶部附检查信息。
- `camX_overlay.png`：原尺寸标注叠加图。
- `camX_mask.png`：原polygon直接栅格化的原尺寸mask，没有修补。
- 八张 `frame_XXXXXX_overview.jpg`：七视角概览。
- 包含全部56张的上述四种可视化；不包含原始视频、训练权重或原始标注JSON。

## 负责人已经确认的约定

**本批现有尾尖标注可接受：不因靠近或落在图像边缘而判错、屏蔽或要求返修，不使用边缘距离硬性阈值。**本包没有附旧版“边界尾尖返修”图，所有尾尖坐标原样保留。

单鼠不需要标注员补实例ID，转换时可统一指定。按仅标可见且能定位的点的约定，未标点不自动算漏标，也不会补零为有效坐标。可视化上没有为缺失点补画。

本包用于人工检查，不等于宣布所有点的3D同源性和分割精度已经通过。并未执行训练导入。

## 编号与颜色

青色：动物左侧；粉色：动物右侧；黄色：尾部；绿色：中轴。左右按动物解剖左右，不按屏幕左右。

01鼻尖；02头中心；03/04左/右耳；05颈；06背中体表；07腰背体表；08尾根；09–13尾中间点；14尾尖；15/16左/右前爪；17/18左/右后爪；19/20左/右肘；21/22左/右膝；23腹面体表。

06/07/23使用已批准的背腹体表语义；原始标签沿用旧通道名，图上按稳定编号展示。头内部中心、耳/肘膝、尾中间点等仍有定义或跨视角对应限制，展示这些原始点不代表全部解禁为训练真值。

三栏图中的 `visibility UNKNOWN / observation only` 是生成时的保守机器状态，不表示点一定不可见或需要逐点另填复杂状态；本包未据此删除标注。

## 反馈建议

如需指出修改，请按“源帧 + 相机 + 编号/轮廓区域”描述，例如“frame259 / camG / 腹足间尾段与mask”。请保留原图坐标和动物左右，勿以整体镜像来修正单点。

文件哈希和打包校验见 `manifest.json`。本包仅用于分享检查，不覆盖本地原始回标。
'''
    (PACKAGE/'README_先读我.md').write_text(readme,encoding='utf-8')
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    records=[];images=0
    for p in files:
        if p.suffix.lower() in ('.png','.jpg'):
            with Image.open(p) as im:im.load()
            images+=1
        if p.suffix=='.html':
            for ref in re.findall(r'(?:src|href)="([^"]+)"',p.read_text(encoding='utf-8')):
                if not ref or ref.startswith(('#','http')):continue
                assert (p.parent/ref).is_file(),(p,ref)
        records.append(dict(path=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    assert images==56*4+8
    # Dynamic gallery targets must all exist, in addition to static HTML links.
    for f in FRAMES:
        for c in 'ABCDEFG':
            for suffix in ('detail.jpg','review.jpg','overlay.png','mask.png'):
                assert (PACKAGE/f'frame_{f:06d}'/f'cam{c}_{suffix}').is_file()
    manifest=dict(package=NAME,source_images=56,source_frames=FRAMES,cameras=list('ABCDEFG'),
        visualization_images=images,tail_boundary_policy='accepted_by_user_no_edge_distance_rejection',
        annotations_modified=False,files=records)
    (PACKAGE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,NAME+'/'+p.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in PACKAGE.rglob('*'):
            if p.is_file():assert hashlib.sha256(z.read(NAME+'/'+p.relative_to(PACKAGE).as_posix())).hexdigest()==sha(p)
    shutil.copy2(archive,delivery);assert sha(archive)==sha(delivery)
    print(json.dumps(dict(zip=str(delivery),bytes=delivery.stat().st_size,sha256=sha(delivery),
                         visualizations=images,zip_crc='passed',all_links='passed',files=len(records)+1),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
