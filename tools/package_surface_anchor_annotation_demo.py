"""Build a standalone reviewer-first synthetic surface-landmark demo patch."""
import hashlib
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/surface_anchor_demo_v1'
OUT=ROOT/'output/annotation_packages/PyMouseStudio-SurfaceAnchors-Demo-v3-Review'
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
COLORS={5:'#ff923d',6:'#56ed80',22:'#38c6ff'}
LABELS={5:'06 背中表面',6:'07 腰背表面',22:'23 腹面表面'}


def font(n):return ImageFont.truetype(FONT,n)


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def marked_crop(raw,uv,visible,crop,old=False,all_points=False):
    x0,y0,x1,y1=crop
    scale=min(720/(x1-x0),440/(y1-y0))
    w,h=round((x1-x0)*scale),round((y1-y0)*scale)
    base=raw.crop(crop).resize((w,h),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(800,540),'#16202b');canvas.paste(base,((800-w)//2,50))
    draw=ImageDraw.Draw(canvas)
    draw.text((18,10),'旧版：骨骼起点（6 / 23 重合）' if old else '新版：固定网格表面锚点',font=font(25),fill='white')
    px=(800-w)//2
    if all_points:
        for k,p in enumerate(uv):
            if k in COLORS:continue
            x,y=(p[0]-x0)*scale+px,(p[1]-y0)*scale+50
            if not (0<x<800 and 45<y<500):continue
            draw.ellipse((x-2,y-2,x+2,y+2),fill='#dedede',outline='black')
            draw.text((x+3,y-13),str(k+1),font=font(14),fill='white',stroke_width=1,stroke_fill='black')
    # Distinct callout rails; do not overlay labels 6 and 23 even in the old view.
    for n,k in enumerate(COLORS):
        p=uv[k];x,y=(p[0]-x0)*scale+px,(p[1]-y0)*scale+50
        col=COLORS[k]
        fill=col if visible[k] else None
        draw.ellipse((x-6,y-6,x+6,y+6),fill=fill,outline=col,width=3)
        tx=20 if n<2 else 575;ty=85+n*80
        label=f'{k+1:02d}（旧）' if old else LABELS[k]
        draw.line((tx+75,ty+23,x,y),fill=col,width=2)
        draw.text((tx,ty),label,font=font(24),fill=col,stroke_width=2,stroke_fill='#16202b')
    draw.text((18,502),'实心：裸鼠网格视线通达；空心：被鼠体遮挡' if not old else '旧坐标仅用于说明差异，不再作为这三个点的示范',font=font(19),fill='#e3e8ed')
    return canvas


def main(argv=None):
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-generated',action='store_true')
    parser.add_argument('--approved-surface',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip-zip',action='store_true')
    args=parser.parse_args(argv)
    if args.output is not None:
        candidate=args.output.resolve()
        assert candidate.parent==(ROOT/'output/annotation_packages').resolve(), 'Output must be an immediate annotation_packages child'
        OUT=candidate
    if OUT.exists():
        assert args.refresh_generated and (OUT/'core23_demo_definitions.json').is_file(), 'Refuse overwrite of unrecognized output'
    OUT.mkdir(parents=True,exist_ok=args.refresh_generated)
    examples=OUT/'synthetic_examples';examples.mkdir(exist_ok=args.refresh_generated)
    geometry=OUT/'surface_geometry';geometry.mkdir(exist_ok=args.refresh_generated)
    anchors=json.loads((SOURCE/'rest_surface_anchors.json').read_text())
    shutil.copy2(SOURCE/'rest_surface_anchors.json',OUT/'rest_surface_anchors.json')
    for p in SOURCE.glob('*SurfaceAnchorReview.blend'):shutil.copy2(p,geometry/p.name)
    for p in SOURCE.glob('*_surface.png'):shutil.copy2(p,geometry/p.name)
    for p in SOURCE.glob('*_surface_tracks.json'):shutil.copy2(p,geometry/p.name)
    # Rendered canonical rest localization, not a frame from another video.
    rest_manifest=json.loads((SOURCE/'WalkAround_render_manifest.json').read_text())[0]
    sheet=Image.new('RGB',(1800,780),'#16202b');d=ImageDraw.Draw(sheet)
    d.text((24,16),'默认静止网格：先固定位置，再随网格变形',font=font(35),fill='white')
    d.text((24,68),'橙 06 背中表面     绿 07 腰背表面     蓝 23 腹面表面   |   彩色面片 = 固定顶点区域',font=font(23),fill='#e0e7ee')
    for i,role in enumerate(['top','bottom','side']):
        rec=next(v for v in rest_manifest['views'] if v['role']==role)
        im=Image.open(SOURCE/rec['image']).convert('RGB')
        crop=(500,260,780,710) if role!='side' else (20,175,1240,560)
        dest=(420,620) if role!='side' else (880,460)
        resized=im.crop(crop);resized.thumbnail(dest,Image.Resampling.LANCZOS)
        ox=[20,460,900][i];oy=130 if i<2 else 220
        sheet.paste(resized,(ox,oy));d.text((ox,105),{'top':'俯视：背部定位','bottom':'底视：腹部定位','side':'侧视：背 / 腹是不同表面'}[role],font=font(25),fill='white')
        sx=resized.width/(crop[2]-crop[0]);sy=resized.height/(crop[3]-crop[1])
        indices=[0,1] if role=='top' else [2] if role=='bottom' else [0,1,2]
        for j in indices:
            x,y=rec['uv'][j];x=ox+(x-crop[0])*sx;y=oy+(y-crop[1])*sy;k=anchors['anchors'][j]['core23_id']
            dx=35 if i<2 else 20;dy=-30 if j<2 else 20
            d.line((x,y,x+dx,y+dy),fill=COLORS[k],width=2)
            d.text((x+dx,y+dy-12),str(k+1),font=font(27),fill=COLORS[k],stroke_width=2,stroke_fill='#16202b')
    d.text((24,747),('06/07/23位置已确认；' if args.approved_surface else '位置提案待确认；')+'表面点不是内部骨骼中心。侧面看不到时不能强行标 visible。',font=font(23),fill='#ffd18b')
    sheet.save(OUT/'01_rest_surface_definition.jpg',quality=96)
    manifests=[];side_cards=[];core_names=None
    for action in ['WalkAround','WashFace','ClimbWall']:
        track=json.loads((SOURCE/f'{action}_surface_tracks.json').read_text())
        folder=ROOT/f'dataset/BlenderPyMouse/PyMouseStudio-BrownFur-{action}-Core23/frame_000031'
        legacy=json.loads((folder/'keypoints.json').read_text());core_names=legacy['names']
        comparison=Image.new('RGB',(1600,3*590+100),'#16202b');cd=ImageDraw.Draw(comparison)
        cd.text((20,15),f'{action} 第31帧 | 姿势、相机和RGB未修改，仅更换三个点的定义',font=font(28),fill='white')
        for row,view_id in enumerate([0,6,7]):
            role={0:'side',6:'bottom',7:'top'}[view_id]
            label={'side':'侧视','bottom':'底视','top':'俯视'}[role]
            rawpath=folder/'blended_images'/f'{view_id:08d}.jpg';raw=Image.open(rawpath).convert('RGB')
            oldview=next(v for v in legacy['views'] if v['view_id']==view_id)
            proposed=next(v for v in track['frame31_views'] if v['view_id']==view_id)
            olduv=np.array([p['uv'] for p in oldview['keypoints']]);newuv=olduv.copy()
            vis={k:bool(oldview['keypoints'][k]['visible']) for k in range(23)}
            for j,k in enumerate(COLORS):newuv[k]=proposed['uv'][j];vis[k]=proposed['body_mesh_line_of_sight'][j]
            body=[k for k in range(23) if k not in range(7,14)]
            bounds=np.concatenate([olduv[body],newuv[body]])
            lo=np.maximum([0,0],np.floor(bounds.min(0)-32)).astype(int)
            hi=np.minimum(raw.size,np.ceil(bounds.max(0)+32)).astype(int)
            crop=(*lo,*hi)
            old=marked_crop(raw,olduv,{k:False for k in COLORS},crop,old=True)
            new=marked_crop(raw,newuv,vis,crop)
            cd.text((20,72+row*590),f'{label} · 原相机 {view_id:02d} · 聚焦身体区域',font=font(22),fill='#a7bdcc')
            comparison.paste(old,(0,110+row*590));comparison.paste(new,(800,110+row*590))
            stem=f'{action}_frame000031_view{view_id:02d}_{role}'
            raw.save(examples/f'{stem}_clean.png')
            # All 23 IDs have separated external rails; unchanged points remain
            # hollow and explicitly deferred, never implied newly approved.
            full=Image.new('RGB',(1800,1000),'#16202b');fd=ImageDraw.Draw(full)
            fd.text((20,14),f'{action} | 第31帧 {label} | 仅06/07/23更新，其余旧定义待审',font=font(29),fill='white')
            original=raw.resize((400,400),Image.Resampling.LANCZOS);full.paste(original,(15,220))
            fd.text((20,175),'原始带毛RGB（无改动）',font=font(22),fill='white')
            lo_all=np.maximum([0,0],np.floor(np.minimum(olduv.min(0),newuv.min(0))-15)).astype(int)
            hi_all=np.minimum(raw.size,np.ceil(np.maximum(olduv.max(0),newuv.max(0))+15)).astype(int)
            crop_all=(*lo_all,*hi_all);z=raw.crop(crop_all)
            sc=min(650/z.width,730/z.height);z=z.resize((round(z.width*sc),round(z.height*sc)),Image.Resampling.LANCZOS)
            ox=530+(650-z.width)//2;oy=140+(730-z.height)//2
            full.paste(z,(ox,oy))
            mapped=np.array([[(p[0]-lo_all[0])*sc+ox,(p[1]-lo_all[1])*sc+oy] for p in newuv])
            sides=[[k for k in range(23) if mapped[k,0]<ox+z.width/2],
                   [k for k in range(23) if mapped[k,0]>=ox+z.width/2]]
            for si,ids in enumerate(sides):
                ids.sort(key=lambda k:(mapped[k,1],k))
                for k,yy in zip(ids,np.linspace(145,860,len(ids))):
                    x,y=mapped[k];col=COLORS.get(k,'#bcb2c7');tx=458 if si==0 else 1230
                    fd.line((tx+15,yy+15,x,y),fill=col,width=1)
                    fd.ellipse((x-4,y-4,x+4,y+4),fill=col if k in COLORS and vis[k] else None,outline=col,width=2)
                    fd.rounded_rectangle((tx-7,yy-3,tx+35,yy+32),radius=4,fill='#16202b',outline=col,width=1)
                    fd.text((tx,yy),f'{k+1:02d}',font=font(22),fill=col)
            for k in range(23):
                text=LABELS[k] if k in COLORS else f'{k+1:02d} {core_names[k]}'
                fd.text((1330,105+k*32),text,font=font(20),fill=COLORS.get(k,'#c9c2d1'))
            fd.text((25,921),'新表面点：实心 = 裸鼠网格视线通达；空心 = 鼠体自遮挡。旧点均以空心作定义参考，不代表可见性。',font=font(23),fill='white')
            fd.text((25,958),'人工不应强行填满23点；未明确的耳、肘膝、内部点与尾部中间点仍需冻结定义。',font=font(22),fill='#ffd18b')
            full.save(examples/f'{stem}_numbered.png')
            new.save(examples/f'{stem}_surface_focus.png')
            old.save(examples/f'{stem}_legacy_focus.png')
            entries=[]
            for k,p in enumerate(oldview['keypoints']):
                entries.append(dict(display_id=k+1,core23_id=k,
                    name=next((a['name'] for a in anchors['anchors'] if a['core23_id']==k),core_names[k]),
                    legacy_name=core_names[k],uv=newuv[k].tolist(),
                    definition=('surface_anchor_v1_user_confirmed' if args.approved_surface else 'surface_anchor_proposal_v1') if k in COLORS else 'unchanged_legacy_bone_head_deferred',
                    body_mesh_line_of_sight=vis[k] if k in COLORS else None,
                    train_valid=False))
            record=dict(action=action,frame=31,view_id=view_id,role=role,width=raw.width,height=raw.height,
                schema='core23_surface_demo_v3_approved_definition' if args.approved_surface else 'core23_surface_demo_v3_review_only',reference_only=True,train_allowed=False,
                image_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),
                keypoints=entries,changed_ids=[6,7,23],
                warning='Only these three surface definitions changed; other bone/ear/paw definitions not newly validated. Visibility is skin-only, not fur/scene.')
            write(examples/f'{stem}_gt.json',record);manifests.append(record)
            if view_id==0:side_cards.append((action,new))
        comparison.save(OUT/f'02_{action}_before_after.jpg',quality=95)
    side_sheet=Image.new('RGB',(2400,625),'#16202b');d=ImageDraw.Draw(side_sheet)
    d.text((22,10),'三种动作同一固定表面定义：不是每帧最高点 / 最低点，也不是沿侧面轮廓重新挑点',font=font(30),fill='white')
    for i,(name,card) in enumerate(side_cards):
        d.text((800*i+20,55),name+'  frame 31',font=font(25),fill='white');side_sheet.paste(card,(800*i,85))
    side_sheet.save(OUT/'03_three_actions_surface_points.jpg',quality=95)
    write(examples/'index.json',manifests)
    definition_rows=[]
    for k,name in enumerate(core_names):
        a=next((a for a in anchors['anchors'] if a['core23_id']==k),None)
        definition_rows.append(dict(display_id=k+1,core23_id=k,legacy_name=name,
            proposed_name=a['name'] if a else name,status=('approved_surface_definition' if args.approved_surface else 'review_surface_proposal') if a else 'unchanged_not_newly_approved',
            source='fixed_rest_mesh_anchor' if a else 'legacy_bone_head'))
    write(OUT/'core23_demo_definitions.json',definition_rows)
    readme='''# 背部 / 腹部表面锚点示意 v3（负责人先审，不是已批准批量标注规则）

先看 01_rest_surface_definition.jpg，再看 02_*_before_after.jpg 和 03_three_actions_surface_points.jpg。

- 06 橙色：背中表面；07 绿色：腰背表面；23 蓝色：腹面表面。
- 在原始未变形网格的正中线上选取固定表面位置。06/23取Spine3对应的同一纵向截面，07取Spine1对应截面；该位置提案需负责人目视确认。
- 固定三角形顶点编号及重心权重决定坐标，随网格变形。彩色区域是周边21个固定连通顶点，仅用于展示位置范围；不是逐帧按最高/最低点重选。
- 这些是皮肤表面参考点，不再是内部脊柱中心。若采用，需要合成导出和真实标注共同版本升级，不能与旧6/7/23标签静默混用。
- 实心/空心只表示裸鼠网格视线通达/被自身遮挡；不代表毛发、场景遮挡或真人能精确定位。侧面看不到腹面时留空或按规定标为推测，不能把背部同一像素重复标成腹部。
- 原有走路、洗脸、攀墙各第31帧的带毛RGB、姿态与相机不变。标注图更新的是3D表面点投影；原训练集未改。灰模是新渲染的几何辅助示意，不是毛色修改。
- surface_geometry 的灰模图使用1280×720 Cycles / OptiX，正交相机只用于说明解剖位置；synthetic_examples保留原512×512透视RGB及原相机投影，没改训练相机。
- 除06/07/23之外的头、耳、肘膝、尾中间点等仍有旧定义待审；不能因本包更新就强制补齐23个点。
- *.blend是轻量、自包含的静态表面检查项目（Rest或第31帧），不是替换生产动画。选择 *_Surface 网格，可查看 KP06/KP07/KP23 的 anchor/region 顶点组。
- 该包不含待标真实图，不需重新标定或重新矫正真实图。ZIP是给负责人审核的demo补充包，未自动改写外置硬盘上的交付包。
'''
    if args.approved_surface:
        readme='''# Synthetic表面锚点示意（06/07/23已由负责人目视确认）

先看01静止网格定位，再看02三动作新旧对照与synthetic_examples完整编号图。
06背中表面、07腰背表面、23腹面表面为本次确认定义；它们不是内部骨骼中心。
固定顶点区域随网格变形，不逐帧重选最高/最低点。灰模隐藏毛发仅为展示网格。
带毛RGB沿用原始512×512透视图；灰模是1280×720正交辅助说明，不更改训练相机。
新点实心/空心仅为裸鼠网格视线判断，不代替人工可见性判断。
旧版对照仅解释错误来源，不用于继续按旧位置标注。其他有歧义点仍按正式定义表暂缓。
surface_geometry内Blender文件是静态表面检查项目，不是生产动画。
本包不修改既有训练集，示例坐标也不是待标真实图的预标注。
'''
    readme_name='README_示意说明.md' if args.approved_surface else 'README_先审核再交付.md'
    (OUT/readme_name).write_text(readme,encoding='utf-8')
    images=['01_rest_surface_definition.jpg','03_three_actions_surface_points.jpg']+[f'02_{a}_before_after.jpg' for a in ['WalkAround','WashFace','ClimbWall']]
    html='<!doctype html><meta charset="utf-8"><title>表面锚点示意 v3</title><body style="font-family:sans-serif;background:#16202b;color:white;margin:24px"><h1>背部 / 腹部：固定网格表面定义（待审核）</h1><p>只更新06、07、23。鼠体姿态与原训练RGB不变。空心点不是要求人工强行点击。</p>'
    for name in images:html+=f'<h2>{name}</h2><a href="{name}"><img style="max-width:100%" src="{name}"></a>'
    html+='<h2>附带原分辨率图和完整编号参考</h2>'
    for p in sorted(examples.glob('*_numbered.png')):html+=f'<p><a href="synthetic_examples/{p.name}">{p.name}</a></p>'
    html+=f'<p><a href="{readme_name}">说明</a></p></body>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'}
    write(OUT/'checksums.json',hashes)
    if args.skip_zip:
        return
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w' if args.refresh_generated else 'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(output=str(OUT),zip_bytes=archive.stat().st_size,files=len(hashes),synthetic_views=len(manifests))))


if __name__=='__main__':main()
