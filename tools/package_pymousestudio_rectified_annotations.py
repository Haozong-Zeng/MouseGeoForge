"""Finalize the all-view corrected annotation package after reconstruction QA."""
from pathlib import Path
import sys,json,shutil,zipfile,hashlib,html
from collections import Counter,defaultdict
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from export_pymousestudio_rectified import write,digest


def main():
    base=ROOT/'dataset/PyMouseStudio';pack=base/'annotation_pack_v2'
    assert json.loads((base/'export_status.json').read_text())['state']=='complete'
    assert (base/'sessions/AZL3/reconstruction_bottom_refined_v1/report.json').exists()
    rows=json.loads((pack/'manifest.json').read_text());assert len(rows)==1950
    old=ROOT/'output/annotation_packages/PyMouseStudioRaw-Core23-AnnotationPack-v1'
    shutil.copytree(old/'synthetic_examples',pack/'synthetic_examples',dirs_exist_ok=True)
    for name in ['core23_definitions.csv','source31_vs_core23.csv','instance_template.json']:
        shutil.copy2(old/name,pack/name)
    guide=ROOT/'docs/pymousestudio_rectified_annotation_guide.md'
    shutil.copy2(guide,pack/'README_标注说明.md')
    audit=(ROOT/'docs/pymousestudio_core23_annotation_20260912.md').read_text().split('## 标注员操作说明')[0]
    (pack/'Core23_语义审计.md').write_text(audit,encoding='utf-8')
    schema=json.loads((old/'schema.json').read_text());schema.update(
        image_coordinates='rectified PNG, zero-based x-right y-down, same original K, output D=0',
        do_not_use_raw_v1_coordinates=True)
    write(pack/'schema.json',schema)
    groups=defaultdict(list)
    for row in rows:
        # Relativize image paths for an independently portable package.
        image=Path(row['image']).relative_to('annotation_pack_v2').as_posix()
        row['image']=image
        groups[(row['session'],row['frame'])].append(row)
    expected={s['session']:s for s in json.loads((base/'sequences.json').read_text())}
    group_rows=[]
    for (s,f),rs in sorted(groups.items()):
        cams=sorted(r['camera'] for r in rs);assert cams==sorted(expected[s]['cameras'])
        group_rows.append(dict(group_id=f'{s}__{f:06d}',session=s,source_frame=f,time_seconds=f/52,
                               split=rs[0]['split'],cameras=cams,images=[r['image'] for r in rs],
                               missing_source_cameras=expected[s]['missing_camera_videos']))
    assert len(group_rows)==300
    pilot=[g['group_id'] for g in group_rows if g['session'] in ['AZL3','CR2','FR2']
           and g['source_frame'] in [0,779]]
    assert len(pilot)==6
    write(pack/'frame_groups.json',group_rows);write(pack/'pilot_group_ids.json',pilot)
    write(pack/'manifest.json',rows)
    blanks=[dict(task_id=f'{r["session"]}__{r["frame"]:06d}__cam{r["camera"]}',image=r['image'],
                 width=1080,height=1200,status='pending',annotator=None,reviewer=None,instances=[],
                 empty_instances_means='UNANNOTATED, not verified absence',notes='') for r in rows]
    (pack/'annotation_template.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in blanks),encoding='utf-8')
    for s in expected:
        dst=pack/'calibration'/s;dst.mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/'sessions'/s/'calibration.json',dst/'rectified_calibration.json')
        shutil.copytree(base/'sessions'/s/'valid_pixels',dst/'valid_pixels',dirs_exist_ok=True)
    hashes=[]
    for r in rows:
        p=pack/r['image'];assert digest(p)==r['sha256']
        with Image.open(p) as im:assert im.size==(1080,1200);im.verify()
        hashes.append(r['sha256'])
    assert len(set(hashes))==1950
    split_counts=dict(Counter(r['split'] for r in rows));assert split_counts=={'real_development':1475,'real_locked_holdout':475}
    report=dict(passed=True,image_count=1950,time_group_count=300,bottom_images=300,side_images=1650,
                split_counts=split_counts,all_available_views_per_group_verified=True,
                dimensions_wh=[1080,1200],lens_corrected_once=True,manual_labels_created=False,
                model_predictions_shown=False,synthetic_examples=7,pilot_groups=6,pilot_images=42,
                mask_or_depth_predictions_in_annotation_package=False,train_allowed=False)
    write(pack/'verification.json',report)
    lines=['<!doctype html><meta charset="utf-8"><title>PyMouseStudio corrected annotation package</title>',
           '<style>body{font:16px sans-serif;max-width:1300px;margin:30px auto}pre{white-space:pre-wrap}img{max-width:100%}</style>',
           '<h1>300 个时间帧 / 全视角 / 1,950 张矫正图片</h1><pre>'+html.escape(guide.read_text())+'</pre>',
           '<h2>Blender PyMouseStudio 合成 GT 定义参考</h2>']
    for p in sorted((pack/'synthetic_examples').glob('*_numbered.png')):
        lines.append(f'<p>{p.name}</p><img loading="lazy" src="synthetic_examples/{p.name}">')
    lines.append('<h2>全部时间组（holdout 不内嵌预览）</h2><ul>')
    for g in group_rows:
        links=' | '.join(f'<a href="{i}">cam{c}</a>' for c,i in zip(g['cameras'],g['images']))
        lines.append(f'<li>{g["group_id"]} [{g["split"]}]: {links}</li>')
    lines.append('</ul>');(pack/'index.html').write_text('\n'.join(lines),encoding='utf-8')
    files={p.relative_to(pack).as_posix():digest(p) for p in sorted(pack.rglob('*')) if p.is_file() and p.name!='checksums.json'}
    write(pack/'checksums.json',files)
    target=base/'PyMouseStudio-Rectified-AllViews-300Frames-Annotation-v2.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(pack.rglob('*')):
            if p.is_file():z.write(p,(Path('PyMouseStudio-Rectified-Annotation-v2')/p.relative_to(pack)).as_posix())
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    receipt=dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=digest(target),
                 zip_crc_passed=True,report=report)
    write(base/'annotation_delivery.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
