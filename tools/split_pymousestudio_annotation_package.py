"""Two standalone ZIPs below decimal 1 GB, preserving existing data splits."""
from pathlib import Path
import hashlib,html,json,sys,zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from export_pymousestudio_rectified import write,digest


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def main():
    base=ROOT/'dataset/PyMouseStudio'
    receipt=json.loads((base/'annotation_delivery.json').read_text())
    source=ROOT/receipt['path'];assert digest(source)==receipt['sha256']
    parts=[('Part1-Development','real_development'),('Part2-Holdout','real_locked_holdout')]
    targets=[base/f'PyMouseStudio-Rectified-Annotation-v2-{name}.zip' for name,_ in parts]
    assert not any(p.exists() for p in targets),'Refusing to overwrite an existing delivery'
    all_images=set();all_groups=set();deliveries=[]
    with zipfile.ZipFile(source) as original:
        roots={n.split('/')[0] for n in original.namelist()};assert len(roots)==1
        prefix=roots.pop()+'/'
        files={i.filename[len(prefix):]:i for i in original.infolist() if not i.is_dir()}
        manifest=json.loads(original.read(prefix+'manifest.json'))
        groups=json.loads(original.read(prefix+'frame_groups.json'))
        templates=[json.loads(line) for line in original.read(prefix+'annotation_template.jsonl').decode('utf-8').splitlines()]
        pilots=json.loads(original.read(prefix+'pilot_group_ids.json'))
        regenerated={'manifest.json','frame_groups.json','annotation_template.jsonl','pilot_group_ids.json',
                     'verification.json','checksums.json','index.html','README_标注说明.md'}
        for (name,split),target in zip(parts,targets):
            selected=[r for r in manifest if r['split']==split]
            selected_groups=[g for g in groups if g['split']==split]
            image_names={r['image'] for r in selected};group_ids={g['group_id'] for g in selected_groups}
            sessions={r['session'] for r in selected}
            assert all(set(g['images'])<=image_names for g in selected_groups)
            assert not (all_images&image_names) and not (all_groups&group_ids)
            all_images|=image_names;all_groups|=group_ids
            hashes={};archive_root=f'PyMouseStudio-Rectified-Annotation-v2-{name}/'
            selected_pilots=[g for g in pilots if g in group_ids]
            intro=(f'# 独立分包：{name}\n\n本包包含 {len(selected_groups)} 个完整时间组、{len(selected)} 张图片，'
                   f'保留原 split `{split}`。可单独解压和标注，不需要另一个分包。\n\n'
                   '每个时间组的全部可用相机均在本包内，图片字节、尺寸和坐标没有变化。'
                   '本包 manifest、frame_groups、annotation_template 和校验清单均仅针对本包。\n\n'
                   '开发集与留出集不得混用。留出集不得用于训练、调参或选择模型。\n\n'
                   f'本包试标组数量：{len(selected_pilots)}。留出包没有开发集试标任务。\n\n'
                   '以下是原完整包的统一标注规范；其中 300 组、1,950 张等是两包合计数量，'
                   '不是本包数量。标注定义与原规范保持不变。\n\n---\n\n')
            guide=intro+original.read(prefix+'README_标注说明.md').decode('utf-8')
            verification=dict(passed=True,part=name,split=split,image_count=len(selected),time_group_count=len(selected_groups),
                all_available_views_per_group_preserved=True,image_bytes_unchanged=True,standalone=True,
                source_zip_sha256=receipt['sha256'],size_wh=[1080,1200],train_allowed=False)
            generated={
                'manifest.json':encoded(selected),'frame_groups.json':encoded(selected_groups),
                'pilot_group_ids.json':encoded(selected_pilots),'verification.json':encoded(verification),
                'annotation_template.jsonl':(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in templates if r['image'] in image_names)).encode('utf-8'),
                'README_标注说明.md':guide.encode('utf-8')}
            index=['<!doctype html><meta charset="utf-8"><title>'+name+'</title>',
                   '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}img{max-width:100%}</style>',
                   '<h1>'+name+'</h1><pre>'+html.escape(guide)+'</pre><h2>Blender 合成参考</h2>']
            for path in sorted(files):
                if path.startswith('synthetic_examples/') and path.endswith('_numbered.png'):
                    index.append(f'<p>{html.escape(path)}</p><img loading="lazy" src="{html.escape(path)}">')
            index.append('<h2>本包时间组</h2><ul>')
            for g in selected_groups:
                links=' | '.join(f'<a href="{html.escape(p)}">cam{c}</a>' for c,p in zip(g['cameras'],g['images']))
                index.append(f'<li>{g["group_id"]} [{split}]: {links}</li>')
            index.append('</ul>');generated['index.html']='\n'.join(index).encode('utf-8')
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as dest:
                for path,info in sorted(files.items()):
                    if path in regenerated:continue
                    if path.startswith('real_images/') and path not in image_names:continue
                    if path.startswith('calibration/') and path.split('/')[1] not in sessions:continue
                    data=original.read(info);dest.writestr(archive_root+path,data)
                    hashes[path]=hashlib.sha256(data).hexdigest()
                for path,data in generated.items():
                    dest.writestr(archive_root+path,data);hashes[path]=hashlib.sha256(data).hexdigest()
                dest.writestr(archive_root+'checksums.json',encoded(hashes))
            assert target.stat().st_size<1_000_000_000
            with zipfile.ZipFile(target) as check:
                assert check.testzip() is None
                names={n[len(archive_root):] for n in check.namelist()}
                assert {n for n in names if n.startswith('real_images/')}==image_names
                for path,sha in hashes.items():assert hashlib.sha256(check.read(archive_root+path)).hexdigest()==sha
                for r in selected:assert hashes[r['image']]==r['sha256']
                assert len(check.read(archive_root+'annotation_template.jsonl').decode('utf-8').splitlines())==len(selected)
            result=dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=digest(target),
                groups=len(selected_groups),images=len(selected),split=split,standalone=True,zip_crc_passed=True)
            deliveries.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    assert all_images=={r['image'] for r in manifest} and all_groups=={g['group_id'] for g in groups}
    assert len(all_images)==1950 and len(all_groups)==300
    assert digest(source)==receipt['sha256']
    write(base/'annotation_split_delivery.json',dict(passed=True,parts=deliveries,total_unique_images=1950,total_unique_time_groups=300,
        original_zip_unchanged=True,images_reencoded=False,maximum_bytes_per_part_exclusive=1_000_000_000))


if __name__=='__main__':main()
