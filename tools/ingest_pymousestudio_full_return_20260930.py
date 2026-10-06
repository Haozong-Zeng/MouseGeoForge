"""Archive-preserving, split-preserving ingestion of the full manual return.

No training import, inferred visibility, or coordinate corrections. Hash-match
every image to the exact rectified annotation delivery before geometry auditing.
"""
from pathlib import Path,PurePosixPath
import hashlib,io,json,re,shutil,stat,zipfile
from collections import Counter
import numpy as np
from PIL import Image
from review_pymousestudio_labelme_return import NAMES

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
PACK=ROOT/'dataset/PyMouseStudio/annotation_pack_v2'
SEMANTICS=NAMES.copy()
for k,n in [(5,'dorsal_thorax_surface'),(6,'dorsal_lumbar_surface'),(22,'ventral_abdomen_surface')]:SEMANTICS[k]=n

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not BASE.exists(),'Never overwrite a previous return.'
    files=sorted(Path('/mnt/c/Users/55395/Downloads').glob('PyMouseStudio-Rectified-Annotation-v2-Part*---ok.zip'))
    assert len(files)==2
    expected={r['image']:r for r in json.loads((PACK/'manifest.json').read_text())}
    groups=json.loads((PACK/'frame_groups.json').read_text())
    BASE.mkdir(parents=True);archives=[];members=[];seen=set();issues=[]
    for archive in files:
        entry=dict(file=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,files=0)
        with zipfile.ZipFile(archive) as z:
            assert sum(i.file_size for i in z.infolist())<3*1024**3
            for item in z.infolist():
                p=PurePosixPath(item.filename.replace('\\','/'))
                assert not p.is_absolute() and '..' not in p.parts and not any(':' in s for s in p.parts)
                assert not stat.S_ISLNK(item.external_attr>>16)
                if item.is_dir():continue
                assert len(p.parts)>=2 and p.parts[1]=='real_images',p
                rel=Path(*p.parts[1:]);assert rel.as_posix() not in seen;seen.add(rel.as_posix())
                assert rel.suffix.lower() in ('.png','.json'),rel
                data=z.read(item) # Reading to EOF validates each member CRC.
                h=hashlib.sha256(data).hexdigest()
                if rel.suffix=='.png':
                    assert rel.as_posix() in expected,rel
                    exp=expected[rel.as_posix()];assert exp['sha256']==h,('Image mismatch',rel)
                    with Image.open(io.BytesIO(data)) as im:
                        assert im.size==(exp['width'],exp['height']);im.verify()
                dest=BASE/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(data)
                assert digest(dest)==h
                members.append(dict(path=rel.as_posix(),sha256=h,archive=archive.name,member=item.filename))
                entry['files']+=1
        entry['crc']='all_members_passed';archives.append(entry)
        print('EXTRACTED',entry['file'],entry['files'],flush=True)
    pngs={p for p in seen if p.endswith('.png')};assert pngs==set(expected),('Image inventory mismatch',len(pngs),len(expected))
    assert {p[:-5]+'.png' for p in seen if p.endswith('.json')}==pngs
    rows=[];pergroup=[];labels=Counter()
    for g in groups:
        session=g['session'];split=g['split'];frame=g['source_frame'];cameras=g['cameras'];uv=np.full((len(cameras),23,2),np.nan,np.float64)
        for vi,cam in enumerate(cameras):
            rel=Path('real_images')/split/session/f'frame_{frame:06d}'/f'cam{cam}.json'
            a=json.loads((BASE/rel).read_text());errors=[];points=[];polys=[];duplicates=set()
            if (a.get('imageWidth'),a.get('imageHeight'))!=(1080,1200):errors.append('wrong_image_dimensions')
            if str(a.get('imagePath','')).replace('\\','/').split('/')[-1]!=f'cam{cam}.png':errors.append('wrong_image_path')
            for s in a.get('shapes',[]):
                label=s.get('label','');labels[label]+=1;xy=np.asarray(s.get('points',[]),dtype=float)
                if xy.ndim!=2 or xy.shape[-1]!=2 or not np.isfinite(xy).all():errors.append('invalid_shape:'+label);continue
                if (xy<0).any() or (xy[:,0]>=1080).any() or (xy[:,1]>=1200).any():errors.append('out_of_bounds:'+label)
                if s.get('shape_type')=='polygon':
                    polys.append(dict(label=label,vertices=len(xy)))
                    if len(xy)<3:errors.append('short_polygon:'+label)
                elif s.get('shape_type')=='point':
                    match=re.fullmatch(r'(\d{2})([a-z0-9_]+)',label)
                    if not match or int(match[1]) not in range(1,24) or len(xy)!=1:errors.append('unknown_point:'+label);continue
                    k=int(match[1])-1
                    if match[2] not in (NAMES[k],SEMANTICS[k]):errors.append('name_slot_mismatch:'+label);continue
                    if k in points:errors.append('duplicate:'+label);duplicates.add(k);continue
                    points.append(k)
                    if 0<=xy[0,0]<1080 and 0<=xy[0,1]<1200:uv[vi,k]=xy[0]
                else:errors.append('extra_shape_type:'+str(s.get('shape_type')))
            for k in duplicates:uv[vi,k]=np.nan
            if errors:issues.append(dict(image=rel.with_suffix('.png').as_posix(),errors=errors))
            rows.append(dict(split=split,session=session,frame=frame,camera=cam,points=len(points),
                point_ids=[k+1 for k in points],polygons=polys,errors=errors,
                explicit_point_states=any(bool(s.get('flags')) or bool(s.get('description')) for s in a.get('shapes',[]) if s.get('shape_type')=='point')))
        target=BASE/'normalized_observations'/split/session/f'frame_{frame:06d}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,uv=uv[:,None],xy_observed=np.isfinite(uv).all(-1)[:,None],names=np.asarray(NAMES),
            semantic_names=np.asarray(SEMANTICS),camera_names=np.asarray(cameras),instance_ids=np.asarray([0]),
            frame=np.asarray(frame),timestamp_seconds=np.asarray(g['time_seconds']),split=np.asarray(split),
            train_allowed=np.asarray(False),tail_edge_filter_applied=np.asarray(False))
        pergroup.append(dict(split=split,session=session,frame=frame,cameras=cameras,normalized_file=target.relative_to(BASE).as_posix()))
    sessions=sorted({g['session'] for g in groups});calibration=[]
    for s in sessions:
        src=PACK/'calibration'/s/'rectified_calibration.json';dst=BASE/'calibration'/s/'rectified_calibration.json';dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);assert digest(src)==digest(dst)
        calibration.append(dict(session=s,path=dst.relative_to(BASE).as_posix(),sha256=digest(dst)))
    result=dict(status='extracted_and_parsed',train_allowed=False,archives=archives,images=len(rows),groups=len(groups),
        sessions=sessions,total_points=sum(r['points'] for r in rows),split_counts=dict(Counter(r['split'] for r in rows)),
        total_polygons=sum(len(r['polygons']) for r in rows),issues=issues,views=rows,frame_groups=pergroup,calibration=calibration,
        files=members,point_label_counts=dict(labels),tail_policy='User accepted edge tail tips; all finite in-bounds observations retained, no distance-to-edge filter.',
        missing_policy='Missing coordinate is unavailable, never (0,0) ground truth; single known mouse assigned instance_id=0.',
        holdout_policy='Preserved locked split; geometry quality audit only, no calibration/model tuning or train import.')
    (BASE/'ingestion_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ['status','images','groups','sessions','total_points','split_counts','total_polygons','issues']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
