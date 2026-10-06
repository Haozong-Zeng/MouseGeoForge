"""Compare a returned review package against the exact sent ZIP, without label promotion."""
import collections,hashlib,html,itertools,json,shutil,stat,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_pymousestudio_manual_multiview import fundamental,epi,solve,projection,stats
ROOT=Path(__file__).resolve().parents[1];DL=Path('/mnt/c/Users/55395/Downloads')
OUT=ROOT/'output/pymousestudio_repair_audit_20261003';DEST=ROOT/'dataset/PyMouseStudio/annotation_returns/20261003_multiview_review_repaired'
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
FONT='/mnt/c/Windows/Fonts/msyh.ttc';FOCUS=[6,7,15,16,17,18]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def extract(zp,dest):
    dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None
        names=set()
        for info in z.infolist():
            name=info.filename.replace('\\','/');parts=PurePosixPath(name).parts
            assert parts and not any(x in ('..','/') or ':' in x for x in parts)
            assert not stat.S_ISLNK(info.external_attr>>16)
            assert name not in names;names.add(name)
            assert dest.joinpath(*parts).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    return next(p for p in dest.iterdir() if p.is_dir())
def points(a):
    d=collections.defaultdict(list)
    for s in a.get('shapes',[]):
        if s.get('shape_type')=='point':d[int(s['label'][:2])].append(s['points'][0])
    return dict(d)
def panel(image,ann,title,box):
    im=Image.open(image).convert('RGB');draw=ImageDraw.Draw(im);colors={6:'#ff70ff',7:'#ffff40',15:'#40ff60',16:'#ff9950',17:'#40ddff',18:'#ff60a0'}
    for k,vs in points(ann).items():
        if k not in FOCUS:continue
        for x,y in vs:
            draw.ellipse((x-6,y-6,x+6,y+6),outline=colors[k],width=3);draw.text((min(x+9,1000),max(y-29,0)),f'{k:02}',font=ImageFont.truetype(FONT,25),fill=colors[k],stroke_width=2,stroke_fill='black')
    im=im.crop(box);im.thumbnail((490,480));canvas=Image.new('RGB',(500,540),'#18212d');canvas.paste(im,((500-im.width)//2,55));ImageDraw.Draw(canvas).text((8,10),title,font=ImageFont.truetype(FONT,19),fill='white');return canvas
def main():
    OUT.mkdir(exist_ok=False)
    original=DL/'PyMouseStudio-Multiview-HumanReview-20261002.zip';returned=DL/'PyMouseStudio-Multiview-HumanReview-20261002---ok---fanxiu2.zip'
    old=extract(original,OUT/'sent_package');new=extract(returned,DEST)
    groups=json.loads((old/'human_review_pilot/selection_manifest.json').read_text())['groups']
    cases=json.loads((ROOT/'output/pymousestudio_numbered_review_20261001/review_manifest.json').read_text())['cases']
    entries=[dict(**g,folder='human_review_pilot/'+f"{g['id']}_{g['session']}_f{g['frame']:06d}") for g in groups]
    for ident in ['C17','C18']:
        case=next(c for c in cases if c['id']==ident);folder='supplemental_return_cases/'+Path(case['folder']).name
        entries.append(dict(id=ident,session=case['session'],frame=case['frame'],folder=folder,cameras=[p.stem[3:] for p in sorted((old/folder).glob('*.json'))]))
    result=dict(archive=str(returned),archive_sha256=sha(returned),reference_zip_sha256=sha(original),extracted=str(new),training_modified=False,calibration_modified=False,automatic_acceptance=False,groups=[],files=[],feedback=[],geometry_pairs=[],loo_observations=[])
    sections=[];by_source=collections.defaultdict(list)
    for group in entries:
        ident=group['id'];folder=group['folder'];cal=json.loads((BASE/'calibration'/group['session']/'rectified_calibration.json').read_text());before={};after={};changed=[];pictures=[]
        for op in sorted((old/folder).glob('*.json')):
            rp=new/folder/op.name;assert rp.exists();a=json.loads(op.read_text());b=json.loads(rp.read_text());pa,pb=points(a),points(b);cam=op.stem[3:];before[cam]=pa;after[cam]=pb;image=rp.with_suffix('.png');same_image=sha(image)==sha(op.with_suffix('.png'));assert same_image
            assert Image.open(image).size==(1080,1200)
            issues=[]
            if (b.get('imageWidth'),b.get('imageHeight'))!=(1080,1200):issues.append('dimensions')
            for k,vs in pb.items():
                if len(vs)!=1:issues.append(f'duplicate_{k}')
                if k<1 or k>23:issues.append(f'unknown_id_{k}')
                for xy in vs:
                    if len(xy)!=2 or not np.isfinite(xy).all() or not(0<=xy[0]<1080 and 0<=xy[1]<1200):issues.append(f'invalid_{k}')
            delta=[dict(id=k,before=pa.get(k,[]),after=pb.get(k,[])) for k in sorted(pa.keys()|pb.keys()) if pa.get(k,[])!=pb.get(k,[])]
            rec=dict(group=ident,session=group['session'],frame=group['frame'],camera=cam,changes=delta,issues=issues,image_unchanged=same_image,annotation=str(rp),shape_flags=[s.get('flags',{}) for s in b.get('shapes',[])],file_flags=b.get('flags',{}));result['files'].append(rec);by_source[(group['session'],group['frame'],cam)].append(pb)
            if delta:changed.append(rec)
            coords=[xy for d in (pa,pb) for vs in d.values() for xy in vs];xy=np.array(coords);lo=np.maximum(xy.min(0)-80,0).astype(int);hi=np.minimum(xy.max(0)+80,[1080,1200]).astype(int);box=(*lo,*hi)
            board=Image.new('RGB',(1000,540),'#18212d');board.paste(panel(image,a,f'{ident} cam{cam} 返修前',box),(0,0));board.paste(panel(image,b,f'{ident} cam{cam} 返修后',box),(500,0));name=f'{ident}_cam{cam}.jpg';board.save(OUT/name,quality=95);pictures.append(name)
        pair_rows=[];loo_rows=[]
        for k in range(1,24):
            common=[c for c in before if len(before[c].get(k,[]))==1 and len(after[c].get(k,[]))==1]
            for c,d in itertools.combinations(common,2):
                f=fundamental(np.array(cal[c]['K']),np.array(cal[c]['world_to_camera_mm']),np.array(cal[d]['K']),np.array(cal[d]['world_to_camera_mm']))
                pair_rows.append(dict(group=ident,id=k,cam1=c,cam2=d,before=epi(before[c][k][0],before[d][k][0],f),after=epi(after[c][k][0],after[d][k][0],f)))
            if len(common)>=3:
                K=np.array([cal[c]['K'] for c in common]);E=np.array([cal[c]['world_to_camera_mm'] for c in common]);uv=[np.array([ann[c][k][0] for c in common]) for ann in (before,after)]
                for vi,cam in enumerate(common):
                    keep=[j for j in range(len(common)) if j!=vi];values=[]
                    for u in uv:
                        x=solve(u[keep],K[keep],E[keep]);value=None
                        if x is not None:
                            p,z=projection(x,K[vi:vi+1],E[vi:vi+1])
                            if z[0]>0:value=float(np.linalg.norm(p[0]-u[vi]))
                        values.append(value)
                    loo_rows.append(dict(group=ident,id=k,camera=cam,before=values[0],after=values[1]))
        result['geometry_pairs'].extend(pair_rows);result['loo_observations'].extend(loo_rows)
        def summarize(rows,ids):
            rows=[r for r in rows if r['id'] in ids and r['before'] is not None and r['after'] is not None]
            return dict(before=stats([r['before'] for r in rows]),after=stats([r['after'] for r in rows]))
        rec=dict(id=ident,session=group['session'],frame=group['frame'],images=len(before),changed_images=len(changed),changed_ids=sorted({d['id'] for r in changed for d in r['changes']}),epipolar_focus=summarize(pair_rows,FOCUS),loo_focus=summarize(loo_rows,FOCUS));result['groups'].append(rec)
        sections.append(f'<h2>{ident} {group["session"]} / {group["frame"]}</h2><pre>{html.escape(json.dumps(rec,ensure_ascii=False,indent=2))}</pre>'+''.join(f'<p><a href="{p}">{p}</a></p><img loading="lazy" src="{p}">' for p in pictures))
    for op in old.rglob('*.txt'):
        if '反馈' not in op.name:continue
        rp=new/op.relative_to(old);text=rp.read_text(encoding='utf-8-sig') if rp.exists() else None
        result['feedback'].append(dict(path=op.relative_to(old).as_posix(),exists=rp.exists(),changed=rp.exists() and sha(op)!=sha(rp),text=text))
    result['copy_conflicts']=[list(k) for k,v in by_source.items() if any(x!=v[0] for x in v[1:])]
    counts=collections.Counter(d['id'] for r in result['files'] for d in r['changes']);result['counts']=dict(groups=len(entries),images=len(result['files']),changed_images=sum(bool(r['changes']) for r in result['files']),changed_groups=sum(r['changed_images']>0 for r in result['groups']),changed_id_images=dict(counts),issues=sum(bool(r['issues']) for r in result['files']),removed=sum(bool(d['before']) and not d['after'] for r in result['files'] for d in r['changes']),added=sum(not d['before'] and bool(d['after']) for r in result['files'] for d in r['changes']))
    result['metrics']={label:{'epipolar':summarize(result['geometry_pairs'],ids),'loo':summarize(result['loo_observations'],ids)} for label,ids in [('focus',FOCUS),('back',[6,7]),('paws',[15,16,17,18]) ]}
    result['per_id']={str(k):dict(epipolar=summarize(result['geometry_pairs'],[k]),loo=summarize(result['loo_observations'],[k])) for k in FOCUS}
    result['caveats']=['Matched same camera/ID observations only; additions/deletions separately counted.','Fixed calibration in rectified pixels; no second undistortion.','Geometric consistency does not certify anatomical truth; no training merge or removal of exclusions.','Returned contact sheets may be stale; these overlays are regenerated from actual JSON.']
    write(OUT/'summary.json',result);write(DEST/'audit_manifest.json',dict(archive_sha256=sha(returned),report=str(OUT/'summary.json'),training_allowed=False))
    (OUT/'index.html').write_text('<meta charset="utf-8"><title>第二轮返修核验</title><style>body{max-width:1100px;margin:32px auto;font:17px/1.6 sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>2026-10-03 第二轮返修核验</h1><p>只做核验，未覆盖训练标注。共同观测、冻结标定的几何诊断不等于解剖正确性。</p><pre>'+html.escape(json.dumps(dict(counts=result['counts'],metrics=result['metrics']),ensure_ascii=False,indent=2))+'</pre>'+''.join(sections),encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'],metrics=result['metrics'],feedback=[{k:v for k,v in r.items() if k!='text'} for r in result['feedback']]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
