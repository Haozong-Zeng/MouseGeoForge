"""Audit independent Labelme surface fragments; never export real curve GT."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from PIL import Image,ImageDraw
from mousegeoforge.surface_curve_review import parse_annotation,compare_fragments,ENDPOINT_LABELS
from mousegeoforge.multiview_point_audit import triangulate_observations,project_point
from prepare_surface_curve_human_pilot_20261004 import OUT as PILOT,sha,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--a',type=Path,required=True);parser.add_argument('--b',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert args.a.resolve()!=args.b.resolve(),'A/B must be two separate returned folders, not the same annotation twice'
    assert not args.out.exists();args.out.mkdir(parents=True);(args.out/'overlays').mkdir()
    manifests={tag:json.loads((PILOT/f'Annotator_{tag}/manifest.json').read_text()) for tag in ('A','B')}
    assert manifests['A']['files']==manifests['B']['files']
    rows=[];parsed={};input_hashes={};completed=collections.Counter();valid=collections.Counter();calibrations={}
    for spec in manifests['A']['files']:
        canonical=PILOT/'Annotator_A'/spec['path'];assert sha(canonical)==spec['sha256']
        row={k:spec[k] for k in ('group','camera','role','session','frame')};row['annotators']={}
        for tag,base in [('A',args.a),('B',args.b)]:
            path=base/spec['annotation'];returned_image=base/spec['path'];issue=[]
            if returned_image.exists() and sha(returned_image)!=spec['sha256']:issue.append('returned_image_changed')
            if not path.exists():row['annotators'][tag]=dict(reviewed=False,valid=False,errors=['missing_json']);continue
            input_hashes[str(path.resolve())]=sha(path)
            try:annotation=json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError,UnicodeError):row['annotators'][tag]=dict(reviewed=False,valid=False,errors=['invalid_json']);continue
            p=parse_annotation(annotation,width=spec['width'],height=spec['height'],image_name=Path(spec['path']).name)
            p['errors']+=issue;p['valid']=p['valid'] and not issue
            row['annotators'][tag]={k:p[k] for k in ('reviewed','valid','errors','no_visible')}
            row['annotators'][tag]['confirmed_fragments']=len(p['lines']['dorsal_surface'])
            completed[tag]+=int(p['reviewed']);valid[tag]+=int(p['reviewed'] and p['valid'])
            if p['valid'] and p['reviewed']:parsed[(tag,spec['group'],spec['camera'])]=p
        a=parsed.get(('A',spec['group'],spec['camera']));b=parsed.get(('B',spec['group'],spec['camera']))
        if a is not None and b is not None:
            row['identical_annotation_files']=sha(args.a/spec['annotation'])==sha(args.b/spec['annotation'])
            row['agreement']=compare_fragments(a['lines']['dorsal_surface'],b['lines']['dorsal_surface'])
            row['visibility_disagreement']=bool(a['lines']['dorsal_surface'])!=bool(b['lines']['dorsal_surface'])
            row['endpoint_separation_px']={k:float(np.linalg.norm(a['endpoints'][k]-b['endpoints'][k])) for k in ENDPOINT_LABELS if k in a['endpoints'] and k in b['endpoints']}
            with Image.open(canonical) as im:im=im.convert('RGB')
            draw=ImageDraw.Draw(im)
            for tag,p,color in [('A',a,'#33ff66'),('B',b,'#ff66ee')]:
                for line in p['lines']['dorsal_surface']:draw.line([tuple(x) for x in line],fill=color,width=3)
                for point in p['endpoints'].values():x,y=point;draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
            draw.rectangle((0,0,1080,35),fill='black');draw.text((10,10),'A green / B magenta; confirmed visible fragments only; NOT 3D truth',fill='white')
            filename=f'{spec["group"]}_cam{spec["camera"]}.jpg';im.save(args.out/'overlays'/filename,quality=95);row['overlay']='overlays/'+filename
        rows.append(row)
    # Only actual shared anatomical SURFACE endpoints may have correspondence.
    # Never equal-arclength triangulate the independent polylines.
    multiview=[];groups={s['group']:s for s in manifests['A']['files']}
    for group,spec in groups.items():
        session=spec['session'];assert session not in ('GR2','UR2','UR3')
        if session not in calibrations:
            path=ROOT/f'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/{session}/rectified_calibration.json'
            calibrations[session]=json.loads(path.read_text());input_hashes[str(path)]=sha(path)
        cal=calibrations[session]
        for tag in ('A','B'):
            for label in sorted(ENDPOINT_LABELS):
                observations={camera:p['endpoints'][label] for (t,g,camera),p in parsed.items() if t==tag and g==group and label in p['endpoints']}
                if len(observations)<3:
                    multiview.append(dict(group=group,annotator=tag,endpoint=label,status='insufficient_visible_endpoints',views=len(observations)));continue
                for held in sorted(observations):
                    sources=[c for c in sorted(observations) if c!=held]
                    fit=triangulate_observations([observations[c] for c in sources],[cal[c]['K'] for c in sources],[cal[c]['world_to_camera_mm'] for c in sources])
                    row=dict(group=group,annotator=tag,endpoint=label,held_camera=held,source_cameras=sources,valid=fit['valid'])
                    if fit['valid']:
                        pixel,depth=project_point(fit['xyz_mm'],cal[held]['K'],cal[held]['world_to_camera_mm'])
                        row['valid']=depth>0
                        if row['valid']:row.update(held_error_px=float(np.linalg.norm(pixel-observations[held])),projected_xy=pixel.tolist())
                    else:row['reason']=fit.get('reason')
                    multiview.append(row)
    summary=dict(images=len(rows),reviewed=dict(completed),reviewed_structurally_valid=dict(valid),paired_reviewed=sum('agreement' in r for r in rows),
                 comparable_visible_curves=sum(r.get('agreement',{}).get('comparable',False) for r in rows),
                 identical_reviewed_annotation_files=sum(r.get('identical_annotation_files',False) for r in rows),
                 all_reviewed_and_structurally_valid=all(valid[t]==len(rows) for t in ('A','B')),
                 training_authorized_by_this_report=False,groundtruth_exported=False)
    report=dict(summary=summary,rows=rows,multiview_endpoints=multiview,input_hashes=input_hashes,
                policy='Native1080x1200 rectified coordinates, fixedcalibration. No imaginary endpoints, no merginglabels, no pseudoGT, no acceptance fromselfreprojection. LOOonlyactualannotatedsurfaceendpoints. Sparse/internal06/07notused.',
                limitations='Doubleannotatoragreement isnotanatomicaltruth. Endpointdefinition stillneedsvisualadjudication. Twoorlessviews cannotgiveLOO. Curvefragmentswithdifferentvisiblecoveragearediagnosticnotautomaticallyfailed.')
    save(args.out/'report.json',report)
    html=['<!doctype html><meta charset="utf-8"><h1>双人背侧体表线复核</h1><p>绿色A，紫色B。只是人工一致性诊断，不导出3D真值。空白模板仍是待标注。</p><pre>'+json.dumps(summary,ensure_ascii=False,indent=2)+'</pre>']
    for row in rows:
        if 'overlay' in row:html.append(f'<h2>{row["group"]} cam{row["camera"]}</h2><img width="540" src="{row["overlay"]}">')
    (args.out/'index.html').write_text('\n'.join(html),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
