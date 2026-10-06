"""Same-camera manual back-vs-nose/paw consistency control, not calibration fit."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.multiview_point_audit import project_point,triangulate_observations
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'manual_multiview_controls_v1'


def main():
    assert not OUT.exists();OUT.mkdir();(OUT/'visual_review').mkdir();torch.set_num_threads(2)
    saved=torch.load(CYCLE/'expanded_real_point_features_v1/real_inputs.pt',map_location='cpu',weights_only=False);meta=saved['metadata'];groups=collections.defaultdict(list)
    for i,m in enumerate(meta):assert m['session'] not in ('GR2','UR2','UR3');groups[(m['session'],m['frame'],m['role'])].append(i)
    records={r['key']:r for r in json.loads((CYCLE/'training_v1/pilot/records.json').read_text()) if r['kind']=='real'};records.update({r['key']:r for r in json.loads((CYCLE/'expanded_real_point_features_v1/records.json').read_text())})
    affine=np.asarray(json.loads((CYCLE/'training_v1/protocol.json').read_text())['real_affine']);cal={};uv=[];valid=[];hashes={};cache={}
    for m in meta:
        path=ROOT/m['scene']/'surface_aux.npz'
        if str(path) not in cache:
            with np.load(path) as z:cache[str(path)]={k:z[k] for k in ['view_ids','uv','observed']}
            hashes[str(path.relative_to(ROOT))]=fingerprint(path)
        z=cache[str(path)];v=z['view_ids'].tolist().index(m['view']);uv.append(z['uv'][v,0]);valid.append(z['observed'][v,0])
        if m['session'] not in cal:
            p=ROOT/f'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/{m["session"]}/rectified_calibration.json';cal[m['session']]=json.loads(p.read_text());hashes[str(p.relative_to(ROOT))]=fingerprint(p)
    uv=np.asarray(uv);valid=np.asarray(valid)&np.isfinite(uv).all(-1);ids_of_interest=[0,5,6,14,15,16,17];rows=[];pairs=[]
    def fit_case(session,held,sources,k):
        ks=[];es=[]
        for i in sources:
            c=cal[session][chr(65+meta[i]['view'])];ks.append(affine@np.asarray(c['K']));es.append(np.asarray(c['world_to_camera_mm']))
        fit=triangulate_observations(uv[sources,k],ks,es)
        if not fit['valid']:return dict(valid=False,reason=fit.get('reason',''))
        c=cal[session][chr(65+meta[held]['view'])];pixel,depth=project_point(fit['xyz_mm'],affine@np.asarray(c['K']),np.asarray(c['world_to_camera_mm']))
        if depth<=0:return dict(valid=False,reason='held_cheirality')
        return dict(valid=True,error_px=float(np.linalg.norm(pixel-uv[held,k])),projected_xy=pixel.tolist(),source_error_px=fit['fit_source_reprojection_px'])
    for (session,frame,role),ids in sorted(groups.items()):
        for held in ids:
            hv=meta[held]['view'];other=[i for i in ids if i!=held and hv not in records[meta[i]['cache_key']]['views']]
            for k in ids_of_interest:
                if not valid[held,k]:continue
                sources=[i for i in other if valid[i,k]]
                if len(sources)>=2:rows.append(dict(session=session,frame=frame,role=role,held=held,held_view=hv,point=k+1,source_views=[meta[i]['view'] for i in sources],**fit_case(session,held,sources,k)))
            for back in [5,6]:
                if not valid[held,back]:continue
                for control in [0,14,15,16,17]:
                    if not valid[held,control]:continue
                    sources=[i for i in other if valid[i,back] and valid[i,control]]
                    if len(sources)<2:continue
                    br=fit_case(session,held,sources,back);cr=fit_case(session,held,sources,control)
                    pairs.append(dict(session=session,frame=frame,role=role,held=held,held_view=hv,back_point=back+1,control_point=control+1,source_views=[meta[i]['view'] for i in sources],back=br,control=cr))
    summary=[]
    for role in ['train','validation']:
        for k in ids_of_interest:
            rr=[r for r in rows if r['role']==role and r['point']==k+1];accepted=[r['error_px'] for r in rr if r['valid']]
            if accepted:summary.append(dict(role=role,point=k+1,eligible=len(rr),accepted=len(accepted),mean_px=float(np.mean(accepted)),median_px=float(np.median(accepted))))
    matched=[]
    for role in ['train','validation']:
        for control in [1,15,16,17,18]:
            pp=[p for p in pairs if p['role']==role and p['control_point']==control and p['back']['valid'] and p['control']['valid']]
            if pp:matched.append(dict(role=role,control=control,n=len(pp),back_mean=float(np.mean([p['back']['error_px'] for p in pp])),control_mean=float(np.mean([p['control']['error_px'] for p in pp])),mean_paired_difference=float(np.mean([p['back']['error_px']-p['control']['error_px'] for p in pp]))))
    eligible=[p for p in pairs if p['role']=='validation' and p['back']['valid'] and p['control']['valid']]
    eligible.sort(key=lambda p:p['back']['error_px']-p['control']['error_px']);chosen=eligible[:6]+eligible[-6:];panels=[]
    for index,p in enumerate(chosen):
        m=meta[p['held']]
        with Image.open(ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png') as im:im=im.convert('RGB').resize((352,384))
        canvas=Image.new('RGB',(352,430),(22,22,28));canvas.paste(im,(0,46));draw=ImageDraw.Draw(canvas)
        draw.text((3,3),f'back{p["back_point"]}: {p["back"]["error_px"]:.1f}px',fill='#ffe454');draw.text((3,22),f'control{p["control_point"]}: {p["control"]["error_px"]:.1f}px',fill='#55ff55')
        for key,k,color,crosscolor in [('back',p['back_point']-1,'#ffe454','#ff55ff'),('control',p['control_point']-1,'#55ff55','#39dfff')]:
            x,y=uv[p['held'],k]/[704,768]*[352,384]+[0,46];draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
            x,y=np.asarray(p[key]['projected_xy'])/[704,768]*[352,384]+[0,46];draw.line((x-4,y,x+4,y),fill=crosscolor,width=2);draw.line((x,y-4,x,y+4),fill=crosscolor,width=2)
        name=f'{index:03d}.jpg';canvas.save(OUT/'visual_review'/name,quality=95);panels.append(dict(file=name,**p))
    report=dict(summary=summary,matched_camera_controls=matched,rows=rows,pairs=pairs,visuals=panels,source_hashes=hashes,
        policy='Manual labels only. Same context-independent heldcamera rule asmodelaudit. Matchedpairs use EXACTsamesourcecameras andheldframe forback/control. No calibrationoptimization/labelmodification or3DGT.',
        limitations='Nose/paws alsomayhaveannotation/occlusion/leftrightambiguity. Differentimagepositions/refraction/geometry canaffecterrors, so thiscannotuniquelyattributeerror tobackdefinitions. Repeatedpairs correlated, notindependentsamples. Selectedvisuals showboth extremesofpairedgap, explicitlydiagnostic notrandomprevalence.')
    write_json(OUT/'results.json',report)
    html=['<!doctype html><meta charset="utf-8"><title>Manual back/control consistency</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Manual back versus nose/paw: SAME camera subsets</h1><p>Backmanual=yellow, backprojection=magenta; controlmanual=green, controlprojection=cyan. Bothuseexactsameothercameras; heldcameraimagecontextexcluded. Thesearepaired-gap extremesinBOTHdirections, notrandomprevalence. Controlsalsocanbeambiguous; cannotuniquelyblamecalibrationorlabels. Noannotationchanges.</p>']
    for p in panels:html.append(f'<h2>{p["session"]} frame{p["frame"]}, held{p["held_view"]}, sources{p["source_views"]}</h2><img src="{p["file"]}">')
    (OUT/'visual_review/index.html').write_text('\n'.join(html));print(json.dumps(dict(summary=[s for s in summary if s['role']=='validation'],matched=[s for s in matched if s['role']=='validation']),indent=2))


if __name__=='__main__':main()
