"""Leave-one-camera-coordinate/image-context-out diagnostic; never create3DGT."""
import collections,datetime as dt,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.multiview_point_audit import project_point,triangulate_observations
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'real_point_multiview_audit_v1'


def main():
    assert (CYCLE/'real_point_heatmap_v1/results.json').exists() and not OUT.exists();OUT.mkdir();(OUT/'visual_review').mkdir();torch.set_num_threads(2)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');path=CYCLE/'expanded_real_point_features_v1/real_inputs.pt';saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['metadata'];d={k:v.numpy() for k,v in saved['tensors'].items()}
    assert all(m['session'] not in ('GR2','UR2','UR3') for m in meta)
    records={r['key']:r for r in json.loads((CYCLE/'training_v1/pilot/records.json').read_text()) if r['kind']=='real'}
    records.update({r['key']:r for r in json.loads((CYCLE/'expanded_real_point_features_v1/records.json').read_text())})
    groups=collections.defaultdict(list)
    for i,m in enumerate(meta):groups[(m['session'],m['frame'],m['role'])].append(i)
    affine=np.asarray(json.loads((CYCLE/'training_v1/protocol.json').read_text())['real_affine']);cal={};hashes={}
    for session in sorted({m['session'] for m in meta}):
        p=ROOT/f'dataset/PyMouseStudio/annotation_returns/20260930_full_v2/calibration/{session}/rectified_calibration.json';cal[session]=json.loads(p.read_text());hashes[session]=fingerprint(p)
        for c in cal[session].values():
            assert c['size_wh']==[1080,1200] and not np.any(c['distortion']);R=np.asarray(c['world_to_camera_mm'])[:3,:3];assert np.allclose(R@R.T,np.eye(3),atol=1e-3)
    arrays={'manual':d['gt_xy']}
    for folder,prefixes in [('real_point_data_scaling_v1',['expanded']),('real_point_heatmap_v1',['coordinate','heatmap'])]:
        with np.load(CYCLE/f'{folder}/predictions.npz') as z:
            for prefix in prefixes:
                for seed in [17,29,43]:arrays[f'{prefix}_seed{seed}']=z[f'{prefix}_seed{seed}__real']
    assert all(a.shape==d['gt_xy'].shape for a in arrays.values())
    write_json(OUT/'protocol.json',dict(scope='OnlyexistingrealTRAIN/developmentVAL, noholdout. TRAINdiagnostic notgeneralization. Manual06/07are uncertain surfaceanchors, notguaranteedcommon3Dlandmarks.',
        holdout='For each observed targetcamera/point, dropitscoordinate AND any sourceprediction whose pairedfeaturecontext includedthatcamera. Samefilteredsourceviews for manualbaseline and allmodelarms. Image-context exclusion prevents heldimage entering other predictions throughpairedVGGTfeatures.',
        eligibility='Atleast2otherannotated pointviews aftercontextfilter; no invisible/missingviewimputation; no confidenceweightingorvalidationthresholdselection.',
        method='NormalizedDLT then Huber8px leastsquares, max100evals; rejectmaxacutebearingangle<5degrees, nonconvergence, nonpositivecameradepth. Parametersfixedbeforeevaluating.',
        metrics='Held projection versus existingmanual2Dcoordinate; directheldprediction error and selfconsistency diagnostic separate. Sourcefitreprojection isNOT independenttest. 3Dcoordinates for audit ONLY, notgroundtruth.',
        camera_units='World-to-camera translation mm; K letterboxed bythe exact704x768affine, distortionalreadyzero.',
        calibration_hashes=hashes,input_sha256=fingerprint(path),limitations='8VALgroups/58manualpoints only, samecamera set anddevelopmentreuse. Backanchorsmaynotrepresentconsistent3Dmaterialpoints acrosshumanviews. No modelselection/labelupdate/3DGTexport.'))
    rows=[];cases=[];skipped=collections.Counter()
    for n,((session,frame,role),ids) in enumerate(sorted(groups.items())):
        for held in ids:
            hv=meta[held]['view']
            for point in range(2):
                if not d['valid'][held,point]:continue
                source=[i for i in ids if i!=held and d['valid'][i,point] and hv not in records[meta[i]['cache_key']]['views']]
                if len(source)<2:skipped[role+'_insufficient_context_independent_views']+=1;continue
                kk=[];ee=[]
                for i in source:
                    c=cal[session][chr(65+meta[i]['view'])];kk.append(affine@np.asarray(c['K']));ee.append(np.asarray(c['world_to_camera_mm']))
                ch=cal[session][chr(65+hv)];hk=affine@np.asarray(ch['K']);he=np.asarray(ch['world_to_camera_mm']);target=d['gt_xy'][held,point]*d['pixels'][held]
                case=dict(held_observation=held,point=point+6,session=session,frame=frame,role=role,held_view=hv,source_views=[meta[i]['view'] for i in source],source_observations=source)
                cases.append(case)
                for arm,a in arrays.items():
                    uv=a[source,point]*d['pixels'][source];fit=triangulate_observations(uv,kk,ee);row=dict(arm=arm,**case,fit=fit,held_target=target.tolist(),direct_held_xy=(a[held,point]*d['pixels'][held]).tolist())
                    row['direct_held_error_px']=float(np.linalg.norm(a[held,point]*d['pixels'][held]-target))
                    if fit['valid']:
                        pixel,depth=project_point(fit['xyz_mm'],hk,he)
                        if depth<=0 or not np.isfinite(pixel).all():fit.update(valid=False,reason='held_cheirality')
                        else:row.update(held_projected_xy=pixel.tolist(),held_reprojection_error_px=float(np.linalg.norm(pixel-target)),held_prediction_consistency_px=float(np.linalg.norm(pixel-a[held,point]*d['pixels'][held])))
                    rows.append(row)
        if (n+1)%20==0:status('auditing',completed_groups=n+1,total_groups=len(groups))
    summaries=[]
    for role in ['train','validation']:
        role_rows=[r for r in rows if r['role']==role];bycase=collections.defaultdict(list)
        for r in role_rows:bycase[(r['held_observation'],r['point'])].append(r)
        common={k for k,rs in bycase.items() if all(r['fit']['valid'] for r in rs)}
        for arm in arrays:
            rr=[r for r in role_rows if r['arm']==arm];good=[r for r in rr if r['fit']['valid']];matched=[r for r in good if (r['held_observation'],r['point']) in common]
            def metrics(items):
                return {k:dict(n=len(items),mean=float(np.mean([r[k] for r in items])),median=float(np.median([r[k] for r in items])),p90=float(np.quantile([r[k] for r in items],.9))) for k in ['held_reprojection_error_px','direct_held_error_px','held_prediction_consistency_px']} if items else {}
            summaries.append(dict(role=role,arm=arm,eligible=len(rr),accepted=len(good),rejected=dict(collections.Counter(r['fit'].get('reason','') for r in rr if not r['fit']['valid'])),all_accepted=metrics(good),all_method_common=metrics(matched)))
    chosen={}
    for c in cases:
        if c['role']=='validation':chosen.setdefault((c['session'],c['frame'],c['point']),c)
    rowmap={(r['arm'],r['held_observation'],r['point']):r for r in rows};panels=[];arms=['manual','expanded_seed17','coordinate_seed17','heatmap_seed17']
    for index,c in enumerate(chosen.values()):
        i=c['held_observation'];m=meta[i]
        with Image.open(ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png') as im:im=im.convert('RGB').resize((320,349))
        canvas=Image.new('RGB',(1280,395),(22,22,28));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(arms):
            r=rowmap[(arm,i,c['point'])];canvas.paste(im,(320*col,46));draw.text((320*col+3,3),arm,fill='white')
            def screen(p):return np.asarray(p)/d['pixels'][i]*[320,349]+[320*col,46]
            gx,gy=screen(r['held_target']);draw.ellipse((gx-3,gy-3,gx+3,gy+3),fill='#ffe454')
            px,py=screen(r['direct_held_xy']);draw.ellipse((px-4,py-4,px+4,py+4),outline='#39dfff',width=2)
            if r['fit']['valid']:
                x,y=screen(r['held_projected_xy']);draw.line((x-4,y,x+4,y),fill='#ff55ff',width=2);draw.line((x,y-4,x,y+4),fill='#ff55ff',width=2)
                draw.text((320*col+3,22),f'held error: {r["held_reprojection_error_px"]:.1f}px',fill='white')
            else:draw.text((320*col+3,22),'REJECTED: '+r['fit'].get('reason',''),fill='#ff8888')
        name=f'{index:03d}.jpg';canvas.save(OUT/'visual_review'/name,quality=94);panels.append(dict(file=name,**c))
    write_json(OUT/'results.json',dict(summary=summaries,skipped=dict(skipped),rows=rows,visuals=panels,no_ground_truth_created=True))
    html=['<!doctype html><meta charset="utf-8"><title>Held-camera consistency audit</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Held-camera/image-context-out audit</h1><p>Yellow=heldmanualpoint; cyan=directprediction; magenta=reprojection fromOTHERobservedcameras. Manualcolumn triangulatesothermanualpoints. Sourceswhosepairedfeatures sawheldcamera areexcluded. No missing-pointimputation, no new3DGT. Manualbackpointscannotbeassumed anatomicallyidentical acrossviews. Trainingfit isnotindependentaccuracy. Thesearedevelopmentvalidationframes,notpristineexternaltest.</p>']
    for r in panels:html.append(f'<h2>{r["session"]} frame{r["frame"]}, point{r["point"]}, held{r["held_view"]}, sources{r["source_views"]}</h2><img src="{r["file"]}">')
    (OUT/'visual_review/index.html').write_text('\n'.join(html));status('complete');print(json.dumps([r for r in summaries if r['role']=='validation'],indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
