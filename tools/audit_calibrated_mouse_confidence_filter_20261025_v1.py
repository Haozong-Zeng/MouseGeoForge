"""Finite CPU-only filter diagnostic on fixed native DEV predictions; no selection."""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'output/calibrated_mouse_resume8h_20261025'
OLD=ROOT/'output/calibrated_mouse_10h_20261025'
RULES=('none','confidence10','confidence20','confidence50','edge3','confidence20_edge3','random_same_count_as_confidence20','black16','white240')
PIPELINES=[f'roi_pipeline_s{s}_{mode}_v1' for mode in ('full','roi') for s in (17,29,43)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def upstream_edge():
    tree=ast.parse((ROOT/'vggt-omega/visual_util.py').read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='depth_edge')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned-upstream-depth-edge>','exec'),ns)
    return ns['depth_edge']
def percentile_mask(conf,valid,p):
    out=np.zeros_like(valid)
    if np.any(valid):
        threshold=float(np.percentile(conf[valid],p));out=valid&(conf>=threshold)&(conf>1e-5)
    else:threshold=None
    return out,threshold
def filters(z,conf,valid,rgb,scene_index):
    masks={'none':valid.copy()};thresholds={}
    for p in (10,20,50):masks[f'confidence{p}'],thresholds[str(p)]=percentile_mask(conf,valid,p)
    edge=upstream_edge()(z,rtol=.03)
    masks['edge3']=valid&~edge
    adjusted=conf.copy();adjusted[edge]=0
    masks['confidence20_edge3'],thresholds['edge_then20']=percentile_mask(adjusted,valid,20)
    # GT-independent deterministic, exactly matched retained count, including score ties.
    positions=np.flatnonzero(valid);count=int(masks['confidence20'].sum())
    rng=np.random.default_rng(20261025+scene_index)
    random_mask=np.zeros_like(valid);random_mask.flat[rng.permutation(positions)[:count]]=True
    masks['random_same_count_as_confidence20']=random_mask
    masks['black16']=valid&(rgb.astype(np.int32).sum(-1)>=16)
    masks['white240']=valid&~(rgb>240).all(-1)
    assert tuple(masks)==RULES
    assert all(not np.any(v&~valid) for v in masks.values())
    return masks,thresholds,edge

def points(z,mask,k,e):
    y,x=np.where(mask);uv=np.stack((x,y,np.ones_like(x)),-1)
    camera=(uv@np.linalg.inv(k).T)*z[y,x,None]
    return (camera-e[:3,3])@e[:3,:3]
def cloud_stats(predicted,truth,d_pred):
    if not len(predicted):
        return dict(points=0,accuracy_mm=None,completeness_mm=None,precision={str(t):0. for t in (1,2,5)},recall={str(t):0. for t in (1,2,5)},fscore={str(t):0. for t in (1,2,5)})
    d_true=cKDTree(predicted).query(truth,workers=2)[0]
    result=dict(points=len(predicted),accuracy_mm=float(np.mean(d_pred)),completeness_mm=float(np.mean(d_true)),precision={},recall={},fscore={})
    for t in (1,2,5):
        key=str(t);p=float(np.mean(d_pred<=t));r=float(np.mean(d_true<=t))
        result['precision'][key]=p;result['recall'][key]=r;result['fscore'][key]=2*p*r/max(p+r,1e-12)
    return result

def run():
    registration=NEW/'confidence_filter_registration_v1.json'
    reg=json.loads(registration.read_text())
    for f,h in reg['source_sha256'].items():assert sha(ROOT/f)==h,(f,'changed after registration')
    out=NEW/'confidence_filter_audit_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir();started=time.monotonic()
    manifest=json.loads((OLD/'data_manifest.json').read_text());inputs={str(registration.relative_to(ROOT)):sha(registration)}
    dev=[(i,s) for i,s in enumerate(manifest['scenes']) if s['split']=='dev'];assert len(dev)==144
    truth_cache={}
    for i,s in dev:
        truth=[];body=[];depth=[];rgb=[]
        for v,source in enumerate(s['views']):
            f=ROOT/source['truth'];assert sha(f)==source['truth_sha256'];inputs[str(f.relative_to(ROOT))]=sha(f)
            img=ROOT/source['image'];assert sha(img)==source['image_sha256'];inputs[str(img.relative_to(ROOT))]=sha(img)
            with np.load(f,allow_pickle=False) as item:
                z=item['depth'];b=item['body'].astype(bool);truth.append(points(z,b,item['K'],item['E']));body.append(b);depth.append(z)
            rgb.append(np.asarray(Image.open(img).convert('RGB')))
        truth_cache[i]=(np.concatenate(truth),np.stack(body),np.stack(depth),np.stack(rgb))
    reports={}
    for name in PIPELINES:
        folder=NEW/name;reference=json.loads((folder/'evaluation.json').read_text());inputs[str((folder/'evaluation.json').relative_to(ROOT))]=sha(folder/'evaluation.json')
        scenes=[];max_baseline_error=0.
        for count,(index,scene) in enumerate(dev,1):
            truth,body,target,rgb=truth_cache[index];z=[];conf=[];valid=[];world=[]
            for v in range(3):
                f=folder/f'scene{index:06d}'/f'view{v}.npz';inputs[str(f.relative_to(ROOT))]=sha(f)
                with np.load(f,allow_pickle=False) as a:
                    d=a['depth_mm'];c=a['plane_probability_max'];m=a['mask']&np.isfinite(d)&(d>0)&np.isfinite(c)
                    z.append(d);conf.append(c);valid.append(m);world.append(points(d,m,a['K'],a['E']))
            z=np.stack(z);conf=np.stack(conf);valid=np.stack(valid);pred=np.concatenate(world)
            masks,thresholds,edge=filters(z,conf,valid,rgb,index)
            d_pred=cKDTree(truth).query(pred,workers=2)[0]
            result={}
            for rule,mask in masks.items():
                keep=mask[valid];cloud=cloud_stats(pred[keep],truth,d_pred[keep]);views=[]
                for v in range(3):
                    overlap=body[v]&mask[v];res=z[v][overlap]-target[v][overlap]
                    views.append(dict(gt_pixels=int(body[v].sum()),retained_pixels=int(mask[v].sum()),gt_depth_coverage=float(overlap.sum()/max(1,body[v].sum())),rmse_mm_on_overlap=float(np.sqrt(np.mean(res.astype(np.float64)**2))) if len(res) else None,overlap_pixels=int(overlap.sum())))
                result[rule]=dict(cloud=cloud,views=views,retained_fraction=float(mask.sum()/max(1,valid.sum())))
            reference_scene=reference['scenes'][count-1];assert reference_scene['id']==scene['id']
            error=max(abs(result['none']['cloud']['fscore'][str(t)]-reference_scene['pointcloud']['fscore'][str(t)]) for t in (1,2,5));max_baseline_error=max(max_baseline_error,error);assert error<1e-10
            scenes.append(dict(index=index,id=scene['id'],source=scene['source'],thresholds=thresholds,original_predicted_points=int(valid.sum()),edge_rejected_gt_body_pixels=int((edge&valid&body).sum()),rules=result))
            if count%24==0:print(name,count,144,'seconds',round(time.monotonic()-started,1),flush=True)
        summary={}
        for rule in RULES:
            clouds=[s['rules'][rule]['cloud'] for s in scenes];views=[v for s in scenes for v in s['rules'][rule]['views']]
            summary[rule]=dict(scene_count=144,view_count=432,nonempty_overlap_views=sum(v['rmse_mm_on_overlap'] is not None for v in views),retained_fraction=float(sum(c['points'] for c in clouds)/sum(s['original_predicted_points'] for s in scenes)),gt_depth_coverage=float(np.mean([v['gt_depth_coverage'] for v in views])),rmse_mm_on_overlap=float(np.mean([v['rmse_mm_on_overlap'] for v in views if v['rmse_mm_on_overlap'] is not None])),precision={str(t):float(np.mean([c['precision'][str(t)] for c in clouds])) for t in (1,2,5)},recall={str(t):float(np.mean([c['recall'][str(t)] for c in clouds])) for t in (1,2,5)},fscore={str(t):float(np.mean([c['fscore'][str(t)] for c in clouds])) for t in (1,2,5)})
        reports[name]=dict(summary=summary,baseline_fscore_replay_max_abs_error=max_baseline_error,scenes=scenes)
        write(out/(name+'.json'),reports[name])
    aggregate={}
    for mode in ('full','roi'):
        aggregate[mode]={}
        for rule in RULES:
            rows=[reports[f'roi_pipeline_s{s}_{mode}_v1']['summary'][rule] for s in (17,29,43)]
            aggregate[mode][rule]={k:float(np.mean([r[k] for r in rows])) for k in ('retained_fraction','gt_depth_coverage','rmse_mm_on_overlap')}
            for metric in ('precision','recall','fscore'):aggregate[mode][rule][metric]={str(t):float(np.mean([r[metric][str(t)] for r in rows])) for t in (1,2,5)}
    for f,h in inputs.items():assert sha(ROOT/f)==h,(f,'changed during audit')
    result=dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=sha(registration),aggregate_equal_seed=aggregate,pipelines={k:dict(summary=v['summary'],baseline_fscore_replay_max_abs_error=v['baseline_fscore_replay_max_abs_error']) for k,v in reports.items()},input_sha256=inputs,limitations=reg['limitations'],seconds=time.monotonic()-started)
    write(out/'completion.json',result)
    print(json.dumps(aggregate),flush=True)
if __name__=='__main__':run()
