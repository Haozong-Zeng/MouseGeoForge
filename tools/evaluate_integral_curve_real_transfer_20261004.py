"""Synthetic-only surface integral heads on frozen real DEVELOPMENT inputs."""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from mousegeoforge.models.surface_curve_integral import SurfaceCurveIntegralProbe
from mousegeoforge.surface_curve_supervision import cubic_bspline,points_to_polyline
from probe_diverse_curve_features_20261003 import FeatureProbe
from review_sparse_real_curve_adaptation_20261003 import crossings
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'integral_curve_real_transfer_v1'


def main():
    assert not OUT.exists();lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();OUT.mkdir();(OUT/'visual_review').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');source=CYCLE/'real_curve_adaptation_inputs_v1/frozen_inputs.pt';saved=torch.load(source,map_location='cpu',weights_only=False);meta=saved['metadata'];data={k:v.cuda() for k,v in saved['tensors'].items()}
    other=torch.load(CYCLE/'back_point_probe_inputs_v1/real_inputs.pt',map_location='cpu',weights_only=False)
    assert saved['metadata']==other['metadata'] and torch.equal(saved['tensors']['pose'],other['tensors']['pose'])
    assert all(m['session'] not in ('GR2','UR2','UR3') for m in meta)
    maps=[None]*len(meta);manifest=json.loads((CYCLE/'real_point_spatial_memory_v1/manifest.json').read_text())
    for record in manifest['records']:
        assert fingerprint(ROOT/record['path'])==record['sha256'];cache=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
        memory=F.interpolate(cache['memory'].float(),(48,48),mode='bilinear',align_corners=False)
        for b in record['bindings']:assert maps[b['observation']] is None;maps[b['observation']]=memory[b['view_index']]
    assert all(m is not None for m in maps);memory=torch.stack(maps).cuda();arrays={};hashes={}
    with torch.no_grad():
        for seed in [17,29,43]:
            path=CYCLE/f'curve_data_scaling_v1/expanded_seed{seed}.pt';base=FeatureProbe(False).cuda().eval();base.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);hashes[f'base{seed}']=fingerprint(path)
            controls=data['boxes'][:,:2,None].transpose(-1,-2)+(data['base_logits'].sigmoid()-.5)*data['boxes'][:,2:,None].transpose(-1,-2)+.5*base.delta(data['pose']).reshape(-1,6,2).tanh()
            arrays[f'base_seed{seed}']=cubic_bspline(controls,33).cpu().numpy()
            for mode in ['global','spatial']:
                assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
                path=CYCLE/f'integral_surface_curve_probe_v1/{mode}_seed{seed}.pt';model=SurfaceCurveIntegralProbe(mode).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state']);hashes[f'{mode}{seed}']=fingerprint(path)
                arrays[f'{mode}_seed{seed}']=model(data['pose'],memory,controls).cpu().numpy();status('evaluating',seed=seed,mode=mode)
    results=[];rows=[]
    for arm,array in arrays.items():
        for role in ['train','validation']:
            values=collections.defaultdict(list)
            for i,m in enumerate(meta):
                if m['role']!=role:continue
                pred=torch.from_numpy(array[i]).cuda()*data['pixels'][i];good=data['observed_valid'][i];gt=data['observed_xy'][i]*data['pixels'][i]
                metrics={}
                if good.any():
                    dist,param=points_to_polyline(gt[good],pred,torch.ones(33,device='cuda',dtype=torch.bool));metrics['observed_point_to_curve_px']=dist.sqrt().tolist()
                    if good.all():metrics['observed_order_reversal']=[float(param[0]>param[1])]
                metrics['self_crossing']=[float(crossings(array[i])>0)];metrics['curve_length_px']=[float((pred[1:]-pred[:-1]).norm(dim=-1).sum())]
                if data['mask_valid'][i]:
                    xy=torch.from_numpy(array[i]).cuda()*pred.new_tensor([176,192]);xy=xy.round().long();inside=(xy[:,0]>=0)&(xy[:,0]<176)&(xy[:,1]>=0)&(xy[:,1]<192)
                    metrics['mask_membership_approx']=[float((data['mask_small'][i][xy[:,1].clamp(0,191),xy[:,0].clamp(0,175)]&inside).float().mean())]
                for k,v in metrics.items():values[k].extend(v)
                rows.append(dict(arm=arm,observation=i,session=m['session'],frame=m['frame'],view=m['view'],role=role,metrics=metrics))
            results.append(dict(arm=arm,role=role,metrics={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()}))
    chosen={(r['session'],r['frame'],r['view']) for r in json.loads((CYCLE/'real_curve_transfer_v1/results.json').read_text())['visuals']};panels=[]
    for i,m in enumerate(meta):
        if (m['session'],m['frame'],m['view']) not in chosen:continue
        with Image.open(ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png') as im:im=im.convert('RGB').resize((352,384))
        canvas=Image.new('RGB',(1056,412),(22,22,28));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['base_seed17','global_seed17','spatial_seed17']):
            canvas.paste(im,(352*col,28));draw.text((352*col+5,5),arm,fill='white');line=arrays[arm][i]*[352,384]+[352*col,28];draw.line([tuple(p) for p in line],fill='#39dfff',width=2)
            for k in range(2):
                if saved['tensors']['observed_valid'][i,k]:
                    x,y=saved['tensors']['observed_xy'][i,k].numpy()*[352,384]+[352*col,28];draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffe454');draw.text((x+4,y-8),str(k+6),fill='#ffe454')
        name=f'{i:03d}.jpg';canvas.save(OUT/'visual_review'/name,quality=95);panels.append(dict(file=name,observation=i,**m))
    np.savez_compressed(OUT/'predictions.npz',**arrays);write_json(OUT/'results.json',dict(summary=results,rows=rows,visuals=panels,no_training=True,no_pseudolabels=True,checkpoint_hashes=hashes,source_sha256=fingerprint(source)))
    html=['<!doctype html><meta charset="utf-8"><title>Synthetic-only integral curve transfer</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Synthetic-only fullcurve transfer to real DEVELOPMENT</h1><p>All curveheads trained on synthetic only, no realadaptation here. Yellow=manual06/07observations, NOT fullcurveGT or endpoints. Cyan=predictedcurve. Samefrozen features/querybindings/pairedcontexts. No pseudoGT created. Maskcontainment and weakpointdistances do not prove surfaceanatomy.</p>']
    for r in panels:html.append(f'<h2>{r["role"]}: {r["session"]} frame{r["frame"]} view{r["view"]}</h2><img src="{r["file"]}">')
    (OUT/'visual_review/index.html').write_text('\n'.join(html));status('complete');print(json.dumps([r for r in results if r['role']=='validation'],indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
