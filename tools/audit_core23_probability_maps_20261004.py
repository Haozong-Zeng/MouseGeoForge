"""Diagnose soft expectation versus modes; labels used for evaluation only."""
import collections,datetime as dt,fcntl,json,os,sys,math,shutil,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from mousegeoforge.models.core23_spatial_readout import Core23SpatialReadout
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004'
SOURCE=CYCLE/'features_expanded_all_v1'
TRAIN=CYCLE/'readout_expanded_all_soft_box_extended4800_v1'
OUT=CYCLE/'probability_audit_v1'


def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='auditing_spatial_probability_modes',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    manifest=json.loads((SOURCE/'manifest.json').read_text());rows=[];maps=[]
    for record in manifest['records']:
        assert fingerprint(ROOT/record['path'])==record['sha256']
        p=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
        mm=F.interpolate(p['memory'].float(),(48,48),mode='bilinear',align_corners=False)
        for o in p['observations']:
            rows.append(dict(o,pixels=torch.tensor(p['image_hw'][::-1],dtype=torch.float32)))
            maps.append(mm[o['metadata']['view_index']])
    data={k:torch.stack([o[k] for o in rows]).cuda() for k in ['pose','coarse_xy','gt_xy','pixels','valid']};memory=torch.stack(maps).cuda();del maps
    for i,o in enumerate(rows):
        if o['metadata']['instance_count']>1:data['valid'][i,7:14]=False
    pm=json.loads((SOURCE/'prior_manifest.json').read_text());assert fingerprint(ROOT/pm['path'])==pm['sha256']
    boxes=torch.load(ROOT/pm['path'],map_location='cpu',weights_only=False)['boxes_cxcywh'].cuda()
    with np.load(TRAIN/'predictions.npz') as a:reference={k:a[k] for k in a.files if k.startswith('spatial')}
    y,x=torch.meshgrid((torch.arange(48,device='cuda')+.5)/48,(torch.arange(48,device='cuda')+.5)/48,indexing='ij');grid=torch.stack([x,y],-1).reshape(-1,2)
    selected={61,188,303,313};visuals={};results=[];details=[]
    for seed in [17,29,43]:
        model=Core23SpatialReadout('spatial').cuda().eval();cp=TRAIN/f'spatial_seed{seed}_latest.pt'
        model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['state'],strict=True)
        agg=collections.defaultdict(lambda:collections.defaultdict(list));maxdelta=0.
        for start in range(0,len(rows),16):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            ids=list(range(start,min(start+16,len(rows))))
            with torch.no_grad():
                xy,prob=model(data['pose'][ids],data['coarse_xy'][ids],memory[ids],True,predicted_boxes=boxes[ids]);p=prob.flatten(2)
                peak=p.argmax(-1);mode=grid[peak]
                delta=(grid[None,None]-mode[:,:,None])*48
                local=p*(delta.abs().amax(-1)<=2);local=local/local.sum(-1,keepdim=True).clamp_min(1e-12)
                local_xy=torch.einsum('nkp,pd->nkd',local,grid)
                ent=-(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(2304)
                radial=((grid[None,None]-data['gt_xy'][ids,:,None])*data['pixels'][ids,None,None]).norm(dim=-1)
                mass16=(p*(radial<=16)).sum(-1)
                scale=data['pixels'][ids,None];gt=data['gt_xy'][ids]
                values={'expectation_px':((xy-gt)*scale).norm(dim=-1),'argmax_px':((mode-gt)*scale).norm(dim=-1),'local_mode_px':((local_xy-gt)*scale).norm(dim=-1),'entropy_normalized':ent,'peak_probability':p.max(-1).values,'gt16px_mass':mass16}
            diff=float((xy.cpu()-torch.tensor(reference[f'spatial_seed{seed}'][ids])).abs().max());maxdelta=max(maxdelta,diff);assert diff<2e-5
            for j,i in enumerate(ids):
                meta=rows[i]['metadata'];r=meta['record'];kind='real' if r['kind']=='real' else 'synthetic';valid=data['valid'][i].cpu();group=kind+'_'+r['role']
                for k in range(23):
                    if not valid[k]:continue
                    tags=[group+'/all',group+f'/point{k+1:02d}',group+('/dual' if meta['instance_count']>1 else '/single')]
                    if k in [0,14,15,16,17]:tags.append(group+'/nose_feet')
                    if k in [14,15,16,17]:tags.append(group+'/feet')
                    if k in [5,6]:tags.append(group+'/back')
                    for tag in tags:
                        for name,value in values.items():agg[tag][name].append(float(value[j,k]))
                if seed==17 and i in selected:
                    visuals[i]=dict(prob=prob[j].cpu(),xy=xy[j].cpu(),mode=mode[j].cpu(),local=local_xy[j].cpu())
                    details.append(dict(observation=i,metadata=meta,metrics={n:v[j].cpu().tolist() for n,v in values.items()},valid=valid.tolist()))
            write_json(OUT/'status.json',dict(stage='auditing',pid=os.getpid(),seed=seed,completed=min(start+16,len(rows)),total=len(rows)))
        results.append(dict(seed=seed,checkpoint_sha256=fingerprint(cp),max_prediction_delta=maxdelta,metrics={group:{k:stats(v) for k,v in d.items()} for group,d in agg.items()}))
    write_json(OUT/'metrics.json',dict(results=results,selected_details=details,semantics='Argmax and mode-local expectation are prediction-only alternative decoders, not new GT. GT16px mass is diagnostic only. Human sparse observations are not exact anatomy; no labels changed.'))
    page=['<!doctype html><meta charset="utf-8"><title>Core23 probability audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Probability map audit, seed17 at4800steps</h1><p>Each row: point01,15,16,17,18. Yellow cross=observed label; cyan circle=expectation; magenta square=argmax. Red overlay=relative heatmap strength (per-point scaled, not calibrated confidence). Fixed earlier failure cases; no GT crops.</p>']
    for i,v in visuals.items():
        o=rows[i];m=o['metadata'];r=m['record'];cache=torch.load(ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt',map_location='cpu',weights_only=False)
        pixels=(cache['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=pixels.shape[:2];pw=300;ph=round(h*pw/w);im=Image.fromarray(pixels).resize((pw,ph))
        canvas=Image.new('RGB',(5*pw,ph+35),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,k in enumerate([0,14,15,16,17]):
            heat=v['prob'][k].numpy();heat=heat/max(float(heat.max()),1e-12);alpha=np.asarray(Image.fromarray((heat*150).astype('uint8')).resize((pw,ph),Image.Resampling.BILINEAR))
            red=Image.new('RGB',(pw,ph),(255,25,0));overlay=Image.composite(red,im,Image.fromarray(alpha))
            canvas.paste(overlay,(col*pw,35));draw.text((col*pw+5,5),f'point{k+1:02d}',fill='white')
            for name,color in [('xy','#44ddff'),('mode','#ff55ff')]:
                xx,yy=v[name][k].numpy()*[pw,ph]+[col*pw,35]
                if name=='xy':draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline=color,width=2)
                else:draw.rectangle((xx-3,yy-3,xx+3,yy+3),outline=color,width=2)
            if o['valid'][k]:
                gx,gy=o['gt_xy'][k].numpy()*[pw,ph]+[col*pw,35];draw.line((gx-4,gy,gx+4,gy),fill='#ffe45c',width=2);draw.line((gx,gy-4,gx,gy+4),fill='#ffe45c',width=2)
        name=f'heatmap_{i:03d}.jpg';canvas.save(OUT/name,quality=95);page.append('<h2>'+html.escape(str(r))+'</h2><img src="'+name+'">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid()))
    state.update(stage='probability_audit_ready',active_job=None,active_pid=None,next='Read probability_audit_v1/metrics.json and view heatmap061/188/303/313. Decide whether expectation averaging or wrong modes dominate before changing loss/architecture. 4800-step mean improvements modest; avoid another blind extension.')
    state['completed'].append('probability_expectation_vs_modes_3seed_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
