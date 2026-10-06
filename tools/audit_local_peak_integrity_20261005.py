"""New-cycle frozen candidate audit; FIT/CAL only, no training or DEV access."""
import os, sys, json, datetime as dt, fcntl, time, traceback, shutil
os.environ['DLClight']='True'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone, load_rgb, sha, digest
from train_mixed_local_backbone_20261005 import buffers_digest, atomic_save
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import fixed_top_peaks
from mousegeoforge.models.regional_peak_candidates import regional_top_peaks
C=ROOT/'output/local_grouping_cycle_20261005'
OLD=ROOT/'output/singleview_fusion_cycle_20261005'
INPUT=OLD/'mixed_local_rgb_inputs_v1'
OUT=C/'peak_integrity_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005/peak_integrity_v1')

def now():return dt.datetime.now(dt.timezone.utc)
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,x):
    temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)

def score(p,r,feet=False):
    mask=r['eligible'].clone()
    if feet:mask[:,0]=False
    near=[];assigned=[];top1=[];missing=0
    diagonal=float(r['pixels'].square().sum().sqrt())
    for k in range(5):
        targets=r['target'][mask[:,k],k]*r['pixels']
        if not len(targets):continue
        cost=torch.cdist(targets,p['xy'][k].cpu()*r['pixels'])
        available=p['available'][k].cpu()
        cost[:,~available]=diagonal  # explicit maximum image-distance penalty
        near.extend(cost.min(1).values.tolist())
        ii,jj=linear_sum_assignment(cost.numpy());assigned.extend(cost[ii,jj].tolist())
        if r['metadata']['kind']!='dual':top1.extend(cost[:,0].tolist())
        if not available.any():missing+=len(targets)
    return dict(nearest=near,assigned=assigned,top1=top1,missing=missing,
                available_slots=int(p['available'].sum()),total_slots=p['available'].numel())

def summary(details):
    result={}
    for stage in ['fit','calibration']:
        result[stage]={}
        for kind in ['real','single','dual','real_bottom_feet']:
            selected=[d for d in details if d['metadata']['stage']==stage and
                      (d['metadata']['kind']==kind if kind!='real_bottom_feet' else
                       d['metadata']['kind']=='real' and d['metadata']['physical_bottom'])]
            z={}
            for method in ['greedy','regional']:
                selected_metrics=[d[method+'_feet' if kind=='real_bottom_feet' else method] for d in selected]
                a=np.asarray(sum([m['nearest'] for m in selected_metrics],[]))
                b=np.asarray(sum([m['assigned'] for m in selected_metrics],[]))
                t=np.asarray(sum([m['top1'] for m in selected_metrics],[]))
                z[method]=dict(n=len(a),nearest_capped_mean=float(a.mean()) if len(a) else None,
                               one_to_one_capped_mean=float(b.mean()) if len(b) else None,
                               candidate_recall16=float((a<=16).mean()) if len(a) else None,
                               candidate_recall32=float((a<=32).mean()) if len(a) else None,
                               top1_capped_mean=float(t.mean()) if len(t) else None,
                               missing_points=sum(m['missing'] for m in selected_metrics),
                               available_slot_fraction=sum(m['available_slots'] for m in selected_metrics)/max(1,sum(m['total_slots'] for m in selected_metrics)))
            result[stage][kind]=z
    return result

def picture(i,r,manifest,pred,arm):
    im=Image.open(ROOT/manifest['image']).convert('RGB');w,h=im.size
    colors=['#ff4422','#00bbff','#ee44dd','#99ff33','#ffbb00'];ids=[1,15,16,17,18]
    canvas=Image.new('RGB',(3*w,h+65),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for slot,name in enumerate(['eligible_GT','greedy','regional']):
        pane=im.copy();d=ImageDraw.Draw(pane)
        if slot==0:
            for q,k in r['eligible'].nonzero().tolist():
                x,y=(r['target'][q,k]*r['pixels']).tolist()
                d.line((x-6,y,x+6,y),fill=colors[k],width=2);d.line((x,y-6,x,y+6),fill=colors[k],width=2)
                d.text((x+6,y),str(ids[k]),font=font,fill=colors[k])
        else:
            p=pred[name]
            for k in range(5):
                for rank in range(6):
                    if not p['available'][k,rank]:continue
                    x,y=(p['xy'][k,rank]*r['pixels']).tolist();rad=5 if rank==0 else 2
                    d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=colors[k],width=2)
                    if rank==0:d.text((x+6,y),str(ids[k]),font=font,fill=colors[k])
        canvas.paste(pane,(slot*w,35));draw.text((slot*w+5,5),f'{arm} {name} index{i}',font=font,fill='black')
    draw.text((5,h+40),'Big=Top1, small=Top2-6. Point types, NOT animal IDs. GT only used for evaluation.',font=font,fill='black')
    name=f'{arm}_{r["metadata"]["stage"]}_{r["metadata"]["kind"]}_{i}.jpg'
    canvas.save(OUT/name,quality=93);return name

def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    state=json.loads((C/'state.json').read_text());state.update(stage='peak_integrity_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat());write(C/'state.json',state)
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')}
    manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    assert len(rows)==362 and all(r['metadata']['stage'] in ['fit','calibration'] for r in rows.values())
    selected=[next(i for i,r in rows.items() if r['metadata']['stage']==s and r['metadata']['kind']==k) for s in ['fit','calibration'] for k in ['real','single','dual']]
    selected+= [i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom'] and i not in selected]
    arms=['native_rgb','quarter_turn'];sources={a:OLD/'mixed_rotation4000_v1'/f'{a}_step4000.pt' for a in arms};hashes={a:sha(p) for a,p in sources.items()}
    protocol=dict(start_utc=now().isoformat(),images=362,stages=['fit','calibration'],checkpoints=hashes,
                  count=6,radius_px=8,regional='3x3 regional maxima;8-connected equal plateaus collapse tofirst row-major cell; flatmap yieldszero; suppress8px; no GT input',
                  metrics='Eligible known-real/visible-synthetic. Missing candidates penalized by image diagonal, not dropped; same-class one-to-one oracle is diagnostic only.',
                  visual_indices=selected,visual_selection='First image per FIT/CAL domain plus all CAL real-bottom; no error selection',
                  code_sha256=sha(Path(__file__)),extractor_sha256=sha(ROOT/'mousegeoforge/models/regional_peak_candidates.py'),
                  deadline=plan['training_stop_utc'],training=False,development_read=False)
    write(OUT/'protocol.json',protocol);results={};examples=[];begin=time.monotonic()
    for arm in arms:
        status('loading_checkpoint',arm=arm);backbone,_=build_backbone();model=MixedLocalEvidence(backbone).cuda();cp=read(sources[arm]);model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);del cp;model.eval()
        before=digest(model.backbone);head_before=digest(model.readout);bn=buffers_digest(model.backbone)
        oldpred=read(OLD/'mixed_rotation4000_v1'/f'{arm}_predictions.pt');predictions={};details=[];maxdelta=0.;nongenuine=0;greedyslots=0
        with torch.no_grad():
            for n,(i,r) in enumerate(rows.items(),1):
                if now()>=deadline:raise TimeoutError('Audit deadline')
                out=model(load_rgb(manifest[i]).cuda(),manifest[i]['image_hw']);greedy=fixed_top_peaks(out,manifest[i]['image_hw']);greedy={k:v.cpu() for k,v in greedy.items()}
                delta=float((greedy['xy']-oldpred[i]['xy']).abs().max());maxdelta=max(maxdelta,delta)
                assert delta==0 and torch.equal(greedy['available'],oldpred[i]['available']),f'Native replay failed {arm} {i}'
                regional=regional_top_peaks(out,manifest[i]['image_hw'])
                hh,ww=regional['regional_mask'].shape[-2:]
                coords=(greedy['xy']*r['pixels']/4-.5).round().long()
                genuine=regional['regional_mask'][torch.arange(5)[:,None],coords[...,1],coords[...,0]]
                nongenuine+=int((~genuine & greedy['available']).sum());greedyslots+=int(greedy['available'].sum())
                details.append(dict(index=i,metadata=r['metadata'],greedy=score(greedy,r),regional=score(regional,r),greedy_feet=score(greedy,r,True),regional_feet=score(regional,r,True)))
                predictions[i]={name:{k:v for k,v in p.items() if k in ['xy','probability','available']} for name,p in [('greedy',greedy),('regional',regional)]}
                if i in selected:examples.append(dict(arm=arm,index=i,file=picture(i,r,manifest[i],predictions[i],arm)))
                if n%25==0:status('evaluating',arm=arm,images=n,total=len(rows));print(arm,n,flush=True)
        assert digest(model.backbone)==before and digest(model.readout)==head_before and buffers_digest(model.backbone)==bn
        assert sha(sources[arm])==hashes[arm]
        results[arm]=dict(summary=summary(details),old_prediction_max_delta=maxdelta,greedy_nonregional_slot_fraction=nongenuine/max(1,greedyslots),source_unchanged=True,weights_and_bn_unchanged=True)
        write(OUT/f'{arm}_details.json',details);atomic_save(OUT/f'{arm}_predictions.pt',predictions);write(OUT/'results_partial.json',results)
        del model,backbone,oldpred;torch.cuda.empty_cache()
    assert all(sha(ROOT/m['image'])==m['sha256'] for m in manifest.values())
    write(OUT/'results.json',dict(arms=results,total_seconds=time.monotonic()-begin,all_input_hashes_unchanged=True));write(OUT/'examples.json',examples)
    page=['<!doctype html><meta charset="utf-8"><title>候选峰完整性审计</title><style>body{font:17px system-ui;margin:28px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>冻结权重：Top6是否重复占用同一峰？</h1><p>原362张FIT/CAL，未读DEV或锁定源。最多6候选、8px抑制半径不变，仅比较greedy与3×3区域极大值平台合并。缺失候选用图像对角线惩罚，不丢弃。候选覆盖不是最终实例姿态，所有图片颜色均为部位而非鼠身份。</p>']
    page.append('<pre>'+json.dumps(results,indent=2)+'</pre>')
    for e in examples:page.append(f'<h2>{e["arm"]} index{e["index"]}</h2><img src="{e["file"]}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');DEST.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.html','.jpg','.json']:shutil.copy2(p,DEST/p.name)
    status('complete');state.update(stage='peak_integrity_complete',active_job=None,active_pid=None,updated_utc=now().isoformat(),next='Visuallyreview fixed/allCALbottom peak_integrity_v1; compare candidate recall andmissing. ThenauditTRAIN center/instance supervision andimplementmatchedgrouping perdoc. NoDEVtuning.');state['completed'].append('frozen362_peak_integrity');write(C/'state.json',state)
    print('COMPLETE',json.dumps(results),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),pid=os.getpid(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',utc=now().isoformat(),pid=os.getpid()))
        raise
