"""Frozen development diagnostics of confidence/disagreement; no acceptance gate fit."""
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import fcntl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from bottom_anchor_common_20261002 import ROOT, OUT, PRIOR, budget, write_state, exclusions


def rank_fraction(x):
    # Average ranks for exact ties; independent of labels.
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    del values
    return (np.cumsum(counts) - (counts+1)/2)[inverse] / max(1,len(x)-1)


def review_stats(score, endpoints):
    order = np.argsort(-score, kind='stable')
    result = []
    for fraction in (.05,.10,.20):
        n = math.ceil(len(score)*fraction); selected = order[:n]
        result.append(dict(review_fraction=fraction, reviewed=n, population=len(score),
            captured={name:dict(total=int(np.sum(labels)), found=int(np.sum(labels[selected])),
                recall=float(np.mean(labels[selected])*n/np.sum(labels)) if np.sum(labels) else None)
                for name,labels in endpoints.items()},
            tied_at_boundary=int(np.sum(score==score[order[n-1]]))))
    return result


def main():
    plan,_,check=budget(300); check()
    dest=OUT/'quadruped_uncertainty_audit_v1'; assert not dest.exists()
    dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']];ids=dm['ids_1based']
    assert len(entries)==281
    assert all(e['session'] in plan['development_sessions'] and (e['session'],int(e['frame'])) not in exclusions(plan) for e in entries)
    arrays=[]
    for folder in ('quadruped_epoch110_review_v1','quadruped_seed29_epoch085_review_v1'):
        with np.load(OUT/folder/'predictions_source_pixels.npz') as z:
            arrays.append({k:z[k].copy() for k in z.files})
    a,b=arrays; assert np.allclose(a['gt'],b['gt'],equal_nan=True)
    assert np.array_equal(a['context_manifest_indices'],b['context_manifest_indices'])
    observed=np.isfinite(a['gt']).all(axis=-1)
    e1=np.linalg.norm(a['uv']-a['gt'],axis=-1);e2=np.linalg.norm(b['uv']-a['gt'],axis=-1)
    worst=np.maximum(e1,e2);best=np.minimum(e1,e2)
    disagree=np.linalg.norm(a['uv']-b['uv'],axis=-1)
    conf=np.minimum(a['confidence'],b['confidence'])
    assert np.isfinite(disagree).all() and np.isfinite(conf).all()
    camera=np.array([e['camera'] for e in entries]);sessions=np.array([e['session'] for e in entries])
    groups={'all':observed,'paws':observed&np.isin(ids,[15,16,17,18])[None,:],
        'dorsal':observed&np.isin(ids,[6,7])[None,:], 'bottom':observed&(camera=='A')[:,None],
        'side':observed&(camera!='A')[:,None]}
    for session in plan['development_sessions']:groups['session_'+session]=observed&(sessions==session)[:,None]
    rankings={};raw_scores={'disagreement':disagree,'low_min_confidence':-conf}
    # Combined rank is exploratory equal weighting, never optimized against GT.
    for name,mask in groups.items():
        scores={k:v[mask] for k,v in raw_scores.items()}
        scores['equal_rank_mean']=(rank_fraction(scores['disagreement'])+rank_fraction(scores['low_min_confidence']))/2
        endpoints={f'{kind}_over{t}':val[mask]>t for kind,val in [('incumbent',e1),('either_seed',worst)] for t in (25,50,100)}
        rankings[name]={k:review_stats(v,endpoints) for k,v in scores.items()}
    # Image review uses all7 predicted points, including unlabeled points; GT only scores known errors.
    image_scores={k:np.max(v,axis=1) for k,v in raw_scores.items()}
    image_scores['equal_rank_mean']=(rank_fraction(image_scores['disagreement'])+rank_fraction(image_scores['low_min_confidence']))/2
    image_endpoints={f'{kind}_over{t}':np.any((val>t)&observed,axis=1) for kind,val in [('incumbent',e1),('either_seed',worst)] for t in (25,50,100)}
    image_rankings={k:review_stats(v,image_endpoints) for k,v in image_scores.items()}
    shared=observed&(disagree<=10)&(best>25)
    cases=[]
    for i,j in np.argwhere(shared):
        e=entries[i]
        cases.append(dict(image_index=int(i),keypoint_index=int(j),session=e['session'],frame=e['frame'],camera=e['camera'],id=ids[j],
            seed17_error=float(e1[i,j]),seed29_error=float(e2[i,j]),disagreement=float(disagree[i,j]),min_confidence=float(conf[i,j])))
    cases.sort(key=lambda r:min(r['seed17_error'],r['seed29_error']),reverse=True)
    dest.mkdir();font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',22);gallery=[]
    for rank,r in enumerate(cases[:6]):
        check();i=r['image_index'];j=r['keypoint_index'];e=entries[i]
        im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
        for label,points,color in [('GT',a['gt'],'#70ff70'),('S17',a['uv'],'#ff77df'),('S29',b['uv'],'#40cfff')]:
            x,y=points[i,j];d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
            d.text((x+10,y+({'GT':-30,'S17':0,'S29':25}[label])),f'{label} id{ids[j]}',font=font,fill=color,stroke_width=1,stroke_fill='black')
        filename=f'shared_{rank+1:02d}_{e["session"]}_{e["frame"]}_cam{e["camera"]}_id{ids[j]}.jpg'
        im.save(dest/filename,quality=95);gallery.append(filename)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colors=np.broadcast_to(np.isin(ids,[6,7])[None,:],observed.shape)[observed]
    axes[0].scatter(disagree[observed],worst[observed],c=colors,cmap='coolwarm',s=9,alpha=.5)
    axes[0].set(xlabel='Best-seed disagreement (source px)',ylabel='Worse seed error (source px)',title='Agreement does not prove correctness')
    for name,records in rankings['all'].items():
        axes[1].plot([r['review_fraction']*100 for r in records],[r['captured']['either_seed_over50']['recall']*100 for r in records],'o-',label=name)
    axes[1].plot([5,10,20],[5,10,20],'k--',label='random expected')
    axes[1].set(xlabel='Observed points reviewed (%)',ylabel='Either-seed >50px errors captured (%)',title='Development-only ranking diagnostic')
    axes[1].legend();[ax.grid(alpha=.25) for ax in axes]
    fig.savefig(dest/'uncertainty_diagnostics.png',dpi=140);plt.close(fig)
    with (ROOT/plan['gpu_lock']).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=False
        except BlockingIOError:held=True
    report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),observed_points=int(observed.sum()),images=281,
        point_review_rankings=rankings,image_review_rankings=image_rankings,shared_error_cases=cases,gallery=gallery,
        shared_error_definition='Both errors>25 source pixels and seed disagreement<=10; descriptive diagnostic only, not production threshold.',
        confidence_ranges={name:[float(z['confidence'].min()),float(z['confidence'].max())] for name,z in [('seed17',a),('seed29',b)]},
        point_ranking_caveat='Only observed points included for evaluation; image ranking uses all7 predictions. Different denominators; do not compare as equal annotation effort.',
        policy='No confidence-filtered full-cohort accuracy claim. Equal rank mean is exploratory and uncalibrated. No automatic acceptance/relabeling.',
        next='Use findings to design a small label-preserving temporal reliability check on allowed development clips; keep geometry correction and strong3D export paused.',
        shared_GPU_lock_held=held,prior_STOP_preserved=(PRIOR/'STOP').exists(),locked_holdouts_accessed=False,pseudo_gt_released=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    page='<meta charset="utf-8"><h1>置信度与双种子分歧诊断</h1><p>仅开发集复核排序，不是自动接纳伪标签的阈值。绿圈人工、粉圈seed17、蓝圈seed29。以下展示两模型相近但共同偏离人工点的六个最大误差例子。</p><img style="max-width:100%" src="uncertainty_diagnostics.png">'
    page+=''.join(f'<h2>{p}</h2><img width="720" src="{p}">' for p in gallery)
    (dest/'index.html').write_text(page+'<pre>'+json.dumps(report,indent=2)+'</pre>')
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name;shutil.copytree(dest,export)
    write_state(stage='teacher_uncertainty_audit_complete',latest_decision=str(dest/'summary.json'),latest_training_visual_directory=str(export),next_action=report['next'])
    print(json.dumps(dict(export=str(export),all_points=rankings['all'],image_rankings=image_rankings,shared_count=len(cases),shared_top6=cases[:6],confidence_ranges=report['confidence_ranges']),indent=2))


if __name__=='__main__':main()
