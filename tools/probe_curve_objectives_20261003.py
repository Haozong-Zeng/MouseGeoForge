"""Controlled objective alternatives on SAME frozen features/init/data as feature probe."""
import collections,datetime as dt,fcntl,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from mousegeoforge.surface_curve_supervision import curve_loss,material_curve_loss,points_to_polyline
from probe_diverse_curve_features_20261003 import FeatureProbe
from run_depth_pose_ablation import write_json,fingerprint
SOURCE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'
OUT=ROOT/'output/surface_curve_cycle_20261003/curve_objectives_v1'

def main():
    assert (SOURCE/'results.json').exists() and not OUT.exists()
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=__import__('os').getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_frozen_features');saved=torch.load(SOURCE/'frozen_inputs.pt',map_location='cpu',weights_only=False)
    data={k:v.cuda() for k,v in saved['tensors'].items()};meta=saved['metadata'];train=[i for i,m in enumerate(meta) if m['role']=='train'];val=[i for i,m in enumerate(meta) if m['role']=='validation']
    buckets=collections.defaultdict(list)
    for i in train:buckets[meta[i]['dataset']].append(i)
    rng=random.Random(17);keys=sorted(buckets);schedule=[[rng.choice(buckets[rng.choice(keys)]) for _ in range(8)] for _ in range(2400)]
    write_json(OUT/'protocol.json',dict(source_features_sha256=fingerprint(SOURCE/'frozen_inputs.pt'),baseline_results=str(SOURCE/'results.json'),
        steps=2400,seed=17,batch_size=8,learning_rate=.001,weight_decay=.001,conditioning=['det','pose'],parameterization='free_residual',
        objective_alternatives={'reduced_order':'Existing curve loss, order_weight0.005 instead of0.05','material_huber':'Known rest-mesh material correspondence Huber(beta0.01)/768 +0.1*shape loss with order_weight0'},
        provenance_gate='All150observations are synthetic65uniformRESTmaterial samples, not2D/3D arc length and not manual Core23fraction points',
        same_initialization_and_schedule_as_source=True,model_heads_frozen=True,baseline_free_arms_reused_not_retrained=True,
        decision='Fixed final2400steps; no validation checkpoint selection. Loss scale/shape changes are part of objective alternative, not an isolated equal-gradient order-only attribution for material_huber.',
        constraints='Never apply material correspondence to real manual curves or assign fixed fractions to06/07. No production change.'))
    def evaluate(probe,feature,ids):
        values=collections.defaultdict(list)
        with torch.no_grad():
            pred=probe(data[feature][ids],data['boxes'][ids],data['base_logits'][ids])*data['pixels'][ids,None];gt=data['gt'][ids]*data['pixels'][ids,None]
            for p,g in zip(pred,gt):
                values['pred_to_gt_px'].extend(points_to_polyline(p,g,torch.ones(len(g),device='cuda',dtype=torch.bool))[0].sqrt().tolist())
                values['gt_to_pred_px'].extend(points_to_polyline(g,p,torch.ones(len(p),device='cuda',dtype=torch.bool))[0].sqrt().tolist());values['endpoints_px'].extend((p[[0,-1]]-g[[0,-1]]).norm(dim=-1).tolist())
        return {k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in values.items()}
    results=[];predictions={};valid=torch.ones(65,device='cuda',dtype=torch.bool);u=torch.linspace(0,1,65,device='cuda')
    for feature in ['det','pose']:
        for objective in ['reduced_order','material_huber']:
            arm=feature+'_'+objective;torch.manual_seed(17);probe=FeatureProbe(False).cuda();optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.001)
            history=[dict(step=0,train=evaluate(probe,feature,train),validation=evaluate(probe,feature,val))]
            reference=json.loads((SOURCE/f'{feature}_free_history.json').read_text())[0]
            assert history[0]['train']==reference['train'] and history[0]['validation']==reference['validation'],'Initial predictions must match baseline exactly'
            status('training',arm=arm,step=0)
            for step,ids in enumerate(schedule,1):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                optimizer.zero_grad(set_to_none=True);pixels=data['pixels'][ids];p=probe(data[feature][ids],data['boxes'][ids],data['base_logits'][ids])*pixels[:,None];gt=data['gt'][ids]*pixels[:,None]
                terms=[]
                for pp,gg in zip(p,gt):
                    if objective=='reduced_order':term=curve_loss(pp,gg,valid,full_curve=True,scale=768,order_weight=.005)
                    else:term=material_curve_loss(pp,gg,valid,u,material_parameter_known=True,scale=768)+.1*curve_loss(pp,gg,valid,full_curve=True,scale=768,order_weight=0.)
                    terms.append(term)
                loss=torch.stack(terms).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);assert torch.isfinite(norm);optimizer.step()
                if step%400==0:
                    row=dict(step=step,loss=float(loss.detach()),train=evaluate(probe,feature,train),validation=evaluate(probe,feature,val));history.append(row);write_json(OUT/f'{arm}_history.json',history);status('training',arm=arm,step=step);print(arm,step,row['train']['pred_to_gt_px']['mean'],row['validation']['pred_to_gt_px']['mean'],flush=True)
            torch.save(dict(state=probe.state_dict(),feature=feature,objective=objective,step=2400),OUT/f'{arm}.pt');results.append(dict(arm=arm,**history[-1]));write_json(OUT/'partial_results.json',results)
            with torch.no_grad():predictions[arm]=probe(data[feature],data['boxes'],data['base_logits']).cpu().numpy()
    np.savez_compressed(OUT/'predictions.npz',**predictions);write_json(OUT/'results.json',dict(results=results,inspection_pending=True,no_production_promotion=True));status('complete')
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
