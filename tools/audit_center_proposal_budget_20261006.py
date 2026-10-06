"""Fixed Top2/Top8 center proposal coverage, not final instance detection."""
import os
os.environ['DLClight']='True'
import sys,json,fcntl,datetime as dt,traceback,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw,ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest
from train_dense_center_warmup_20261005 import read,write,now
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,decode_centers
C=ROOT/'output/local_grouping_cycle_20261005';OUT=C/'center_proposal_budget_audit_v1'
INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'

def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    assert json.loads((C/'center_task_gradient_audit_v2/status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);examples=OUT/'examples';examples.mkdir();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='center_proposal_budget_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup_hash=sha(sup_path);sup={r['index']:r for r in read(sup_path)}
    samples=json.loads((C/'center_task_gradient_audit_v2/protocol.json').read_text())['samples']
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    indices=list(dict.fromkeys(fixed+[s['index'] for s in samples]+[i for i,r in rows.items() if r['metadata']['stage']=='calibration']))
    selected=list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i in indices if rows[i]['metadata']['stage']=='calibration' and rows[i]['metadata']['kind']=='real' and rows[i]['metadata']['physical_bottom']]))
    assert all(rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3'] for i in indices)
    write(OUT/'protocol.json',dict(indices=indices,arms=['focal','gaussian'],step=9216,budgets=[2,8],radius_px=8,
        description='Native768 images. Top8 independent regional center peaks, no score gate for candidate coverage; Top2 nested control. GT only posthoc perimage one-to-one Hungarian distance matching; absent candidates incur image diagonal. NoGTcrops/count/input. This is an oracle prerequisite, NOTdetection/poseaccuracy.',
        purpose='Check whether predicted-center grouping with learned rejection is worth a matchedpilot; do not gate on raw heatmap0.5. K8=4xknownmaxcapacity2 is a predeclaredengineeringbudget, notCALsearched orCenterGroup exactsetting.',
        references=['https://raw.githubusercontent.com/dvl-tum/center-group/main/configs/centergroup/base.py','https://raw.githubusercontent.com/dvl-tum/center-group/main/centergroup/core/inference.py'],
        scope='same89CAL plusfixed3FIT and48seed61stratifiedFIT; notcompleteFITcoverage or independentbenchmark',
        code_sha256=sha(Path(__file__)),supervision_sha256=sup_hash,deadline=plan['training_stop_utc']))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);summary={};hashes={};start=time.monotonic()
    for arm in ['focal','gaussian']:
        source=C/f'dense_center_loss_{arm}_9216_v1/center_step9216.pt';hashes[arm]=sha(source);cp=read(source);backbone,_=build_backbone()
        model=MixedLocalEvidence(backbone,adapt_backbone=False).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.eval()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);center.eval();del cp
        before=(digest(model),digest(center),buffers_digest(backbone));old=read(C/f'dense_center_loss_{arm}_9216_v1/step9216_predictions.pt');details=[];predictions={}
        with torch.no_grad():
            for n,i in enumerate(indices,1):
                if now()>=deadline:raise TimeoutError('Proposal audit cutoff')
                r=rows[i];hw=manifest[i]['image_hw'];out=center(model(load_rgb(manifest[i]).cuda(),hw),hw)
                p={k:v.cpu() for k,v in decode_centers(out,hw,capacity=8).items()};p2=decode_centers(out,hw)
                assert torch.equal(p['centers'][:2],p2['centers'].cpu()) and torch.equal(p['available'][:2],p2['available'].cpu())
                if i in old:assert all(torch.equal(v.cpu(),old[i]['decoded'][k]) for k,v in p2.items()),(arm,i,'replay')
                predictions[i]=p;gt=sup[i]['mask_centroid'][sup[i]['instance_valid']]*r['pixels'];diag=float(r['pixels'].float().norm());metrics={}
                for budget in [2,8]:
                    xy=p['centers'][:budget][p['available'][:budget]]*r['pixels'];errors=[diag]*len(gt)
                    if len(xy):
                        distance=torch.cdist(xy,gt);rr,cc=linear_sum_assignment(distance.numpy())
                        for a,b in zip(rr,cc):errors[b]=float(distance[a,b])
                    metrics[budget]=dict(errors=errors,available=len(xy),all_within64=all(e<=64 for e in errors))
                details.append(dict(index=i,metadata=r['metadata'],metrics=metrics))
                if i in selected:
                    rgb=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=rgb.size;canvas=Image.new('RGB',(2*w,h+40),'white');canvas.paste(rgb,(0,40));canvas.paste(rgb,(w,40));d=ImageDraw.Draw(canvas)
                    d.text((4,4),f'{arm} index{i}: GT mask center | Top8 proposals NOT detections; numbers=rank',font=font,fill='black')
                    for x,y in gt.tolist():d.line((x-6,y+40,x+6,y+40),fill='cyan',width=2);d.line((x,y+34,x,y+46),fill='cyan',width=2)
                    for rank in p['available'].nonzero().flatten().tolist():
                        x,y=(p['centers'][rank]*r['pixels']).tolist();x+=w;y+=40;color='orange' if rank<2 else 'lime';d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);d.text((x+5,y),str(rank+1),fill=color,font=font)
                    canvas.save(examples/f'{arm}_{i}.jpg',quality=93)
                if n%20==0:status('auditing',arm=arm,images=n,total=len(indices));print(arm,n,flush=True)
        assert before==(digest(model),digest(center),buffers_digest(backbone)) and sha(source)==hashes[arm]
        write(OUT/f'{arm}_details.json',details);torch.save(predictions,OUT/f'{arm}_predictions.pt');summary[arm]={}
        for stage in ['fit','calibration']:
            summary[arm][stage]={}
            for kind in ['real','real_bottom','single','dual']:
                ds=[d for d in details if d['metadata']['stage']==stage and (d['metadata']['kind']==kind or (kind=='real_bottom' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']))]
                summary[arm][stage][kind]={}
                for budget in [2,8]:
                    ee=np.array([e for d in ds for e in d['metrics'][budget]['errors']]);summary[arm][stage][kind][budget]=dict(images=len(ds),instances=len(ee),mean_error_px=float(ee.mean()),recall16=float((ee<=16).mean()),recall32=float((ee<=32).mean()),recall64=float((ee<=64).mean()),all_instance_coverage64=float(np.mean([d['metrics'][budget]['all_within64'] for d in ds])))
        del model,center,backbone,old;torch.cuda.empty_cache()
    assert sha(sup_path)==sup_hash and all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in indices)
    result=dict(summary=summary,checkpoint_sha256=hashes,weights_bn_sources_unchanged=True,replay_existing92_exact=True,training=False,seconds=time.monotonic()-start);write(OUT/'results.json',result)
    page=['<!doctype html><meta charset="utf-8"><title>中心候选覆盖审计</title><h1>Top2/Top8中心候选覆盖，不是检测成绩</h1><pre>'+json.dumps(result,indent=2)+'</pre>']
    for i in selected:
        for arm in ['focal','gaussian']:page.append(f'<img style="max-width:100%" src="examples/{arm}_{i}.jpg">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True);status('complete');state.update(stage='center_proposal_budget_audit_complete',active_job=None,active_pid=None,next='Review candidate coverage and visuals; decide if a matched predicted-center grouping/rejection pilot is justified. Oracle is not final accuracy.',updated_utc=now().isoformat());state['completed'].append('center_proposal_budget_audit');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
