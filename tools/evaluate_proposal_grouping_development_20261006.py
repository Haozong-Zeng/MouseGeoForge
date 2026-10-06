"""Frozen fixed endpoints, all144 reused DEV images. No further training."""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,time,traceback,collections,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest,atomic_save
from train_dense_center_warmup_20261005 import read,write,now
from train_local_instance_grouping_20261005 import metrics,render_predictions
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters
from mousegeoforge.models.proposal_instance_grouping import ProposalInstanceGrouping,decode_proposal_grouped
C=ROOT/'output/local_grouping_cycle_20261005';OLD=ROOT/'output/singleview_fusion_cycle_20261005';INPUT=OLD/'mixed_local_development_inputs_v1';OUT=C/'proposal_final_development144_v1';KP=[0,14,15,16,17]

def aggregate(details):
    result={}
    selectors={k:lambda m,k=k:m['kind']==k for k in ['real','single','dual']}
    selectors.update(real_bottom_feet=lambda m:m['kind']=='real' and m['physical_bottom'],real_other=lambda m:m['kind']=='real' and not m['physical_bottom'])
    for source in ['FR2','AZR2L2']:selectors[source]=lambda m,s=source:m['kind']=='real' and m['source']==s
    for name,select in selectors.items():
        ds=[d for d in details if select(d['metadata'])];points=[p for d in ds for p in d['metrics']['points'] if name!='real_bottom_feet' or p['keypoint']>0]
        rr=dict(images=len(ds),points=len(points),images_without_eligible=sum(not d['metrics']['points'] for d in ds),count_accuracy=float(np.mean([d['metrics']['count_correct'] for d in ds])))
        for key in ['tp64','fp64','fn64']:rr[key]=sum(d['metrics'][key] for d in ds)
        if points:rr.update(joint64_pck16=float(np.mean([p['joint64_error']<=16 for p in points])),joint64_error_px=float(np.mean([p['joint64_error'] for p in points])),candidate_oracle_px=float(np.mean([p['oracle'] for p in points])),accepted_fraction=float(np.mean([p['accepted'] for p in points])))
        result[name]=rr
    return result

@torch.no_grad()
def main():
    freeze=json.loads((C/'training_frozen.json').read_text());assert freeze['training_closed_for_this_cycle'];plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['hard_deadline_utc'].replace('Z','+00:00'))-dt.timedelta(minutes=10);assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='final_development_evaluation',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),training_closed_for_this_cycle=True,next='NO moretraining afterDEV; fixed4endpoints144DEV thenvisual/report.',updated_utc=now().isoformat());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('preparing_supervision');manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};assert len(rows)==144
    fit=read(OLD/'mixed_local_fulltrain_rgb_inputs_v1/evaluation_only.pt');fit_scenes={r['metadata']['scene'] for r in fit};assert all(r['metadata']['scene'] not in fit_scenes for r in rows.values());del fit
    groups=collections.defaultdict(list)
    for i,r in rows.items():
        m=r['metadata'];assert m['stage']=='validation' and m.get('source') not in ['GR2','UR2','UR3'];assert m['kind']!='real' or m['source'] in ['FR2','AZR2L2'];groups[(m['scene'],tuple(manifest[i]['image_hw']),m['kind'])].append(i)
    sources=dict(zip(['independent3200','within3200','bce_add640','focal_add640'],[C/p for p in freeze['fixed_endpoints']]));hashes={a:sha(p) for a,p in sources.items()}
    selected=[]
    for source in ['FR2','AZR2L2']:
        for bottom in [True,False]:selected.extend([i for i,r in rows.items() if r['metadata']['source']==source and bool(r['metadata']['physical_bottom'])==bottom][:2])
    for kind in ['single','dual']:selected.extend([i for i,r in rows.items() if r['metadata']['kind']==kind][:4])
    selected=list(dict.fromkeys(selected+[i for i,r in rows.items() if r['metadata']['kind']=='real' and r['metadata']['physical_bottom']]))
    protocol=dict(freeze=freeze,source_sha256=hashes,visual_indices=selected,indices=list(rows),code_sha256=sha(Path(__file__)),input_manifest_sha256=sha(INPUT/'inference_manifest.json'),evaluation_labels_sha256=sha(INPUT/'evaluation_only.pt'),new_training=False,deadline=deadline.isoformat(),mask_definition='Approved projected mask centroid(mean pixel+.5)/WH including tail; GT ONLYmetrics. No GT inforward. Unknownpointsignored; all144includingemptyeligibleimage.')
    write(OUT/'protocol.json',protocol);sup={};source_files={}
    for n,((scene,hw,kind),ids) in enumerate(groups.items(),1):
        if now()>=deadline:raise TimeoutError('Preparationcutoff')
        views=sorted({rows[i]['metadata']['view'] for i in ids});item=load_item(dict(scene=scene,views=views,image_hw=list(hw),kind=kind));t=instance_targets_from_batch(item['batch']);source_files.update(item['source_hashes']);h,w=hw
        for i in ids:
            r=rows[i];v=views.index(r['metadata']['view']);rgb=(item['batch']['images'][0,v].permute(1,2,0)*255).round().byte().numpy();assert sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'];assert np.array_equal(rgb,np.asarray(Image.open(ROOT/manifest[i]['image']).convert('RGB')))
            iv=t['instance_valid'][0,v].bool();known=t['keypoint_valid'][0,v,:,KP]&iv[:,None];assert torch.equal(known,r['coordinate_valid']);assert torch.allclose(t['keypoint_xy'][0,v,:,KP][known],r['target'][known],atol=1e-6,rtol=0)
            assert torch.equal(t['keypoint_visibility_valid'][0,v,:,KP],r['visibility_valid']) and torch.equal(t['keypoint_visible'][0,v,:,KP],r['visible']);assert torch.equal(iv,r['instance_valid'])
            masks=t['masks'][0,v];mv=t['mask_valid'][0,v].bool();assert (mv[iv]).all();centers=torch.zeros(len(iv),2)
            for q in iv.nonzero().flatten().tolist():
                yy,xx=masks[q].bool().nonzero(as_tuple=True);assert len(xx);centers[q]=torch.tensor([(xx.float().mean()+.5)/w,(yy.float().mean()+.5)/h])
            sup[i]=dict(instance_valid=iv,mask_centroid=centers,centroid_valid=iv.clone())
        if n%8==0:status('preparing_supervision',scenes=n,total=len(groups));print('SUP',n,flush=True)
    assert len(sup)==144;atomic_save(OUT/'evaluation_instance_targets.pt',sup);write(OUT/'source_hashes.json',source_files)
    examples=OUT/'examples';examples.mkdir();summary={};performance={}
    for arm,path in sources.items():
        cp=read(path);backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.eval()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);center.eval();group=ProposalInstanceGrouping(interaction=arm!='independent3200').cuda();group.load_state_dict(cp['grouping']);group.eval();del cp
        before=[digest(m) for m in [model,center,group]]+[buffers_digest(backbone)];torch.cuda.reset_peak_memory_stats();pred={};details=[];start=time.monotonic()
        for n,(i,r) in enumerate(rows.items(),1):
            if now()>=deadline:raise TimeoutError('Inferencecutoff')
            x=load_rgb(manifest[i]).cuda();hw=manifest[i]['image_hw'];evidence=model(x,hw);out=group(evidence,center(evidence,hw),hw);decoded=decode_proposal_grouped(out);cpu={k:v.cpu() for k,v in out.items()};dec={k:v.cpu() for k,v in decoded.items()};pred[i]=dict(output=cpu,decoded=dec)
            details.append(dict(index=i,metadata=r['metadata'],metrics=metrics(cpu,dec,r,sup[i],hw)))
            if n%24==0:status('evaluating',arm=arm,images=n,total=144);print(arm,n,flush=True)
        assert before==[digest(m) for m in [model,center,group]]+[buffers_digest(backbone)] and sha(path)==hashes[arm]
        summary[arm]=aggregate(details);performance[arm]=dict(seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),weights_and_bn_unchanged=True);atomic_save(OUT/f'{arm}_predictions.pt',pred);write(OUT/f'{arm}_details.json',details);write(OUT/'results_partial.json',dict(summary=summary,performance=performance));render_predictions(examples,arm,pred,rows,manifest,selected);del model,center,group,backbone,pred;torch.cuda.empty_cache()
    assert all(sha(ROOT/p)==h for p,h in source_files.items());assert all(sha(ROOT/m['image'])==m['sha256'] for m in manifest.values());assert sha(INPUT/'evaluation_only.pt')==protocol['evaluation_labels_sha256']
    result=dict(summary=summary,performance=performance,input_parity=True,source_files_unchanged=True,optimizer_steps=0,production_changed=False,no_post_development_training=True);write(OUT/'results.json',result)
    page=['<!doctype html><meta charset="utf-8"><h1>冻结终点：144张开发来源验证</h1><p>不是独立外部测试。四个固定终点，所有144图，无GT前向或数量强制；缺失/拒绝按图像对角线计惩罚。此后本轮不再训练或调阈值。</p><pre>'+json.dumps(result,indent=2)+'</pre>']
    for p in sorted(examples.glob('*.jpg')):page.append(f'<img style="max-width:100%" src="examples/{p.name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');status('complete');state.update(stage='final_development_complete',active_job=None,active_pid=None,next='NO moretraining. Visuallyreview4fixedendpoints and report all failures; harddeadline10:14:45Z.',updated_utc=now().isoformat());state['completed'].append('proposal_final_development144');write(C/'state.json',state)
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
