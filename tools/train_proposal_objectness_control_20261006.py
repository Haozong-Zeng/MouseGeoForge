"""Matched BCE/focal control from one common within-instance3200 checkpoint."""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,time,random,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest,atomic_save
from train_dense_center_warmup_20261005 import read,write,now
from train_local_instance_grouping_20261005 import metrics,summarize,render_predictions
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,dense_center_loss
from mousegeoforge.models.proposal_instance_grouping import ProposalInstanceGrouping,decode_proposal_grouped
from mousegeoforge.models.proposal_objectness_losses import grouping_objectness_loss
from mousegeoforge.models.local_instance_grouping import rotate_mask_centroids
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
C=ROOT/'output/local_grouping_cycle_20261005';INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'
PREVIOUS=C/'proposal_grouping_continue3200_v1';SOURCE=PREVIOUS/'within_instance_step3200.pt';OUT=C/'proposal_objectness_control640_v1';ARMS=['balanced_bce','balanced_focal'];STEPS=640

def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    assert json.loads((C/'proposal_matching_stability_audit_v1/status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='proposal_objectness_control_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat(),next='Wait for matched BCE/focal640; no second GPU job.');write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup_hash=sha(sup_path);sup={r['index']:r for r in read(sup_path)};source_hash=sha(SOURCE)
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    pools={b:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==b] for b in [True,False]}
    syn={k:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k] for k in ['single','dual']}
    rng=random.Random(17);rot=random.Random(43);all_schedule=[[rng.choice(pools[j%2==0]),rng.choice(syn['single']),rng.choice(syn['dual'])] for j in range(3200+STEPS)];all_turns=[[rot.randrange(4) for _ in range(3)] for j in range(3200+STEPS)]
    previous_protocol=json.loads((PREVIOUS/'protocol.json').read_text());assert all_schedule[:3200]==previous_protocol['schedule'] and all_turns[:3200]==previous_protocol['rotations'];schedule=all_schedule[3200:];turns=all_turns[3200:]
    assert sha(ROOT/'mousegeoforge/models/proposal_instance_grouping.py')==previous_protocol['module_sha256']
    evaluation=fixed+[i for i,r in rows.items() if r['metadata']['stage']=='calibration'];selected=list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i in evaluation if rows[i]['metadata']['stage']=='calibration' and rows[i]['metadata']['kind']=='real' and rows[i]['metadata']['physical_bottom']]))
    needed=set(evaluation)|{i for ids in schedule for i in ids}
    for i in needed:
        assert rows[i]['metadata']==sup[i]['metadata'] and rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3']
        assert torch.equal(sup[i]['instance_valid'],sup[i]['centroid_valid']) and bool(sup[i]['detection_annotation_complete'])
    models={};opts={};bn={};initial={};cp=read(SOURCE);assert cp['step']==3200 and cp['arm']=='within_instance';old=read(PREVIOUS/'within_instance_step3200_predictions.pt')
    for arm in ARMS:
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.train()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);group=ProposalInstanceGrouping(interaction=True).cuda();group.load_state_dict(cp['grouping'])
        models[arm]=(model,center,group);bn[arm]=buffers_digest(backbone);initial[arm]=[digest(m) for m in models[arm]]
        opt=torch.optim.AdamW([dict(params=group.parameters(),lr=3e-4),dict(params=center.parameters(),lr=3e-4),dict(params=model.readout.parameters(),lr=1e-4),dict(params=backbone.parameters(),lr=1e-5)],weight_decay=1e-4)
        # Independent copies of identical AdamW state, including step counters.
        import copy
        opt.load_state_dict(copy.deepcopy(cp['optimizer']));opts[arm]=opt;assert [g['lr'] for g in opt.param_groups]==[3e-4,3e-4,1e-4,1e-5]
    del cp
    assert initial[ARMS[0]]==initial[ARMS[1]]
    protocol=dict(source_checkpoint=str(SOURCE.relative_to(ROOT)),source_sha256=source_hash,initial_model_hashes=initial,source_group_step=3200,additional_steps=STEPS,checks=[0,320,640],schedule=schedule,rotations=turns,seed=17,rotations_seed=43,optimizer='Identical copies of within_instance3200 fullAdamW; noreset',architecture='Both fullwithininstanceattention; unchangedTop8/pointTop6/learnedobjectgate. This is loss comparison NOT attentionablation.',difference='Only objectnessgamma0 versus2; same equal positive/negative mean BCE weighting, alpha=-1. Focal suppresses easy cases; absolute loss magnitude also changes, no compensating tuned scalar.',loss='localCE+densecenterfocal+assignmentCE+balancedobjectness; matching nearest64pxHungarian unchanged',inference='Unchanged .5 learnedobjectnessgate, atmost2,8pxcenterNMS,exclusivepointassignment; noGTforward/count',caveat='ExploratoryafterCAL. InspiredbyCenterGroupgamma2but their allproposalmean differs; NOTexactreproduction. Source selectedfromexistingwithin3200,compareagainstmatchedBCEnotagainstindependent3200.',code_sha256=sha(Path(__file__)),loss_module_sha256=sha(ROOT/'mousegeoforge/models/proposal_objectness_losses.py'),supervision_sha256=sup_hash,deadline=plan['training_stop_utc'],production_changed=False)
    write(OUT/'protocol.json',protocol);images={}
    for i in sorted(needed):
        if now()>=deadline:raise TimeoutError('Inputcutoff')
        images[i]=load_rgb(manifest[i]).cuda()
    def compute(arm,i,k=0):
        model,center,group=models[arm];r=rows[i];s=sup[i];hw=manifest[i]['image_hw'];x=images[i];xy=r['target'].cuda();eligible=r['eligible'].cuda();gt=s['mask_centroid'].cuda();cv=s['centroid_valid'].cuda()
        if k:x,xy,hw=quarter_turn(x,xy,eligible,hw,k);gt=rotate_mask_centroids(gt,cv,k)
        evidence=model(x,hw);co=center(evidence,hw);out=group(evidence,co,hw)
        losses=grouping_objectness_loss(out,xy,eligible,s['instance_valid'].cuda(),gt,cv,hw,gamma=2 if arm=='balanced_focal' else 0,annotation_complete=True)
        losses['local']=point_evidence_loss(evidence,xy,eligible,hw);losses['center']=dense_center_loss(co,gt,cv,hw);losses['total']=losses['total']+losses['local']+losses['center'];return out,losses
    curves=[];predictions={};histories={a:[] for a in ARMS};seconds={a:0. for a in ARMS};gradients={a:[] for a in ARMS};peak=0
    @torch.no_grad()
    def evaluate(arm,step):
        details=[];pred={}
        for i in evaluation:
            if now()>=deadline:raise TimeoutError('Evalcutoff')
            out,loss=compute(arm,i);decoded=decode_proposal_grouped(out);cpu={k:v.cpu() for k,v in out.items()};dec={k:v.cpu() for k,v in decoded.items()}
            if step==0:
                assert all(torch.equal(v,old[i]['output'][k]) for k,v in cpu.items()),(arm,i,'sourceoutput')
                assert all(torch.equal(v,old[i]['decoded'][k]) for k,v in dec.items()),(arm,i,'sourcedecode')
            details.append(dict(index=i,metadata=rows[i]['metadata'],loss={k:float(v) for k,v in loss.items() if v.ndim==0},metrics=metrics(cpu,dec,rows[i],sup[i],manifest[i]['image_hw'])));pred[i]=dict(output=cpu,decoded=dec)
        curves.append(dict(arm=arm,step=step,summary=summarize(details)));write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',details);atomic_save(OUT/f'{arm}_step{step:04d}_predictions.pt',pred);predictions[arm]=pred
    def checkpoint(arm,step):
        model,center,group=models[arm];atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(step=step,arm=arm,source_sha256=source_hash,backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},center={k:v.cpu() for k,v in center.state_dict().items()},grouping={k:v.cpu() for k,v in group.state_dict().items()},optimizer=opts[arm].state_dict()))
    start=time.monotonic()
    for arm in ARMS:status('evaluating',arm=arm,step=0);evaluate(arm,0)
    del old
    for step,ids in enumerate(schedule,1):
        for arm in ARMS:
            if now()>=deadline:raise TimeoutError('Trainingcutoff')
            model,center,group=models[arm];opt=opts[arm];opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic();value=0.
            for slot,i in enumerate(ids):
                out,losses=compute(arm,i,turns[step-1][slot]);loss=losses['total']/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach());del out,losses,loss
            if step in [1,2,STEPS]:
                norms={name:sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5 for name,m in [('backbone',model.backbone),('local',model.readout),('center',center),('group',group)]};assert all(np.isfinite(v) and v>0 for v in norms.values());gradients[arm].append(dict(step=step,norms=norms))
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(center.parameters())+list(group.parameters()),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds[arm]+=time.monotonic()-t;peak=max(peak,torch.cuda.max_memory_allocated());histories[arm].append(value)
        if step%8==0:status('training',step=step,total=STEPS,seconds=seconds,recent_loss={a:float(np.mean(histories[a][-8:])) for a in ARMS});print('STEP',step,flush=True)
        if step in [320,STEPS]:
            for arm in ARMS:assert buffers_digest(models[arm][0].backbone)==bn[arm];checkpoint(arm,step);status('evaluating',arm=arm,step=step);evaluate(arm,step)
    assert sha(SOURCE)==source_hash and sha(sup_path)==sup_hash and all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed)
    summary={a:next(r['summary'] for r in reversed(curves) if r['arm']==a) for a in ARMS};result=dict(summary=summary,histories=histories,gradients=gradients,train_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak,bn_sources_unchanged=True,initial_all92_source_replay_exact=True,production_changed=False);write(OUT/'results.json',result)
    examples=OUT/'examples';examples.mkdir()
    for arm in ARMS:render_predictions(examples,arm,predictions[arm],rows,manifest,selected)
    page=['<!doctype html><meta charset="utf-8"><h1>实例objectness：均衡BCE与均衡focal</h1><p>两组同一个within3200检查点和AdamW，只改变gamma0/2。固定追加640步，不是注意力结构消融。无GT前向、无自动晋升。</p><pre>'+json.dumps({k:v for k,v in result.items() if k!='histories'},indent=2)+'</pre>']
    for p in sorted(examples.glob('*.jpg')):page.append(f'<img style="max-width:100%" src="examples/{p.name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True);status('complete',step=STEPS);state.update(stage='proposal_objectness_control_complete',active_job=None,active_pid=None,next='Read finalbothmetrics/visualallbottom+fixed; noautomaticcontinuationorpromotion.',updated_utc=now().isoformat());state['completed'].append('proposal_objectness_control640');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
