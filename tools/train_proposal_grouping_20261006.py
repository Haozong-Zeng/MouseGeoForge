"""Matched predicted-center grouping, three-FIT overfit then gated mixed pilot."""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,time,random,traceback,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest,atomic_save
from train_dense_center_warmup_20261005 import read,write,now
from train_local_instance_grouping_20261005 import metrics,summarize,render_predictions
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,dense_center_loss,decode_centers
from mousegeoforge.models.proposal_instance_grouping import ProposalInstanceGrouping,proposal_grouping_loss,decode_proposal_grouped
from mousegeoforge.models.local_instance_grouping import rotate_mask_centroids
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
C=ROOT/'output/local_grouping_cycle_20261005';INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'
SOURCE=C/'dense_center_loss_focal_9216_v1/center_step9216.pt';ARMS=['independent','within_instance'];OUT=None

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['overfit128','pilot640','continue1920','continue3200'],required=True);args=parser.parse_args()
    overfit=args.stage=='overfit128';continuing=args.stage.startswith('continue');initial_step={'continue1920':640,'continue3200':1920}.get(args.stage,0)
    steps={'overfit128':128,'pilot640':640,'continue1920':1920,'continue3200':3200}[args.stage];checks=[initial_step,initial_step+640,steps] if continuing else ([0,32,64,128] if overfit else [0,160,320,640])
    OUT=C/f'proposal_grouping_{args.stage}_v1';plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    assert json.loads((C/'center_proposal_budget_audit_v1/status.json').read_text())['stage']=='complete'
    if not overfit:
        gate=json.loads((C/'proposal_grouping_overfit128_v1/results.json').read_text());assert all(gate['overfit_gate'].values()),'Resolve overfit count/point rejection first'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='proposal_grouping_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid(),updated_utc=now().isoformat(),next='Wait for matched proposal grouping; do not start second GPU job.');write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup_path=C/'instance_supervision_audit_v1/evaluation_instance_targets.pt';sup={r['index']:r for r in read(sup_path)};sup_hash=sha(sup_path);source_hash=sha(SOURCE)
    assert source_hash=='b2e96a4d3099695245df8cc8c60c80cc48bd586e0ba5d175367ab90067674c5d'
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    pools={b:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']==b] for b in [True,False]}
    syn={k:[i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k] for k in ['single','dual']}
    rng=random.Random(17);rot=random.Random(43)
    schedule=[fixed[:] for _ in range(steps)] if overfit else [[rng.choice(pools[j%2==0]),rng.choice(syn['single']),rng.choice(syn['dual'])] for j in range(steps)]
    turns=[[0,0,0] for _ in range(steps)] if overfit else [[rot.randrange(4) for _ in range(3)] for j in range(steps)]
    previous=C/('proposal_grouping_continue1920_v1' if initial_step==1920 else 'proposal_grouping_pilot640_v1');resume_hashes={};resume_predictions={}
    if continuing:
        assert json.loads((previous/'status.json').read_text())['stage']=='complete'
        prior_protocol=json.loads((previous/'protocol.json').read_text())
        assert schedule[:initial_step]==prior_protocol['schedule'] and turns[:initial_step]==prior_protocol['rotations']
        assert sha(ROOT/'mousegeoforge/models/proposal_instance_grouping.py')==prior_protocol['module_sha256']
    evaluation=fixed if overfit else fixed+[i for i,r in rows.items() if r['metadata']['stage']=='calibration']
    selected=fixed if overfit else list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i in evaluation if rows[i]['metadata']['stage']=='calibration' and rows[i]['metadata']['kind']=='real' and rows[i]['metadata']['physical_bottom']]))
    needed=set(evaluation)|{i for ids in schedule for i in ids}
    for i in needed:
        assert rows[i]['metadata']==sup[i]['metadata'] and rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3']
        assert torch.equal(sup[i]['instance_valid'],sup[i]['centroid_valid']) and bool(sup[i]['detection_annotation_complete'])
    shared=ProposalInstanceGrouping().state_dict();atomic_save(OUT/'shared_group_initial.pt',shared)
    models={};opts={};bn={};initial_hashes={}
    cp=read(SOURCE);old=read(SOURCE.parent/'step9216_predictions.pt')
    for arm in ARMS:
        backbone,_=build_backbone();model=MixedLocalEvidence(backbone,adapt_backbone=True).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.train()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);group=ProposalInstanceGrouping(interaction=arm=='within_instance').cuda();group.load_state_dict(shared)
        models[arm]=(model,center,group);bn[arm]=buffers_digest(backbone);initial_hashes[arm]=[digest(m) for m in models[arm]]
        opts[arm]=torch.optim.AdamW([dict(params=group.parameters(),lr=3e-4),dict(params=center.parameters(),lr=3e-4),dict(params=model.readout.parameters(),lr=1e-4),dict(params=backbone.parameters(),lr=1e-5)],weight_decay=1e-4)
        if continuing:
            path=previous/f'{arm}_step{initial_step:04d}.pt';resume_hashes[arm]=sha(path);resume=read(path);assert resume['step']==initial_step and resume['arm']==arm and resume['source_sha256']==source_hash
            model.backbone.load_state_dict(resume['backbone']);model.readout.load_state_dict(resume['readout']);center.load_state_dict(resume['center']);group.load_state_dict(resume['grouping']);opts[arm].load_state_dict(resume['optimizer'])
            assert [g['lr'] for g in opts[arm].param_groups]==[3e-4,3e-4,1e-4,1e-5]
            assert buffers_digest(backbone)==bn[arm]
            resume_predictions[arm]=read(previous/f'{arm}_step{initial_step:04d}_predictions.pt');del resume
    del cp
    assert initial_hashes[ARMS[0]]==initial_hashes[ARMS[1]]
    protocol=dict(stage=args.stage,steps=steps,initial_step=initial_step,resume_sha256=resume_hashes,evaluation_steps=checks,source_checkpoint=str(SOURCE.relative_to(ROOT)),source_sha256=source_hash,supervision_sha256=sup_hash,
        seed=17,rotations_seed=43,schedule=schedule,rotations=turns,initial_model_hashes=initial_hashes,
        initialization=(f'Both resume corresponding{initial_step} weights and AdamW exactly; schedule/rotationsprefix{initial_step} exact, sameLR/loss/architecture. Only additional{steps-initial_step}updates.' if continuing else 'Both fresh same focal9216 backbone/readout/center plus identical newgroup state. FreshAdamW, not source optimizer. Pilot also freshfromsamefocal9216, NOToverfitweights.'),
        architecture='8predictedcenterregionalpeaks without rawscoregate; visual/xy/heatmapscore/rank embedding +centercontextattention;5typesTop6pointcandidates+relativegeometry/null;learnedobjectness afterpointcontext. NoGTforward.',
        difference='Only 5pointtype selfattention full vsdiagonal within eachcenter. Sameparams and allothermechanisms.',
        loss='localCE + originaldensecenterfocal + pointassignmentCE + balancedpositive/negativeproposalBCE. GTmatchingONLYloss: maximize center matches<=64px then minimizedistance, onepositiveperGT. Missingcenterproposalsreported, notinjected. Unknownpointsignored; unknowninstancesnevernegative.',
        inference='Learnedobjectsigmoid>=.5 andatmost2topobjects, no distanceNMS beyond8pxproposalextraction; GTfree pertypeone-to-onepointcandidate/null selection. NoGTcount. Rejection not visibility.',
        lrs=dict(group=3e-4,center=3e-4,local=1e-4,backbone=1e-5),fixed_bn=True,
        caveat='Exploratoryrevision afterCAL; notfairvsoldquery640 dueextra backbonepretrain. Candidatepoolselectedhistorically, notunbiasedbenchmark. FIT3overfitnotgeneralization.',
        overfit_gate='Final allFITcounts1/1/2,zero64pxFPFN,and knownpointjoint64PCK16>=.8 for everydomain; botharms beforemixedpilot.',
        code_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'mousegeoforge/models/proposal_instance_grouping.py'),deadline=plan['training_stop_utc'],production_changed=False)
    write(OUT/'protocol.json',protocol)
    images={}
    for i in sorted(needed):
        if now()>=deadline:raise TimeoutError('Input cutoff')
        images[i]=load_rgb(manifest[i]).cuda()
    with torch.no_grad():
        for arm,(model,center,group) in models.items():
            if continuing:continue
            for i in fixed:
                dec=decode_centers(center(model(images[i],manifest[i]['image_hw']),manifest[i]['image_hw']),manifest[i]['image_hw'])
                assert all(torch.equal(v.cpu(),old[i]['decoded'][k]) for k,v in dec.items()),(arm,i,'source replay')
    del old
    def compute(arm,i,k=0):
        model,center,group=models[arm];r=rows[i];s=sup[i];hw=manifest[i]['image_hw'];x=images[i];xy=r['target'].cuda();eligible=r['eligible'].cuda();gt=s['mask_centroid'].cuda();cv=s['centroid_valid'].cuda()
        if k:x,xy,hw=quarter_turn(x,xy,eligible,hw,k);gt=rotate_mask_centroids(gt,cv,k)
        evidence=model(x,hw);co=center(evidence,hw);out=group(evidence,co,hw)
        losses=proposal_grouping_loss(out,xy,eligible,s['instance_valid'].cuda(),gt,cv,hw,annotation_complete=True)
        losses['local']=point_evidence_loss(evidence,xy,eligible,hw);losses['center']=dense_center_loss(co,gt,cv,hw,annotation_complete=True)
        losses['total']=losses['total']+losses['local']+losses['center'];return out,losses
    curves=[];predictions={};histories={a:[] for a in ARMS};seconds={a:0. for a in ARMS};gradients={a:[] for a in ARMS};peak=0
    @torch.no_grad()
    def evaluate(arm,step):
        details=[];pred={}
        for i in evaluation:
            if now()>=deadline:raise TimeoutError('Evaluation cutoff')
            out,loss=compute(arm,i);decoded=decode_proposal_grouped(out);cpu={k:v.cpu() for k,v in out.items()};dec={k:v.cpu() for k,v in decoded.items()}
            if continuing and step==initial_step:
                prior=resume_predictions[arm][i]
                assert all(torch.equal(v,prior['output'][k]) for k,v in cpu.items()),(arm,i,'resume output mismatch')
                assert all(torch.equal(v,prior['decoded'][k]) for k,v in dec.items()),(arm,i,'resume decode mismatch')
            details.append(dict(index=i,metadata=rows[i]['metadata'],loss={k:float(v) for k,v in loss.items() if v.ndim==0},metrics=metrics(cpu,dec,rows[i],sup[i],manifest[i]['image_hw'])));pred[i]=dict(output=cpu,decoded=dec)
        summary=summarize(details);curves.append(dict(arm=arm,step=step,summary=summary));write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',details);atomic_save(OUT/f'{arm}_step{step:04d}_predictions.pt',pred);predictions[arm]=pred
    def checkpoint(arm,step):
        model,center,group=models[arm];atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(step=step,arm=arm,source_sha256=source_hash,backbone={k:v.cpu() for k,v in model.backbone.state_dict().items()},readout={k:v.cpu() for k,v in model.readout.state_dict().items()},center={k:v.cpu() for k,v in center.state_dict().items()},grouping={k:v.cpu() for k,v in group.state_dict().items()},optimizer=opts[arm].state_dict()))
    start=time.monotonic()
    for arm in ARMS:status('evaluating',arm=arm,step=initial_step);evaluate(arm,initial_step)
    resume_predictions.clear()
    for step,ids in enumerate(schedule[initial_step:],initial_step+1):
        for arm in ARMS:
            if now()>=deadline:
                for a in ARMS:checkpoint(a,initial_step+len(histories[a]))
                raise TimeoutError('Training cutoff')
            model,center,group=models[arm];opt=opts[arm];opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic();value=0.
            for slot,i in enumerate(ids):
                out,losses=compute(arm,i,turns[step-1][slot]);loss=losses['total']/3;assert torch.isfinite(loss);loss.backward();value+=float(loss.detach());del out,losses,loss
            if step in [initial_step+1,initial_step+2,steps]:
                norms={name:sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5 for name,m in [('backbone',model.backbone),('local',model.readout),('center',center),('group',group)]};assert all(np.isfinite(v) and v>0 for v in norms.values());gradients[arm].append(dict(step=step,norms=norms))
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(center.parameters())+list(group.parameters()),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();seconds[arm]+=time.monotonic()-t;peak=max(peak,torch.cuda.max_memory_allocated());histories[arm].append(value)
        if step%8==0:status('training',step=step,total=steps,seconds=seconds,recent_loss={a:float(np.mean(histories[a][-8:])) for a in ARMS});print('STEP',step,flush=True)
        if step in checks:
            for arm in ARMS:assert buffers_digest(models[arm][0].backbone)==bn[arm];checkpoint(arm,step);status('evaluating',arm=arm,step=step);evaluate(arm,step)
    assert sha(SOURCE)==source_hash and sha(sup_path)==sup_hash and all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed)
    if continuing:assert all(sha(previous/f'{a}_step{initial_step:04d}.pt')==resume_hashes[a] for a in ARMS)
    summary={a:next(r['summary'] for r in reversed(curves) if r['arm']==a) for a in ARMS}
    gate={a:all(summary[a]['fit'][k]['count_accuracy']==1 and summary[a]['fit'][k]['fp64']==0 and summary[a]['fit'][k]['fn64']==0 and summary[a]['fit'][k]['joint64_pck16']>=.8 for k in ['real','single','dual']) for a in ARMS}
    result=dict(summary=summary,overfit_gate=gate if overfit else {},histories=histories,gradients=gradients,train_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak,bn_sources_unchanged=True,source_fixed3_center_replay_exact=not continuing,resume_all92_replay_exact=continuing,production_changed=False);write(OUT/'results.json',result)
    examples=OUT/'examples';examples.mkdir()
    for arm in ARMS:render_predictions(examples,arm,predictions[arm],rows,manifest,selected)
    page=[f'<!doctype html><meta charset="utf-8"><title>预测中心分组对照</title><h1>{args.stage}: 独立点与实例内交互</h1><p>完整实例输出，不使用GT中心或数量。未知点不计错，漏检/拒绝用图像对角线惩罚。候选oracle不是最终精度。</p><pre>'+json.dumps({k:v for k,v in result.items() if k!='histories'},indent=2)+'</pre>']
    for p in sorted(examples.glob('*.jpg')):page.append(f'<img style="max-width:100%" src="examples/{p.name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True);status('complete',step=steps);state.update(stage='proposal_grouping_complete',active_job=None,active_pid=None,next='Review both finalmetrics/allimages. Overfit128 gate required beforefresh mixed640. No automaticpromotion.',updated_utc=now().isoformat());state['completed'].append('proposal_grouping_'+args.stage);write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT is not None and OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
