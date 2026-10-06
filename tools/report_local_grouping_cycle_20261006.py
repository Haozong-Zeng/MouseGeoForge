"""Read-only saved-output decomposition and scientific curves; never retrains."""
import sys,json,hashlib,collections,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from mousegeoforge.models.proposal_instance_grouping import decode_proposal_grouped
C=ROOT/'output/local_grouping_cycle_20261005';DEV=C/'proposal_final_development144_v1';OUT=C/'final_error_report_v1';INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_development_inputs_v1'
ARMS=['independent3200','within3200','bce_add640','focal_add640']
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((DEV/'status.json').read_text())['stage']=='complete' and json.loads((C/'training_frozen.json').read_text())['training_closed_for_this_cycle']
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};sup=read(DEV/'evaluation_instance_targets.pt');manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};reports={};hashes={}
    for arm in ARMS:
        path=DEV/f'{arm}_predictions.pt';hashes[arm]=sha(path);pred=read(path);details=[]
        original={r['index']:r for r in json.loads((DEV/f'{arm}_details.json').read_text())}
        for i,item in pred.items():
            out,d=item['output'],item['decoded'];replay=decode_proposal_grouped(out)
            for k,v in d.items():
                if 'probability' in k:torch.testing.assert_close(v,replay[k],atol=1e-7,rtol=1e-6)
                else:assert torch.equal(v,replay[k]),(arm,i,k)
            r=rows[i];s=sup[i];scale=torch.tensor([manifest[i]['image_hw'][1],manifest[i]['image_hw'][0]]);gt=s['instance_valid'].nonzero().flatten();active=d['instance_active'].nonzero().flatten();available=out['center_available'].nonzero().flatten();matches={};coverage=0
            if len(available) and len(gt):
                cd=torch.cdist(out['centers'][available]*scale,s['mask_centroid'][gt]*scale);cost=cd.masked_fill(cd>64,1e6);a,b=linear_sum_assignment(cost.numpy());coverage=sum(float(cd[x,y])<=64 for x,y in zip(a,b))
            if len(active) and len(gt):
                dist=torch.cdist(d['centers'][active]*scale,s['mask_centroid'][gt]*scale);a,b=linear_sum_assignment(dist.numpy());matches={int(gt[y]):int(active[x]) for x,y in zip(a,b) if float(dist[x,y])<=64}
            points=[]
            for g,k in r['eligible'].nonzero().tolist():
                cd=((out['candidate_xy'][k]-r['target'][g,k])*scale).norm(dim=-1).masked_fill(~out['candidate_available'][k],float('inf'));has_candidate=bool((cd<=16).any());category='center_missing';harmed=helped=False;raw_correct=False
                if g in matches:
                    q=matches[g];accepted=bool(d['accepted'][q,k]);correct=accepted and float(((d['xy'][q,k]-r['target'][g,k])*scale).norm())<=16
                    raw=int(out['assignment_logits'][q,k].argmax());raw_correct=raw<6 and bool(out['candidate_available'][k,raw]) and float(cd[raw])<=16
                    harmed=raw_correct and not correct;helped=correct and not raw_correct
                    category='correct' if correct else ('candidate_missing' if not has_candidate else ('rejected_with_candidate' if not accepted else 'wrong_assignment_with_candidate'))
                points.append(dict(instance=g,keypoint=k,category=category,candidate_exists16=has_candidate,exclusivity_harmed=harmed,exclusivity_helped=helped,raw_argmax_correct=raw_correct))
            assert sum(p['category']=='correct' for p in points)==sum(p['joint64_error']<=16 for p in original[i]['metrics']['points'])
            details.append(dict(index=i,metadata=r['metadata'],instances=len(gt),proposal_coverage64=coverage,points=points))
        summary={}
        for kind in ['real','single','dual','real_bottom_feet']:
            ds=[d for d in details if d['metadata']['kind']==kind or kind=='real_bottom_feet' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']];pts=[p for d in ds for p in d['points'] if kind!='real_bottom_feet' or p['keypoint']>0]
            summary[kind]=dict(images=len(ds),instances=sum(d['instances'] for d in ds),proposal_covered64=sum(d['proposal_coverage64'] for d in ds),points=len(pts),categories=dict(collections.Counter(p['category'] for p in pts)),candidate_exists16=sum(p['candidate_exists16'] for p in pts),exclusivity_harmed=sum(p['exclusivity_harmed'] for p in pts),exclusivity_helped=sum(p['exclusivity_helped'] for p in pts),raw_argmax_correct=sum(p['raw_argmax_correct'] for p in pts))
        assert sha(path)==hashes[arm];reports[arm]=dict(summary=summary,details=details)
    result=dict(arms=reports,source_hashes=hashes,model_inference_count=0,optimizer_steps=0,decode_replay=True,caveat='Categories are ordered: center missing, then candidate missing, rejection or wrong assignment. Candidate coverage is GT oracle, NOT model accuracy. Exclusivity harm/help uses unchanged predicted active instances and same GT posthocmatching; removing exclusivity is NOT a deployable improvement. Unknown points ignored. DEV previously reused; not external test.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves={}
    for folder in ['proposal_grouping_pilot640_v1','proposal_grouping_continue1920_v1','proposal_grouping_continue3200_v1']:
        for c in json.loads((C/folder/'curves.json').read_text()):curves[c['arm'],c['step']]=c['summary']['calibration']
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,(domain,title) in enumerate([('real_bottom_feet','Real bottom (16 images)'),('single','Synthetic single (38 images)'),('dual','Synthetic dual (19 images)')]):
        for arm,color in [('independent','#1479b8'),('within_instance','#db7620')]:
            steps=sorted(s for a,s in curves if a==arm);axs[0,col].plot(steps,[curves[arm,s][domain].get('joint64_pck16',0)*100 for s in steps],'-o',ms=3,label=arm,color=color);axs[1,col].plot(steps,[curves[arm,s][domain]['fp64'] for s in steps],'-o',ms=3,label=arm,color=color)
        axs[0,col].set_title(title);axs[0,col].set_ylim(0,100);axs[0,col].set_ylabel('Joint64 PCK16 (%)');axs[1,col].set_ylabel('False positives @64px');axs[1,col].set_xlabel('Grouping updates (common focal9216 pretrained)')
        for ax in axs[:,col]:ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle('CAL trajectories: no consistent all-domain convergence; fixed endpoints reported');fig.savefig(OUT/'calibration_curves.png',dpi=150);plt.close(fig)
    dev=json.loads((DEV/'results.json').read_text())['summary'];fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(4);labels=['Real','Bottom feet','Synth single','Synth dual'];domains=['real','real_bottom_feet','single','dual']
    for j,arm in enumerate(ARMS):
        axs[0].bar(x+(j-1.5)*.19,[dev[arm][d]['joint64_pck16']*100 for d in domains],width=.19,label=arm);axs[1].bar(x+(j-1.5)*.19,[dev[arm][d]['fp64'] for d in domains],width=.19,label=arm)
    for ax in axs:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylim(0,100);axs[0].set_ylabel('Joint64 PCK16 (%)');axs[1].set_ylabel('False positives @64px');axs[0].legend(fontsize=8);fig.suptitle('Reused DEV144 fixed endpoints; bottom only8 images/32feet, NOT independent externaltest');fig.savefig(OUT/'development_endpoints.png',dpi=150);plt.close(fig)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>只读误差分解与曲线</h1><p>候选oracle不代表预测精度；互斥损害不意味着应关闭双鼠互斥。无模型推理或训练，不据此调阈值。</p><img style="max-width:100%" src="calibration_curves.png"><img style="max-width:100%" src="development_endpoints.png"><pre>'+json.dumps({a:r['summary'] for a,r in reports.items()},indent=2)+'</pre>',encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps(dict(stage='complete')))
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;shutil.copytree(OUT,dest)
    print(json.dumps({a:r['summary'] for a,r in reports.items()}))
if __name__=='__main__':main()
