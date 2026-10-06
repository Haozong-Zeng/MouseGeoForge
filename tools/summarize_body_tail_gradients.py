"""CPU-only descriptive follow-up; no new sample selection or optimization."""
import json
import numpy as np
import torch
from run_depth_pose_ablation import ROOT,write_json,fingerprint


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'output/body_tail_gradient_diagnostic_v1'
    out=ROOT/'output/body_tail_gradient_summary_v1';out.mkdir(exist_ok=False)
    assert json.loads((source/'verification.json').read_text())['passed']
    rows=json.loads((source/'records.json').read_text())
    gradients=torch.load(source/'gradients.pt',map_location='cpu',weights_only=True)
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');results={}
    for i,state in enumerate(('initial','mixed400')):
        for j,field in enumerate(('variant','source_group')):
            groups=sorted({r[field] for r in rows if r['state']==state});means={}
            for group in groups:
                selected=[r for r in rows if r['state']==state and r[field]==group and r['schedule_count']]
                total=sum((gradients[state+'/'+r['key']]['body'].double()+gradients[state+'/'+r['key']]['tail'].double())*r['schedule_count'] for r in selected)
                means[group]=total/sum(r['schedule_count'] for r in selected)
            values=np.array([[float(means[a]@means[b]/(means[a].norm()*means[b].norm())) for b in groups] for a in groups])
            results[state+'/'+field]=dict(groups=groups,cosines=values.tolist())
            ax=axes[i,j];im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu')
            labels=[g.replace('crim13/021709_A25_Block22_BCfe1_t','CRIM13').replace('mouse_climb_wall_91','ClimbWall').replace('moseq_sample10','MoSeq') for g in groups]
            ax.set_xticks(range(len(groups)),labels,rotation=20);ax.set_yticks(range(len(groups)),labels)
            ax.set_title(state+' / '+field)
            for a in range(len(groups)):
                for b in range(len(groups)):ax.text(b,a,f'{values[a,b]:+.3f}',ha='center',va='center',color='white' if abs(values[a,b])>.65 else 'black')
    fig.colorbar(im,ax=axes,label='Cosine of schedule-weighted total-loss gradients',shrink=.75)
    fig.suptitle('Training-only gradient directions (not AdamW updates or causal proof)')
    fig.savefig(out/'group_gradient_cosines.png',dpi=140);plt.close(fig)
    write_json(out/'summary.json',results)
    write_json(out/'verification.json',dict(passed=True,training_steps=0,new_image_reads=0,
        source_hashes={p.name:fingerprint(p) for p in (source/'records.json',source/'gradients.pt',source/'verification.json')},
        code_sha256=fingerprint(ROOT/'tools/summarize_body_tail_gradients.py'),warning='Post-hoc descriptive grouping, not pre-registered causal test.'))
    print('COMPLETED')


if __name__=='__main__':main()
