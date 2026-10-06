"""Read-only annotation/score decomposition, never changes candidate predictions."""
import json
import time
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT, fingerprint, write_json, instance_targets_from_batch
from train_cylinder_view_balance import oriented_batch, scene_digest
from mousegeoforge.dense_center_probe import candidate_metrics

OUT=ROOT/'output/center_supervision_geometry_audit_v1'
SOURCE=ROOT/'output/dense_center_context_diversity_v1/on'


def error_type(score, iou):
    if score>=.5 and iou>=.5:return 'found'
    if score<.5 and iou>=.5:return 'score_only'
    if score>=.5:return 'geometry_only'
    return 'score_and_geometry'


def run():
    started=time.monotonic();torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve prior audit')
    spec=json.loads((ROOT/'output/dual_pose_diversity_seed17_v1/protocol.json').read_text())
    source=json.loads((SOURCE/'report.json').read_text());assert source['passed']
    for p,h in source['code_sha256'].items():assert fingerprint(ROOT/p)==h
    inputs={str(p):fingerprint(p) for p in (SOURCE/'report.json',SOURCE/'protocol.json',ROOT/'output/dual_pose_diversity_seed17_v1/protocol.json')}
    write_json(OUT/'protocol.json',dict(input_sha256=inputs,code_sha256=fingerprint(ROOT/'tools/audit_center_supervision_geometry.py'),
        scope='No model forward/optimization.16 dual train inputs annotation audit plus all88 saved development candidates.',
        thresholds=dict(score=.5,iou=.5,foreground_fraction=.5),definition='GT center pixel uses floor(cx*W,cy*H); native cell foreground fraction uses area downsampling of own instance mask. No GT candidate selection or threshold modification.',
        max_seconds=600,production_promoted=False))
    rows=[];visuals=[]
    train=list(dict.fromkeys(x[0] for x in spec['arms']['diversity_on'] if spec['records'][x[0]]['expected_instances']==2))
    entries=[('train_dual',None,k) for k in train]+[('dev',i,e['key']) for i,e in enumerate(spec['evaluation'])]
    try:
        for split,index,key in entries:
            if time.monotonic()-started>600 or time.time()>=1789194663:raise RuntimeError('Audit time cap')
            r=spec['records'][key];assert scene_digest(ROOT/r['scene'])==spec['source_sha256'][r['scene']]
            batch=oriented_batch(r,num_views=len(r['views']));t=instance_targets_from_batch(batch)
            h,w=batch['images'].shape[-2:];gh,gw=h//16,w//16
            saved=None
            if split=='dev':
                path=SOURCE/f'candidates_{index:03d}.pt';inputs[str(path)]=fingerprint(path)
                saved=torch.load(path,map_location='cpu',weights_only=True)
                assert saved['record']==r
                torch.testing.assert_close(saved['boxes'],t['boxes_cxcywh'][0],atol=0,rtol=0)
                torch.testing.assert_close(saved['valid'],t['instance_valid'][0],atol=0,rtol=0)
            for v,role in enumerate(r['physical_roles']):
                ids=torch.where(t['instance_valid'][0,v])[0]
                boxes=t['boxes_cxcywh'][0,v,ids];masks=t['masks'][0,v,ids].float()
                assert masks.shape[-2:]==(h,w)
                native=F.interpolate(masks[:,None],size=(gh,gw),mode='area')[:,0]
                matches={}
                if saved is not None:
                    d=saved['decoded'];metrics=candidate_metrics(d['boxes'][v],d['scores'][v],boxes)
                    matches={m['target']:m for m in metrics['matches']}
                for j,b in enumerate(boxes):
                    px=min(w-1,int(b[0]*w));py=min(h-1,int(b[1]*h))
                    ix=min(gw-1,int(b[0]*gw));iy=min(gh-1,int(b[1]*gh))
                    row=dict(split=split,index=index,frame=r['frame'],role=role,view=r['views'][v],
                        dual=r['expected_instances']==2,target=j,center_foreground=bool(masks[j,py,px]>.5),
                        native_foreground_fraction=float(native[j,iy,ix]),cell=[ix,iy])
                    if saved is not None:
                        m=matches[j];heat=d['heatmap'][v]
                        row.update(error=error_type(m['score'],m['iou']),score=m['score'],iou=m['iou'],
                            gt_cell_score=float(heat[iy,ix]),near3x3_max=float(heat[max(0,iy-1):iy+2,max(0,ix-1):ix+2].max()))
                    rows.append(row)
                if split=='dev' and index in (0,7,10):visuals.append((r['frame'],role,native.numpy(),d['heatmap'][v].numpy(),[x for x in rows if x['split']=='dev' and x['index']==index and x['view']==r['views'][v]]))
            write_json(OUT/'status.json',dict(state='auditing',inputs_processed=entries.index((split,index,key))+1,elapsed_s=time.monotonic()-started))
        groups=defaultdict(list)
        for r in rows:
            group='train_dual' if r['split']=='train_dual' else 'dev_dual' if r['dual'] else 'dev_single'
            groups[group].append(r);groups[group+'/'+r['role']].append(r)
        summary={}
        for name,rr in groups.items():
            values=[r['native_foreground_fraction'] for r in rr]
            item=dict(gt=len(rr),center_on_background=sum(not r['center_foreground'] for r in rr),
                cell_foreground_below_half=sum(x<.5 for x in values),mean_cell_foreground=float(np.mean(values)))
            if rr[0]['split']=='dev':
                item['errors']=dict(Counter(r['error'] for r in rr))
                item['by_center_foreground']={str(f):dict(Counter(r['error'] for r in rr if r['center_foreground']==f)) for f in (False,True)}
                item['gt_cell_score_median']=float(np.median([r['gt_cell_score'] for r in rr]))
                item['gt_cell_score_ge_half']=sum(r['gt_cell_score']>=.5 for r in rr)
                item['near3x3_score_ge_half']=sum(r['near3x3_max']>=.5 for r in rr)
            summary[name]=item
        fig,axes=plt.subplots(len(visuals),2,figsize=(11,3*len(visuals)),layout='constrained')
        for i,(frame,role,masks,heat,rr) in enumerate(visuals):
            for col,field in enumerate((masks.max(0),heat)):
                axes[i,col].imshow(field,vmin=0,vmax=1,cmap='gray' if col==0 else 'magma')
                for r in rr:
                    x,y=r['cell'];axes[i,col].plot(x,y,'+',color='cyan')
                    axes[i,col].text(x+1,y,f"GT{r['target']}",color='cyan',fontsize=8)
                axes[i,col].set_title(f"Frame{frame} {role}: "+('GT mask fraction / GT center' if col==0 else 'Predicted center heat / GT center'))
                axes[i,col].axis('off')
        fig.savefig(OUT/'center_mask_heat.png',dpi=120);plt.close(fig)
        for p,h in inputs.items():assert fingerprint(__import__('pathlib').Path(p))==h
        write_json(OUT/'report.json',dict(passed=True,summary=summary,rows=rows,input_sha256=inputs,model_forwards=0,optimizer_steps=0,
            limitations=['GT-center score/nearby maxima are annotation diagnostics, not oracle detections or alternative evaluation.',
                         'Foreground correlation is descriptive, not causal evidence.'],production_promoted=False))
        write_json(OUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-started))
        print(json.dumps(summary,indent=2))
    except BaseException as error:
        write_json(OUT/'status.json',dict(state='failed',error=repr(error),elapsed_s=time.monotonic()-started));raise


if __name__=='__main__':run()
