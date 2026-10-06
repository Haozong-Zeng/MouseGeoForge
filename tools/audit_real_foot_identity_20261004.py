"""Diagnostic-only observed-foot permutations; never relabel targets or inference."""
import collections,html,itertools,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.detection_readout_loss import match_detection
from run_depth_pose_ablation import fingerprint,write_json
from run_surface_curve_pilot_20261003 import load_item
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'foot_identity_audit_v1'

def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);rows=[]
    for r in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    counts=collections.defaultdict(lambda:dict(images=0,points=[0]*23));real_ids=[]
    for i,o in enumerate(rows):
        m=o['metadata'];r=m['record']
        if r['kind']!='real':continue
        assert r.get('session') not in ['GR2','UR2','UR3']
        key=r['role']+f'/cam{chr(65+int(m["view"]))}';counts[key]['images']+=1
        for k in range(23):counts[key]['points'][k]+=int(o['point_valid'][0,k])
    arrays=np.load(CYCLE/'pose_query_alignment_v1/predictions.npz');ids=arrays['image_indices'].tolist();det=np.load(CYCLE/'detection_context_roi_v1/predictions.npz');details=[];results=[];visual={}
    for seed in [17,29,43]:
        agg=collections.defaultdict(lambda:collections.defaultdict(list))
        for j,i in enumerate(ids):
            o=rows[i];m=o['metadata'];r=m['record']
            if r['kind']!='real':continue
            p={k:torch.tensor(det[f'seed{seed}__{k}'][i:i+1]) for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
            qs,ms=match_detection(p,o['boxes'][None],(o['box_valid']&o['instance_valid'])[None])[0];assert len(ms)==1
            q=int(qs[0]);inst=int(ms[0]);valid=o['point_valid'][inst].numpy();feet=[k for k in [14,15,16,17] if valid[k]]
            if not feet:continue
            xy=arrays[f'adapted_seed{seed}'][j,q]*o['pixels'].numpy();gt=o['gt_xy'][inst].numpy()*o['pixels'].numpy()
            sem=np.linalg.norm(xy[feet]-gt[feet],axis=-1)
            # LR swaps only when BOTH members of that pair are observed.
            maps=[list(feet)]
            for pair in [[14,15],[16,17]]:
                if all(k in feet for k in pair):
                    maps=maps+[[pair[1] if k==pair[0] else pair[0] if k==pair[1] else k for k in perm] for perm in maps]
            best_lr=min(maps,key=lambda perm:np.linalg.norm(xy[perm]-gt[feet],axis=-1).sum())
            best_any=min(itertools.permutations(feet),key=lambda perm:np.linalg.norm(xy[list(perm)]-gt[feet],axis=-1).sum())
            lr=np.linalg.norm(xy[best_lr]-gt[feet],axis=-1);anyerr=np.linalg.norm(xy[list(best_any)]-gt[feet],axis=-1)
            # Missing labels never treated as evidence that other paws invisible.
            camera=f'cam{chr(65+int(m["view"]))}';tags=['all',camera,r['session'],f'observed{len(feet)}']
            for tag in tags:
                for name,values in [('semantic',sem),('oracle_lr',lr),('oracle_any',anyerr)]:agg[tag][name]+=values.tolist()
            row=dict(seed=seed,observation=i,metadata=m,query=q,observed_ids=[k+1 for k in feet],semantic_mean=float(sem.mean()),oracle_lr_mean=float(lr.mean()),oracle_any_mean=float(anyerr.mean()),oracle_lr_source_ids=[k+1 for k in best_lr],oracle_any_source_ids=[k+1 for k in best_any],lr_changed=best_lr!=feet)
            details.append(row)
            if seed==17:visual[i]=dict(row=row,xy=xy,gt=gt,feet=feet,lr=best_lr)
        summary={k:{name:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for name,v in values.items()} for k,values in agg.items()};results.append(dict(seed=seed,metrics=summary))
    report=dict(results=results,label_counts_by_camera=counts,details=details,contract='Oracle LR/any permutation uses GT after inference for diagnosis only, within observed IDs only. Not valid deployment metrics, not new labels, not evidence annotators swapped limbs. Partial missingness unknown. Point errors at704x768 letterbox. Query chosen by box/objectness Hungarian, not point error. Fixed strict-back adapted-assignment1600 readouts; detector frozen. Real development only.')
    write_json(OUT/'results.json',report)
    selected=[]
    for camera in sorted({v['row']['metadata']['view'] for v in visual.values()}):
        candidates=[i for i,v in visual.items() if v['row']['metadata']['view']==camera];candidates.sort(key=lambda i:visual[i]['row']['semantic_mean']);selected.append(candidates[len(candidates)//2])
    largest=sorted(visual,key=lambda i:visual[i]['row']['semantic_mean']-visual[i]['row']['oracle_lr_mean'],reverse=True)[:2]
    selected+= [i for i in largest if i not in selected]
    page=['<!doctype html><meta charset="utf-8"><title>Foot identity diagnostic</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Foot identity vs localization diagnostic</h1><p>Left semantic predictions (not all-query deployment view); right GT-assisted best left/right swap DIAGNOSTIC ONLY. Coordinates unchanged; IDs may change only if both feet in the pair observed. Yellow crosses=labels. No labels edited. One median-error frame per camera plus two largest oracle LR improvements. Error metrics are development-only and oracle is not a model result.</p>'];manifest=[]
    colors={14:'#ff7777',15:'#55ccff',16:'#77ff77',17:'#ee88ff'}
    for index,i in enumerate(selected):
        v=visual[i];o=rows[i];m=o['metadata'];r=m['record'];cache=ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else load_item(r)
        rgb=(item['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=rgb.shape[:2];pw=440;ph=round(h*pw/w);scale=np.array([pw/w,ph/h]);im=Image.fromarray(rgb).resize((pw,ph));canvas=Image.new('RGB',(2*pw,ph+30),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,perm in enumerate([v['feet'],v['lr']]):
            canvas.paste(im,(col*pw,30));draw.text((col*pw+5,5),'semantic' if col==0 else 'oracle LR - NOT inference',fill='white')
            for target,source in zip(v['feet'],perm):
                x,y=v['xy'][source]*scale+[col*pw,30];gx,gy=v['gt'][target]*scale+[col*pw,30];color=colors[target];draw.line((x,y,gx,gy),fill=color,width=1);draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);draw.text((x+4,y),str(target+1),fill=color);draw.line((gx-4,gy,gx+4,gy),fill='#ffee55',width=2);draw.line((gx,gy-4,gx,gy+4),fill='#ffee55',width=2)
        name=f'{index:02d}_cam{chr(65+int(m["view"]))}.jpg';canvas.save(OUT/name,quality=95);manifest.append(dict(file=name,**v['row']));page.append('<h2>'+html.escape(str(v['row']))+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',manifest);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)

if __name__=='__main__':main()
