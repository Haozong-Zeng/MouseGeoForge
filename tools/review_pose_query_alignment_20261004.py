"""Fixed confidence-selected queries: initial/control/new-association comparison."""
import argparse,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'pose_query_alignment_v1';OUT=CYCLE/'review_pose_query_alignment_v1'

def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--back-control',action='store_true');parser.add_argument('--decoder-control',action='store_true');parser.add_argument('--bottom-balanced',action='store_true');parser.add_argument('--extend-decoder',action='store_true');args=parser.parse_args()
    assert not args.extend_decoder or args.bottom_balanced
    if args.bottom_balanced:args.decoder_control=True
    assert not (args.back_control and args.decoder_control)
    if args.back_control:BASE=CYCLE/'real_back_weight_control_v1';OUT=CYCLE/'review_real_back_weight_v1'
    if args.decoder_control:BASE=CYCLE/'decoder_unfreeze_control_v1';OUT=CYCLE/'review_decoder_unfreeze_v1'
    if args.bottom_balanced:BASE=CYCLE/'bottom_balanced_decoder_v1';OUT=CYCLE/'review_bottom_balanced_v1'
    if args.extend_decoder:BASE=CYCLE/'bottom_balanced_extended2400_v1';OUT=CYCLE/'review_bottom_balanced_extended2400_v1'
    assert (BASE/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    results=json.loads((BASE/'results.json').read_text());rows=[]
    for r in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    control_key='frozen_seed17' if args.decoder_control else 'downweight_seed17' if args.back_control else 'original_seed17'
    new_key='unfrozen_seed17' if args.decoder_control else 'omit_seed17' if args.back_control else 'adapted_seed17'
    if args.bottom_balanced:
        uniform_base=CYCLE/('decoder_unfrozen_extended2400_v1' if args.extend_decoder else 'decoder_unfreeze_control_v1')
        with np.load(BASE/'predictions.npz') as a:ids=a['image_indices'].tolist();new=a['balanced_seed17'].copy()
        with np.load(uniform_base/'predictions.npz') as a:assert ids==a['image_indices'].tolist();control=a['unfrozen_seed17'].copy()
        previous=json.loads((uniform_base/'results.json').read_text());results['results']=[dict(r,arm='uniform') for r in previous['results'] if r['arm']=='unfrozen']+results['results']
    else:
        with np.load(BASE/'predictions.npz') as a:ids=a['image_indices'].tolist();control=a[control_key].copy();new=a[new_key].copy()
    if args.back_control:
        with np.load(CYCLE/'pose_query_alignment_v1/predictions.npz') as a:assert ids==a['image_indices'].tolist();initial=a['adapted_seed17'].copy()
        strict=json.loads((CYCLE/'pose_query_alignment_v1/results.json').read_text());results['results']=[dict(r,arm='strict1') for r in strict['results'] if r['arm']=='adapted']+results['results']
    else:
        with np.load(CYCLE/'detection_pose_context_roi_audit_v1/predictions.npz') as a:assert ids==a['image_indices'].tolist();initial=a['seed17'][:,1].copy()
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:boxes=a['seed17__instance_boxes_cxcywh'].copy();logits=a['seed17__instance_objectness_logits'].copy()
    selected=[r['observation'] for r in json.loads((CYCLE/'review_box_contract_v1/manifest.json').read_text()) if r['observation'] in ids];seen=set()
    for i in ids:
        m=rows[i]['metadata']
        if m['record']['kind']=='real' and m['view'] not in seen:
            seen.add(m['view'])
            if i not in selected:selected.append(i)
    page=['<!doctype html><meta charset="utf-8"><title>Pose query alignment</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}td,th{padding:8px}</style><h1>Frozen detector, pose query-assignment control</h1><p>Left initial6400 readout; middle +1600steps original TRAIN query assignment; right +1600steps adapted TRAIN query assignment. Same frozen ROI boxes/scores for all three. All confidence>=.5queries, no GT filtering/noNMS. Colored circles=predicted IDs1/15/16/17/18; yellow crosses=observed labels. Colors are query indices, not persistent animal IDs. Seed17 images, all3seeds below. Development data only.</p><table><tr><th>Arm</th><th>Seed</th><th>Real reliable px</th><th>Real gatedPCK32</th><th>Dual px</th><th>Dual gatedPCK32</th></tr>']
    for r in results['results']:
        real=r['metrics']['real_validation/reliable'];dual=r['metrics']['synthetic_validation/dual'];page.append(f'<tr><td>{r["arm"]}</td><td>{r["seed"]}</td><td>{real["error_mean"]:.2f}</td><td>{real["gated_pck32"]:.3f}</td><td>{dual["error_mean"]:.2f}</td><td>{dual["gated_pck32"]:.3f}</td></tr>')
    page.append('</table>');report=[];colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544']
    if args.back_control:
        page[0]=page[0].replace('Pose query alignment','Real back weight sensitivity').replace('Frozen detector, pose query-assignment control','Real back06/07 training weight control').replace('Left initial6400 readout; middle +1600steps original TRAIN query assignment; right +1600steps adapted TRAIN query assignment.','Left strict weight1; middle weight0.1; right weight0. All resume same6400checkpoint and1600steps. Synthetic back unchanged. This is incremental fine-tuning sensitivity, not from-scratch omission.').replace('IDs1/15/16/17/18','IDs1/6/7/15/16/17/18')
    if args.decoder_control:
        page[0]=page[0].replace('Pose query alignment','Decoder trainable-scope comparison').replace('Frozen detector, pose query-assignment control','Frozen versus unfrozen keypoint decoder').replace('Left initial6400 readout; middle +1600steps original TRAIN query assignment; right +1600steps adapted TRAIN query assignment.','Left initial6400 readout; middle +800steps frozen decoder; right +800steps unfrozen decoder. Both train spatial readout; same frozen ROI detector/data/matching, all138queries retained. NativeFP32 memory; no GT crop. This compares trainable scope, not equal parameter counts.')
        page.append('<h2>Per-camera real foot error, pixels</h2><table><tr><th>Arm</th><th>Seed</th>'+''.join(f'<th>cam{c}</th>' for c in 'ABCDEFG')+'</tr>')
        for result in results['results']:
            page.append('<tr><td>'+result['arm']+'</td><td>'+str(result['seed'])+'</td>'+''.join('<td>'+str(round(result['metrics'][f'real_validation/cam{c}/feet']['error_mean'],2))+'</td>' for c in 'ABCDEFG')+'</tr>')
        page.append('</table>')
    if args.bottom_balanced:
        page[0]=page[0].replace('Frozen versus unfrozen keypoint decoder','Real bottom sampling balance').replace('Left initial6400 readout; middle +800steps frozen decoder; right +800steps unfrozen decoder. Both train spatial readout; same frozen ROI detector/data/matching, all138queries retained. NativeFP32 memory; no GT crop. This compares trainable scope, not equal parameter counts.','Left initial6400; middle uniform real sampling; right4bottom+4otherreal per batch, both800steps and bothunfrozen decoder. Exact same8synthetic samples per step, same initialization/LR/loss/frozenROI detector. Camera metadata affects TRAIN sampling only; no inference routing. Same TRAINpool but different per-image exposures.')
    if args.extend_decoder:page[0]=page[0].replace('both800steps','both2400steps (matched resume from800, optimizer state restored and low-LR restart)')
    for ix,i in enumerate(selected):
        o=rows[i];m=o['metadata'];r=m['record'];j=ids.index(i);cache=ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt';item=torch.load(cache,map_location='cpu',weights_only=False) if cache.exists() else load_item(r)
        rgb=(item['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=rgb.shape[:2];pw=420;ph=round(h*pw/w);im=Image.fromarray(rgb).resize((pw,ph));canvas=Image.new('RGB',(3*pw,ph+32),(22,24,30));draw=ImageDraw.Draw(canvas)
        score=1/(1+np.exp(-np.clip(logits[i],-80,80)));selected_q=np.flatnonzero(score>=.5)
        panels=[('strict realback1',initial[j]),('realback0.1',control[j]),('realback0',new[j])] if args.back_control else [('initial6400',initial[j]),('old assignment+1600',control[j]),('new assignment+1600',new[j])]
        if args.decoder_control:panels=[('initial6400',initial[j]),('frozen decoder+800',control[j]),('unfrozen decoder+800',new[j])]
        if args.bottom_balanced:panels=[('initial6400',initial[j]),('uniform real+800',control[j]),('bottom balanced+800',new[j])]
        if args.extend_decoder:panels=[('initial6400',initial[j]),('uniform real+2400',control[j]),('bottom balanced+2400',new[j])]
        point_ids=[0,5,6,14,15,16,17] if args.back_control else [0,14,15,16,17]
        for col,(name,xy) in enumerate(panels):
            canvas.paste(im,(col*pw,32));draw.text((col*pw+4,5),name,fill='white')
            for q in selected_q:
                color=colors[q];cx,cy,bw,bh=boxes[i,q];x0,y0=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,32];x1,y1=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,32];draw.rectangle((x0,y0,x1,y1),outline=color,width=2)
                for k in point_ids:
                    x,y=xy[q,k]*[pw,ph]+[col*pw,32];draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=2);draw.text((x+3,y),str(k+1),fill=color)
            for inst in torch.where(o['instance_valid'])[0]:
                for k in point_ids:
                    if o['point_valid'][inst,k]:
                        x,y=o['gt_xy'][inst,k].numpy()*[pw,ph]+[col*pw,32];draw.line((x-4,y,x+4,y),fill='#ffee55',width=2);draw.line((x,y-4,x,y+4),fill='#ffee55',width=2)
        name=f'{ix:02d}_{r["kind"]}.jpg';canvas.save(OUT/name,quality=95);report.append(dict(file=name,observation=i,metadata=m,queries=selected_q.tolist()));page.append('<h2>'+html.escape(str(m))+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',report);write_json(OUT/'results.json',results);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)

if __name__=='__main__':main()
