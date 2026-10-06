"""Matched raw prediction views for detector-context and frozen-pose audit."""
import argparse,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'review_detection_context_pose_v1';BASE=CYCLE/'detection_context_pose_audit_v1'

def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--order-control',action='store_true');parser.add_argument('--full-context-control',action='store_true');args=parser.parse_args();assert not(args.order_control and args.full_context_control);arms=['baseline','paired','full'] if args.full_context_control else ['baseline','fixed','random_ref'] if args.order_control else ['baseline','paired','mixed'];contexts=['single','shared','full'] if args.full_context_control else ['single','paired','reversed'] if args.order_control else ['single','paired']
    if args.order_control:BASE=CYCLE/'detection_reference_pose_audit_v1';OUT=CYCLE/'review_detection_reference_pose_v1'
    if args.full_context_control:BASE=CYCLE/'detection_full_context_pose_audit_v1';OUT=CYCLE/'review_detection_full_context_pose_v1'
    assert (BASE/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    rows=[]
    for rec in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    with np.load(BASE/'predictions.npz') as a:indices=a['image_indices'].tolist();points={k:a[k].copy() for k in a.files if k!='image_indices'}
    with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:det={k:a[k].copy() for k in a.files}
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:oldpair={k:a['seed17__'+k].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
    singlemanifest=json.loads((CYCLE/'single_detection_context_v1/manifest.json').read_text());oldsingle={k:v.numpy() for k,v in torch.load(ROOT/singlemanifest['golden_path'],map_location='cpu',weights_only=False)[17].items()}
    if args.order_control:
        oldpair={k:det[f'mixed_seed17_paired__{k}'] for k in oldpair};oldsingle={k:det[f'mixed_seed17_single__{k}'] for k in oldsingle}
        rm=json.loads((CYCLE/'reversed_detection_context_v1/manifest.json').read_text());oldreverse={k:v.numpy() for k,v in torch.load(ROOT/rm['golden_path'],map_location='cpu',weights_only=False)[17].items()}
        with np.load(CYCLE/'detection_reference_control_v1/predictions.npz') as a:det={k:a[k].copy() for k in a.files}
    if args.full_context_control:
        oldsingle={k:det[f'mixed_seed17_single__{k}'] for k in oldsingle}
        sm=json.loads((CYCLE/'shared_detection_context_v1/manifest.json').read_text());oldpair={k:v.numpy() for k,v in torch.load(ROOT/sm['golden_path'],map_location='cpu',weights_only=False)[17].items()}
        fm=json.loads((CYCLE/'fullshared_detection_context_v1/manifest.json').read_text());oldreverse={k:v.numpy() for k,v in torch.load(ROOT/fm['golden_path'],map_location='cpu',weights_only=False)[17].items()}
        with np.load(CYCLE/'detection_full_context_control_v1/predictions.npz') as a:det={k:a[k].copy() for k in a.files}
    picked=list(dict.fromkeys(r['image_index'] for r in json.loads((CYCLE/'review_view_context_control_v1/manifest.json').read_text())))
    colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];manifest=[]
    page=['<!doctype html><meta charset="utf-8"><title>Detector context control</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detector context adaptation, frozen pose</h1><p>Columns: original ROI detector, paired-only detector continuation, mixed detector continuation. SAME frozen mixed800 pose model; predicted boxes and association may change. Seed17 fixed. All confidence>=0.5 queries drawn WITHOUT GT filtering/count or NMS. Yellow crosses accepted labels; query colors are not persistent identities. Same fixed examples as previous context review; not selected by improvement. Original resized image, no crop masking of false positives.</p>']
    if args.order_control:page=['<!doctype html><meta charset="utf-8"><title>Detector reference control</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detector reference-order adaptation</h1><p>Columns baseline mixed800 detector / fixed-order continuation / random-reference continuation. SAME frozen random_ref800 pose. Seed17; all confidence>=0.5 queries, no GT filtering, count or NMS. Yellow crosses accepted labels; query colors are not persistent identity. Same fixed examples, not selected by improvement. Boxes are predicted; no GTcrop.</p>']
    if args.full_context_control:page=['<!doctype html><meta charset="utf-8"><title>Detectorfullcontext control</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detector view-count adaptation, frozenpose</h1><p>Columns baseline mixed800 / sharedpair continuation / eligiblefullcontext continuation. SAME frozen full_context_pose_control full800pose. Both50percent actualsingle; treatmentadds25percent fulleligiblecontext. Allconfidence>=.5queries, noGTfilter/count/NMS. Seed17 samefixedexamples, not independenttest. Yellowacceptedlabels; querycolors notpersistentidentity. Syntheticeligiblefull only2views.</p>']
    for index,i in enumerate(picked):
        o=rows[i];m=o['metadata'];r=m['record'];item=load_item(dict(r,views=[m['view']]))
        imdata=(item['batch']['images'][0,0].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=imdata.shape[:2];pw=420;ph=round(h*pw/w);im=Image.fromarray(imdata).resize((pw,ph));j=indices.index(i)
        for context in contexts:
            canvas=Image.new('RGB',(3*pw,ph+40),(22,24,30));draw=ImageDraw.Draw(canvas);drawn={}
            for col,arm in enumerate(arms):
                if arm=='baseline':p=oldpair if context in ['paired','shared'] else oldsingle if context=='single' else oldreverse;boxes=p['instance_boxes_cxcywh'][i];logits=p['instance_objectness_logits'][i]
                else:boxes=det[f'{arm}_seed17_{context}__instance_boxes_cxcywh'][i];logits=det[f'{arm}_seed17_{context}__instance_objectness_logits'][i]
                scores=1/(1+np.exp(-np.clip(logits,-80,80)));visible=np.flatnonzero(scores>=.5);drawn[arm]=visible.tolist();canvas.paste(im,(col*pw,40));draw.text((col*pw+5,6),f'{arm} DET / {context} INPUT',fill='white')
                for q in visible:
                    cx,cy,bw,bh=boxes[q];x0,y0=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,40];x1,y1=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,40];draw.rectangle((x0,y0,x1,y1),outline=colors[q],width=2);draw.text((x0,y0-10),f'q{q}:{scores[q]:.2f}',fill=colors[q])
                    for k in [0,14,15,16,17]:
                        x,y=points[f'{arm}_seed17_{context}'][j,q,k]*[pw,ph]+[col*pw,40];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y+1),str(k+1),fill=colors[q])
                for target in range(len(o['instance_valid'])):
                    if not o['instance_valid'][target]:continue
                    for k in [0,14,15,16,17]:
                        if not o['point_valid'][target,k]:continue
                        x,y=o['gt_xy'][target,k].numpy()*[pw,ph]+[col*pw,40];draw.line((x-4,y,x+4,y),fill='#fff06a',width=2);draw.line((x,y-4,x,y+4),fill='#fff06a',width=2)
            name=f'{index:02d}_{r["kind"]}_cam{chr(65+int(m["view"]))}_{context}.jpg';canvas.save(OUT/name,quality=95);manifest.append(dict(file=name,image_index=i,context=context,metadata=m,queries=drawn));page.append('<h2>'+html.escape(r['scene']+' '+context)+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',manifest);shutil.copy2(BASE/'results.json',OUT/'results.json');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)

if __name__=='__main__':main()
