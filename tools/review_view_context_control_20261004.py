"""Raw six-query paired/mixed context pilot review, seed17 fixed in advance."""
import argparse,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'view_context_control_v1';OUT=CYCLE/'review_view_context_control_v1'

def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--order-control',action='store_true');parser.add_argument('--shared-control',action='store_true');parser.add_argument('--full-context-control',action='store_true');args=parser.parse_args()
    if args.full_context_control:
        assert not(args.order_control or args.shared_control);args.shared_control=True
    assert not(args.order_control and args.shared_control);advanced=args.order_control or args.shared_control;alternate='shared' if args.shared_control else 'reversed'
    if args.order_control:BASE=CYCLE/'view_order_control_v1';OUT=CYCLE/'review_view_order_control_v1'
    if args.shared_control:BASE=CYCLE/'shared_token_pose_control_v1';OUT=CYCLE/'review_shared_token_pose_control_v1'
    if args.full_context_control:BASE=CYCLE/'full_context_pose_control_v1';OUT=CYCLE/'review_full_context_pose_control_v1'
    assert (BASE/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    rows=[]
    for rec in json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records']:rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    with np.load(BASE/'predictions.npz') as a:indices=a['image_indices'].tolist();pred={k:a[k].copy() for k in a.files if k!='image_indices'}
    with np.load(CYCLE/'detection_context_roi_v1/predictions.npz') as a:paired_boxes=a['seed17__instance_boxes_cxcywh'].copy();paired_logits=a['seed17__instance_objectness_logits'].copy()
    singles={}
    for rec in json.loads((CYCLE/'single_view_validation_context_v1/manifest.json').read_text())['records']:
        part=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);singles[rec['image_index']]={k:v.numpy() for k,v in part['adapted'][17].items()}
    reversed_det={}
    if advanced:
        with np.load(CYCLE/'detection_view_context_control_v1/predictions.npz') as a:
            paired_boxes=a['mixed_seed17_paired__instance_boxes_cxcywh'].copy();paired_logits=a['mixed_seed17_paired__instance_objectness_logits'].copy()
            singles={i:{k:a[f'mixed_seed17_single__{k}'][i].copy() for k in ['instance_boxes_cxcywh','instance_objectness_logits']} for i in indices}
        for rec in json.loads((CYCLE/f'{alternate}_view_validation_context_v1/manifest.json').read_text())['records']:
            part=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);reversed_det[rec['image_index']]={k:v.numpy() for k,v in part['adapted'][17].items()}
    if args.full_context_control:
        full_det={}
        for rec in json.loads((CYCLE/'fullshared_view_validation_context_v1/manifest.json').read_text())['records']:
            part=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);full_det[rec['image_index']]={k:v.numpy() for k,v in part['adapted'][17].items()}
    picked=[]
    for cam in range(7):
        candidates=[i for i in indices if rows[i]['metadata']['record']['kind']=='real' and int(rows[i]['metadata']['view'])==cam]
        if candidates:picked.append(candidates[0])
    for count in [1,2]:
        candidates=[i for i in indices if rows[i]['metadata']['record']['kind']!='real' and rows[i]['metadata']['expected_instances']==count]
        if candidates:picked.append(candidates[0])
    colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];manifest=[]
    page=['<!doctype html><meta charset="utf-8"><title>Context augmentation pilot</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Paired-only versus single/paired training</h1><p>Left paired-only continuation; right mixed continuation. Both800steps from same balanced2400, seed17 fixed. Every query with confidence>=0.5 drawn; no GT filtering or NMS. Yellow crosses are accepted annotations, not anatomical certainty. Query colors do not mean persistent animal IDs. Each camera example is first eligible development image, not chosen by treatment gain. Single and paired panels show the same image with distinct encoder context. Modelinput pixels, not original resolution.</p>']
    if args.order_control:page=['<!doctype html><meta charset="utf-8"><title>Reference-order augmentation pilot</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Fixed versus random reference order TRAIN</h1><p>Left fixed-order; right random reference. Both800steps from SAME mixed800pose, SAME frozen mixed800detector. Both50percent single/50percent paired exposure; treatment reverses half paired occurrences. No shared-special-token intervention. All confidence>=.5queries drawn, noGTfiltering/count orNMS. Yellow acceptedpoints. Same first-camera examples, fixedseed17, not independenttest.</p>']
    if args.shared_control:page=['<!doctype html><meta charset="utf-8"><title>Shared-token2D adaptation</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Original versus shared-token2D TRAIN</h1><p>Both800steps sameinitialmixed800pose,50percent single/50percent paired. Originalimageorder unchanged; shared treatment uses firstspecialtokens forallpairedviews. Detectorweights frozenmixed800; featurecontext canchange detectorpredictions, identical betweenarms within eachpanel. Allconfidence>=.5queries, noGTfiltering/NMS. Geometry fromsharedpass isNOTgroundtruth. Same fixedcameraexamples, seed17. No promotion.</p>']
    if args.full_context_control:page=['<!doctype html><meta charset="utf-8"><title>Eligiblefullcontext adaptation</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Shared-token2D: paired versus fullcontext TRAIN</h1><p>Both800steps fromidentical shared800pose, samefrozenmixed800detector. Both50percent single, control50percent sharedpair; treatment25percent sharedpair25percent eligiblefullcontext. Same supervisedimages/bottomexposure. Realfull5-7cameras; syntheticfullhere only2eligiblecameras, NOT4/8viewTRAIN coverage. Allconfidence>=.5queries noGTfilter/count orNMS. Same fixedexamples seed17, no promotion.</p>']
    for index,i in enumerate(picked):
        o=rows[i];m=o['metadata'];r=m['record'];item=load_item(dict(r,views=[m['view']]))
        imdata=(item['batch']['images'][0,0].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=imdata.shape[:2];pw=500;ph=round(h*pw/w);im=Image.fromarray(imdata).resize((pw,ph));j=indices.index(i)
        for context in (['single','shared','full'] if args.full_context_control else ['single','paired',alternate] if advanced else ['single','paired']):
            boxes=paired_boxes[i] if context=='paired' else singles[i]['instance_boxes_cxcywh'];logits=paired_logits[i] if context=='paired' else singles[i]['instance_objectness_logits'];scores=1/(1+np.exp(-np.clip(logits,-80,80)));visible=np.flatnonzero(scores>=.5)
            if context==alternate:
                boxes=reversed_det[i]['instance_boxes_cxcywh'];logits=reversed_det[i]['instance_objectness_logits'];scores=1/(1+np.exp(-np.clip(logits,-80,80)));visible=np.flatnonzero(scores>=.5)
            if context=='full':
                boxes=full_det[i]['instance_boxes_cxcywh'];logits=full_det[i]['instance_objectness_logits'];scores=1/(1+np.exp(-np.clip(logits,-80,80)));visible=np.flatnonzero(scores>=.5)
            canvas=Image.new('RGB',(2*pw,ph+40),(22,24,30));draw=ImageDraw.Draw(canvas)
            for col,arm in enumerate(['paired','full'] if args.full_context_control else ['original','shared'] if args.shared_control else ['fixed','random_ref'] if args.order_control else ['paired','mixed']):
                canvas.paste(im,(col*pw,40));draw.text((col*pw+5,6),f'{arm} TRAIN / {context} INPUT',fill='white')
                for q in visible:
                    cx,cy,bw,bh=boxes[q];x0,y0=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,40];x1,y1=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,40];draw.rectangle((x0,y0,x1,y1),outline=colors[q],width=2);draw.text((x0,y0-10),f'q{q}:{scores[q]:.2f}',fill=colors[q])
                    for k in [0,14,15,16,17]:
                        x,y=pred[f'{arm}_seed17_{context}'][j,q,k]*[pw,ph]+[col*pw,40];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y+1),str(k+1),fill=colors[q])
                for target in range(len(o['instance_valid'])):
                    if not o['instance_valid'][target]:continue
                    for k in [0,14,15,16,17]:
                        if not o['point_valid'][target,k]:continue
                        x,y=o['gt_xy'][target,k].numpy()*[pw,ph]+[col*pw,40];draw.line((x-5,y,x+5,y),fill='#fff06a',width=2);draw.line((x,y-5,x,y+5),fill='#fff06a',width=2)
            name=f'{index:02d}_{r["kind"]}_cam{chr(65+int(m["view"]))}_{context}.jpg';canvas.save(OUT/name,quality=95)
            manifest.append(dict(file=name,image_index=i,context=context,metadata=m,queries=visible.tolist()));page.append('<h2>'+html.escape(r['scene']+' '+context)+'</h2><img src="'+name+'">')
    results=json.loads((BASE/'results.json').read_text());summary=[]
    for result in results['results']:
        summary.append(dict(arm=result['arm'],seed=result['seed'],step=result['step'],metrics={k:v for k,v in result['metrics'].items() if '_validation/' in k}))
    page.insert(3,'<p>Numerical results for all3seeds in results.json. Visual review still required; no production promotion.</p>')
    write_json(OUT/'manifest.json',manifest);write_json(OUT/'results.json',summary);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)

if __name__=='__main__':main()
