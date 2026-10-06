"""Raw confidence-selected queries: no GT chooses which predictions are drawn."""
import argparse,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'live_view_audit_v2';OUT=CYCLE/'review_live_queries_v1'


def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--detector-roi',action='store_true');parser.add_argument('--decoder-adapted',choices=['uniform','balanced']);parser.add_argument('--context-control',choices=['paired','mixed']);parser.add_argument('--detector-context',choices=['paired','mixed']);args=parser.parse_args()
    if args.detector_context:
        assert args.context_control in [None,'mixed'];args.context_control='mixed'
    assert not (args.context_control and args.decoder_adapted)
    if args.context_control:args.decoder_adapted='context_'+args.context_control
    if args.decoder_adapted:args.detector_roi=True
    if args.detector_roi:BASE=CYCLE/'live_view_roi_detector_audit_v1';OUT=CYCLE/'review_live_roi_queries_v1'
    if args.decoder_adapted:BASE=CYCLE/f'live_view_decoder_{args.decoder_adapted}2400_audit_v1';OUT=CYCLE/f'review_live_decoder_{args.decoder_adapted}2400_v1'
    if args.context_control:BASE=CYCLE/f'live_view_context_{args.context_control}800_audit_v1';OUT=CYCLE/f'review_live_context_{args.context_control}800_v1'
    if args.detector_context:BASE=CYCLE/f'live_view_detector_context_{args.detector_context}800_audit_v1';OUT=CYCLE/f'review_live_detector_context_{args.detector_context}800_v1'
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    reports=json.loads((BASE/'results.json').read_text())['reports'];selected=[]
    previous=json.loads((CYCLE/'live_view_audit_v2/results.json').read_text())['reports'] if args.detector_roi else reports
    if args.decoder_adapted:
        previous_dir='live_view_decoder_uniform2400_audit_v1' if args.decoder_adapted=='balanced' else 'live_view_roi_detector_audit_v1'
        if args.context_control:previous_dir='live_view_context_paired800_audit_v1' if args.context_control=='mixed' else 'live_view_decoder_balanced2400_audit_v1'
        if args.detector_context:previous_dir='live_view_detector_context_paired800_audit_v1' if args.detector_context=='mixed' else 'live_view_context_mixed800_audit_v1'
        previous=json.loads((CYCLE/previous_dir/'results.json').read_text())['reports']
    current_lookup={(r['task']['record']['scene'],r['task']['variant']):r for r in reports}
    previous_lookup={(r['task']['record']['scene'],r['task']['variant']):r for r in previous}
    for kind in ['real','synthetic']:
        candidates=[r for r in previous if r['task']['record']['kind']==kind and r['task']['variant']=='paired']
        candidates.sort(key=lambda r:r['metrics']['spatial/anchor'].get('mean',0))
        for picked in [candidates[len(candidates)//2],candidates[-1]]:
            scene=picked['task']['record']['scene'];selected.extend(r for r in reports if r['task']['record']['scene']==scene)
    page=['<!doctype html><meta charset="utf-8"><title>Raw all-query inference</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Raw inference: all confidence-selected queries</h1><p>Left original; right opt-in spatial readout. Every query with objectness probability at least0.5 is drawn, WITHOUT GT filtering or NMS. Boxes unchanged. Point labels shown1/15/16/17/18. Conditional landmark gains do not repair the frozen detector. Scenes chosen by conditional paired error median/worst, then same anchor shown under1/2/allview contexts.</p>'];meta=[]
    colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544']
    if args.detector_roi:page=['<!doctype html><meta charset="utf-8"><title>Live ROI detector audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Full RGB inference, same spatial pose readout</h1><p>Left original detector; right ROI residual detector. All queries with confidence>=.5, no GT filtering/noNMS. Same scenes chosen in previous audit, shown under1/2/all views. Colors identify decoder queries, not persistent animal identities. Points1/15/16/17/18.</p>']
    if args.decoder_adapted:
        left_label='uniform2400' if args.decoder_adapted=='balanced' else 'original decoder'
        if args.context_control:left_label='paired-only +800' if args.context_control=='mixed' else 'balanced2400 initial'
        if args.detector_context:left_label='paired DET +800' if args.detector_context=='mixed' else 'original ROI DET'
        page=['<!doctype html><meta charset="utf-8"><title>Live decoder adaptation audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Full RGB decoder adaptation</h1><p>Left '+left_label+'; right '+args.decoder_adapted+'2400. Same fixed ROI detector; all confidence>=0.5 queries drawn, no GT filtering or NMS. Colors are query IDs, not animal identities. Seed17 fixed; development images, not external test.</p>']
        if args.context_control:page=[p.replace(args.decoder_adapted+'2400',args.context_control+' context +800') for p in page]
        if args.detector_context:page=['<!doctype html><meta charset="utf-8"><title>Detector context RGB audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detector context adaptation, frozen mixed800 pose</h1><p>Left '+left_label+'; right '+args.detector_context+' detector +800. SAME frozen pose; detector boxes and assignments change. Allconfidence>=.5queries drawn, no GTfiltering/count or NMS. Seed17 same development scenes, not external test.</p>']
    for index,row in enumerate(selected):
        task=row['task'];r=task['record'];item=load_item(r);v=r['views'].index(task['anchor_view']);imdata=(item['batch']['images'][0,v].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=imdata.shape[:2];pw=500;ph=round(h*pw/w);im=Image.fromarray(imdata).resize((pw,ph));canvas=Image.new('RGB',(pw*2,ph+40),(22,24,30));draw=ImageDraw.Draw(canvas)
        with np.load(ROOT/row['path']) as a:data={k:a[k] for k in a.files}
        olddata=data
        if args.detector_roi:
            with np.load(ROOT/previous_lookup[(r['scene'],task['variant'])]['path']) as a:olddata={k:a[k] for k in a.files}
        for col,key in enumerate(['instance_keypoint_original_xy','instance_keypoint_xy']):
            drawdata=olddata if col==0 else data
            if args.detector_roi:key='instance_keypoint_xy'
            logits=drawdata['instance_objectness_logits'][0,v];scores=1/(1+np.exp(-np.clip(logits,-80,80)));visible=np.flatnonzero(scores>=.5)
            canvas.paste(im,(col*pw,40));draw.text((col*pw+5,6),('original' if col==0 else 'spatial')+' '+task['variant'],fill='white')
            if args.decoder_adapted:
                right_label=args.context_control+' +800' if args.context_control else args.decoder_adapted+'2400'
                if args.detector_context:right_label=args.detector_context+' DET +800'
                draw.rectangle((col*pw,0,(col+1)*pw-1,39),fill=(22,24,30));draw.text((col*pw+5,6),(left_label if col==0 else right_label)+' '+task['variant'],fill='white')
            for q in visible:
                color=colors[q];cx,cy,bw,bh=drawdata['instance_boxes_cxcywh'][0,v,q];x0,y0=(np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,40]);x1,y1=(np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,40]);draw.rectangle((x0,y0,x1,y1),outline=color,width=2);draw.text((x0,y0-10),f'q{q} {scores[q]:.2f}',fill=color)
                for k in [0,14,15,16,17]:
                    x,y=drawdata[key][0,v,q,k]*[pw,ph]+[col*pw,40];draw.ellipse((x-2,y-2,x+2,y+2),fill=color);draw.text((x+3,y+1),str(k+1),fill=color)
        name=f'{index:02d}_{r["kind"]}_{task["variant"]}.jpg';canvas.save(OUT/name,quality=95);meta.append(dict(file=name,task=task,queries=visible.tolist(),detector=row['detector']));page.append('<h2>'+html.escape(r['scene']+' '+task['variant'])+'</h2><img src="'+name+'">')
    write_json(OUT/'manifest.json',meta);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)


if __name__=='__main__':main()
