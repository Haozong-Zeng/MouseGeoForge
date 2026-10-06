"""Physical-view aligned order audit; prediction-only set matching, raw overlays."""
import argparse,collections,datetime as dt,html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';BASE=CYCLE/'live_view_order_mixed800_audit_v1';OUT=CYCLE/'review_view_order_v1'

def main():
    global BASE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--shared-first-tokens',action='store_true');parser.add_argument('--reference-control',choices=['fixed','random_ref']);parser.add_argument('--reference-detector',choices=['fixed','random_ref']);parser.add_argument('--independent-pose',action='store_true');parser.add_argument('--shared-pose-control',choices=['original','shared']);parser.add_argument('--full-context-control',choices=['paired','full']);parser.add_argument('--full-context-detector',choices=['paired','full']);args=parser.parse_args()
    if args.full_context_detector:
        assert args.full_context_control in [None,'full'];args.full_context_control='full'
    if args.full_context_control:
        assert not args.shared_pose_control;args.shared_pose_control=args.full_context_control
    if args.shared_pose_control:assert not any([args.shared_first_tokens,args.reference_control,args.reference_detector,args.independent_pose])
    if args.independent_pose:args.reference_detector='random_ref'
    if args.reference_detector:
        assert args.reference_control in [None,'random_ref'];args.reference_control='random_ref'
    if args.shared_first_tokens:BASE=CYCLE/'live_view_shared_first_tokens_audit_v1';OUT=CYCLE/'review_shared_first_tokens_v1'
    assert not (args.shared_first_tokens and args.reference_control)
    if args.reference_control:BASE=CYCLE/f'live_view_reference_{args.reference_control}800_audit_v1';OUT=CYCLE/f'review_reference_{args.reference_control}_order_v1'
    if args.reference_detector:BASE=CYCLE/f'live_view_reference_detector_{args.reference_detector}800_audit_v1';OUT=CYCLE/f'review_reference_detector_{args.reference_detector}_order_v1'
    if args.independent_pose:BASE=CYCLE/'live_view_independent_pose_audit_v1';OUT=CYCLE/'review_independent_pose_order_v1'
    if args.shared_pose_control:BASE=CYCLE/f'live_view_shared_pose_{args.shared_pose_control}800_audit_v1';OUT=CYCLE/f'review_shared_pose_{args.shared_pose_control}_order_v1'
    if args.full_context_control:BASE=CYCLE/f'live_view_full_context_{args.full_context_control}800_audit_v1';OUT=CYCLE/f'review_full_context_{args.full_context_control}_order_v1'
    if args.full_context_detector:BASE=CYCLE/f'live_view_full_detector_{args.full_context_detector}800_audit_v1';OUT=CYCLE/f'review_full_detector_{args.full_context_detector}_order_v1'
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    rows=json.loads((BASE/'results.json').read_text())['reports'];previous=json.loads((CYCLE/'live_view_detector_context_mixed800_audit_v1/results.json').read_text())['reports'];lookup={(r['task']['record']['scene'],r['task']['variant']):r for r in rows};oldlookup={(r['task']['record']['scene'],r['task']['variant']):r for r in previous}
    def read(row):
        with np.load(ROOT/row['path']) as a:return {k:a[k].copy() for k in a.files}
    max_original=0.;count=0
    for key,old in oldlookup.items():
        if args.reference_control or args.shared_pose_control:continue # Changed trained weights: equality to old checkpoint is not expected.
        if args.shared_first_tokens and key[1]!='single':continue
        current=lookup[key];assert current['task']==old['task'];a,b=read(current),read(old)
        for k in ['instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits']:
            delta=float(np.max(np.abs(a[k]-b[k])));assert delta<2e-5;max_original=max(max_original,delta)
        count+=1
    summary=[]
    for kind in ['real','synthetic']:
        for variant in ['single','paired','paired_reversed','all','all_reversed']:
            group=[r for r in rows if r['task']['record']['kind']==kind and r['task']['variant']==variant];points=sum(r['gated_pose']['total_points'] for r in group);anchor_count=sum(r['metrics']['spatial/anchor']['n'] for r in group)
            summary.append(dict(kind=kind,variant=variant,points=points,gated_pck32=sum(r['gated_pose']['pck32_count'] for r in group)/max(points,1),anchor_error=sum(r['metrics']['spatial/anchor']['n']*r['metrics']['spatial/anchor']['mean'] for r in group)/max(anchor_count,1),tp=sum(r['detector']['tp'] for r in group),gt=sum(r['detector']['gt'] for r in group),predictions=sum(r['detector']['predictions'] for r in group)))
    changes=[]
    for row in rows:
        variant=row['task']['variant']
        if variant not in ['paired','all']:continue
        r=row['task']['record'];reverse=lookup[(r['scene'],variant+'_reversed')];a,b=read(row),read(reverse);viewids=r['views'];reversedids=reverse['task']['record']['views'];assert set(viewids)==set(reversedids)
        assert row['task']['anchor_view']==reverse['task']['anchor_view']
        for i,view in enumerate(viewids):
            j=reversedids.index(view)
            # Explicitly verify label ordering: comparisons follow physical views,
            # not tensor indices. Labels are not used for query set association.
            np.testing.assert_allclose(a['gt_xy'][0,i],b['gt_xy'][0,j],atol=0,rtol=0,equal_nan=True);np.testing.assert_array_equal(a['valid'][0,i],b['valid'][0,j]);np.testing.assert_allclose(a['gt_boxes'][0,i],b['gt_boxes'][0,j],atol=0,rtol=0)
            sa=1/(1+np.exp(-np.clip(a['instance_objectness_logits'][0,i],-80,80)));sb=1/(1+np.exp(-np.clip(b['instance_objectness_logits'][0,j],-80,80)))
            cost=np.abs(a['instance_boxes_cxcywh'][0,i,:,None]-b['instance_boxes_cxcywh'][0,j,None]).sum(-1)+.1*np.abs(sa[:,None]-sb[None])
            qa,qb=linear_sum_assignment(cost);selected=sa[qa]>=.5;qa,qb=qa[selected],qb[selected];pixels=np.array(r['image_hw'][::-1]);kp=[0,14,15,16,17]
            shift=np.linalg.norm((a['instance_keypoint_xy'][0,i,qa][:,kp]-b['instance_keypoint_xy'][0,j,qb][:,kp])*pixels,axis=-1)
            changes.append(dict(scene=r['scene'],kind=r['kind'],variant=variant,view=view,was_reference=i==0,is_reference_after_reverse=j==0,baseline_confident_queries=qa.tolist(),matched_reverse_queries=qb.tolist(),mean_point_shift_px=float(shift.mean()) if shift.size else None,max_point_shift_px=float(shift.max()) if shift.size else None,reverse_confident_matched=int((sb[qb]>=.5).sum())))
    oldgallery=json.loads((CYCLE/'review_live_detector_context_mixed800_v1/manifest.json').read_text());scenes=list(dict.fromkeys(m['task']['record']['scene'] for m in oldgallery));colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];gallery=[]
    page=['<!doctype html><meta charset="utf-8"><title>Input order audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Same images, different input order</h1><p>Frozen mixed-context pose+detector, seed17. Columns: paired original/reversed, all original/reversed. Each row shows exactly the SAME physical anchor camera, not arrayindex0. RGB pixels and GT ordering checked after physicalcamera alignment. All confidence>=0.5 queries shown WITHOUT GTfiltering or NMS. Colors are query IDs, not persistent identities. Numbers1/15/16/17/18; yellow crosses accepted labels. This is an order-sensitivity diagnostic, NOT choosing a best order using GT.</p>']
    if args.shared_first_tokens:page.append('<h2>Shared first-frame camera/register tokens, inference-only probe</h2><p>All views temporarily use the first-frame special-token values. Other weights unchanged; checkpoint files untouched. Single-view unchanged control. Symmetry does NOT imply accuracy or calibrated depth. <a href="../review_view_order_v1/index.html">Original special-token comparison</a></p>')
    if args.reference_control and not args.reference_detector:page.append('<h2>Reference-order training: '+args.reference_control+'</h2><p>800 additional matched steps; original pretrained special tokens retained. Frozen mixed800 detector. Both arms have identical single/paired counts. Random_ref reverses half paired TRAIN inputs. Changed pose weights; old-checkpoint output equality is not expected. Cached-to-live parity is checked separately by validator. No model promotion.</p>')
    if args.reference_detector:page.append('<h2>Detector reference-order training: '+args.reference_detector+'</h2><p>Frozen random_ref800 pose, '+args.reference_detector+' detector800. Original pretrained special tokens. Changed detector weights; old-checkpoint output equality is not expected. Cached-to-live parity separately checked. No promotion.</p>')
    if args.independent_pose:page.append('<h2>Independent-view2D diagnostic</h2><p>Same weights, eachphysicalcamera separatelyencoded for2Dpose/detection. Original multiview geometry retained. Auxiliaryposefields not reconciled. Extra encoderpasses, notcompute-matched, no crossviewinstanceidentity. Singleview unchanged control. Consistency is notaccuracy.</p>')
    if args.shared_pose_control:page.append('<h2>Shared-token2D pose adaptation: '+args.shared_pose_control+'</h2><p>Both compared arms use shared-token2D inference, samefrozenmixed800detector; pose was trained800steps with '+args.shared_pose_control+' context. The generic heading refers to the common mixed800 initialization, not final poseweights. Originalmultiview geometry preserved; six2Dfields replaced, auxiliaries unreconciled. Extraencoderpass; nojointtraining use orpromotion. Single12 andsharedpaired24 live/cachegoldens separatelychecked.</p>')
    if args.full_context_control:page.append('<h2>Full-eligible-context TRAIN control: '+args.full_context_control+'</h2><p>Initial shared800pose then800matched steps paired/fullcontext training. Realeligiblefull5-7cameras; syntheticfullcache2cameras only, so4/8viewresults arestress-only. FullRGB48realview goldens plus24sharedpaired and12single checked. Originalgeometry remains unchanged.</p>')
    if args.full_context_detector:page=['<!doctype html><meta charset="utf-8"><title>Detector full context audit</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Detector view-count control: '+args.full_context_detector+'</h1><p>Frozen full800 pose; matched '+args.full_context_detector+'800 detector. Seed17 fixed. Same physical anchor, paired/reverse/all/reverse. Every confidence&gt;=.5 query drawn, no GT filtering/count/NMS. Colors are query IDs, not cross-view identities; yellow crosses are accepted labels. Actual RGB candidate goldens: 12single/24shared-pair/48real-full; original-token gate used baseline detector before loading candidate. Original geometry retained; auxiliary pose fields unreconciled. Synthetic4/8view is untrained stress. No production change. Order consistency is not accuracy.</p>']
    for index,scene in enumerate(scenes):
        base=lookup[(scene,'paired')];r=base['task']['record'];anchor=base['task']['anchor_view'];item=load_item(dict(r,views=[anchor]));rgb=item['batch']['images'][0,0];h,w=rgb.shape[-2:];pw=380;ph=round(h*pw/w);im=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')).resize((pw,ph));canvas=Image.new('RGB',(pw*4,ph+40),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,variant in enumerate(['paired','paired_reversed','all','all_reversed']):
            current=lookup[(scene,variant)];rr=current['task']['record'];v=rr['views'].index(anchor);loaded=load_item(rr);torch.testing.assert_close(rgb,loaded['batch']['images'][0,v],atol=0,rtol=0);a=read(current);canvas.paste(im,(col*pw,40));draw.text((col*pw+5,5),variant+' / cam'+chr(65+anchor),fill='white')
            score=1/(1+np.exp(-np.clip(a['instance_objectness_logits'][0,v],-80,80)))
            for q in np.flatnonzero(score>=.5):
                cx,cy,bw,bh=a['instance_boxes_cxcywh'][0,v,q];x0,y0=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,40];x1,y1=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,40];draw.rectangle((x0,y0,x1,y1),outline=colors[q],width=2);draw.text((x0,y0-10),f'q{q}:{score[q]:.2f}',fill=colors[q])
                for k in [0,14,15,16,17]:
                    x,y=a['instance_keypoint_xy'][0,v,q,k]*[pw,ph]+[col*pw,40];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y),str(k+1),fill=colors[q])
            for target in range(a['gt_xy'].shape[2]):
                for k in [0,14,15,16,17]:
                    if not a['valid'][0,v,target,k]:continue
                    x,y=a['gt_xy'][0,v,target,k]*[pw,ph]+[col*pw,40];draw.line((x-4,y,x+4,y),fill='#fff06a',width=2);draw.line((x,y-4,x,y+4),fill='#fff06a',width=2)
        name=f'{index:02d}_{r["kind"]}_order.jpg';canvas.save(OUT/name,quality=95);gallery.append(dict(file=name,scene=scene,anchor=anchor));page.append('<h2>'+html.escape(scene)+'</h2><img src="'+name+'">')
    write_json(OUT/'results.json',dict(summary=summary,changes=changes,original_order_replay_cases=count,max_original_order_delta=max_original,replay_scope='single-view only, sharedtoken intervention' if args.shared_first_tokens else 'all36original-order tasks',physical_view_gt_alignment=True,rgb_alignment_checked=True,association='Prediction-only boxL1+.1confidence difference, all6queries Hungarian. Displacement reported for reference-order confidence>=.5 queries; not GT association nor accuracy metric.',visual_review_pending=True));write_json(OUT/'manifest.json',gallery);(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)
    print(json.dumps(dict(summary=summary,original_order_replay_cases=count,max_original_order_delta=max_original),indent=2))

if __name__=='__main__':main()
