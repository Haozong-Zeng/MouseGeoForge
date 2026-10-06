"""TRAIN-only visualization of additional synthetic view detection qualification."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np,torch
from PIL import Image,ImageDraw
from review_pixel_adapter_20261005 import font
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from mousegeoforge.pilot_metrics import box_iou
from mousegeoforge.detection_readout_loss import match_detection
from run_depth_pose_ablation import write_json
OUT=ROOT/'output/singleview_fusion_cycle_20261005/single_train_allview_rgb_inputs_v1'
def main():
    torch.set_num_threads(2);rows=torch.load(OUT/'rows.pt',map_location='cpu',weights_only=False);lookup={(r['metadata']['scene'],r['metadata']['view']):r for r in rows};records=[];items={}
    top=[r for r in rows if 'BrownFur-KeypointMoseq' in r['metadata']['scene'] and r['metadata']['synthetic_orientation']=='top']
    for r in top:
        scene=r['metadata']['scene'];item=load_item(r['record']);items[scene]=item;b=item['batch'];vi=r['record']['views'].index(r['metadata']['view']);t=instance_targets_from_batch(b);p=r['baseline'];valid=t['instance_valid'][0,vi]&t['box_valid'][0,vi]&t['mask_valid'][0,vi];gt=t['boxes_cxcywh'][0,vi][valid];score=p['objectness_logits'].sigmoid()
        pairs=match_detection(dict(instance_boxes_cxcywh=p['boxes'][None],instance_objectness_logits=p['objectness_logits'][None]),t['boxes_cxcywh'][0,vi][None],valid[None])[0]
        matches=[dict(query=int(q),target=int(j),iou=float(box_iou(p['boxes'][q:q+1],t['boxes_cxcywh'][0,vi,j:j+1])[0,0]),score=float(score[q])) for q,j in zip(*pairs)]
        records.append(dict(scene=scene,view=r['metadata']['view'],qualified_known=int(r['valid'].sum()),gt_instances=int(valid.sum()),gt_nose_feet_valid=int(t['keypoint_valid'][0,vi,valid][:,[0,14,15,16,17]].sum()),confident_queries=int((score>=.5).sum()),matched=matches))
    chosen=[next(r for r in records if r['qualified_known']==0),next(r for r in records if r['qualified_known']>0)]
    canvas=Image.new('RGB',(1830,1300),'white');draw=ImageDraw.Draw(canvas)
    for ri,e in enumerate(chosen):
        item=items[e['scene']];t=instance_targets_from_batch(item['batch']);source=lookup[e['scene'],e['view']]['record']
        for ci,v in enumerate([0,2,7]):
            r=lookup[e['scene'],v];vi=source['views'].index(v);rgb=item['batch']['images'][0,vi];im=Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8));w,h=im.size;di=ImageDraw.Draw(im);p=r['baseline'];scores=p['objectness_logits'].sigmoid()
            def rect(box,color,label):
                x,y,bw,bh=box.tolist();bounds=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h);di.rectangle(bounds,outline=color,width=3);di.text((bounds[0],max(0,bounds[1]-17)),label,font=font(16),fill=color)
            for j in torch.where(t['instance_valid'][0,vi]&t['box_valid'][0,vi]&t['mask_valid'][0,vi])[0].tolist():rect(t['boxes_cxcywh'][0,vi,j],'#00ee66','GT mask box')
            for q in torch.where(scores>=.5)[0].tolist():rect(p['boxes'][q],'red',f'q{q} score{scores[q]:.2f}')
            im.thumbnail((590,450));xx=ci*610;yy=ri*650;canvas.paste(im,(xx+(610-im.width)//2,yy+145))
            draw.text((xx+12,yy+10),('Top qualification FAIL' if ri==0 else 'Top qualification PASS'),font=font(22),fill='black')
            draw.text((xx+12,yy+48),f'view{v} {r["metadata"]["synthetic_orientation"]} known={int(r["valid"].sum())}',font=font(21),fill='black')
            draw.text((xx+12,yy+85),Path(e['scene']).name,font=font(18),fill='black')
    canvas.save(OUT/'top_qualification_examples.jpg',quality=95)
    write_json(OUT/'top_qualification_audit.json',dict(records=records,visual_cases=chosen,scope='TRAIN-only diagnostic, no thresholdtuning/annotationchanges. Redscore>=.5predictedboxes, greenmaskGT evaluationonly. Top exclusion fromazimuthcontrol preservesoriginalorientationexposure, not hidingVAL.'))
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()

