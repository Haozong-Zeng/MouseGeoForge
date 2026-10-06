"""Document unknown-label ranking caveat and repair derived overlay text clipping."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bottom_anchor_common_20261002 import ROOT, OUT, budget, write_state


def main():
    _,_,check=budget(120);check()
    dest=OUT/'quadruped_uncertainty_audit_v1'
    report=json.loads((dest/'summary.json').read_text())
    assert not (dest/'ranking_followup.json').exists()
    dm=json.loads((OUT/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text())
    entries=[dm['entries'][i] for i in dm['validation_indices']]
    arrays=[]
    for folder in ('quadruped_epoch110_review_v1','quadruped_seed29_epoch085_review_v1'):
        with np.load(OUT/folder/'predictions_source_pixels.npz') as z: arrays.append({k:z[k].copy() for k in z.files})
    a,b=arrays;known=np.isfinite(a['gt']).all(axis=-1)
    risks={'disagreement':np.linalg.norm(a['uv']-b['uv'],axis=-1),
           'low_min_confidence':-np.minimum(a['confidence'],b['confidence'])}
    influence={}
    for name,risk in risks.items():
        winner=np.argmax(risk,axis=1);score=np.max(risk,axis=1)
        selected=np.argsort(-score,kind='stable')[:57]
        influence[name]=dict(top20_percent_images=57,driving_point_unannotated=int(np.sum(~known[selected,winner[selected]])),
            all281_driving_point_unannotated=int(np.sum(~known[np.arange(281),winner])),
            unknown_does_not_mean_invisible=True)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',22)
    for r,filename in zip(report['shared_error_cases'][:6],report['gallery']):
        i=r['image_index'];j=r['keypoint_index'];e=entries[i]
        im=Image.open(ROOT/e['source']).convert('RGB');d=ImageDraw.Draw(im)
        for label,points,color in [('GT',a['gt'],'#70ff70'),('S17',a['uv'],'#ff77df'),('S29',b['uv'],'#40cfff')]:
            x,y=points[i,j];d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
            text=f'{label} id{dm["ids_1based"][j]}'
            bbox=d.textbbox((0,0),text,font=font,stroke_width=1)
            tx=min(max(x+10,2),im.width-(bbox[2]-bbox[0])-5)
            ty=min(max(y+{'GT':-30,'S17':0,'S29':25}[label],2),im.height-35)
            d.text((tx,ty),text,font=font,fill=color,stroke_width=1,stroke_fill='black')
        im.save(dest/filename,quality=95)
    followup=dict(image_ranking_unknown_driver=influence,
        interpretation='Many image-level maximum-uncertainty scores are driven by unannotated outputs. Their errors cannot be evaluated, so poor recall of known errors is not proof they are bad annotation targets. It does invalidate treating observed-point ranking performance as whole-image known-error detection performance.',
        next='Prepare bounded continuous-clip temporal reliability audit with frozen Quadruped checkpoints, no retraining/adaptation, no label correction. Preserve unknown visibility; no new visibility negatives inferred from missing labels. Choose clips away from excluded groups, use all cameras, evaluate only existing center-frame labels.',
        original_labels_changed=False,production_acceptance_threshold_fit=False,locked_holdouts_accessed=False)
    (dest/'ranking_followup.json').write_text(json.dumps(followup,indent=2))
    export=Path('/mnt/c/Users/55395/Downloads/MouseGeoForge-BottomAnchor-20261002')/dest.name
    for filename in report['gallery']+['ranking_followup.json']:shutil.copy2(dest/filename,export/filename)
    write_state(latest_decision=str(dest/'ranking_followup.json'),next_action=followup['next'])
    print(json.dumps(followup,indent=2))


if __name__=='__main__':main()
