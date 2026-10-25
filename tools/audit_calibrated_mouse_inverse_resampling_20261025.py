"""Fixed inverse-resampling diagnostic of already-complete DEV predictions.

This is an exploratory operator comparison, never replacement of primary masks,
threshold selection, a new checkpoint evaluation, or model promotion.
"""
from pathlib import Path
from collections import defaultdict
import argparse,datetime as dt,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from mousegeoforge.semseg_training import digest,mask_metrics
OLD=ROOT/'output/calibrated_mouse_10h_20261025';OUT=ROOT/'output/calibrated_mouse_resume8h_20261025'

def threshold_saved_probability(probability):
    assert probability.ndim==2 and np.isfinite(probability).all()
    assert probability.min()>=0 and probability.max()<=1
    # Class argmax tie policy is background. Saved probabilities are float16.
    return (probability.astype(np.float32)>.5).astype(np.uint8)

def target_roundtrip(target,side):
    h,w=target.shape;scale=side/max(h,w);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    small=cv2.resize(target.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small,(w,h),interpolation=cv2.INTER_NEAREST)

def group_summary(rows):
    result={}
    for key in ['legacy_nearest_mask','bilinear_probability_fixed_half','GT_nearest_roundtrip_self_agreement']:
        result[key]={k:float(np.mean([r[key][k] for r in rows])) for k in ['iou','dice','boundary_f1']}
    for key in ['changed_valid_pixels','valid_pixels','GT_foreground_pixels','legacy_false_negative_pixels','bilinear_false_negative_pixels','legacy_false_positive_pixels','bilinear_false_positive_pixels','saved_probability_ambiguous_near_half_pixels']:
        result[key]=sum(r[key] for r in rows)
    result['images']=len(rows)
    result['changed_valid_fraction']=result['changed_valid_pixels']/max(1,result['valid_pixels'])
    result['native_original_iou_delta_mean']=result['bilinear_probability_fixed_half']['iou']-result['legacy_nearest_mask']['iou']
    result['native_original_boundary_f1_delta_mean']=result['bilinear_probability_fixed_half']['boundary_f1']-result['legacy_nearest_mask']['boundary_f1']
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT/'inverse_resampling_audit_v1.json');args=ap.parse_args()
    assert not args.output.exists()
    manifest=OLD/'semseg_items.json';items=[r for r in json.loads(manifest.read_text())['items'] if r['split']=='dev'];assert len(items)==879
    reference_audit=json.loads((OUT/'native_metric_audit_v1.json').read_text());references={a:{r['id']:r for r in v['rows']} for a,v in reference_audit['arms'].items()}
    groups={};all_rows=[];receipts={};truth_cache={}
    for arm in ['L','M']:
        directory=OUT/(arm+'_dev_v1');saved=json.loads((directory/'evaluation.json').read_text());protocol=json.loads((directory/'protocol.json').read_text());assert {r['id'] for r in saved['ema']}=={r['id'] for r in items}
        assert saved['checkpoint_sha256']==reference_audit['arms'][arm]['checkpoint_sha256']
        receipts[arm]=dict(checkpoint_sha256=saved['checkpoint_sha256'],evaluation_sha256=digest(directory/'evaluation.json'),protocol_sha256=digest(directory/'protocol.json'),side=protocol['side'])
        for item in items:
            label=ROOT/item['label'];assert digest(label)==item['label_sha256']
            if item['id'] not in truth_cache:
                if label.suffix=='.npz':
                    with np.load(label) as z:truth=z['body'].astype(np.uint8)
                else:
                    raw=np.array(Image.open(label).convert('L'));truth=raw if item.get('label_encoding')=='class_index_0_1_255' else (raw>0).astype(np.uint8)
                truth_cache[item['id']]=truth
            truth=truth_cache[item['id']];maskpath=directory/'masks_ema'/(item['id']+'.png');probpath=directory/'masks_ema'/(item['id']+'_probability.npz')
            legacy=(np.array(Image.open(maskpath).convert('L'))>0).astype(np.uint8)
            with np.load(probpath) as z:prob=z['foreground'];affine=z['pixel_center_affine']
            assert prob.dtype==np.float16 and prob.shape==legacy.shape==truth.shape
            alternate=threshold_saved_probability(prob);valid=truth!=255;gt=truth==1
            metrics=mask_metrics(legacy,truth,1)
            # Independent replay must match the already-fixed original-native audit.
            reference=references[arm][item['id']];expected=reference['native_original']
            assert digest(maskpath)==reference['mask_sha256']
            for key in ['iou','dice','boundary_f1']:assert abs(metrics[key]-expected[key])<1e-12
            row=dict(id=item['id'],arm=arm,domain=item['domain'],bottom=item['bottom'],group=item['group'],image=item['image'],
                label_sha256=item['label_sha256'],mask_sha256=digest(maskpath),probability_sha256=digest(probpath),pixel_center_affine=affine.tolist(),
                legacy_nearest_mask=metrics,bilinear_probability_fixed_half=mask_metrics(alternate,truth,1),
                GT_nearest_roundtrip_self_agreement=mask_metrics(target_roundtrip(truth,protocol['side']),truth,1),
                changed_valid_pixels=int(((alternate!=legacy)&valid).sum()),valid_pixels=int(valid.sum()),GT_foreground_pixels=int((gt&valid).sum()),
                legacy_false_negative_pixels=int(((legacy==0)&gt&valid).sum()),bilinear_false_negative_pixels=int(((alternate==0)&gt&valid).sum()),
                legacy_false_positive_pixels=int(((legacy==1)&~gt&valid).sum()),bilinear_false_positive_pixels=int(((alternate==1)&~gt&valid).sum()),
                saved_probability_ambiguous_near_half_pixels=int(((np.abs(prob.astype(np.float32)-.5)<=2**-11)&valid).sum()))
            all_rows.append(row)
        selected=[r for r in all_rows if r['arm']==arm];subsets={'all':selected}
        for domain in ['real','synthetic']:
            for bottom in [True,False]:subsets[f'{domain}/{bottom}']=[r for r in selected if r['domain']==domain and r['bottom']==bottom]
        groups[arm]={k:group_summary(v) for k,v in subsets.items()}
    result=dict(schema='fixed-inverse-resampling-diagnostic-v1',completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=digest(__file__),manifest_sha256=digest(manifest),reference_original_native_audit_sha256=digest(OUT/'native_metric_audit_v1.json'),
        receipts=receipts,groups=groups,rows=all_rows,threshold=.5,tie_policy='background',threshold_or_checkpoint_selection=False,
        primary_prediction_masks_unchanged=True,probability_precision='saved float16; pixels within2^-11 of.5 separately reported, not assumed exact FP32 inference',
        limitations=['Prospective fixed inverse operator diagnostic on reused DEV; no new independent generalization claim.',
            'GT down/up self-agreement quantifies this operator only, is NOT a model accuracy upper bound or attainable ceiling.',
            'Manual real masks and synthetic opaque-skin surrogate kept separate.',
            'No alternative masks written into current pipelines, no threshold tuning/filtering or model promotion.'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    compact={k:v for k,v in result.items() if k!='rows'};compact['full_audit_sha256']=digest(args.output)
    (ROOT/'docs/experiments/calibrated_mouse_inverse_resampling_20261025.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps({'groups':groups}))
if __name__=='__main__':main()
