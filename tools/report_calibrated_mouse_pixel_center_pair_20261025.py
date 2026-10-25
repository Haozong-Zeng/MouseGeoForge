"""Compact matched-label pilot report; absent/partial attempts stay explicit."""
from pathlib import Path
import json,datetime,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';BASE=OUT/'pixel_center_pair_v1'
def main():
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),comparable=False,arms={},primary_native='FP32 bilinear probability>.5; original native labels for both',selection='One fixed seed17/fixed6000 pair, reused DEV exploratory only; no selection/promotion',encoder_unfreeze=False,existing_ROI_mask_generator_unchanged=True)
    complete=BASE/'completion.json';result['campaign']=json.loads(complete.read_text()) if complete.exists() else {'status':'not_completed'}
    protocols={}
    expected={r['id'] for r in json.loads((ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json').read_text())['items'] if r['split']=='dev'}
    for policy in ['legacy','center_exact']:
        folder=BASE/(policy+'_dev');p=folder/'evaluation.json'
        if not p.exists():result['arms'][policy]={'status':'evaluation_unavailable'};continue
        d=json.loads(p.read_text());assert len(d['ema'])==879 and {r['id'] for r in d['ema']}==expected
        protocol=json.loads((folder/'protocol.json').read_text());assert protocol['label_policy']==policy and protocol['native_truth_policy']=='original pixels; no GT resize'
        protocols[policy]=protocol
        groups={}
        for name,rows in [('all',d['ema']),*[(f'{domain}/{bottom}',[r for r in d['ema'] if r['domain']==domain and r['bottom']==bottom]) for domain in ['real','synthetic'] for bottom in [True,False]]]:
            groups[name]=dict(images=len(rows),**{key:float(np.mean([r['native'][key] for r in rows])) for key in ['iou','dice','boundary_f1']})
        result['arms'][policy]=dict(status='complete',checkpoint_sha256=d['checkpoint_sha256'],evaluation_sha256=digest(p),groups=groups)
    if len(protocols)==2 and result['campaign'].get('status')=='completed':
        assert {k:v for k,v in protocols['legacy'].items() if k!='label_policy'}=={k:v for k,v in protocols['center_exact'].items() if k!='label_policy'}
        assert result['campaign']['full_paired_endpoint']==6000
        result['comparable']=True;result['center_minus_legacy']={g:{k:result['arms']['center_exact']['groups'][g][k]-result['arms']['legacy']['groups'][g][k] for k in ['iou','dice','boundary_f1']} for g in result['arms']['legacy']['groups']}
    (OUT/'pixel_center_pair_report_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'docs/experiments/calibrated_mouse_pixel_center_pair_20261025.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='campaign'}))
if __name__=='__main__':main()
