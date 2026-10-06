"""Measure repeated-control numerical drift; do not assume deterministic training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
C=ROOT/'output/singleview_fusion_cycle_20261005';OLD=C/'mixed_real_pool_control_v1';NEW=C/'mixed_rotation_control_v1'
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def main():
    a=json.loads((OLD/'protocol.json').read_text());b=json.loads((NEW/'protocol.json').read_text());assert a['source_sha256']==b['source_sha256'] and a['schedules']['expanded_real']==b['schedules']['native_rgb'];p=read(OLD/'expanded_real_predictions.pt');q=read(NEW/'native_rgb_predictions.pt');rows={r['index']:r for r in read(C/'mixed_local_rgb_inputs_v1/evaluation_only.pt')};details=[]
    for i in p:
        delta=torch.linalg.vector_norm((p[i]['xy']-q[i]['xy'])*rows[i]['pixels'],dim=-1);details.append(dict(index=i,stage=rows[i]['metadata']['stage'],kind=rows[i]['metadata']['kind'],peak_mean_px=float(delta.mean()),top1_mean_px=float(delta[:,0].mean()),peak_max_px=float(delta.max()),probability_map_max_abs=float((p[i]['probability_map']-q[i]['probability_map']).abs().max()) if 'probability_map' in p[i] else None))
    cp=read(OLD/'expanded_real_step1600.pt');cq=read(NEW/'native_rgb_step1600.pt');weightdelta={part:max(float((cp[part][k]-cq[part][k]).abs().max()) for k in cp[part] if cp[part][k].is_floating_point()) for part in ['backbone','readout']};summary={}
    for stage in ['fit','calibration']:
        summary[stage]={}
        for kind in ['real','single','dual']:
            rr=[r for r in details if r['stage']==stage and r['kind']==kind];summary[stage][kind]=dict(images=len(rr),top1_mean_px=float(np.mean([r['top1_mean_px'] for r in rr])),peak_max_px=max(r['peak_max_px'] for r in rr))
    result=dict(same_initial_checkpoint=True,same_training_schedule=True,bitwise_equal=all(z==0 for z in weightdelta.values()),max_abs_weight_delta=weightdelta,prediction_drift=summary,details=details,caveat='Training used CUDA without deterministic-algorithm enforcement. Small weight drift can switch nearly tied peak rankings; peak_max_px compares identical RANK slots and is NOT candidate-set drift or GT error. Native repetition is contemporaneous primary control, previous expanded arm is only replication audit. Do not claim exact reproducibility.')
    (NEW/'native_replay_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
