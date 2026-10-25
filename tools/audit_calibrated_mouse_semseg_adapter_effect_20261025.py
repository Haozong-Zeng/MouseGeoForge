"""CPU-only final effect audit; fixed completed endpoints, no model calls/selection."""
from pathlib import Path
import datetime as dt,hashlib,json,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from mousegeoforge.semseg_training import digest
from mousegeoforge.semseg_adapter_safety_20261025 import violations
OUT=ROOT/'output/calibrated_mouse_next12h_20261025';BASE=OUT/'semseg_adapter_pair_v1'
def main():
 dest=OUT/'semseg_adapter_effect_audit_v1.json';assert not dest.exists();reg=json.loads((OUT/'registrations/semseg_adapter_effect_audit_v1.json').read_text())
 for f,h in reg['source_sha256'].items():assert digest(ROOT/f)==h
 report=json.loads((OUT/'semseg_adapter_pair_report_v1.json').read_text());assert report['comparable'] and report['full_registered_endpoint_completed'];evaluation={policy:json.loads((BASE/f'{policy}_dev/evaluation.json').read_text()) for policy in ['control','adapter']};rows={policy:{r['id']:r for r in data['ema']} for policy,data in evaluation.items()};assert rows['control'].keys()==rows['adapter'].keys() and len(rows['control'])==879
 pixels=changed=identical=0;deltas=[];max_image_diff=0;input_hashes={};perimage=[]
 for identifier,a in rows['control'].items():
  b=rows['adapter'][identifier];masks=[]
  for policy in ['control','adapter']:
   f=BASE/f'{policy}_dev/masks_ema/{identifier}.png';input_hashes[str(f.relative_to(ROOT))]=digest(f);masks.append(np.asarray(Image.open(f))>0)
  assert masks[0].shape==masks[1].shape;count=int(np.count_nonzero(masks[0]!=masks[1]));pixels+=masks[0].size;changed+=count;identical+=count==0;max_image_diff=max(max_image_diff,count);delta=b['native']['iou']-a['native']['iou'];deltas.append(delta);perimage.append(dict(id=identifier,changed_pixels=count,native_iou_delta=delta))
 assert abs(float(np.mean(deltas))-report['adapter_minus_control']['all']['iou'])<1e-12
 checks={};baseline=json.loads((OUT/'semseg_roi_pair_v2/full_006000/train_monitor_006000.json').read_text())
 for policy in ['control','adapter']:
  ckfile=BASE/f'{policy}_003000/step_003000.pt';ck=torch.load(ckfile,map_location='cpu',weights_only=False);assert ck['step']==3000;assert all(not v for k,v in ck['requires_grad'].items() if k.startswith('encoder.'));assert all(v==(policy=='adapter') for k,v in ck['requires_grad'].items() if k.startswith('adapters.'))
  norms={state:{k:float(v.double().norm()) for k,v in ck[state].items() if k.startswith('adapters.') and '.2.weight' in k} for state in ['model','ema']}
  if policy=='adapter':assert all(len(values)==4 and all(x>0 for x in values.values()) for values in norms.values())
  else:assert all(not values for values in norms.values())
  windows={}
  for endpoint in range(500,3001,500):
   f=BASE/f'{policy}_{endpoint:06d}/train_monitor_{endpoint:06d}.json';windows[str(endpoint)]=violations(json.loads(f.read_text()),baseline);assert windows[str(endpoint)]==[]
  checks[policy]=dict(checkpoint_sha256=digest(ckfile),encoder_frozen=True,adapter_last_layer_norms=norms,all_six_TRAIN_violation_lists=windows)
 parent=json.loads((OUT/'semseg_roi_pair_report_v2.json').read_text())['arms']['full']['groups']['all']['iou'];result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=digest(Path(__file__)),paired_report_sha256=digest(OUT/'semseg_adapter_pair_report_v1.json'),images=879,total_native_pixels=pixels,changed_native_pixels=changed,changed_fraction=changed/pixels,identical_mask_images=identical,different_mask_images=879-identical,max_changed_pixels_per_image=max_image_diff,perimage_iou=dict(improved=sum(x>0 for x in deltas),degraded=sum(x<0 for x in deltas),equal=sum(x==0 for x in deltas),mean_delta=float(np.mean(deltas))),original_parent_native_iou=parent,final_control_minus_parent=report['arms']['control']['groups']['all']['iou']-parent,final_adapter_minus_parent=report['arms']['adapter']['groups']['all']['iou']-parent,actual_checkpoints_and_TRAIN_checks=checks,input_mask_sha256=input_hashes,rows=perimage,optimization_updates=0,model_calls=0,interpretation='Actualadapterweightsareupdated/nonzero andallregisteredTRAINsafetywindowspass; observednativeDEV effectis negligibleatthisfixed3000schedule. No practicalbenefit established; no extension/promotion or claimalladapterroutesfail.',limitations=report['limitations'])
 dest.write_text(json.dumps(result,indent=2)+'\n');compact={k:v for k,v in result.items() if k not in ['input_mask_sha256','rows']};compact['full_audit_sha256']=digest(dest);(ROOT/'docs/experiments/calibrated_mouse_next12h_semseg_adapter_effect_20261025.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(compact))
if __name__=='__main__':main()
