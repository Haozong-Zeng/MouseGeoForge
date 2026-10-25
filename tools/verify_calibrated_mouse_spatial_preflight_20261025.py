"""Check actual engine logs, cached-feature GPU receipt and seeded base initializations."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025 import CalibratedRayDepthSpatial
from mousegeoforge.semseg_training import digest
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
def main():
 dest=OUT/'spatial_depth_preflight_gate_v1.json';assert not dest.exists();reg=json.loads((OUT/'registrations/spatial_depth_pair_v1.json').read_text())
 for f,h in reg['source_sha256'].items():assert digest(ROOT/f)==h
 assert digest(OUT/'spatial_depth_CPU_preflight_v1/receipt.json')==reg['CPU_preflight_sha256']
 gpu=json.loads((OUT/'spatial_depth_actual_GPU_preflight_v1/receipt.json').read_text());assert gpu['passed'] and gpu['device']=='cuda' and gpu['actual_released_cached_features'] and gpu['feature_values_and_file_unchanged']
 for f,h in gpu['source_sha256'].items():assert reg['source_sha256'][f]==h
 assert gpu['roi_cache_receipt_sha256']==reg['roi_cache_receipt_sha256']
 checks={};basehash={}
 for kernel in (1,3):
  folder=OUT/f'spatial_depth_k{kernel}_engine_smoke_v1';receipt=json.loads((folder/'completion.json').read_text());assert receipt['passed'] and receipt['steps']==3 and receipt['checkpoint_forward_roundtrip_exact'] and receipt['full_next_update_resume_tested']
  ck=torch.load(folder/'step_000003.pt',map_location='cpu',weights_only=False);p=ck['protocol'];assert p['command']=='smoke' and p['neck_kernel']==kernel and p['neck_parameters']==18432 and p['feature_mode']=='roi';assert (p['warmup'],p['cosine_floor'],p['peak_lr'])==(100,.1,.001)
  assert len(ck['records'])==3
  for record in ck['records']:
   step=record['step']-1;lr=.001*min(1,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/3)));assert abs(lr-record['lr'])<1e-16
  assert json.loads((folder/'neck_second_update_gradient.json').read_text())['first_neck_nonzero']
  checks[str(kernel)]=dict(receipt_sha256=digest(folder/'completion.json'),actual_all_engine_smoke_LRs_verified=True,next_update_max_difference=receipt['next_update_parameter_max_difference'])
 for seed in (17,29,43):
  values={}
  for kernel in (1,3):
   torch.manual_seed(seed);m=CalibratedRayDepthSpatial(neck_kernel=kernel);h=hashlib.sha256()
   for name,value in m.state_dict().items():
    if name.startswith('neck.'):continue
    h.update(name.encode());h.update(value.detach().numpy().tobytes())
   values[str(kernel)]=h.hexdigest()
  assert values['1']==values['3'];basehash[str(seed)]=values['1']
 result=dict(passed=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),engineering_only=True,actual_GPU_operator_receipt_sha256=digest(OUT/'spatial_depth_actual_GPU_preflight_v1/receipt.json'),engine_smokes=checks,identical_initial_base_sha256_by_seed=basehash,neck_parameters_each=18432,parameter_matching='1x1 hidden288 vs3x3 hidden32; same18432biasfreeweights, differentactivationwidth disclosed',no_scientific_convergence_claim=True)
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
