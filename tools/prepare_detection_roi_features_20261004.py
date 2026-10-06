"""Prediction-only ROI features from immutable frozen patch memories."""
import datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from torch.nn import functional as F
from mousegeoforge.models.spatial_detection_readout import sample_predicted_roi
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'detection_roi_features_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    manifest=json.loads((CYCLE/'detection_features_v1/manifest.json').read_text());rows=[]
    for r in manifest['records']:
        assert fingerprint(ROOT/r['path'])==r['sha256'];rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    allowed={}
    for source in ['features_v1','features_expanded_synthetic_v1','features_expanded_all_v1']:
        for r in json.loads((CYCLE/source/'manifest.json').read_text())['records']:allowed[r['path']]=r['sha256']
    roi=[];global_features=[];hashes={};last_path=None;memory=None
    for i,o in enumerate(rows):
        m=o['metadata'];assert m['record'].get('session') not in ['GR2','UR2','UR3'];path=m['memory_part']
        if path!=last_path:
            if path not in hashes:hashes[path]=fingerprint(ROOT/path);assert hashes[path]==allowed[path]
            memory=F.interpolate(torch.load(ROOT/path,map_location='cpu',weights_only=False)['memory'].float(),(48,48),mode='bilinear',align_corners=False).cuda();last_path=path
        with torch.no_grad():a,b=sample_predicted_roi(memory[m['view_index']:m['view_index']+1],o['original_boxes'][None].cuda())
        roi.append(a[0].half().cpu());global_features.append(b[0].half().cpu())
    dest=OUT/'features.pt';torch.save(dict(roi=torch.stack(roi),global_memory=torch.stack(global_features),metadata=[o['metadata'] for o in rows]),dest)
    write_json(OUT/'manifest.json',dict(path=dest.relative_to(ROOT).as_posix(),sha256=fingerprint(dest),images=len(rows),memory_hashes=hashes,source_sha256=fingerprint(CYCLE/'detection_features_v1/manifest.json'),script_sha256=fingerprint(Path(__file__)),contract='Float32 48x48 resized original frozen patch memory. Prediction-only5x5 ROI spanning2x original box; border padding; stored fp16. Global full-image mean is matched control. No GT used for feature sampling.'))
    write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))

if __name__=='__main__':main()
