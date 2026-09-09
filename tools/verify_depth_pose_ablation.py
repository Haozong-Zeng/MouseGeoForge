"""Reload every final head checkpoint and independently repeat held-out evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_depth_pose_ablation import (ROOT, torch, VGGTOmega, DepthPoseModel, load_batch,
    fingerprint, restore, evaluate, write_json)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/depth_pose_ablation_pilot_v1')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'vggt-omega/vggt_omega_1b_512.pt')
    a=p.parse_args()
    results=json.loads((a.output/'results.json').read_text())
    assert fingerprint(a.checkpoint)==results['provenance']['checkpoint_sha256']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    original=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    selected={k:v for k,v in original.items() if k.startswith(('aggregator.','dense_head.'))}
    result=spatial.load_state_dict(selected,strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys)
    del original,selected
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval(); del spatial
    manifest=json.loads((a.output/'manifest.json').read_text())
    data=[]
    for r in manifest['records']:
        if r['split']!='val': continue
        b=load_batch(r)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached,patch_start=model.aggregator(b['images'].cuda())
        data.append(dict(record=r,batch=b,patch_start=patch_start,
                         cached=[x.cpu() if x is not None and i in (4,11,17,23) else None for i,x in enumerate(cached)]))
    records=[]
    for r in results['runs']:
        folder=a.output/f"seed{r['seed']}_{r['mode']}"
        ckpt=torch.load(folder/'final_heads.pt',map_location='cpu',weights_only=True)
        assert ckpt['seed']==r['seed'] and ckpt['coupling']==r['mode']
        assert ckpt['checkpoint_sha256']==results['provenance']['checkpoint_sha256']
        restore(model,ckpt['model']); del ckpt
        check=evaluate(model,data,r['mode'],'val')
        expected=json.loads((folder/'validation.json').read_text())
        max_error=0.
        for actual,saved in zip(check['records'],expected['records'],strict=True):
            assert actual['scene']==saved['scene']
            for key,value in actual['metrics'].items():
                target=saved['metrics'][key]
                if value is None: assert target is None; continue
                error=abs(value-target); max_error=max(max_error,error)
                assert error<1e-6,(r['seed'],r['mode'],key,error)
        records.append(dict(seed=r['seed'],mode=r['mode'],max_metric_difference=max_error,
                            validated_scenes=len(check['records']),passed=True))
        print(records[-1],flush=True)
    write_json(a.output/'checkpoint_verification.json',dict(passed=True,runs=records))


if __name__=='__main__': main()
