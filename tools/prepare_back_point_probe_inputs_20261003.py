"""CPU-only immutable06/07point probe inputs; no label or model update."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.uncertain_back_points import back_axes
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
OUT=CYCLE/'back_point_probe_inputs_v1'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    initial=CYCLE/'training_v1/pilot/initial_heads.pt';state=torch.load(initial,map_location='cpu',weights_only=True)
    head=MultiInstanceMouseHead(num_keypoints=23).eval();prefix='instance_pose_head.'
    head.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);head.requires_grad_(False)
    sources={'synthetic':CYCLE/'expanded_curve_features_v1/frozen_inputs.pt','real':CYCLE/'real_curve_adaptation_inputs_v1/frozen_inputs.pt'}
    manifest={};file_hashes={}
    for kind,path in sources.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);old=saved['tensors'];meta=saved['metadata'];values=collections.defaultdict(list)
        with torch.no_grad():coarse=head.coarse_keypoint_xy(old['pose'].reshape(-1,4,256)[:,1:3]).sigmoid()
        for i,m in enumerate(meta):
            if kind=='real':
                assert m['session'] not in ('GR2','UR2','UR3')
                target=old['observed_xy'][i];valid=old['observed_valid'][i]
                source=ROOT/m['scene']/'surface_aux.npz';file_hashes[str(source.relative_to(ROOT))]=fingerprint(source)
                with np.load(source) as z:
                    v=z['view_ids'].tolist().index(m['view']);tangent=torch.tensor(z['tangent'][v,0,5:7]);tv=torch.tensor(z['tangent_valid'][v,0,5:7])
            else:
                source=ROOT/m['scene']/'keypoints.npz';file_hashes[str(source.relative_to(ROOT))]=fingerprint(source)
                with np.load(source) as z:
                    v=z['view_ids'].tolist().index(m['view']);uv=z['uv'][v];flags=z['xy_label_valid'][v] if 'xy_label_valid' in z else z['in_frame'][v]
                    if uv.ndim==3:uv=uv[m['instance']];flags=flags[m['instance']]
                images=list((ROOT/m['scene']/'blended_images').glob(f'{m["view"]:08d}.*'));assert len(images)==1
                with Image.open(images[0]) as image:width,height=image.size
                target=torch.tensor(uv[5:7]/[width,height],dtype=torch.float32)
                valid=torch.tensor(flags[5:7],dtype=torch.bool)&torch.isfinite(target).all(-1)&(target>=0).all(-1)&(target<=1).all(-1)
                tangent=torch.zeros(2,2);tv=torch.zeros(2,dtype=torch.bool)
            pixels=old['pixels'][i].float();primary_t,primary_r=back_axes(target*pixels,valid,primary_tangent=tangent,primary_valid=tv)
            fallback_t,fallback_r=back_axes(target*pixels,valid,primary_tangent=tangent,primary_valid=tv,use_pair_fallback=True)
            for key,val in dict(pose=old['pose'][i],coarse_xy=coarse[i],gt_xy=target,valid=valid,pixels=pixels,
                primary_tangent=primary_t,primary_reliability=primary_r,pair_tangent=fallback_t,pair_reliability=fallback_r).items():values[key].append(val)
        tensors={k:torch.stack(v) for k,v in values.items()}
        torch.save(dict(tensors=tensors,metadata=meta),OUT/f'{kind}_inputs.pt')
        manifest[kind]=dict(observations=len(meta),supervised_points=int(tensors['valid'].sum()),
            primary_axes=int((tensors['primary_reliability']>0).sum()),fallback_policy_axes=int((tensors['pair_reliability']>0).sum()),
            source_sha256=fingerprint(path))
    write_json(OUT/'manifest.json',dict(inputs=manifest,source_files=file_hashes,initial_heads_sha256=fingerprint(initial),
        prediction_base='Frozen ORIGINAL coarse_keypoint_xy of cached06/07decoder tokens, NOT existing local-refined point output; any probe must explicitly report this distinction.',
        scope='New two-point experimental readout only; original Core23outputs unchanged. Synthetic positions exact; real manual positions weak/unknownvisibility.',
        fallback='06->07chord if bothobserved and length>=20px, reliability0.5, onlywhereprimaryaxis unavailable. No default productionpolicychanged.',
        validation='Existing source roles frozen; no validation observation in training inputs chosen beyond role metadata. Lockedrealneverloaded.',
        no_groundtruth_in_prediction_base=True,no_labels_changed=True))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
