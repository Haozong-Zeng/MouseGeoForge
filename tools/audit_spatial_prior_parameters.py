"""CPU checkpoint audit, base geometry only; not image-conditioned trajectories."""
import json
import torch
from run_depth_pose_ablation import ROOT, fingerprint, write_json
from mousegeoforge.spatial_query_prior import ConditionalSpatialPrior


def main():
    folder=ROOT/'output/spatial_query_prior_seed17_v1';path=folder/'parameter_audit.json'
    if path.exists():raise RuntimeError('Preserve completed audit')
    initial=ConditionalSpatialPrior(2,6,256).state_dict();arms={};hashes={}
    for arm in ('spatial_prior_off','spatial_prior_on'):
        file=folder/arm/'final_heads.pt';state=torch.load(file,map_location='cpu',weights_only=True)
        hashes[str(file)]=fingerprint(file);rows={}
        for name,value in initial.items():
            trained=state['model']['instance_pose_head.spatial_query_prior.'+name]
            delta=trained-value
            rows[name]=dict(delta_l2=float(delta.norm()),delta_max=float(delta.abs().max()),changed_elements=int(delta.count_nonzero()))
            if name=='center_logits':rows[name]['base_xy']=trained.sigmoid().tolist()
            elif name=='sigma_logits':rows[name]['base_sigma']=(.08+.92*trained.sigmoid()).tolist()
            elif name=='strength_logits':rows[name]['strength']=torch.nn.functional.softplus(trained).clamp(max=2.).tolist()
        assert state['steps']==1600
        if arm.endswith('_off'):assert all(v['delta_l2']==0 for v in rows.values())
        else:assert all(v['delta_l2']>0 for v in rows.values())
        arms[arm]=rows
    write_json(path,dict(passed=True,arms=arms,checkpoint_sha256=hashes,model_forwards=0,optimizer_steps=0,
                        limitation='Base centers/scales exclude conditional offsets. Nonzero parameter updates do not prove useful attention or instance recovery.'))
    print('Parameter audit saved')


if __name__=='__main__':main()
