"""Verify the second-order-seed paired experiment without changing its protocol."""
import fcntl
import json
import math
from collections import Counter
from train_cylinder_temperature_seed29 import ROOT,OUTPUT,ARMS,build_protocol,first
from run_depth_pose_ablation import torch,write_json,fingerprint,state_hash
from mousegeoforge.adaptive_temperature import PREFIX


def main():
    torch.set_num_threads(2)
    spec=json.loads((OUTPUT/'protocol.json').read_text());result=json.loads((OUTPUT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert spec==build_protocol(spec['config'])
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    parent=json.loads((first.OUTPUT/'protocol.json').read_text())
    assert all(spec[k]==parent[k] for k in ('records','evaluation','checkpoint_sha256','encoder_sha256','manifest_sha256'))
    order=spec['seed29_permutation'];assert sorted(order)==list(range(800)) and order!=list(range(800))
    assert spec['schedule']==list(range(800))
    assert spec['arms'][ARMS[0]]==spec['arms'][ARMS[1]]
    assert Counter(tuple(x) for x in spec['arms'][ARMS[0]])==Counter(tuple(x) for x in parent['arms'][first.ARM])
    assert json.loads((OUTPUT/'baseline_pretrain_verification.json').read_text())['passed']
    assert len(result['baseline']['records'])==88
    previous=json.loads((first.OUTPUT/first.ARM/'history.json').read_text())
    reference=None;initials=[];verified={}
    for arm in ARMS:
        rows=json.loads((OUTPUT/arm/'history.json').read_text())
        scales=json.loads((OUTPUT/arm/'temperature_history.json').read_text());assert len(rows)==len(scales)==800
        images=tokens=0
        for i,(row,scale) in enumerate(zip(rows,scales)):
            old=previous[order[i]]
            for k in ('scene','views','image_hw','slot_kind','learning_rate'):assert row[k]==old[k]
            assert row['step']==scale['step']==i+1 and row['pose_loss']==scale['pose_loss']
            images+=len(row['views']);h,w=row['image_hw'];tokens+=len(row['views'])*(h//16)*(w//16)
            assert row['images_seen']==images and row['patch_tokens_seen']==tokens
            assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
            assert .5<=scale['min']<=scale['max']<=8.000001
            if arm==ARMS[0]:assert scale['min']==scale['max']==1
            if reference:
                for k in ('scene','views','image_hw','slot_kind','step','learning_rate','images_seen','patch_tokens_seen'):assert row[k]==reference[i][k]
        assert scales[0]['min']==scales[0]['max']==1
        assert images==1800 and tokens==3540384
        reference=rows
        cp=torch.load(OUTPUT/arm/'final_heads.pt',weights_only=True,map_location='cpu')
        assert state_hash(cp['model'])==cp['trained_hash']!=cp['initial_hash']
        assert cp['protocol_sha256']==result['protocol_sha256'];initials.append(cp['initial_hash'])
        extra={k:v for k,v in cp['model'].items() if k.startswith(PREFIX)}
        assert len(extra)==2 and sum(x.numel() for x in extra.values())==257
        updated=any(bool(x.abs().max()>0) for x in extra.values());assert updated==(arm==ARMS[1])
        res=result['arms'][arm];assert res['checkpoint_reload_verified'] and res['encoder_gradients_absent']
        assert res['images_seen']==images and res['patch_tokens_seen']==tokens and len(res['records'])==88
        for idx in range(88):
            saved=torch.load(OUTPUT/arm/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            diag=torch.load(OUTPUT/arm/f'diagnostics_{idx:03d}.pt',weights_only=True,map_location='cpu')
            assert saved['record']==diag['record'] and saved['oracle_assignment']==diag['oracle_assignment']
            for k,v in saved['predictions'].items():torch.testing.assert_close(v,diag['predictions'][k],atol=0,rtol=0)
            scale=diag['temperature_scale'];assert torch.isfinite(scale).all() and scale.min()>=.5 and scale.max()<=8.000001
            if arm==ARMS[0]:assert (scale==1).all()
        verified[arm]=dict(steps=800,images=images,patch_tokens=tokens,new_parameters_updated=updated,
            checkpoint_sha256=fingerprint(OUTPUT/arm/'final_heads.pt'),training_seconds=res['training_seconds'])
    assert len(set(initials))==1
    report=dict(passed=True,protocol_rebuilt_exactly=True,identical_input_multiset_to_seed17=True,
        paired_stepwise_budgets_equal=True,initial_parameter_states_equal=True,baseline_prediction_difference=0,
        diagnostics_prediction_difference=0,seed_scope='Adaptation input order only; same pretrained weights and data.',arms=verified)
    write_json(OUTPUT/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
