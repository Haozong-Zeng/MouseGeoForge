"""Verify exact sampling control and real-data loss parity with shared matching."""
import argparse
import fcntl
import json
import math
from unittest.mock import patch

from train_cylinder_coarse_aux import ROOT,OUTPUT,PARENT,ARM,build_protocol
from run_depth_pose_ablation import torch,fingerprint,write_json,state_hash
from mousegeoforge.coarse_pose_supervision import coarse_aux_pose_loss,upstream


def loss_smoke():
    saved=torch.load(ROOT/'output/cylinder_localization_audit_v1/new_sampling/prediction_000.pt',weights_only=True,map_location='cpu')
    p,t=saved['predictions'],saved['targets']
    weights=upstream.InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
    baseline,_=upstream.instance_pose_loss(p,t,weights,image_size=tuple(saved['record']['image_hw']))
    original_matcher=upstream.hungarian_match_instances
    with patch.object(upstream,'hungarian_match_instances',wraps=original_matcher) as matcher:
        zero,_=coarse_aux_pose_loss(p,t,weights,aux_weight=0.,image_size=tuple(saved['record']['image_hw']))
        assert matcher.call_count==1
    torch.testing.assert_close(zero,baseline,atol=0,rtol=0)
    total,logs=coarse_aux_pose_loss(p,t,weights,aux_weight=2.5,image_size=tuple(saved['record']['image_hw']))
    torch.testing.assert_close(total,baseline+2.5*logs['loss_instance_coarse_body'],atol=0,rtol=0)
    return dict(real_dual_sample_base_loss_parity=True,shared_hungarian_calls=1,
        smoke_supervised_body_points=logs['coarse_body_points'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);smoke=loss_smoke()
    if args.smoke_only:print(json.dumps(smoke));return
    spec=json.loads((OUTPUT/'protocol.json').read_text());old=json.loads((PARENT/'protocol.json').read_text())
    result=json.loads((OUTPUT/'results.json').read_text());prior=json.loads((PARENT/'results.json').read_text())
    assert json.loads((OUTPUT/'status.json').read_text())['state']=='completed'
    with (OUTPUT/'running.lock').open() as file:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert build_protocol(spec['config'])==spec
    for key in ('records','evaluation','schedule','slots','checkpoint_sha256','encoder_sha256','manifest_sha256'):
        assert spec[key]==old[key],key
    assert spec['arms'][ARM]==old['arms']['combination_coverage']
    assert fingerprint(OUTPUT/'protocol.json')==result['protocol_sha256']
    assert fingerprint(PARENT/'results.json')==spec['parent_results_sha256']
    assert fingerprint(PARENT/'combination_coverage/final_heads.pt')==spec['parent_candidate_sha256']
    current=result['arms'][ARM];reference=prior['arms']['combination_coverage']
    for key in ('images_seen','patch_tokens_seen','initialization_hash'):assert current[key]==reference[key]
    history=json.loads((OUTPUT/ARM/'history.json').read_text())
    previous=json.loads((PARENT/'combination_coverage/history.json').read_text())
    auxiliary=json.loads((OUTPUT/ARM/'coarse_loss_history.json').read_text())
    assert len(history)==len(previous)==len(auxiliary)==800
    for row,before,aux in zip(history,previous,auxiliary):
        for key in ('scene','views','image_hw','images_seen','patch_tokens_seen','slot_kind','step','learning_rate'):
            assert row[key]==before[key],key
        assert aux['step']==row['step'] and aux['aux_weight']==2.5 and aux['supervised_body_points']>0
        assert abs(row['pose_loss']-aux['combined_pose_loss'])<1e-6
        assert abs(aux['combined_pose_loss']-aux['base_pose_loss']-2.5*aux['coarse_body_loss'])<1e-5
        assert all(math.isfinite(row[k]) for k in ('loss','pose_loss','depth_loss','gradient_norm'))
    parity=[]
    for a,b in zip(result['baseline']['records'],prior['baseline']['records']):
        assert all(a[k]==b[k] for k in ('group','scene','views'))
        parity.extend(abs(a['metrics'][k]-v) for k,v in b['metrics'].items() if v is not None)
    assert len(result['baseline']['records'])==88 and max(parity)<1e-6
    checkpoint=torch.load(OUTPUT/ARM/'final_heads.pt',weights_only=True,map_location='cpu')
    assert state_hash(checkpoint['model'])==checkpoint['trained_hash']!=checkpoint['initial_hash']
    assert checkpoint['protocol_sha256']==result['protocol_sha256']
    assert current['checkpoint_reload_verified'] and current['encoder_gradients_absent']
    report=dict(passed=True,**smoke,protocol_rebuilt_exactly=True,all_source_code_weights_verified=True,
        exact_slot_sampling_and_budget=True,baseline_metric_max_difference=max(parity),
        images=current['images_seen'],patch_tokens=current['patch_tokens_seen'],
        final_checkpoint_sha256=fingerprint(OUTPUT/ARM/'final_heads.pt'),
        first100_coarse_loss=sum(r['coarse_body_loss'] for r in auxiliary[:100])/100,
        last100_coarse_loss=sum(r['coarse_body_loss'] for r in auxiliary[-100:])/100)
    write_json(OUTPUT/'delivery_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
