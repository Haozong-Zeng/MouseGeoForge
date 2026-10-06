"""Separate existing backward repeatability from zero-query-position gradients."""
import fcntl
import json
import warnings
from pathlib import Path
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.query_position import install_query_position,query_position_context
from training.instance_pose_losses import instance_pose_loss,InstancePoseLossWeights

OUT=ROOT/'output/query_position_gradient_probe_v1'
PARENT=ROOT/'output/cylinder_adaptive_temperature_v1'


def differences(a,b):
    assert a.keys()==b.keys()
    rows=[]
    for name in a:
        d=(a[name]-b[name]).double()
        rows.append(dict(name=name,max_abs=float(d.abs().max()),delta_l2=float(d.norm()),baseline_l2=float(a[name].double().norm())))
    return dict(max_abs=max(r['max_abs'] for r in rows),
                delta_l2=sum(r['delta_l2']**2 for r in rows)**.5,
                baseline_l2=sum(r['baseline_l2']**2 for r in rows)**.5,rows=rows)


def main():
    if OUT.exists():raise RuntimeError('Preserve existing probe')
    OUT.mkdir(parents=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=json.loads((PARENT/'protocol.json').read_text())
        fail=ROOT/'output/query_position_preflight_v1'
        assert json.loads((fail/'status.json').read_text())['state']=='failed'
        prereg=json.loads((fail/'protocol.json').read_text())
        for p,h in prereg['code_sha256'].items():assert fingerprint(ROOT/p)==h
        trainkey=prereg['train_probe_key'];r=spec['records'][trainkey]
        write_json(OUT/'protocol.json',dict(train_key=trainkey,sequence=['OFF_a','OFF_b','ON_a','ON_b'],
            criterion='Measure before deciding; do not change original zero-forward requirement or claim exact gradient equality. Full loss repeatability plus detection/segmentation-only isolation and deterministic warning diagnosis.',
            frozen_failed_protocol_sha256=fingerprint(fail/'protocol.json'),code_sha256=fingerprint(Path(__file__)),optimizer_steps=0))
        torch.set_num_threads(2);torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
        encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
        w=torch.load(encoder,weights_only=True,mmap=True)
        loaded=spatial.load_state_dict({k:v for k,v in w.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
        assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
        del w
        model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
        cp=PARENT/'adaptive_temperature/final_heads.pt';original=torch.load(cp,map_location='cpu',weights_only=True)['model'];restore_adaptive(model,original)
        install_query_position(model);batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
        targets={k:v.cuda() for k,v in instance_targets_from_batch(batch).items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
        model.train();weights=InstancePoseLossWeights(keypoint_3d=0.,reprojection=0.,local_heatmap=0.,keypoint_uncertainty=0.)
        base_predictions=None
        def evaluate(enabled,component):
            nonlocal base_predictions
            model.zero_grad(set_to_none=True)
            with query_position_context(model,enabled=enabled):
                pred=model.forward_cached(images,cached,start,coupling='off')
                saved={k:v.detach().cpu() for k,v in pred.items()}
                if base_predictions is None:base_predictions=saved
                else:
                    for k,v in base_predictions.items():torch.testing.assert_close(saved[k],v,atol=0,rtol=0)
                total,logs=instance_pose_loss(pred,targets,weights,image_size=images.shape[-2:])
                if component=='detection_segmentation':
                    total=sum(logs[k]*weight for k,weight in (('loss_instance_objectness',1),('loss_instance_box_l1',5),('loss_instance_box_giou',2),('loss_instance_mask_bce',2),('loss_instance_mask_dice',2)))
                total.backward()
            grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None and not n.startswith('instance_pose_head.query_position.')}
            g=model.instance_pose_head.query_position.offsets.grad
            new_norm=g.flatten(1).norm(dim=1).tolist() if g is not None else None
            return grads,float(total.detach()),new_norm
        results={}
        for component in ('full_pose','detection_segmentation'):
            rows=[evaluate(enabled,component) for enabled in (False,False,True,True)]
            assert len(set(x[1] for x in rows))==1
            results[component]=dict(loss=rows[0][1],off_repeat=differences(rows[0][0],rows[1][0]),
                on_repeat=differences(rows[2][0],rows[3][0]),off_on=differences(rows[0][0],rows[2][0]),
                new_gradients_per_layer=[x[2] for x in rows[2:]])
            del rows
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');torch.use_deterministic_algorithms(True,warn_only=True)
            try:evaluate(False,'full_pose')
            finally:torch.use_deterministic_algorithms(False)
            diagnostic_warnings=sorted({str(w.message) for w in caught})
        model.zero_grad(set_to_none=True);model.eval()
        assert state_hash({k:v for k,v in tensor_state(model).items() if not k.startswith('instance_pose_head.query_position.')})==state_hash(original)
        assert all(len(layer.multihead_attn._forward_pre_hooks)==0 for layer in model.instance_pose_head.instance_decoder.layers)
        assert torch.count_nonzero(model.instance_pose_head.query_position.offsets)==0
        report=dict(passed_probe=True,results=results,deterministic_warnings=diagnostic_warnings,
                    all_forwards_exact=True,encoder_forwards=1,head_forwards=9,optimizer_steps=0,
                    original_state_unchanged=True,production_promoted=False,
                    caveat='Diagnostic only; no relaxed preflight accepted and no training authorized by this report alone.')
        write_json(OUT/'report.json',report)
        print(json.dumps(dict(results={k:{n:{m:v[n][m] for m in ('max_abs','delta_l2','baseline_l2')} for n in ('off_repeat','on_repeat','off_on')} for k,v in results.items()},warnings=diagnostic_warnings),indent=2))


if __name__=='__main__':main()
