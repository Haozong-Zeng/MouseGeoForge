"""Export both independently paired candidates and audit opt-in inference replay."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,write_json,fingerprint,tensor_state,state_hash
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.mask_box_candidate import MaskBoxCandidate,tensor_mapping_sha256

OUT=ROOT/'output/mask_box_candidates_v1'
SOURCES={s:ROOT/f'output/mask_reference_box_{s}_v1' for s in ('seed17','seed29')}
BASES={'seed17':ROOT/'output/cylinder_adaptive_temperature_v1/adaptive_temperature/final_heads.pt',
       'seed29':ROOT/'output/cylinder_temperature_seed29_v1/temperature_on/final_heads.pt'}


def run():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing export/audit; never overwrite')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            specs={s:json.loads((p/'protocol.json').read_text()) for s,p in SOURCES.items()}
            for s,p in SOURCES.items():
                assert json.loads((p/'verification.json').read_text())['passed']
                report=json.loads((p/'acceptance.json').read_text());assert report['practical_criterion_passed']
                assert all(report['gates'][a]['passed'] for a in ('reference_on','reference_on_corner_stress'))
                for path,h in specs[s]['code_sha256'].items():assert fingerprint(ROOT/path)==h
            spec=specs['seed17'];assert specs['seed29']['records']==spec['records'] and specs['seed29']['evaluation']==spec['evaluation']
            code={p:fingerprint(ROOT/p) for p in ('mousegeoforge/mask_box_candidate.py','mousegeoforge/mask_reference_box.py','mousegeoforge/mask_box_refiner.py','mousegeoforge/mask_shape_audit.py','tools/export_audit_mask_box_candidates.py')}
            inputs={str(p):fingerprint(p) for folder in SOURCES.values() for p in [folder/'protocol.json',folder/'acceptance.json',folder/'verification.json',folder/'reference_on_final.pt']}
            inputs.update({str(p):fingerprint(p) for p in BASES.values()})
            sources=spec['source_sha256'];assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            write_json(OUT/'protocol.json',dict(code_sha256=code,input_sha256=inputs,source_sha256=sources,
                scope='Export both800-step reference-ON branches only, no optimizer/base weights. Strict in-memory base+encoder pairing, full88 replay per seed, zero training. Default disabled, production not promoted.'))
            torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';encoder_file_hash=fingerprint(encoder)
            original_spec=json.loads((ROOT/'output/cylinder_adaptive_temperature_v1/protocol.json').read_text());assert encoder_file_hash==original_spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            encoder_state_hash=tensor_mapping_sha256({k:v for k,v in model.state_dict().items() if k.startswith('aggregator.')})
            flags={n:p.requires_grad for n,p in model.named_parameters()};modes={n:m.training for n,m in model.named_modules()}
            manifests={};latencies=[];forward_count=0
            for seed,folder in SOURCES.items():
                if time.monotonic()-began>1200 or time.time()>1789148363:raise RuntimeError('Audit deadline')
                original=torch.load(BASES[seed],weights_only=True,map_location='cpu')['model'];restore_adaptive(model,original)
                branch=torch.load(folder/'reference_on_final.pt',weights_only=True,map_location='cpu');assert branch['step']==800
                candidate_dir=OUT/seed;candidate_dir.mkdir(exist_ok=True)
                torch.save(branch['model'],candidate_dir/'refiner.pt')
                manifest=dict(schema_version=1,seed=seed,default_enabled=False,production_promoted=False,weights='refiner.pt',
                    weights_sha256=fingerprint(candidate_dir/'refiner.pt'),refiner_state_sha256=state_hash(branch['model']),
                    base_checkpoint=str(BASES[seed].relative_to(ROOT)),base_checkpoint_sha256=fingerprint(BASES[seed]),base_state_sha256=state_hash(original),
                    encoder_checkpoint=str(encoder.relative_to(ROOT)),encoder_checkpoint_sha256=encoder_file_hash,encoder_state_sha256=encoder_state_hash,
                    source_sha256=code,training_protocol_sha256=fingerprint(folder/'protocol.json'),training_verification_sha256=fingerprint(folder/'verification.json'),
                    policy='Only normalized cxcywh boxes may change. FP32 CPU postprocessing, actual image HW, no GT/identity/role input. Candidate for the paired temperature model only.',
                    limitations=['Same reused synthetic development, not real Cylinder validation.','Does not resolve missing/duplicated instances or improve keypoint coordinates.','Connected-component CPU prototype; context validates hashes once, not a real-time guarantee.'])
                write_json(candidate_dir/'manifest.json',manifest);manifests[seed]=manifest
                candidate=MaskBoxCandidate.load(candidate_dir,base_checkpoint=BASES[seed],encoder_checkpoint=encoder,repository_root=ROOT)
                other=BASES['seed29' if seed=='seed17' else 'seed17']
                try:MaskBoxCandidate.load(candidate_dir,base_checkpoint=other,encoder_checkpoint=encoder,repository_root=ROOT)
                except ValueError:pass
                else:raise AssertionError('Wrong base checkpoint was accepted')
                hooks_before=len(model.instance_pose_head.instance_decoder._forward_hooks)
                with candidate.capture(model):
                    assert len(model.instance_pose_head.instance_decoder._forward_hooks)==hooks_before
                    dummy={'sentinel':object()};assert candidate(dummy) is dummy
                with candidate.capture(model,enabled=True):
                    for idx,e in enumerate(spec['evaluation']):
                        r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                        with torch.no_grad():pred=model.forward_cached(images,cached,start,coupling='off')
                        old=torch.load(folder/'regressed'/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
                        for k,v in old['predictions'].items():torch.testing.assert_close(pred[k].cpu(),v,atol=0,rtol=0)
                        assert candidate(pred) is pred
                        t=time.perf_counter();updated=candidate(pred,r['image_hw'],enabled=True);elapsed_ms=(time.perf_counter()-t)*1000
                        expected=torch.load(folder/'reference_on'/f'prediction_{idx:03d}.pt',map_location='cpu',weights_only=True)
                        for k,v in expected['predictions'].items():torch.testing.assert_close(updated[k].cpu(),v,atol=0,rtol=0)
                        assert all(updated[k] is v for k,v in pred.items() if k!='instance_boxes_cxcywh')
                        latencies.append(dict(seed=seed,index=idx,views=len(r['views']),image_hw=r['image_hw'],postprocess_ms=elapsed_ms))
                        forward_count+=1
                        del pred,updated,cached,images,batch
                        if (idx+1)%16==0:status('replaying',seed=seed,records=idx+1,total=88);print(seed,idx+1,flush=True)
                assert len(model.instance_pose_head.instance_decoder._forward_hooks)==hooks_before
                assert state_hash(tensor_state(model))==state_hash(original)
                assert {n:p.requires_grad for n,p in model.named_parameters()}==flags
                assert {n:m.training for n,m in model.named_modules()}==modes
                assert all(p.grad is None for p in model.parameters())
            assert tensor_mapping_sha256({k:v for k,v in model.state_dict().items() if k.startswith('aggregator.')})==encoder_state_hash
            from pathlib import Path
            assert all(fingerprint(Path(p))==h for p,h in inputs.items())
            assert all(fingerprint(ROOT/p)==h for p,h in code.items())
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            timing={seed:{name:float(fn([r['postprocess_ms'] for r in latencies if r['seed']==seed])) for name,fn in [('median_ms',np.median),('max_ms',np.max)]} for seed in SOURCES}
            write_json(OUT/'latencies.json',latencies)
            write_json(OUT/'verification.json',dict(passed=True,optimizer_steps=0,records_per_seed=88,encoder_forwards=forward_count,head_forwards=forward_count,
                original_prediction_max_difference=0,exported_refinement_max_difference=0,other_output_objects_unchanged=True,default_off_identity=True,
                incorrect_base_rejected=True,actual_base_and_encoder_checked=True,hooks_removed=True,base_weights_flags_modes_unchanged=True,
                candidate_manifest_sha256={s:fingerprint(OUT/s/'manifest.json') for s in SOURCES},latency=timing,
                latency_scope='Postprocess call only including transfers, excludes encoder/base forward and context hash validation; single run, not real-time benchmark.',production_promoted=False,elapsed_s=time.monotonic()-began))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched; inspect existing state')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:run()
