"""Two frozen checkpoints x original88 records, regressed vs predicted-mask boxes."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,tensor_state,state_hash,fingerprint,write_json,instance_targets_from_batch
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch,physical_metrics,scene_digest
from train_variable_view_replay import mean_metrics
from mousegeoforge.adaptive_temperature import restore_adaptive
from mousegeoforge.mask_box_decode import mask_boxes_from_logits
from mousegeoforge.pilot_metrics import box_iou

OUT=ROOT/'output/mask_box_decode_v1'
SOURCES={
    'seed17':ROOT/'output/cylinder_adaptive_temperature_v1',
    'seed29':ROOT/'output/cylinder_temperature_seed29_v1',
}
ARMS={'seed17':'adaptive_temperature','seed29':'temperature_on'}


def run():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as global_lock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing audit; never silently restart')
        def status(state,**kw):write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),elapsed_s=time.monotonic()-began,**kw))
        status('preparing')
        try:
            specs={k:json.loads((p/'protocol.json').read_text()) for k,p in SOURCES.items()}
            spec=specs['seed17']
            for tag,folder in SOURCES.items():
                assert json.loads((folder/'delivery_verification.json').read_text())['passed']
                assert json.loads((folder/'status.json').read_text())['state']=='completed'
                for key in ('records','evaluation','encoder_sha256','source_sha256'):assert specs[tag][key]==spec[key]
                for p,h in specs[tag]['code_sha256'].items():assert fingerprint(ROOT/p)==h
            sources=spec['source_sha256'];assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            folders={k:p/ARMS[k] for k,p in SOURCES.items()}
            checkpoints={k:p/'final_heads.pt' for k,p in folders.items()}
            saved_hash={str(p/f'prediction_{i:03d}.pt'):fingerprint(p/f'prediction_{i:03d}.pt') for p in folders.values() for i in range(88)}
            code={p:fingerprint(ROOT/p) for p in ('tools/audit_mask_box_decode.py','mousegeoforge/mask_box_decode.py','mousegeoforge/pilot_metrics.py','tools/train_cylinder_view_balance.py','tools/run_depth_pose_ablation.py')}
            protocol=dict(config=spec['config'],records=spec['records'],evaluation=spec['evaluation'],source_sha256=sources,
                checkpoint_sha256={str(p):fingerprint(p) for p in checkpoints.values()},saved_prediction_sha256=saved_hash,
                code_sha256=code,encoder_sha256=spec['encoder_sha256'],
                parent_protocol_sha256={str(p/'protocol.json'):fingerprint(p/'protocol.json') for p in SOURCES.values()},
                scope='Two frozen temperature-ON adaptation-order seeds, same88 reused synthetic development inputs; no training or GT-conditioned decoding.',
                decoding='Bilinear upsample logits to actual input HW, >0, all foreground exclusive pixel bbox; empty mask falls back to original box. No denoising, tail exclusion, threshold sweep or component filtering.',
                evaluation_policy='Keep GT, query coordinates, objectness, depth and original17 gates. Rerun original box-only evaluation assignment. Also report fixed old-assignment box IoU. PCK improvement is not coordinate improvement.')
            write_json(OUT/'protocol.json',protocol)
            torch.set_num_threads(2);torch.manual_seed(17)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
            spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
            weights=torch.load(encoder,weights_only=True,mmap=True)
            loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            del weights
            model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
            states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in checkpoints.items()}
            results={tag:{arm:[] for arm in ('regressed','mask')} for tag in folders};box_rows=[];parity=[];changed_assignments=[];query_masks=[]
            for idx,e in enumerate(spec['evaluation']):
                if time.monotonic()-began>1200:raise RuntimeError('20minute audit safety limit')
                r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));target=instance_targets_from_batch(batch)
                images=batch['images'].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
                for tag,state in states.items():
                    restore_adaptive(model,state)
                    with torch.no_grad():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start,coupling='off').items()}
                    saved=torch.load(folders[tag]/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
                    assert r==saved['record']
                    for key,value in saved['predictions'].items():torch.testing.assert_close(pred[key],value,atol=0,rtol=0)
                    for key in ('boxes_cxcywh','keypoint_xy','keypoint_valid'):torch.testing.assert_close(target[key],saved['targets'][key],atol=0,rtol=0)
                    off,flag=mask_boxes_from_logits(None,pred['instance_boxes_cxcywh'],r['image_hw'])
                    assert off is pred['instance_boxes_cxcywh'] and not flag.any()
                    decoded,used=mask_boxes_from_logits(pred['instance_mask_logits'],off,r['image_hw'],enabled=True)
                    alt=dict(pred,instance_boxes_cxcywh=decoded)
                    assert all(alt[k] is v for k,v in pred.items() if k!='instance_boxes_cxcywh')
                    evaluated={}
                    for arm,p in (('regressed',pred),('mask',alt)):
                        metrics,assignment=physical_metrics(p,batch,target);evaluated[arm]=(metrics,assignment)
                        if arm=='regressed':
                            assert assignment==saved['oracle_assignment']
                            for key,value in saved['metrics'].items():
                                if value is None:assert metrics[key] is None
                                else:parity.append(abs(metrics[key]-value))
                        folder=OUT/tag/arm;folder.mkdir(parents=True,exist_ok=True)
                        torch.save(dict(record=r,group=e['group'],oracle_assignment=assignment,metrics=metrics,
                            predictions={k:p[k] for k in ('instance_keypoint_xy','instance_boxes_cxcywh','instance_objectness_logits')},
                            targets={k:v for k,v in target.items() if k!='masks'}),folder/f'prediction_{idx:03d}.pt')
                        results[tag][arm].append(dict(scene=r['scene'],views=r['views'],group=e['group'],metrics=metrics))
                    if evaluated['mask'][1]!=evaluated['regressed'][1]:changed_assignments.append(dict(seed=tag,index=idx,scene=r['scene'],group=e['group'],old=evaluated['regressed'][1],new=evaluated['mask'][1]))
                    for v,(qs,ms) in enumerate(saved['oracle_assignment']):
                        for q,m in zip(qs,ms):
                            gt=target['boxes_cxcywh'][0,v,m:m+1];old_iou=float(box_iou(gt,off[0,v,q:q+1])[0,0]);new_iou=float(box_iou(gt,decoded[0,v,q:q+1])[0,0])
                            box_rows.append(dict(seed=tag,index=idx,scene=r['scene'],group=e['group'],view=v,role=r['physical_roles'][v],query=q,instance=m,
                                old_iou=old_iou,mask_iou=new_iou,mask_used=bool(used[0,v,q]),score=float(pred['instance_objectness_logits'][0,v,q].sigmoid())))
                    query_masks.append(dict(seed=tag,index=idx,queries=used.numel(),nonempty=int(used.sum()),high_score_empty=int(((pred['instance_objectness_logits'].sigmoid()>=.5)&~used).sum())))
                    del pred,alt,decoded,saved
                del images,cached,batch,target
                if (idx+1)%8==0:status('evaluating',records=idx+1,total=88);print('Completed',idx+1,'/88 x2 checkpoints',flush=True)
            assert max(parity)<1e-6,max(parity)
            for tag,state in states.items():restore_adaptive(model,state);assert state_hash(tensor_state(model))==state_hash(state)
            for p,h in code.items():assert fingerprint(ROOT/p)==h
            from pathlib import Path
            for mapping in (protocol['checkpoint_sha256'],saved_hash,protocol['parent_protocol_sha256']):
                assert all(fingerprint(Path(p))==h for p,h in mapping.items())
            assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
            packaged={tag:{arm:dict(records=rows,summary=mean_metrics(rows)) for arm,rows in arms.items()} for tag,arms in results.items()}
            write_json(OUT/'results.json',dict(seeds=packaged,config=spec['config'],protocol_sha256=fingerprint(OUT/'protocol.json'),production_promoted=False))
            write_json(OUT/'box_rows.json',box_rows);write_json(OUT/'assignment_changes.json',changed_assignments);write_json(OUT/'mask_availability.json',query_masks)
            write_json(OUT/'verification.json',dict(passed=True,original_predictions_difference=0,original_metric_max_difference=max(parity),only_boxes_changed=True,
                weights_source_code_unchanged=True,encoder_forwards=88,head_forwards=176,optimizer_steps=0,records_per_checkpoint=88,elapsed_s=time.monotonic()-began))
            status('completed');print('COMPLETED',flush=True)
        except BaseException as error:status('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        if (OUT/'status.json').exists():raise RuntimeError('Already launched; inspect existing run')
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'run.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))));return
    run()


if __name__=='__main__':main()
