"""Expand existing single-animal synthetic TRAIN contexts, labels untouched."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.candidate_inputs import local_candidate_inputs
from mousegeoforge.multiview_camera_geometry import projection_matrix,project_points
from training.instance_pose_losses import instance_targets_from_batch
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime,KP,OLD
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';OUT=CYCLE/'synthetic_full_candidate_context_v1'


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();(OUT/'scenes').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='expanding_synthetic_train_contexts',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');started=time.monotonic();runtime=FrozenSingleViewCandidateRuntime()
    inventorypath=CYCLE/'candidate_repeats_review_v1/synthetic_train_view_inventory.json';inventory=json.loads(inventorypath.read_text());sourcepath=SOURCE/'train.pt'
    sm=json.loads((SOURCE/'manifest.json').read_text());assert fingerprint(sourcepath)==next(r['sha256'] for r in sm['parts'] if r['role']=='train');data=torch.load(sourcepath,map_location='cpu',weights_only=False)
    geompath=CYCLE/'camera_geometry_audit_v2/train_cameras.pt';groups={g['scene']:g for g in torch.load(geompath,map_location='cpu',weights_only=False) if g['kind']=='synthetic'}
    native={r['image_index']:r for r in json.loads((OLD/'single_view_train_context_v1/manifest.json').read_text())['records']}
    mappath=CYCLE/'singleview_probability_diagnostic_v1/train_probabilities.npz'
    with np.load(mappath) as a:probability=a['probability'];maprow={int(i):j for j,i in enumerate(a['row_indices'])}
    assert len(inventory)==69 and set(groups)=={r['scene'] for r in inventory}
    write_json(OUT/'protocol.json',dict(source_sha256=fingerprint(sourcepath),inventory_sha256=fingerprint(inventorypath),geometry_sha256=fingerprint(geompath),probabilities_sha256=fingerprint(mappath),model_sha256=runtime.hashes,
        scope='Only existing69syntheticTRAINframes;308extraimages atsame768-basedshape, no rendering/newactions/VAL. Original153views reused from immutableV1 caches, all138queries represented in decoder. Shared normalization per scene retained. Candidate0 originalmean +5NMSlocalpeaks; labels separate, never used for ROI/peaks/features. No production or annotations modified.'))
    records=[];fresh=reused=0;maximumprojection=maximumxy=0.;replays=[]
    for si,r in enumerate(inventory):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        assert r['role']=='train' and not r['camera_missing'];g=groups[r['scene']];old={int(data['metadata'][i]['view']):i for i in g['row_indices']};meta=data['metadata'][g['row_indices'][0]];record=dict(meta['record'],views=r['available_views'],normalization_scale=g['scale'])
        assert record['role']=='train' and record['kind']!='real'
        item=load_item(record);batch=item['batch'];targets=instance_targets_from_batch(batch);views=r['available_views'];h,w=record['image_hw'];pixels=np.array([w,h],np.float32)
        assert int(batch['actual_instance_count'][0])==1 and batch['instance_supervision_available'].all()
        assert batch['camera_available'].all() and batch['depth_available'].all()
        for path,sha in g['source_hashes'].items():assert item['source_hashes'][path]==sha
        k=batch['intrinsics'][0].numpy();e=batch['extrinsics'][0].numpy();projections=np.stack([projection_matrix(a,b) for a,b in zip(k,e)])
        values=[];source_rows=[]
        for vi,physical in enumerate(views):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            if physical in old:
                i=old[physical];source_rows.append(i)
                torch.testing.assert_close(targets['keypoint_xy'][0,vi],data['gt_xy'][i],rtol=0,atol=0,equal_nan=True);torch.testing.assert_close(targets['keypoint_valid'][0,vi],data['point_valid'][i],rtol=0,atol=0)
                torch.testing.assert_close(targets['boxes_cxcywh'][0,vi],data['gt_boxes'][i],rtol=0,atol=0)
                rec=native[data['image_indices'][i]];assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)
                maps=F.interpolate(raw['memory'][None].cuda().half().float(),(48,48),mode='bilinear',align_corners=False)
                value=dict(xy=data['xy'][i],boxes=data['boxes'][i],objectness_logits=data['objectness_logits'][i],tokens=data['tokens'][i,:,KP],probability=torch.from_numpy(probability[maprow[i]]))
                if si in [0,22,66] and physical==views[0]:
                    replay=runtime.from_images(batch['images'][:,vi:vi+1]);delta={key:float((replay[key].cpu()-value[key]).abs().max()) for key in ['xy','boxes','objectness_logits','tokens','probability']};assert max(delta.values())<1e-4,delta;replays.append(dict(scene=r['scene'],view=physical,delta=delta));del replay
                reused+=1
            else:
                source_rows.append(None);value=runtime.from_images(batch['images'][:,vi:vi+1]);maps=value.pop('maps');value={key:tensor.cpu() for key,tensor in value.items()};fresh+=1
            aux=local_candidate_inputs(value['probability'].numpy(),value['xy'][:,KP].numpy(),maps)
            value.update(aux);values.append(value);del maps
            xyz=batch['instance_keypoint_world_xyz'][0,0].numpy();valid=targets['keypoint_valid'][0,vi,0].numpy()&batch['instance_world_valid'][0,0].numpy();uv,z=project_points(projections[vi],xyz);valid&=z>0
            err=np.linalg.norm(uv-targets['keypoint_xy'][0,vi,0].numpy()*pixels,axis=-1)[valid]
            if len(err):maximumprojection=max(maximumprojection,float(err.max()))
            assert maximumprojection<.002,(r['scene'],physical,maximumprojection)
            status('extracting',scenes_done=si,fresh=fresh,reused=reused,total_fresh=308,elapsed_seconds=time.monotonic()-started)
        predictions={key:torch.stack([v[key] for v in values]) for key in values[0]}
        target_fields={key:targets[key][0] for key in ['keypoint_xy','keypoint_valid','keypoint_visible','keypoint_visibility_valid','boxes_cxcywh','box_valid','instance_valid','instance_ids','keypoint_world_xyz','world_valid','surface_valid']}
        path=OUT/'scenes'/f'{si:03d}.pt';torch.save(dict(scene=r['scene'],role='train',record=record,physical_views=views,source_rows=source_rows,pixels=pixels,
            predictions=predictions,targets=target_fields,cameras=dict(intrinsics=k,extrinsics=e,projections=projections,normalization_scale=g['scale']),
            source_hashes=item['source_hashes'],annotation_availability=batch['instance_supervision_available'][0],actual_instance_count=batch['actual_instance_count'][0]),path)
        records.append(dict(scene=r['scene'],role='train',path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),views=views,source_rows=source_rows,bytes=path.stat().st_size,source_hashes=item['source_hashes']))
        write_json(OUT/'partial_manifest.json',dict(records=records,fresh=fresh,reused=reused));print(si+1,'/69',fresh,'new',reused,'reused',flush=True)
    assert fresh==308 and reused==153
    write_json(OUT/'manifest.json',dict(records=records,fresh=fresh,reused=reused,images=fresh+reused,scenes=len(records),max_synthetic_projection_px=maximumprojection,old_view_target_exact=True,replays=replays,
        protocol_sha256=fingerprint(OUT/'protocol.json'),script_sha256=fingerprint(Path(__file__)),runtime_sha256=fingerprint(ROOT/'tools/singleview_candidate_runtime.py'),candidate_input_sha256=fingerprint(ROOT/'mousegeoforge/candidate_inputs.py'),bytes=sum(r['bytes'] for r in records),elapsed_seconds=time.monotonic()-started))
    status('complete',images=fresh+reused,fresh=fresh,reused=reused);state.update(stage='synthetic_full_candidate_context_ready',active_job=None,active_pid=None,next='Audit synthetic_full_candidate_context_v1 predictions and camera/label parity; then evaluate fixed baseline/unary/geometry in fullsyntheticTRAIN before training learnedcandidate route. No VAL data or original annotations changed.');state['completed'].append('synthetic_full_train_candidate_contexts');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
