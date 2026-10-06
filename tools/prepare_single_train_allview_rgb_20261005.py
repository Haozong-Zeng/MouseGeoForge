"""All rendered cameras of the same43 TRAIN single-mouse poses; labels stay separate."""
import collections,datetime as dt,fcntl,json,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from training.finetune_blendedmvs import _normalized_box_from_mask
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from audit_bottom_view_coverage import orientation_from_extrinsic
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'single_train_allview_rgb_inputs_v1'
def now():return dt.datetime.now(dt.timezone.utc)
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='single_train_allview_rgb_inputs_running',active_job=OUT.name,active_pid=os.getpid(),updated_utc=now().isoformat());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=now().isoformat(),**kw))
    original=prepare_rows();singles={(r['metadata']['scene'],r['metadata']['view']):r for r in original if r['metadata']['stage']=='fit' and r['metadata']['kind']=='single'}
    scenes={r['metadata']['scene']:r['record'] for r in singles.values()};assert len(singles)==86 and len(scenes)==43
    assert all(r['metadata']['source']=='moseq_sample10' for r in singles.values())
    nonfit={r['metadata']['scene'] for r in original if r['metadata']['stage']!='fit'};assert not set(scenes)&nonfit
    write_json(OUT/'protocol.json',dict(scope='Same43 existingfit poses only:21Cylinder+22Homecage, allrendered8/4cameras. No newframes/source/video/cal/VAL and noRealCapture21. IndependentV1 originalRGB percamera. Baselinepredictedboxes/logits notGT.',
        source_reference='Current sourcegroupmoseq_sample10 fromexistingfrozenmanifest. These exactscenes alreadyTRAIN; no needtoinventprovenanceforunseenframes.',scene_records=scenes,planned_images=256,
        labels='Originalmask-backedGTboxes, originalCore23validpoints fromloader; no sparsepoint boxfallback. GTmatchingonlyafterRGBforward tostoreknownnose4feet loss targets; missing/failedqualification excluded notnegative.',
        purpose='Data-view coverage next: existing head-only400 is control; same initialhead/LR/schedulecohorts/poseexposure, replace onlysinglemousecamera draw. No simultaneousLoRA ornewposeexpansion. Do not trainuntilold86viewprediction+targetparity verified.',
        script_sha256=fingerprint(Path(__file__))))
    status('loading_model');runtime=FrozenSingleViewCandidateRuntime();rows=[];parity=[];hashes={};stats=collections.Counter()
    for si,(scene,source_record) in enumerate(scenes.items()):
        if now()>=deadline:raise TimeoutError('Training deadline')
        meta=json.loads((ROOT/scene/'scene_meta.json').read_text());views=sorted(v['view_id'] for v in meta['views']);r=dict(source_record,views=views)
        assert r['role']=='train' and r['source_group']=='moseq_sample10'
        item=load_item(r);hashes.update(item['source_hashes']);b=item['batch'];t=instance_targets_from_batch(b);h,w=b['images'].shape[-2:]
        for vi,v in enumerate(views):
            with torch.no_grad():raw=runtime.from_images(b['images'][:,vi:vi+1].cuda())
            baseline={k:raw[k].cpu() for k in ['xy','boxes','objectness_logits']}
            d=dict(gt_xy=t['keypoint_xy'][0,vi],gt_boxes=t['boxes_cxcywh'][0,vi],point_valid=t['keypoint_valid'][0,vi],box_valid=t['box_valid'][0,vi]&t['mask_valid'][0,vi],instance_valid=t['instance_valid'][0,vi])
            for j in torch.where(d['box_valid']&d['instance_valid'])[0].tolist():
                box=torch.tensor(_normalized_box_from_mask(t['masks'][0,vi,j].numpy()));assert (box-d['gt_boxes'][j]).abs().max()<1e-6
            qs,ts=match_detection(dict(instance_boxes_cxcywh=baseline['boxes'][None],instance_objectness_logits=baseline['objectness_logits'][None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0]
            gt=torch.zeros(6,5,2);valid=torch.zeros(6,5,dtype=torch.bool)
            for q,j in zip(qs.tolist(),ts.tolist()):
                if baseline['objectness_logits'][q].sigmoid()<.5 or box_iou(baseline['boxes'][q:q+1],d['gt_boxes'][j:j+1])[0,0]<.5:continue
                gt[q]=d['gt_xy'][j,KP];valid[q]=d['point_valid'][j,KP]
            cam=next(c for c in meta['views'] if c['view_id']==v);elevation,orientation=orientation_from_extrinsic(cam['extrinsic'])
            m=dict(role='train',stage='fit',kind='single',source='moseq_sample10',scene=scene,view=v,physical_bottom=False,synthetic_orientation=orientation,expanded_single_index=len(rows))
            row=dict(metadata=m,record=r,baseline=baseline,pixels=torch.tensor([w,h]).float(),gt=gt,valid=valid)
            if (scene,v) in singles:
                old=singles[scene,v];delta={k:float((baseline[k]-old['baseline'][k]).abs().max()) for k in baseline};assert max(delta.values())<1e-5,(scene,v,delta)
                assert torch.equal(valid,old['valid']);assert (gt[valid]-old['gt'][valid]).abs().max()<1e-6
                parity.append(dict(scene=scene,view=v,delta=delta))
            rows.append(row);stats[(Path(scene).parts[2],orientation,'images')]+=1;stats[(Path(scene).parts[2],orientation,'known_points')]+=int(valid.sum());stats[(Path(scene).parts[2],orientation,'zero_known_images')]+=int(not valid.any())
            del raw
        status('extracting',scenes=si+1,total=43,images=len(rows));print('ALLVIEW_SINGLE',si+1,len(rows),flush=True)
    assert len(rows)==256 and len(parity)==86
    torch.save(rows,OUT/'rows.pt');write_json(OUT/'source_hashes.json',hashes);write_json(OUT/'manifest.json',dict(images=len(rows),scenes=len(scenes),old86_parity=parity,counts=[dict(dataset=k[0],orientation=k[1],metric=k[2],value=v) for k,v in sorted(stats.items())],runtime_hashes=runtime.hashes,rows_sha256=fingerprint(OUT/'rows.pt'),no_data_or_weights_changed=True))
    status('complete',images=len(rows));state.update(stage='single_train_allview_rgb_inputs_complete',active_job=None,active_pid=None,next='Inspect256images qualification/orientation andold86parity; freeze data-view control protocol before training, no LoRA/newposes simultaneous.',updated_utc=now().isoformat());state['completed'].append('single_train_allview256_rgb_parity86');write_json(CYCLE/'state.json',state)
    print('COMPLETE',json.dumps([dict(dataset=k[0],orientation=k[1],metric=k[2],value=v) for k,v in sorted(stats.items())]))
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=traceback.format_exc()))
        raise

