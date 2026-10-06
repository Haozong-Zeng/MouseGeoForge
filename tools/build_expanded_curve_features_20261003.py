"""Append only newly verified TRAIN observations; freeze existing150input tensors/labels."""
import collections,datetime as dt,fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega'),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image
from vggt_omega.models import VGGTOmega
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel
from training.instance_pose_losses import instance_targets_from_batch,hungarian_match_instances
from training.finetune_blendedmvs import load_scene
from run_depth_pose_ablation import fingerprint,write_json,move_batch
from run_surface_curve_pilot_20261003 import load_item
SOURCE=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_features_v1'
GT=ROOT/'output/surface_curve_cycle_20261003/diverse_curve_gt_v2'
OUT=ROOT/'output/surface_curve_cycle_20261003/expanded_curve_features_v1'

def main():
    assert not OUT.exists() and json.loads((GT/'qa.json').read_text())['numeric_passed']
    lock=(ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    deadline=dt.datetime(2026,10,4,8,5,16,tzinfo=dt.timezone.utc)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=__import__('os').getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading_frozen_encoder');saved=torch.load(SOURCE/'frozen_inputs.pt',map_location='cpu',weights_only=False);original={k:v.clone() for k,v in saved['tensors'].items()};oldmeta=saved['metadata'];meta=list(oldmeta)
    oldscenes={r['scene'] for r in oldmeta};manifest=json.loads((GT/'manifest.json').read_text())
    eligible=set(json.loads((GT/'qa.json').read_text())['effective_accepted_scenes'])
    rows=[r for r in manifest['records'] if r['accepted_for_training'] and r['scene'] in eligible and r['scene'] not in oldscenes]
    assert all(r['role']=='train' for r in rows)
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    checkpoint=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    chosen={k:v for k,v in checkpoint.items() if k.startswith(('aggregator.','dense_head.'))};result=spatial.load_state_dict(chosen,strict=False)
    assert not result.unexpected_keys and all(k.startswith('instance_pose_head.') for k in result.missing_keys);del checkpoint,chosen
    model=SurfaceCurveDepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    initial=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot/initial_heads.pt'
    assert fingerprint(initial)==json.loads((SOURCE/'protocol.json').read_text())['source_initial_sha256']
    result=model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True),strict=False)
    assert not result.unexpected_keys and all(k.startswith('aggregator.') for k in result.missing_keys);model.requires_grad_(False)
    new=collections.defaultdict(list);provenance=[]
    for n,row in enumerate(rows):
        assert dt.datetime.now(dt.timezone.utc)<deadline
        scene=load_scene(ROOT/row['scene']);views=[0,2];assert set(views)<=set(scene.frames)
        assert fingerprint(ROOT/row['scene']/'keypoints.npz')==row['source_annotation_sha256']
        with Image.open(scene.frames[0].image_path) as im:w0,h0=im.size
        hw=[max(16,round(h0*768/max(w0,h0)/16)*16),max(16,round(w0*768/max(w0,h0)/16)*16)]
        r=dict(scene=row['scene'],kind='synthetic',role='train',dataset=row['dataset'],source_group=row['source_group'],views=views,image_hw=hw,key=f'x{n:04d}')
        item=load_item(r);inst=[];pose=[]
        ah=model.instance_pose_head.instance_decoder.register_forward_hook(lambda m,a,o:inst.append(o.detach()))
        bh=model.instance_pose_head.keypoint_decoder.register_forward_hook(lambda m,a,o:pose.append(o.detach()))
        try:
            with torch.no_grad():pred=model(item['batch']['images'].cuda(),coupling='off')
        finally:ah.remove();bh.remove()
        pt=pose[0].reshape(2,6,23,256);target=instance_targets_from_batch(move_batch(item['batch'],'cuda'));mt=dict(target);mt['keypoint_valid']=torch.zeros_like(mt['keypoint_valid']);matches=hungarian_match_instances(pred,mt)[0]
        with np.load(ROOT/row['sidecar']) as a:
            indices=[a['view_ids'].tolist().index(i) for i in views];cur=a['uv'][indices].copy();assert a['position_valid'][indices].all()
        for v,vid in enumerate(views):
            with Image.open(scene.frames[vid].image_path) as im:sw,sh=im.size
            cur[v]/=[sw,sh]
        h,w=hw
        for v,(qs,ms) in enumerate(matches):
            for q,m in zip(qs.tolist(),ms.tolist()):
                tok=inst[0][v,q];new['det'].append(tok.repeat(4).cpu());new['pose'].append(pt[v,q,[4,5,6,7]].flatten().cpu());new['boxes'].append(pred['instance_boxes_cxcywh'][0,v,q].cpu())
                new['gt'].append(torch.tensor(cur[v,m],dtype=torch.float32));new['pixels'].append(torch.tensor([w,h],dtype=torch.float32));new['base_logits'].append(model.surface_curve_head.net(tok).reshape(6,2).cpu())
                meta.append(dict(key=r['key'],scene=r['scene'],role='train',kind='synthetic',dataset=r['dataset'],source_group=r['source_group'],view=views[v],query=q,instance=m,
                    curve_sha256=fingerprint(ROOT/row['sidecar']),source_curve=row['sidecar'],source_image=scene.frames[views[v]].image_path.relative_to(ROOT).as_posix(),image_hw=hw))
        provenance.append(dict(key=r['key'],record=r,source_hashes=item['source_hashes'],curve_sha256=fingerprint(ROOT/row['sidecar'])))
        status('extracting_new_train_features',completed=n+1,total=len(rows));print('EXPAND',n+1,len(rows),row['scene'],flush=True)
    tensors={k:torch.cat([original[k],torch.stack(new[k])],dim=0) for k in original}
    for k in original:assert torch.equal(tensors[k][:len(oldmeta)],original[k]),'Original train/validation tensors changed'
    assert meta[:len(oldmeta)]==oldmeta
    torch.save(dict(tensors=tensors,metadata=meta),OUT/'frozen_inputs.pt')
    info=dict(source_input_sha256=fingerprint(SOURCE/'frozen_inputs.pt'),source_initial_sha256=fingerprint(initial),source_encoder_sha256=fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt'),
        original_observations=len(oldmeta),new_train_observations=len(meta)-len(oldmeta),train_observations=sum(r['role']=='train' for r in meta),validation_observations=sum(r['role']=='validation' for r in meta),
        train_scenes=len({m['scene'] for m in meta if m['role']=='train'}),validation_scenes=len({m['scene'] for m in meta if m['role']=='validation'}),
        original150tensors_bitwise_unchanged=True,all_validation_unchanged=True,source_records=provenance,
        storage_policy='Only compact frozen instance/pose features saved; new full encoder caches not duplicated',status='ready_for_controlled_training')
    write_json(OUT/'manifest.json',info);status('complete');print(json.dumps({k:v for k,v in info.items() if k!='source_records'}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()));raise
