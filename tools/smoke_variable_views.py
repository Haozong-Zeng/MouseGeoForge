"""One real Blender pose, V=1/2/3/4 frozen GPU forwards; no model training."""
import argparse
import json
import numpy as np

from diagnose_occlusion_pose import ROOT, torch, infer, fingerprint, write_json, load_batch
from training.finetune_blendedmvs import load_scene, median_camera_baseline
from mousegeoforge.view_sampling import subset_record, view_cache_key
from vggt_omega.utils.robust_pose_geometry import triangulate_robust


def variable_batch(record):
    return load_batch(record,num_views=len(record['views']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    run = ROOT/'output/pose_diverse_core23_768_v1'
    destination = ROOT/'output/variable_view_pose_v1/smoke'
    manifest = json.loads((run/'manifest.json').read_text())
    record = next(r for r in manifest['records'] if r['split']=='train' and r['expected_instances']==2)
    scene = load_scene(ROOT/record['scene']); available = sorted(scene.frames)
    if len(available)<4: raise ValueError('Need four real cameras for this smoke')
    scene_scale = median_camera_baseline(np.stack([scene.frames[v].extrinsic for v in available]))
    encoder_hash = fingerprint(ROOT/'vggt-omega/vggt_omega_1b_512.pt')
    records = []
    for i in range(4):
        r = subset_record(record,available,sample_index=i,scene_scale=scene_scale,max_views=4)
        r['split'] = 'train_source_functionality_smoke'
        rgb_revision = {str(v):fingerprint(scene.frames[v].image_path) for v in r['views']}
        r['encoder_cache_identity'] = view_cache_key(scene=r['scene'],view_ids=r['views'],
            rgb_revision=rgb_revision,encoder_sha256=encoder_hash,
            preprocessing=dict(image_hw=r['image_hw'],resize='PIL_BICUBIC',rgb='original'))
        records.append(r)
    protocol = dict(records=records,head_sha256=fingerprint(run/'seed17_off/final_heads.pt'),
                    model_updated=False,scope='one training-source pose; functionality only, not accuracy validation')
    path = destination/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Smoke protocol changed')
    write_json(path,protocol)
    if not args.report_only: infer(run,destination,records,batch_loader=variable_batch)
    rows = []; depths_by_camera = {}; xyz = None
    for i,r in enumerate(records):
        item = torch.load(destination/'predictions'/f'{i:03d}.pt',weights_only=True)
        if item['record'] != r: raise ValueError('Cache record mismatch')
        p,t = item['predictions'],item['targets']; v = len(r['views'])
        batch = variable_batch(r)
        if p['instance_keypoint_xy'].shape[:2] != (1,v): raise AssertionError('Wrong predicted view count')
        if not all(torch.isfinite(x).all() for x in p.values()): raise AssertionError('Nonfinite prediction')
        if abs(float(batch['scale'][0])-scene_scale)>1e-3: raise AssertionError('Scale changes with V')
        if xyz is None: xyz = t['keypoint_world_xyz']
        else: torch.testing.assert_close(xyz,t['keypoint_world_xyz'])
        for j,camera in enumerate(r['diagnostic_view_order']):
            if camera in depths_by_camera: torch.testing.assert_close(depths_by_camera[camera],batch['depths'][0,j])
            else: depths_by_camera[camera] = batch['depths'][0,j].clone()
        if v==1:
            result = triangulate_robust(p['instance_keypoint_xy'][0,:,0,0],(t['intrinsics'][0]@t['extrinsics'][0,:,:3]))
            if result.supported or not torch.isnan(result.xyz).all(): raise AssertionError('Single-view 3D fabricated')
        rows.append(dict(views=v,cameras=r['views'],scale=float(batch['scale'][0]),
            predicted_xy_shape=list(p['instance_keypoint_xy'].shape),
            labeled_2d_points=int(t['keypoint_valid'].sum()),all_predictions_finite=True,
            diagnostic_2d_metrics=item['baseline_2d']))
    write_json(destination/'report.json',dict(rows=rows,scale_constant_across_subsets=True,
        native_depth_and_world_label_consistency=True,weights_updated=False,
        warning='Functionality smoke only. Existing fixed-two-view checkpoint is not variable-view adapted.'))
    print(json.dumps([{k:v for k,v in row.items() if k!='diagnostic_2d_metrics'} for row in rows],indent=2),flush=True)


if __name__=='__main__': main()
