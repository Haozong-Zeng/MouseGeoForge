"""Frozen localization decomposition; exact 88-record development protocol."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from run_depth_pose_ablation import (ROOT,torch,VGGTOmega,DepthPoseModel,restore,
    fingerprint,write_json,instance_targets_from_batch)
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.localization_audit import refinement_statistics,summarize

OUTPUT=ROOT/'output/cylinder_localization_audit_v1'
RUN=ROOT/'output/cylinder_combinations_v2'
PARENT=ROOT/'output/cylinder_view_balance_v1'


def part(name,index):
    if 7<=index<14:return 'tail'
    low=name.lower()
    if any(x in low for x in ('paw','wrist','ankle','elbow','knee','hand','foot')):return 'limbs'
    if any(x in low for x in ('nose','ear','eye','head','snout')):return 'head'
    return 'trunk'


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'protocol.json').exists():raise RuntimeError('Audit output already exists; do not overwrite')
    start=time.monotonic()
    spec=json.loads((RUN/'protocol.json').read_text())
    assert json.loads((RUN/'delivery_verification.json').read_text())['passed']
    checkpoints={'original':ROOT/spec['config']['checkpoint'],
        'old_sampling':PARENT/'balanced_views/final_heads.pt',
        'new_sampling':RUN/'combination_coverage/final_heads.pt'}
    code=dict(spec['code_sha256'])
    for path in ('tools/audit_cylinder_localization.py','mousegeoforge/localization_audit.py'):
        code[path]=fingerprint(ROOT/path)
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    scenes={spec['records'][e['key']]['scene']:spec['records'][e['key']]['source_revision'] for e in spec['evaluation']}
    assert all(scene_digest(ROOT/p)==h for p,h in scenes.items())
    base=ROOT/'vggt-omega/vggt_omega_1b_512.pt'
    assert fingerprint(base)==spec['encoder_sha256']
    protocol=dict(parent_protocol_sha256=fingerprint(RUN/'protocol.json'),encoder_sha256=spec['encoder_sha256'],
        checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in checkpoints.items()},
        code_sha256=code,source_sha256=scenes,weights_updated=False,
        scope='Frozen three-checkpoint decomposition on all 88 reused synthetic development records; no real Cylinder.',
        matching='Same saved GT-box/objectness matching as parent; coarse/refined share assignments; not end-to-end metrics.',
        aggregation='Point-weighted valid labels; visibility split uses visibility_valid, unknown kept separate.',
        counterfactual='Scales 1/4/8/16 applied to saved local logits only, coarse/assignment fixed; posthoc diagnostic, no tuned checkpoint.',
        evaluation=spec['evaluation'])
    write_json(OUTPUT/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(base,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in checkpoints.items()}
    reference={'original':(RUN/'initial'),'old_sampling':(PARENT/'balanced_views'),'new_sampling':(RUN/'combination_coverage')}
    rows=[];parity=0.;names=None
    for idx,e in enumerate(spec['evaluation']):
        r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']))
        target=instance_targets_from_batch(batch);images=batch['images'].cuda()
        with np.load(ROOT/r['scene']/'keypoints.npz',allow_pickle=False) as a:
            current=[str(x) for x in a['names']]
        if names is None:names=current;print('Keypoint names:',names,flush=True)
        assert current==names and len(names)==23
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):cached,patch_start=model.aggregator(images)
        for tag,state in states.items():
            restore(model,state)
            with torch.inference_mode():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,patch_start,coupling='off').items() if k.startswith('instance_') and 'mask' not in k}
            saved=torch.load(reference[tag]/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
            assert saved['record']==r
            for key,value in saved['predictions'].items():
                difference=float((pred[key]-value).abs().max());parity=max(parity,difference)
                assert difference<1e-6,(tag,idx,key,difference)
            folder=OUTPUT/tag;folder.mkdir(exist_ok=True)
            torch.save(dict(record=r,group=e['group'],predictions=pred,targets=saved['targets'],
                oracle_assignment=saved['oracle_assignment'],names=names),folder/f'prediction_{idx:03d}.pt')
            for v,(qs,ms) in enumerate(saved['oracle_assignment']):
                for q,m in zip(qs,ms):
                    for k in range(23):
                        if not target['keypoint_valid'][0,v,m,k]:continue
                        visible=bool(target['keypoint_visible'][0,v,m,k]);known=bool(target['keypoint_visibility_valid'][0,v,m,k])
                        stats=refinement_statistics(pred['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),
                            pred['instance_keypoint_xy'][0,v,q,k].numpy(),target['keypoint_xy'][0,v,m,k].numpy(),
                            pred['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),r['image_hw'],
                            target['boxes_cxcywh'][0,v,m,2:].numpy(),model.instance_pose_head.local_window_radius)
                        rows.append(dict(checkpoint=tag,record=idx,group=e['group'],scene=r['scene'],camera=r['views'][v],
                            role=r['physical_roles'][v],instance=m,keypoint=k,name=names[k],part=part(names[k],k),
                            visibility='visible' if visible and known else 'occluded' if known else 'unknown',**stats))
            del pred
        del cached,images,batch,target
        if (idx+1)%8==0 or idx==87:
            write_json(OUTPUT/'status.json',dict(state='evaluating',records=idx+1,total=88,elapsed_s=time.monotonic()-start))
            print('Completed',idx+1,'/88',flush=True)
    grouped=defaultdict(list)
    for row in rows:
        if row['part']=='tail':continue
        key=(row['checkpoint'],row['group'],row['role'])
        for suffix in ('body',row['part'],row['visibility'],'inside' if row['inside_window'] else 'outside'):
            grouped[key+(suffix,)].append(row)
    summary=[dict(checkpoint=k[0],group=k[1],role=k[2],subset=k[3],**summarize(v)) for k,v in sorted(grouped.items())]
    write_json(OUTPUT/'points.json',rows);write_json(OUTPUT/'summary.json',summary)
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in checkpoints.items())
    assert all(fingerprint(ROOT/p)==h for p,h in code.items())
    assert all(scene_digest(ROOT/p)==h for p,h in scenes.items())
    write_json(OUTPUT/'verification.json',dict(passed=True,prediction_parity_max_difference=parity,
        checkpoints_unchanged=True,source_and_code_unchanged=True,encoder_forwards=88,head_forwards=264,
        valid_point_rows=len(rows),keypoint_names=names,local_radius=model.instance_pose_head.local_window_radius,
        grid_size=model.instance_pose_head.local_heatmap_size,elapsed_s=time.monotonic()-start))
    write_json(OUTPUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'run.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid)));return
    try:run()
    except BaseException as e:
        write_json(OUTPUT/'status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':main()
