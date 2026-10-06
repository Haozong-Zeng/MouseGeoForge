"""Frozen OFF/ON coarse/local decomposition on the unchanged 88 records."""
from collections import defaultdict
import fcntl
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,DepthPoseModel,restore,tensor_state,state_hash,fingerprint,write_json
from train_cylinder_view_balance import oriented_batch,scene_digest
from mousegeoforge.localization_audit import refinement_statistics,summarize
from train_cylinder_epipolar import OUTPUT,PARENT,ARM

OUT=ROOT/'output/cylinder_epipolar_localization_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'output/cylinder_training.lock').open('a+') as glock,(OUT/'running.lock').open('a+') as lock:
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing localization audit; do not overwrite')
        run()


def run():
    start=time.monotonic();torch.set_num_threads(2)
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    for folder in (PARENT,OUTPUT):
        assert json.loads((folder/'delivery_verification.json').read_text())['passed']
        assert json.loads((folder/'status.json').read_text())['state']=='completed'
        with (folder/'running.lock').open() as file:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths={'off':PARENT/'source_mix/final_heads.pt','on':OUTPUT/ARM/'final_heads.pt'}
    folders={'off':PARENT/'source_mix','on':OUTPUT/ARM}
    sources={spec['records'][e['key']]['scene']:spec['records'][e['key']]['source_revision'] for e in spec['evaluation']}
    for p,h in spec['code_sha256'].items():assert fingerprint(ROOT/p)==h
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    inputs={str(folder/f'prediction_{i:03d}.pt'):fingerprint(folder/f'prediction_{i:03d}.pt') for folder in folders.values() for i in range(88)}
    protocol=dict(checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=fingerprint(p)) for k,p in paths.items()},
        source_sha256=sources,predictions_sha256=inputs,training_protocol_sha256=fingerprint(OUTPUT/'protocol.json'),
        code_sha256={p:fingerprint(ROOT/p) for p in ('tools/audit_cylinder_epipolar_localization.py','mousegeoforge/localization_audit.py')},
        evaluation=spec['evaluation'],scope='Same 88 reused synthetic development records; frozen source-mixture OFF vs epipolar ON; GT-box-assisted assignment reused; no updates.',
        aggregation='Valid body16 point-weighted; head4/limbs8/trunk4 and known visible/occluded/unknown separately. No tail points in new auxiliary or this body diagnostic.')
    write_json(OUT/'protocol.json',protocol)
    torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';assert fingerprint(encoder)==spec['encoder_sha256']
    weights=torch.load(encoder,weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    states={k:torch.load(p,weights_only=True,map_location='cpu')['model'] for k,p in paths.items()}
    points=[]
    for idx,e in enumerate(spec['evaluation']):
        r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));images=batch['images'].cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start_patch=model.aggregator(images)
        for tag,state in states.items():
            restore(model,state)
            with torch.inference_mode():pred={k:v.cpu() for k,v in model.forward_cached(images,cached,start_patch,coupling='off').items()}
            saved=torch.load(folders[tag]/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu');target=saved['targets']
            assert saved['record']==r
            for key,value in saved['predictions'].items():torch.testing.assert_close(pred[key],value,atol=0,rtol=0)
            for v,(qs,ms) in enumerate(saved['oracle_assignment']):
                for q,m in zip(qs,ms):
                    for k in list(range(7))+list(range(14,23)):
                        if not target['keypoint_valid'][0,v,m,k]:continue
                        vis='unknown' if not target['keypoint_visibility_valid'][0,v,m,k] else 'visible' if target['keypoint_visible'][0,v,m,k] else 'occluded'
                        part='head' if k<4 else 'limbs' if 14<=k<22 else 'trunk'
                        stats=refinement_statistics(pred['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),pred['instance_keypoint_xy'][0,v,q,k].numpy(),
                            target['keypoint_xy'][0,v,m,k].numpy(),pred['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),
                            r['image_hw'],target['boxes_cxcywh'][0,v,m,2:].numpy(),scales=(1,))
                        points.append(dict(checkpoint=tag,record=idx,scene=r['scene'],group=e['group'],camera=r['views'][v],role=r['physical_roles'][v],instance=m,keypoint=k,part=part,visibility=vis,**stats))
            del pred,saved
        del cached,batch,images
        if (idx+1)%22==0:print('Checked',idx+1,'/88',flush=True)
    grouped=defaultdict(list)
    for p in points:
        for part in ('body',p['part']):
            for vis in ('all',p['visibility']):grouped[p['checkpoint'],p['group'],p['role'],part,vis].append(p)
    summary=[dict(checkpoint=k[0],group=k[1],role=k[2],part=k[3],visibility=k[4],**summarize(v)) for k,v in sorted(grouped.items())]
    for tag,state in states.items():restore(model,state);assert state_hash(tensor_state(model))==state_hash(state)
    assert all(fingerprint(p)==protocol['checkpoints'][k]['sha256'] for k,p in paths.items())
    assert all(fingerprint(ROOT/p)==h for p,h in protocol['code_sha256'].items())
    from pathlib import Path
    assert all(fingerprint(Path(p))==h for p,h in inputs.items())
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    write_json(OUT/'points.json',points);write_json(OUT/'summary.json',summary)
    report=dict(passed=True,encoder_forwards=88,head_forwards=176,weights_source_code_inputs_unchanged=True,optimizer_steps=0,
        prediction_max_difference=0,body_points=len(points),elapsed_s=time.monotonic()-start)
    write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    gs=('bottom_only','side_bottom','sides_bottom','sides_bottom_top')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for tag in states:
        data=[next(r for r in summary if (r['checkpoint'],r['group'],r['role'],r['part'],r['visibility'])==(tag,g,'bottom','body','all')) for g in gs]
        for ax,key in zip(axes,('coarse_nme','refined_nme','inside_window')):ax.plot(range(4),[r[key] for r in data],marker='o',label=tag)
    for ax,title in zip(axes,('Coarse body NME','Final body NME','GT inside local window')):
        ax.set_title(title);ax.set_xticks(range(4),gs,rotation=25,ha='right');ax.legend()
    fig.suptitle('Frozen OFF/ON localization | same 88 development inputs | body16')
    fig.savefig(OUT/'localization.png',dpi=145);plt.close(fig)
    print(json.dumps(report,indent=2))
    print(json.dumps([r for r in summary if r['group']=='side_bottom' and r['role']=='bottom'],indent=2))


if __name__=='__main__':main()
