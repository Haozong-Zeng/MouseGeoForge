"""Recheck local-window coverage on the completed auxiliary-loss candidate."""
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_cylinder_coarse_aux import ROOT,OUTPUT,ARM
from run_depth_pose_ablation import torch,VGGTOmega,DepthPoseModel,restore,instance_targets_from_batch,fingerprint,write_json
from train_cylinder_view_balance import oriented_batch
from audit_cylinder_localization import part
from mousegeoforge.localization_audit import refinement_statistics,summarize


def main():
    assert json.loads((OUTPUT/'delivery_verification.json').read_text())['passed']
    spec=json.loads((OUTPUT/'protocol.json').read_text())
    checkpoint=OUTPUT/ARM/'final_heads.pt';digest=fingerprint(checkpoint)
    torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
    weights=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',weights_only=True,mmap=True)
    loaded=spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
    del weights
    model=DepthPoseModel.from_spatial_model(spatial).cuda().eval();del spatial
    state=torch.load(checkpoint,weights_only=True,map_location='cpu');restore(model,state['model']);del state
    names=json.loads((ROOT/'output/cylinder_localization_audit_v1/verification.json').read_text())['keypoint_names']
    rows=[];parity=0.
    for idx,e in enumerate(spec['evaluation']):
        r=spec['records'][e['key']];batch=oriented_batch(r,num_views=len(r['views']));t=instance_targets_from_batch(batch)
        with torch.inference_mode():p={k:v.cpu() for k,v in model(batch['images'].cuda(),coupling='off').items() if k.startswith('instance_') and 'mask' not in k}
        saved=torch.load(OUTPUT/ARM/f'prediction_{idx:03d}.pt',weights_only=True,map_location='cpu')
        for key,value in saved['predictions'].items():
            err=float((p[key]-value).abs().max());parity=max(parity,err);assert err<1e-6
        if r['scene'].endswith('000055') and e['group'] in ('bottom_only','side_bottom','sides_bottom'):
            folder=OUTPUT/'localization';folder.mkdir(exist_ok=True)
            torch.save(dict(record=r,group=e['group'],predictions=p,targets=saved['targets'],oracle_assignment=saved['oracle_assignment']),folder/f'prediction_{idx:03d}.pt')
        for v,(qs,ms) in enumerate(saved['oracle_assignment']):
            for q,m in zip(qs,ms):
                for k in range(23):
                    if 7<=k<14 or not t['keypoint_valid'][0,v,m,k]:continue
                    visible=bool(t['keypoint_visible'][0,v,m,k]);known=bool(t['keypoint_visibility_valid'][0,v,m,k])
                    s=refinement_statistics(p['instance_keypoint_coarse_xy'][0,v,q,k].numpy(),p['instance_keypoint_xy'][0,v,q,k].numpy(),
                        t['keypoint_xy'][0,v,m,k].numpy(),p['instance_keypoint_local_heatmap_logits'][0,v,q,k].numpy(),
                        r['image_hw'],t['boxes_cxcywh'][0,v,m,2:].numpy())
                    rows.append(dict(checkpoint=ARM,record=idx,group=e['group'],scene=r['scene'],camera=r['views'][v],
                        role=r['physical_roles'][v],instance=m,keypoint=k,name=names[k],part=part(names[k],k),
                        visibility='visible' if visible and known else 'occluded' if known else 'unknown',**s))
        if (idx+1)%16==0:print('localization',idx+1,'/88',flush=True)
    assert fingerprint(checkpoint)==digest
    write_json(OUTPUT/'localization_points.json',rows)
    old=[r for r in json.loads((ROOT/'output/cylinder_localization_audit_v1/points.json').read_text()) if r['checkpoint']=='new_sampling' and r['part']!='tail']
    key=lambda r:(r['group'],r['scene'],r['camera'],r['instance'],r['keypoint'])
    assert {key(r) for r in rows}=={key(r) for r in old}
    groups=['bottom_only','side_bottom','sides_bottom','sides_bottom_top'];comparison=[]
    for group in groups:
        for subset in ('body','head','limbs','visible','occluded'):
            choose=lambda r:r['group']==group and r['role']=='bottom' and (subset=='body' or r['part']==subset or r['visibility']==subset)
            a=summarize([r for r in old if choose(r)]);b=summarize([r for r in rows if choose(r)])
            comparison.append(dict(group=group,subset=subset,off=a,on=b))
    report=dict(passed=True,checkpoint_sha256=digest,prediction_parity_max_difference=parity,valid_body_points=len(rows),comparison=comparison,
        weights_updated=False,scope='Same reused synthetic development; fixed GT-box assisted matches, body16; not independent validation')
    write_json(OUTPUT/'localization_report.json',report)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    body=[r for r in comparison if r['subset']=='body']
    for ax,k,title in zip(axes,('coarse_nme','refined_nme','inside_window'),('Coarse body NME','Refined body NME','GT inside local window')):
        for i,(tag,color) in enumerate((('off','#b27649'),('on','#218d89'))):
            ax.bar(np.arange(4)+(i-.5)*.3,[r[tag][k] for r in body],.3,label='aux '+tag,color=color)
        ax.set_title(title);ax.set_xticks(range(4),groups,rotation=25,ha='right');ax.legend()
    fig.suptitle('Bottom view | body16 | exact 800-step sampling control')
    fig.savefig(OUTPUT/'localization_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps([dict(group=r['group'],off_inside=r['off']['inside_window'],on_inside=r['on']['inside_window'],off_coarse=r['off']['coarse_nme'],on_coarse=r['on']['coarse_nme']) for r in body],indent=2))


if __name__=='__main__':main()
