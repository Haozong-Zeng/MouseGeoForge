"""Audit every registered diverse scene and plot a fixed training-only GT montage."""
import json
from pathlib import Path
from run_depth_pose_ablation import ROOT,torch,load_batch,depth_alignment_audit,write_json


def main():
    torch.set_num_threads(2)
    folder=ROOT/'output/pose_diverse_core23_preflight_v1'
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[]; examples=[]; seen={}
    for r in manifest['records']:
        b=load_batch(r); h,w=r['image_hw']; errors=[]
        for v in range(2):
            world=b['instance_keypoint_world_xyz'][0]
            e=b['extrinsics'][0,v]; k=b['intrinsics'][0,v]
            camera=world@e[:3,:3].T+e[:3,3]; p=camera@k.T
            uv=p[...,:2]/p[...,2:]
            gt=b['instance_keypoint_xy'][0,v]*torch.tensor([w,h])
            valid=b['instance_keypoint_valid'][0,v]
            errors.extend((uv-gt).norm(dim=-1)[valid].tolist())
        assert errors and max(errors)<.01,(r['scene'],max(errors))
        audit=depth_alignment_audit(b)
        assert audit['surface_anchor_absrel_median']<.15
        rows.append(dict(scene=r['scene'],image_hw=r['image_hw'],max_projection_px=max(errors),**audit))
        if r['split']=='train' and (r['dataset'] not in seen or r['frame'] in (40,98) and r['expected_instances']==2):
            examples.append((r,b)); seen[r['dataset']]=True
        if len(rows)%64==0: print(f'Audited {len(rows)}/432',flush=True)
    write_json(folder/'geometry_audit.json',dict(passed=True,scenes=len(rows),records=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(examples),2,figsize=(10,3*len(examples)),layout='constrained')
    for row,(r,b) in zip(axes,examples):
        h,w=r['image_hw']
        for v,ax in enumerate(row):
            ax.imshow(b['images'][0,v].permute(1,2,0).numpy())
            for m in range(r['expected_instances']):
                good=b['instance_keypoint_valid'][0,v,m]
                xy=b['instance_keypoint_xy'][0,v,m]*torch.tensor([w,h])
                ax.scatter(xy[good,0],xy[good,1],s=5,color=('red','cyan')[m])
            ax.set_title(f"{r['dataset'][:35]} / frame {r['frame']} / tensor view {v}",fontsize=8)
            ax.axis('off')
    fig.savefig(folder/'training_sources.png',dpi=120); plt.close(fig)
    print('DIVERSE GEOMETRY PASS')


if __name__=='__main__': main()
