"""Post-hoc label/camera sanity checks and figures; not another tuned experiment."""
from collections import Counter, defaultdict
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_view_redundancy import ROOT, OLD, METHODS, torch, oracle, view_indices, write_json
from vggt_omega.utils.robust_pose_geometry import triangulate_robust, project
from mousegeoforge.pilot_metrics import box_iou


def main():
    folder = ROOT/'output/occlusion_pose_v4/view_redundancy'
    report = json.loads((folder/'report.json').read_text())
    data = [torch.load(p,weights_only=True) for p in sorted((folder/'predictions').glob('*.pt'))]
    uv_errors, gt_errors = [], []
    per_view = defaultdict(list); detection = defaultdict(Counter)
    for item in data:
        t,p = item['targets'],item['predictions']; matches = oracle(item)
        h,w = item['record']['image_hw']; scale = torch.tensor([w,h])
        cameras = list(map(int,item['actual_view_ids'][0].split(',')))
        proj = t['intrinsics'][0]@t['extrinsics'][0,:,:3]
        native = float(item['scale'].flatten()[0])
        for v,camera in enumerate(cameras):
            for m,q in matches[v].items():
                iou = float(box_iou(t['boxes_cxcywh'][0,v,m:m+1],p['instance_boxes_cxcywh'][0,v,q:q+1])[0,0])
                detected = iou>=.5 and float(p['instance_objectness_logits'][0,v,q].sigmoid())>=.5
                detection[camera]['instances']+=1; detection[camera]['detected']+=int(detected)
                valid = t['keypoint_valid'][0,v,m]
                e = ((p['instance_keypoint_xy'][0,v,q]-t['keypoint_xy'][0,v,m])*scale).norm(dim=-1)
                diag = (t['boxes_cxcywh'][0,v,m,2:]*scale).norm().clamp_min(1.)
                per_view[camera].extend((float(error),float(error/diag)) for error in e[valid])
        for m in range(2):
            for k in range(23):
                gt = t['keypoint_world_xyz'][0,m,k]
                predicted_uv,_ = project(proj,gt)
                target_uv = t['keypoint_xy'][0,:,m,k]*scale
                valid = t['keypoint_valid'][0,:,m,k]
                uv_errors.extend((predicted_uv-target_uv).norm(dim=-1)[valid].tolist())
                solved = triangulate_robust(target_uv,proj,valid,huber_delta_px=3.)
                if not solved.supported: raise AssertionError('Ground-truth observations failed triangulation')
                gt_errors.append(float((solved.xyz-gt).norm())*native)
    manifest = json.loads((ROOT/'output/pose_diverse_core23_768_v1/manifest.json').read_text())
    result = dict(posthoc_diagnostic=True,
        gt_reprojection=dict(points=len(uv_errors),mean_px=float(np.mean(uv_errors)),max_px=max(uv_errors)),
        gt_triangulation=dict(points=len(gt_errors),mean_mm=float(np.mean(gt_errors)),max_mm=max(gt_errors)),
        four_input_per_camera={str(k):dict(points=len(v),mean_px=float(np.mean([x[0] for x in v])),
            nme=float(np.mean([x[1] for x in v])),**detection[k]) for k,v in per_view.items()},
        dual_training_view_pairs=dict(Counter(str(r['views']) for r in manifest['records'] if r['split']=='train' and r['expected_instances']==2)))
    write_json(folder/'camera_label_audit.json',result)
    print(json.dumps(result,indent=2))

    fig, axes = plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    labels = ['2 input\n2 rays','4 input\n2 rays','4 input\n4 rays','4 input\n>=3 inliers']
    colors = ['#6b8799','#d39955','#b04f5e','#558d7b']
    for ax,key,factor,title in ((axes[0],'coverage',100,'Supported 3D points (%)'),
                                (axes[1],'mean_mm',1,'Mean 3D error (mm; differing subsets)')):
        values = [report['summary'][m][key]*factor for m in METHODS]
        ax.bar(range(4),values,color=colors)
        for i,val in enumerate(values): ax.text(i,val,f'{val:.1f}',ha='center',va='bottom')
        ax.set(xticks=range(4),xticklabels=labels,title=title,ylim=(0,max(values)*1.2))
    hist = report['four_view_inlier_histogram']
    axes[2].bar(range(4),[hist[str(k)] for k in (0,2,3,4)],color=colors)
    axes[2].set(xticks=range(4),xticklabels=['Rejected','2 inliers','3 inliers','4 inliers'],title='4-ray consensus support')
    for i,k in enumerate((0,2,3,4)): axes[2].text(i,hist[str(k)],str(hist[str(k)]),ha='center',va='bottom')
    axes[2].set_ylim(0,1050)
    fig.suptitle('Frozen 768 model | 24 reused development scenes | GT-assisted instance matching')
    fig.supxlabel('Higher coverage is not higher accuracy. No model or production fusion changes.',fontsize=10)
    fig.savefig(folder/'view_redundancy_summary.png',dpi=150); plt.close(fig)

    # Explicit worst shared-view 2D degradation, not an unlabelled cherry-picked example.
    scores = defaultdict(list)
    for row in report['rows_2d']:
        scores[row['scene']].append(row['four_input']['nme']-row['two_input']['nme'])
    worst = max(scores,key=lambda s:np.mean(scores[s]))
    idx = next(i for i,x in enumerate(data) if x['record']['scene']==worst)
    new = data[idx]; old = torch.load(OLD/'predictions'/f'{idx:03d}.pt',weights_only=True)
    pair = list(map(int,old['actual_view_ids'][0].split(',')))
    def panel(ax,item,camera,title):
        h,w = item['record']['image_hw']; scale = np.array([w,h])
        image = Image.open(ROOT/worst/'blended_images'/f'{camera:08d}.jpg').convert('RGB').resize((w,h),Image.Resampling.BICUBIC)
        ax.imshow(image); v = view_indices(item,[camera])[0]
        for m,q in oracle(item)[v].items():
            gt = item['targets']['keypoint_xy'][0,v,m].numpy()*scale
            pred = item['predictions']['instance_keypoint_xy'][0,v,q].numpy()*scale
            valid = item['targets']['keypoint_valid'][0,v,m].numpy()
            vis = item['targets']['keypoint_visible'][0,v,m].numpy()
            for k in np.flatnonzero(valid): ax.plot([gt[k,0],pred[k,0]],[gt[k,1],pred[k,1]],color='white',alpha=.5,lw=.7)
            ax.scatter(gt[valid&vis,0],gt[valid&vis,1],s=25,facecolors='none',edgecolors='#44ff70',linewidths=1)
            ax.scatter(gt[valid&~vis,0],gt[valid&~vis,1],s=25,facecolors='none',edgecolors='#ffbf35',linewidths=1)
            ax.scatter(pred[valid,0],pred[valid,1],s=20,c='#ff4364',marker='x')
        ax.set_title(title,fontsize=11); ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis('off')
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,item in enumerate((old,new)):
        for col,camera in enumerate(pair): panel(axes[row,col],item,camera,f'{2 if row==0 else 4}-image input | camera ID {camera}')
    fig.suptitle(f'Frame {new["record"]["frame"]}: largest shared-view NME degradation\nGreen: visible GT | Amber: occluded GT | Red: prediction',fontsize=13)
    fig.savefig(folder/'worst_context_comparison.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for camera,ax in enumerate(axes.flat): panel(ax,new,camera,f'4-image input | camera ID {camera}')
    fig.suptitle(f'Frame {new["record"]["frame"]}: all four cameras | GT box-assisted correspondence',fontsize=13)
    fig.savefig(folder/'worst_four_views.png',dpi=150); plt.close(fig)


if __name__=='__main__': main()
