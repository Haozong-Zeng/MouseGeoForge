"""Read-only saved-output decomposition, without encoder calls or training."""
from collections import Counter,defaultdict
import fcntl
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

from run_depth_pose_ablation import ROOT,torch,fingerprint,write_json
from train_cylinder_view_balance import scene_digest
from mousegeoforge.retention_geometry_audit import BODY,matched_views,geometry_points

OUT=ROOT/'output/cylinder_retention_geometry_v1'
PARENT=ROOT/'output/cylinder_combinations_v2'
MIX=ROOT/'output/cylinder_source_mix_v1'
GROUPS=('old_dual_pair','side_bottom','sides_bottom','sides_bottom_top')


def mean(values):return float(np.mean(values)) if values else None


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'running.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'protocol.json').exists():raise RuntimeError('Existing audit; do not overwrite')
        run()


def run():
    start=time.monotonic();torch.set_num_threads(2)
    for folder in (PARENT,MIX):
        assert json.loads((folder/'status.json').read_text())['state']=='completed'
        assert json.loads((folder/'delivery_verification.json').read_text())['passed']
        with (folder/'running.lock').open() as file:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=json.loads((MIX/'protocol.json').read_text())
    selected=[(i,e) for i,e in enumerate(spec['evaluation']) if e['group'] in GROUPS]
    assert len(selected)==40
    folders={'original':PARENT/'initial','v2':PARENT/'combination_coverage','mixture':MIX/'source_mix'}
    files=[folder/f'prediction_{i:03d}.pt' for folder in folders.values() for i,e in selected]
    files += [PARENT/'results.json',MIX/'results.json',MIX/'protocol.json']
    code=['tools/audit_cylinder_retention_geometry.py','mousegeoforge/retention_geometry_audit.py',
          'mousegeoforge/pilot_metrics.py','vggt-omega/vggt_omega/utils/robust_pose_geometry.py']
    for path,digest in spec['code_sha256'].items():assert fingerprint(ROOT/path)==digest
    sources={spec['records'][e['key']]['scene']:spec['records'][e['key']]['source_revision'] for i,e in selected}
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    protocol=dict(input_sha256={str(p.relative_to(ROOT)):fingerprint(p) for p in files},
        code_sha256={p:fingerprint(ROOT/p) for p in code},source_sha256=sources,
        selected_evaluation_indices=[i for i,e in selected],checkpoints=list(folders),
        scope='Saved predictions only; 16 dual records +8 Cylinder poses x3 multiview groups. Reused synthetic development, not independent test.',
        association='Reuse saved GT-box/objectness assignment. Do not alter queries for tail/body analysis.',
        geometry='Original triangulate_robust with 8px reprojection, 1deg ray-angle, Huber3; report failure reasons, no-objectness and GT replacements only as labeled counterfactuals.',
        aggregation='Point-weighted dual decomposition; parity with saved scene metrics checked. Common-support 3D point-weighted versus original scene-macro conditional means; do not conflate.',
        visuals='Fixed frame55 for Cylinder; dual example selected by most lost detections mixture vs original, scene-name tie-break, after analysis only.')
    write_json(OUT/'protocol.json',protocol)
    loaded={};detection=[];points=[];geometry=[];parity=[]
    for tag,folder in folders.items():
        for i,e in selected:
            s=torch.load(folder/f'prediction_{i:03d}.pt',weights_only=True,map_location='cpu');loaded[tag,i]=s
            assert s['record']==spec['records'][e['key']] and s['group']==e['group']
            rows=matched_views(s)
            if e['group']=='old_dual_pair':
                local=[]
                for row in rows:
                    det={k:v for k,v in row.items() if k not in ('errors','valid','visible','visibility_valid')}
                    detection.append(dict(checkpoint=tag,record=i,scene=s['record']['scene'],**det))
                    for k in torch.where(row['valid'])[0].tolist():
                        vis='unknown' if not row['visibility_valid'][k] else 'visible' if row['visible'][k] else 'occluded'
                        part='tail' if k not in BODY else 'head' if k<4 else 'limbs' if 14<=k<22 else 'trunk'
                        point=dict(checkpoint=tag,record=i,scene=s['record']['scene'],role=row['role'],instance=row['instance'],
                            keypoint=k,part=part,visibility=vis,nme=float(row['errors'][k]),found=row['found'])
                        points.append(point);local.append(point)
                for metric,subset in (('box_matched_nme',local),('body_keypoint_nme',[p for p in local if p['part']!='tail']),('tail_nme',[p for p in local if p['part']=='tail'])):
                    parity.append(abs(mean([p['nme'] for p in subset])-s['metrics'][metric]))
                parity.append(abs(mean([float(p['found'] and p['nme']<=.1) for p in local])-s['metrics']['detection_aware_pck_0.1']))
            else:
                result=geometry_points(s)
                parity.append(abs(sum(p['supported'] for p in result)/16-s['metrics']['calibrated_body3d_coverage']))
                errors=[p['error_native'] for p in result if p['supported']]
                if errors:parity.append(abs(mean(errors)-s['metrics']['calibrated_body3d_error_native']))
                geometry.extend(dict(checkpoint=tag,record=i,scene=s['record']['scene'],group=e['group'],**p) for p in result)
    assert max(parity)<1e-5,max(parity)
    dual_summary=[]
    for tag in folders:
        for role in ('all','top','side'):
            det=[r for r in detection if r['checkpoint']==tag and (role=='all' or r['role']==role)]
            failures=Counter('score_and_box' if r['score']<.5 and r['iou']<.5 else 'score_only' if r['score']<.5 else 'box_only' for r in det if not r['found'])
            for part in ('all','body','tail','head','limbs','trunk'):
                pts=[r for r in points if r['checkpoint']==tag and (role=='all' or r['role']==role) and
                     (part=='all' or (part=='body' and r['part']!='tail') or r['part']==part)]
                dual_summary.append(dict(checkpoint=tag,role=role,part=part,count=len(pts),nme=mean([r['nme'] for r in pts]),
                    matched_pck=mean([float(r['nme']<=.1) for r in pts]),detection_aware_pck=mean([float(r['found'] and r['nme']<=.1) for r in pts]),
                    instances=len(det),found=sum(r['found'] for r in det),failure_reasons=dict(failures),
                    visibility={v:dict(count=len(pv),nme=mean([p['nme'] for p in pv])) for v in ('visible','occluded','unknown')
                        for pv in [[p for p in pts if p['visibility']==v]]}))
    geo_summary=[];common=[]
    for g in GROUPS[1:]:
        for tag in folders:
            rows=[r for r in geometry if (r['group'],r['checkpoint'])==(g,tag)]
            geo_summary.append(dict(group=g,checkpoint=tag,total=len(rows),supported=sum(r['supported'] for r in rows),
                reasons=dict(Counter(r['reason'] for r in rows)),without_objectness_supported=sum(r['without_objectness_supported'] for r in rows),
                gt_supported=sum(r['gt_supported'] for r in rows),
                epipolar_mean_px=mean([r['epipolar_distance_px'] for r in rows if 'epipolar_distance_px' in r]),
                gt_replace_view_support=[sum(r['gt_replace_one_view_supported'][j] for r in rows) for j in range(2)] if g=='side_bottom' else None))
        for a,b in (('original','mixture'),('v2','mixture')):
            ra={(r['scene'],r['keypoint']):r for r in geometry if (r['group'],r['checkpoint'])==(g,a)}
            rb={(r['scene'],r['keypoint']):r for r in geometry if (r['group'],r['checkpoint'])==(g,b)}
            keys=[k for k in ra if ra[k]['supported'] and rb[k]['supported']]
            transitions=Counter(f'{int(ra[k]["supported"])}->{int(rb[k]["supported"])}' for k in ra)
            common.append(dict(group=g,reference=a,candidate=b,common_points=len(keys),transitions=dict(transitions),
                reference_common_error=mean([ra[k]['error_native'] for k in keys]),candidate_common_error=mean([rb[k]['error_native'] for k in keys])))
    write_json(OUT/'dual_points.json',points);write_json(OUT/'detections.json',detection);write_json(OUT/'geometry_points.json',geometry)
    write_json(OUT/'dual_summary.json',dual_summary);write_json(OUT/'geometry_summary.json',geo_summary);write_json(OUT/'common_support.json',common)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for j,tag in enumerate(folders):
        dual=next(r for r in dual_summary if (r['checkpoint'],r['role'],r['part'])==(tag,'all','body'))
        geo=next(r for r in geo_summary if (r['checkpoint'],r['group'])==(tag,'side_bottom'))
        axes[0].bar(j,dual['nme']);axes[1].bar(j,geo['supported']/128);axes[2].bar(j,geo['epipolar_mean_px'])
    for ax,title in zip(axes,('Dual body16 NME','Side-bottom supported /128','Side-bottom epipolar distance (px)')):
        ax.set_xticks(range(3),list(folders));ax.set_title(title)
    fig.suptitle('Frozen saved outputs | same GT-assisted assignments | original gates unchanged')
    fig.savefig(OUT/'decomposition.png',dpi=145);plt.close(fig)
    # Side and bottom are both shown; no cherry-picked per-view checkpoint.
    i=next(i for i,e in selected if e['group']=='side_bottom' and loaded['original',i]['record']['scene'].endswith('000055'))
    fig,axes=plt.subplots(3,2,figsize=(9,12),layout='constrained')
    for j,tag in enumerate(folders):
        s=loaded[tag,i];rec=s['record'];t=s['targets'];p=s['predictions'];pixel=np.array(rec['image_hw'][::-1])
        for row in matched_views(s):
            v=row['view'];q=row['query'];ax=axes[j,v]
            xy=p['instance_keypoint_xy'][0,v,q].numpy()*pixel;gt=t['keypoint_xy'][0,v,0].numpy()*pixel
            ok=row['valid'].numpy().copy();ok[7:14]=False
            ax.imshow(Image.open(ROOT/rec['scene']/'blended_images'/f'{rec["views"][v]:08d}.jpg').resize(tuple(pixel)))
            ax.scatter(gt[ok,0],gt[ok,1],c='lime',s=12);ax.scatter(xy[ok,0],xy[ok,1],c='magenta',marker='x',s=17)
            for a,b in zip(gt[ok],xy[ok]):ax.plot([a[0],b[0]],[a[1],b[1]],color='magenta',lw=.5)
            lo=np.maximum(gt[ok].min(0)-80,0);hi=np.minimum(gt[ok].max(0)+80,pixel)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.axis('off')
            ax.set_title(f'{tag} / {row["role"]} / score={row["score"]:.3f}')
    fig.suptitle('Fixed frame55, side-bottom inputs | GT green, prediction magenta')
    fig.savefig(OUT/'fixed_frame55_views.png',dpi=145);plt.close(fig)
    losses=[]
    for idx,e in selected:
        if e['group']!='old_dual_pair':continue
        old=sum(r['found'] for r in detection if r['checkpoint']=='original' and r['record']==idx)
        new=sum(r['found'] for r in detection if r['checkpoint']=='mixture' and r['record']==idx)
        losses.append((new-old,loaded['mixture',idx]['record']['scene'],idx))
    _,scene,idx=min(losses)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,tag in zip(axes,folders):
        s=loaded[tag,idx];rec=s['record'];v=rec['physical_roles'].index('top');pixel=np.array(rec['image_hw'][::-1])
        ax.imshow(Image.open(ROOT/scene/'blended_images'/f'{rec["views"][v]:08d}.jpg').resize(tuple(pixel)))
        titles=[]
        for r in matched_views(s):
            if r['view']!=v:continue
            for box,color in ((s['targets']['boxes_cxcywh'][0,v,r['instance']],'lime'),(s['predictions']['instance_boxes_cxcywh'][0,v,r['query']],'magenta')):
                box=box.numpy();xy=(box[:2]-box[2:]/2)*pixel;wh=box[2:]*pixel
                ax.add_patch(Rectangle(xy,*wh,fill=False,edgecolor=color,lw=1))
            titles.append(f'm{r["instance"]}: score {r["score"]:.2f}, IoU {r["iou"]:.2f}')
        ax.set_title(tag+'\n'+'\n'.join(titles),fontsize=9);ax.axis('off')
    fig.suptitle('Posthoc detection-loss example: '+scene.split('/')[-1]+' | top | GT green / predicted magenta')
    fig.savefig(OUT/'dual_detection_example.png',dpi=145);plt.close(fig)
    for p,h in protocol['input_sha256'].items():assert fingerprint(ROOT/p)==h
    for p,h in protocol['code_sha256'].items():assert fingerprint(ROOT/p)==h
    assert all(scene_digest(ROOT/p)==h for p,h in sources.items())
    report=dict(passed=True,encoder_forwards=0,optimizer_steps=0,source_and_inputs_unchanged=True,
        saved_metric_max_difference=max(parity),saved_records=120,dual_points=len(points),geometry_points=len(geometry),
        gt_epipolar_max_px=max(r['gt_epipolar_distance_px'] for r in geometry if 'gt_epipolar_distance_px' in r),
        dual_visual_scene=scene,elapsed_s=time.monotonic()-start)
    write_json(OUT/'report.json',report);write_json(OUT/'status.json',dict(state='completed'))
    print(json.dumps(report,indent=2));print(json.dumps(geo_summary,indent=2));print(json.dumps(common,indent=2))
    print(json.dumps([r for r in dual_summary if r['part'] in ('all','body','tail')],indent=2))


if __name__=='__main__':main()
