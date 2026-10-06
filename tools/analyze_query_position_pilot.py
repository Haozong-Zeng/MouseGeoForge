"""Fixed final-budget gates, box/instance diagnostics and all-active-query visuals."""
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.nn import functional as F
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint
from analyze_cylinder_view_balance import gate_results
from mousegeoforge.retention_geometry_audit import matched_views

OUT=ROOT/'output/query_position_seed17_v1'


def main():
    if (OUT/'acceptance.json').exists():raise RuntimeError('Preserve existing analysis')
    torch.set_num_threads(2)
    assert json.loads((OUT/'verification.json').read_text())['passed']
    results=json.loads((OUT/'results.json').read_text());arms={'initial':results['baseline'],**results['arms']}
    baseline=json.loads((ROOT/'output/cylinder_source_mix_v1/results.json').read_text())['baseline']
    control=json.loads((ROOT/'output/cylinder_view_balance_v1/results.json').read_text())['arms']['side_control']
    gates={a:gate_results(dict(config=results['config'],baseline=baseline,arms=dict(side_control=control,balanced_views=v)))['balanced_views'] for a,v in arms.items()}
    assert all(g['total']==17 for g in gates.values())
    stats=[];hashes={};quality={};coords=[]
    for a,values in arms.items():
        quality[a]=json.loads((OUT/f'{a}_instance_quality.json').read_text())
        for i,r in enumerate(values['records']):
            p=OUT/a/f'prediction_{i:03d}.pt';hashes[str(p)]=fingerprint(p)
            saved=torch.load(p,map_location='cpu',weights_only=True)
            for v in matched_views(saved):
                stats.append(dict(arm=a,group=r['group'],index=i,frame=saved['record']['frame'],view=v['view'],instance=v['instance'],iou=v['iou'],found=v['found']))
                if r['group']=='old_dual_pair':
                    for part,indices in (('body',list(range(7))+list(range(14,23))),('tail',list(range(7,14)))):
                        valid=v['valid'][indices];e=v['errors'][indices][valid]
                        coords.extend(dict(arm=a,part=part,nme=float(x)) for x in e)
    summary=[]
    for a,values in arms.items():
        for group in values['summary']:
            rows=[r for r in stats if r['arm']==a and r['group']==group]
            qq=[q for q in quality[a] if q['group']==group]
            summary.append(dict(arm=a,group=group,mean_box_iou=float(np.mean([r['iou'] for r in rows])),
                found=sum(r['found'] for r in rows),gt_instances=len(rows),
                active_queries=sum(q['active_queries'] for q in qq),
                duplicate_mask_views=sum(q['duplicate_strong_mask'] for q in qq)))
    def dual(a):return next(r for r in summary if r['arm']==a and r['group']=='old_dual_pair')
    off=arms['query_position_off']['summary']['old_dual_pair'];on=arms['query_position_on']['summary']['old_dual_pair']
    criteria=dict(all17_gates=gates['query_position_on']['passed'],
        pck_gain_at_least_01=on['detection_aware_pck_0.1']-off['detection_aware_pck_0.1']>=.01,
        coordinate_nme_ratio_at_most_102=on['box_matched_nme']<=off['box_matched_nme']*1.02,
        tp_not_lower=dual('query_position_on')['found']>=dual('query_position_off')['found'],
        fp_not_higher=sum(x['active_queries']-x['found'] for x in summary if x['arm']=='query_position_on')<=sum(x['active_queries']-x['found'] for x in summary if x['arm']=='query_position_off'),
        duplicate_views_not_higher=sum(x['duplicate_mask_views'] for x in summary if x['arm']=='query_position_on')<=sum(x['duplicate_mask_views'] for x in summary if x['arm']=='query_position_off'))
    report=dict(passed_analysis=True,gates=gates,box_instance_summary=summary,
        dual_coordinate_parts=[dict(arm=a,part=p,nme=float(np.mean([x['nme'] for x in coords if x['arm']==a and x['part']==p]))) for a in arms for p in ('body','tail')],
        pck_gain=on['detection_aware_pck_0.1']-off['detection_aware_pck_0.1'],
        dual_nme_ratio=on['box_matched_nme']/off['box_matched_nme'],criteria=criteria,
        seed29_eligible=all(criteria.values()),production_promoted=False,
        decision='Second seed eligible after visual inspection.' if all(criteria.values()) else 'Second new mechanism did not meet fixed practical criteria: stop new training in this ten-hour cycle; no third mechanism or extension.',
        source_sha256={str(OUT/'results.json'):fingerprint(OUT/'results.json'),**hashes},
        analysis_code_sha256=fingerprint(ROOT/'tools/analyze_query_position_pilot.py'),
        limitations=['Reused synthetic development, no real Cylinder evidence.',
                    'Native-grid mask duplicate statistics are diagnostic; unchanged full-image metrics remain authoritative.',
                    'OFF and ON both receive800 warm-start updates; gain must be compared to OFF, not initial.'])
    write_json(OUT/'acceptance.json',report)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for a in results['arms']:
        h=json.loads((OUT/a/'history.json').read_text())
        axes[0].plot(np.arange(50,801),np.convolve([r['pose_loss'] for r in h],np.ones(50)/50,mode='valid'),label=a)
    axes[0].set(title='50-slot pose loss (same train poses)',xlabel='step');axes[0].legend()
    axes[1].bar(list(arms),[v['summary']['old_dual_pair']['detection_aware_pck_0.1'] for v in arms.values()])
    axes[1].set(title='Dual detection-aware PCK; fixed final800',ylim=(0,1))
    fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)
    ids=[0,7,9,13];fig,axes=plt.subplots(4,3,figsize=(15,11),layout='constrained')
    for row,i in enumerate(ids):
        for col,a in enumerate(arms):
            p=torch.load(OUT/a/f'prediction_{i:03d}.pt',map_location='cpu',weights_only=True)
            masks=torch.load(OUT/a/f'mask_visual_{i:03d}.pt',map_location='cpu',weights_only=True)
            r=p['record'];v=r['physical_roles'].index('top');h,w=r['image_hw'];scale=np.array([w,h])
            rgb=np.array(Image.open(ROOT/r['scene']/'blended_images'/f'{r["views"][v]:08d}.jpg').resize((w,h)))
            ax=axes[row,col];ax.imshow(rgb)
            for m in torch.where(p['targets']['instance_valid'][0,v])[0]:
                b=p['targets']['boxes_cxcywh'][0,v,m].numpy()
                ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor='lime',lw=1))
            active=torch.where(p['predictions']['instance_objectness_logits'][0,v].sigmoid()>=.5)[0]
            palette=plt.get_cmap('tab10')
            for q in active:
                color=np.array(palette(int(q))[:3]);mask=F.interpolate(masks['mask_logits'][0,v,q][None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()>0
                ax.imshow(np.dstack([np.broadcast_to(color,(h,w,3)),mask*.4]))
                b=p['predictions']['instance_boxes_cxcywh'][0,v,q].numpy()
                ax.add_patch(Rectangle((b[:2]-b[2:]/2)*scale,*(b[2:]*scale),fill=False,edgecolor=color,lw=1.2))
                ax.text(b[0]*w,b[1]*h,f'q{int(q)}',color=color,fontsize=9)
            ax.set_title(f'{a} / frame{r["frame"]} top / {len(active)} queries',fontsize=9);ax.axis('off')
    fig.suptitle('All score>=.5 queries; query color is not identity. GT boxes green; no GT-guided selection.')
    fig.savefig(OUT/'fixed_dual_masks.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(gates={a:(g['passed_count'],g['total']) for a,g in gates.items()},dual_boxes=[dual(a) for a in arms],criteria=criteria,pck_gain=report['pck_gain'],decision=report['decision']),indent=2))


if __name__=='__main__':main()


