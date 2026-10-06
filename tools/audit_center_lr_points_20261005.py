"""Frozen matched center-LR endpoint audit: point candidates are not poses."""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,traceback,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest
from train_dense_center_warmup_20261005 import read,write,now,score as center_score
from audit_local_peak_integrity_20261005 import score
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,decode_centers
from mousegeoforge.models.regional_peak_candidates import regional_top_peaks
C=ROOT/'output/local_grouping_cycle_20261005';INPUT=ROOT/'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'
OUT=C/'dense_center_lr_point_audit_v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--loss-pair',action='store_true');modes.add_argument('--loss-pair9216',action='store_true');args=parser.parse_args()
    loss_pair=args.loss_pair or args.loss_pair9216
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    sources={'base':('dense_center_continue4096_v1',4096),'low':('dense_center_lr_low_6144_v1',6144),'high':('dense_center_lr_high_6144_v1',6144)}
    pair=['low','high']
    if args.loss_pair:
        OUT=C/'dense_center_loss_point_audit_v1';pair=['focal','gaussian']
        sources={'base':('dense_center_lr_high_6144_v1',6144),'focal':('dense_center_loss_focal_7168_v1',7168),'gaussian':('dense_center_loss_gaussian_7168_v1',7168)}
    if args.loss_pair9216:
        OUT=C/'dense_center_loss9216_point_audit_v1';pair=['focal','gaussian']
        sources={'prior_focal':('dense_center_loss_focal_7168_v1',7168),'prior_gaussian':('dense_center_loss_gaussian_7168_v1',7168),
                 'focal':('dense_center_loss_focal_9216_v1',9216),'gaussian':('dense_center_loss_gaussian_9216_v1',9216)}
    for name,step in sources.values():assert json.loads((C/name/'status.json').read_text())['stage']=='complete'
    protocols={a:json.loads((C/n/'protocol.json').read_text()) for a,(n,s) in sources.items()}
    for key in ['schedule','rotations','source_sha256','supervision_sha256','evaluation_steps']+(['common_6144_sha256'] if args.loss_pair9216 else ['resume_sha256']):
        assert protocols[pair[0]][key]==protocols[pair[1]][key],key
    if args.loss_pair9216:
        for arm in pair:
            prior=protocols['prior_'+arm];current=protocols[arm]
            assert current['resume_sha256']==sha(C/sources['prior_'+arm][0]/'center_step7168.pt')
            assert current['common_6144_sha256']==prior['resume_sha256']
            assert current['schedule'][:7168]==prior['schedule'] and current['rotations'][:7168]==prior['rotations']
            assert current['loss']==prior['loss'] and current['lrs']==prior['lrs']
    if loss_pair:assert protocols['focal']['lrs']==protocols['gaussian']['lrs']==[3e-4,1e-4,1e-5]
    else:assert protocols['low']['lrs']==[3e-4,1e-4,1e-6] and protocols['high']['lrs']==[3e-4,1e-4,1e-5]
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);examples=OUT/'examples';examples.mkdir();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='center_lr_point_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    indices=fixed+[i for i,r in rows.items() if r['metadata']['stage']=='calibration']
    selected=list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i in indices if rows[i]['metadata']['stage']=='calibration' and rows[i]['metadata']['kind']=='real' and rows[i]['metadata']['physical_bottom']]))
    assert all(rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3'] for i in indices)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);colors=['#ff4422','#00bbff','#ee44dd','#99ff33','#ffbb00'];ids=[1,15,16,17,18]
    summaries={};hashes={};center_summaries={}
    sup={r['index']:r for r in read(C/'instance_supervision_audit_v1/evaluation_instance_targets.pt')}
    train_distances=[]
    for i,r in rows.items():
        if r['metadata']['stage']=='fit' and int(sup[i]['instance_valid'].sum())==2:
            xy=sup[i]['mask_centroid'][sup[i]['instance_valid']]*r['pixels'];train_distances.append(float((xy[0]-xy[1]).norm()))
    write(OUT/'protocol.json',dict(loss_pair=loss_pair,continuation9216=args.loss_pair9216,source_pairs=pair,diagnostic_radius=[8,32] if args.loss_pair else [8],
          radius_reason='32px is2x preexistingGaussian sigma16, declared beforeevaluation; NOTtunedonCAL. Diagnostic only, decoderdefault8 unchanged.',
          train_dual_centroid_distance_min=min(train_distances),train_dual_centroid_distance_median=float(np.median(train_distances)),
          training=False,deadline=plan['training_stop_utc'],code_sha256=sha(Path(__file__))))
    for arm,(name,step) in sources.items():
        status('loading',arm=arm);source=C/name/f'center_step{step:04d}.pt';hashes[arm]=sha(source);cp=read(source);backbone,_=build_backbone()
        model=MixedLocalEvidence(backbone,adapt_backbone=False).cuda();model.backbone.load_state_dict(cp['backbone']);model.readout.load_state_dict(cp['readout']);model.eval()
        center=DenseInstanceCenters().cuda();center.load_state_dict(cp['center']);center.eval();del cp
        before=(digest(model),digest(center),buffers_digest(backbone));old=read(C/name/f'step{step:04d}_predictions.pt');details=[];predictions={};center_details=[]
        with torch.no_grad():
            for n,i in enumerate(indices,1):
                if now()>=deadline:raise TimeoutError('Audit cutoff')
                r=rows[i];hw=manifest[i]['image_hw'];out=model(load_rgb(manifest[i]).cuda(),hw);center_out=center(out,hw);decoded=decode_centers(center_out,hw)
                assert all(torch.equal(v.cpu(),old[i]['decoded'][k]) for k,v in decoded.items()),(arm,i,'centerreplay')
                for radius in ([8,32] if args.loss_pair else [8]):
                    decoded_radius={k:v.cpu() for k,v in decode_centers(center_out,hw,radius_px=radius).items()}
                    metrics=center_score(decoded_radius,sup[i],hw)
                    xy=decoded_radius['centers'][decoded_radius['instance_active']]*r['pixels'];gt=sup[i]['mask_centroid'][sup[i]['instance_valid']]*r['pixels']
                    near=int((torch.cdist(xy,gt).min(1).values<=64).sum()) if len(xy) and len(gt) else 0
                    center_details.append(dict(index=i,metadata=r['metadata'],radius=radius,metrics=metrics,near_gt_surplus=near-metrics['tp64'],far_fp=len(xy)-near))
                p=regional_top_peaks(out,hw);predictions[i]={k:p[k] for k in ['xy','available','probability']}
                details.append(dict(index=i,metadata=r['metadata'],points=score(p,r),feet=score(p,r,True)))
                if i in selected:
                    rgb=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=rgb.size;canvas=Image.new('RGB',(2*w,h+60),'white');canvas.paste(rgb,(0,35));canvas.paste(rgb,(w,35));d=ImageDraw.Draw(canvas)
                    d.text((4,4),f'{arm} index{i}: eligible GT | regional point candidates (NOT instances)',fill='black',font=font)
                    for q,k in r['eligible'].nonzero().tolist():
                        x,y=(r['target'][q,k]*r['pixels']).tolist();y+=35;d.line((x-5,y,x+5,y),fill=colors[k],width=2);d.line((x,y-5,x,y+5),fill=colors[k],width=2);d.text((x+5,y),str(ids[k]),fill=colors[k],font=font)
                    for k in range(5):
                        for rank in range(6):
                            if not p['available'][k,rank]:continue
                            x,y=(p['xy'][k,rank]*r['pixels']).tolist();x+=w;y+=35;rad=5 if rank==0 else 2;d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=colors[k],width=2)
                            if rank==0:d.text((x+5,y),str(ids[k]),fill=colors[k],font=font)
                    d.text((4,h+38),'Large=Top1, small=Top2-6. Colors indicate bodyparts, not mouse IDs.',fill='black',font=font);canvas.save(examples/f'{arm}_{i}.jpg',quality=93)
                if n%20==0:status('auditing',arm=arm,images=n,total=len(indices));print(arm,n,flush=True)
        assert before==(digest(model),digest(center),buffers_digest(backbone)) and sha(source)==hashes[arm]
        summaries[arm]={}
        for kind in ['real','single','dual','real_bottom_feet']:
            ds=[d for d in details if d['metadata']['stage']=='calibration' and (d['metadata']['kind']==kind or (kind=='real_bottom_feet' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']))]
            metrics=[d['feet' if kind=='real_bottom_feet' else 'points'] for d in ds];z={}
            for metric in ['nearest','assigned','top1']:
                v=np.array([x for m in metrics for x in m[metric]]);z[metric+'_n']=len(v);z[metric+'_mean_px']=float(v.mean()) if len(v) else None;z[metric+'_recall16']=float((v<=16).mean()) if len(v) else None
            z['missing_points']=sum(m['missing'] for m in metrics);summaries[arm][kind]=z
        write(OUT/f'{arm}_details.json',details);torch.save(predictions,OUT/f'{arm}_predictions.pt');del model,center,backbone,old;torch.cuda.empty_cache()
        write(OUT/f'{arm}_center_details.json',center_details);center_summaries[arm]={}
        for radius in ([8,32] if args.loss_pair else [8]):
            center_summaries[arm][radius]={}
            for kind in ['real','single','dual','real_bottom']:
                ds=[d for d in center_details if d['radius']==radius and d['metadata']['stage']=='calibration' and (d['metadata']['kind']==kind or (kind=='real_bottom' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']))]
                center_summaries[arm][radius][kind]=dict(images=len(ds),count_accuracy=float(np.mean([d['metrics']['count_correct'] for d in ds])),
                    tp64=sum(d['metrics']['tp64'] for d in ds),fp64=sum(d['metrics']['fp64'] for d in ds),fn64=sum(d['metrics']['fn64'] for d in ds),near_gt_surplus=sum(d['near_gt_surplus'] for d in ds),far_fp=sum(d['far_fp'] for d in ds))
    assert all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in indices)
    result=dict(summary=summaries,center_summary=center_summaries,source_hashes=hashes,paired_protocol_exact=True,center_replay_exact=True,images=len(indices),weights_bn_input_unchanged=True,training=False,interpretation='Candidateoracle not poseaccuracy. Top1 onlyfor singleinstances; dual uses pertype one-to-one diagnosticmatching. Missingcandidates includedwithdiagonalpenalty. NoDEV/locked sources. Radius32optional diagnosis NOTdefaultchange or independenttest; nearGTsurplus isgeometricduplicateproxy notproof.')
    write(OUT/'results.json',result)
    page=['<!doctype html><meta charset="utf-8"><title>匹配对照关键点候选审计</title><style>body{font:18px system-ui;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>冻结匹配对照终点：关键点候选审计</h1><p>GT只用于评价。颜色是关键点类型不是动物身份。候选oracle不是最终姿态准确率；单鼠Top1另报，双鼠按同类型一对一候选匹配诊断。</p><pre>'+json.dumps(result,indent=2)+'</pre>']
    for i in selected:
        for arm in sources:page.append(f'<img src="examples/{arm}_{i}.jpg">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True);status('complete');state.update(stage='center_lr_point_audit_complete',active_job=None,active_pid=None,next='Review matchedLR center/pointmetrics andallbottomimages before nextoptimization. No automaticpromotion.',updated_utc=now().isoformat());state['completed'].append('center_lr_point_audit');write(C/'state.json',state);print('COMPLETE',json.dumps(summaries),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
