"""Frozen full-frame scale diagnostic; all errors remapped to original input px.

No optimizer, no crops/GT/count supplied to forward, no confidence tuning.
This tests an object-scale/context hypothesis, not a deployable improvement.
"""
import os
os.environ['DLClight']='True'
import sys,json,datetime as dt,fcntl,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone,load_rgb,sha,digest
from train_mixed_local_backbone_20261005 import buffers_digest
from train_dense_center_warmup_20261005 import score,read,write,now
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.dense_instance_centers import DenseInstanceCenters,decode_centers
C=ROOT/'output/local_grouping_cycle_20261005';OLD=ROOT/'output/singleview_fusion_cycle_20261005'
INPUT=OLD/'mixed_local_fulltrain_rgb_inputs_v1';BASE=C/'dense_center_continue1024_v1'
OUT=C/'dense_center_scale_audit_v1'

def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    assert json.loads((BASE/'status.json').read_text())['stage']=='complete'
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((C/'state.json').read_text());state.update(stage='dense_center_scale_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading')
    rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())}
    sup={r['index']:r for r in read(C/'instance_supervision_audit_v1/evaluation_instance_targets.pt')}
    fixed=[next(i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k) for k in ['real','single','dual']]
    indices=fixed+[i for i,r in rows.items() if r['metadata']['stage']=='calibration']
    selected=list(dict.fromkeys(fixed+[next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k) for k in ['real','single','dual']]+[i for i in indices if rows[i]['metadata']['stage']=='calibration' and rows[i]['metadata']['kind']=='real' and rows[i]['metadata']['physical_bottom']]))
    for i in indices:assert rows[i]['metadata'].get('source') not in ['FR2','AZR2L2','GR2','UR2','UR3']
    source=BASE/'center_step1024.pt';source_hash=sha(source);cp=read(source);backbone,_=build_backbone()
    model=MixedLocalEvidence(backbone,adapt_backbone=False).cuda();model.backbone.load_state_dict(cp['backbone'],strict=True);model.readout.load_state_dict(cp['readout'],strict=True);model.eval()
    detector=DenseInstanceCenters().cuda();detector.load_state_dict(cp['center'],strict=True);detector.eval();del cp
    before=(digest(model),digest(detector),buffers_digest(model.backbone));oldpred=read(BASE/'step1024_predictions.pt')
    predictions={};details=[];scales={'native':1.,'half':.5,'quarter':.25}
    with torch.no_grad():
        for n,i in enumerate(indices,1):
            if now()>=deadline:raise TimeoutError('Audit cutoff')
            x=load_rgb(manifest[i]).cuda();h,w=manifest[i]['image_hw'];predictions[i]={}
            for name,scale in scales.items():
                if scale==1.:image=x;hw=(h,w)
                else:
                    nh,nw=round(h*scale),round(w*scale)
                    resized=torch.nn.functional.interpolate(x[...,:h,:w],size=(nh,nw),mode='bilinear',align_corners=False,antialias=True)
                    black=-x.new_tensor([.485,.456,.406])/x.new_tensor([.229,.224,.225])
                    image=black[None,:,None,None].expand(1,3,nh+(-nh)%32,nw+(-nw)%32).clone();image[...,:nh,:nw]=resized;hw=(nh,nw)
                output=detector(model(image,hw),hw);decoded={k:v.cpu() for k,v in decode_centers(output,hw).items()}
                if name=='native':
                    assert all(torch.equal(v,oldpred[i]['decoded'][k]) for k,v in decoded.items()),('native parity',i)
                predictions[i][name]=decoded
                # Cell-center normalized coordinates are preserved under whole
                # frame resizing. Score in ORIGINAL h,w, not shrunken pixels.
                details.append(dict(index=i,metadata=rows[i]['metadata'],scale=name,inference_hw=list(hw),metrics=score(decoded,sup[i],(h,w))))
            if n%10==0:status('auditing',images=n,total=len(indices));print('AUDIT',n,flush=True)
    summary={}
    for name in scales:
        summary[name]={}
        for stage in ['fit','calibration']:
            summary[name][stage]={}
            for kind in ['real','single','dual','real_bottom']:
                dd=[d for d in details if d['scale']==name and d['metadata']['stage']==stage and
                    (d['metadata']['kind']==kind or (kind=='real_bottom' and d['metadata']['kind']=='real' and d['metadata']['physical_bottom']))]
                if not dd:continue
                summary[name][stage][kind]=dict(images=len(dd),count_accuracy=float(np.mean([d['metrics']['count_correct'] for d in dd])),
                    tp64=sum(d['metrics']['tp64'] for d in dd),fp64=sum(d['metrics']['fp64'] for d in dd),fn64=sum(d['metrics']['fn64'] for d in dd),
                    center_error_original_px=float(np.mean([v for d in dd for v in d['metrics']['error']])),
                    candidate_oracle_original_px=float(np.mean([v for d in dd for v in d['metrics']['candidate_oracle']])))
    assert before==(digest(model),digest(detector),buffers_digest(model.backbone))
    assert sha(source)==source_hash and all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in indices)
    result=dict(summary=summary,source_sha256=source_hash,images=len(indices),native_replay_exact=True,model_bn_sources_unchanged=True,
                training=False,interpretation='Read-only whole-frame scale diagnostic. Coordinates/errors expressed in original768longsideinput. No GTcrop/count/center, no thresholdchange. Smallerimagesalterbothcontextandantialiasing; notproof ofsinglecause or deployableposegain.')
    write(OUT/'results.json',result);write(OUT/'details.json',details);torch.save(predictions,OUT/'predictions.pt')
    examples=OUT/'examples';examples.mkdir();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for i in selected:
        rgb=Image.open(ROOT/manifest[i]['image']).convert('RGB');w,h=rgb.size
        for name in scales:
            canvas=Image.new('RGB',(2*w,h+40),'white');canvas.paste(rgb,(0,40));canvas.paste(rgb,(w,40));d=ImageDraw.Draw(canvas);decoded=predictions[i][name]
            d.text((4,4),f'{name} index{i}: mask centroid GT | frozen predictedcenters, original pixelcoordinates, count {int(decoded["instance_active"].sum())}',font=font,fill='black')
            for xy in sup[i]['mask_centroid'][sup[i]['instance_valid']]:
                a,b=(xy*torch.tensor([w,h])).tolist();d.line((a-7,b+40,a+7,b+40),fill='cyan',width=3);d.line((a,b+33,a,b+47),fill='cyan',width=3)
            for q in decoded['available'].nonzero().flatten().tolist():
                a,b=(decoded['centers'][q]*torch.tensor([w,h])).tolist();a+=w;color='lime' if decoded['instance_active'][q] else 'orange'
                d.ellipse((a-5,b+35,a+5,b+45),outline=color,width=2);d.text((a+6,b+40),f'{float(decoded["object_probability"][q]):.3f}',font=font,fill=color)
            canvas.save(examples/f'{name}_{i}.jpg',quality=93)
    page=['<!doctype html><meta charset="utf-8"><title>中心检测尺度诊断</title><style>body{font:18px system-ui;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>冻结1024步中心检测器：整图缩放诊断</h1><p>无训练、无GT裁剪或数量输入、阈值0.5不变；所有误差换回原始768长边输入像素。不是通过缩小分辨率把像素误差数字缩小。</p><pre>'+json.dumps(result,indent=2)+'</pre>']
    for i in selected:
        for name in scales:page.append(f'<img src="examples/{name}_{i}.jpg">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name;dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True)
    status('complete');state.update(stage='dense_center_scale_audit_complete',active_job=None,active_pid=None,next='Inspect scale diagnostic and originalpixelmetrics; chooseone evidence-backed detector adjustment, noGTforward/defaultpromotion.',updated_utc=now().isoformat());state['completed'].append('dense_center_scale_audit');write(C/'state.json',state);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
