"""Read-only provenance/parity audit of existing FIT/CAL instance supervision."""
import sys,json,datetime as dt,os,traceback,collections,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import label
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
C=ROOT/'output/local_grouping_cycle_20261005';OLD=ROOT/'output/singleview_fusion_cycle_20261005'
INPUT=OLD/'mixed_local_fulltrain_rgb_inputs_v1';OUT=C/'instance_supervision_audit_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005/instance_supervision_audit_v1')
KP=[0,14,15,16,17]
def now():return dt.datetime.now(dt.timezone.utc)
def read(p):return torch.load(p,weights_only=False,map_location='cpu')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
    plan=json.loads((C/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));assert now()<deadline
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    state=json.loads((C/'state.json').read_text());state.update(stage='instance_supervision_audit_running',active_job=str(OUT.relative_to(ROOT)),active_pid=os.getpid());write(C/'state.json',state)
    def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=now().isoformat(),**kw))
    status('loading');rows={r['index']:r for r in read(INPUT/'evaluation_only.pt')};manifest={r['index']:r for r in json.loads((INPUT/'inference_manifest.json').read_text())};groups=collections.defaultdict(list)
    for i,r in rows.items():
        m=r['metadata'];assert m['stage'] in ['fit','calibration'];assert m.get('source') not in ['GR2','UR2','UR3','FR2','AZR2L2'];groups[(m['scene'],tuple(manifest[i]['image_hw']),m['kind'])].append(i)
    selected={next(i for i,r in rows.items() if r['metadata']['stage']==s and r['metadata']['kind']==k) for s in ['fit','calibration'] for k in ['real','single','dual']}
    details=[];targets=[];sources={};examples=[];counts=collections.Counter();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for number,((scene,hw,kind),indices) in enumerate(groups.items(),1):
        if now()>=deadline:raise TimeoutError('Input audit deadline')
        views=sorted({rows[i]['metadata']['view'] for i in indices});record=dict(scene=scene,views=views,image_hw=list(hw),kind=kind)
        item=load_item(record);t=instance_targets_from_batch(item['batch']);sources.update(item['source_hashes'])
        for i in indices:
            r=rows[i];v=views.index(r['metadata']['view']);h,w=hw
            rgb=(item['batch']['images'][0,v].permute(1,2,0)*255).round().byte().numpy()
            assert sha(ROOT/manifest[i]['image'])==manifest[i]['sha256']
            assert np.array_equal(rgb,np.asarray(Image.open(ROOT/manifest[i]['image']).convert('RGB'))),('RGB parity',i)
            known=t['keypoint_valid'][0,v,:,KP]&t['instance_valid'][0,v,:,None]
            assert torch.equal(known,r['coordinate_valid'])
            assert torch.allclose(t['keypoint_xy'][0,v,:,KP][known],r['target'][known],atol=1e-6,rtol=0)
            masks=t.get('masks');mask_valid=t.get('mask_valid',torch.zeros_like(t['instance_valid']))[0,v].bool()
            boxes=t['boxes_cxcywh'][0,v].clone();bv=t.get('box_valid',torch.zeros_like(t['instance_valid']))[0,v].bool();iv=t['instance_valid'][0,v].bool()
            centers=torch.zeros(len(iv),2);center_valid=torch.zeros_like(iv);instance_details=[]
            for q in iv.nonzero().flatten().tolist():
                counts[r['metadata']['stage']+'_'+kind+'_instances']+=1
                dd=dict(instance=q,box_valid=bool(bv[q]),mask_valid=bool(mask_valid[q]),eligible_points=int(r['eligible'][q].sum()))
                if mask_valid[q] and masks is not None:
                    mask=masks[0,v,q].bool();assert mask.shape==(h,w);yy,xx=mask.nonzero(as_tuple=True);assert len(xx)>0
                    centers[q]=torch.tensor([(xx.float().mean()+.5)/w,(yy.float().mean()+.5)/h]);center_valid[q]=True
                    cc,n=label(mask.numpy());sizes=np.bincount(cc.ravel())[1:]
                    box=torch.tensor([(xx.min()+xx.max()+1)/2/w,(yy.min()+yy.max()+1)/2/h,(xx.max()-xx.min()+1)/w,(yy.max()-yy.min()+1)/h])
                    dd.update(mask_area=int(mask.sum()),area_fraction=float(mask.float().mean()),components=n,largest_fraction=float(sizes.max()/sizes.sum()),mask_centroid=centers[q].tolist(),mask_box=box.tolist(),existing_box=boxes[q].tolist(),box_max_delta=float((box-boxes[q]).abs().max()))
                    counts[r['metadata']['stage']+'_'+kind+'_mask_valid']+=1
                else:counts[r['metadata']['stage']+'_'+kind+'_mask_missing']+=1
                instance_details.append(dd)
            extra=dict(index=i,metadata=r['metadata'],instance_valid=iv,boxes=boxes,box_valid=bv,mask_valid=mask_valid,mask_centroid=centers,centroid_valid=center_valid,detection_annotation_complete=t.get('detection_annotation_complete',torch.zeros(1,len(views),dtype=torch.bool))[0,v].clone())
            targets.append(extra);details.append(dict(index=i,metadata=r['metadata'],instances=instance_details))
            if i in selected:
                arr=rgb.astype(float)
                if masks is not None:
                    for q in iv.nonzero().flatten().tolist():
                        if mask_valid[q]:
                            mask=masks[0,v,q].bool().numpy();color=np.array([255,50,50] if q==0 else [40,180,255]);arr[mask]=arr[mask]*.65+color*.35
                im=Image.fromarray(arr.astype('uint8'));d=ImageDraw.Draw(im)
                for q in iv.nonzero().flatten().tolist():
                    color='yellow' if q==0 else 'cyan';cx,cy,bw,bh=(boxes[q]*torch.tensor([w,h,w,h])).tolist()
                    if bv[q]:d.rectangle((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2),outline=color,width=2)
                    if center_valid[q]:
                        x,y=(centers[q]*torch.tensor([w,h])).tolist();d.line((x-8,y,x+8,y),fill=color,width=3);d.line((x,y-8,x,y+8),fill=color,width=3)
                    for k in r['eligible'][q].nonzero().flatten().tolist():
                        x,y=(r['target'][q,k]*r['pixels']).tolist();d.ellipse((x-3,y-3,x+3,y+3),outline='white',width=2)
                canvas=Image.new('RGB',(w,h+50),'white');canvas.paste(im,(0,40));ImageDraw.Draw(canvas).text((5,5),f'index{i}: approved mask, box, mask-centroid; GT only',font=font,fill='black');name=f'{r["metadata"]["stage"]}_{kind}_{i}.jpg';canvas.save(OUT/name,quality=93);examples.append(dict(index=i,file=name))
        if number%10==0:status('auditing',scenes=number,total_scenes=len(groups),images=len(targets));print('AUDIT',number,len(targets),flush=True)
    assert len(targets)==1055 and all(sha(ROOT/p)==digest for p,digest in sources.items())
    torch.save(targets,OUT/'evaluation_instance_targets.pt');write(OUT/'details.json',details);write(OUT/'source_hashes.json',sources);write(OUT/'examples.json',examples)
    result=dict(counts=dict(counts),images=len(targets),all_rgb_and_point_parity=True,sources_unchanged=True,center_definition='Centroid of approved visible projected mask including whatever its mask encodes; NOT anatomicalbodycenter, NOTmeanofavailablepoints. No useasGTuntilmasksemanticsreview.',scope='Existing1055selectedFIT/CAL pool, selectedhistorically withbaselinequalifiedknownpoints; NOTunbiasedall-sourcebenchmark. NoDEV/lockedread. Allinstancesfromfullimageincluded.',new_training=False)
    write(OUT/'results.json',result);status('complete');DEST.mkdir(parents=True,exist_ok=True)
    page=['<!doctype html><meta charset="utf-8"><title>实例监督来源审计</title><style>body{font:18px system-ui;margin:30px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>实例掩码、框及几何中心：不是解剖中心</h1><pre>'+json.dumps(result,indent=2)+'</pre>']
    for e in examples:page.append('<img src="'+e['file']+'">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for p in OUT.iterdir():
        if p.suffix in ['.json','.jpg','.html']:shutil.copy2(p,DEST/p.name)
    state.update(stage='instance_supervision_audit_complete',active_job=None,active_pid=None,updated_utc=now().isoformat(),next='Reviewmasksemantics/boxfallbackandfirstsixvisualsbeforechoosingcenterorquerygrouping; preserveunknownpoints. Implementmatchedgroupingnext, notGTforward.');state['completed'].append('instance_supervision1055_source_audit');write(C/'state.json',state);print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc(),utc=now().isoformat()));write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
