"""All-query frozen-pose audit for pure-V1 background detection control."""
import collections,datetime as dt,fcntl,html,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
from run_surface_curve_pilot_20261003 import load_item
OLD=ROOT/'output/core23_spatial_cycle_20261004';CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';DET=CYCLE/'single_detector_background_control_v1';OUT=CYCLE/'single_detector_pose_audit_v1';ARMS=['baseline','control','negative04']

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (DET/'results.json').exists() and not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='auditing_pure_single_detector_frozen_pose',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');rows=[];source=OLD/'detection_features_v1/manifest.json'
    for rec in json.loads(source.read_text())['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];rows.extend(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['observations'])
    with np.load(OLD/'full_context_pose_control_v1/predictions.npz') as a:indices=a['image_indices'].tolist();gold={s:a[f'full_seed{s}_single'].copy() for s in [17,29,43]}
    meta=[rows[i]['metadata'] for i in indices];assert len(indices)==112 and all(m['record']['role']=='validation' and m['record'].get('session') not in ['GR2','UR2','UR3'] for m in meta)
    data={k:torch.stack([rows[i][k] for i in indices]).cuda() for k in ['boxes','box_valid','instance_valid','gt_xy','point_valid','pixels']};valid=data['box_valid']&data['instance_valid'];n=len(indices)
    for j,m in enumerate(meta):
        if m['expected_instances']>1:data['point_valid'][j,:,7:14]=False
    lookup={i:j for j,i in enumerate(indices)};memories=[None]*n;tokens=torch.zeros(n,6,256,device='cuda')
    for rec in json.loads((OLD/'single_view_validation_context_v1/manifest.json').read_text())['records']:
        assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);j=lookup[rec['image_index']];assert raw['metadata']==meta[j];memories[j]=raw['memory'].cuda();tokens[j]=raw['instance_tokens'].cuda()
    assert all(m is not None for m in memories)
    det={}
    with np.load(OLD/'detection_view_context_control_v1/predictions.npz') as a:
        for s in [17,29,43]:det[(s,'baseline')]={k:torch.tensor(a[f'mixed_seed{s}_single__{k}'][indices],device='cuda') for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
    with np.load(DET/'predictions.npz') as a:
        for s in [17,29,43]:
            for arm in ARMS[1:]:det[(s,arm)]={k:torch.tensor(a[f'{arm}_seed{s}__{k}'][indices],device='cuda') for k in ['instance_boxes_cxcywh','instance_objectness_logits']}
    @torch.no_grad()
    def predict(model,boxes):
        parts=[None]*n;groups=collections.defaultdict(list)
        for j,m in enumerate(memories):groups[tuple(m.shape[-2:])].append(j)
        for ids in groups.values():
            for start in range(0,len(ids),8):
                group=ids[start:start+8];p=model(tokens[group],torch.stack([memories[j] for j in group]),boxes[group])['xy']
                for k,j in enumerate(group):parts[j]=p[k]
        return torch.stack(parts)
    def evaluate(xy,p,baseline_matches):
        matches=match_detection(p,data['boxes'],valid);accum=collections.defaultdict(lambda:dict(error=[],same_query_error=[],points=0,pck16=0,pck32=0));counts=collections.defaultdict(lambda:dict(images=0,tp=0,gt=0,predictions=0,query_switches=0))
        for i,m in enumerate(meta):
            kind='real' if m['record']['kind']=='real' else 'synthetic';cam='cam'+chr(65+int(m['view']));subtype='dual' if m['expected_instances']>1 else 'single';qs,ms=matches[i];bq={int(t):int(q) for q,t in zip(*baseline_matches[i])};scores=p['instance_objectness_logits'][i].sigmoid();tp=switches=0
            for q,target in zip(qs.tolist(),ms.tolist()):
                hit=bool(scores[q]>=.5 and box_iou(p['instance_boxes_cxcywh'][i,q:q+1],data['boxes'][i,target:target+1])[0,0]>=.5);tp+=hit;switches+=int(q!=bq[target]);good=data['point_valid'][i,target].cpu()
                err=((xy[i,q]-data['gt_xy'][i,target])*data['pixels'][i]).norm(dim=-1).cpu();same=((xy[i,bq[target]]-data['gt_xy'][i,target])*data['pixels'][i]).norm(dim=-1).cpu()
                reliable=good&torch.tensor([k in [0,14,15,16,17] if kind=='real' else k not in [5,6] for k in range(23)]);groups={'all':good,'reliable':reliable,subtype:good}
                if kind=='real':groups[cam+'/feet']=good&torch.tensor([k in [14,15,16,17] for k in range(23)])
                for group,mask in groups.items():
                    a=accum[kind+'/'+group];values=err[mask].tolist();a['error']+=values;a['same_query_error']+=same[mask].tolist();a['points']+=len(values)
                    if hit:a['pck16']+=sum(v<=16 for v in values);a['pck32']+=sum(v<=32 for v in values)
            for group in [kind,kind+'/'+subtype]+([kind+'/'+cam] if kind=='real' else []):
                a=counts[group];a['images']+=1;a['tp']+=tp;a['gt']+=len(ms);a['predictions']+=int((scores>=.5).sum());a['query_switches']+=switches
        metrics={}
        for group,a in accum.items():
            for name in ['error','same_query_error']:
                vals=a.pop(name);a[name+'_mean']=float(np.mean(vals)) if vals else None
            a.update(gated_pck16=a['pck16']/max(a['points'],1),gated_pck32=a['pck32']/max(a['points'],1));metrics[group]=a
        for a in counts.values():a.update(precision=a['tp']/max(a['predictions'],1),recall=a['tp']/max(a['gt'],1))
        return metrics,dict(counts)
    write_json(OUT/'protocol.json',dict(source_sha256=fingerprint(source),predictions_sha256=fingerprint(DET/'predictions.npz'),script_sha256=fingerprint(Path(__file__)),pose='Frozen OLD/full_context_pose_control_v1/full800, same withinseed across3detectors.',scope='112 genuine single-view validation images, full138queries before GTbox-only association. RealreliableIDs1/15-18; syntheticdualtail8-14off. Baselinegoldens required.',metrics='conditionalerror uses new association; same_query_error uses old association with new predictedbox prior. GatedPCK counts missed detections, not every duplicatepoint FP; separate counts/precision. Not externaltest or AP.',production_changed=False))
    results=[];saved={};maxdelta=0.
    for seed in [17,29,43]:
        path=OLD/f'full_context_pose_control_v1/full_seed{seed}_latest.pt';cp=torch.load(path,map_location='cuda',weights_only=False);assert cp['seed']==seed and cp['step']==800 and cp['arm']=='full';model=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False);model.load_state_dict(cp['state'],strict=True);baseline_matches=match_detection(det[(seed,'baseline')],data['boxes'],valid)
        for arm in ARMS:
            assert dt.datetime.now(dt.timezone.utc)<deadline;p=det[(seed,arm)];xy=predict(model,p['instance_boxes_cxcywh'])
            if arm=='baseline':
                delta=float((xy.cpu()-torch.tensor(gold[seed])).abs().max());assert delta<2e-5;maxdelta=max(maxdelta,delta)
            metrics,counts=evaluate(xy,p,baseline_matches);row=dict(seed=seed,arm=arm,metrics=metrics,detector=counts,pose_sha256=fingerprint(path));results.append(row);saved[f'{arm}_seed{seed}']=xy.cpu().numpy();write_json(OUT/'partial_results.json',results);status('auditing',seed=seed,arm=arm);print('SINGLEPOSE',seed,arm,metrics['real/reliable'],metrics['real/camA/feet'],counts['real'],flush=True)
    np.savez_compressed(OUT/'predictions.npz',image_indices=np.array(indices),**saved);write_json(OUT/'results.json',dict(results=results,max_baseline_golden_delta=maxdelta,production_changed=False))
    gallery=OUT/'gallery';gallery.mkdir();picked=list(dict.fromkeys(r['image_index'] for r in json.loads((OLD/'review_view_context_control_v1/manifest.json').read_text())));colors=['#ff6a6a','#66ff88','#55ccff','#ffff55','#ee77ff','#ffa544'];pages=['<!doctype html><meta charset="utf-8"><title>Single-view negative control</title><style>body{font:17px sans-serif;color:#eee;background:#20232b}img{max-width:100%}</style><h1>纯单视角检测负例对照</h1><p>同一full800姿态头，seed17固定。baseline/control/negative04；全部confidence≥0.5查询，无NMS/GT数量过滤。黄色为人工点，颜色是query而非跨视角身份。完整图保留反光/背景。固定上一轮样例，非按改善挑选。真实缺点非不可见，尾巴/背部不在此可靠点图中。</p>'];gallery_rows=[]
    for index,i in enumerate(picked):
        j=lookup[i];m=meta[j];r=m['record'];imdata=load_item(dict(r,views=[m['view']]))['batch']['images'][0,0];pw=460;ph=round(imdata.shape[-2]*pw/imdata.shape[-1]);im=Image.fromarray((imdata.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')).resize((pw,ph));canvas=Image.new('RGB',(pw*3,ph+38),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(ARMS):
            canvas.paste(im,(pw*col,38));draw.text((pw*col+5,6),arm+' / actual V1',fill='white');p=det[(17,arm)];scores=p['instance_objectness_logits'][j].sigmoid().cpu().numpy();boxes=p['instance_boxes_cxcywh'][j].cpu().numpy()
            for q in np.flatnonzero(scores>=.5):
                cx,cy,bw,bh=boxes[q];lo=np.array([cx-bw/2,cy-bh/2])*[pw,ph]+[col*pw,38];hi=np.array([cx+bw/2,cy+bh/2])*[pw,ph]+[col*pw,38];draw.rectangle(tuple(lo)+tuple(hi),outline=colors[q],width=2);draw.text((lo[0],lo[1]-11),f'q{q}:{scores[q]:.2f}',fill=colors[q])
                for k in [0,14,15,16,17]:
                    x,y=saved[f'{arm}_seed17'][j,q,k]*[pw,ph]+[col*pw,38];draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[q]);draw.text((x+3,y),str(k+1),fill=colors[q])
            for target in range(2):
                for k in [0,14,15,16,17]:
                    if not data['point_valid'][j,target,k]:continue
                    x,y=data['gt_xy'][j,target,k].cpu().numpy()*[pw,ph]+[col*pw,38];draw.line((x-4,y,x+4,y),fill='#fff06a',width=2);draw.line((x,y-4,x,y+4),fill='#fff06a',width=2)
        name=f'{index:02d}_{r["kind"]}_cam{chr(65+int(m["view"]))}.jpg';canvas.save(gallery/name,quality=95);pages.append('<h2>'+html.escape(r['scene'])+'</h2><img src="'+name+'">');gallery_rows.append(dict(file=name,image_index=i,metadata=m))
    write_json(gallery/'manifest.json',gallery_rows);(gallery/'index.html').write_text('\n'.join(pages),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005');dest.mkdir(exist_ok=True);assert not (dest/OUT.name).exists();shutil.copytree(OUT,dest/OUT.name)
    status('complete');state.update(stage='single_detector_pose_and_gallery_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Inspect3seed results and fixedcamA/camD/dual gallery. Negativepenalty reduces duplicates but may remove correctcamA query: distinguish scorethreshold and association. ActualRGBV1parity before promoting baseline. Next controlledsingleviewlocalization/fusion setup perplan, not simplymoreheadsteps.');state['completed'].append('single_detector_frozen_pose_3seed_and_raw_gallery');write_json(CYCLE/'state.json',state)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
