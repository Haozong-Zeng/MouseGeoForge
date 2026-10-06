"""Stage-specific grouped cache for point accuracy quality, no coordinate edits."""
import collections,datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'point_quality_inputs_v1';KP=[0,14,15,16,17]
SCALARS=['entropy','log1p_pixel_std/log769','peak_probability','objectness','log1p_peak_offset_px/log769','box_relative_x/4','box_relative_y/4']+[f'point{k+1}' for k in KP]


def features(xy,boxes,logits,tokens,entropy,variance,peak_probability,peak_xy,pixels):
    pixel=torch.as_tensor(pixels).float();std=(variance.float()*pixel.square()).sum(-1).clamp_min(0).sqrt()
    offset=((peak_xy.float()-xy[:,:,KP] if xy.ndim==4 else peak_xy.float()-xy[:,KP])*pixel).norm(dim=-1)
    base=xy[:,:,KP] if xy.ndim==4 else xy[:,KP]
    relative=((base-boxes[...,None,:2])/boxes[...,None,2:].clamp_min(.02)).clamp(-4,4)/4
    score=logits.float().sigmoid()[...,None].expand_as(entropy)
    scalar=torch.stack([entropy.float(),std.log1p()/np.log(769),peak_probability.float(),score,offset.log1p()/np.log(769),relative[...,0],relative[...,1]],-1)
    onehot=torch.eye(5).expand(*scalar.shape[:-2],5,5)
    return tokens.float(),torch.cat([scalar,onehot],-1),std


def supervision(xy,boxes,logits,d,pixels):
    q,t=match_detection(dict(instance_boxes_cxcywh=boxes[None],instance_objectness_logits=logits[None]),d['gt_boxes'][None],(d['box_valid']&d['instance_valid'])[None])[0]
    valid=torch.zeros(6,5,dtype=torch.bool);error=torch.full((6,5),float('nan'));iou=torch.full((6,),float('nan'))
    for query,target in zip(q.tolist(),t.tolist()):
        iou[query]=box_iou(boxes[query:query+1],d['gt_boxes'][target:target+1])[0,0]
        if logits[query].sigmoid()<.5 or iou[query]<.5:continue
        valid[query]=d['point_valid'][target,KP]
        error[query]=((xy[query,KP]-d['gt_xy'][target,KP])*pixels).norm(dim=-1)
    error[~valid]=float('nan')
    return valid,error,iou


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    deadline=json.loads((CYCLE/'plan.json').read_text())['training_stop_utc'];assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(deadline.replace('Z','+00:00'))
    protocol=dict(real_calibration='FR3',single_calibration='mouse_climb_wall_91',dual_calibration_groups=[5,7,12,14,19],
        target='P(2D localization error <=32 input pixels), not visibility/objectness.',scalar_names=SCALARS,
        omitted='21curve-kind TRAIN images lack source_group; omit rather than assume independence. Replace42TRAIN/32VAL dual2view images with84TRAIN/64VAL4view images, no duplicates.',
        supervision='All6queries x5points enter feature extraction. Loss/evaluation known only if original predconfidence>=.5, fixed boxmatch IoU>=.5 and point_valid. Missing labels/unknown identity are NOT negatives. No GT for feature construction.',
        split_scope='All base model TRAIN data already seen by frozen detector/pose. FR3 real source/session held out only for new quality head; single climb action held out; dual5cal frames same sourceA. Outer development reused realFR2/AZR2L2,syntheticwash,dualactionB. No pristine/generalreal3D claim.')
    write_json(OUT/'protocol.json',protocol)
    rows=[];metadata=[];hashes={};parity=[];old_dual={};omitted=[]
    for role in ['train','validation']:
        file=CYCLE/'singleview_fusion_inputs_seed17_v1'/f'{role}.pt';hashes[str(file.relative_to(ROOT))]=fingerprint(file);data=torch.load(file,map_location='cpu',weights_only=False)
        for i,m in enumerate(data['metadata']):
            r=m['record'];assert r.get('session') not in ['GR2','UR2','UR3']
            if m['expected_instances']>1:
                old_dual[role,r['scene'],m['view']]={k:data[k][i] for k in ['xy','boxes','objectness_logits']};continue
            if r['kind']!='real' and not r.get('source_group'):
                omitted.append(dict(role=role,scene=r['scene'],view=m['view'],reason='unverified_action_source'));continue
            kind='real' if r['kind']=='real' else 'single';source=r['session'] if kind=='real' else r['source_group']
            stage='validation' if role=='validation' else 'calibration' if source in ['FR3','mouse_climb_wall_91'] else 'fit'
            xy,boxes,logits,pixel=[data[k][i] for k in ['xy','boxes','objectness_logits','pixels']]
            tok,sca,std=features(xy,boxes,logits,data['tokens'][i,:,KP],data['entropy'][i,:,KP],data['variance'][i,:,KP],data['peak_probability'][i,:,KP],data['peak_xy'][i,:,KP],pixel)
            d={k:data[k][i] for k in ['gt_boxes','gt_xy','point_valid','box_valid','instance_valid']};valid,error,iou=supervision(xy,boxes,logits,d,pixel)
            rows.append(dict(tokens=tok,scalars=sca,std_px=std,valid=valid,error=error,known_iou=iou,confident=logits.sigmoid()>=.5,xy=xy,boxes=boxes))
            metadata.append(dict(role=role,stage=stage,kind=kind,source=source,scene=r['scene'],view=int(m['view']),image_index=data['image_indices'][i],physical_bottom=kind=='real' and m['view']==0))
    for role,folder in [('train','dual_train_allview_affinity_context_v1'),('validation','dual_allscene_association_audit_v1')]:
        file=CYCLE/folder/'groups.pt';hashes[str(file.relative_to(ROOT))]=fingerprint(file);groups=torch.load(file,map_location='cpu',weights_only=False)
        for gi,g in enumerate(groups):
            p=g['predictions'];prob=p['probability'].float();h,w=prob.shape[-2:];y,x=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij');grid=torch.stack([x,y],-1)
            entropy=-(prob*prob.clamp_min(1e-12).log()).sum((-1,-2))/np.log(h*w)
            variance=(prob[...,None]*(grid[None,None,None]-p['xy'][:,:,KP,None,None,:])**2).sum((-3,-2))
            peak=prob.flatten(-2).max(-1);peakxy=grid.reshape(-1,2)[peak.indices]
            for vi,view in enumerate(g['record']['views']):
                key=role,g['record']['scene'],view
                if key in old_dual:
                    delta={name:float((p[name][vi]-old_dual[key][name]).abs().max()) for name in old_dual[key]}
                    assert max(delta.values())<1e-5,delta;parity.append(dict(role=role,scene=key[1],view=view,delta=delta))
                stage='validation' if role=='validation' else 'calibration' if gi in protocol['dual_calibration_groups'] else 'fit'
                xy,boxes,logits=[p[k][vi] for k in ['xy','boxes','objectness_logits']];pixel=torch.tensor(g['pixels']).float()
                tok,sca,std=features(xy,boxes,logits,p['tokens'][vi],entropy[vi],variance[vi],peak.values[vi],peakxy[vi],pixel)
                d={k:g['evaluation_data'][k][vi] for k in ['gt_boxes','gt_xy','point_valid','box_valid','instance_valid']};valid,error,iou=supervision(xy,boxes,logits,d,pixel)
                rows.append(dict(tokens=tok,scalars=sca,std_px=std,valid=valid,error=error,known_iou=iou,confident=logits.sigmoid()>=.5,xy=xy,boxes=boxes))
                metadata.append(dict(role=role,stage=stage,kind='dual',source=g['record']['source_group'],scene=g['record']['scene'],view=view,group_index=gi,physical_bottom=False))
    assert len(parity)==74 and len(omitted)==21
    keys=[(m['role'],m['scene'],m['view']) for m in metadata];assert len(keys)==len(set(keys))
    scene_stages={}
    for m in metadata:assert scene_stages.setdefault(m['scene'],m['stage'])==m['stage']
    packed={k:torch.stack([x[k] for x in rows]) for k in rows[0]};packed['target']=(packed['error']<=32).float();packed['metadata']=metadata
    assert torch.isfinite(packed['tokens']).all() and torch.isfinite(packed['scalars']).all()
    stats=[]
    for stage in ['fit','calibration','validation']:
        for kind in ['real','single','dual']:
            ids=[i for i,m in enumerate(metadata) if m['stage']==stage and m['kind']==kind];assert ids
            known=packed['valid'][ids];conf=packed['confident'][ids][...,None].expand(-1,-1,5)
            stats.append(dict(stage=stage,kind=kind,images=len(ids),known_points=int(known.sum()),confident_points=int(conf.sum()),unknown_confident_points=int((conf&~known).sum()),
                sources=sorted({metadata[i]['source'] for i in ids}),positive_points=int((packed['target'][ids]*known).sum()) if stage!='validation' else None))
    torch.save(packed,OUT/'inputs.pt');write_json(OUT/'manifest.json',dict(stats=stats,source_sha256=hashes,inputs_sha256=fingerprint(OUT/'inputs.pt'),omitted=omitted,dual_old_view_parity=parity,script_sha256=fingerprint(Path(__file__))))
    write_json(OUT/'status.json',dict(stage='complete'));print(json.dumps(stats,indent=2))
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='point_quality_inputs_ready',active_job=None,active_pid=None);state['completed'].append('point_quality_stage_inputs');write_json(CYCLE/'state.json',state)


if __name__=='__main__':main()

