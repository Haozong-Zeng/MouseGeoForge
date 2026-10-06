"""Frozen V1 probability replay and post-forward localization diagnostics.

All six queries are retained: GT matching never decides which query is decoded.
Only five reliable semantic points are stored, with no RGB/token duplication.
"""
import collections,datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.models.decoder_spatial_pose_readout import DecoderSpatialPoseReadout
from mousegeoforge.detection_readout_loss import match_detection
from run_depth_pose_ablation import fingerprint,write_json

OLD=ROOT/'output/core23_spatial_cycle_20261004'
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005'
SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1'
CAMERAS=CYCLE/'camera_geometry_audit_v2'
OUT=CYCLE/'singleview_probability_diagnostic_v1'
KP=[0,14,15,16,17]


def nms_peaks(probability,count=5,radius=2):
    """Fixed greedy spatial peaks, never selected by distance to GT."""
    value=np.array(probability,copy=True);points=[];scores=[]
    for _ in range(count):
        flat=int(np.argmax(value));y,x=np.unravel_index(flat,value.shape)
        points.append([(x+.5)/value.shape[1],(y+.5)/value.shape[0]])
        scores.append(float(probability[y,x]))
        value[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=-1
    return np.array(points),scores


def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'))
    lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists() and torch.cuda.is_available();OUT.mkdir();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='extracting_probability_diagnostic',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading')
    posepath=OLD/'full_context_pose_control_v1/full_seed17_latest.pt'
    model=DecoderSpatialPoseReadout().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(posepath,map_location='cuda',weights_only=False)['state'],strict=True)
    manifest=json.loads((SOURCE/'manifest.json').read_text());cammanifest=json.loads((CAMERAS/'results.json').read_text())
    y,x=np.meshgrid((np.arange(48)+.5)/48,(np.arange(48)+.5)/48,indexing='ij');grid=np.stack([x,y],-1)
    maxdelta=0.;parts=[];observations=[]
    for part in manifest['parts']:
        role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
        cp=next(p for p in cammanifest['camera_artifacts'] if p['role']==role);assert fingerprint(ROOT/cp['path'])==cp['sha256']
        scenes=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=False);row_ids=sorted(i for g in scenes for i in g['row_indices'])
        assert len(row_ids)==len(set(row_ids))
        native_manifest=OLD/f'single_view_{role}_context_v1/manifest.json';native={r['image_index']:r for r in json.loads(native_manifest.read_text())['records']}
        buckets=collections.defaultdict(list)
        for i in row_ids:
            m=data['metadata'][i];assert m['record'].get('session') not in ['GR2','UR2','UR3'];assert m['record']['role']==role
            buckets[tuple(m['record']['image_hw'])].append(i)
        saved=np.empty((len(row_ids),6,5,48,48),np.float32);local={i:j for j,i in enumerate(row_ids)};completed=0
        for ids_in_shape in buckets.values():
            for start in range(0,len(ids_in_shape),8):
                assert dt.datetime.now(dt.timezone.utc)<deadline
                ids=ids_in_shape[start:start+8];memories=[]
                for i in ids:
                    rec=native[data['image_indices'][i]];assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)
                    assert raw['metadata']==data['metadata'][i];memories.append(raw['memory'])
                memory=torch.stack(memories).cuda();b=len(ids)
                # Decoder outputs are immutable cached results from all138
                # queries. No GT-based query selection occurs here.
                token=data['tokens'][ids].cuda();coarse=data['coarse'][ids].cuda();boxes=data['boxes'][ids].cuda()
                maps=F.interpolate(memory.half().float(),(48,48),mode='bilinear',align_corners=False)
                expanded=maps[:,None].expand(-1,6,-1,-1,-1).reshape(b*6,256,48,48)
                with torch.no_grad():
                    xy,prob=model.spatial_readout(token.reshape(b*6,23,256),coarse.reshape(b*6,23,2),expanded,
                        return_probability=True,predicted_boxes=boxes.reshape(b*6,4),detach_pose_tokens=False)
                delta=float((xy.reshape(b,6,23,2).cpu()-data['xy'][ids]).abs().max());assert delta<2e-5;maxdelta=max(maxdelta,delta)
                probability=prob.reshape(b,6,23,48,48)[:,:,KP].cpu().numpy()
                assert np.isfinite(probability).all();np.testing.assert_allclose(probability.sum((-1,-2)),1,atol=2e-6)
                for j,i in enumerate(ids):saved[local[i]]=probability[j]
                completed+=len(ids);status('extracting',role=role,completed=completed,total=len(row_ids))
                if completed%80==0:print(role,completed,'/',len(row_ids),flush=True)
        path=OUT/f'{role}_probabilities.npz';np.savez_compressed(path,probability=saved,row_indices=row_ids,image_indices=[data['image_indices'][i] for i in row_ids],keypoint_indices=KP)
        parts.append(dict(role=role,path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),images=len(row_ids),source_sha256=part['sha256'],native_manifest_sha256=fingerprint(native_manifest),bytes=path.stat().st_size))
        det=dict(instance_boxes_cxcywh=data['boxes'],instance_objectness_logits=data['objectness_logits'])
        matches=match_detection(det,data['gt_boxes'],data['box_valid']&data['instance_valid'])
        for i in row_ids:
            m=data['metadata'][i];kind='real' if m['record']['kind']=='real' else 'synthetic';pixel=data['pixels'][i].numpy()
            for qq,tt in zip(*matches[i]):
                q,t=int(qq),int(tt)
                for ki,kp in enumerate(KP):
                    if not data['point_valid'][i,t,kp]:continue
                    p=saved[local[i],q,ki];gt=data['gt_xy'][i,t,kp].numpy();mean=data['xy'][i,q,kp].numpy()
                    peaks,scores=nms_peaks(p);peak_error=np.linalg.norm((peaks-gt)*pixel,axis=-1)
                    dist=np.linalg.norm((grid-gt)*pixel,axis=-1);cell=np.minimum(np.maximum((gt*48).astype(int),0),47);gtvalue=p[cell[1],cell[0]]
                    observations.append(dict(role=role,kind=kind,row=i,image_index=data['image_indices'][i],view=int(m['view']),point=kp+1,query=q,target=t,
                        mean_error=float(np.linalg.norm((mean-gt)*pixel)),peak_error=float(peak_error[0]),
                        oracle_top5_error=float(peak_error.min()),mean_to_peak=float(np.linalg.norm((mean-peaks[0])*pixel)),
                        mass16=float(p[dist<=16].sum()),mass32=float(p[dist<=32].sum()),
                        gt_cell_rank=int((p>gtvalue).sum()+1),peak_scores=scores,peaks=peaks.tolist(),
                        query_score=float(data['objectness_logits'][i,q].sigmoid())))
    summaries=[]
    for role in ['train','validation']:
        for kind in ['real','synthetic']:
            for name in ['reliable','bottom_feet','bad_bottom_feet']:
                rows=[r for r in observations if r['role']==role and r['kind']==kind and (name=='reliable' or r['view']==0 and r['point']!=1) and (name!='bad_bottom_feet' or r['mean_error']>32)]
                if not rows:continue
                summary=dict(role=role,kind=kind,part=name,points=len(rows))
                for k in ['mean_error','peak_error','oracle_top5_error','mean_to_peak','mass16','mass32','gt_cell_rank']:summary[k]=float(np.mean([r[k] for r in rows]))
                for key in ['mean_error','peak_error','oracle_top5_error']:summary[key+'_pck32']=float(np.mean([r[key]<=32 for r in rows]))
                summaries.append(summary)
    write_json(OUT/'observations.json',observations);write_json(OUT/'results.json',dict(summaries=summaries,
        warning='GT mass, rank and best-of-five distance are diagnostic oracles, NOT inference or an achieved pose score. Peak candidates selected from predictions only. Conditional postforward GTbox association; repeated development validation, not externaltest.'))
    write_json(OUT/'manifest.json',dict(parts=parts,source_manifest_sha256=fingerprint(SOURCE/'manifest.json'),pose_sha256=fingerprint(posepath),
        max_expected_xy_replay_delta=maxdelta,script_sha256=fingerprint(Path(__file__)),keypoint_indices=KP,queries='All6 retained, no GTselection',production_changed=False))
    status('complete');state.update(stage='probability_diagnostic_ready',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        next='Read singleview_probability_diagnostic_v1/results.json and visually inspect probability overlays. Oracle top5/GTmass are diagnostics only, not deployable improvements. Choose next spatial evidence route on TRAIN, not VAL; no groundtruth in forward.')
    state['completed'].append('fixed_v1_probability_diagnostic');write_json(CYCLE/'state.json',state)
    print(json.dumps(dict(parts=parts,maxdelta=maxdelta,summaries=summaries)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
