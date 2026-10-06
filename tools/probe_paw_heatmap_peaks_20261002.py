"""Actual LP heatmap candidate audit; GT-oracle recall is NOT deployable accuracy."""
import datetime as dt,fcntl,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from torchvision import transforms
from lightning_pose.models.heatmap_tracker import HeatmapTracker
from run_teacher_cycle_dlc_20261001 import ROOT,OUT,write_state

def main():
    guard=(OUT/'gpu_job.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text());deadline=min(time.time()+900,dt.datetime.fromisoformat(plan['deadline_utc']).timestamp())
    def check():assert time.time()<deadline and not (OUT/'STOP').exists()
    check();dest=OUT/'paw_heatmap_peaks_v1';dest.mkdir(exist_ok=False)
    settings=dict(seeds=[17,31,73],nms_kernel=7,topk=5,source_recall_radius_px=15,relative_peak_threshold=.1,
        coordinate_mapping='LP native192x176 heatmap grid times4 then inverse704x768 letterbox affine',
        warning='GT used only to assess candidate recall. NOT a decoder improvement or inferred label. Same development split.',train_allowed=False)
    (dest/'settings.json').write_text(json.dumps(settings,indent=2))
    m=json.loads((OUT/'dlc_dataset_manifest.json').read_text());A=np.array(m['affine_source_to_input']);ind=np.array(m['validation_indices'])
    ids=m['ids_1based'];js=[ids.index(k) for k in [15,16,17,18]]
    assert all(m['entries'][i]['session'] in plan['development_validation_sessions'] for i in ind)
    excluded={(r['session'],r['frame']) for r in json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']}
    assert not any((m['entries'][i]['session'],m['entries'][i]['frame']) in excluded for i in ind)
    torch.set_num_threads(8);assert torch.cuda.is_available()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    write_state(stage='paw_heatmap_candidate_audit_running',pid=os.getpid(),active_run=dest.name,mode='GPU_inference',job_deadline_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat())
    records={};all_peaks=[];all_ratios=[]
    for seed in settings['seeds']:
        check();run=OUT/f'lp_seed{seed}_kl';ck=json.loads((run/'summary.json').read_text())['checkpoint']
        model=HeatmapTracker.load_from_checkpoint(ck,map_location='cpu',pretrained=False,loss_factory=None,weights_only=False).cuda().eval()
        with np.load(run/'predictions_source_pixels.npz') as z:saved=dict(z)
        i=int(ind[0]);start=(i//8)*8
        def load(entries):return torch.stack([transform(cv2.cvtColor(cv2.imread(str(ROOT/e['derived'])),cv2.COLOR_BGR2RGB)) for e in entries]).cuda()
        with torch.inference_mode():
            p,_=model.head.run_subpixelmaxima(model(load(m['entries'][start:start+8])))
            parity=float(np.max(np.abs(p.cpu().numpy().reshape(-1,7,2)-(saved['uv'][start:start+8]*np.diag(A)[:2]+A[:2,2]))))
            assert parity<.1,parity
            coords=[];scores=[]
            for begin in range(0,len(ind),8):
                check();batch=ind[begin:begin+8];hm=model(load([m['entries'][i] for i in batch]))[:,js]
                assert tuple(hm.shape[-2:])==(192,176)
                local=hm.eq(F.max_pool2d(hm,7,stride=1,padding=3));candidate=hm.masked_fill(~local,-torch.inf)
                values,idx=candidate.flatten(-2).topk(5,dim=-1)
                xy=torch.stack([idx%176,idx//176],dim=-1)*4
                coords.append((xy.cpu().numpy()-A[:2,2])/np.diag(A)[:2]);scores.append(values.cpu().numpy())
        peaks=np.concatenate(coords);score=np.concatenate(scores);ratio=score/score[...,:1]
        gt=saved['gt'][ind][:,js];baseline=np.linalg.norm(saved['uv'][ind][:,js]-gt,axis=-1)
        error=np.linalg.norm(peaks-gt[:,:,None],axis=-1);valid=np.isfinite(baseline);severe=valid&(baseline>50)
        report=dict(checkpoint=ck,parity_max_px=parity,n=int(valid.sum()),severe_n=int(severe.sum()),baseline_under15=int((baseline[valid]<=15).sum()),oracle_recall={})
        for k in [1,3,5]:
            for threshold in [0,.1]:
                ok=(error[:,:,:k]<=15)&(ratio[:,:,:k]>=threshold)
                hit=ok.any(-1)
                report['oracle_recall'][f'top{k}_ratio{threshold}']=dict(all_hit=int((hit&valid).sum()),severe_recovered=int((hit&severe).sum()))
        np.savez_compressed(dest/f'seed{seed}.npz',indices=ind,ids=[15,16,17,18],peaks_source_px=peaks,relative_scores=ratio,gt=gt,baseline_error_px=baseline,train_allowed=False)
        records[str(seed)]=report;all_peaks.append(peaks);all_ratios.append(ratio)
        print('PEAK_AUDIT',seed,json.dumps(report),flush=True)
        del model;torch.cuda.empty_cache()
    with np.load(OUT/'lp_ensemble_17_31_73/predictions_source_pixels.npz') as z:ens=z['uv_median'][ind][:,js]
    error=np.linalg.norm(np.stack(all_peaks)-gt[None,:,:,None],axis=-1);ratios=np.stack(all_ratios)
    baseline=np.linalg.norm(ens-gt,axis=-1);valid=np.isfinite(baseline);severe=valid&(baseline>50)
    hit=((error<=15)&(ratios>=.1)).any(0).any(-1)
    report=dict(seeds=records,union_top5_ratio01=dict(n=int(valid.sum()),baseline_under15=int((baseline[valid]<=15).sum()),
        oracle_hit=int((hit&valid).sum()),severe_n=int(severe.sum()),severe_recovered=int((hit&severe).sum())),train_allowed=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['union_top5_ratio01']))
    write_state(stage='paw_heatmap_candidate_audit_complete',pid=None,active_run=None,mode='idle_between_heartbeats',job_deadline_utc=None,
        latest_result=str(dest/'summary.json'),next_action='Inspect candidate oracle recall and peak strengths before any identity-aware decoding; oracle choices must never enter pseudo labels.')

if __name__=='__main__':main()
