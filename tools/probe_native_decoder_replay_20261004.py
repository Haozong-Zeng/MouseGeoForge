"""Quantify native-size fp16 memory replay against saved full-query decoder goldens."""
import datetime as dt,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'native_decoder_replay_v1'

def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='checking_native_decoder_replay',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    initial=ROOT/'output/surface_curve_cycle_20261003/training_v1/pilot/initial_heads.pt';source=json.loads((CYCLE/'features_expanded_all_v1/manifest.json').read_text());assert fingerprint(initial)==source['initial_sha256']
    raw=torch.load(initial,map_location='cpu',weights_only=True);head=MultiInstanceMouseHead(num_keypoints=23).cuda().eval().requires_grad_(False);head.load_state_dict({k[len('instance_pose_head.'):]:v for k,v in raw.items() if k.startswith('instance_pose_head.')},strict=True);del raw
    records=json.loads((CYCLE/'detection_features_v1/manifest.json').read_text())['records'];allowed=json.loads((CYCLE/'detection_roi_features_v1/manifest.json').read_text())['memory_hashes'];details=[];last=None;verified=set();allpixels=[]
    for record in records:
        assert dt.datetime.now(dt.timezone.utc)<deadline;assert fingerprint(ROOT/record['path'])==record['sha256']
        observations=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)['observations']
        for o in observations:
            m=o['metadata'];assert m['record'].get('session') not in ['GR2','UR2','UR3'];path=m['memory_part']
            if path!=last:
                if path not in verified:assert fingerprint(ROOT/path)==allowed[path];verified.add(path)
                memory=torch.load(ROOT/path,map_location='cpu',weights_only=False)['memory'].float().cuda();last=path
            native=memory[m['view_index']:m['view_index']+1].flatten(2).transpose(1,2)
            with torch.no_grad():
                seed=o['instance_tokens'][None].cuda()[:,:,None]+head.keypoint_queries.weight[None,None]
                tokens=head.keypoint_decoder(seed.flatten(1,2),native).reshape(6,23,256)
                coarse=head.coarse_keypoint_xy(tokens).sigmoid()
            token_delta=float((tokens.cpu()-o['pose_tokens']).abs().max());xy_delta=float((coarse.cpu()-o['coarse_xy']).abs().max());px=((coarse.cpu()-o['coarse_xy'])*o['pixels']).norm(dim=-1).numpy();allpixels+=px.flatten().tolist()
            details.append(dict(metadata=m,token_delta=token_delta,coarse_normalized_delta=xy_delta,coarse_px_mean=float(px.mean()),coarse_px_max=float(px.max())))
        write_json(OUT/'status.json',dict(stage='checking',pid=os.getpid(),images=len(details),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    report=dict(images=len(details),max_token_delta=max(r['token_delta'] for r in details),max_coarse_normalized_delta=max(r['coarse_normalized_delta'] for r in details),coarse_px_mean=float(np.mean(allpixels)),coarse_px_p99=float(np.quantile(allpixels,.99)),coarse_px_max=max(allpixels),token_tolerance=2e-4,coarse_tolerance=2e-5,details=details,initial_sha256=fingerprint(initial),script_sha256=fingerprint(Path(__file__)),contract='Original keypointdecoder/semanticembeddings/coarsehead, ALL6x23 queries, original instance tokens, native spatial memory shape; fp16 stored memory reconstructed float32, no resize. No GT or loss or new training. Not a model accuracy score.')
    report['strict_parity_passed']=report['max_token_delta']<2e-4 and report['max_coarse_normalized_delta']<2e-5
    write_json(OUT/'results.json',report);write_json(OUT/'status.json',dict(stage='complete',pid=os.getpid(),strict_parity_passed=report['strict_parity_passed']))
    state.update(stage='native_decoder_replay_checked',active_job=None,active_pid=None,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Read native_decoder_replay_v1/results.json. If strict gate fails extract nativefp32 memory via originalhead hooks and verify full23replay; do not silently train on drifted features. Then bounded matched decoder freeze/unfreeze control.');state['completed'].append('native_memory_decoder_replay_gate');write_json(CYCLE/'state.json',state)

if __name__=='__main__':main()
