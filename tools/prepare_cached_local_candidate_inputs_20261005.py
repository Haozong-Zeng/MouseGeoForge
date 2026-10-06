"""CPU extraction of candidate-local evidence from immutable existing V1 caches.

Real TRAIN and unchanged real/synthetic development cohorts only. Synthetic
TRAIN full contexts live in a different, already audited manifest. No new
probability/RGB copies; only candidate positions/masses and sampled features.
"""
import datetime as dt,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from torch.nn import functional as F
from mousegeoforge.candidate_inputs import local_candidate_inputs
from run_depth_pose_ablation import fingerprint,write_json
OLD=ROOT/'output/core23_spatial_cycle_20261004';CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';SOURCE=CYCLE/'singleview_fusion_inputs_seed17_v1';MAPS=CYCLE/'singleview_probability_diagnostic_v1';OUT=CYCLE/'cached_local_candidate_inputs_v1';KP=[0,14,15,16,17]


def main():
    assert not OUT.exists();OUT.mkdir();(OUT/'scenes').mkdir();torch.set_num_threads(2)
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));state=json.loads((CYCLE/'state.json').read_text());state.update(stage='preparing_cached_local_candidate_inputs',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    sm=json.loads((SOURCE/'manifest.json').read_text());pm=json.loads((MAPS/'manifest.json').read_text());gm=json.loads((CYCLE/'camera_geometry_audit_v2/results.json').read_text());records=[];images=0;started=time.monotonic();sources=[]
    for part in sm['parts']:
        role=part['role'];assert fingerprint(ROOT/part['path'])==part['sha256'];data=torch.load(ROOT/part['path'],map_location='cpu',weights_only=False)
        mp=next(r for r in pm['parts'] if r['role']==role);assert fingerprint(ROOT/mp['path'])==mp['sha256']
        with np.load(ROOT/mp['path']) as a:prob=a['probability'];lookup={int(i):j for j,i in enumerate(a['row_indices'])}
        gp=next(r for r in gm['camera_artifacts'] if r['role']==role);assert fingerprint(ROOT/gp['path'])==gp['sha256'];groups=torch.load(ROOT/gp['path'],map_location='cpu',weights_only=False)
        nativepath=OLD/f'single_view_{role}_context_v1/manifest.json';native={r['image_index']:r for r in json.loads(nativepath.read_text())['records']};sources.append(dict(role=role,packed=part['sha256'],probabilities=mp['sha256'],geometry=gp['sha256'],native_manifest=fingerprint(nativepath)))
        for g in groups:
            if role=='train' and g['kind']=='synthetic':continue
            assert dt.datetime.now(dt.timezone.utc)<deadline
            ids=g['row_indices'];values=[];native_hashes={}
            for i in ids:
                m=data['metadata'][i];assert m['record']['role']==role and m['record'].get('session') not in ['GR2','UR2','UR3']
                rec=native[data['image_indices'][i]];assert fingerprint(ROOT/rec['path'])==rec['sha256'];raw=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False);assert raw['metadata']==m;native_hashes[rec['path']]=rec['sha256']
                memory=F.interpolate(raw['memory'][None].half().float(),(48,48),mode='bilinear',align_corners=False)
                p=prob[lookup[i]];baseline=data['xy'][i,:,KP].numpy();aux=local_candidate_inputs(p,baseline,memory)
                torch.testing.assert_close(aux['candidate_xy'][:,:,0],torch.from_numpy(baseline),atol=0,rtol=0)
                values.append(dict(xy=data['xy'][i],boxes=data['boxes'][i],objectness_logits=data['objectness_logits'][i],tokens=data['tokens'][i,:,KP],
                    probability_entropy=data['entropy'][i,:,KP],probability_variance=data['variance'][i,:,KP],**aux));images+=1
            pred={k:torch.stack([v[k] for v in values]) for k in values[0]}
            targets={k:data[source][ids] for k,source in [('keypoint_xy','gt_xy'),('keypoint_valid','point_valid'),('boxes_cxcywh','gt_boxes'),('box_valid','box_valid'),('instance_valid','instance_valid')]}
            # No fabricated real visibility/3D fields. Source labels are only
            # copied as distinct supervision arrays, never used in sampling.
            path=OUT/'scenes'/f'{len(records):03d}.pt';torch.save(dict(scene=g['scene'],role=role,kind=g['kind'],record=data['metadata'][ids[0]]['record'],physical_views=g['physical_views'],source_rows=ids,
                image_indices=[data['image_indices'][i] for i in ids],pixels=data['pixels'][ids[0]].numpy(),predictions=pred,targets=targets,cameras=dict(intrinsics=g['intrinsics'],extrinsics=g['extrinsics'],projections=g['projections'],normalization_scale=g['scale']),source_hashes=g['source_hashes'],native_hashes=native_hashes),path)
            records.append(dict(scene=g['scene'],role=role,kind=g['kind'],path=path.relative_to(ROOT).as_posix(),sha256=fingerprint(path),views=g['physical_views'],images=len(ids),bytes=path.stat().st_size))
            write_json(OUT/'status.json',dict(stage='extracting',pid=os.getpid(),scenes=len(records),images=images,role=role,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            if len(records)%30==0:print(len(records),'scenes',images,'images',flush=True)
    assert len(records)==180 and images==1108
    write_json(OUT/'manifest.json',dict(records=records,images=images,sources=sources,script_sha256=fingerprint(Path(__file__)),candidate_module_sha256=fingerprint(ROOT/'mousegeoforge/candidate_inputs.py'),bytes=sum(r['bytes'] for r in records),elapsed_seconds=time.monotonic()-started,
        contract='CPU only. FrozenV1 local evidence at prediction-defined six alternatives: originalmean+fivepeaks. All6instancequeries,5reliablepoints, noGTcrop/selection. TRAINreal156scenes1028images; validation24scenes80images. SyntheticTRAIN separate fullcontext461images. No real visibility/3D fabrication.'))
    write_json(OUT/'status.json',dict(stage='complete',images=images,scenes=len(records),pid=os.getpid()));state.update(stage='candidate_learning_inputs_ready',active_job=None,active_pid=None,next='Read fullsynthetic audit and both input manifests. Implement matched candidate selector with originalmean fallback, geometry on/off control, labels only in loss. No promotion/validation search.');state['completed'].append('cached_local_candidate_inputs');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
