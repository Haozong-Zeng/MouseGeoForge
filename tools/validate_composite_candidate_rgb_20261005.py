"""Fresh RGB engineering replay of bounded local + geometry composition."""
import datetime as dt,fcntl,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import numpy as np
import torch
from PIL import Image
from singleview_candidate_runtime import FrozenSingleViewCandidateRuntime
from audit_composite_candidate_cache_20261005 import load_wrapper
from mousegeoforge.candidate_inputs import local_candidate_inputs
from run_surface_curve_pilot_20261003 import load_item
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'composite_candidate_rgb_audit_v1'


@torch.no_grad()
def main():
    plan=json.loads((CYCLE/'plan.json').read_text());deadline=dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z','+00:00'));lock=(ROOT/plan['gpu_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();(OUT/'rgb').mkdir();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    state=json.loads((CYCLE/'state.json').read_text());state.update(stage='composite_candidate_rgb_running',active_job=OUT.name,active_pid=os.getpid());write_json(CYCLE/'state.json',state)
    def status(stage,**kw):write_json(OUT/'status.json',dict(stage=stage,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
    status('loading');started=time.monotonic();source=CYCLE/'singleview_fusion_inputs_seed17_v1/validation.pt';d=torch.load(source,map_location='cpu',weights_only=False)
    manifests=json.loads((CYCLE/'cached_local_candidate_inputs_v1/manifest.json').read_text())['records'];selected=[]
    for kind,count in [('real',2),('synthetic',1)]:
        matches=[r for r in manifests if r.get('role')=='validation' and r.get('kind')==kind]
        # Some manifests store role in the scene only; resolve without labels.
        if not matches:
            matches=[r for r in manifests if r.get('kind')==kind and torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['role']=='validation']
        assert len(matches)>=count;selected+=matches[:count]
    groups=[]
    for record in selected:
        assert fingerprint(ROOT/record['path'])==record['sha256'];s=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False);groups.append(dict(scene=s['scene'],record=s['record'],views=s['physical_views'],projections=s['cameras']['projections'],pixels=s['pixels'],cached=s,kind=record['kind']))
    firstdual=next(m for m in d['metadata'] if m['expected_instances']>1);scene=firstdual['record']['scene'];dualrows=[i for i,m in enumerate(d['metadata']) if m['record']['scene']==scene]
    groups.append(dict(scene=scene,record=firstdual['record'],views=[d['metadata'][i]['view'] for i in dualrows],projections=None,pixels=d['pixels'][dualrows[0]],cached=None,kind='synthetic_dual'))
    lookup={(m['record']['scene'],int(m['view'])):i for i,m in enumerate(d['metadata'])};native={r['image_index']:r for r in json.loads((OLD/'single_view_validation_context_v1/manifest.json').read_text())['records']}
    write_json(OUT/'protocol.json',dict(groups=[dict(scene=g['scene'],views=g['views'],kind=g['kind']) for g in groups],selection='First2 eligible real validation scenes, first synthetic single scene, first existing dual scene, before fresh outputs. Not error-selected.',scope='Fresh independentV1 encoder per uniqueRGB. Images ONLY passed to runtime; labels loaded by dataset only for hash/eval, never forward. Cached outputs comparison after fresh inference. View subset/order/duplicate fusion reruns use these fresh independent outputs (not separately re-encoded contexts). Repeat firstRGB once checks stateless numeric replay. Not compute-matched with a joint multiview encoder. Dual pair calibration unused/no fusion claim.',source_sha256=fingerprint(source)))
    runtime=FrozenSingleViewCandidateRuntime();wrapper=load_wrapper();records=[];group_results=[];saved=[];repeat_delta=None
    modes=dict(baseline=(False,False),local=(True,False),fusion_only=(False,True),composite=(True,True))
    for gi,g in enumerate(groups):
        live=[];rows=[];images=[]
        for vi,view in enumerate(g['views']):
            assert dt.datetime.now(dt.timezone.utc)<deadline
            row=lookup[(g['scene'],int(view))];m=d['metadata'][row];assert m['record']['role']=='validation' and m['record'].get('session') not in ['GR2','UR2','UR3']
            item=load_item(dict(m['record'],views=[int(view)]));rgb=item['batch']['images']
            torch.cuda.synchronize();t0=time.monotonic();raw=runtime.from_images(rgb);torch.cuda.synchronize();elapsed=time.monotonic()-t0
            local=local_candidate_inputs(raw['probability'].cpu().numpy(),raw['xy'][:,[0,14,15,16,17]].cpu().numpy(),raw['maps'])
            p={k:v.cpu() for k,v in raw.items() if k!='maps'};p.update(local);live.append(p);rows.append(row)
            # Comparison only after complete fresh prediction/candidate creation.
            rec=native[int(d['image_indices'][row])];assert item['source_hashes']==rec['source_hashes']
            delta={name:float((p[key]-d[sourcekey][row]).abs().max()) for name,key,sourcekey in [('xy','xy','xy'),('boxes','boxes','boxes'),('logits','objectness_logits','objectness_logits')]};assert max(delta.values())<2e-5,delta
            name=f'rgb/group{gi:02d}_view{view}.png';im=(rgb[0,0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8);Image.fromarray(im).save(OUT/name);images.append(name)
            records.append(dict(group=gi,row=row,image_index=int(d['image_indices'][row]),view=int(view),delta=delta,seconds=elapsed,rgb_path=name,source_hashes=item['source_hashes']))
            if repeat_delta is None:
                again=runtime.from_images(rgb);repeat_delta=max(float((again[k].cpu()-p[k]).abs().max()) for k in ['xy','boxes','objectness_logits','tokens','probability']);assert repeat_delta<2e-5,repeat_delta
            status('fresh_RGB',completed=len(records),total=sum(len(x['views']) for x in groups));print(gi,view,delta,flush=True)
            del raw,rgb,item
        prediction={k:torch.stack([p[k] for p in live]) for k in live[0]};v=len(live)
        def call(ids,mode='composite',**kw):
            local,fusion=modes[mode];projection=None if g['projections'] is None else g['projections'][ids]
            return wrapper({k:x[ids] for k,x in prediction.items()},projection,[g['views'][i] for i in ids],g['pixels'],local_enabled=local,fusion_enabled=fusion,**kw)
        outputs={name:call(list(range(v)),name) for name in modes};tests={}
        if g['cached'] is not None:
            s=g['cached'];gold=wrapper(s['predictions'],s['cameras']['projections'],s['physical_views'],s['pixels'],local_enabled=True,fusion_enabled=True)
            tests['cached_composite_xy']=float((outputs['composite']['xy']-gold['xy']).abs().max());assert tests['cached_composite_xy']<2e-5,tests
        else:
            tests['dual_composite_abstention']=float((outputs['composite']['xy']-prediction['xy']).abs().max());assert tests['dual_composite_abstention']==0
        for label,ids in [('reverse',list(reversed(range(v)))),('duplicate',list(range(v))*2)]:
            out=call(ids)['xy'];out=out.flip(0) if label=='reverse' else out[:v];tests[label]=float((out-outputs['composite']['xy']).abs().max());assert tests[label]==0
        for count in [1,2]:
            maximum=0.
            for target in range(v):
                ids=[target]+[i for i in range(v) if i!=target][:count-1];out=call(ids)['xy'][0];maximum=max(maximum,float((out-outputs['local']['xy'][target]).abs().max()))
            tests[f'V{count}_local_fallback']=maximum;assert maximum==0
        tests['all_invalid']=float((call(list(range(v)),view_valid=[False]*v)['xy']-prediction['xy']).abs().max());assert tests['all_invalid']==0
        group_results.append(dict(scene=g['scene'],kind=g['kind'],rows=rows,views=g['views'],tests=tests,local_eligible=int(outputs['composite']['local_eligible'].sum()),fusion_eligible=int(outputs['composite']['fusion_eligible'].sum()),confident_predictions_per_image=(prediction['objectness_logits'].sigmoid()>=.5).sum(-1).tolist(),images=images))
        saved.append(dict(scene=g['scene'],kind=g['kind'],rows=rows,views=g['views'],pixels=g['pixels'],projections=g['projections'],predictions=prediction,outputs=outputs,images=images))
    torch.save(saved,OUT/'fresh_groups.pt');write_json(OUT/'results.json',dict(groups=group_results,records=records,encoder_calls=len(records)+1,unique_images=len(records),repeat_RGB_delta=repeat_delta,elapsed_seconds=time.monotonic()-started,runtime_sha256=runtime.hashes,wrapper_heads_sha256={arm:fingerprint(CYCLE/f'bounded_prior_candidate_pilot_v1/{arm}_latest.pt') for arm in ['unary','geometry']},script_sha256=fingerprint(Path(__file__)),module_sha256=fingerprint(ROOT/'mousegeoforge/models/bounded_candidate_fusion.py'),claim='Engineering RGB replay and fusion gating only. Not new independent accuracy study. No model or label promotion. Single-subject association heuristic not proof of multi-mouse identity.',production_changed=False));status('complete');state.update(stage='composite_candidate_rgb_complete',active_job=None,active_pid=None,next='Read composite_candidate_rgb_audit_v1, cacheaudit, visualize exportedactualRGB with allqueries; report timings/parity/limits before nextstep. No training/promotion.');state['completed'].append('composite_candidate_fresh_rgb_audit');write_json(CYCLE/'state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write_json(OUT/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise
