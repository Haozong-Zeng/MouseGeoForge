"""Balanced, source-split actual RGB inputs for a mixed-domain backbone control."""
import sys,json,random,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch
from mousegeoforge.detection_readout_loss import match_detection
from mousegeoforge.pilot_metrics import box_iou
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'mixed_local_rgb_inputs_v1'
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    global OUT
    full_real='--full-real' in sys.argv
    if full_real:OUT=CYCLE/'mixed_local_fulltrain_rgb_inputs_v1'
    OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();torch.set_num_threads(2);rows=prepare_rows();rng=random.Random(17);small=json.loads((CYCLE/'highres_preflight_inputs_v1/protocol.json').read_text())['indices'];chosen=[];pools={}
    for stage,real_n in [('fit',128),('calibration',32)]:
        for kind in ['real','single','dual']:
            pool=[i for i,r in enumerate(rows) if r['metadata']['stage']==stage and r['metadata']['kind']==kind and r['valid'].any()];pools[stage+'_'+kind]=len(pool)
            if kind=='real':
                selected=[]
                for bottom in [True,False]:
                    candidates=[i for i in pool if rows[i]['metadata']['physical_bottom']==bottom];keep=[i for i in small if i in candidates];take=min(real_n//2,len(candidates));assert len(keep)<=take;selected+=keep+rng.sample([i for i in candidates if i not in keep],take-len(keep))
            else:selected=pool
            chosen.extend(selected)
    if full_real:
        previous=json.loads((CYCLE/'mixed_local_rgb_inputs_v1/protocol.json').read_text())['indices'];assert chosen==previous
        chosen += [i for i,r in enumerate(rows) if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real' and r['valid'].any() and i not in chosen]
    assert len(chosen)==len(set(chosen));assert set(small).issubset(chosen)
    manifests=[];evaluation=[];hashes={};items={};counts=collections.Counter();allpointcounts=collections.Counter()
    for number,i in enumerate(chosen):
        r=rows[i];m=r['metadata'];key=(r['record']['scene'],tuple(r['record']['views']))
        if key not in items:items[key]=load_item(r['record'])
        item=items[key];batch=item['batch'];t=instance_targets_from_batch(batch);v=r['record']['views'].index(m['view']);rgb=batch['images'][0,v].permute(1,2,0);a=(rgb*255).round().to(torch.uint8);assert torch.equal(a.float()/255,rgb);path=OUT/'images'/f'{i:04d}.png';Image.fromarray(a.numpy()).save(path);assert torch.equal(torch.tensor(np.asarray(Image.open(path)).copy()).float()/255,rgb)
        target=t['keypoint_xy'][0,v,:,KP].clone();known=t['keypoint_valid'][0,v,:,KP].clone()&t['instance_valid'][0,v,:,None];vv=t['keypoint_visibility_valid'][0,v,:,KP].clone();visible=t['keypoint_visible'][0,v,:,KP].clone();eligible=known.clone() if m['kind']=='real' else known&vv&visible
        if m['kind']=='real':assert not vv.any() and m['source'] not in ['GR2','UR2','UR3','FR2','AZR2L2']
        assert eligible.any() and torch.isfinite(target[eligible]).all()
        b=r['baseline'];qs,ts=match_detection(dict(instance_boxes_cxcywh=b['boxes'][None],instance_objectness_logits=b['objectness_logits'][None]),t['boxes_cxcywh'][:,v],(t['box_valid']&t['instance_valid'])[:,v])[0];recreated=torch.zeros_like(r['valid']);gt=torch.zeros_like(r['gt'])
        for q,j in zip(qs.tolist(),ts.tolist()):
            if b['objectness_logits'][q].sigmoid()<.5 or box_iou(b['boxes'][q:q+1],t['boxes_cxcywh'][0,v,j:j+1])[0,0]<.5:continue
            recreated[q]=known[j];gt[q]=target[j]
        assert torch.equal(recreated,r['valid']) and torch.allclose(gt[recreated],r['gt'][recreated],atol=1e-6,rtol=0),('original target mismatch',i)
        manifests.append(dict(index=i,image=str(path.relative_to(ROOT)),sha256=sha(path),image_hw=list(a.shape[:2])))
        evaluation.append(dict(index=i,metadata=m,target=target,coordinate_valid=known,visibility_valid=vv,visible=visible,eligible=eligible,pixels=r['pixels'],instance_valid=t['instance_valid'][0,v].clone(),qualified_target=r['gt'],qualified_valid=r['valid'],baseline=r['baseline']))
        hashes.update(item['source_hashes']);occ=ROOT/r['record']['scene']/'keypoint_occlusion.npz'
        if occ.exists():hashes[str(occ.relative_to(ROOT))]=sha(occ)
        counts[m['stage']+'_'+m['kind']]+=1;allpointcounts[m['stage']+'_'+m['kind']]+=int(eligible.sum())
        if (number+1)%25==0:print('RGB',number+1,'of',len(chosen),flush=True)
    torch.save(evaluation,OUT/'evaluation_only.pt');write(OUT/'inference_manifest.json',manifests);write(OUT/'source_hashes.json',hashes)
    schedule_rng=random.Random(17);fitpools={k:[z['index'] for z in evaluation if z['metadata']['stage']=='fit' and z['metadata']['kind']==k] for k in ['real','single','dual']};schedule=[[schedule_rng.choice(fitpools[k]) for k in fitpools] for _ in range(800)]
    protocol=dict(seed=17,images=len(chosen),image_counts=dict(counts),eligible_point_counts=dict(allpointcounts),original_pools=pools,indices=chosen,sampling='Use every eligible originalFIT synthetic single/dual row; real128FIT/32internalcal halfphysicalbottom, include original42 thenseed17random remainder. Same original source-stage split. No selection bypoint error. Original42 retained for initialfeatureparity.',labels='ALL coordinatevalid real points (visibilityunknown), ALL explicitlyvisible synthetic points forbothinstances, not merely detectorqualified points. Separate originalqualifiedtarget retained solelyforevaluation. OriginalqualifiedGT/valid parity asserted. Missingpoint excluded; no relabeling.',new_validation=0,roundtrip_exact=True,train_steps=800,schedule=schedule,planned_arms=['frozen_official_backbone','train_official_backbone'],initialization='Same local officialSuperAnimalQuadrupedHRNetW32 and freshsame19,973parameter5pointhead, no pretraineddecodersemantic mapping. Backbonecopyfullweights trainable inadaptarm; all BNrunningbuffers frozen eval inboth.',objective='Same Gaussianmixture spatialCE sigma8px for5types; real/single/dual1each perstep, noGTforward. Diagnosticlocalization notinstancepose/AP.',optimizer='Planned backboneAdamW1e-5 andreadout1e-3 cosinefloor0.1 wd1e-4 clip1; fixed800endpointfirst. Sameimageupdatebudget, extra trainableparams/compute explicitlyreported.',preflight='Verify official nativefeatures againstold42cache, pixel-normalization parity, frozenbuffers, finite gradient through fullbackbone, isolatedsourcehashes; smokeupdatesdiscarded. UseDLCtorch2.7 botharms. No hugefeaturecache: imagePNGsonly.',extension='No automaticextension. InspectFIT/internalcal convergence at200/400/800; extendmatchedarms onlyif evidencejustifies andbefore18:11:41UTC. No externaltest/productionpromotion.',png_bytes=sum((ROOT/z['image']).stat().st_size for z in manifests))
    if full_real:
        protocol.update(sampling='Preserve original362 inoriginalorder and append allremainingeligibleFITreal rows; originalrealFIT128 expandedto821. No newcal orsyntheticimages. No errorselection.',planned_arms=['same_pool_continuation','expanded_real_continuation'],initialization='Both start from SAME train_official_backbone_step0800.pt includingoptimizer; nofrozenarm ornewinitialization.',train_steps=800,optimizer='Continuation: preserve800step optimizerstate; hold lastLRconstant forbotharms. NoLRrestart oraugmentationchanges.',objective='SameGaussianmixtureCE andperstep real/single/dual1each. Matchreal physicalbottom samplingstratum andsyntheticschedule betweenarms; onlyrealFITpool differs.',extension='This dataset file schedule is provisional and MUST NOTbeusedforcontrolledcontinuation; runnerwillpreregister pairedold/newrealstratifiedschedule andidenticalsyntheticdraws. Evaluatecommonoriginal362 only, no newvalidation; deadlineunchanged.')
        protocol.pop('schedule')
    write(OUT/'protocol.json',protocol);print(json.dumps({k:v for k,v in protocol.items() if k not in ['indices','schedule']},indent=2))
if __name__=='__main__':main()
