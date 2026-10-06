"""Synthetic-only unknown-along-curve correspondence study; virtual anchors."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_depth_pose_ablation import fingerprint,write_json
CYCLE=ROOT/'output/surface_curve_cycle_20261003';OUT=CYCLE/'controlled_back_noise_inputs_v1'


def sample(curve,u):
    index=u*(curve.shape[1]-1);lo=index.floor().long().clamp(0,curve.shape[1]-2);alpha=index-lo
    rows=torch.arange(len(curve))[:,None]
    return curve[rows,lo]*(1-alpha[...,None])+curve[rows,lo+1]*alpha[...,None]


def main():
    assert not OUT.exists();OUT.mkdir();(OUT/'visual_review').mkdir();torch.set_num_threads(2)
    ppath=CYCLE/'back_point_probe_inputs_v1/synthetic_inputs.pt';cpath=CYCLE/'expanded_curve_features_v1/frozen_inputs.pt'
    points=torch.load(ppath,map_location='cpu',weights_only=False);curves=torch.load(cpath,map_location='cpu',weights_only=False);assert points['metadata']==curves['metadata'];meta=points['metadata'];d=points['tensors'];curve=curves['tensors']['gt'];n=len(meta)
    u=torch.tensor([.3,.7]).repeat(n,1);clean=sample(curve,u);generator=torch.Generator().manual_seed(20261004)
    noisy_u=u+(torch.rand((n,2),generator=generator)-.5)*.3
    train=torch.tensor([m['role']=='train' for m in meta]);noisy_u[~train]=u[~train]
    tangent=sample(curve,(noisy_u+.005).clamp(0,1))-sample(curve,(noisy_u-.005).clamp(0,1));tangent=tangent*d['pixels'][:,None];length=tangent.norm(dim=-1);tangent=tangent/length.clamp_min(1e-8)[...,None]
    normal=torch.stack([-tangent[...,1],tangent[...,0]],-1);jitter=torch.randn((n,2,1),generator=generator)*2.;jitter[~train]=0
    noisy=sample(curve,noisy_u)+normal*jitter/d['pixels'][:,None]
    valid=torch.isfinite(clean).all(-1)&torch.isfinite(noisy).all(-1)&(clean>=0).all(-1)&(clean<=1).all(-1)&(noisy>=0).all(-1)&(noisy<=1).all(-1)
    assert torch.equal(noisy[~train],clean[~train])
    chord=(noisy[:,1]-noisy[:,0])*d['pixels'];chord_length=chord.norm(dim=-1);chord=chord/chord_length.clamp_min(1e-8)[:,None]
    chord_t=chord[:,None].expand(-1,2,-1).clone();chord_r=((chord_length>=10)&valid.all(-1)).float()[:,None].expand(-1,2).clone()*.5
    oracle_r=(length>1e-6).float();values=dict(pose=d['pose'],coarse_xy=d['coarse_xy'],pixels=d['pixels'],clean_xy=clean,noisy_xy=noisy,valid=valid,
        chord_tangent=chord_t,chord_reliability=chord_r,oracle_tangent=tangent,oracle_reliability=oracle_r,clean_u=u,noisy_u=noisy_u,curve_gt=curve)
    torch.save(dict(tensors=values,metadata=meta),OUT/'inputs.pt')
    noise=((noisy-clean)*d['pixels'][:,None]).norm(dim=-1)[train[:,None]&valid]
    report=dict(source_point_sha256=fingerprint(ppath),source_curve_sha256=fingerprint(cpath),observations=n,train=int(train.sum()),validation=int((~train).sum()),
        anchors='VIRTUAL A/B atknownsyntheticmaterialu0.30/0.70, NOT redefinedCore23points06/07. Originallabels untouched. No realdata loaded.',
        corruption='TRAINonly: independentper-view/anchor fixeduniformmaterialoffset[-.15,+.15] plusGaussian2pxnormaljitter. Fixednoiseseed20261004 sharedacrossallmodelseeds. Notmeasuredhuman-noisedistribution.',
        targets='CleanGT keptforevaluation andclean-supervisedreferenceonly. Noisy labels forstrict/chordweak/oracletangentweak experimentalarms. Oracleaxis usestruecurvetangent andmustbelabeledupperbound, notdeployable.',
        forward='Onlyoriginalfrozenpose/imagefeatures andoriginalcoarse06/07priors (imperfectpriorforvirtualanchors) allowed. NoGTcurve/materialu/tangent inforward.',
        proposed_controls=['clean_strict_reference','noisy_strict','noisy_weak_chord_reliability0.5','noisy_weak_oracle_reliability1_upperbound'],
        proposed_budget='Same3seeds17/29/43,2400updates,sameinit/syntheticdataset-uniformbatches,strictHuber versusexistingalongtolerance4normal1loss. Fixedvalidation andsourcegroups.',
        metrics='Cleanvirtualanchorerror AND distanceofpredictedpointtotruecurve; latterdoesnotestablishmaterialcorrespondence. Trainfitandheldsourceposemetrics separated. Neverjudgeonlydistance tonoisylabels.',
        train_noise_px=dict(mean=float(noise.mean()),median=float(noise.median()),p90=float(torch.quantile(noise,.9))),
        limitations='Controlledsynthetichypothesisdiagnostic, notrealperformanceclaim. Sameassets andfrozenpretrainedfeatures. Fullcurvegeometry onlyforGT/explicitoraclecontrol, neverinferenceinputs.')
    write_json(OUT/'protocol.json',report)
    groups=collections.defaultdict(list)
    for i,m in enumerate(meta):
        if m['role']=='train':groups[m['dataset']].append(i)
    panels=[]
    for dataset,ids in sorted(groups.items()):
        i=ids[len(ids)//2];m=meta[i];images=list((ROOT/m['scene']/'blended_images').glob(f'{m["view"]:08d}.*'));assert len(images)==1
        w,h=map(int,d['pixels'][i].tolist())
        with Image.open(images[0]) as im:im=im.convert('RGB').resize((w,h))
        draw=ImageDraw.Draw(im);line=curve[i].numpy()*[w,h];draw.line([tuple(p) for p in line],fill='#ffe454',width=2)
        for k in range(2):
            x,y=clean[i,k].numpy()*[w,h];draw.ellipse((x-4,y-4,x+4,y+4),outline='#55ff55',width=2);draw.text((x+4,y-12),'AB'[k],fill='#55ff55')
            x,y=noisy[i,k].numpy()*[w,h];draw.line((x-5,y,x+5,y),fill='#ff5555',width=2);draw.line((x,y-5,x,y+5),fill='#ff5555',width=2)
        name=f'{i:03d}.jpg';im.save(OUT/'visual_review'/name,quality=95);panels.append(dict(file=name,observation=i,**m))
    html=['<!doctype html><meta charset="utf-8"><title>Controlled synthetic annotation noise</title><style>body{background:#202027;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Virtual-anchor TRAIN annotation perturbations</h1><p>Yellow=knownmaterialsurfacecurve. GreenA/B=cleanvirtualanchorsu0.30/0.70, NOTCore23redefinitions. Redcross=perturbedTRAINlabelalongcurve plus2pxnormalnoise. Thisisachosenhypothesisstress, NOTobservedhuman-errorstatistics. NoREALlabelsmodified. No training yet.</p>']
    for p in panels:html.append(f'<h2>{p["scene"]}, view{p["view"]}</h2><img src="{p["file"]}">')
    (OUT/'visual_review/index.html').write_text('\n'.join(html));write_json(OUT/'visual_review/manifest.json',panels);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
