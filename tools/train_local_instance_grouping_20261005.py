"""Bounded matched, full-backbone RGB grouping: smoke overfit or 640-step pilot.

Each update uses exactly one real, one single synthetic and one dual synthetic
image. Arms share all initialization, image draws and rotations; only the
within-instance point-attention mask differs. No development/locked input.
"""
import os
os.environ['DLClight'] = 'True'
import sys, json, datetime as dt, argparse, fcntl, time, random, traceback, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont
from preflight_mixed_local_backbone_20261005 import build_backbone, load_rgb, sha, digest
from train_mixed_local_backbone_20261005 import buffers_digest, atomic_save
from mousegeoforge.models.mixed_local_evidence import MixedLocalEvidence
from mousegeoforge.models.local_point_evidence_probe import point_evidence_loss
from mousegeoforge.models.quarter_turn_evidence import quarter_turn
from mousegeoforge.models.local_instance_grouping import (
    LocalInstanceGrouping, grouping_loss, decode_grouped, rotate_mask_centroids,
)

C = ROOT / 'output/local_grouping_cycle_20261005'
OLD = ROOT / 'output/singleview_fusion_cycle_20261005'
INPUT = OLD / 'mixed_local_fulltrain_rgb_inputs_v1'
SOURCE = OLD / 'mixed_local_backbone800_v1/train_official_backbone_step0800.pt'
ARMS = ['independent', 'within_instance']
OUT = None


def now():
    return dt.datetime.now(dt.timezone.utc)


def read(path):
    return torch.load(path, weights_only=False, map_location='cpu')


def write(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metrics(output, decoded, row, supervision, hw):
    """Hungarian posthoc center matching; no GT is passed to inference/decoding."""
    scale = torch.tensor([hw[1], hw[0]])
    diagonal = float(scale.float().norm())
    active = decoded['instance_active'].nonzero().flatten()
    gt = supervision['instance_valid'].nonzero().flatten()
    matches = {}
    center_errors = []
    true_positive = 0
    if len(active) and len(gt):
        distance = torch.cdist(decoded['centers'][active] * scale,
                               supervision['mask_centroid'][gt] * scale)
        a, b = linear_sum_assignment(distance.numpy())
        for pa, pb in zip(a, b):
            d = float(distance[pa, pb]); center_errors.append(d)
            true_positive += int(d <= 64.)
            matches[int(gt[pb])] = (int(active[pa]), d)
    points = []
    for g in gt.tolist():
        for k in row['eligible'][g].nonzero().flatten().tolist():
            known_xy = row['target'][g, k]
            candidate_distance = ((output['candidate_xy'][k] - known_xy) * scale).norm(dim=-1)
            candidate_distance = candidate_distance.masked_fill(~output['candidate_available'][k], diagonal)
            oracle = float(candidate_distance.min())
            error = diagonal; accepted = False; joint = diagonal
            if g in matches:
                query, center_error = matches[g]
                accepted = bool(decoded['accepted'][query, k])
                if accepted:
                    error = min(diagonal, float(((decoded['xy'][query, k] - known_xy) * scale).norm()))
                    if center_error <= 64.:
                        joint = error
            points.append(dict(instance=g, keypoint=k, error=error, joint64_error=joint,
                               accepted=accepted, oracle=oracle))
    return dict(points=points, gt_count=len(gt), predicted_count=len(active),
                count_correct=int(len(gt) == len(active)), center_errors=center_errors,
                tp64=true_positive, fp64=len(active)-true_positive, fn64=len(gt)-true_positive)


def summarize(details):
    result = {}
    for stage in ['fit', 'calibration']:
        result[stage] = {}
        for kind in ['real', 'single', 'dual', 'real_bottom_feet']:
            subset = [d for d in details if d['metadata']['stage'] == stage and
                      (d['metadata']['kind'] == kind or (kind == 'real_bottom_feet' and
                       d['metadata']['kind'] == 'real' and d['metadata']['physical_bottom']))]
            points = [p for d in subset for p in d['metrics']['points']
                      if kind != 'real_bottom_feet' or p['keypoint'] > 0]
            if not subset:
                continue
            m = dict(images=len(subset), points=len(points),
                     loss=float(np.mean([d['loss']['total'] for d in subset])),
                     count_accuracy=float(np.mean([d['metrics']['count_correct'] for d in subset])),
                     tp64=sum(d['metrics']['tp64'] for d in subset),
                     fp64=sum(d['metrics']['fp64'] for d in subset),
                     fn64=sum(d['metrics']['fn64'] for d in subset))
            if points:
                m.update(error_px=float(np.mean([p['error'] for p in points])),
                         joint64_error_px=float(np.mean([p['joint64_error'] for p in points])),
                         accepted_fraction=float(np.mean([p['accepted'] for p in points])),
                         pck16=float(np.mean([p['error'] <= 16 for p in points])),
                         joint64_pck16=float(np.mean([p['joint64_error'] <= 16 for p in points])),
                         candidate_oracle_px=float(np.mean([p['oracle'] for p in points])))
            result[stage][kind] = m
    return result


def render_predictions(directory, arm, predictions, rows, manifest, selected):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    colors = ['#f33d42', '#2bd6f5', '#f472df', '#8ed641', '#ffc736']
    for i in selected:
        if i not in predictions:
            continue
        rgb = Image.open(ROOT / manifest[i]['image']).convert('RGB')
        w, h = rgb.size
        canvas = Image.new('RGB', (w*2, h+45), 'white')
        canvas.paste(rgb, (0, 40)); canvas.paste(rgb, (w, 40))
        draw = ImageDraw.Draw(canvas)
        decoded = predictions[i]['decoded']
        draw.text((4, 4), f'{arm} index{i}: known GT | RGB-only prediction; count {int(decoded["instance_active"].sum())}', font=font, fill='black')
        for g, k in rows[i]['eligible'].nonzero().tolist():
            x, y = (rows[i]['target'][g, k] * torch.tensor([w, h])).tolist()
            draw.ellipse((x-3,y+37,x+3,y+43), outline=colors[k], width=2)
            draw.text((x+4,y+40), f'{g}:{[1,15,16,17,18][k]}', fill=colors[k], font=font)
        for q in range(len(decoded['object_probability'])):
            center = decoded['centers'][q] * torch.tensor([w,h])
            x,y = center.tolist()
            draw.text((w+x,y+40), f'q{q} p={float(decoded["object_probability"][q]):.2f}', fill='white',font=font)
            if not decoded['instance_active'][q]:
                continue
            for k in decoded['accepted'][q].nonzero().flatten().tolist():
                x,y = (decoded['xy'][q,k] * torch.tensor([w,h])).tolist(); x += w
                draw.ellipse((x-4,y+36,x+4,y+44), outline=colors[k], width=2)
                draw.text((x+5,y+40), f'{q}:{[1,15,16,17,18][k]}', fill=colors[k],font=font)
        canvas.save(directory / f'{arm}_{i}.jpg', quality=93)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['overfit', 'overfit256', 'pilot'], required=True)
    args = parser.parse_args()
    is_overfit = args.stage != 'pilot'
    steps = {'overfit':32, 'overfit256':256, 'pilot':640}[args.stage]
    initial_step = 32 if args.stage == 'overfit256' else 0
    checkpoints = {'overfit':[0,16,32], 'overfit256':[32,96,160,256], 'pilot':[0,160,320,640]}[args.stage]
    OUT = C / f'grouping_{args.stage}_v1'
    plan = json.loads((C / 'plan.json').read_text())
    deadline = dt.datetime.fromisoformat(plan['training_stop_utc'].replace('Z', '+00:00'))
    assert now() < deadline
    if args.stage == 'pilot':
        preflight = json.loads((C / 'grouping_overfit256_v1/results.json').read_text())
        assert preflight['gradient_finite_and_nonzero'] and preflight['bn_unchanged']
        assert all(preflight['overfit_loss_decreased'].values()), 'Do not launch larger pilot before overfit gate passes'
        assert all(preflight['instance_overfit_gate'].values()), 'Resolve count/center separation before larger pilot'
    lock = (ROOT / plan['gpu_lock']).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = json.loads((C / 'state.json').read_text())
    state.update(stage=f'grouping_{args.stage}_running', active_job=str(OUT.relative_to(ROOT)),
                 active_pid=os.getpid(), updated_utc=now().isoformat(),
                 next='Wait for matched grouping job; inspect status/failure, do not duplicate.')
    write(C / 'state.json', state)

    def status(stage, **kw):
        write(OUT / 'status.json', dict(stage=stage, pid=os.getpid(), utc=now().isoformat(), **kw))

    status('loading')
    rows = {r['index']: r for r in read(INPUT / 'evaluation_only.pt')}
    manifest = {r['index']: r for r in json.loads((INPUT / 'inference_manifest.json').read_text())}
    sup_path = C / 'instance_supervision_audit_v1/evaluation_instance_targets.pt'
    supervision = {r['index']: r for r in read(sup_path)}
    assert set(rows) == set(supervision) and len(rows) == 1055
    for i, r in rows.items():
        assert r['metadata']['stage'] in ['fit', 'calibration']
        assert r['metadata'].get('source') not in ['FR2', 'AZR2L2', 'GR2', 'UR2', 'UR3']
        assert r['metadata'] == supervision[i]['metadata']
        assert torch.equal(r['instance_valid'].bool(), supervision[i]['instance_valid'])
        assert torch.equal(supervision[i]['instance_valid'], supervision[i]['centroid_valid'])
        # Completeness is the existing single-mouse curation/synthetic scene
        # contract, not something established from partial keypoint labels.
        assert bool(supervision[i]['detection_annotation_complete'])
    fixed_fit = [next(i for i, r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k)
                 for k in ['real','single','dual']]
    real_pools = {b: [i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']=='real'
                     and r['metadata']['physical_bottom']==b] for b in [True,False]}
    synthetic = {k: [i for i,r in rows.items() if r['metadata']['stage']=='fit' and r['metadata']['kind']==k]
                 for k in ['single','dual']}
    rng = random.Random(17); rotation_rng = random.Random(43)
    schedule = [fixed_fit[:] for _ in range(steps)] if is_overfit else [
        [rng.choice(real_pools[j%2==0]), rng.choice(synthetic['single']), rng.choice(synthetic['dual'])]
        for j in range(steps)]
    turns = [[0,0,0] for _ in range(steps)] if is_overfit else [
        [rotation_rng.randrange(4) for _ in range(3)] for _ in range(steps)]
    eval_indices = fixed_fit if is_overfit else fixed_fit + [i for i,r in rows.items() if r['metadata']['stage']=='calibration']
    selected = list(dict.fromkeys(fixed_fit + [next(i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']==k)
                 for k in ['real','single','dual']] + [i for i,r in rows.items() if r['metadata']['stage']=='calibration' and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']]))
    source_hash = sha(SOURCE); supervision_hash = sha(sup_path)
    assert source_hash == '601505a17ddd90ea374c1f6c2f6cb98d8692a6a360e268c7eae20134d096d8b7'
    shared = LocalInstanceGrouping(interaction=False).state_dict()
    atomic_save(OUT / 'shared_grouping_initial.pt', shared)
    models = {}; optimizers = {}; bn = {}
    for arm in ARMS:
        backbone, _ = build_backbone()
        evidence = MixedLocalEvidence(backbone, adapt_backbone=True).cuda()
        cp = read(SOURCE)
        evidence.backbone.load_state_dict(cp['backbone'], strict=True)
        evidence.readout.load_state_dict(cp['readout'], strict=True)
        evidence.train(); del cp
        assert digest(evidence.backbone) == '18594fc7cd3a5587ce595ceac8c00c34f99c5572cffa6bcc5fec2a05ca753692'
        group = LocalInstanceGrouping(interaction=arm=='within_instance').cuda()
        group.load_state_dict(shared, strict=True)
        models[arm] = (evidence, group)
        bn[arm] = buffers_digest(evidence.backbone)
        optimizers[arm] = torch.optim.AdamW([
            dict(params=group.parameters(), lr=3e-4),
            dict(params=evidence.readout.parameters(), lr=1e-4),
            dict(params=evidence.backbone.parameters(), lr=1e-6)], weight_decay=1e-4)
    assert digest(models[ARMS[0]][0]) == digest(models[ARMS[1]][0])
    assert digest(models[ARMS[0]][1]) == digest(models[ARMS[1]][1])
    resume_hashes = {}
    if initial_step:
        prior = C / 'grouping_overfit_v1'
        assert json.loads((prior/'status.json').read_text())['stage']=='complete'
        for arm in ARMS:
            path = prior/f'{arm}_step0032.pt'; resume_hashes[arm] = sha(path)
            cp = read(path); assert cp['step']==32 and cp['arm']==arm and cp['source_sha256']==source_hash
            evidence, group = models[arm]
            evidence.backbone.load_state_dict(cp['backbone'],strict=True)
            evidence.readout.load_state_dict(cp['readout'],strict=True)
            group.load_state_dict(cp['grouping'],strict=True)
            optimizers[arm].load_state_dict(cp['optimizer']);del cp
            assert buffers_digest(evidence.backbone)==bn[arm]
    protocol = dict(stage=args.stage, steps=steps, initial_step=initial_step, resume_sha256=resume_hashes,
                    evaluation_steps=checkpoints, seed=17, rotations_seed=43,
                    schedule=schedule, rotations=turns, source_sha256=source_hash,
                    source_checkpoint=str(SOURCE.relative_to(ROOT)), supervision_sha256=supervision_hash,
                    grouping_parameters=sum(p.numel() for p in models[ARMS[0]][1].parameters()),
                    trainable_backbone_parameters=sum(p.numel() for p in models[ARMS[0]][0].backbone.parameters()),
                    same_all_initial_weights_at_step0=True, optimizer='Fresh AdamW atstep0 both arms; overfit256 preserves each32-step model and optimizer. Constant LR group3e-4/local1e-4/backbone1e-6; fixed BN.',
                    difference='Within-instance five-type self-attention full versus diagonal mask. Identical parameters, candidate extractor and losses.',
                    architecture='Two learned instance queries over pooled full-image HRNet features. Projected mask-centroid L1 supervised in loss. Candidate features + relative predicted center geometry + null. RGB-only inference; no GT count/center/box.',
                    loss='local Gaussian-mixture CE + candidate-assignment CE + object BCE + 10*normalized-center L1; unknown point labels ignored. Candidate-null supervised only when known point lacks candidate within16px.',
                    inference='At most two mice (pilot capacity, not supplied GT count); object probability>=0.5, per-type candidate/null assignment without reuse. Rejection is not a visibility prediction.',
                    evaluation='CAL89 + fixedFIT3 for pilot, fixedFIT3 only for overfit. GT center matching posthoc, diagonal penalty for rejection/missedinstance, joint64 includes center64px gate. Candidate oracle diagnostic only; no AP claim.',
                    selected_pool_caveat='Existing1055 includes historical baseline/known-point selection, not unbiased all-source benchmark.',
                    code_sha256=sha(Path(__file__)), module_sha256=sha(ROOT/'mousegeoforge/models/local_instance_grouping.py'),
                    deadline=plan['training_stop_utc'], production_default_changed=False)
    write(OUT / 'protocol.json', protocol)
    needed = set(eval_indices) | {i for ids in schedule for i in ids}
    images = {}
    for n,i in enumerate(sorted(needed)):
        if now() >= deadline: raise TimeoutError('Input cutoff')
        images[i] = load_rgb(manifest[i]).cuda()
        if (n+1)%100 == 0: status('loading', images=n+1, total=len(needed))
    curves=[]; histories={a:[] for a in ARMS}; gradients={a:[] for a in ARMS}
    seconds={a:0. for a in ARMS}; peak=0; initial_loss={}; final_loss={}; predictions={}

    def compute(arm, i, turn=0):
        evidence, group = models[arm]
        r = rows[i]; s = supervision[i]; hw = manifest[i]['image_hw']
        target = r['target'].cuda(); eligible=r['eligible'].cuda(); centers=s['mask_centroid'].cuda()
        center_valid=s['centroid_valid'].cuda(); x=images[i]
        if turn:
            x,target,hw=quarter_turn(x,target,eligible,hw,turn)
            centers=rotate_mask_centroids(centers,center_valid,turn)
        local=evidence(x,hw); output=group(local,hw)
        losses=grouping_loss(output,target,eligible,s['instance_valid'].cuda(),centers,center_valid,hw,
                             annotation_complete=bool(s['detection_annotation_complete']))
        local_loss=point_evidence_loss(local,target,eligible,hw)
        losses['grouping']=losses['total']; losses['local']=local_loss
        losses['total']=losses['total']+local_loss
        return output,losses

    @torch.no_grad()
    def evaluate(arm, step):
        details=[]; pred={}
        for i in eval_indices:
            if now() >= deadline: raise TimeoutError('Evaluation cutoff')
            out,losses=compute(arm,i)
            decoded=decode_grouped(out)
            cpu={k:v.detach().cpu() for k,v in out.items()}
            dec={k:v.detach().cpu() for k,v in decoded.items()}
            details.append(dict(index=i,metadata=rows[i]['metadata'],
                                loss={k:float(v) for k,v in losses.items() if v.ndim==0},
                                metrics=metrics(cpu,dec,rows[i],supervision[i],manifest[i]['image_hw'])))
            pred[i]=dict(output=cpu,decoded=dec)
        summary=summarize(details)
        curves.append(dict(arm=arm,step=step,summary=summary))
        write(OUT/'curves.json',curves);write(OUT/f'{arm}_step{step:04d}_details.json',details)
        atomic_save(OUT/f'{arm}_step{step:04d}_predictions.pt',pred)
        return pred,float(np.mean([d['loss']['total'] for d in details]))

    def checkpoint(arm,step):
        evidence,group=models[arm]
        atomic_save(OUT/f'{arm}_step{step:04d}.pt',dict(
            backbone={k:v.cpu() for k,v in evidence.backbone.state_dict().items()},
            readout={k:v.cpu() for k,v in evidence.readout.state_dict().items()},
            grouping={k:v.cpu() for k,v in group.state_dict().items()},
            optimizer=optimizers[arm].state_dict(), step=step, arm=arm, source_sha256=source_hash))

    start=time.monotonic()
    for arm in ARMS:
        status('initial_evaluation',arm=arm,step=initial_step)
        predictions[arm],initial_loss[arm]=evaluate(arm,initial_step)
        if initial_step:
            old_predictions=read(C/'grouping_overfit_v1'/f'{arm}_step0032_predictions.pt')
            for i in eval_indices:
                for key,value in predictions[arm][i]['output'].items():
                    assert torch.equal(value,old_predictions[i]['output'][key]),(arm,i,key,'resume parity')
    for step,ids in enumerate(schedule[initial_step:],initial_step+1):
        for arm in ARMS:
            if now() >= deadline:
                for name in ARMS: checkpoint(name,initial_step+len(histories[name]))
                raise TimeoutError('Training cutoff; both latest actual steps preserved')
            evidence,group=models[arm]; optimizer=optimizers[arm]
            optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t=time.monotonic(); value=0.
            for slot,i in enumerate(ids):
                out,losses=compute(arm,i,turns[step-1][slot]); loss=losses['total']/3
                assert torch.isfinite(loss); loss.backward(); value+=float(loss.detach())
                del out,losses,loss
            if step in [initial_step+1,initial_step+2,steps]:
                norms={name:sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
                       for name,module in [('backbone',evidence.backbone),('readout',evidence.readout),('group',group)]}
                assert all(np.isfinite(z) and z>0 for z in norms.values()),norms
                gradients[arm].append(dict(step=step,norms=norms))
            torch.nn.utils.clip_grad_norm_(list(evidence.parameters())+list(group.parameters()),1,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();seconds[arm]+=time.monotonic()-t
            peak=max(peak,torch.cuda.max_memory_allocated());histories[arm].append(value)
        if step%8==0:
            status('training',step=step,total=steps,seconds=seconds,recent_loss={a:float(np.mean(histories[a][-8:])) for a in ARMS})
            print('STEP',step,{a:float(np.mean(histories[a][-8:])) for a in ARMS},seconds,flush=True)
        if step in checkpoints:
            for arm in ARMS:
                assert buffers_digest(models[arm][0].backbone)==bn[arm]
                checkpoint(arm,step);status('evaluating',arm=arm,step=step,total=steps)
                predictions[arm],final_loss[arm]=evaluate(arm,step)
    assert sha(SOURCE)==source_hash and sha(sup_path)==supervision_hash
    assert all(sha(ROOT/manifest[i]['image'])==manifest[i]['sha256'] for i in needed)
    assert all(sha(C/'grouping_overfit_v1'/f'{arm}_step0032.pt')==value for arm,value in resume_hashes.items())
    decreased={a:final_loss[a]<initial_loss[a] for a in ARMS}
    summary={a:next(z['summary'] for z in reversed(curves) if z['arm']==a) for a in ARMS}
    instance_gate={a:all(summary[a]['fit'][k]['count_accuracy']==1. and
                       summary[a]['fit'][k]['fp64']==0 and summary[a]['fit'][k]['fn64']==0
                       for k in ['real','single','dual']) for a in ARMS}
    result=dict(stage=args.stage,steps=steps,gradient_finite_and_nonzero=True,gradients=gradients,
                bn_unchanged=True,sources_unchanged=True,overfit_loss_decreased=decreased if is_overfit else {},
                instance_overfit_gate=instance_gate if is_overfit else {},resume_exact_parity=bool(initial_step),
                initial_evaluation_loss=initial_loss,final_evaluation_loss=final_loss,
                training_seconds=seconds,total_seconds=time.monotonic()-start,peak_gpu_bytes=peak,
                histories=histories,production_changed=False,summary=summary)
    write(OUT/'results.json',result)
    examples=OUT/'examples';examples.mkdir()
    for arm in ARMS:render_predictions(examples,arm,predictions[arm],rows,manifest,selected)
    page=['<!doctype html><meta charset="utf-8"><title>局部实例分组对照</title><style>body{font:18px system-ui;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}</style>',
          f'<h1>{args.stage}: 独立点 / 实例内交互，固定{steps}步</h1>',
          '<p>两组均从RGB预测实例数量和点分配；未知点不计错、拒绝和漏检按图像对角线惩罚。候选oracle不是模型成绩。此处没有独立外部测试。</p>',
          '<pre>'+json.dumps({k:v for k,v in result.items() if k!='histories'},indent=2)+'</pre>']
    for p in sorted(examples.glob('*.jpg')):page.append(f'<h3>{p.stem}</h3><img src="examples/{p.name}">')
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/OUT.name
    dest.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ['.json','.html']:shutil.copy2(p,dest/p.name)
    shutil.copytree(examples,dest/'examples',dirs_exist_ok=True)
    status('complete',step=steps)
    state.update(stage=f'grouping_{args.stage}_complete',active_job=None,active_pid=None,
                 next='Review curves and visuals. Overfit gate before640 matched pilot. If pilot completed, decide fixed3200 experiment; no automatic model promotion.',
                 updated_utc=now().isoformat())
    state['completed'].append(f'grouping_{args.stage}_{steps}_matched');write(C/'state.json',state)
    print('COMPLETE',json.dumps({k:v for k,v in result.items() if k!='histories'}),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        if OUT is not None and OUT.exists():
            write(OUT/'failure.json',dict(utc=now().isoformat(),traceback=traceback.format_exc()))
            write(OUT/'status.json',dict(stage='failed',pid=os.getpid(),utc=now().isoformat()))
        raise
