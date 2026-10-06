"""Frozen paired mask-box real-development replay. No manual accuracy claims."""
import fcntl
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
import torch
from run_depth_pose_ablation import ROOT, VGGTOmega, DepthPoseModel, fingerprint, write_json
from mousegeoforge.adaptive_temperature import install_adaptive_temperature, restore_adaptive
from mousegeoforge.mask_box_candidate_safe import MaskBoxCandidate
from mousegeoforge.mask_box_candidate import tensor_mapping_sha256

SOURCE = ROOT / 'output/real_cylinder_bottom_frozen_v1'
OUT = ROOT / 'output/real_cylinder_mask_box_v1_retry3'
DEADLINE = datetime(2026, 9, 12, 17, 19, 27, tzinfo=timezone.utc).timestamp()
BOX = 'instance_boxes_cxcywh'
BODY = list(range(7)) + list(range(14, 23))


def corners(box):
    x, y, w, h = box
    return np.array([x-w/2, y-h/2, x+w/2, y+h/2])


def box_iou(a, b):
    a, b = corners(a), corners(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    areas = (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod()
    return float(inter / max(areas-inter, 1e-12))


def mask_geometry(logits, h, w):
    mask = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), 8)
    if count <= 1:
        return mask, None
    k = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    x, y, bw, bh = stats[k, :4]
    return mask, np.array([(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h])


def main():
    began = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        (OUT/'review').mkdir()
        def status(state, **kwargs):
            write_json(OUT/'status.json', dict(state=state, pid=os.getpid(), elapsed_s=time.monotonic()-began, **kwargs))
        status('preparing')
        try:
            protocol = json.loads((SOURCE/'protocol.json').read_text())
            checkpoint = ROOT/protocol['head_paths']['cylinder800']
            encoder = ROOT/'vggt-omega/vggt_omega_1b_512.pt'
            candidate = MaskBoxCandidate.load(ROOT/'output/mask_box_candidates_v1/seed17',
                base_checkpoint=checkpoint, encoder_checkpoint=encoder, repository_root=ROOT)
            spatial = VGGTOmega(enable_camera=False, enable_depth=True, enable_multi_instance_pose=True, num_keypoints=23)
            weights = torch.load(encoder, map_location='cpu', weights_only=True, mmap=True)
            loaded = spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.', 'dense_head.'))}, strict=False)
            assert not loaded.unexpected_keys and all(k.startswith('instance_pose_head.') for k in loaded.missing_keys)
            model = install_adaptive_temperature(DepthPoseModel.from_spatial_model(spatial)).cuda().eval()
            restore_adaptive(model, torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
            del weights, spatial
            flags = {n:p.requires_grad for n,p in model.named_parameters()}
            original_hash = tensor_mapping_sha256(model.state_dict())
            write_json(OUT/'protocol.json', dict(source_protocol_sha256=fingerprint(SOURCE/'protocol.json'),
                candidate='output/mask_box_candidates_v1/seed17', sources=protocol['sources'], conditions=protocol['conditions'],
                network_hw=[768,704], train_steps=0, deadline_utc='2026-09-12T17:19:27Z',
                scope='Predicted-mask geometric consistency and translation equivariance, NOT manual box/keypoint accuracy.',
                code_sha256=fingerprint(Path(__file__)), holdout_accessed=False))
            rows, translations, records, on_original = [], [], {}, {}
            replay_max_abs = {}
            replay_semantics = []
            count = 0
            with candidate.capture(model, enabled=True):
                for session, frames in protocol['sources'].items():
                    for frame in frames:
                        for condition, cameras in protocol['conditions'].items():
                            if time.time() >= DEADLINE:
                                raise TimeoutError('Authorized six-hour window ended')
                            rgb = [cv2.cvtColor(cv2.imread(str(SOURCE/f'review/{session}_{frame:06d}_{cam}_input.png')), cv2.COLOR_BGR2RGB) for cam in cameras]
                            x = torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).unsqueeze(0).cuda().float()/255
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                cached, start = model.aggregator(x)
                            with torch.inference_mode():
                                pred = model.forward_cached(x, cached, start, coupling='off')
                                assert candidate(pred, enabled=False) is pred
                                updated = candidate(pred, tuple(x.shape[-2:]), enabled=True)
                            assert all(updated[k] is v for k,v in pred.items() if k != BOX)
                            assert torch.isfinite(updated[BOX]).all()
                            old = torch.load(SOURCE/f'predictions/{session}_{frame:06d}_{condition}.pt', map_location='cpu', weights_only=True)['outputs']['cylinder800']
                            for k,v in old.items():
                                replay_max_abs[k] = max(replay_max_abs.get(k, 0.), float((pred[k].cpu()-v).abs().max()))
                                atol = 1e-4 if k=='instance_mask_logits' else 1e-5
                                torch.testing.assert_close(pred[k].cpu(), v, atol=atol, rtol=1e-5)
                            # Additional semantic checks: near-zero mask-logit rounding must not
                            # alter instance selection/count or materially change geometry.
                            score_now=pred['instance_objectness_logits'].cpu().detach()
                            score_old=old['instance_objectness_logits']
                            assert torch.equal(score_now>=0,score_old>=0)
                            assert torch.equal(score_now.argmax(-1),score_old.argmax(-1))
                            mask_flip=float(((pred['instance_mask_logits'].cpu()>0)!=(old['instance_mask_logits']>0)).float().mean())
                            pose_delta=float((pred['instance_keypoint_xy'].cpu()-old['instance_keypoint_xy']).abs().max())
                            box_delta=float((pred[BOX].cpu()-old[BOX]).abs().max())
                            assert mask_flip<1e-4 and max(pose_delta,box_delta)<1e-5
                            replay_semantics.append(dict(session=session,frame=frame,condition=condition,mask_flip_fraction=mask_flip,
                                normalized_pose_delta=pose_delta,normalized_box_delta=box_delta))
                            records[f'{session}_{frame:06d}_{condition}'] = updated[BOX].cpu()
                            panels = []
                            for j, cam in enumerate(cameras):
                                scores = pred['instance_objectness_logits'][0,j].sigmoid().detach().cpu().numpy().reshape(-1)
                                q = int(scores.argmax()); h,w = rgb[j].shape[:2]
                                boxes0 = pred[BOX][0,j].detach().cpu().numpy()
                                boxes1 = updated[BOX][0,j].cpu().numpy()
                                mask, reference = mask_geometry(old['instance_mask_logits'][0,j,q].numpy(), h, w)
                                row = dict(session=session, frame=frame, condition=condition, camera=cam, query=q,
                                    positive_slots=int((scores>=.5).sum()), box_center_change_px=float(np.linalg.norm((boxes1[q,:2]-boxes0[q,:2])*[w,h])),
                                    off_box=boxes0[q].tolist(), on_box=boxes1[q].tolist(), reference_is_predicted_mask_not_gt=True)
                                if reference is not None:
                                    row.update(off_iou_to_predicted_mask_box=box_iou(boxes0[q], reference), on_iou_to_predicted_mask_box=box_iou(boxes1[q], reference))
                                rows.append(row)
                                images = []
                                for label, boxes in [('OFF', boxes0), ('ON', boxes1)]:
                                    image = cv2.cvtColor(rgb[j],cv2.COLOR_RGB2BGR)
                                    image[mask] = (.6*image[mask]+.4*np.array([0,220,0])).astype(np.uint8)
                                    for slot in np.where(scores>=.5)[0]:
                                        p = np.rint(corners(boxes[slot])*[w,h,w,h]).astype(int)
                                        cv2.rectangle(image,tuple(p[:2]),tuple(p[2:]),(255,0,255) if slot==q else (255,255,0),2)
                                    points = old['instance_keypoint_xy'][0,j,q].numpy()*[w,h]
                                    for point in points[BODY]:cv2.circle(image,tuple(np.rint(point).astype(int)),3,(0,255,255),-1)
                                    cv2.rectangle(image,(0,0),(w,35),(20,20,20),-1)
                                    cv2.putText(image,f'{session} {frame} {cam} {label} N={row["positive_slots"]}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                                    images.append(cv2.resize(image,(352,384)))
                                panels.append(np.concatenate(images,axis=1))
                            cv2.imwrite(str(OUT/f'review/{session}_{frame:06d}_{condition}.png'),np.concatenate(panels,axis=0))
                            if session=='AZL3' and condition=='bottom':on_original[frame]=updated[BOX][0,0].cpu().numpy()
                            count += 1; status('replaying', forwards=count, total=57)
                            del pred, updated, cached, x
                for frame in protocol['sources']['AZL3']:
                    if time.time() >= DEADLINE:raise TimeoutError('Authorized window ended')
                    image=cv2.cvtColor(cv2.imread(str(SOURCE/f'review/AZL3_{frame:06d}_A_input.png')),cv2.COLOR_BGR2RGB)
                    h,w=image.shape[:2]; matrix=np.float32([[1,0,64],[0,1,0]])
                    shifted=cv2.warpAffine(image,matrix,(w,h))
                    x=torch.from_numpy(shifted).permute(2,0,1)[None,None].cuda().float()/255
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(x)
                    with torch.inference_mode():
                        pred=model.forward_cached(x,cached,start,coupling='off')
                        updated=candidate(pred,tuple(x.shape[-2:]),enabled=True)
                    old=torch.load(SOURCE/f'predictions/AZL3_{frame:06d}_bottom.pt',map_location='cpu',weights_only=True)['outputs']['cylinder800']
                    q=int(old['instance_objectness_logits'][0,0].sigmoid().argmax())
                    mask,_=mask_geometry(old['instance_mask_logits'][0,0,q].numpy(),h,w)
                    expected=cv2.warpAffine(np.uint8(mask),matrix,(w,h))>0
                    masks=[mask_geometry(m,h,w)[0] for m in pred['instance_mask_logits'][0,0].detach().cpu().numpy()]
                    ious=[np.count_nonzero(m&expected)/max(1,np.count_nonzero(m|expected)) for m in masks]
                    qn=int(np.argmax(ious))
                    a=(pred[BOX][0,0,qn,:2].detach().cpu().numpy()-old[BOX][0,0,q,:2].numpy())*[w,h]
                    b=(updated[BOX][0,0,qn,:2].cpu().numpy()-on_original[frame][q,:2])*[w,h]
                    translations.append(dict(frame=frame,query_before=q,matched_query=qn,mask_equivariance_iou=ious[qn],
                        off_center_shift_px=a.tolist(),on_center_shift_px=b.tolist(),
                        off_translation_residual_px=float(np.linalg.norm(a-[64,0])),on_translation_residual_px=float(np.linalg.norm(b-[64,0]))))
                    count+=1; status('translation',forwards=count,total=57)
                    del pred,updated,cached,x
            assert count==57
            assert tensor_mapping_sha256(model.state_dict())==original_hash
            assert flags=={n:p.requires_grad for n,p in model.named_parameters()}
            assert all(p.grad is None for p in model.parameters())
            assert not any(m.training for m in model.modules())
            grouped=[]
            for condition,cameras in protocol['conditions'].items():
                for cam in cameras:
                    r=[e for e in rows if e['condition']==condition and e['camera']==cam and 'on_iou_to_predicted_mask_box' in e]
                    grouped.append(dict(condition=condition,camera=cam,n=len(r),**{k:float(np.mean([e[k] for e in r])) for k in ['off_iou_to_predicted_mask_box','on_iou_to_predicted_mask_box','box_center_change_px']}))
            torch.save(records,OUT/'on_boxes.pt')
            write_json(OUT/'per_view.json',rows)
            write_json(OUT/'historical_replay_semantics.json',replay_semantics)
            write_json(OUT/'translation.json',translations)
            write_json(OUT/'summary.json',dict(groups=grouped,translation=translations,manual_accuracy=False))
            write_json(OUT/'verification.json',dict(passed=True,forwards=count,off_replay_records=54,
                historical_replay_max_abs=replay_max_abs,historical_replay_atol=1e-5,historical_replay_rtol=1e-5,
                historical_mask_logits_atol=1e-4,semantic_replay_checks=True,
                other_outputs_identity=True,model_state_flags_modes_unchanged=True,optimizer_steps=0,
                holdout_accessed=False,production_promoted=False))
            status('completed',forwards=count,total=57)
            print(json.dumps(dict(groups=grouped,translation=translations)),flush=True)
        except BaseException as e:
            status('failed',error=repr(e))
            raise


if __name__=='__main__':main()
