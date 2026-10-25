"""Fixed seed17 ROI depth with final new FULL/ROI native masks; old M anchor replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI as CalibratedRayDepth
from mousegeoforge.semseg_training import digest, mask_metrics, write_json
from calibrated_mouse_data_20261025 import OUT, gpu_lock, points
from train_calibrated_mouse_depth_roi_20261025 import write_ply
from evaluate_calibrated_mouse_spatial_pipeline_20261025 import cloud_metrics
NEW=ROOT/'output/calibrated_mouse_next12h_20261025'
RESUME=ROOT/'output/calibrated_mouse_resume8h_20261025'
ANCHOR=RESUME/'roi_pipeline_s17_roi_v1'


def box(mask):
    y,x = np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def box_iou(first, second):
    if first is None or second is None:
        return 0.
    overlap = max(0,min(first[2],second[2])-max(first[0],second[0]))*max(0,min(first[3],second[3])-max(first[1],second[1]))
    areas = (first[2]-first[0])*(first[3]-first[1])+(second[2]-second[0])*(second[3]-second[1])
    return overlap/max(1,areas-overlap)



def compare_common_depth(prediction,mask,old_depth,old_mask,tolerance=1e-4):
    common=mask&old_mask&(prediction>0)&(old_depth>0)
    maximum=float(np.max(np.abs(prediction[common]-old_depth[common]))) if common.any() else 0.
    assert maximum<=tolerance,('fixed-depth replay failed',maximum,tolerance)
    return dict(common_valid_pixels=int(common.sum()),maximum_abs_difference_mm=maximum,tolerance_mm=tolerance)


def verify_new_masks(arm,directory):
    report=json.loads((NEW/'semseg_roi_pair_report_v2.json').read_text());assert report['comparable'] and report['campaign']['full_paired_endpoint']==6000
    protocol=json.loads((NEW/f'semseg_roi_pair_v2/{arm}_006000/protocol.json').read_text())
    assert protocol['crop_mode']==arm and protocol['steps']==6000 and protocol['label_policy']=='center_exact' and protocol['precision']=='fp32'
    assert protocol['roi_warmup']==600 and protocol['roi_cosine_floor']==.1 and protocol['head_phase_exposure_offset']==18000
    ck=NEW/f'semseg_roi_pair_v2/{arm}_006000/step_006000.pt';checkpoint_sha=digest(ck)
    data=json.loads((directory/'evaluation.json').read_text());audit=json.loads((NEW/f'semseg_roi_native_{arm}_v2.json').read_text())
    assert audit['denominator']==879 and audit['checkpoint_sha256']==data['checkpoint_sha256']==checkpoint_sha
    assert audit['evaluation_receipt_sha256']==digest(directory/'evaluation.json')
    assert report['arms'][arm]['checkpoint_sha256']==checkpoint_sha
    return dict(role=f'new semsegROI v2 {arm} final6000 EMA',checkpoint_sha256=checkpoint_sha,evaluation_sha256=digest(directory/'evaluation.json'),native_audit_sha256=digest(NEW/f'semseg_roi_native_{arm}_v2.json'),protocol_sha256=digest(NEW/f'semseg_roi_pair_v2/{arm}_006000/protocol.json'),native_policy='originalGT;FP32bilinear probability>.5,tiebackground;zerooutsidepredictedcrop')


def verify_registration():
    path=NEW/'registrations/semseg_chain_v1.json';reg=json.loads(path.read_text())
    for name,h in reg['source_sha256'].items():assert digest(ROOT/name)==h
    for name,h in reg['input_sha256'].items():assert digest(ROOT/name)==h
    return reg

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['anchor','full','roi'],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reg=verify_registration();args.depth_checkpoint=ROOT/reg['depth_checkpoint'];args.roi_cache=RESUME/'roi_cache_v1';args.feature_mode='roi'
    args.segmentation_evaluation=RESUME/'M_dev_v1' if args.arm=='anchor' else NEW/f'semseg_roi_pair_v2/{args.arm}_dev'
    lineage=dict(role='historical originalM18000EMA fixed anchor; legacy native mask operator retained',evaluation_sha256=digest(args.segmentation_evaluation/'evaluation.json')) if args.arm=='anchor' else verify_new_masks(args.arm,args.segmentation_evaluation)
    if args.arm!='anchor':
        smoke=json.loads((NEW/'semseg_chain_anchor_GPU_smoke_v1/evaluation.json').read_text());assert smoke['anchor_replay_passed'] and smoke['full_scene_failure_denominator']==1
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir()
    masks=args.segmentation_evaluation/'masks_ema'
    audited={r['id']:r for r in json.loads((NEW/f'semseg_roi_native_{args.arm}_v2.json').read_text())['rows']} if args.arm!='anchor' else {}
    roi_receipt=json.loads((args.roi_cache/'cache_receipt.json').read_text());roi_hashes={r['scene_index']:r['sha256'] for r in roi_receipt['files']}
    rows=json.loads((OUT/'semseg_items.json').read_text())['items']
    by_image={r['image']:r for r in rows if r['domain']=='synthetic' and r['split']=='dev'}
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    receipt=json.loads((OUT/'cache_receipt.json').read_text())
    for entry in receipt['files']:
        assert digest(ROOT/entry['path'])==entry['sha256']
    saved=torch.load(args.depth_checkpoint,map_location='cuda',weights_only=False)
    assert saved['protocol']['feature_mode']==args.feature_mode and saved['step']==6000 and saved['protocol']['seed']==17 and saved['protocol']['gradient_weight']==0.
    assert digest(args.depth_checkpoint)==reg['depth_checkpoint_sha256']
    model=CalibratedRayDepth(planes=saved['protocol']['planes']).cuda().eval()
    model.load_state_dict(saved['ema'])
    bounds=torch.tensor(saved['protocol']['capture_aabb_mm'],device='cuda',dtype=torch.float32)
    began=time.monotonic();results=[]
    with gpu_lock(),torch.inference_mode():
        for index,scene in enumerate(manifest['scenes']):
            if scene['split']!='dev':continue
            if args.arm=='anchor' and results:break
            cache=args.roi_cache if args.feature_mode=='roi' else OUT/'depth_features'
            cache_file=cache/f'{index:06d}.pt';assert digest(cache_file)==roi_hashes[index]
            item=torch.load(cache_file,map_location='cpu',weights_only=False)
            features=item['features'].cuda().float();k=item['K'].cuda().float();e=item['E'].cuda().float()
            affine=item['feature_affines'].cuda().float() if args.feature_mode=='roi' else torch.eye(3,device='cuda').repeat(3,1,1)
            encoded=model.encode(features)
            scene_dir=args.output/f'scene{index:06d}';scene_dir.mkdir()
            predicted_cloud,truth_cloud,views=[],[],[]
            for view,source in enumerate(scene['views']):
                row=by_image[source['image']];mask_file=masks/(row['id']+'.png')
                mask=cv2.imread(str(mask_file),cv2.IMREAD_GRAYSCALE)
                if mask is None:raise FileNotFoundError(mask_file)
                mask=mask>0;gt=item['body'][view].numpy();target=item['depth'][view].numpy()
                if args.arm!='anchor':
                    assert digest(mask_file)==audited[row['id']]['prediction_sha256'];assert digest(masks/(row['id']+'_probability.npz'))==audited[row['id']]['probability_sha256']
                    with np.load(masks/(row['id']+'_probability.npz')) as zz:prob=zz['foreground'].copy()
                    assert prob.dtype==np.float32 and np.isfinite(prob).all();np.testing.assert_array_equal(mask,prob>.5)
                assert mask.shape==gt.shape==(384,384)
                y,x=np.where(mask);prediction=np.zeros_like(target);confidence=np.zeros_like(target)
                if len(x):
                    xy=torch.from_numpy(np.stack([x,y],-1)).cuda().float();values=[];confidences=[];validity=[]
                    for chunk in xy.split(512):
                        z,conf,valid=model(encoded,chunk,view,k,e,bounds,feature_affines=affine)
                        values.append(z.cpu());confidences.append(conf.cpu());validity.append(valid.cpu())
                    z=torch.cat(values).numpy();valid=torch.cat(validity).numpy();conf=torch.cat(confidences).numpy()
                    assert np.isfinite(z).all()
                    prediction[y[valid],x[valid]]=z[valid];confidence[y[valid],x[valid]]=conf[valid]
                with np.load(ANCHOR/f'scene{index:06d}/view{view}.npz') as zz:old_depth,old_mask=zz['depth_mm'].copy(),zz['mask'].copy()
                replay=compare_common_depth(prediction,mask,old_depth,old_mask)
                if args.arm=='anchor':
                    np.testing.assert_array_equal(mask,old_mask);np.testing.assert_array_equal(prediction>0,old_depth>0);assert replay['common_valid_pixels']==int((old_depth>0).sum())
                camera_k,camera_e=k[view].cpu().numpy(),e[view].cpu().numpy()
                np.savez_compressed(scene_dir/f'view{view}.npz',depth_mm=prediction,mask=mask,
                    plane_probability_max=confidence,K=camera_k,E=camera_e)
                predicted_cloud.append(points(prediction,mask&(prediction>0),camera_k,camera_e))
                truth_cloud.append(points(target,gt,camera_k,camera_e))
                overlap=gt&mask&(prediction>0);residual=prediction[overlap]-target[overlap]
                strict=prediction[gt]-target[gt]
                views.append(dict(view=view,bottom=view==0,segmentation=mask_metrics(mask.astype('uint8'),gt.astype('uint8'),1),
                    bbox_xyxy=box(mask),gt_bbox_xyxy=box(gt),bbox_iou=box_iou(box(mask),box(gt)),
                    ground_truth_foreground_pixels=int(gt.sum()),predicted_pixels=int(mask.sum()),
                    gt_depth_coverage=float(overlap.sum()/max(1,gt.sum())),
                    rmse_mm_on_overlap=float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
                    rmse_mm_missing_as_zero=float(np.sqrt(np.mean(strict**2))),
                    rejected_predicted_rays=int((mask&(prediction==0)).sum()),
                    mask_sha256=digest(mask_file),fixed_depth_common_replay=replay,old_depth_input_sha256=digest(ANCHOR/f'scene{index:06d}/view{view}.npz')))
            predicted,truth=np.concatenate(predicted_cloud),np.concatenate(truth_cloud)
            geometry=cloud_metrics(predicted,truth)
            write_ply(scene_dir/'predicted_world_mm.ply',predicted)
            write_json(scene_dir/'cameras.json',dict(depth_units='millimeter',depth_semantics='camera_z',
                extrinsic='OpenCV world-to-camera',pixel_centres='integer array indices',
                K=k.cpu().tolist(),E=e.cpu().tolist(),fusion='same synchronized instant only',
                confidence='uncalibrated maximum depth-plane probability'))
            results.append(dict(id=scene['id'],source=scene['source'],rig=scene['rig'],condition=scene['condition'],
                                pointcloud=geometry,views=views))
            if len(results)<=4:
                rgb=cv2.imread(str(ROOT/scene['views'][0]['image']));pred=cv2.imread(str(masks/(by_image[scene['views'][0]['image']]['id']+'.png')),0)
                gt=item['body'][0].numpy();overlay=rgb.copy();overlay[pred>0]=(overlay[pred>0]*.5+np.array([0,180,0])*.5).astype('uint8')
                panel=np.concatenate([rgb,np.repeat(gt[...,None]*255,3,axis=-1).astype('uint8'),overlay],axis=1)
                cv2.imwrite(str(scene_dir/'bottom_rgb_truth_prediction.jpg'),panel)
            if len(results)%6==0:
                print('PIPELINE',len(results),144,flush=True)
                write_json(args.output/'progress.json',dict(scenes=len(results),seconds=time.monotonic()-began))
    views=[v for scene in results for v in scene['views']]
    summary=dict(mean_iou=float(np.mean([v['segmentation']['iou'] for v in views])),
        mean_bbox_iou=float(np.mean([v['bbox_iou'] for v in views])),
        gt_depth_coverage=float(np.mean([v['gt_depth_coverage'] for v in views])),
        pointcloud_fscore={str(t):float(np.mean([r['pointcloud']['fscore'][str(t)] for r in results])) for t in (1,2,5)})
    write_json(args.output/'evaluation.json',dict(summary=summary,scenes=results,
        depth_checkpoint_sha256=digest(args.depth_checkpoint),segmentation_arm=lineage['role'],segmentation_lineage=lineage,registration_sha256=digest(NEW/'registrations/semseg_chain_v1.json'),
        segmentation_evaluation_receipt_sha256=digest(args.segmentation_evaluation/'evaluation.json'),
        feature_mode=args.feature_mode,feature_affine_sampling=True,
        roi_cache_receipt_sha256=digest(args.roi_cache/'cache_receipt.json'),
        oracle_mask_inference=False,calibration_fixed=True,gt_scale_alignment=False,
        single_mouse_bbox_only=True,multiple_mouse_instance_detection_claim=False,
        independent_real_mm_accuracy_claim=False,unseen_pose_claim=False,
        visible_skin_surrogate=True,full_scene_failure_denominator=len(results),view_denominator=len(views),fixed_depth_common_replay_passed=True,anchor_replay_passed=args.arm=='anchor',old_anchor_evaluation_sha256=digest(ANCHOR/'evaluation.json'),source_sha256=reg['source_sha256'],seconds=time.monotonic()-began))
    assert len(results)==(1 if args.arm=='anchor' else 144) and len(views)==(3 if args.arm=='anchor' else 432)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
