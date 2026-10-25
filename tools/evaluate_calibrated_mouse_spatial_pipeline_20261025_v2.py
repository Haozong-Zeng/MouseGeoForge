"""Predicted full-image masks + fixed calibrated depth head, visible surfaces only."""
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
from mousegeoforge.models.calibrated_mouse_depth_spatial_20261025_v2 import CalibratedRayDepthSpatialV2 as CalibratedRayDepth
from mousegeoforge.semseg_training import digest, mask_metrics, write_json
from calibrated_mouse_data_20261025 import OUT, gpu_lock, points
from train_calibrated_mouse_depth_roi_20261025 import write_ply
from scipy.spatial import cKDTree


def cloud_metrics(predicted,target):
    if not len(target):raise ValueError('Empty GT cloud')
    if not len(predicted):return dict(accuracy_mm=None,completeness_mm=None,precision={str(t):0. for t in (1,2,5)},recall={str(t):0. for t in (1,2,5)},fscore={str(t):0. for t in (1,2,5)},failed=True)
    dp=cKDTree(target).query(predicted,workers=2)[0];dt=cKDTree(predicted).query(target,workers=2)[0]
    result=dict(accuracy_mm=float(dp.mean()),completeness_mm=float(dt.mean()),precision={},recall={},fscore={},failed=False)
    for t in (1,2,5):
        precision=float((dp<=t).mean());recall=float((dt<=t).mean());key=str(t)
        result['precision'][key]=precision;result['recall'][key]=recall;result['fscore'][key]=2*precision*recall/max(precision+recall,1e-12)
    return result


def box(mask):
    y,x = np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def box_iou(first, second):
    if first is None or second is None:
        return 0.
    overlap = max(0,min(first[2],second[2])-max(first[0],second[0]))*max(0,min(first[3],second[3])-max(first[1],second[1]))
    areas = (first[2]-first[0])*(first[3]-first[1])+(second[2]-second[0])*(second[3]-second[1])
    return overlap/max(1,areas-overlap)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--depth-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--segmentation-evaluation',type=Path,required=True);parser.add_argument('--feature-mode',choices=['full','roi'],required=True);parser.add_argument('--roi-cache',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    if not (args.segmentation_evaluation/'evaluation.json').exists():
        raise FileNotFoundError('Complete fixed M DEV evaluation required')
    masks=args.segmentation_evaluation/'masks_ema'
    rows=json.loads((OUT/'semseg_items.json').read_text())['items']
    by_image={r['image']:r for r in rows if r['domain']=='synthetic' and r['split']=='dev'}
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    receipt=json.loads((OUT/'cache_receipt.json').read_text())
    for entry in receipt['files']:
        assert digest(ROOT/entry['path'])==entry['sha256']
    saved=torch.load(args.depth_checkpoint,map_location='cuda',weights_only=False)
    assert saved['protocol']['feature_mode']==args.feature_mode
    model=CalibratedRayDepth(planes=saved['protocol']['planes'],neck_kernel=saved['protocol']['neck_kernel']).cuda().eval()
    model.load_state_dict(saved['ema'])
    bounds=torch.tensor(saved['protocol']['capture_aabb_mm'],device='cuda',dtype=torch.float32)
    began=time.monotonic();results=[]
    with gpu_lock(),torch.inference_mode():
        for index,scene in enumerate(manifest['scenes']):
            if scene['split']!='dev':continue
            cache=args.roi_cache if args.feature_mode=='roi' else OUT/'depth_features'
            item=torch.load(cache/f'{index:06d}.pt',map_location='cpu',weights_only=False)
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
                    mask_sha256=digest(mask_file)))
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
        depth_checkpoint_sha256=digest(args.depth_checkpoint),score_bias_trainable=saved['protocol']['score_bias_trainable'],neck_kernel=saved['protocol']['neck_kernel'],neck_parameters=saved['protocol']['neck_parameters'],segmentation_arm='M common endpoint18000 EMA, fixed before resumed DEV',
        segmentation_evaluation_receipt_sha256=digest(args.segmentation_evaluation/'evaluation.json'),
        feature_mode=args.feature_mode,feature_affine_sampling=True,
        roi_cache_receipt_sha256=digest(args.roi_cache/'cache_receipt.json'),
        oracle_mask_inference=False,calibration_fixed=True,gt_scale_alignment=False,
        single_mouse_bbox_only=True,multiple_mouse_instance_detection_claim=False,
        independent_real_mm_accuracy_claim=False,unseen_pose_claim=False,
        visible_skin_surrogate=True,full_scene_failure_denominator=144,seconds=time.monotonic()-began))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
