"""Bounded limb repair: tracked silhouette + reliable DLC ROI, not drawn bones.

Preserve every previously published depth value. Fill missing foreground only
using new, main-body-connected voxels with bottom and >=5/6 side support.
This remains unverified pseudo geometry, never enabled as training ground truth.
"""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from reconstruct_pymousestudio_pilot import triangulate
from add_dannce_real_multiview_pseudo_depth import project, rasterize_depth, read_pfm, write_pfm
from export_pymousestudio_rectified import write

cv2.setNumThreads(1)


def landmark_roi(shape, projections, radius=90):
    """A search region only. It cannot create foreground without a silhouette."""
    roi = np.zeros(shape, np.uint8)
    for uv in projections:
        if np.isfinite(uv).all():
            x, y = np.rint(uv).astype(int)
            if -radius <= x < shape[1]+radius and -radius <= y < shape[0]+radius:
                cv2.circle(roi, (x, y), radius, 255, -1)
    return roi


def connected_additions(old, proposed, shape):
    """New voxels must connect to the largest old body component in 3D."""
    structure = np.ones((3, 3, 3), bool)
    old_ids, count = label(old.reshape(shape), structure=structure)
    if not count:
        return np.zeros_like(old)
    sizes = np.bincount(old_ids.ravel()); sizes[0] = 0
    seed = np.flatnonzero(old_ids.ravel() == sizes.argmax())[0]
    union_ids, _ = label((old | proposed).reshape(shape), structure=structure)
    keep = union_ids.ravel() == union_ids.ravel()[seed]
    return proposed & ~old & keep


def component_metrics(mask):
    _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), 8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if not len(sizes):
        return dict(components_ge20=0, orphan_pixels=0)
    return dict(components_ge20=int(np.count_nonzero(sizes >= 20)),
                orphan_pixels=int(sizes.sum()-sizes.max()))


def reject_new_islands(before, additions):
    """Conservatively reject new 2D components with no old foreground contact."""
    _, components = cv2.connectedComponents(np.uint8(before | additions), connectivity=8)
    touched = np.unique(components[before])
    return additions & np.isin(components, touched[touched>0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, nargs='+', default=list(range(780, 812)))
    parser.add_argument('--output', default='reconstruction_limb_guarded_v1')
    args = parser.parse_args()
    s = ROOT/'dataset/PyMouseStudio/sessions/AZL3'
    base = s/'reconstruction_tail_guarded_v2'
    body = s/'reconstruction_bottom_refined_v1'
    tracked = s/'tail_dlc_video_masks_v1'
    out = s/args.output
    out.mkdir(exist_ok=False)
    (out/'review').mkdir()
    cal = json.loads((s/'calibration.json').read_text())
    K = {n: np.array(cal[n]['K']) for n in 'ABCDEFG'}
    E = {n: np.array(cal[n]['world_to_camera_mm']) for n in 'ABCDEFG'}
    data = {n: dict(np.load(s/f'keypoints_source31/cam{n}.npz')) for n in 'BCDEFG'}
    cfg = json.loads((body/'config.json').read_text())
    low, high = np.array(cfg['bounds_mm'])
    axes = [np.arange(low[i], high[i]+.5, 1.) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    cache = {}
    for n in 'ABCDEFG':
        uv, z = project(points, K[n], E[n]); x, y = np.rint(uv).astype(int).T
        valid = (z>0)&(x>=0)&(x<1080)&(y>=0)&(y<1200)
        cache[n] = (np.clip(y, 0, 1199)*1080+np.clip(x, 0, 1079), valid)
    records = []; audits = []; video = None
    # Paw, ankle, digit and nearby limb candidates; semantics remain source31.
    for f in args.frames:
        xyz = []; ids = []
        for j in range(5, 19):
            obs = {n: d['uv'][f,j] for n,d in data.items()
                   if d['likelihood'][f,j] >= .8 and np.isfinite(d['uv'][f,j]).all()}
            X, err = triangulate(obs, K, E)
            audits.append(dict(frame=f, source_id=j, name=str(data['B']['source_names'][j]),
                               xyz_mm=None if X is None else X.tolist(), reprojection_px=err,
                               confident_views=list(obs)))
            if X is not None:
                xyz.append(X); ids.append(j)
        uv = project(np.array(xyz).reshape(-1,3), K['A'], E['A'])[0] if xyz else []
        roi = landmark_roi((1200,1080), uv)
        old_bottom = cv2.imread(str(body/f'frame_{f:06d}/bottom_sam2_mask.png'), 0)
        full_track = cv2.imread(str(tracked/f'masks/{f:06d}.png'), 0)
        tail = cv2.imread(str(tracked/f'tail_roi/{f:06d}.png'), 0)
        # No all-body SAM adoption: exclude tail, reflection arcs and unsupported pixels.
        extension = (roi>0)&(full_track>0)&(old_bottom==0)&(tail==0)
        masks = {n: cv2.imread(str(s/f'pilot_input_masks/{f:06d}/cam{n}.png'), 0) for n in 'BCDEFG'}
        votes = np.zeros(len(points), np.uint8)
        for n in 'BCDEFG':
            ind, valid = cache[n]
            votes += ((masks[n].ravel()[ind]>0)&valid).astype(np.uint8)
        ind, valid = cache['A']
        old = (votes>=5)&(old_bottom.ravel()[ind]>0)&valid
        proposal = (votes>=5)&extension.ravel()[ind]&valid
        added = connected_additions(old, proposal, shape)
        cloud = points[added]
        scene = out/f'frame_{f:06d}'; scene.mkdir()
        cv2.imwrite(str(scene/'bottom_limb_search_roi.png'), roi)
        cv2.imwrite(str(scene/'bottom_limb_mask_extension.png'), np.uint8(extension)*255)
        view_rows = []; panels = [[],[],[]]
        for v,n in enumerate('ABCDEFG'):
            before = read_pfm(base/f'frame_{f:06d}/rendered_depth_maps/{v:08d}.pfm')
            mask = masks[n] if n!='A' else np.uint8(extension)*255
            candidate = rasterize_depth(cloud, mask, K[n], E[n], 1.) if len(cloud) else np.zeros_like(before)
            # Append-only delivery: never perturb existing tail/body depths.
            added_pixels = (before==0)&(candidate>0)
            if n=='A':
                added_pixels &= extension
            added_pixels = reject_new_islands(before>0, added_pixels)
            after = before.copy(); after[added_pixels] = candidate[added_pixels]
            path = scene/'rendered_depth_maps'/f'{v:08d}.pfm'; write_pfm(path, after)
            for folder, img in [('masks',after>0),('limb_added_pixels',added_pixels)]:
                path = scene/folder/f'{v:08d}.png'; path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(path), np.uint8(img)*255)
            assert np.array_equal(after[before>0], before[before>0])
            assert np.array_equal(read_pfm(scene/'rendered_depth_maps'/f'{v:08d}.pfm'),after)
            assert np.isfinite(after).all() and (after>=0).all()
            view_rows.append(dict(camera=n, view_id=v, added_pixels=int(added_pixels.sum()),
                                  before=component_metrics(before>0), after=component_metrics(after>0)))
            rgb = cv2.imread(str(s/f'pilot_rgb/{f:06d}/cam{n}.png'))
            images = []
            for region, color, title in [(before>0,(0,130,255),f'cam{n} BEFORE'),
                                         (after>0,(70,255,0),f'cam{n} AFTER +{added_pixels.sum()}px')]:
                im=rgb.copy();im[region]=(.6*im[region]+.4*np.array(color)).astype(np.uint8)
                cv2.putText(im,title,(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                images.append(im)
            depth_img = cv2.applyColorMap(np.uint8(np.clip((after-100)/220*255,0,255)),cv2.COLORMAP_TURBO)
            depth_img[after==0]=255
            for row,im in zip(panels,images+[depth_img]):
                row.append(cv2.resize(im,(270,300)))
            if f in [780,796,811]:
                # Native-resolution leg crop, include actual RGB and source silhouette.
                centers = project(np.array(xyz).reshape(-1,3),K[n],E[n])[0] if xyz else np.array([[540,800]])
                x0,y0 = np.maximum(np.floor(centers.min(0)-100).astype(int),[0,0])
                x1,y1 = np.minimum(np.ceil(centers.max(0)+100).astype(int),[1080,1200])
                source=full_track if n=='A' else masks[n]
                si=rgb.copy(); sel=source>0;si[sel]=(.6*si[sel]+.4*np.array([255,180,0])).astype(np.uint8)
                montage=np.concatenate([im[y0:y1,x0:x1] for im in [rgb,si,images[0],images[1]]],axis=1)
                cv2.imwrite(str(out/f'review/limbs_{f:06d}_cam{n}.png'),montage)
        montage = np.concatenate([np.concatenate(row,axis=1) for row in panels],axis=0)
        cv2.imwrite(str(out/f'review/{f:06d}.png'),montage)
        if video is None:
            video=cv2.VideoWriter(str(out/'review/limb_repair_8fps.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(1890,900))
        video.write(montage)
        records.append(dict(frame=f, reliable_limb_ids=ids, tracked_extension_pixels=int(extension.sum()),
                            proposed_voxels=int(proposal.sum()), accepted_voxels=int(added.sum()), views=view_rows))
        write(out/'progress.json',dict(completed=len(records),total=len(args.frames)))
        print('LIMBS',f,'landmarks',ids,'new voxels',int(added.sum()),'new pixels',sum(x['added_pixels'] for x in view_rows),flush=True)
    if video: video.release()
    write(out/'report.json',dict(frames=records,landmark_audit=audits,train_allowed=False))
    write(out/'config.json',dict(method='DLC-localized temporal bottom silhouette repair; append-only depth',
        base=str(base.relative_to(ROOT)),frames=args.frames,voxel_mm=1.,depth_unit='mm',depth_semantics='camera_z',
        size_wh=[1080,1200],min_side_votes=5,landmark_confidence=.8,landmark_min_inlier_views=3,
        landmark_max_reprojection_px=15,roi_radius_px=90,connectivity='26-neighbor to largest original 3D body',
        no_silhouette_gap_filling=True,existing_depths_unchanged=True,train_allowed=False,
        reject_new_2d_islands=True,
        limitation='Pseudo masks, not manual GT; genuine occlusions and missing input silhouettes may remain disconnected.'))


if __name__=='__main__':
    main()
