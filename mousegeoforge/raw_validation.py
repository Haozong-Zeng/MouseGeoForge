"""Explicitly held-out real-video access; no implicit training or invented GT."""
import json
from pathlib import Path
import pickletools
import cv2
import numpy as np


def primitive_column_metadata(blob):
    """Interpret only primitive protocol0 metadata, never execute pickle objects."""
    if len(blob)>100000:raise ValueError('Oversize metadata')
    stack=[];memo={};mark=object()
    for count,(op,arg,_) in enumerate(pickletools.genops(bytes(blob))):
        if count>10000:raise ValueError('Too many metadata operations')
        if op.name=='MARK':stack.append(mark)
        elif op.name in ('UNICODE','STRING'):stack.append(arg)
        elif op.name=='NONE':stack.append(None)
        elif op.name=='PUT':memo[arg]=stack[-1]
        elif op.name=='GET':stack.append(memo[arg])
        elif op.name in ('LIST','TUPLE'):
            index=next(i for i in range(len(stack)-1,-1,-1) if stack[i] is mark)
            values=stack[index+1:];stack[index:]=[values if op.name=='LIST' else tuple(values)]
        elif op.name=='APPEND':value=stack.pop();stack[-1].append(value)
        elif op.name=='STOP':
            if len(stack)!=1:raise ValueError('Invalid metadata stack')
            return stack[0]
        else:raise ValueError(f'Executable or unsupported metadata opcode: {op.name}')
    raise ValueError('Incomplete metadata')


def undistort_points_compat(uv,K,dist,P=None):
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-10)
    if hasattr(cv2,'undistortPointsIter'):
        return cv2.undistortPointsIter(uv,K,dist,None,P,criteria)
    return cv2.undistortPoints(uv,K,dist,R=None,P=P,criteria=criteria)


def rectify_points(uv,K,dist):
    uv=np.asarray(uv,dtype=np.float64);result=np.full_like(uv,np.nan)
    valid=np.isfinite(uv).all(-1)
    if valid.any():
        result[valid]=undistort_points_compat(uv[valid,None,:],K,dist,P=K).reshape(-1,2)
    return result


def group_for_session(session):
    # Conservative family grouping, not a claim that this prefix is an animal ID.
    if session.startswith('AZ'):return 'AZ'
    return session[0]


class RawVideoValidation:
    """Native-resolution frame reader for the NEW real validation manifest only.

    Does not claim DLC predictions/masks are manual GT. No model calls or training.
    Lens correction is explicit, paired across RGB, masks and keypoints.
    """
    def __init__(self,manifest,project_root,allow_locked_holdout=False):
        self.root=Path(project_root)
        self.records={r['sequence_id']:r for r in map(json.loads,Path(manifest).read_text().splitlines())}
        if any(r['split'] not in ('real_development','real_locked_holdout') or r.get('train_allowed',True) for r in self.records.values()):
            raise ValueError('This reader accepts explicitly non-training validation sources only')
        self.allow_locked_holdout=allow_locked_holdout

    def read(self,sequence_id,frame,cameras=None,undistort=False):
        r=self.records[sequence_id]
        if r['split']=='real_locked_holdout' and not self.allow_locked_holdout:
            raise PermissionError('Holdout is locked; freeze the evaluation protocol before accessing it')
        if frame<0 or frame>=r['frame_count']:raise IndexError(frame)
        names=cameras or list(r['cameras'])
        calibration=json.loads((self.root/r['prepared_calibration']).read_text())
        images=[];masks=[];mask_valid=[];uvs=[];conf=[];Ks=[];Ds=[];Es=[]
        def read_video(path):
            cap=cv2.VideoCapture(str(self.root/path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,arr=cap.read();cap.release()
            if not ok:raise OSError(f'Unable to decode {path} frame{frame}')
            return arr
        for name in names:
            c=r['cameras'][name];cal=calibration[name]
            rgb=cv2.cvtColor(read_video(c['video']),cv2.COLOR_BGR2RGB)
            K=np.asarray(cal['K'],dtype=np.float64);D=np.asarray(cal['distortion'],dtype=np.float64)
            h,w=rgb.shape[:2]
            if [w,h]!=cal['size_wh']:raise ValueError('Image/calibration size mismatch')
            if c['keypoints']:
                with np.load(self.root/c['keypoints'],allow_pickle=False) as a:
                    indices=np.flatnonzero(a['frame_indices']==frame)
                    if len(indices)!=1:raise ValueError('Unknown keypoint/video synchronization')
                    uv=a['uv'][indices[0]].astype(np.float64);likelihood=a['likelihood'][indices[0]]
            else:uv=np.full((31,2),np.nan);likelihood=np.zeros(31,dtype=np.float32)
            hasmask=c['mask'] is not None
            mask=cv2.cvtColor(read_video(c['mask']),cv2.COLOR_BGR2GRAY) if hasmask else np.zeros((h,w),dtype=np.uint8)
            if mask.shape!=(h,w):raise ValueError('Mask/RGB alignment mismatch')
            if undistort:
                mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(w,h),cv2.CV_32FC1)
                rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                mask=cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                uv=rectify_points(uv,K,D);D=np.zeros_like(D)
            images.append(rgb);masks.append(mask);mask_valid.append(hasmask)
            uvs.append(uv[None]);conf.append(likelihood[None]);Ks.append(K);Ds.append(D);Es.append(cal['world_to_camera_mm'])
        return dict(images=np.stack(images),foreground_mask_uint8=np.stack(masks),mask_valid=np.array(mask_valid),
            keypoints2d_source31=np.stack(uvs),keypoint_confidence=np.stack(conf),K=np.stack(Ks),distortion=np.stack(Ds),
            world_to_camera_mm=np.asarray(Es),camera_names=names,frame=frame,timestamp_seconds=frame/r['fps'],
            source_group=r['source_group'],split=r['split'],annotation_kind='DLC_prediction_and_unverified_mask',
            manual_ground_truth=False,detection_annotation_complete=False,depth=None,keypoints3d=None,
            lens_correction_applied=undistort)
