"""Diagnostic ONLY: test reference calibration and a hypothesized 280px crop."""
import json
from pathlib import Path
import cv2
import numpy as np

BASE = Path('/home/haozong/projects/MouseGeoForge')
OUT = BASE / 'output/orbbec_openfield_audit'
SOURCE = Path('/home/haozong/projects/Orbbec OpenField/Orbbec OpenField')
meta = json.loads((BASE / 'dataset/Orbbec-Openfield/unpacked_calibration/Cage531450_R2/metadata.json').read_text())
cam = meta['camera']
kd = cam['depth_intrinsic']
kc = meta['stream_profiles'][0]['intrinsic']
K = np.array([[kc['fx'],0,kc['cx']-280],[0,kc['fy'],kc['cy']],[0,0,1.]])
R = np.array(cam['depth_to_color_rotation_row_major']).reshape(3,3)
t = np.array(cam['depth_to_color_translation_mm'])
ds = cam['color_distortion']
D = np.array([ds[k] for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
panels=[]
for row in json.loads((OUT / 'labeled_pairs.json').read_text()):
    rgbpath = SOURCE / row['sequence'] / row['sample']['file']
    rgb = cv2.imread(str(rgbpath))
    dep = cv2.imread(str(rgbpath.with_name(rgbpath.stem.replace('rgb_','depth_')+'.png')),cv2.IMREAD_UNCHANGED)[:,:,0].astype(float)*2.3529
    h,w = dep.shape
    y,x = np.indices(dep.shape)
    ok = dep>0
    xyz = np.stack([(x+280-kd['cx'])*dep/kd['fx'],(y-kd['cy'])*dep/kd['fy'],dep],axis=-1)[ok]
    xyz = xyz @ R.T + t
    uv,_ = cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)
    uv = np.rint(uv.reshape(-1,2)).astype(int)
    inside = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&(xyz[:,2]>0)
    zbuffer = np.full(h*w,np.inf)
    np.minimum.at(zbuffer,uv[inside,1]*w+uv[inside,0],xyz[inside,2])
    registered = zbuffer.reshape(h,w)
    registered[~np.isfinite(registered)] = 0
    images=[]
    for depth,title in ((dep,'RAW overlap'),(registered,'REFERENCE registration; crop=280 hypothesis')):
        overlay = rgb.copy()
        vals = depth[depth>0]
        # Draw depth isolines on the RGB image; large floor/mouse boundaries are visible.
        gray = np.uint8(np.clip((depth-np.percentile(vals,1))/max(np.percentile(vals,99)-np.percentile(vals,1),1)*255,0,255))
        edges=cv2.Canny(gray,35,80)
        overlay[edges>0]=(0,0,255)
        overlay=cv2.resize(overlay,(480,480))
        head=np.full((45,480,3),255,np.uint8)
        cv2.putText(head,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1)
        cv2.putText(head,row['sequence'].replace('Orbbec-','')[:70],(5,37),cv2.FONT_HERSHEY_SIMPLEX,.28,(0,0,0),1)
        images.append(np.vstack([head,overlay]))
    panel=np.hstack(images)
    panels.append(panel)
    cv2.imwrite(str(OUT / (row['sequence']+'_registration.jpg')),panel)
cv2.imwrite(str(OUT / 'registration_preview.jpg'),np.vstack(panels[:3]))
print('Saved diagnostic previews; reference calibration and crop offsets remain unverified.')
