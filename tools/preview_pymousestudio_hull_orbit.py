"""Simple orbit of the actual reconstructed voxel cloud, no generated anatomy."""
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import binary_erosion
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'dataset/PyMouseStudio/sessions/AZL3/reconstruction_bottom_refined_v1'
    p=base/'frame_000780/hull_voxels_mm.ply'
    with p.open('rb') as f:
        while f.readline().strip()!=b'end_header':pass
        a=np.fromfile(f,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    points=np.stack([a[n] for n in ['x','y','z']],-1)
    grid=np.rint(points-points.min(0)).astype(int)+1
    occupancy=np.zeros(tuple(grid.max(0)+2),bool);occupancy[tuple(grid.T)]=True
    surface=occupancy&~binary_erosion(occupancy)
    keep=surface[tuple(grid.T)];points=points[keep];points-=np.median(points,axis=0)
    w,h=960,720;target=base/'review/orbit_voxel_cloud_frame000780.mp4'
    enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
       '-s',f'{w}x{h}','-r','15','-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','veryfast',
       '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    for i in range(90):
        angle=2*np.pi*i/90;ca,sa=np.cos(angle),np.sin(angle)
        R=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
        tilt=.3;T=np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
        q=points@R.T@T.T;scale=4.;xy=np.rint(q[:,:2]*[scale,-scale]+[w/2,h/2]).astype(int)
        z=q[:,2];shade=np.clip(100+120*(z-z.min())/max(1,np.ptp(z)),0,255).astype(np.uint8)
        panel=np.full((h,w,3),245,np.uint8)
        for idx in np.argsort(z):
            x,y=xy[idx]
            if 0<=x<w and 60<=y<h-40:
                c=int(shade[idx]);cv2.circle(panel,(int(x),int(y)),2,(c,int(c*.8),int(c*.55)),-1)
        cv2.putText(panel,'AZL3 frame780 | actual 1mm voxel SURFACE cloud | not mesh GT',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),1,cv2.LINE_AA)
        cv2.putText(panel,'Orbit in calibration coordinates; no ground-alignment assumption',(15,53),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,50,50),1,cv2.LINE_AA)
        cv2.line(panel,(35,h-45),(115,h-45),(20,20,20),3)
        cv2.putText(panel,'20 mm',(35,h-17),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1,cv2.LINE_AA)
        if i in [0,22,45,67]:cv2.imwrite(str(base/'review'/f'orbit_{i:02d}.png'),panel)
        enc.stdin.write(panel.tobytes())
    enc.stdin.close();assert enc.wait()==0;print(target)


if __name__=='__main__':main()
