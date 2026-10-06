"""Fixed first dual training scene, original and wide placements, GT only."""
import json
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.per_view_affine_batch import augment_per_view


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    plan=json.loads((ROOT/'output/wide_position_library_v1/plan.json').read_text());parent=json.loads((ROOT/'output/spatial_anchor_scale_heatmap_v1/protocol.json').read_text())
    key=next(k for k in parent['train_keys'] if parent['records'][k]['expected_instances']==2);record=parent['records'][key]
    batch=oriented_batch(record,num_views=len(record['views']));h,w=batch['images'].shape[-2:];panels=[]
    for name in ('identity','left_up','right_down'):
        row=next(r for r in plan['records'] if r['key']==key and r['variant']['name']==name);out,_=augment_per_view(batch,row['affines']);cols=[]
        for v in range(out['images'].shape[1]):
            canvas=cv2.cvtColor(np.uint8(np.clip(out['images'][0,v].permute(1,2,0).numpy()*255,0,255)),cv2.COLOR_RGB2BGR)
            for m in torch.where(out['instance_valid'][0,v])[0]:
                color=(0,255,0) if int(m)==0 else (255,255,0)
                for k in torch.where(out['instance_keypoint_valid'][0,v,m])[0]:
                    cv2.circle(canvas,tuple(np.rint(out['instance_keypoint_xy'][0,v,m,k].numpy()*[w,h]).astype(int)),3,color,1)
            cv2.putText(canvas,name,(6,23),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);cols.append(cv2.resize(canvas,(512,round(h*512/w))))
        panels.append(np.concatenate(cols,1))
    path=ROOT/'output/wide_position_library_v1/review/dual_first_fixed.png';assert not path.exists();cv2.imwrite(str(path),np.concatenate(panels,0))
    print(path)


if __name__=='__main__':main()
