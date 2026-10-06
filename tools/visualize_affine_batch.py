"""Synthetic GT overlay audit for actual RGB augmentation, not model output."""
import json
import cv2,numpy as np,torch
from run_depth_pose_ablation import ROOT
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.affine_batch_experiment import augment_batch


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    out=ROOT/'output/spatial_anchor_affine_batch_review_v1';out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'output/spatial_anchor_affine_batch_preflight_v1_retry1/plan.json').read_text())
    protocol=json.loads((ROOT/'output/spatial_anchor_diverse32_v1/protocol.json').read_text())
    chosen=[next(k for k in protocol['train_keys'] if 'bottom' in protocol['records'][k]['physical_roles']),
            next(k for k in protocol['train_keys'] if protocol['records'][k]['expected_instances']==2)]
    for scene_index,key in enumerate(chosen):
        record=protocol['records'][key];batch=oriented_batch(record,num_views=len(record['views']));variants=[]
        for var in plan['variants']:
            transformed,_=augment_batch(batch,**{k:v for k,v in var.items() if k!='name'});rows=[]
            h,w=transformed['images'].shape[-2:]
            for v in range(transformed['images'].shape[1]):
                im=np.uint8(np.clip(transformed['images'][0,v].permute(1,2,0).numpy()*255,0,255));im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                for m in torch.where(transformed['instance_valid'][0,v])[0]:
                    color=[(0,255,0),(255,100,0)][int(m)]
                    contours,_=cv2.findContours(transformed['masks'][0,v,m].numpy().astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im,contours,-1,color,1)
                    for k in range(23):
                        if transformed['instance_keypoint_valid'][0,v,m,k]:
                            xy=tuple(np.rint(transformed['instance_keypoint_xy'][0,v,m,k].numpy()*[w,h]).astype(int));cv2.circle(im,xy,3,color,-1)
                cv2.rectangle(im,(0,0),(w,28),(20,20,20),-1)
                cv2.putText(im,f'{var["name"]} {record["physical_roles"][v]} SYNTHETIC GT',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                rows.append(cv2.resize(im,(384,round(h*384/w))))
            variants.append(np.concatenate(rows,0))
        cv2.imwrite(str(out/f'scene_{scene_index}.png'),np.concatenate(variants,1))


if __name__=='__main__':main()
