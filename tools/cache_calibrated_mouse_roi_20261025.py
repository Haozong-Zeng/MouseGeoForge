"""Fixed M18000 predicted boxes, half-pixel affine K, frozen VGGT ROI cache."""
import argparse,pathlib,json,sys,time,math,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image
from mousegeoforge.semseg_training import MouseSemseg,make_encoder,letterbox,native_mask,digest,write_json
from mousegeoforge.models.mouse_semseg import semantic_scores
from calibrated_mouse_data_20261025 import gpu_lock
OLD=ROOT/'output/calibrated_mouse_10h_20261025';NEW=ROOT/'output/calibrated_mouse_resume8h_20261025'
M_SHA='30a50a5cfb67b7639b1d2e24988789a11460b8787ed9eea46963b2e9e148b8dc'
def roi_affine(mask):
 y,x=np.where(mask)
 if not len(x):return (0,0,384),np.eye(3,dtype=np.float32)
 size=min(384,max(96,math.ceil(max(x.max()-x.min()+1,y.max()-y.min()+1)*1.4)))
 x0=int(np.clip(math.floor((x.max()+x.min()+1-size)/2),0,384-size));y0=int(np.clip(math.floor((y.max()+y.min()+1-size)/2),0,384-size));scale=384/size
 a=np.array([[scale,0,scale*(.5-x0)-.5],[0,scale,scale*(.5-y0)-.5],[0,0,1]],np.float32)
 return (x0,y0,size),a
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=pathlib.Path,required=True);p.add_argument('--limit',type=int);args=p.parse_args();args.run_dir.mkdir(exist_ok=False)
 torch.set_num_threads(4);cv2.setNumThreads(1);manifest=json.loads((OLD/'data_manifest.json').read_text());rows=manifest['scenes'];indices=list(range(len(rows)))
 if args.limit:indices=[int(i) for i in np.linspace(0,len(rows)-1,args.limit)]
 checkpoint=OLD/'semseg_campaign_s17_v1/M_018000/step_018000.pt';assert digest(checkpoint)==M_SHA
 began=time.monotonic();files=[];diagnostics=[]
 with gpu_lock(),torch.inference_mode():
  encoder=make_encoder(ROOT/'vggt-omega/vggt_omega_1b_512.pt');model=MouseSemseg(encoder).cuda().eval().requires_grad_(False);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);parameters=dict(model.named_parameters())
  for name,value in saved['ema'].items():assert name.startswith('head.');parameters[name].copy_(value)
  for index in indices:
   row=rows[index];assert row['split'] in ['train','dev'];crops=[];affines=[];masks=[];boxes=[]
   for view in row['views']:
    assert digest(ROOT/view['image'])==view['image_sha256'];rgb=np.array(Image.open(ROOT/view['image']).convert('RGB'));assert rgb.shape==(384,384,3)
    image,label,meta=letterbox(rgb,np.zeros((384,384),np.uint8),512)
    with torch.autocast('cuda',dtype=torch.bfloat16):predictions=model(image[None].cuda(),'M');scores=semantic_scores(predictions[-1],(512,512))
    mask=native_mask(scores.argmax(1)[0].cpu().numpy(),meta)>0;box,a=roi_affine(mask);x,y,size=box
    crop=cv2.resize(rgb[y:y+size,x:x+size],(384,384),interpolation=cv2.INTER_LINEAR);crops.append(torch.from_numpy(crop).permute(2,0,1).float()/255);affines.append(a);masks.append(mask);boxes.append(box)
   batch=torch.stack(crops).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder.get_intermediate_layers((batch-model.image_mean)/model.image_std,n=[23],reshape=True,norm=True)[0]
   item=torch.load(OLD/'depth_features'/f'{index:06d}.pt',map_location='cpu',weights_only=False)
   # Truth is only copied after inference/box creation, never used by the crop policy.
   item.update(features=features.cpu().half(),feature_affines=torch.from_numpy(np.stack(affines)),predicted_body=torch.from_numpy(np.stack(masks)),crop_xy_size=boxes)
   item['K_feature']=item['feature_affines'].double()@item['K'].double()
   path=args.run_dir/f'{index:06d}.pt';torch.save(item,path);files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'scene_index':index,'id':row['id']})
   for v,(x,y,size) in enumerate(boxes):
    body=item['body'][v].numpy();diagnostics.append({'scene_index':index,'view':v,'split':row['split'],'crop':boxes[v],'predicted_foreground':int(masks[v].sum()),'gt_fraction_inside_crop':float(body[y:y+size,x:x+size].sum()/max(1,body.sum()))})
   if len(files)%12==0:write_json(args.run_dir/'progress.json',{'scenes':len(files),'total':len(indices),'seconds':time.monotonic()-began});print('ROI_CACHE',len(files),len(indices),flush=True)
 receipt={'passed':True,'files':files,'manifest_sha256':digest(OLD/'data_manifest.json'),'encoder_sha256':digest(ROOT/'vggt-omega/vggt_omega_1b_512.pt'),'segmentation_checkpoint_sha256':M_SHA,'layer':23,'shape':[3,1024,24,24],'crop_policy':'all predicted components union bbox,40percent total square padding,min96px,max384px,empty falls back full','pixel_center_policy':'u_crop=s*(u_native-x0+.5)-.5; K_crop=A@K_native; physical rays and metric GT stay native','mask_generator_pose_exposure':'M head trained on all admitted TRAIN source poses; not an unseen-pose experiment','truth_not_used_for_crops':True,'diagnostics_after_crop_only':diagnostics,'seconds':time.monotonic()-began,'source_sha256':digest(pathlib.Path(__file__))};write_json(args.run_dir/'cache_receipt.json',receipt);print(json.dumps({'passed':True,'scenes':len(files),'seconds':receipt['seconds']}))
if __name__=='__main__':main()
