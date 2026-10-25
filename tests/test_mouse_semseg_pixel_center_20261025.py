import json,unittest
from pathlib import Path
import cv2,numpy as np,torch
from mousegeoforge.semseg_training import ROOT,letterbox,read_sample,digest
from mousegeoforge.semseg_pixel_center_20261025 import label_letterbox,read_center_sample,native_probability,native_label
class PixelCenterTests(unittest.TestCase):
 def test_legacy_real_TRAIN_preprocessing_bitwise(self):
  rows=json.loads((ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json').read_text())['items']
  for domain,bottom in [('real',True),('real',False),('synthetic',True),('synthetic',False)]:
   selected=sorted([r for r in rows if r['split']=='train' and (r['domain'],r['bottom'])==(domain,bottom)],key=lambda r:r['id'])[:3]
   for row in selected:
    self.assertEqual(digest(ROOT/row['image']),row['image_sha256']);self.assertEqual(digest(ROOT/row['label']),row['label_sha256'])
    old=read_sample(row,ROOT,512);new=read_center_sample(row,ROOT,512,'legacy')
    self.assertTrue(torch.equal(old[0],new[0]));self.assertTrue(torch.equal(old[1],new[1]));self.assertEqual(old[2],new[2])
    center=read_center_sample(row,ROOT,512,'center_exact');self.assertTrue(torch.equal(old[0],center[0]));self.assertEqual(old[2],center[2])
 def test_ignore_preserved_and_padding_background(self):
  image=np.zeros((13,31,3),np.uint8);label=np.zeros((13,31),np.uint8);label[2:7,10:15]=255;label[7:12,1:5]=1
  _,target,meta=label_letterbox(image,label,64,'center_exact');self.assertEqual(set(target.unique().tolist()),{0,1,255})
  y,x,h,w=meta['roi'];self.assertTrue(bool((target[:y]==0).all()));self.assertTrue(bool((target[y+h:]==0).all()))
 def test_center_target_uses_declared_interpolation(self):
  image=np.zeros((384,384,3),np.uint8);label=np.zeros((384,384),np.uint8);label[20:60,37:57]=1
  old=label_letterbox(image,label,512,'legacy');new=label_letterbox(image,label,512,'center_exact')
  np.testing.assert_array_equal(new[1].numpy(),cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST_EXACT));self.assertFalse(torch.equal(old[1],new[1]))
 def test_native_probability_uses_ROI_and_halfpixel(self):
  p=np.zeros((8,8),np.float32);p[2:6,1:7]=np.arange(6,dtype=np.float32)[None,:]/5
  meta={'roi':[2,1,4,6],'native_hw':[7,13]};native,binary=native_probability(p,meta)
  expected=cv2.resize(p[2:6,1:7],(13,7),interpolation=cv2.INTER_LINEAR)
  np.testing.assert_array_equal(native,expected);np.testing.assert_array_equal(binary,(expected>.5).astype(np.uint8));self.assertEqual(native.dtype,np.float32)
 def test_unknown_policy_and_tie_background(self):
  with self.assertRaises(ValueError):label_letterbox(np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint8),4,'bad')
  _,mask=native_probability(np.full((2,2),.5,np.float32),{'roi':[0,0,2,2],'native_hw':[2,2]});self.assertFalse(bool(mask.any()))
if __name__=='__main__':unittest.main()
