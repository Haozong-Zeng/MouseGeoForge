import unittest
import numpy as np
from mousegeoforge.semseg_roi_20261025 import predicted_box,roi_sample,native_probability
from mousegeoforge.semseg_pixel_center_20261025 import label_letterbox,native_probability as old_inverse
class ROI(unittest.TestCase):
 def test_empty_fallback(self):self.assertEqual(predicted_box(np.zeros((384,384),bool)),[0,0,384,384])
 def test_prediction_only_padding_bounds(self):
  m=np.zeros((384,384),bool);m[180:200,180:220]=True;b=predicted_box(m);self.assertEqual(b,[152,142,248,238]);self.assertTrue(0<=b[0]<b[2]<=384)
 def test_full_preprocess_and_inverse_identical(self):
  rng=np.random.default_rng(4);im=rng.integers(256,size=(67,103,3),dtype=np.uint8);label=rng.integers(2,size=(67,103),dtype=np.uint8)
  a=roi_sample(im,label,128,'full',[1,2,30,40]);b=label_letterbox(im,label,128,'center_exact');np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);p=rng.random((128,128),dtype=np.float32);np.testing.assert_array_equal(native_probability(p,a[2])[0],old_inverse(p,b[2])[0])
 def test_crop_inverse_zeros_outside_full_GT_denominator(self):
  im=np.zeros((100,120,3),np.uint8);label=np.ones((100,120),np.uint8);_,_,meta=roi_sample(im,label,64,'roi',[20,30,80,90]);prob,mask=native_probability(np.ones((64,64),np.float32),meta);self.assertEqual(mask.shape,label.shape);self.assertEqual(mask.sum(),60*60);self.assertFalse(mask[0,0]);self.assertTrue(mask[50,50]);self.assertEqual(label.sum(),12000)
 def test_pixel_center_affine(self):
  im=np.zeros((100,120,3),np.uint8);_,_,meta=roi_sample(im,np.zeros((100,120),np.uint8),64,'roi',[20,30,84,94]);a=np.array(meta['pixel_center_affine']);np.testing.assert_allclose(a@np.array([20,30,1]),[0,0],atol=1e-12)
 def test_tie_background_and_invalid_crop(self):
  im=np.zeros((100,100,3),np.uint8);_,_,meta=roi_sample(im,np.zeros((100,100),np.uint8),64,'roi',[20,20,84,84]);self.assertEqual(native_probability(np.full((64,64),.5,np.float32),meta)[1].sum(),0)
  with self.assertRaises(ValueError):roi_sample(im,np.zeros((100,100),np.uint8),64,'roi',[-1,0,100,100])
if __name__=='__main__':unittest.main()
