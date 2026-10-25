import unittest,cv2,numpy as np
class PixelCenterControl(unittest.TestCase):
 def test_exact_nearest_follows_center_coordinate(self):
  for w,n in [(384,512),(1080,461),(1200,512),(3,4),(4,3)]:
   a=np.arange(w,dtype=np.float64)[None,:]
   source_center=np.clip((np.arange(n)+.5)*w/n-.5,0,w-1)
   got=cv2.resize(a,(n,1),interpolation=cv2.INTER_NEAREST_EXACT)[0]
   self.assertLessEqual(float(np.max(np.abs(got-source_center))),.500001)
 def test_legacy_nearest_can_miss_center_nearest(self):
  a=np.arange(3,dtype=np.float64)[None,:]
  self.assertFalse(np.array_equal(cv2.resize(a,(4,1),interpolation=cv2.INTER_NEAREST),cv2.resize(a,(4,1),interpolation=cv2.INTER_NEAREST_EXACT)))
 def test_center_resize_preserves_upsampled_discrete_label(self):
  a=np.zeros((384,384),np.uint8);a[130:180,123:142]=1;a[100:200,200]=1
  b=cv2.resize(cv2.resize(a,(512,512),interpolation=cv2.INTER_NEAREST_EXACT),(384,384),interpolation=cv2.INTER_NEAREST_EXACT)
  np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
