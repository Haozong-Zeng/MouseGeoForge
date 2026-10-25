import unittest,sys,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np,torch,cv2
from cache_calibrated_mouse_roi_20261025 import roi_affine
from mousegeoforge.models.calibrated_mouse_depth import CalibratedRayDepth,project_world
from mousegeoforge.models.calibrated_mouse_depth_roi import CalibratedRayDepthROI
class TestROICalibration(unittest.TestCase):
 def test_empty_fallback(self):
  box,a=roi_affine(np.zeros((384,384),bool));self.assertEqual(box,(0,0,384));np.testing.assert_array_equal(a,np.eye(3))
 def test_border_and_sparse_components_retained(self):
  m=np.zeros((384,384),bool);m[0:12,0:7]=True;m[80,130]=True;box,a=roi_affine(m);x,y,size=box;self.assertGreaterEqual(size,96);self.assertEqual(int(m[y:y+size,x:x+size].sum()),int(m.sum()))
 def test_half_pixel_resize(self):
  m=np.zeros((384,384),bool);m[101:141,180:200]=True;(x,y,size),a=roi_affine(m)
  self.assertAlmostEqual(float(a[0,0]),384/size,places=6);uv=np.array([x,y,1]);np.testing.assert_allclose((a@uv)[:2],[(384/size-1)/2]*2,atol=1e-5)
 def test_k_projection_and_inverse(self):
  _,a=roi_affine(np.pad(np.ones((20,30),bool),((100,264),(110,244))))
  a=torch.tensor(a,dtype=torch.float64);k=torch.tensor([[420.,0,191.5],[0,410.,191.5],[0,0,1]],dtype=torch.float64);e=torch.eye(4,dtype=torch.float64);w=torch.tensor([[10.,-5.,300.],[-40.,10.,350.]],dtype=torch.float64)
  uv,z=project_world(w,k,e);crop,z2=project_world(w,a@k,e);expected=torch.cat([uv,torch.ones(2,1,dtype=torch.float64)],-1)@a.T;torch.testing.assert_close(crop,expected[:,:2]);torch.testing.assert_close(z,z2)
  original=torch.cat([crop,torch.ones(2,1,dtype=torch.float64)],-1)@torch.linalg.inv(a).T;torch.testing.assert_close(original[:,:2],uv)
 def test_identity_matches_original_forward_and_gradient(self):
  torch.manual_seed(17);old=CalibratedRayDepth(in_channels=8,channels=8,planes=16);new=CalibratedRayDepthROI(in_channels=8,channels=8,planes=16);new.load_state_dict(old.state_dict());f=torch.randn(3,8,24,24);xy=torch.tensor([[180.,180.],[200.,200.]])
  k=torch.tensor([[300.,0,191.5],[0,300.,191.5],[0,0,1]]).repeat(3,1,1);e=torch.eye(4).repeat(3,1,1);e[1,0,3]=15;e[2,1,3]=15;aabb=torch.tensor([[-100.,-100.,100.],[100.,100.,400.]])
  a=old(old.encode(f),xy,0,k,e,aabb);b=new(new.encode(f),xy,0,k,e,aabb,feature_affines=torch.eye(3).repeat(3,1,1))
  for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
  a[0].sum().backward();b[0].sum().backward()
  for x,y in zip(old.parameters(),new.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=0,rtol=0)
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
