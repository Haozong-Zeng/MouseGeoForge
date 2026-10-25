import importlib.util
from pathlib import Path
import unittest
import torch
P=Path(__file__).resolve().parents[1]/'tools/calibrated_mouse_classical_baselines_20261025.py'
spec=importlib.util.spec_from_file_location('classic',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Classical(unittest.TestCase):
 def setup_scene(self,scale=1.):
  k=torch.tensor([[20.,0,8],[0,20,8],[0,0,1]]).repeat(3,1,1);e=torch.eye(4).repeat(3,1,1);centers=torch.tensor([-1.,0,1])*scale;e[:,0,3]=-centers
  yy,xx=torch.meshgrid(torch.arange(17),torch.arange(17),indexing='ij');rgbs=[]
  for c in centers:
   color=.5+.01*((xx-8)*.5+c/scale)+.02*(yy-8)*.5;rgbs.append(color.repeat(3,1,1))
  return torch.stack(rgbs),torch.ones(3,17,17,dtype=torch.bool),k,e,torch.tensor([[-20.,-20.,8.],[20.,20.,12.]])*scale,torch.tensor([[8.,8.]])
 def test_known_textured_plane_camera_z(self):
  rgb,mask,k,e,b,xy=self.setup_scene();r=m.sweep(rgb,mask,k,e,b,1,xy,planes=5);self.assertAlmostEqual(float(r['rgb'][0]),10.,places=5)
 def test_hull_first_occupied_is_near(self):
  rgb,mask,k,e,b,xy=self.setup_scene();r=m.sweep(rgb,mask,k,e,b,1,xy,planes=5);self.assertEqual(float(r['hull'][0]),8.)
 def test_hull_missing_view_not_silently_kept(self):
  rgb,mask,k,e,b,xy=self.setup_scene();mask[0]=False;r=m.sweep(rgb,mask,k,e,b,1,xy,planes=5);self.assertEqual(float(r['hull'][0]),0.);self.assertFalse(r['hull_valid'][0])
 def test_mm_scale_preserved(self):
  rgb,mask,k,e,b,xy=self.setup_scene(10.);r=m.sweep(rgb,mask,k,e,b,1,xy,planes=5);self.assertAlmostEqual(float(r['rgb'][0]),100.,places=4)
 def test_outside_volume_rejected(self):
  rgb,mask,k,e,b,xy=self.setup_scene();xy[0]=torch.tensor([1000.,1000.]);r=m.sweep(rgb,mask,k,e,b,1,xy,planes=5);self.assertEqual(float(r['rgb'][0]),0.);self.assertEqual(float(r['hull'][0]),0.)
if __name__=='__main__':unittest.main()
