import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('export',ROOT/'tools/export_calibrated_mouse_pointcloud_review_20261025.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class PointcloudArtifactTests(unittest.TestCase):
 def test_camera_Z_rotation_translation_and_pixel_colors(self):
  z=np.full((3,4),120.,np.float64);mask=np.zeros_like(z,dtype=bool);mask[0,1]=mask[2,3]=True;k=np.array([[200.,0,1.5],[0,210.,1],[0,0,1.]])
  e=np.eye(4);e[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];e[:3,3]=[-12,34,-100]
  rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3);world,colors,xy,depth=m.unproject(z,mask,k,e,rgb);uv,back=m.project(world,k,e)
  np.testing.assert_allclose(uv,xy,atol=1e-12);np.testing.assert_allclose(back,depth,atol=1e-12);np.testing.assert_array_equal(colors,rgb[mask]);self.assertEqual(world.shape,(2,3))
 def test_invalid_depth_is_counted_out_without_confidence_rule(self):
  z=np.array([[12,0],[np.nan,-1]],float);world,_,_,_=m.unproject(z,np.ones((2,2),bool),np.eye(3),np.eye(4),np.zeros((2,2,3),np.uint8));self.assertEqual(len(world),1)
 def test_PLY_binary_count_colors_views_and_units(self):
  world=np.array([[10.12345678,-20.4,415.00123],[12,14,360]],float);color=np.array([[3,200,255],[0,5,8]],np.uint8);view=np.array([0,2],np.uint8)
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'test.ply';m.write_ply(path,world,color,view);r=m.read_ply(path)
   np.testing.assert_allclose(np.c_[r['x'],r['y'],r['z']],world,atol=3e-5,rtol=0);np.testing.assert_array_equal(np.c_[r['red'],r['green'],r['blue']],color);np.testing.assert_array_equal(r['source_view'],view)
 def test_no_nan_color_indexing_and_shared_scale(self):
  z=np.array([[100,200],[np.nan,0]],float);valid=np.array([[1,1],[0,0]],bool);c=m.palette(z,valid,100,200);np.testing.assert_array_equal(c[~valid],0);self.assertFalse(np.array_equal(c[0,0],c[0,1]))
 def test_nonfinite_export_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaises(AssertionError):m.write_ply(Path(temp)/'bad.ply',np.array([[np.nan,0,0]]),np.zeros((1,3),np.uint8),np.zeros(1,np.uint8))
if __name__=='__main__':unittest.main()
