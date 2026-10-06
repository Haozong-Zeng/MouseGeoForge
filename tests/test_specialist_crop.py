import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from mousegeoforge.models.specialist_crop import make_specialist_crop,decode_specialist_crop,CORE_INDICES

class CropTests(unittest.TestCase):
    def setUp(self):self.im=Image.new('RGB',(100,80),'white');self.base=torch.rand(23,2)
    def test_affine_roundtrip(self):
        _,m=make_specialist_crop(self.im,[.5,.5,.4,.5]);A=np.array(m['affine']);p=np.array([[0,0,1],[20,30,1],[99,79,1]]).T
        self.assertTrue(np.allclose(np.linalg.inv(A)@(A@p),p));self.assertEqual(m['expansion'],1.25)
    def test_invalid(self):
        for b in [[np.nan,0,1,1],[.5,.5,0,1],[5,5,.1,.1],[.5,.5,100,100]]:self.assertIsNone(make_specialist_crop(self.im,b))
    def test_defaultoff(self):self.assertIs(decode_specialist_crop(None,self.base,None),self.base)
    def test_missingmeta_fallback(self):self.assertIs(decode_specialist_crop(None,self.base,None,enabled=True),self.base)
    def test_outside_image_and_letterbox(self):
        im,m=make_specialist_crop(self.im,[0,0,.5,.5]);self.assertEqual(im.size,(704,768));self.assertLess(m['crop_ltrb'][0],0);self.assertGreater(m['valid_ltrb'][0],0)
        self.assertEqual(im.getpixel((0,0)),(0,0,0))
    def test_decode_centers_and_non_targets(self):
        _,m=make_specialist_crop(self.im,[.5,.5,.4,.5]);hm=torch.zeros(5,192,176);hm[:,96,88]=2
        out=decode_specialist_crop(hm,self.base,m,enabled=True);A=np.array(m['affine']);want=(np.array([354,386])-A[:2,2])/np.diag(A)[:2]/[100,80]
        self.assertTrue(np.allclose(out[0],want));other=[k for k in range(23) if k not in CORE_INDICES];self.assertTrue(torch.equal(out[other],self.base[other]))
    def test_padding_peak_excluded(self):
        _,m=make_specialist_crop(self.im,[0,0,.5,.5]);hm=torch.zeros(5,192,176);hm[:,0,0]=100;hm[:,150,140]=2
        out=decode_specialist_crop(hm,self.base,m,enabled=True);self.assertTrue(torch.isfinite(out).all());self.assertTrue((out[list(CORE_INDICES)]>=0).all())
    def test_nan_heatmap_fallback(self):
        _,m=make_specialist_crop(self.im,[.5,.5,.4,.5]);out=decode_specialist_crop(torch.full((5,192,176),float('nan')),self.base,m,enabled=True);self.assertTrue(torch.equal(out,self.base))
if __name__=='__main__':unittest.main()
