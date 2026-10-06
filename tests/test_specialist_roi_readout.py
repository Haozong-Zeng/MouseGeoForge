import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mousegeoforge.models.specialist_roi_readout import specialist_roi_readout,CORE_INDICES
class SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.hm=torch.zeros(5,4,4);self.hm[:,1,1]=2;self.hm[:,2,3]=3;self.b=torch.tensor([[.25,.5,.5,1.],[.75,.5,.5,1.]]);self.xy=torch.rand(2,23,2)
    def run_readout(self,**kw):return specialist_roi_readout(self.hm,self.b,self.xy,(16,16),enabled=True,**kw)
    def test_default_off_identity(self):
        out=specialist_roi_readout(self.hm,self.b,self.xy,(16,16));self.assertIs(out['xy'],self.xy)
    def test_predicted_roi_and_stride_centers(self):
        out=self.run_readout();self.assertTrue(torch.equal(out['xy'][0,0],torch.tensor([6/16,6/16])));self.assertTrue(torch.equal(out['xy'][1,0],torch.tensor([14/16,10/16])))
    def test_non_target_unchanged(self):
        ids=[i for i in range(23) if i not in CORE_INDICES];self.assertTrue(torch.equal(self.run_readout()['xy'][:,ids],self.xy[:,ids]))
    def test_query_permutation(self):
        base=self.run_readout();p=torch.tensor([1,0]);out=specialist_roi_readout(self.hm,self.b[p],self.xy[p],(16,16),enabled=True);self.assertTrue(torch.equal(out['xy'],base['xy'][p]))
    def test_invalid_empty_fallback(self):
        self.b[0]=float('nan');self.b[1]=torch.tensor([.01,.01,.001,.001]);out=self.run_readout();self.assertTrue(torch.equal(out['xy'],self.xy));self.assertFalse(out['supported'].any());self.assertTrue(torch.isfinite(out['scores']).all())
    def test_padding_excluded(self):
        self.hm[:,3]=100;box=torch.tensor([[.5,.5,1.,1.]]);out=specialist_roi_readout(self.hm,box,self.xy[:1],(12,16),enabled=True);self.assertTrue(torch.equal(out['xy'][0,0],torch.tensor([14/16,10/12])))
    def test_nan_channel_fallback(self):
        self.hm[2]=float('nan');out=self.run_readout();self.assertFalse(out['supported'][:,2].any());self.assertTrue(torch.equal(out['xy'][:,CORE_INDICES[2]],self.xy[:,CORE_INDICES[2]]))
if __name__=='__main__':unittest.main()
