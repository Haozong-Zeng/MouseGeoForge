import unittest,torch
from mousegeoforge.models.quarter_turn_evidence import quarter_turn

class QuarterTurnTest(unittest.TestCase):
    def setUp(self):
        self.x=torch.arange(3*5*7).reshape(1,3,5,7).float();self.t=torch.tensor([[[1/7,2/5],[5/7,3/5]]]);self.v=torch.ones(1,2,dtype=torch.bool)
    def test_identity(self):
        x,t,hw=quarter_turn(self.x,self.t,self.v,(5,7),0);self.assertIs(x,self.x);self.assertIs(t,self.t);self.assertEqual(hw,(5,7))
    def test_pixel_correspondence_all_turns(self):
        for k in [1,2,3]:
            x,t,hw=quarter_turn(self.x,self.t,self.v,(5,7),k)
            for n,(xx,yy) in enumerate([(1,2),(5,3)]):
                p=(t[0,n]*torch.tensor([hw[1],hw[0]])).round().long();self.assertTrue(torch.equal(x[0,:,p[1],p[0]],self.x[0,:,yy,xx]))
    def test_roundtrip(self):
        x,t,hw=quarter_turn(self.x,self.t,self.v,(5,7),1);x,t,hw=quarter_turn(x,t,self.v,hw,3);self.assertTrue(torch.equal(x[...,:5,:7],self.x));torch.testing.assert_close(t,self.t)
    def test_padding_black(self):
        x,_,hw=quarter_turn(self.x,self.t,self.v,(5,7),1);torch.testing.assert_close(x[0,:,hw[0],hw[1]],-torch.tensor([.485,.456,.406])/torch.tensor([.229,.224,.225]))
    def test_unknown_nan_preserved(self):
        self.t[0,1]=float('nan');self.v[0,1]=False;_,t,_=quarter_turn(self.x,self.t,self.v,(5,7),2);self.assertTrue(t[0,1].isnan().all())
    def test_no_point_permutation_or_input_mutation(self):
        before=self.t.clone();_,t,_=quarter_turn(self.x,self.t,self.v,(5,7),2);torch.testing.assert_close(self.t,before);torch.testing.assert_close(t[0,0],torch.tensor([5/7,2/5]))
    def test_invalid(self):
        with self.assertRaises(ValueError):quarter_turn(self.x,self.t,self.v,(5,7),4)
if __name__=='__main__':unittest.main()
