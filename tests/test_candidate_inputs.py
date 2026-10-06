import unittest
import numpy as np
import torch
from mousegeoforge.candidate_inputs import local_candidate_inputs


class CandidateInputTests(unittest.TestCase):
    def test_baseline_and_shapes(self):
        p=np.zeros((2,3,8,8),np.float32);p[:,:,2,3]=1
        xy=np.full((2,3,2),.5,np.float32);memory=torch.ones(1,7,8,8,requires_grad=True)
        out=local_candidate_inputs(p,xy,memory)
        np.testing.assert_array_equal(out['candidate_xy'][:,:,0],xy)
        self.assertEqual(out['candidate_memory'].shape,(2,3,6,7))
        self.assertFalse(out['candidate_memory'].requires_grad)
        torch.testing.assert_close(out['candidate_memory'],torch.ones(2,3,6,7,dtype=torch.float16))
    def test_point_sampling_alignment(self):
        p=np.zeros((1,1,8,8),np.float32);p[0,0,2,3]=1
        xy=np.array([[[3.5/8,2.5/8]]],np.float32)
        memory=torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)
        out=local_candidate_inputs(p,xy,memory)
        self.assertEqual(float(out['candidate_memory'][0,0,0,0]),19)
        self.assertEqual(float(out['candidate_memory'][0,0,1,0]),19)
        self.assertEqual(float(out['candidate_mass'][0,0,0]),1)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):local_candidate_inputs(np.zeros((1,1,8,8)),np.zeros((1,1,2)),torch.zeros(1,4,8,8))


if __name__=='__main__':unittest.main()
