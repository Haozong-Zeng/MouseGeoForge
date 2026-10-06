"""Deterministic CPU proof of zero-position Q injection's local derivative."""
import unittest
import torch
from mousegeoforge.query_position import LayerQueryPosition


class QueryPositionDerivativeTests(unittest.TestCase):
    def test_zero_injection_identity_jacobian(self):
        gen=torch.Generator().manual_seed(113)
        query=torch.randn(2,6,16,generator=gen,dtype=torch.float64,requires_grad=True)
        upstream=torch.randn(2,6,16,generator=gen,dtype=torch.float64)
        branch=LayerQueryPosition(2,6,16).double()
        output=branch(query,1)
        torch.testing.assert_close(output,query,atol=0,rtol=0)
        (output*upstream).sum().backward()
        torch.testing.assert_close(query.grad,upstream,atol=0,rtol=0)
        torch.testing.assert_close(branch.offsets.grad[1],upstream.sum(0),atol=0,rtol=0)
        self.assertEqual(int(torch.count_nonzero(branch.offsets.grad[0])),0)


if __name__=='__main__':unittest.main()
