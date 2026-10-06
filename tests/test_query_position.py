import unittest
from types import SimpleNamespace
import torch
from torch import nn
from mousegeoforge.query_position import LayerQueryPosition,install_query_position,query_position_context


class QueryPositionTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(91)
        h=nn.Module();h.hidden_dim=16;h.num_queries=6
        h.instance_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(16,4,dropout=0,batch_first=True,norm_first=True),2,norm=nn.LayerNorm(16))
        h.eval();return h,torch.randn(2,6,16),torch.randn(2,20,16)

    def test_disabled_without_model(self):
        with query_position_context():pass

    def test_zero_parity_rng_and_cleanup(self):
        h,q,m=self.fixture()
        with torch.no_grad():before=h.instance_decoder(q,m)
        rng=torch.random.get_rng_state().clone();install_query_position(h)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        with query_position_context(h,enabled=True),torch.no_grad():after=h.instance_decoder(q,m)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        self.assertTrue(all(len(x.multihead_attn._forward_pre_hooks)==0 for x in h.instance_decoder.layers))

    def test_nonzero_gradient_and_effect(self):
        h,q,m=self.fixture();install_query_position(h)
        with query_position_context(h,enabled=True):h.instance_decoder(q,m).square().mean().backward()
        g=h.query_position.offsets.grad
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
        with torch.no_grad():
            old=h.instance_decoder(q,m);h.query_position.offsets.normal_(0,.2)
            with query_position_context(h,enabled=True):new=h.instance_decoder(q,m)
        self.assertGreater(float((old-new).abs().max()),0)

    def test_nonreentrant_exception_cleanup(self):
        h,q,m=self.fixture();install_query_position(h)
        with self.assertRaises(RuntimeError),query_position_context(h,enabled=True):
            with query_position_context(h,enabled=True):pass
        self.assertFalse(h._query_position_active)
        self.assertTrue(all(len(x.multihead_attn._forward_pre_hooks)==0 for x in h.instance_decoder.layers))

    def test_geometry_guard(self):
        p=LayerQueryPosition(2,6,16)
        with self.assertRaises(ValueError):p(torch.zeros(1,4,16),0)


if __name__=='__main__':unittest.main()
