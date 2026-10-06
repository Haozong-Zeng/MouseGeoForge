import unittest
import torch
from torch import nn
from mousegeoforge.spatial_query_prior import ConditionalSpatialPrior, install_spatial_query_prior, spatial_query_context


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim=16; self.num_queries=6; self.patch_size=4
        self.instance_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(16,4,dropout=0,batch_first=True,norm_first=True),2)

    def forward(self, memory, images, query):
        return self.instance_decoder(query, memory)


class SpatialPriorTests(unittest.TestCase):
    def fixture(self, views=2):
        torch.manual_seed(71)
        head=TinyHead().eval()
        return head, torch.randn(views,6,16), torch.randn(views,20,16), torch.zeros(1,views,3,16,20)

    def test_disabled_without_model(self):
        with spatial_query_context(): pass

    def test_disabled_exact_rng_and_cleanup(self):
        h,q,m,images=self.fixture()
        with torch.no_grad(): expected=h(m,images,q)
        rng=torch.get_rng_state().clone(); install_spatial_query_prior(h)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with spatial_query_context(h,enabled=False),torch.no_grad(): actual=h(m,images,q)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
        with spatial_query_context(h,enabled=True),torch.no_grad(): changed=h(m,images,q)
        self.assertGreater(float((changed-expected).abs().max()),0)
        with torch.no_grad(): restored=h(m,images,q)
        torch.testing.assert_close(expected,restored,atol=0,rtol=0)
        self.assertEqual(len(h._forward_pre_hooks),0)
        self.assertTrue(all(not x.multihead_attn._forward_pre_hooks for x in h.instance_decoder.layers))

    def test_shapes_soft_finite_and_gradients(self):
        p=ConditionalSpatialPrior(2,6,16)
        q=torch.randn(4,6,16,requires_grad=True)
        for layer in range(2):
            b=p(q,layer,(3,5),4)
            self.assertEqual(tuple(b.shape),(16,6,15))
            self.assertTrue(torch.isfinite(b).all())
            self.assertGreaterEqual(float(b.detach().min()),-20.)
            self.assertGreater(float(b.detach().max()-b.detach().min()),0)
            b.square().mean().backward()
        for param in p.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.norm()),0)

    def test_centers_can_coincide_and_move(self):
        p=ConditionalSpatialPrior(2,6,16)
        with torch.no_grad(): p.center_logits.fill_(0.)
        c,s,g=p.geometry(torch.zeros(1,6,16),0)
        torch.testing.assert_close(c,torch.full_like(c,.5),atol=0,rtol=0)
        with torch.no_grad(): p.center_logits[0,0]=torch.tensor([3.,-3.])
        c,_,_=p.geometry(torch.zeros(1,6,16),0)
        self.assertGreater(float(c[0,0,0].detach()),.9)
        self.assertLess(float(c[0,0,1].detach()),.1)

    def test_dynamic_condition_and_non_square_axes(self):
        p=ConditionalSpatialPrior(1,1,4)
        with torch.no_grad(): p.condition_weight[0,0,0]=1
        c,_,_=p.geometry(torch.tensor([[[1.,0,0,0]],[[-1.,0,0,0]]]),0)
        self.assertGreater(float(c[0,0,0].detach()),float(c[1,0,0].detach()))
        with torch.no_grad(): p.condition_weight.zero_();p.center_logits[:]=torch.logit(torch.tensor([.9,.1]))
        b=p(torch.zeros(1,1,4),0,(2,4),1)
        self.assertEqual(int(b[0,0].argmax()),3)

    def test_view_counts_and_backward(self):
        for views in (1,2,4):
            h,q,m,images=self.fixture(views);install_spatial_query_prior(h)
            with spatial_query_context(h,enabled=True): h(m,images,q).square().mean().backward()
            self.assertIsNotNone(h.spatial_query_prior.center_logits.grad)
            self.assertTrue(torch.isfinite(h.spatial_query_prior.center_logits.grad).all())

    def test_reentrancy_and_geometry_errors_cleanup(self):
        h,q,m,images=self.fixture();install_spatial_query_prior(h)
        with self.assertRaises(RuntimeError),spatial_query_context(h,enabled=True):
            with spatial_query_context(h,enabled=True): pass
        self.assertFalse(h._spatial_query_active)
        with self.assertRaises(ValueError),spatial_query_context(h,enabled=True):h(m[:,:10],images,q)
        self.assertFalse(h._spatial_query_active)
        self.assertEqual(len(h._forward_pre_hooks),0)


if __name__=='__main__': unittest.main()
