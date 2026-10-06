import inspect,unittest
import torch
from mousegeoforge.models.dense_instance_centers import (
    DenseInstanceCenters,center_heatmap_targets,dense_center_loss,decode_centers,dense_center_gaussian_loss,
)
from mousegeoforge.models.local_point_evidence_probe import LocalPointEvidenceProbe


class DenseCenterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.features=torch.randn(32,20,24,requires_grad=True)
        self.evidence=LocalPointEvidenceProbe()(self.features,(72,88))
        self.evidence['features']=self.features
        self.centers=torch.tensor([[.25,.3],[.75,.7]])
        self.valid=torch.tensor([True,True])

    def output(self):
        return DenseInstanceCenters()(self.evidence,(72,88))

    def test_forward_no_gt_and_gradient(self):
        self.assertEqual(list(inspect.signature(DenseInstanceCenters.forward).parameters),['self','evidence','image_hw'])
        out=self.output();loss=dense_center_loss(out,self.centers,self.valid,(72,88));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(self.features.grad.norm()),0)

    def test_two_centers_not_mixture_average(self):
        target=center_heatmap_targets(self.output(),self.centers,self.valid,(72,88))
        self.assertEqual(int(target['positive'].sum()),2);self.assertEqual(target['collisions'],0)
        self.assertTrue((target['heatmap'][target['positive']]==1).all())

    def test_permutation_invariant(self):
        out=self.output()
        torch.testing.assert_close(dense_center_loss(out,self.centers,self.valid,(72,88)),
                                   dense_center_loss(out,self.centers.flip(0),self.valid,(72,88)))

    def test_missing_center_not_positive(self):
        out=self.output();xy=self.centers.clone();xy[1]=torch.nan;valid=torch.tensor([True,False])
        self.assertEqual(int(center_heatmap_targets(out,xy,valid,(72,88))['positive'].sum()),1)
        self.assertTrue(torch.isfinite(dense_center_loss(out,xy,valid,(72,88))))

    def test_padding_excluded(self):
        out=self.output();target=center_heatmap_targets(out,self.centers,self.valid,(72,88))
        self.assertFalse(target['heatmap'][~out['spatial_valid']].any())
        original=dense_center_loss(out,self.centers,self.valid,(72,88))
        out['logits']=out['logits'].masked_fill(~out['spatial_valid'],1e8)
        torch.testing.assert_close(original,dense_center_loss(out,self.centers,self.valid,(72,88)))

    def test_empty_complete_and_incomplete(self):
        out=self.output();valid=torch.zeros(2,dtype=torch.bool)
        loss=dense_center_loss(out,self.centers,valid,(72,88));self.assertGreater(float(loss.detach()),0)
        self.assertEqual(float(dense_center_loss(out,self.centers,valid,(72,88),annotation_complete=False).detach()),0)

    def test_exact_peaks_decode_two_without_count_input(self):
        out=self.output();target=center_heatmap_targets(out,self.centers,self.valid,(72,88))
        out['logits']=torch.logit(target['heatmap'].clamp(.001,.999))[None]
        decoded=decode_centers(out,(72,88))
        self.assertEqual(int(decoded['instance_active'].sum()),2)
        dist=torch.cdist(decoded['centers']*torch.tensor([88,72]),self.centers*torch.tensor([88,72]))
        self.assertTrue((dist.min(1).values<3.).all())

    def test_flat_no_forced_detection(self):
        out=self.output();out['logits']=torch.zeros_like(out['logits']).masked_fill(~out['spatial_valid'],-torch.inf)
        self.assertFalse(decode_centers(out,(72,88))['instance_active'].any())

    def test_collision_explicitly_reported(self):
        target=center_heatmap_targets(self.output(),self.centers[:1].expand(2,-1),self.valid,(72,88))
        self.assertEqual(target['collisions'],1)

    def test_gaussian_field_fit_and_permutation(self):
        out=self.output();target=center_heatmap_targets(out,self.centers,self.valid,(72,88))['heatmap']
        before=dense_center_gaussian_loss(out,self.centers,self.valid,(72,88))
        torch.testing.assert_close(before,dense_center_gaussian_loss(out,self.centers.flip(0),self.valid,(72,88)))
        out['logits']=torch.logit(target.clamp(1e-5,1-1e-5))[None]
        after=dense_center_gaussian_loss(out,self.centers,self.valid,(72,88))
        self.assertLess(float(after),float(before));self.assertLess(float(after),1e-6)

    def test_gaussian_field_padding_gradient_unknown_and_empty(self):
        out=self.output();out['logits'].retain_grad()
        loss=dense_center_gaussian_loss(out,self.centers,self.valid,(72,88));loss.backward()
        self.assertTrue(torch.isfinite(self.features.grad).all());self.assertGreater(float(self.features.grad.norm()),0.)
        self.assertFalse(out['logits'].grad[0,~out['spatial_valid']].any())
        modified=dict(out,logits=out['logits'].masked_fill(~out['spatial_valid'],1e8))
        torch.testing.assert_close(loss,dense_center_gaussian_loss(modified,self.centers,self.valid,(72,88)))
        xy=self.centers.clone();xy[1]=torch.nan
        self.assertTrue(torch.isfinite(dense_center_gaussian_loss(out,xy,torch.tensor([True,False]),(72,88))))
        self.assertGreater(float(dense_center_gaussian_loss(out,xy,torch.tensor([False,False]),(72,88)).detach()),0.)
        with self.assertRaises(ValueError):dense_center_gaussian_loss(out,self.centers,self.valid,(72,88),annotation_complete=False)

if __name__=='__main__':unittest.main()
