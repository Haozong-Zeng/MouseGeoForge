import inspect
import unittest

import torch

from mousegeoforge.models.local_instance_grouping import (
    LocalInstanceGrouping, assignment_targets, grouping_loss, decode_grouped, rotate_mask_centroids,
)
from mousegeoforge.models.local_point_evidence_probe import LocalPointEvidenceProbe


class GroupingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def evidence(self, flat=False):
        features = torch.randn(32, 12, 16, requires_grad=True)
        out = LocalPointEvidenceProbe()(features, (40, 60))
        # Avoid the default zero-initialized heatmap unless explicitly testing it.
        out['logits'] = (torch.zeros(5, 12, 16) if flat else torch.randn(5, 12, 16)).masked_fill(~out['spatial_valid'], -torch.inf)
        out['features'] = features
        return out

    def output(self):
        return LocalInstanceGrouping()(self.evidence(), (40, 60))

    def targets(self):
        xy = torch.rand(2, 5, 2)
        known = torch.ones(2, 5, dtype=torch.bool)
        valid = torch.ones(2, dtype=torch.bool)
        centers = torch.tensor([[.25, .4], [.75, .5]])
        return xy, known, valid, centers, valid.clone(), (40, 60)

    def test_no_gt_in_forward_interface(self):
        self.assertEqual(list(inspect.signature(LocalInstanceGrouping.forward).parameters),
                         ['self', 'evidence', 'image_hw'])

    def test_both_arms_identical_parameters_and_default_off(self):
        plain = LocalInstanceGrouping()
        context = LocalInstanceGrouping(interaction=True)
        context.load_state_dict(plain.state_dict(), strict=True)
        self.assertFalse(plain.interaction)
        self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in context.parameters()))

    def test_shape_and_finite_loss_gradient_both_arms(self):
        for interaction in [False, True]:
            evidence = self.evidence()
            model = LocalInstanceGrouping(interaction=interaction)
            out = model(evidence, (40, 60))
            self.assertEqual(out['assignment_logits'].shape, (2, 5, 7))
            loss = grouping_loss(out, *self.targets())['total']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(evidence['features'].grad.abs().sum()), 0)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_instance_permutation_invariant_loss(self):
        out = self.output()
        args = self.targets()
        a = grouping_loss(out, *args)['total']
        flipped = tuple(z.flip(0) for z in args[:-1]) + (args[-1],)
        b = grouping_loss(out, *flipped)['total']
        torch.testing.assert_close(a, b)

    def test_unknown_nan_coordinates_ignored_not_null(self):
        out = self.output()
        xy, known, valid, centers, cv, hw = self.targets()
        xy[:] = torch.nan
        known[:] = False
        distributions = assignment_targets(out, xy, known, hw)
        self.assertEqual(float(distributions.sum()), 0.)
        loss = grouping_loss(out, xy, known, valid, centers, cv, hw)
        self.assertEqual(float(loss['assignment']), 0.)
        self.assertTrue(torch.isfinite(loss['total']))

    def test_unlocalized_flat_evidence_all_null(self):
        out = LocalInstanceGrouping()(self.evidence(flat=True), (40, 60))
        self.assertFalse(out['candidate_available'].any())
        out['object_logits'] = torch.ones(2) * 10
        decoded = decode_grouped(out)
        self.assertFalse(decoded['accepted'].any())
        args = self.targets()
        distribution = assignment_targets(out, args[0], args[1], args[-1])
        self.assertTrue((distribution[..., -1] == 1).all())
        self.assertTrue(torch.isfinite(grouping_loss(out, *args)['total']))

    def test_no_reused_candidate_between_mice(self):
        out = self.output()
        out['object_logits'] = torch.ones(2) * 10
        out['assignment_logits'][:] = -10
        out['assignment_logits'][:, :, 0] = 10
        out['assignment_logits'][:, :, 1] = 9
        decoded = decode_grouped(out)
        self.assertTrue(decoded['instance_active'].all())
        for k in range(5):
            chosen = decoded['chosen'][:, k][decoded['accepted'][:, k]]
            self.assertEqual(len(chosen), len(torch.unique(chosen)))

    def test_second_instance_is_positive_not_background(self):
        out = self.output()
        out['object_logits'] = torch.tensor([2., -2.], requires_grad=True)
        grouping_loss(out, *self.targets())['objectness'].backward()
        self.assertTrue((out['object_logits'].grad < 0).all())

    def test_absent_instance_negative_only_if_annotation_complete(self):
        args = list(self.targets())
        args[2] = args[4] = torch.tensor([True, False])
        args[1][1] = False
        for complete in [True, False]:
            out = self.output()
            out['object_logits'] = torch.zeros(2, requires_grad=True)
            result = grouping_loss(out, *args, annotation_complete=complete)
            result['objectness'].backward()
            unmatched = list({0, 1} - set(result['matched_queries'].tolist()))[0]
            grad = float(out['object_logits'].grad[unmatched])
            self.assertGreater(grad, 0) if complete else self.assertEqual(grad, 0)

    def test_all_instances_missing_is_valid_empty_detection(self):
        out = self.output()
        args = list(self.targets())
        args[1][:] = False
        args[2][:] = False
        args[4][:] = False
        result = grouping_loss(out, *args)
        self.assertTrue(torch.isfinite(result['total']))
        self.assertEqual(float(result['assignment']), 0)

    def test_capacity_overflow_fails_not_silently_truncates(self):
        out = LocalInstanceGrouping(max_instances=1)(self.evidence(), (40, 60))
        with self.assertRaises(ValueError):
            grouping_loss(out, *self.targets())

    def test_padding_not_candidates(self):
        out = self.output()
        xy = out['candidate_xy'][out['candidate_available']]
        self.assertTrue(((xy > 0) & (xy < 1)).all())

    def test_mask_centroid_rotation_uses_cell_centers(self):
        mask = torch.zeros(6, 10, dtype=torch.bool)
        mask[1:4, 2:7] = True
        def center(m):
            y, x = m.nonzero(as_tuple=True)
            return torch.tensor([[(x.float().mean() + .5) / m.shape[1],
                                  (y.float().mean() + .5) / m.shape[0]]])
        for k in range(4):
            torch.testing.assert_close(rotate_mask_centroids(center(mask), torch.tensor([True]), k),
                                       center(torch.rot90(mask, k)))


if __name__ == '__main__':
    unittest.main()
