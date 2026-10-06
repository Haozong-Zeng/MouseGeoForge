import unittest
from unittest.mock import patch
import torch
from mousegeoforge.mask_box_candidate import MaskBoxCandidate as Original
from mousegeoforge.mask_box_candidate_safe import MaskBoxCandidate


class SafeCandidateTests(unittest.TestCase):
    def test_same_inference_methods(self):
        self.assertIs(MaskBoxCandidate.__call__,Original.__call__)
        self.assertIs(MaskBoxCandidate.capture,Original.capture)
    def test_load_preserves_rng(self):
        before=torch.random.get_rng_state().clone();sentinel=object()
        def random_load(*args,**kwargs):torch.rand(100);return sentinel
        with patch.object(Original,'load',side_effect=random_load):self.assertIs(MaskBoxCandidate.load('x'),sentinel)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
    def test_failed_load_preserves_rng(self):
        before=torch.random.get_rng_state().clone()
        def random_failure(*args,**kwargs):torch.rand(100);raise ValueError('wrong pair')
        with patch.object(Original,'load',side_effect=random_failure):
            with self.assertRaises(ValueError):MaskBoxCandidate.load('x')
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))


if __name__=='__main__':unittest.main()
