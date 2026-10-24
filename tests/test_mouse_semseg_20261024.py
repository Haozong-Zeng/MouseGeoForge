"""Engineering invariants, not a score test for a learned mouse model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mousegeoforge.models.mouse_semseg import MouseMaskDecoder, hungarian_loss, semantic_scores
from mousegeoforge.semseg_training import (BalancedSchedule, EMA, MouseSemseg, configure_stage,
    letterbox, load_checkpoint, load_items, make_encoder, mask_metrics, native_mask,
    optimizer_for, rng_state, save_checkpoint, seed_all, set_learning_rates)


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def decoder(self):
        torch.manual_seed(8)
        return MouseMaskDecoder(8, dim=16, queries=4, heads=4, pixel_layers=1, query_layers=1).eval()

    def test_decoder_loss_is_query_permutation_invariant(self):
        model = self.decoder()
        features = [torch.randn(2, 8, 4, 4) for _ in range(4)]
        outputs = model(features)
        target = torch.zeros(2, 64, 64, dtype=torch.long); target[:, 20:40, 10:30] = 1
        loss = hungarian_loss(outputs, target)
        permutation = torch.tensor([2, 0, 3, 1])
        shuffled = [{k: v[:, permutation] for k, v in p.items()} for p in outputs]
        torch.testing.assert_close(loss, hungarian_loss(shuffled, target))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for layer in model.query_layers:
            for attention in [layer.cross, layer.self_attention]:
                dim = attention.embed_dim
                self.assertEqual(float(attention.in_proj_bias.grad[dim:2*dim].abs().sum()), 0)
        self.assertEqual(semantic_scores(outputs[-1], (64, 64)).shape, (2, 2, 64, 64))

    def test_unknown_pixels_do_not_become_background(self):
        outputs = [dict(classes=torch.randn(1, 4, 3, requires_grad=True),
                        masks=torch.randn(1, 4, 8, 8, requires_grad=True))]
        unknown = torch.full((1, 8, 8), 255, dtype=torch.long)
        loss = hungarian_loss(outputs, unknown)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(outputs[0]['masks'].grad.abs().sum()), 0)
        background = torch.zeros_like(unknown)
        self.assertGreater(float(hungarian_loss(outputs, background).detach()), 0)

    def test_dense_loss_gradient_accumulation_matches_batch(self):
        one, two = self.decoder(), self.decoder()
        features = [torch.randn(2, 8, 4, 4) for _ in range(4)]
        labels = torch.zeros(2, 64, 64, dtype=torch.long); labels[:, 12:37, 14:50] = 1
        hungarian_loss(one(features), labels).backward()
        for i in range(2):
            (hungarian_loss(two([x[i:i+1] for x in features]), labels[i:i+1]) / 2).backward()
        for a, b in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=1e-4)

    def test_letterbox_padding_and_inverse(self):
        image = np.zeros((32, 64, 3), np.uint8)
        mask = np.zeros((32, 64), np.uint8); mask[4:23, 7:40] = 1; mask[5:9, 45:48] = 255
        _, target, meta = letterbox(image, mask, 64)
        np.testing.assert_array_equal(native_mask(target.numpy(), meta), mask)
        self.assertEqual(int(target[:16].sum()), 0)
        self.assertEqual(int(target[-16:].sum()), 0)
        self.assertEqual(mask_metrics(mask == 1, mask)['iou'], 1)

    def test_global_schedule_shards_preserve_strata_and_resume(self):
        rows = [dict(id=str(i), domain=d, bottom=b, split='train', group=str(i))
                for i, (d, b) in enumerate([('real', True), ('real', False), ('synthetic', True), ('synthetic', False)])]
        a, b = BalancedSchedule(rows, 8, 17), BalancedSchedule(rows, 8, 17)
        ids = a.at(25)
        self.assertEqual(ids, b.at(25))
        self.assertEqual(sorted(ids), sorted(ids[0::2] + ids[1::2]))
        self.assertEqual([ids.count(i) for i in range(4)], [2] * 4)

    def test_manifest_excludes_test_sources_before_image_access(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'items.json'
            p.write_text(json.dumps([dict(id='x', split='train', group='WalkAround', rig=None,
                         image='missing.png', label='missing_label.png')]))
            with self.assertRaisesRegex(ValueError, 'Excluded source'):
                load_items(p, Path(directory), verify=False)

    def test_multilayer_api_last_output_matches_regular_forward(self):
        seed_all(5)
        enc = make_encoder(fixture=True).eval()
        x = torch.randn(1, 3, 64, 64)
        layers = enc.get_intermediate_layers(x, n=[20, 21, 22, 23], norm=True, reshape=True)
        direct = enc(x)['x_norm_patchtokens'].transpose(1, 2).reshape(1, 32, 4, 4)
        torch.testing.assert_close(layers[-1], direct)

    def test_stage_boundary_resume_includes_new_parameters(self):
        def create():
            seed_all(17)
            m = MouseSemseg(make_encoder(fixture=True), dim=16, queries=4, heads=4, pixel_layers=1, query_layers=1)
            configure_stage(m, 'U', 0, 1)
            return m, optimizer_for(m, 'U', 3e-5), EMA()
        def update(m, optimizer, ema, step):
            configure_stage(m, 'U', step, 1)
            set_learning_rates(optimizer, step, 4, 2, .1, 1)
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(1, 3, 64, 64)
            y = torch.zeros(1, 64, 64, dtype=torch.long); y[:, 16:35, 18:40] = 1
            loss = hungarian_loss(m(x), y); loss.backward(); optimizer.step(); ema.update(m)
            return loss.detach()
        m, opt, ema = create(); ema.update(m)
        frozen = m.encoder.blocks[0].attn.qkv.weight.detach().clone()
        update(m, opt, ema, 0); update(m, opt, ema, 1)
        protocol = dict(fixture=True, world_size=1)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'checkpoint.pt'
            save_checkpoint(p, m, opt, ema, 2, protocol, [rng_state()], [])
            expected = update(m, opt, ema, 2)
            resumed, opt2, ema2 = create()
            step, _ = load_checkpoint(p, resumed, opt2, ema2, protocol)
            self.assertEqual(step, 2)
            actual = update(resumed, opt2, ema2, step)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for a, b in zip(m.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertTrue(resumed.encoder.blocks[20].attn.qkv.weight.requires_grad)
            torch.testing.assert_close(m.encoder.blocks[0].attn.qkv.weight, frozen, atol=0, rtol=0)
            raw = m.head.classes.weight.detach().clone()
            with ema.apply(m):
                pass
            torch.testing.assert_close(m.head.classes.weight, raw, atol=0, rtol=0)
            other_world = dict(protocol, world_size=2)
            with self.assertRaisesRegex(ValueError, 'Resume protocol mismatch'):
                load_checkpoint(p, resumed, opt2, ema2, other_world)
            # Inference alone may use a different world size; not exact continuation.
            load_checkpoint(p, resumed, opt2, ema2, other_world, inference=True)


if __name__ == '__main__':
    unittest.main()
