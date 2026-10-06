"""Negative-contract tests use disposable fixtures, never real candidate files."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from mousegeoforge.mask_box_candidate_safe import MaskBoxCandidate
from mousegeoforge.mask_box_candidate import file_sha256,tensor_mapping_sha256
from mousegeoforge.mask_reference_box import MaskReferenceBoxRefiner
from test_mask_box_candidate import FakeBase


class CandidateBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model=FakeBase().eval();state=self.model.state_dict()
        self.c=MaskBoxCandidate(MaskReferenceBoxRefiner(),
            base_state_sha256=tensor_mapping_sha256({k:v for k,v in state.items() if not k.startswith('aggregator.')}),
            encoder_state_sha256=tensor_mapping_sha256({k:v for k,v in state.items() if k.startswith('aggregator.')}))
    def prediction(self):
        return dict(instance_boxes_cxcywh=torch.full((1,1,6,4),.2),instance_mask_logits=torch.zeros(1,1,6,8,10))
    def forward(self,token=None):self.model.instance_pose_head.instance_decoder(torch.ones(1,6,256) if token is None else token)
    def reusable(self):
        self.assertFalse(self.c._active);self.assertEqual(len(self.model.instance_pose_head.instance_decoder._forward_hooks),0)
        with self.c.capture(self.model,enabled=True):
            self.forward();self.c(self.prediction(),(32,40),enabled=True)
    def test_wrong_encoder_rejected_before_hook(self):
        before=self.model.aggregator.weight.detach().clone()
        with torch.no_grad():self.model.aggregator.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'encoder'):
            with self.c.capture(self.model,enabled=True):pass
        with torch.no_grad():self.model.aggregator.weight.copy_(before)
        self.reusable()
    def test_missing_and_multiple_forwards(self):
        for count in (0,2):
            with self.subTest(count=count),self.assertRaises(RuntimeError):
                with self.c.capture(self.model,enabled=True):
                    for _ in range(count):self.forward()
                    self.c(self.prediction(),(32,40),enabled=True)
            self.reusable()
    def test_nonfinite_mask_token_and_boxes(self):
        for field in ('mask','token','boxes'):
            p=self.prediction();token=torch.ones(1,6,256)
            if field=='token':token[0,0,0]=float('nan')
            else:p['instance_mask_logits' if field=='mask' else 'instance_boxes_cxcywh'].flatten()[0]=float('inf')
            with self.subTest(field=field),self.assertRaises(ValueError):
                with self.c.capture(self.model,enabled=True):self.forward(token);self.c(p,(32,40),enabled=True)
            self.reusable()
    def test_bad_box_dtype_and_hw(self):
        for dtype,hw in ((torch.float16,(32,40)),(torch.int64,(32,40)),(torch.float32,(0,40)),(torch.float32,(32.5,40)),(torch.float32,('32',40)),(torch.float32,None)):
            p=self.prediction();p['instance_boxes_cxcywh']=p['instance_boxes_cxcywh'].to(dtype)
            with self.subTest(dtype=dtype,hw=hw),self.assertRaises(ValueError):
                with self.c.capture(self.model,enabled=True):self.forward();self.c(p,hw,enabled=True)
            self.reusable()
    def test_disabled_malformed_inputs_remain_untouched(self):
        p={'instance_boxes_cxcywh':torch.tensor(float('nan'))}
        with self.c.capture():self.assertIs(self.c(p,('bad',None)),p)
        self.assertFalse(self.c._active)
    def test_corrupt_files_and_source_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='mask-candidate-contract-') as folder:
            root=Path(folder);bundle=root/'bundle';bundle.mkdir()
            base=root/'base.bin';enc=root/'encoder.bin';source=root/'runtime-marker.txt'
            base.write_bytes(b'base fixture');enc.write_bytes(b'encoder fixture');source.write_bytes(b'source fixture')
            state=MaskReferenceBoxRefiner().state_dict();weights=bundle/'refiner.pt';torch.save(state,weights)
            clean_weights=weights.read_bytes();source_bytes=source.read_bytes()
            manifest=dict(schema_version=1,default_enabled=False,base_checkpoint_sha256=file_sha256(base),encoder_checkpoint_sha256=file_sha256(enc),
                source_sha256={'runtime-marker.txt':file_sha256(source)},weights='refiner.pt',weights_sha256=file_sha256(weights),refiner_state_sha256=tensor_mapping_sha256(state),
                base_state_sha256=self.c._base_hash,encoder_state_sha256=self.c._encoder_hash)
            path=bundle/'manifest.json';path.write_text(json.dumps(manifest))
            def load():return MaskBoxCandidate.load(bundle,base_checkpoint=base,encoder_checkpoint=enc,repository_root=root)
            load()
            for failure in ('weights','source','encoder','wrong_state_hash','path_escape'):
                altered=dict(manifest)
                if failure=='weights':weights.write_bytes(b'corrupted fixture')
                if failure=='source':source.write_bytes(b'changed fixture')
                if failure=='encoder':enc.write_bytes(b'wrong encoder')
                if failure=='wrong_state_hash':altered['refiner_state_sha256']='0'*64
                if failure=='path_escape':altered['weights']='../base.bin'
                path.write_text(json.dumps(altered));rng=torch.random.get_rng_state().clone()
                with self.subTest(failure=failure),self.assertRaises(ValueError):load()
                self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
                weights.write_bytes(clean_weights);source.write_bytes(source_bytes);enc.write_bytes(b'encoder fixture');path.write_text(json.dumps(manifest))
                load()


if __name__=='__main__':unittest.main()
