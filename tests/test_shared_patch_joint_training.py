import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock
import importlib.util
import json
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vggt-omega'))
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from mousegeoforge.models.shared_patch_two_heads import SharedPatchTwoHeads
from mousegeoforge.joint_training import joint_train_step, update_ema, save_joint_checkpoint, load_joint_checkpoint


class DepthHead(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(32,2,1)
    def forward(self,taps,rgb):return self.proj(sum(taps)/len(taps))


class KeypointHead(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(32,5,1)
    def forward(self,feat,rgb):return self.proj(feat)


def model():
    enc=DinoVisionTransformer(img_size=32,patch_size=16,embed_dim=32,depth=24,num_heads=2,ffn_ratio=1,
        pos_embed_rope_normalize_coords='max',pos_embed_rope_dtype='fp32',layerscale_init=.1,
        norm_layer='layernormbf16',n_storage_tokens=4,mask_k_bias=True)
    enc.init_weights()
    return SharedPatchTwoHeads(enc,DepthHead(),KeypointHead(),mean=[.485,.456,.406],std=[.229,.224,.225])


def batch():return dict(images=torch.rand(2,3,32,32),split='train')
def depth_loss(out,b):return (out-1).square().mean()
def kp_loss(out,b):return (out+1).square().mean()
IDENTITY={k:'a'*64 for k in ['code_sha256','data_sha256','release_sha256','heads_sha256','recipe_sha256']}


class JointTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(29);torch.set_num_threads(1)

    def test_one_encoder_traversal_and_both_task_gradients(self):
        m=model();calls=[]
        hook=m.encoder.blocks[0].register_forward_hook(lambda *args:calls.append(1))
        out=m(batch()['images']);self.assertEqual(len(calls),1);hook.remove()
        d=torch.autograd.grad(out['depth'].square().mean(),m.encoder.blocks[0].mlp.fc1.weight,retain_graph=True)[0]
        k=torch.autograd.grad(out['keypoints'].square().mean(),m.encoder.blocks[0].mlp.fc1.weight)[0]
        self.assertGreater(float(d.abs().sum()),0);self.assertGreater(float(k.abs().sum()),0)

    def test_task_routing_and_encoder_eval_has_live_autograd(self):
        m=model().train();self.assertFalse(m.encoder.training)
        d=m(batch()['images'],task='depth');self.assertEqual(set(d),{'depth'})
        d['depth'].sum().backward();self.assertIsNone(m.keypoint_head.proj.weight.grad)
        self.assertIsNotNone(m.encoder.blocks[23].mlp.fc1.weight.grad)
        with self.assertRaises(ValueError):m(torch.rand(1,3,31,32))

    def test_accumulated_update_changes_encoder_and_both_heads(self):
        m=model();before=copy.deepcopy(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
        result=joint_train_step(m,opt,batch(),batch(),depth_loss,kp_loss)
        self.assertEqual(result['optimizer_updates'],1)
        for key in ['encoder.blocks.0.mlp.fc1.weight','depth_head.proj.weight','keypoint_head.proj.weight']:
            self.assertFalse(torch.equal(before[key],m.state_dict()[key]),key)

    def test_rejects_dev_before_model_forward(self):
        m=model();opt=torch.optim.AdamW(m.parameters());calls=[]
        hook=m.encoder.blocks[0].register_forward_hook(lambda *args:calls.append(1))
        b=batch();b['split']='dev'
        with self.assertRaises(ValueError):joint_train_step(m,opt,batch(),b,depth_loss,kp_loss)
        self.assertEqual(calls,[]);hook.remove()

    def test_nonfinite_second_task_does_not_apply_first_task_update(self):
        m=model();before=copy.deepcopy(m.state_dict());opt=torch.optim.AdamW(m.parameters())
        with self.assertRaises(ValueError):
            joint_train_step(m,opt,batch(),batch(),depth_loss,lambda out,b:out.sum()*float('nan'))
        for key,value in before.items():self.assertTrue(torch.equal(value,m.state_dict()[key]),key)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_full_resume_reproduces_updates_ema_rng_and_sampler(self):
        m=model();opt=torch.optim.AdamW(m.parameters(),lr=1e-3);ema=copy.deepcopy(m)
        joint_train_step(m,opt,batch(),batch(),depth_loss,kp_loss);update_ema(ema,m)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            save_joint_checkpoint(path,m,opt,ema,step=1,samplers={'depth':17,'kp':29},identity=IDENTITY)
            py=random.random();npval=np.random.random();b1,b2=batch(),batch()
            joint_train_step(m,opt,b1,b2,depth_loss,kp_loss);update_ema(ema,m)
            restored=model();ropt=torch.optim.AdamW(restored.parameters(),lr=9);rema=copy.deepcopy(restored)
            step,samplers=load_joint_checkpoint(path,restored,ropt,rema,identity=IDENTITY)
            self.assertEqual((step,samplers),(1,{'depth':17,'kp':29}))
            self.assertEqual(py,random.random());self.assertEqual(npval,np.random.random())
            rb1,rb2=batch(),batch();self.assertTrue(torch.equal(b1['images'],rb1['images']))
            joint_train_step(restored,ropt,rb1,rb2,depth_loss,kp_loss);update_ema(rema,restored)
            for key,value in m.state_dict().items():self.assertTrue(torch.equal(value,restored.state_dict()[key]),key)
            for key,value in ema.state_dict().items():self.assertTrue(torch.equal(value,rema.state_dict()[key]),key)
            wrong={**IDENTITY,'recipe_sha256':'b'*64}
            with self.assertRaises(ValueError):load_joint_checkpoint(path,restored,ropt,rema,identity=wrong)

    def test_cli_binding_parent_parity_and_discarded_update(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('joint_preflight',root/'tools/preflight_part_balanced_joint_20261028.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        m=model();parent=copy.deepcopy(m).eval();db,kb=batch(),batch()
        def reference_depth(rgb):
            return parent.depth_head(parent.encoder.get_intermediate_layers(
                (rgb-parent.image_mean)/parent.image_std,n=(5,11,17,23),reshape=True,norm=True),rgb)
        def reference_kp(rgb):
            # Independent final-feature parent path, not the new two-head wrapper.
            x=parent.encoder((rgb-parent.image_mean)/parent.image_std)['x_norm_patchtokens']
            return parent.keypoint_head(x.transpose(1,2).reshape(len(rgb),32,2,2),rgb)
        bundle=dict(model=m,optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3),depth_batch=db,keypoint_batch=kb,
            depth_loss=depth_loss,keypoint_loss=kp_loss,reference_depth=reference_depth,
            reference_keypoints=reference_kp,identity=IDENTITY)
        original=cli.importlib.import_module
        adapter=SimpleNamespace(build=lambda *args:bundle)
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'preflight.json'
            argv=['preflight','--factory','fixture:build','--output',str(output)]
            with mock.patch.object(sys,'argv',argv),mock.patch.object(cli.importlib,'import_module',
                    side_effect=lambda name:adapter if name=='fixture' else original(name)):
                cli.main()
            result=json.loads(output.read_text())
            self.assertTrue(result['parent_output_parity'])
            self.assertFalse(result['saved_weights']);self.assertFalse(result['formal_training_started'])


if __name__=='__main__':unittest.main()
