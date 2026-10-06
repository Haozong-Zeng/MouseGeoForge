"""CPU compatibility/gradient gates before the bounded surface-curve pilot."""
import unittest
import torch
from test_depth_pose import TinyEncoder
from vggt_omega.models.heads.dense_head import DenseHead
from vggt_omega.models.heads.instance_pose_head import MultiInstanceMouseHead
from mousegeoforge.models import DepthPoseModel
from mousegeoforge.models.surface_curve_pose import SurfaceCurveDepthPoseModel, SurfaceCurveHead
from mousegeoforge.surface_curve_instance_loss import surface_instance_loss
from training.instance_pose_losses import InstancePoseLossWeights

def make(cls=SurfaceCurveDepthPoseModel):
    torch.manual_seed(42)
    return cls(TinyEncoder(), DenseHead(dim_in=32,features=16,out_channels=[16]*4),
        MultiInstanceMouseHead(num_keypoints=23,num_queries=3,dim_in=32,hidden_dim=16,num_heads=4),bottleneck=8).eval()

class TestSurfaceModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_off_exact_and_on_preserves_existing_outputs(self):
        base,model=make(DepthPoseModel),make()
        missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
        self.assertTrue(missing and all(k.startswith('surface_curve_head.') for k in missing));self.assertFalse(unexpected)
        images=torch.rand(1,2,3,64,64)
        with torch.no_grad():
            expected=base(images);off=model(images);model.surface_curve_enabled=True;on=model(images)
        self.assertEqual(set(expected),set(off))
        for k in expected:
            torch.testing.assert_close(expected[k],off[k],atol=0,rtol=0)
            torch.testing.assert_close(off[k],on[k],atol=0,rtol=0)
        self.assertEqual(on['instance_surface_curve_xy'].shape,(1,2,3,33,2))
        self.assertEqual(len(model.instance_pose_head.instance_decoder._forward_hooks),0)

    def test_curve_gradient_box_detach_and_query_permutation(self):
        head=SurfaceCurveHead(16);tokens=torch.rand(1,2,3,16,requires_grad=True)
        boxes=torch.rand(1,2,3,4,requires_grad=True)
        control,curve=head(tokens,boxes);perm=[2,0,1]
        torch.testing.assert_close(head(tokens[:,:,perm],boxes[:,:,perm])[1],curve[:,:,perm])
        curve.square().mean().backward()
        self.assertIsNone(boxes.grad)
        self.assertGreater(head.net[-1].weight.grad.abs().sum(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_enabled_roundtrip_and_hook_failure_cleanup(self):
        a,b=make(),make();a.surface_curve_enabled=b.surface_curve_enabled=True
        b.load_state_dict(a.state_dict(),strict=True);images=torch.rand(1,1,3,64,64)
        with torch.no_grad():
            aa,bb=a(images),b(images)
        for k in aa:torch.testing.assert_close(aa[k],bb[k],atol=0,rtol=0)
        with self.assertRaises(ValueError):a._decode(images,[None]*24,2)
        self.assertEqual(len(a.instance_pose_head.instance_decoder._forward_hooks),0)

    def test_loss_replacement_common_matching_and_missing_visibility(self):
        for instances in (1,2):
            model=make();model.surface_curve_enabled=True
            with torch.no_grad():out=model(torch.rand(1,2,3,64,64))
            out={k:v.detach().clone() for k,v in out.items()}
            out['instance_keypoint_xy'].requires_grad_(True)
            target=dict(instance_valid=torch.ones(1,2,instances,dtype=torch.bool),
                instance_ids=torch.arange(instances)[None,None].expand(1,2,-1),
                boxes_cxcywh=out['instance_boxes_cxcywh'][:,:,:instances].clone(),
                keypoint_xy=out['instance_keypoint_xy'][:,:,:instances].detach().clone(),
                keypoint_valid=torch.ones(1,2,instances,23,dtype=torch.bool),
                keypoint_visibility_valid=torch.zeros(1,2,instances,23,dtype=torch.bool),
                detection_annotation_complete=torch.ones(1,2,dtype=torch.bool))
            target['keypoint_xy'][...,5:7,0]+=.08
            target['keypoint_valid'][...,22]=False;target['keypoint_xy'][...,22,:]=float('nan')
            weights=InstancePoseLossWeights()
            tang=torch.zeros_like(target['keypoint_xy']);tang[...,0]=1
            tv=torch.ones_like(target['keypoint_valid'])
            results=[];gradients=[]
            for mode in ('strict_points','weak_points','weak_points_curve6'):
                loss,logs,matches=surface_instance_loss(out,target,weights,image_size=(64,64),mode=mode,is_real=True,tangent=tang,tangent_valid=tv)
                self.assertTrue(torch.isfinite(loss));self.assertEqual(float(logs['loss_instance_keypoint_visibility']),0)
                g=torch.autograd.grad(loss,out['instance_keypoint_xy'],retain_graph=True)[0]
                self.assertTrue(torch.isfinite(g).all());gradients.append(g)
                results.append([[(q.tolist(),m.tolist()) for q,m in vs] for vs in matches])
            self.assertEqual(results[0],results[1]);self.assertEqual(results[1],results[2])
            self.assertLess(gradients[1][...,5:7,:].abs().sum(),gradients[0][...,5:7,:].abs().sum())
            torch.testing.assert_close(gradients[1],gradients[2])

if __name__=='__main__':unittest.main()
