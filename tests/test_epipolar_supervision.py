import copy
import unittest
from unittest.mock import patch
import torch
from mousegeoforge.epipolar_supervision import fundamental_from_calibration,signed_sampson_normalized,matched_epipolar_body_loss,epipolar_aux_pose_loss,upstream


class EpipolarSupervisionTests(unittest.TestCase):
    def sample(self,views=2,instances=1):
        xy=torch.full((1,views,instances,23,2),.4)
        for v in range(views):xy[0,v,...,0]-=.05*v
        xy.requires_grad_()
        k=torch.tensor([[100.,0,50],[0,100.,50],[0,0,1.]]).expand(1,views,3,3).clone()
        e=torch.eye(4).expand(1,views,4,4).clone()
        for v in range(views):e[0,v,0,3]=-v
        t=dict(intrinsics=k,extrinsics=e,camera_available=torch.ones(1,views,dtype=torch.bool),
            instance_ids=torch.arange(instances).expand(1,views,instances).clone(),keypoint_valid=torch.ones(1,views,instances,23,dtype=torch.bool))
        m=[[(torch.arange(instances),torch.arange(instances)) for v in range(views)]]
        return dict(instance_keypoint_xy=xy),t,m

    def loss(self,p,t,m):return matched_epipolar_body_loss(p,t,m,image_size=(100,100))

    def test_consistent_zero_and_v1_zero(self):
        for v in (1,2,3,4):
            p,t,m=self.sample(v);loss,counts=self.loss(p,t,m)
            self.assertLess(abs(float(loss.detach())),1e-12);loss.backward()
            self.assertTrue(torch.isfinite(p['instance_keypoint_xy'].grad).all())
            self.assertEqual(counts['epipolar_pairs'],v*(v-1)//2)

    def test_body_gradient_tail_zero_and_occlusion_not_filtered(self):
        p,t,m=self.sample();t['keypoint_visible']=torch.zeros_like(t['keypoint_valid'])
        with torch.no_grad():p['instance_keypoint_xy'][0,1,...,1]+=.2
        loss,counts=self.loss(p,t,m);loss.backward();g=p['instance_keypoint_xy'].grad
        self.assertGreater(float(g.abs().sum()),0);self.assertEqual(float(g[...,7:14,:].abs().sum()),0)
        self.assertEqual(counts['epipolar_points'],16)

    def test_missing_calibration_labels_identity(self):
        for field in ('camera_available','keypoint_valid','instance_ids'):
            p,t,m=self.sample();t[field][:]=-1 if field=='instance_ids' else False
            loss,c=self.loss(p,t,m);self.assertEqual(float(loss.detach()),0);self.assertEqual(c['epipolar_points'],0)

    def test_view_and_query_permutation_invariance(self):
        p,t,m=self.sample(4,2)
        with torch.no_grad():p['instance_keypoint_xy'][0,1,0,:,1]+=.2;p['instance_keypoint_xy'][0,3,1,:,1]+=.1
        expected,_=self.loss(p,t,m)
        order=[2,0,3,1];p2={'instance_keypoint_xy':p['instance_keypoint_xy'][:,order].flip(2)}
        t2={k:v[:,order] for k,v in t.items()};m2=[[(torch.tensor([1,0]),torch.tensor([0,1])) for _ in order]]
        actual,_=self.loss(p2,t2,m2);torch.testing.assert_close(actual,expected)

    def test_per_instance_balance(self):
        p,t,m=self.sample(2,2)
        with torch.no_grad():p['instance_keypoint_xy'][0,1,0,:,1]+=.1;p['instance_keypoint_xy'][0,1,1,:,1]+=.3
        t['keypoint_valid'][...,1,:]=False;t['keypoint_valid'][...,1,0]=True
        both,_=self.loss(p,t,m);single=[]
        for q in (0,1):
            one=[[(torch.tensor([q]),torch.tensor([q])) for _ in range(2)]];single.append(self.loss(p,t,one)[0])
        torch.testing.assert_close(both,torch.stack(single).mean())

    def test_view_specific_slot_order_uses_persistent_identity(self):
        p,t,m=self.sample(2,2)
        with torch.no_grad():
            p['instance_keypoint_xy'][:,:,1,:,1]=.7
            p['instance_keypoint_xy'][:,1]=p['instance_keypoint_xy'][:,1].flip(1)
        wrong,_=self.loss(p,t,m);self.assertGreater(float(wrong.detach()),0)
        t['instance_ids'][:,1]=t['instance_ids'][:,1].flip(1)
        correct,_=self.loss(p,t,m);self.assertLess(abs(float(correct.detach())),1e-12)

    def test_scale_and_resize_invariance(self):
        p,t,m=self.sample();xy=p['instance_keypoint_xy'][0]
        with torch.no_grad():xy[1,...,1]+=.2
        expected,_=self.loss(p,t,m)
        t['extrinsics'][...,:3,3]*=1000;t['intrinsics'][...,:2,:]*=2
        actual,_=matched_epipolar_body_loss(p,t,m,image_size=(200,200));torch.testing.assert_close(actual,expected)

    def test_coincident_cameras_and_duplicate_ids(self):
        p,t,m=self.sample(2,2);t['extrinsics'][0,1]=t['extrinsics'][0,0]
        loss,c=self.loss(p,t,m);self.assertEqual(float(loss.detach()),0);self.assertEqual(c['epipolar_degenerate_pairs'],2)
        t['instance_ids'][:]=0
        with self.assertRaises(ValueError):self.loss(p,t,m)

    def test_gradient_finite_difference(self):
        p,t,m=self.sample();xy=p['instance_keypoint_xy'][0].double();matrix=fundamental_from_calibration(t['intrinsics'][0,0],t['extrinsics'][0,0],t['intrinsics'][0,1],t['extrinsics'][0,1])
        a=xy[0,0,:2].detach().requires_grad_();b=(xy[1,0,:2].detach()+.03).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:signed_sampson_normalized(a,b,matrix,(100,100)),(a,b)))

    def test_zero_weight_shared_matching(self):
        p,t,m=self.sample();namespace=dict(hungarian_match_instances=None)
        exec('def stub(predictions,targets,weights,*,image_size=None):\n hungarian_match_instances(predictions,targets)\n return predictions["instance_keypoint_xy"].sum(), {}',namespace)
        with patch.object(upstream,'instance_pose_loss',namespace['stub']),patch.object(upstream,'hungarian_match_instances',return_value=m) as matcher:
            loss,logs=epipolar_aux_pose_loss(p,t,None,aux_weight=0,image_size=(100,100))
            self.assertEqual(matcher.call_count,1);torch.testing.assert_close(loss,p['instance_keypoint_xy'].sum(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
