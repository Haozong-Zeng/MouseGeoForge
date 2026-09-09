import unittest
import torch
from test_depth_pose import tiny_model
from training.instance_pose_losses import instance_pose_loss, instance_targets_from_batch
from mousegeoforge.pilot_metrics import evaluate_scene, summarize_records


class TestJointPilot(unittest.TestCase):
    def metric_fixture(self):
        boxes=torch.tensor([[[[.25,.5,.2,.4],[.75,.5,.2,.4]]]])
        xy=torch.zeros(1,1,2,23,2)
        xy[:,:,:, :,0]=torch.tensor([.25,.75])[None,None,:,None]
        xy[...,1]=.5
        target=dict(instance_valid=torch.ones(1,1,2,dtype=torch.bool),boxes_cxcywh=boxes,
                    keypoint_xy=xy,keypoint_valid=torch.ones(1,1,2,23,dtype=torch.bool),
                    keypoint_visible=torch.ones(1,1,2,23,dtype=torch.bool),
                    masks=torch.ones(1,1,2,16,32,dtype=torch.bool),mask_valid=torch.ones(1,1,2,dtype=torch.bool))
        batch=dict(images=torch.zeros(1,1,3,16,32),depths=torch.ones(1,1,16,32),
                   depth_masks=torch.ones(1,1,16,32,dtype=torch.bool),
                   depth_available=torch.ones(1,1,dtype=torch.bool),body_masks=torch.ones(1,1,16,32,dtype=torch.bool))
        out=dict(instance_boxes_cxcywh=boxes.clone(),instance_objectness_logits=torch.ones(1,1,2)*10,
                 instance_keypoint_xy=xy.clone(),instance_mask_logits=torch.ones(1,1,2,4,8)*10,
                 depth=torch.ones(1,1,16,32,1))
        return out,batch,target

    def test_metrics_perfect_and_query_permutation(self):
        out,batch,target=self.metric_fixture()
        a,_=evaluate_scene(out,batch,target)
        for name in ('box_matched_nme','body_depth_absrel'): self.assertEqual(a[name],0.)
        for name in ('detection_aware_pck_0.1','matched_mask_iou','detection_recall'): self.assertEqual(a[name],1.)
        rearranged={k:v[:,:,[1,0]] if k.startswith('instance_') else v for k,v in out.items()}
        b,_=evaluate_scene(rearranged,batch,target)
        self.assertEqual(a,b)

    def test_metrics_count_misses_and_do_not_match_using_keypoints(self):
        out,batch,target=self.metric_fixture()
        _,assignment=evaluate_scene(out,batch,target)
        out['instance_keypoint_xy']=out['instance_keypoint_xy'][:,:,[1,0]]
        m,new_assignment=evaluate_scene(out,batch,target)
        self.assertEqual(assignment,new_assignment)
        self.assertGreater(m['box_matched_nme'],0.5)
        out,batch,target=self.metric_fixture()
        out['instance_objectness_logits'].fill_(-10)
        m,_=evaluate_scene(out,batch,target)
        self.assertEqual(m['box_matched_nme'],0.)
        self.assertEqual(m['detection_aware_pck_0.1'],0.)
        self.assertEqual(m['detection_fn'],2)

    def test_metric_summary_weights_sources_equally(self):
        records=[dict(source_group='a',metrics={'error':1.}) for _ in range(9)]
        records.append(dict(source_group='b',metrics={'error':3.}))
        self.assertEqual(summarize_records(records)['source_macro']['error'],2.)

    def test_cached_forward_exact(self):
        m = tiny_model().eval()
        images = torch.rand(1, 2, 3, 64, 64)
        with torch.no_grad():
            cached, start = m.aggregator(images)
            for mode in ('off', 'local_depth_features'):
                a = m(images, coupling=mode)
                b = m.forward_cached(images, cached, start, coupling=mode)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            m.forward_cached(images[:, :1], cached, start)

    def test_unfrozen_normal_forward_and_cache_rejection(self):
        m = tiny_model()
        m.freeze_encoder = False
        m.aggregator.requires_grad_(True)
        images = torch.rand(1, 2, 3, 64, 64)
        m(images)['depth'].mean().backward()
        self.assertIsNotNone(m.aggregator.proj.weight.grad)
        with self.assertRaises(ValueError):
            m.forward_cached(images, *m.aggregator(images))

    def test_incomplete_detection_no_false_negatives(self):
        m = tiny_model()
        prediction = m(torch.rand(1, 2, 3, 64, 64))
        prediction['instance_objectness_logits'].retain_grad()
        target = dict(instance_valid=torch.zeros(1, 2, 1, dtype=torch.bool),
                      instance_ids=torch.zeros(1, 2, 1, dtype=torch.long),
                      boxes_cxcywh=torch.zeros(1, 2, 1, 4),
                      keypoint_xy=torch.zeros(1, 2, 1, 5, 2),
                      keypoint_valid=torch.zeros(1, 2, 1, 5, dtype=torch.bool),
                      keypoint_visible=torch.zeros(1, 2, 1, 5, dtype=torch.bool),
                      detection_annotation_complete=torch.tensor([[False, True]]))
        resolved = instance_targets_from_batch(target)
        loss, _ = instance_pose_loss(prediction, resolved)
        loss.backward()
        grad = prediction['instance_objectness_logits'].grad
        self.assertEqual(float(grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(grad[0, 1].abs().sum()), 0)

    def test_incomplete_detection_positive_still_supervised(self):
        m = tiny_model()
        prediction = m(torch.rand(1, 1, 3, 64, 64))
        prediction['instance_objectness_logits'].retain_grad()
        target = dict(instance_valid=torch.ones(1, 1, 1, dtype=torch.bool),
                      instance_ids=torch.zeros(1, 1, 1, dtype=torch.long),
                      boxes_cxcywh=torch.tensor([[[[.5, .5, .2, .2]]]]),
                      keypoint_xy=torch.zeros(1, 1, 1, 5, 2),
                      keypoint_valid=torch.zeros(1, 1, 1, 5, dtype=torch.bool),
                      keypoint_visible=torch.zeros(1, 1, 1, 5, dtype=torch.bool),
                      detection_annotation_complete=torch.zeros(1, 1, dtype=torch.bool))
        loss, _ = instance_pose_loss(prediction, target)
        loss.backward()
        grad = prediction['instance_objectness_logits'].grad.flatten()
        self.assertEqual(int((grad != 0).sum()), 1)
        self.assertLess(float(grad.min()), 0)
