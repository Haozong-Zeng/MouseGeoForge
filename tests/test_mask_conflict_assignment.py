import unittest
import torch
from mousegeoforge.mask_conflict_assignment import repair_mask_conflicts


class TestMaskConflict(unittest.TestCase):
    def setup_masks(self):
        gt=torch.zeros(1,1,2,8,12,dtype=torch.bool);gt[0,0,0,1:4,1:4]=True;gt[0,0,1,4:7,8:11]=True
        pred=torch.where(gt.flip(2),torch.tensor(10.),torch.tensor(-10.));valid=torch.ones(1,1,2,dtype=torch.bool)
        return pred,gt,valid

    def test_swap_fixed_and_original_unchanged(self):
        p,g,v=self.setup_masks();old=[([0,1],[0,1])]
        fixed,rows=repair_mask_conflicts(p,g,v,v,old)
        self.assertEqual(fixed,[([1,0],[0,1])]);self.assertEqual(old,[([0,1],[0,1])]);self.assertTrue(rows[0]['overridden'])

    def test_correct_or_weak_stays_exact(self):
        p,g,v=self.setup_masks();old=[([1,0],[0,1])]
        self.assertEqual(repair_mask_conflicts(p,g,v,v,old)[0],old)
        p.fill_(-10);self.assertEqual(repair_mask_conflicts(p,g,v,v,old)[0],old)

    def test_missing_mask_and_duplicate_mapping_rejected(self):
        p,g,v=self.setup_masks()
        with self.assertRaises(ValueError):repair_mask_conflicts(p,g,v,torch.zeros_like(v),[([0,1],[0,1])])
        with self.assertRaises(ValueError):repair_mask_conflicts(p,g,v,v,[([0,0],[0,1])])

    def test_multiple_views_and_padded_target_ids(self):
        p,g,v=self.setup_masks();gt=torch.zeros(1,2,4,8,12,dtype=torch.bool)
        gt[:,:,1]=g[0,0,0];gt[:,:,3]=g[0,0,1]
        valid=torch.zeros(1,2,4,dtype=torch.bool);valid[:,:,1]=True;valid[:,:,3]=True
        pred=torch.cat([p,p.flip(2)],1);old=[([0,1],[1,3]),([0,1],[1,3])]
        fixed,rows=repair_mask_conflicts(pred,gt,valid,valid,old)
        self.assertEqual(fixed,[([1,0],[1,3]),([0,1],[1,3])])
        self.assertEqual([r['overridden'] for r in rows],[True,False])


if __name__=='__main__':unittest.main()
