import unittest
import json
from mousegeoforge.cylinder_combinations import sample_orders, coverage


class CylinderCombinationsTests(unittest.TestCase):
    def setUp(self):
        self.roles={**{k:'side' for k in range(6)},6:'bottom',7:'top'}
        self.slots=[(str(i%16),n) for i in range(100) for n in (1,2,3,4)]

    def test_coverage_and_reproducibility(self):
        orders=sample_orders(self.slots,self.roles)
        self.assertEqual(orders,sample_orders(self.slots,self.roles))
        result=coverage(self.slots,orders,self.roles)
        self.assertTrue(result['passed'],result)
        self.assertEqual(result,json.loads(json.dumps(result)))
        self.assertEqual(result['unordered_subset_counts'],{'1':2,'2':12,'3':36,'4':55})
        self.assertGreaterEqual(result['ordered_pairs']['2,6'],4)
        self.assertGreaterEqual(result['ordered_pairs']['6,2'],4)

    def test_pose_renaming_does_not_change_sampling(self):
        renamed=[('pose_'+p,n) for p,n in self.slots]
        self.assertEqual(sample_orders(self.slots,self.roles),sample_orders(renamed,self.roles))

    def test_insufficient_budget_fails_audit(self):
        slots=self.slots[:8]
        self.assertFalse(coverage(slots,sample_orders(slots,self.roles),self.roles)['passed'])

    def test_invalid_rig_and_views_rejected(self):
        with self.assertRaises(ValueError): sample_orders(self.slots,{0:'top'})
        with self.assertRaises(ValueError): sample_orders([('a',5)],self.roles)
        with self.assertRaises(ValueError): coverage([('a',2)],[[0,0]],self.roles)


if __name__=='__main__': unittest.main()
