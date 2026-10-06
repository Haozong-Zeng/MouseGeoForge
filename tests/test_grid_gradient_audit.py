import unittest
from mousegeoforge.grid_gradient_audit import point_gradient,cosine,select_records


class GridGradientTests(unittest.TestCase):
    def test_sign_and_decomposition(self):
        r=point_gradient([2,0],[-1,0],[1,0])
        self.assertTrue(r['harmful_reversal']);self.assertTrue(r['indirect_away'])
        self.assertEqual(r['indirect_norm'],3);self.assertEqual(r['full_direct_cos'],-1)

    def test_aligned_and_zero(self):
        r=point_gradient([-2,0],[-1,0],[1,0])
        self.assertFalse(r['harmful_reversal']);self.assertFalse(r['indirect_away'])
        self.assertIsNone(cosine([0,0],[1,0]))

    def test_invalid_vector(self):
        with self.assertRaises(ValueError):point_gradient([float('nan'),0],[1,0],[1,0])

    def test_selection_ignores_development_metrics(self):
        records={};slots=[];arm=[]
        for j in range(8):
            for order,roles in (([6],['bottom']),([7],['top']),([2,6],['side','bottom']),([2,0,6],['side','side','bottom']),([2,0,6,7],['side','side','bottom','top'])):
                key=str(len(records));records[key]=dict(scene=f'c{j}',split='train',expected_instances=1,views=order,physical_roles=roles)
                slots.append(dict(kind='cylinder'));arm.append([key])
        for j in range(8):
            key=str(len(records));records[key]=dict(scene=f'd{j}',split='train',expected_instances=2,views=[2,0],physical_roles=['side','top'])
            slots.append(dict(kind='dual'));arm.append([key])
        spec=dict(records=records,slots=slots,arms={'combination_coverage':arm})
        chosen=select_records(spec)
        self.assertEqual(len(chosen),24);self.assertEqual(len({r['scene'] for r in chosen}),8)


if __name__=='__main__':unittest.main()
