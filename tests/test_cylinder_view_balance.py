import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from train_cylinder_view_balance import cylinder_order


class CylinderViewBalanceTests(unittest.TestCase):
    roles = {i:'side' for i in range(6)} | {6:'bottom',7:'top'}

    def test_equal_counts_unique_cameras(self):
        for variant in range(16):
            for n in range(1,5):
                for balanced in (False,True):
                    selected = cylinder_order(self.roles,n,variant,balanced)
                    self.assertEqual(len(selected),n)
                    self.assertEqual(len(set(selected)),n)
                    if not balanced:
                        self.assertTrue(all(self.roles[k]=='side' for k in selected))

    def test_three_and_four_views_cover_all_roles(self):
        for n in (3,4):
            for variant in range(16):
                selected = cylinder_order(self.roles,n,variant,True)
                self.assertEqual({self.roles[k] for k in selected},{'top','side','bottom'})

    def test_camera_ids_not_roles(self):
        mapping = {i: 100+i for i in self.roles}
        roles = {mapping[i]:role for i,role in self.roles.items()}
        for n in range(1,5):
            expected = [mapping[k] for k in cylinder_order(self.roles,n,3,True)]
            self.assertEqual(cylinder_order(roles,n,3,True),expected)

    def test_single_view_top_bottom_balanced_across_poses(self):
        roles = [self.roles[cylinder_order(self.roles,1,i,True)[0]] for i in range(16)]
        self.assertEqual(roles.count('top'),8)
        self.assertEqual(roles.count('bottom'),8)

    def test_missing_bottom_rejected(self):
        with self.assertRaises(ValueError):
            cylinder_order({i:'side' for i in range(8)},2,0,True)


if __name__ == '__main__': unittest.main()
