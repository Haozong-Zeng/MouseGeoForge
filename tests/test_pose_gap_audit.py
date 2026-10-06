import unittest
from mousegeoforge.pose_gap_audit import build_training_cohort,GROUPS


class PoseGapTests(unittest.TestCase):
    def spec(self):
        records={};evaluation=[]
        for g,v in zip(GROUPS,([2,0],[6],[2,6],[2,0,6],[2,0,6,7],[2,0])):
            records[g]=dict(scene='dev',views=v,physical_roles=['bottom' if x==6 else 'top' if x==7 else 'side' for x in v],image_hw=[768,768],split='val')
            evaluation.append(dict(key=g,group=g))
        for kind in ('cylinder','dual'):
            records[kind]=dict(records[GROUPS[0]],scene=kind,split='train',normalization_scale=7,source_revision='fixed')
        return dict(records=records,evaluation=evaluation,slots=[dict(kind='cylinder'),dict(kind='dual')],arms={'combination_coverage':[['cylinder'],['dual']]})

    def test_same_camera_conditions_preserve_scale(self):
        rows=build_training_cohort(self.spec());self.assertEqual(len(rows),6)
        self.assertTrue(all(r['record']['normalization_scale']==7 for r in rows))
        self.assertEqual(rows[0]['exact_order_visits'],1);self.assertEqual(rows[1]['exact_order_visits'],0)

    def test_no_validation_pose_in_training_cohort(self):
        spec=self.spec();spec['records']['cylinder']['split']='val'
        with self.assertRaises(ValueError):build_training_cohort(spec)

    def test_shape_mismatch_rejected(self):
        spec=self.spec();spec['records']['dual']['image_hw']=[432,768]
        with self.assertRaises(ValueError):build_training_cohort(spec)


if __name__=='__main__':unittest.main()
