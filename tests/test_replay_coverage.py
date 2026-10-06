import json
import unittest
from mousegeoforge.replay_coverage import select_unreplayed, summarize_views


class ReplayCoverageTests(unittest.TestCase):
    def sample(self):
        return [dict(scene=f'scene{i}', split='train', expected_instances=2,
                     source_group='source', frame=i, source_frame=100-i) for i in range(20)]

    def test_source_frame_order_and_train_only(self):
        rows = self.sample()
        rows += [dict(rows[0], scene='validation', split='val'), dict(rows[0], scene='test', split='test')]
        selected = select_unreplayed(rows, {'scene0', 'scene19'}, 4)
        self.assertEqual([r['frame'] for r in selected], [18, 12, 7, 1])
        self.assertTrue(all(r['split'] == 'train' for r in selected))
        self.assertEqual(select_unreplayed(rows[::-1], {'scene19', 'scene0'}, 4), selected)

    def test_duplicate_missing_replay_and_size_rejected(self):
        rows = self.sample()
        with self.assertRaises(ValueError): select_unreplayed(rows + [rows[0]], {'scene0'}, 4)
        with self.assertRaises(ValueError): select_unreplayed(rows, {'absent'}, 4)
        with self.assertRaises(ValueError): select_unreplayed(rows, {'scene0'}, 20)

    def test_detection_gated_pck_and_serialization(self):
        rows = [dict(checkpoint='mixed', cohort='nonreplayed', role='top', found=found,
                     iou=.9 if found else .4, body_errors=[.05], tail_errors=[.2], core23_errors=[.05, .2])
                for found in (True, False)]
        result = summarize_views(rows)
        row = next(r for r in result if r['role'] == 'all')
        self.assertEqual(row['recall'], .5)
        self.assertEqual(row['body_detection_pck'], .5)
        self.assertEqual(row['core23_detection_pck'], .25)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)


if __name__ == '__main__': unittest.main()
