import unittest
from mousegeoforge.pilot_schedule import learning_rates


class TestPilotSchedule(unittest.TestCase):
    def test_legacy_constant_unchanged(self):
        self.assertEqual(learning_rates({'steps': 5000, 'learning_rate': 1e-4}), [1e-4]*5000)

    def test_hold_cosine_boundaries(self):
        values = learning_rates({'steps': 15000, 'learning_rate': 1e-4,
            'lr_schedule': {'type': 'hold_cosine', 'hold_steps': 5000, 'min_learning_rate': 1e-5}})
        self.assertEqual(len(values), 15000)
        self.assertEqual(values[:5000], [1e-4]*5000)
        self.assertLess(values[5000], values[4999])
        self.assertAlmostEqual(values[9999], 5.5e-5)
        self.assertEqual(values[-1], 1e-5)
        self.assertTrue(all(a >= b for a,b in zip(values, values[1:])))

    def test_invalid_schedule_rejected(self):
        for spec in ({'type': 'unknown'}, {'type': 'hold_cosine', 'hold_steps': 20, 'min_learning_rate': 1e-5},
                     {'type': 'hold_cosine', 'hold_steps': 5, 'min_learning_rate': float('nan')}):
            with self.assertRaises(ValueError):
                learning_rates({'steps': 20, 'learning_rate': 1e-4, 'lr_schedule': spec})

    def test_budget_bounds(self):
        for steps in (0, 15001, 2.5):
            with self.assertRaises(ValueError): learning_rates({'steps': steps, 'learning_rate': 1e-4})
