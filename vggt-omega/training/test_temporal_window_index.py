import unittest
import numpy as np
from training.temporal_window_index import valid_window_starts


class TestTemporalWindowIndex(unittest.TestCase):
    def test_gap_cannot_be_skipped(self):
        ids=np.array([0,1,2,4,5,6,7])
        self.assertEqual(valid_window_starts(ids,[0]*7,[False]+[True]*6,
            window_size=2,frame_step=2).tolist(),[0,3,4])

    def test_action_boundary(self):
        self.assertEqual(valid_window_starts(np.arange(6),[0,0,0,1,1,1],
            [False,True,True,False,True,True],window_size=3).tolist(),[0,3])

    def test_regular_subsampling(self):
        ids=np.arange(1,392,3)
        starts=valid_window_starts(ids,[0]*131,[False]+[True]*130,
            source_step=3,window_size=4,frame_step=2)
        self.assertEqual(len(starts),125)
        self.assertEqual(np.diff(ids[starts[0]+np.arange(4)*2]).tolist(),[6,6,6])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            valid_window_starts([1,1],[0,0],[False,True])


if __name__=='__main__':
    unittest.main()
