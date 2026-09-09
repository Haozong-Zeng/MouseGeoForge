from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core23_temporal_sequence_dataset import Core23TemporalSequenceDataset


class Core23TemporalSequenceDatasetTest(unittest.TestCase):
    def test_pose_window_and_instance_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "sequence.npz"
            world = np.arange(10 * 1 * 23 * 3, dtype=np.float32).reshape(10, 1, 23, 3)
            np.savez_compressed(
                sequence,
                instance_ids=np.asarray([7], dtype=np.int32),
                joint_world_xyz=world,
                world_valid=np.ones((10, 1, 23), dtype=bool),
                frame_indices=np.arange(10, dtype=np.int64),
                timestamps_seconds=np.arange(10, dtype=np.float64) / 50.0,
            )
            manifest = root / "sequences.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sequence_id": "example",
                        "source": "test",
                        "split": "train",
                        "sequence_npz": "sequence.npz",
                        "frame_count": 10,
                        "fps": 50.0,
                        "instance_count": 1,
                        "fixed_camera": True,
                        "camera_available": False,
                        "rgb_available": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = Core23TemporalSequenceDataset(
                manifest,
                window_size=3,
                frame_step=2,
                window_stride=1,
                max_instances=2,
                max_temporal_speed_mm_s=10000.0,
                project_root=root,
            )
            sample = dataset[1]
            self.assertEqual(tuple(sample["keypoint_world_xyz"].shape), (3, 2, 23, 3))
            self.assertEqual(sample["frame_indices"].tolist(), [1, 3, 5])
            self.assertTrue(sample["instance_valid"][:, 0].all())
            self.assertFalse(sample["instance_valid"][:, 1].any())
            self.assertEqual(sample["instance_ids"].tolist(), [7, -1])
            np.testing.assert_allclose(
                sample["delta_seconds"].numpy(), [0.0, 0.04, 0.04], atol=1e-7
            )
            self.assertFalse(sample["temporal_transition_valid"][0].any())
            self.assertTrue(sample["temporal_transition_valid"][1:, 0].all())
            self.assertFalse(sample["temporal_transition_valid"][:, 1].any())
            self.assertEqual(tuple(sample["embedding_action"].shape), (3, 2, 2))
            self.assertFalse(sample["embedding_action_valid"].any())


if __name__ == "__main__":
    unittest.main()
