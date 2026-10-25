from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import convert_v5_to_blenderpymouse_20261025 as c


class ConversionTests(unittest.TestCase):
    def test_pfm_preserves_rows_and_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'depth.pfm'
            a = np.array([[0, .125, 9], [31, 17, 1.5]], np.float32)
            c.write_pfm(f, a)
            np.testing.assert_array_equal(c.read_pfm(f), a)
            with f.open('rb') as handle:
                for _ in range(3):
                    handle.readline()
                np.testing.assert_array_equal(np.fromfile(handle, '<f4').reshape(2, 3), a[::-1])

    def test_camera_roundtrip_preserves_half_pixel_and_world_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            k = np.array([[510.9592692057292, 0, 191.5], [0, 510.9592692057292, 191.5], [0, 0, 1]])
            e = np.eye(4); e[:3, 3] = [12.345678901234567, -23.5, 300.25]
            f = Path(tmp) / 'camera.txt'
            c.write_cam(f, e, k, np.arange(1, 13, dtype=np.float32).reshape(3, 4))
            ee, kk = c.read_cam(f)
            np.testing.assert_array_equal(e, ee)
            np.testing.assert_array_equal(k, kk)

    def test_surface_visibility_conflict_remains_unknown(self):
        world = np.tile([0, 0, 10.], (23, 1))
        support = np.repeat(world[:, None], 16, axis=1)
        k = np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1.]])
        result = c.label_view(world, world, support, np.eye(4), k,
                              np.ones((5, 5), np.float32)*5, np.zeros((5, 5), bool),
                              np.ones(23, bool), np.zeros((23, 16), bool))
        self.assertFalse(result['visibility_valid'][c.IDS].any())
        self.assertFalse(result['visible'][c.IDS].any())
        np.testing.assert_array_equal(result['cause'][c.IDS], [-1, -1, -1])


if __name__ == '__main__':
    unittest.main()
