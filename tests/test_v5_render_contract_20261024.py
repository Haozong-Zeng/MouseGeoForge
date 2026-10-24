import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import v5_render_contract_20261024 as c


class RenderContractTests(unittest.TestCase):
    def old_design(self):
        return dict(sources=[dict(id=x, admitted_split='train', frames=[49, 97, 145]) for x in ('climb', 'moseq')],
                    rigs=[dict(id=f'{s}_rig{i:02d}', split=s) for s, n in [('train', 48), ('val', 12), ('sealed', 8)] for i in range(n)])

    def test_split_counts_and_unambiguous_paths(self):
        d = c.make_design(self.old_design())
        paths = [c.scene_path(c.OUT_REL, r) for r in d['scenes']]
        self.assertEqual(len(paths), 768)
        self.assertEqual(len(set(paths)), 768)
        self.assertEqual(sum(r['condition'] == 'dev_primary' for r in d['scenes']), 72)
        self.assertEqual(sum(r['condition'] == 'dev_attribution' for r in d['scenes']), 72)
        split = {e['id']: e['split'] for e in d['environments']}
        for row in d['scenes']:
            if row['split'] == 'train':
                self.assertEqual(split[row['environment']], 'train')
            if row['split'] == 'sealed':
                self.assertEqual(split[row['environment']], 'sealed')

    def test_layout_candidates_reproducible_and_camera_independent(self):
        d = c.make_design(self.old_design())
        env = d['environments'][0]
        p = c.environment_parameters(env, 0)
        self.assertEqual(p['rotation_z'], 0)
        self.assertEqual(p['material_gain'], 1)
        self.assertEqual(p, c.environment_parameters(env, 0))
        self.assertNotEqual(p, c.environment_parameters(env, 1))
        row = d['scenes'][0]
        light = c.lighting_parameters(row)
        self.assertTrue(light['shared_across_all_views'])
        self.assertNotIn('view', light)

    def test_payload_validator_rejects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'render/train_rig00/train_env00/climb/frame_000049'
            folder.mkdir(parents=True)
            side = 16
            record = dict(kind='synthetic', admitted_motion_split='train', split='train', motion_id='climb',
                          rig='train_rig00', environment_id='train_env00', scene=folder.relative_to(root).as_posix(), views=[])
            k = np.array([[20., 0, 7.5], [0, 20., 7.5], [0, 0, 1]])
            for i in range(3):
                image = folder / f'rgb_{i}.png'; truth = folder / f'truth_{i}.npz'
                Image.new('RGB', (side, side)).save(image)
                np.savez_compressed(truth, depth=np.ones((side, side), np.float32), body=np.ones((side, side), bool),
                                    valid=np.ones((side, side), bool), K=k, E=np.eye(4))
                record['views'].append(dict(id=i, physical_role='bottom' if i == 0 else 'side', passed=True,
                                           foreground_pixels=side * side, projection_closure_max_px=0,
                                           image=image.relative_to(root).as_posix(), truth=truth.relative_to(root).as_posix(),
                                           K=k.tolist(), E=np.eye(4).tolist(),
                                           file_sha256={image.name: c.sha(image), truth.name: c.sha(truth)}))
            self.assertTrue(c.validate_record(root, record, side))
            escaped = copy.deepcopy(record)
            escaped['views'][0]['image'] = '../rgb_0.png'
            with self.assertRaises(AssertionError):
                c.validate_record(root, escaped, side)
            with (folder / 'rgb_0.png').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaises(AssertionError):
                c.validate_record(root, record, side)


if __name__ == '__main__':
    unittest.main()
