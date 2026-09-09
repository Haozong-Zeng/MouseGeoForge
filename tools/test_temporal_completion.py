"""Small, CPU-only tests for completion path/validation guards."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from complete_subsampled_blender_datasets import validate_scene, win


class CompletionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.scene = Path(self.temp.name)/'frame_000002'
        for folder in ('cams','annotations','blended_images','rendered_depth_maps','body_masks'):
            (self.scene/folder).mkdir(parents=True,exist_ok=True)
        self.job = {'cameras':['CameraA'],'resolution':[8,6]}
        (self.scene/'scene_meta.json').write_text(json.dumps({'frame':2,'views':[{'camera':'CameraA'}]}))
        (self.scene/'keypoints.json').write_text('{"frame":2}')
        np.savez(self.scene/'keypoints.npz',names=np.array([f'p{i}' for i in range(29)]),
                 joint_world_xyz=np.zeros((29,3)),view_ids=np.array([0]))
        (self.scene/'cams/pair.txt').write_text('1\n0\n0\n')
        (self.scene/'cams/00000000_cam.txt').write_text('test')
        (self.scene/'annotations/00000000.json').write_text('{"camera":"CameraA"}')
        for suffix in ('','_masked'):
            Image.new('RGB',(8,6)).save(self.scene/f'blended_images/00000000{suffix}.jpg')
        Image.new('L',(8,6),255).save(self.scene/'body_masks/00000000.png')
        self.depth(np.ones((6,8),dtype='<f4'))

    def depth(self, values):
        with (self.scene/'rendered_depth_maps/00000000.pfm').open('wb') as f:
            f.write(b'Pf\n8 6\n-1.0\n')
            f.write(values.tobytes())

    def test_valid(self):
        validate_scene(self.scene,self.job)

    def test_reject_negative_depth(self):
        self.depth(-np.ones((6,8),dtype='<f4'))
        with self.assertRaises(AssertionError):
            validate_scene(self.scene,self.job)

    def test_reject_wrong_frame(self):
        (self.scene/'keypoints.json').write_text('{"frame":3}')
        with self.assertRaises(AssertionError):
            validate_scene(self.scene,self.job)

    def test_missing_old_mask_is_optional_only_during_preflight(self):
        (self.scene/'body_masks/00000000.png').unlink()
        validate_scene(self.scene,self.job,require_mask=False)
        with self.assertRaises(FileNotFoundError):
            validate_scene(self.scene,self.job)

    def test_wrong_mask_size(self):
        Image.new('L',(4,3)).save(self.scene/'body_masks/00000000.png')
        with self.assertRaises(AssertionError):
            validate_scene(self.scene,self.job)

    def test_wsl_windows_path(self):
        self.assertEqual(win(Path('/home/haozong/projects/MouseGeoForge')),
            '\\\\wsl.localhost\\Ubuntu\\home\\haozong\\projects\\MouseGeoForge')


if __name__=='__main__':
    unittest.main()
