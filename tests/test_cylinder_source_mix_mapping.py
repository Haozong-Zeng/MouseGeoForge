import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from preflight_cylinder_source_mix import role_mapping


class SourceMixMappingTests(unittest.TestCase):
    def rig(self):
        rows=[dict(view_id=i,orientation='side',forward=[np.cos(i*np.pi/3),np.sin(i*np.pi/3),0.]) for i in range(6)]
        return rows+[dict(view_id=6,orientation='bottom',forward=[0.,0.,1.]),dict(view_id=7,orientation='top',forward=[0.,0.,-1.])]

    def test_ids_are_not_physical_roles(self):
        source=self.rig(); dest=copy.deepcopy(source)
        expected={i:6-i for i in range(7)};expected[7]=7
        for row in dest:row['view_id']=expected[row['view_id']]
        mapping,angles=role_mapping(source,dest)
        self.assertEqual(mapping,expected)
        self.assertLess(max(angles.values()),1e-5)

    def test_missing_pole_rejected(self):
        with self.assertRaises(ValueError):role_mapping(self.rig(),self.rig()[:-1])

    def test_duplicate_camera_rejected(self):
        dest=self.rig();dest[0]['view_id']=dest[1]['view_id']
        with self.assertRaises(ValueError):role_mapping(self.rig(),dest)


if __name__=='__main__':unittest.main()
