import unittest
from tools.import_pymousestudio_raw import safe_member


class SafeRawImportTests(unittest.TestCase):
    def test_valid(self):self.assertEqual(str(safe_member('FR2/videos/camA.mp4','FR2')),'FR2/videos/camA.mp4')
    def test_escape(self):
        for path in ('/FR2/a','FR2/../../a','FR2/../x','X/a','FR2/C:a','FR2\\a'):
            with self.assertRaises(ValueError):safe_member(path,'FR2')
