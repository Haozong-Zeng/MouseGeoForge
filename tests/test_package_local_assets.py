import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('asset_pack', Path(__file__).resolve().parents[1] / 'tools/package_local_assets.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class PackagingTest(unittest.TestCase):
    def test_roundtrip_resume_and_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = pack.ROOT
            pack.ROOT = Path(temp)
            try:
                source = pack.ROOT / 'dataset/demo'
                source.mkdir(parents=True)
                (source / 'hello.bin').write_bytes(b'hello\0world' * 1000)
                os.link(source / 'hello.bin', source / 'hardlink.bin')
                (source / 'reference').symlink_to('hello.bin')
                (source / 'empty').mkdir()
                out = pack.ROOT / 'backup/test'
                out.mkdir(parents=True)
                rows = [{'path': p.relative_to(pack.ROOT).as_posix(),
                         'mtime_ns': p.lstat().st_mtime_ns,
                         'bytes': p.stat().st_size if p.is_file() and not p.is_symlink() else 0,
                         'type': 'symlink' if p.is_symlink() else 'directory' if p.is_dir() else 'file'} for p in pack.entries(source)]
                pack.dump(out / 'list.json', rows)
                pack.dump(out / 'plan.json', {'root': str(pack.ROOT), 'jobs': [{'name': 'demo', 'manifest': 'list.json', 'members': len(rows), 'bytes': sum(r['bytes'] for r in rows)}]})
                pack.run(out, 1)
                first = json.loads((out / 'jobs/demo.json').read_text())
                self.assertEqual(first['state'], 'verified')
                pack.run(out, 1)
                self.assertEqual(json.loads((out / 'jobs/demo.json').read_text())['sha256'], first['sha256'])
                checks = json.loads((out / 'packages/demo.manifest.json').read_text())['members']
                regular = [r for r in checks if r['type'] == 'file']
                self.assertEqual(len(regular), 2)
                self.assertEqual(regular[0]['sha256'], regular[1]['sha256'])
                # A changed source must fail before creating a final archive.
                second = pack.ROOT / 'backup/changed'
                second.mkdir(parents=True)
                pack.dump(second / 'list.json', rows)
                pack.dump(second / 'plan.json', json.loads((out / 'plan.json').read_text()))
                (source / 'hello.bin').write_bytes(b'changed')
                with self.assertRaises(SystemExit):
                    pack.run(second, 1)
                self.assertFalse((second / 'packages/demo.tar.gz').exists())
            finally:
                pack.ROOT = previous


if __name__ == '__main__':
    unittest.main()
