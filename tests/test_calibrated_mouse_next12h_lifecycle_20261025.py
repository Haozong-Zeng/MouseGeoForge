import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'tools/run_calibrated_mouse_next12h_20261025.py'
spec=importlib.util.spec_from_file_location('nextcycle',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Lifecycle(unittest.TestCase):
 def test_full_identity_and_stale_stop(self):
  current=m.identity(os.getpid());self.assertTrue(m.matches(current));stale={**current,'boot_id':'invalid-test-boot'};self.assertFalse(m.matches(stale));m.stop(stale);self.assertTrue(m.matches(current))
 def test_real_cutoff_stops_owned_fixture_and_completes(self):
  with tempfile.TemporaryDirectory(prefix='mouse12h_lifecycle_') as d:
   out=Path(d)
   for n in ['logs','jobs']:(out/n).mkdir()
   now=dt.datetime.now(dt.timezone.utc)
   m.write(out/'plan.json',dict(optimization_cutoff_utc=(now+dt.timedelta(seconds=2)).isoformat(),hard_finish_utc=(now+dt.timedelta(seconds=7)).isoformat()))
   helper=out/'sleep.py';helper.write_text('import time\ntime.sleep(60)\n')
   h=hashlib.sha256(helper.read_bytes()).hexdigest()
   m.write(out/'queue.json',[dict(id='fixture_sleep',args=[str(helper)],optimization=True,minimum_seconds=0,timeout_seconds=60,source_sha256={str(helper):h})])
   run=subprocess.Popen([sys.executable,str(P),'--output',str(out)],cwd=ROOT,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   _,err=run.communicate(timeout=15);self.assertEqual(run.returncode,0,err.decode())
   completion=m.read(out/'completion.json');self.assertIsNotNone(completion);self.assertEqual(completion['state']['jobs']['fixture_sleep']['status'],'failed');self.assertFalse(m.read(out/'active_child.json'))
   job=m.read(out/'jobs/fixture_sleep.json');self.assertEqual(job['optimization'],True)
   for f in ['controller_process.json','watchdog_process.json']:
    saved=m.read(out/f)
    for _ in range(20):
     if not m.matches(saved):break
     time.sleep(.1)
    self.assertFalse(m.matches(saved),f)
if __name__=='__main__':unittest.main()
