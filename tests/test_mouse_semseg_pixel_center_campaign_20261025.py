import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=module('campaign','tools/calibrated_mouse_pixel_center_pair_20261025.py');e=module('engine','tools/train_mouse_semseg_pixel_center_20261025.py')
class PairedArguments(unittest.TestCase):
 def test_parent_policy_and_exposure_consistent_in_train_resume_eval(self):
  for policy in ['legacy','center_exact']:
   for endpoint in [500,1000,6000]:
    for evaluation in [False,True]:
     command=c.command(policy,endpoint,Path('/tmp/never_executed_fixture'),Path('/tmp/fixture_checkpoint'),evaluation)
     a=e.parser().parse_args(command[2:]);self.assertEqual(a.parent,c.PARENT);self.assertEqual(a.label_policy,policy);self.assertEqual(a.steps,6000);self.assertEqual(a.arm,'M');self.assertEqual(a.seed,17);self.assertEqual(a.global_batch,4)
     self.assertEqual(a.command,'evaluate' if evaluation else 'train');self.assertEqual(a.stop_after,None if evaluation else endpoint)
 def test_encoder_unfreeze_rejected_at_parser(self):
  with self.assertRaises(SystemExit):e.parser().parse_args(['smoke','--label-policy','legacy','--arm','U'])
if __name__=='__main__':unittest.main()
