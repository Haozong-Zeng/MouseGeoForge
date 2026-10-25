from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from audit_calibrated_mouse_dannce_holdout_20261025 import replay_split,parse_namespace,LOADER
class Checks(unittest.TestCase):
 def test_actual80frame_loader_test_is_train_subset(self):
  r=replay_split(LOADER.read_text(),80,6,Path('/fixture'));a={x['path'] for x in r['train']};b={x['path'] for x in r['test']};self.assertEqual((len(a),len(b),len(a&b)),(480,18,18))
 def test_actual_loader_general_count_and_time(self):
  r=replay_split(LOADER.read_text(),12,2,Path('/fixture'));self.assertEqual(len(r['train']),24);self.assertEqual([x['time'] for x in r['test'][:3]],[0.,4/12,8/12]);self.assertEqual(r['train'][11]['time'],11/12)
 def test_literal_cfg(self):
  self.assertEqual(parse_namespace("Namespace(eval=True,llffhold=8,kplanes_config={'a':[1,2]})")['llffhold'],8)
 def test_cfg_does_not_execute_calls(self):
  for source in ['evil(eval=True)',"Namespace(x=__import__('os').getcwd())",'Namespace(**{})']:
   with self.assertRaises((ValueError,TypeError)):parse_namespace(source)
if __name__=='__main__':unittest.main()
