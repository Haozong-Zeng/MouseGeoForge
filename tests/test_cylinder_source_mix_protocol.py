import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_cylinder_source_mix import validate_sequence


class SourceMixProtocolTests(unittest.TestCase):
    def inputs(self):
        # Fully synthetic records: no local assets or GPU required.
        seq=[];before=[];slots=[];records={}
        for i in range(800):
            special=i%2==0
            n=(i//2)%4+1 if special else 2
            j=(i//2)%16 if special else 16+(i//2)%24
            rec=dict(scene=f'train{j}',source_group='train_group',split='train',image_hw=[768,768],views=list(range(n)),
                     diagnostic_view_order=list(range(n)),physical_roles=['side']*n,normalization_scale=2.)
            # 301 replay slots are 768x432, matching the real 3540384-token budget.
            if not special and i//2<301:rec['image_hw']=[432,768]
            records[str(i)]=rec;before.append([str(i)]);slots.append(dict(kind='cylinder' if special else 'single'))
            new=copy.deepcopy(rec)
            if special and j%2:new['scene']=f'new{j}'
            seq.append(new)
        records['dev']=dict(scene='dev',source_group='dev_group')
        return seq,dict(records=records,arms={'combination_coverage':before},slots=slots,evaluation=[dict(key='dev')])

    def test_accept_fixed_budgets(self):
        seq,old=self.inputs();self.assertEqual(validate_sequence(seq,old)['changed_slots'],200)

    def test_replay_change_rejected(self):
        seq,old=self.inputs();seq[1]['normalization_scale']=3.
        with self.assertRaises(ValueError):validate_sequence(seq,old)

    def test_development_source_rejected(self):
        seq,old=self.inputs();seq[2]['source_group']='dev_group'
        with self.assertRaises(ValueError):validate_sequence(seq,old)

    def test_role_change_rejected(self):
        seq,old=self.inputs();seq[2]['physical_roles'][0]='bottom'
        with self.assertRaises(ValueError):validate_sequence(seq,old)


if __name__=='__main__':unittest.main()
