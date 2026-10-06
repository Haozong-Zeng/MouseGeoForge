import copy
import unittest
from mousegeoforge.bottom_view_sampling import balance_bottom_real_batches

class BottomSamplingTests(unittest.TestCase):
    def data(self):
        return [dict(record=dict(kind='real',role='train',session=s),view=v,detection_complete=True) for s in ['a','b'] for v in [0,1,2]]
    def test_counts_determinism_synthetic_preserved(self):
        meta=self.data();schedule=[dict(real=list(range(6))+[0,1],synthetic=list(range(20+i,28+i))) for i in range(12)];before=copy.deepcopy(schedule)
        result=balance_bottom_real_batches(schedule,meta,range(6),17);self.assertEqual(result,balance_bottom_real_batches(schedule,meta,range(6),17));self.assertEqual(schedule,before)
        for old,new in zip(schedule,result):self.assertEqual(old['synthetic'],new['synthetic']);self.assertEqual(sum(meta[i]['view']==0 for i in new['real']),4)
    def test_validation_or_locked_rejected(self):
        for field,value in [('role','validation'),('session','GR2')]:
            meta=self.data();meta[0]['record'][field]=value
            with self.assertRaises(ValueError):balance_bottom_real_batches([],meta,range(6),17)
    def test_missing_pool_rejected(self):
        with self.assertRaises(ValueError):balance_bottom_real_batches([],self.data(),[0,3],17)

if __name__=='__main__':unittest.main()
