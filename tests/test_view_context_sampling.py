import copy, unittest
from mousegeoforge.view_context_sampling import single_context_flags,random_reference_contexts,full_view_contexts

class ContextSamplingTests(unittest.TestCase):
    def setUp(self):
        self.meta=[dict(record=dict(role='train',session='AZL3',kind='real' if i<8 else 'synthetic'),view=0 if i<4 else 1,detection_complete=True) for i in range(16)]
        self.schedule=[dict(real=list(range(8)),synthetic=list(range(8,16))) for _ in range(7)]
    def test_matched_exposure_deterministic_no_mutation(self):
        old=copy.deepcopy(self.schedule);a=single_context_flags(self.schedule,self.meta,17)
        self.assertEqual(a,single_context_flags(self.schedule,self.meta,17));self.assertEqual(old,self.schedule)
        for step in a:
            self.assertEqual(sum(step['real'][:4]),2);self.assertEqual(sum(step['real'][4:]),2);self.assertEqual(sum(step['synthetic']),4)
    def test_validation_and_locked_rejected(self):
        self.meta[0]['record']['role']='validation'
        with self.assertRaises(ValueError):single_context_flags(self.schedule,self.meta,1)
        self.meta[0]['record'].update(role='train',session='GR2')
        with self.assertRaises(ValueError):single_context_flags(self.schedule,self.meta,1)
    def test_wrong_bottom_order_rejected(self):
        self.meta[0]['view']=2
        with self.assertRaises(ValueError):single_context_flags(self.schedule,self.meta,1)
    def test_random_reference_preserves_counts_and_single_flags(self):
        flags=single_context_flags(self.schedule,self.meta,17);old=copy.deepcopy(flags);contexts=random_reference_contexts(self.schedule,flags,self.meta,29)
        self.assertEqual(flags,old);self.assertEqual(contexts,random_reference_contexts(self.schedule,flags,self.meta,29))
        for a,b in zip(flags,contexts):
            for domain in a:
                self.assertEqual(a[domain],[c=='single' for c in b[domain]]);self.assertEqual(b[domain].count('reversed'),2);self.assertEqual(b[domain].count('paired'),2)
            self.assertEqual(b['real'][:4].count('reversed'),1);self.assertEqual(b['real'][4:].count('reversed'),1)
    def test_random_reference_rejects_unbalanced_or_mismatched_flags(self):
        flags=single_context_flags(self.schedule,self.meta,17)
        with self.assertRaises(ValueError):random_reference_contexts(self.schedule,flags[:1],self.meta,29)
        flags[0]['real']=[False]*8
        with self.assertRaises(ValueError):random_reference_contexts(self.schedule,flags,self.meta,29)

    def test_full_view_preserves_single_and_bottom_exposure(self):
        flags=single_context_flags(self.schedule,self.meta,17);before=copy.deepcopy((flags,self.schedule));contexts=full_view_contexts(self.schedule,flags,self.meta,43)
        self.assertEqual((flags,self.schedule),before);self.assertEqual(contexts,full_view_contexts(self.schedule,flags,self.meta,43))
        for f,c in zip(flags,contexts):
            for domain in f:
                self.assertEqual(f[domain],[x=='single' for x in c[domain]])
                self.assertEqual({x:c[domain].count(x) for x in ['single','shared','full']},{'single':4,'shared':2,'full':2})
            self.assertEqual(c['real'][:4].count('full'),1);self.assertEqual(c['real'][4:].count('full'),1)

if __name__=='__main__':unittest.main()
