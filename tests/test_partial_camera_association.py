import unittest
import numpy as np
from mousegeoforge.partial_camera_association import conservative_camera_groups as group


class PartialTests(unittest.TestCase):
    def setup_graph(self,views,edges):
        nodes=[dict(view=v,query=i) for i,v in enumerate(views)];d=np.full((len(nodes),len(nodes)),np.nan)
        for a,b in edges:d[a,b]=d[b,a]=1.
        return nodes,d
    def test_default_off(self):
        n,d=self.setup_graph([0,1,2],[(0,1),(0,2),(1,2)]);self.assertEqual(group(n,d,2)['groups'],[])
    def test_two_separate_subjects(self):
        n,d=self.setup_graph([0,1,2,0,1,2],[(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)])
        self.assertEqual(group(n,d,2,enabled=True)['groups'],[[0,1,2],[3,4,5]])
    def test_cutoff_strict_and_singletons(self):
        n,d=self.setup_graph([0,1,2],[(0,1),(0,2),(1,2)])
        self.assertEqual(group(n,d,1,enabled=True)['groups'],[])
        self.assertEqual(group(n[:2],d[:2,:2],2,enabled=True)['groups'],[])
    def test_ties_abstain(self):
        n,d=self.setup_graph([0,1,1,2],[(0,1),(0,2),(0,3),(1,3),(2,3)])
        self.assertEqual(group(n,d,2,enabled=True)['groups'],[])
    def test_overlapping_cliques_abstain(self):
        n,d=self.setup_graph([0,1,2,3,4],[(0,1),(0,2),(1,2),(0,3),(0,4),(3,4)])
        out=group(n,d,2,enabled=True);self.assertEqual(out['groups'],[]);self.assertEqual(len(out['ambiguous_groups']),2)
    def test_permutation(self):
        n,d=self.setup_graph([0,1,2,3],[(0,1),(0,2),(1,2)]);perm=[2,0,3,1]
        out=group([n[i] for i in perm],d[np.ix_(perm,perm)],2,enabled=True)
        self.assertEqual(sorted(perm[i] for i in out['groups'][0]),[0,1,2])
    def test_invalid_and_samecamera(self):
        n,d=self.setup_graph([0,0,1],[(0,1),(0,2),(1,2)])
        self.assertEqual(group(n,d,2,enabled=True)['groups'],[])
        with self.assertRaises(ValueError):group(n,d,float('nan'),enabled=True)
        n[1]=n[0].copy()
        with self.assertRaises(ValueError):group(n,d,2,enabled=True)


if __name__=='__main__':unittest.main()
