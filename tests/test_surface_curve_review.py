import unittest
import numpy as np
from mousegeoforge.surface_curve_review import parse_annotation,sample_fragments,distance_to_fragments,compare_fragments


class HumanCurveReviewTests(unittest.TestCase):
    def annotation(self,shapes=None,**flags):
        return dict(imageWidth=100,imageHeight=100,imagePath='camA.png',flags=flags,shapes=shapes or [])
    def parse(self,a):return parse_annotation(a,width=100,height=100,image_name='camA.png')
    def line(self,label='dorsal_surface',points=None):
        return dict(label=label,shape_type='linestrip',group_id=0,points=points or [[10,20],[80,20]],flags={})
    def test_blank_is_pending_not_invisible(self):
        r=self.parse(self.annotation());self.assertTrue(r['valid']);self.assertFalse(r['reviewed']);self.assertFalse(r['no_visible'])
    def test_explicit_empty_review(self):
        self.assertFalse(self.parse(self.annotation(reviewed=True))['valid'])
        self.assertTrue(self.parse(self.annotation(reviewed=True,no_visible_dorsal_surface=True))['valid'])
    def test_silhouette_never_enters_surface(self):
        r=self.parse(self.annotation([self.line('silhouette_only')],reviewed=True,only_silhouette_identifiable=True))
        self.assertTrue(r['valid']);self.assertEqual(len(r['lines']['dorsal_surface']),0)
    def test_bad_geometry_rejected(self):
        for p in ([[0,0],[100,0]],[[0,0],[float('nan'),1]],[[1,1],[1,1]]):
            self.assertFalse(self.parse(self.annotation([self.line(points=p)]))['valid'])
    def test_curve_gap_not_bridged(self):
        f=[np.array([[0.,0],[1,0]]),np.array([[3.,0],[4,0]])]
        d,_=distance_to_fragments(np.array([[2.,0]]),f);self.assertAlmostEqual(d[0],1.)
        self.assertFalse(np.any(np.all(sample_fragments(f)==[2,0],axis=1)))
    def test_parallel_agreement(self):
        a=[np.array([[0.,0],[10,0]])];b=[np.array([[0.,3],[10,3]])]
        self.assertAlmostEqual(compare_fragments(a,b)['equal_direction_mean_px'],3.)
    def test_missing_not_scored_zero(self):self.assertFalse(compare_fragments([],[])['comparable'])
    def test_uncertain_endpoint_not_anchor(self):
        endpoint=dict(label='neck_surface_endpoint',shape_type='point',points=[[10,20]],flags={'uncertain':True})
        r=self.parse(self.annotation([self.line(),endpoint],reviewed=True));self.assertEqual(r['endpoints'],{})
    def test_endpoint_away_from_line_rejected(self):
        endpoint=dict(label='neck_surface_endpoint',shape_type='point',points=[[10,90]],flags={})
        self.assertFalse(self.parse(self.annotation([self.line(),endpoint],reviewed=True))['valid'])
    def test_malformed_json_structure_rejected(self):
        for a in ([],{'shapes':'wrong'},self.annotation([42])):self.assertFalse(self.parse(a)['valid'])
    def test_string_flag_not_boolean(self):
        self.assertFalse(self.parse(self.annotation(reviewed=True,no_visible_dorsal_surface='false'))['valid'])


if __name__=='__main__':unittest.main()
