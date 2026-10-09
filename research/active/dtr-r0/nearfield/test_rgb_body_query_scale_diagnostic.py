import unittest
import numpy as np

from rgb_body_query_scale_diagnostic import fit_scale, paired_mask, depth_error, summarize


class ScaleDiagnosticTests(unittest.TestCase):
    def test_multiplicative_fit_uses_all_paired_observed_pixels(self):
        truth=np.array([1.,2.,4.,np.nan,0.,1.])
        prediction=np.array([.5,1.,2.,1.,1.,1.])
        mask=paired_mask(truth,prediction,np.array([True]*5+[False]))
        self.assertEqual(mask.tolist(),[True,True,True,False,False,False])
        scale=fit_scale(np.log(truth[mask]/prediction[mask]))
        self.assertEqual(scale,2.)
        self.assertEqual(depth_error(truth,prediction,mask,scale)['absrel_mean'],0.)
        with self.assertRaises(ValueError): fit_scale([])

    def test_summary_keeps_witness_and_support_separate(self):
        rows=[dict(scan='s',frame=0,tp=15,fn=5,fp=2,tn=8,reference_state='POSITIVE',predicted_positive=True),
              dict(scan='s',frame=0,tp=0,fn=0,fp=3,tn=7,reference_state='FREE_ON_SAMPLED_RAYS',predicted_positive=False),
              dict(scan='s',frame=1,tp=16,fn=4,fp=0,tn=4,reference_state='POSITIVE',predicted_positive=True)]
        s=summarize(rows)
        self.assertEqual([s[k] for k in ('tp','fn','fp','tn')],[31,9,5,19])
        self.assertEqual(s['query_positive_hits'],2)
        self.assertEqual(s['query_positive_known_witness_hits'],1)
        self.assertEqual(s['sampled_free_false_support'],0)
        self.assertEqual(s['frames'],2)


if __name__=='__main__': unittest.main()
