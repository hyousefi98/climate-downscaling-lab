import unittest
from pathlib import Path
import tempfile
import numpy as np

from baselines.ridge import fit_ridge, predict_ridge
from evaluation.metrics import Accumulator, basic_metrics, csi, crps, ensemble_mean, select_oracle_member
from evaluation.quantiles import fit_training_quantile
from evaluation.percentile_fss import exact_quantiles
from evaluation.spatial import exceedance, fractions, fss


class MetricTests(unittest.TestCase):
    def test_streaming_matches_direct_and_signed_bias(self):
        o=np.array([-3.,-1.,2.,7.]);p=np.array([-2.,0.,4.,8.])
        a=Accumulator();a.add(o[:1],p[:1]);a.add(o[1:],p[1:])
        b=basic_metrics(o,p)
        for key in b:self.assertAlmostEqual(a.result()[key],b[key])
        self.assertAlmostEqual(b['PBIAS_percent'],100*5/13)

    def test_missing_prediction_cannot_shrink_reference_mask(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            basic_metrics([1.,2.],[1.,np.nan])
        with self.assertRaisesRegex(ValueError,'member'):
            ensemble_mean(np.array([[1.,2.],[1.,np.nan]]),np.array([True,True]))
        self.assertEqual(basic_metrics([1.,np.nan],[2.,np.nan])['count'],1)

    def test_no_implicit_post_mean_threshold(self):
        m=np.array([[0.,0.],[0.12,0.]],dtype=np.float32)
        p=ensemble_mean(m,np.ones(2,bool))
        self.assertGreater(p[0],0)
        self.assertLess(p[0],.1)
        self.assertAlmostEqual(basic_metrics(np.zeros(2),p)['RMSE'],float(p[0])/np.sqrt(2))

    def test_csi_empty_and_equal_threshold(self):
        self.assertIsNone(csi([0,0],[0,0],1))
        self.assertEqual(csi([1,0],[1,1],1),.5)
        self.assertEqual(csi([1],[1],1),1)

    def test_empirical_crps_and_single_member(self):
        self.assertEqual(crps(np.array([1.]),np.array([[0.],[2.]])),.5)
        self.assertEqual(crps(np.array([1.]),np.array([[3.]])),2)

    def test_oracle_member_ties_and_undefined_kge(self):
        o=np.array([1.,2.,4.]);a=np.array([o*2,o,o])
        self.assertEqual(select_oracle_member(o,a,'kge'),1)
        self.assertEqual(select_oracle_member(o,a,'rmse'),1)
        with self.assertRaisesRegex(ValueError,'undefined'):
            select_oracle_member(np.zeros(3),np.zeros((2,3)),'kge')

    def test_fss_probability_differs_from_ensemble_mean(self):
        o=np.array([[20.,0.],[0.,0.]])
        a=np.array([[[30.,0.],[0.,0.]],np.zeros((2,2))])
        self.assertEqual(fss(o,a.mean(0),20,1),0)
        self.assertAlmostEqual(fss(o,a,20,1,ensemble_probability=True),.8)

    def test_fss_temporal_variants_are_distinct(self):
        o=np.ones((2,3,3));o[0]=0;o[0,0,0]=1
        p=np.ones_like(o);p[0]=0
        self.assertAlmostEqual(fss(o,p,1,1,temporal='pooled'),18/19)
        self.assertAlmostEqual(fss(o,p,1,1,temporal='mean_time'),.5)
        self.assertIsNone(fss(np.zeros((2,2)),np.zeros((2,2)),1,1))

    def test_mask_normalized_neighborhood_not_zero_padding(self):
        x=np.array([[1.,np.nan,np.nan]])
        mask=np.array([[True,False,False]])
        valid,_=fractions(x,mask,3,'valid','land')
        old,_=fractions(x,mask,3,'zero_padded','land')
        self.assertAlmostEqual(valid[0,0],1)
        self.assertAlmostEqual(old[0,0],1/9)

    def test_empty_neighborhood_does_not_gain_roundoff_support(self):
        from scipy.ndimage import convolve
        rng=np.random.default_rng(123)
        mask=rng.random((80,80))<.3;mask[20:60,20:60]=False
        expected=convolve(mask.astype(int),np.ones((9,9),int),mode='constant',cval=0)>0
        values,support=fractions(mask.astype(float),mask,9,'valid','expanded')
        np.testing.assert_array_equal(support,expected)
        np.testing.assert_allclose(values[support],1)

    def test_quantiles_include_dry_values_and_prevent_test_fitting(self):
        q=fit_training_quantile(np.array([[0.,0.],[2.,4.]]),['2000-01-01','2000-01-02'],.5,
                                product='AORC',evaluation_start='2012-01-01')
        self.assertEqual(q.threshold,1)
        self.assertEqual(q.zero_count,2)
        with self.assertRaisesRegex(ValueError,'before evaluation'):
            fit_training_quantile([[1]],['2012-01-01'],.95,product='AORC',evaluation_start='2012-01-01')

    def test_float32_quantile_comparison_does_not_round_threshold_down(self):
        value=np.float32(1.)
        threshold=1.0+float(np.finfo(np.float32).eps)/4
        self.assertEqual(exceedance(np.array([value]),threshold)[0],0)

    def test_ridge_intercept_not_penalized(self):
        x=np.array([[-1.],[0.],[1.]])
        y=2*x+7
        w=fit_ridge(x,y,alpha=100)
        self.assertAlmostEqual(w[-1,0],7)
        self.assertLess(w[0,0],.1)
        self.assertTrue(np.allclose(predict_ridge(x,fit_ridge(x,y,0)),y))

    def test_streamed_zero_inclusive_quantiles_match_numpy(self):
        x=np.array([0.,0.,0.,1.,2.,7.,8.])
        with tempfile.TemporaryDirectory() as p:
            result=exact_quantiles([x[:2],x[2:5],x[5:]],[0,.5,.95,1],Path(p)/'values')
            self.assertFalse((Path(p)/'values').exists())
        np.testing.assert_allclose([r['threshold_mm'] for r in result],np.quantile(x,[0,.5,.95,1]))
        with tempfile.TemporaryDirectory() as p:
            result=exact_quantiles([np.zeros(5)],[.95],Path(p)/'values')
        self.assertEqual(result[0]['threshold_mm'],0)


if __name__=='__main__':unittest.main()
