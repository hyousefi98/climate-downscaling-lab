"""Tiny generated fixtures check the complete frozen-quantile FSS pipeline."""
import json
from pathlib import Path
import tempfile
import unittest

import netCDF4
import numpy as np

from evaluation.percentile_fss import compare


def rainfall_file(path, times, values):
    with netCDF4.Dataset(path,'w') as d:
        for name,size in [('time',len(times)),('y',2),('x',2)]:d.createDimension(name,size)
        t=d.createVariable('time','f8',('time',));t.units='hours since 2011-01-01';t[:]=times
        for name in ['lat','lon']:
            v=d.createVariable(name,'f8',('y','x'));v[:]=np.arange(4).reshape(2,2)
        for group in ['input','truth']:
            g=d.createGroup(group);v=g.createVariable('rain','f4',('time','y','x'));v.units='mm';v[:]=values


class PercentilePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        rainfall_file(self.p/'train_p.nc',[0,6],[[[0,0],[1,2]],[[0,0],[3,4]]])
        rainfall_file(self.p/'train_o.nc',[0,6],[[[0,0],[2,4]],[[0,0],[6,8]]])
        rainfall_file(self.p/'eval_p.nc',[12],[[[0,2],[0,4]]])
        rainfall_file(self.p/'eval_o.nc',[12],[[[0,4],[0,8]]])
        np.save(self.p/'mask.npy',np.ones((2,2),bool))

    def run_compare(self,**kwargs):
        return compare(self.p/'train_p.nc',self.p/'train_o.nc',self.p/'eval_p.nc',self.p/'eval_o.nc',
                       self.p/'mask.npy',self.p/'result',predictor_variable='rain',reference_variable='rain',
                       quantiles=(.5,),windows=(1,3),absolute_thresholds=(3.,),**kwargs)

    def test_frozen_product_thresholds_and_pooled_fss(self):
        rows=self.run_compare()
        quantile=[r for r in rows if r['experiment']=='quantile_0.5']
        self.assertEqual(len(quantile),2)
        self.assertEqual(quantile[0]['predictor_threshold_mm'],.5)
        self.assertEqual(quantile[0]['reference_threshold_mm'],1.)
        self.assertTrue(all(r['FSS']==1 for r in quantile))
        absolute=[r for r in rows if r['experiment']=='absolute_3mm'][0]
        self.assertAlmostEqual(absolute['FSS'],2/3)
        audit=json.loads((self.p/'result/audit.json').read_text())
        self.assertEqual(audit['predictor_quantiles'][0]['zero_count'],4)
        self.assertEqual(audit['evaluation_count'],1)

    def test_training_overlap_is_rejected(self):
        rainfall_file(self.p/'eval_p.nc',[6],[[[0,2],[0,4]]])
        rainfall_file(self.p/'eval_o.nc',[6],[[[0,4],[0,8]]])
        with self.assertRaisesRegex(ValueError,'strictly precede'):self.run_compare()
        self.assertFalse((self.p/'result').exists())

    def test_zero_quantile_does_not_report_trivial_perfect_fss(self):
        rainfall_file(self.p/'train_p.nc',[0,6],np.zeros((2,2,2)))
        with self.assertRaisesRegex(ValueError,'quantile is zero'):self.run_compare()
        self.assertFalse((self.p/'result').exists())


if __name__=='__main__':unittest.main()
