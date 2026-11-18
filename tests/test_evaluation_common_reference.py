import json
from pathlib import Path
import tempfile
import unittest

import netCDF4
import numpy as np

from evaluation.score import score_files


def fixture(path, *, truth_offset=0., prediction_offset=0., times=(0,6), missing_member=False, lon_offset=0.):
    with netCDF4.Dataset(path,'w') as d:
        for name,size in [('time',2),('ensemble',2),('y',2),('x',2)]:d.createDimension(name,size)
        t=d.createVariable('time','f8',('time',));t.units='hours since 2012-01-01';t[:]=times
        for name,offset in [('lat',0.),('lon',lon_offset)]:
            v=d.createVariable(name,'f8',('y','x'));v[:]=np.arange(4).reshape(2,2)+offset
        g=d.createGroup('truth');v=g.createVariable('q','f4',('time','y','x'),fill_value=np.nan)
        a=np.arange(1,9,dtype=np.float32).reshape(2,2,2)+truth_offset;a[:,0,0]=np.nan;v[:]=a
        g=d.createGroup('prediction');v=g.createVariable('q','f4',('ensemble','time','y','x'),fill_value=np.nan)
        a=np.stack([a,a])+prediction_offset
        if missing_member:a[1,0,1,1]=np.nan
        v[:]=a


class CommonReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        fixture(self.p/'reference.nc');fixture(self.p/'prediction.nc',prediction_offset=1)

    def run_score(self,**kwargs):
        return score_files(self.p/'reference.nc',[self.p/'prediction.nc'],self.p/'scores.csv',
                           reference_id='AORC',require_stored_truth=True,expected_members=2,**kwargs)

    def test_units_applied_symmetrically_and_output_audit(self):
        u=self.p/'units.json';u.write_text(json.dumps({'q':{'factor':.1,'units':'kg kg-1'}}))
        rows=self.run_score(units_path=u)
        self.assertEqual(rows[0]['count'],6)
        self.assertAlmostEqual(rows[0]['RMSE'],.1)
        self.assertEqual(rows[0]['units'],'kg kg-1')
        self.assertIsNone(json.loads((self.p/'scores.audit.json').read_text())['post_mean_threshold'])

    def test_same_product_name_does_not_allow_different_truth(self):
        fixture(self.p/'prediction.nc',truth_offset=1)
        with self.assertRaisesRegex(ValueError,'truth values or masks'):self.run_score()
        self.assertFalse((self.p/'scores.csv').exists())

    def test_different_times_rejected_before_scoring(self):
        fixture(self.p/'prediction.nc',times=(6,0))
        with self.assertRaisesRegex(ValueError,'Timestamp'):self.run_score()

    def test_different_grid_rejected(self):
        fixture(self.p/'prediction.nc',lon_offset=.01)
        with self.assertRaisesRegex(ValueError,'lon coordinates'):self.run_score()

    def test_one_missing_member_is_not_silently_averaged(self):
        fixture(self.p/'prediction.nc',missing_member=True)
        with self.assertRaisesRegex(ValueError,'member'):self.run_score()

    def test_optional_strict_source_units(self):
        with self.assertRaisesRegex(ValueError,'unit attributes'):self.run_score(require_unit_attributes=True)


if __name__=='__main__':unittest.main()
