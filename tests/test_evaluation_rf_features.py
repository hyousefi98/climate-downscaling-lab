"""Frozen RF feature state must never learn from evaluation values."""
from types import SimpleNamespace
import unittest

import numpy as np

from baselines.historical.rf_feature_state import FLAGS,make_state,restore_state,check_feature_count


class Dataset:
    input_variables=['rain','temperature']
    output_variables=['rain']
    land_mask=np.ones((2,2),bool)
    def latitude(self):return np.zeros((2,2))
    def longitude(self):return np.ones((2,2))
    def __getitem__(self,index):raise AssertionError('Inference must not inspect held-out fields to fit climatology')


def options(**changes):
    value={k:False for k in FLAGS};value.update(add_climatology=True,lead_time_scale=1.)
    value.update(changes);return SimpleNamespace(**value)


class FrozenFeatureTests(unittest.TestCase):
    def setUp(self):
        self.ds=Dataset();self.normals=np.full((367,2),7.,np.float32)
        self.state=make_state(options(),self.ds,{'train':True},self.normals,['2000-01-01','2011-12-31'],8)
        self.model=SimpleNamespace(n_features_in_=8,downscaling_feature_state_=self.state)

    def test_heldout_values_cannot_change_saved_training_normals(self):
        a=options(add_climatology=False)
        restored=restore_state(self.model,a,self.ds,{'train':False},['2012-01-01'])
        np.testing.assert_array_equal(restored,self.normals)
        self.assertTrue(a.add_climatology)
        restored[:]=99
        np.testing.assert_array_equal(self.state['climatology'],self.normals)

    def test_overlap_and_changed_feature_schema_rejected(self):
        with self.assertRaisesRegex(ValueError,'follow'):
            restore_state(self.model,options(),self.ds,{'train':False},['2011-12-31'])
        changed=Dataset();changed.input_variables=['temperature','rain']
        with self.assertRaisesRegex(ValueError,'variables'):
            restore_state(self.model,options(),changed,{'train':False},['2012-01-01'])
        with self.assertRaisesRegex(ValueError,'schema mismatch'):
            check_feature_count(SimpleNamespace(n_features_in_=31),np.zeros((10,29)))

    def test_old_model_has_no_implicit_climatology(self):
        self.assertIsNone(restore_state(SimpleNamespace(n_features_in_=31),options(),self.ds,{},['2012-01-01']))


if __name__=='__main__':unittest.main()
