"""Optional CPU check of the actual historical RF training/inference builders."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch','sklearn','physicsnemo']),
                     'Requires optional train and baselines dependencies')
class FeatureBuilderIntegrationTests(unittest.TestCase):
    def test_training_and_inference_feature_order_match(self):
        script=r'''
from datetime import datetime
import numpy as np
import RF_3km as rf
from generate_rf_nc import _build_features

class Dataset:
    land_mask=np.array([[True,True,False],[True,True,True]])
    def latitude(self):return np.array([[1,1,1],[2,2,2]],dtype=np.float32)
    def longitude(self):return np.array([[3,4,5],[3,4,5]],dtype=np.float32)
dataset=Dataset()
x=np.arange(48,dtype=np.float32).reshape(8,2,3)/10
y=x[:2].copy()
normals=np.arange(367*8,dtype=np.float32).reshape(367,8)/100
time=np.datetime64('2012-07-10T06:00:00')
common=dict(add_coords=True,add_latlon=True,add_landmask=True,add_climatology=True,
            include_lead_time=False,lead_time_scale=1.,time_value=time,climatology=normals)
training=rf._extract_pixels(dataset,(y,x),dataset.land_mask,np.random.default_rng(42),
                            samples_per_image=0,use_physical=False,coords_cache=None,
                            latlon_cache=None,landmask_cache=None,**common)[0]
inference,_=_build_features(x,dataset,lead_time=None,mask=dataset.land_mask,**common)
assert training.shape==(5,31)
np.testing.assert_array_equal(training,inference)
assert rf._time_to_fields(time)==(7,192)
print('Actual RF training/inference feature order matches exactly (31 columns).')
'''
        directory=Path(__file__).resolve().parents[1]/'baselines/historical'
        result=subprocess.run([sys.executable,'-c',script],cwd=directory,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
