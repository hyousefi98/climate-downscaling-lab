"""CPU workflow: native fixture -> corrected stores/stats -> ridge/RF outputs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import netCDF4
import numpy as np
import xarray as xr
import yaml

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(all(importlib.util.find_spec(x) is not None for x in ['torch','sklearn','physicsnemo']),
                     'Requires optional train and baselines dependencies')
class PhysicalBaselineWorkflowTests(unittest.TestCase):
    def test_corrected_stores_train_and_predict_without_legacy_scaling(self):
        from data.source_profiles import AORC_CHANNELS
        from data.preprocessing import write_zarr_new,training_statistics
        spec=importlib.util.spec_from_file_location('baseline_fixture_prepare',ROOT/'scripts/preprocess/aorc.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            times=np.arange('2011-12-31T01','2012-01-01T13',dtype='datetime64[h]')
            bases=[10,3000,2000,9000,100,2900,30,40]
            fields={name:(('time','latitude','longitude'),np.full((len(times),24,24),base,dtype=np.int16)+np.arange(len(times),dtype=np.int16)[:,None,None]) for name,base in zip(AORC_CHANNELS,bases)}
            raw=xr.Dataset(fields,coords={'time':times,'latitude':30+np.arange(24)/120,'longitude':-115+np.arange(24)/120})
            inp,target=module.prepare(raw)
            write_zarr_new(inp,p/'input.zarr');write_zarr_new(target,p/'target.zarr')
            stats=training_statistics(inp,target,np.ones((8,8),bool),input_array='aorc',target_array='aorc',start='2011-01-01',end='2012-01-01')
            (p/'stats.json').write_text(json.dumps(stats))
            dataset=dict(type='physical_pair',input_data_path=str(p/'input.zarr'),target_data_path=str(p/'target.zarr'),stats_path=str(p/'stats.json'),input_variables=list(AORC_CHANNELS),output_variables=list(AORC_CHANNELS),train=True,train_years=[2011,2011],valid_years=[2012,2012],crop_shape=[8,8],coastal_buffer_pixels=0,keep_ocean_as_nan=True,precip_threshold=.1)
            train={'dataset':dataset,'validation':{'train':False}}
            gen={'dataset':dict(dataset,train=False),'generation':{'io':{'output_filename':str(p/'default.nc')},'data':{'max_samples':2}}}
            (p/'train.yaml').write_text(yaml.safe_dump(train));(p/'gen.yaml').write_text(yaml.safe_dump(gen))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            def run(args,ok=True):
                result=subprocess.run([sys.executable,'-m','baselines.run',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
                if ok:self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-3000:])
                return result
            run(['forest','--','--config',str(p/'train.yaml'),'--model-out',str(p/'rf.joblib'),'--n-estimators','3','--max-depth','3','--max-images','2','--samples-per-image','12','--max-samples','24','--n-jobs','1','--add-coords','--add-latlon','--add-landmask','--add-climatology','--climatology-max-images','3','--use-physical'])
            # No feature flags are repeated: generation must restore the training state.
            run(['predict-forest','--','--config',str(p/'gen.yaml'),'--model',str(p/'rf.joblib'),'--output',str(p/'rf.nc')])
            run(['ridge','--','--train-config',str(p/'train.yaml'),'--gen-config',str(p/'gen.yaml'),'--output',str(p/'ridge.nc'),'--coeff-out',str(p/'coeff.json'),'--max-images','3','--samples-per-image','0','--max-samples','48'])
            for name in ['rf.nc','ridge.nc']:
                with netCDF4.Dataset(p/name) as d:
                    self.assertEqual(len(d.dimensions['time']),2)
                    q=np.ma.filled(d.groups['truth']['SPFH_2maboveground'][:],np.nan)
                    self.assertGreater(float(np.nanmin(q)),.01)
                    self.assertLess(float(np.nanmax(q)),.02) # Physical q, never learned 10q.
                    temp=np.ma.filled(d.groups['truth']['TMP_2maboveground'][:],np.nan)
                    self.assertGreater(float(np.nanmin(temp)),290)
                    for v in d.groups['prediction'].variables.values():
                        a=np.ma.filled(v[:],np.nan)
                        self.assertEqual(int(np.isfinite(a).sum()),32)
            legacy={'dataset':dict(dataset,type='aorc_legacy')}
            (p/'legacy.yaml').write_text(yaml.safe_dump(legacy))
            rejected=run(['forest','--','--config',str(p/'legacy.yaml'),'--no-validation','--model-out',str(p/'must_not_exist.joblib')],ok=False)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('Legacy AORC scaling cannot consume',rejected.stderr)
            self.assertFalse((p/'must_not_exist.joblib').exists())


if __name__=='__main__':unittest.main()
