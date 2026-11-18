#!/usr/bin/env python3
"""Recreate the current paper's matched archived-grid inputs and AORC reference.

Uses external original hourly spatially processed stores, baseline inference
NetCDF, and frozen statistics. It does not download data, change grid labels,
retrain models, alter the mask, or apply the new native-data preprocessing path.
"""
import argparse
from data.matched_reference import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('synthetic-hourly','target-hourly','baseline-netcdf','statistics','output','reference-output','audit-output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--times-output',help='Optional new JSON file of baseline timestamps for workflows.reproduce_matched --times')
    a=p.parse_args()
    prepare(a.synthetic_hourly,a.target_hourly,a.baseline_netcdf,a.statistics,a.output,a.reference_output,a.audit_output,a.times_output)


if __name__=='__main__':main()
