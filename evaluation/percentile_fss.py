"""Training-climatology FSS for two gridded precipitation products.

Inputs are grouped NetCDF files in physical mm per six-hour interval and an
explicit boolean .npy mask on their common grid. Training quantiles include zeros,
use exact linear order statistics, and are held fixed for evaluation. Positive
values are streamed to temporary scratch files rather than loading the full
training archive into RAM. This cleaned interface does not repeat historical
NaN-to-zero or normalization-roundtrip preprocessing: provide the intended fields
explicitly, and record their provenance. Native temporal support must be aligned
before this command; equal labels alone do not prove equal accumulation windows.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import tempfile

import netCDF4
import numpy as np

from .io import timestamps, time_slice, validate_grid
from .spatial import exceedance, fractions


def exact_quantiles(chunks, probabilities, scratch):
    """Exact zero-inclusive quantiles of nonnegative finite chunks."""
    probabilities=np.asarray(probabilities,dtype=float)
    if not len(probabilities) or np.any(~np.isfinite(probabilities)) or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError("Quantiles must be in [0,1]")
    scratch=Path(scratch)
    if scratch.exists():raise FileExistsError(scratch)
    total=zeros=0
    try:
        with scratch.open('xb') as f:
            for chunk in chunks:
                x=np.asarray(chunk,dtype=np.float64).ravel()
                if not np.isfinite(x).all() or (x<0).any():raise ValueError("Invalid training rainfall")
                total+=len(x);zeros+=int((x==0).sum())
                x[x>0].tofile(f)
        if not total:raise ValueError("Empty quantile training sample")
        positions=(total-1)*probabilities
        low=np.floor(positions).astype(np.int64);high=np.ceil(positions).astype(np.int64)
        wanted=sorted(set(int(x-zeros) for x in np.r_[low,high] if x>=zeros))
        values=np.memmap(scratch,dtype='float64',mode='r+',shape=(total-zeros,)) if total>zeros else None
        try:
            if wanted:values.partition(wanted)
            result=[]
            for q,pos,lo,hi in zip(probabilities,positions,low,high):
                a=0. if lo<zeros else float(values[lo-zeros]);b=0. if hi<zeros else float(values[hi-zeros])
                result.append({'quantile':float(q),'threshold_mm':a+(b-a)*(pos-lo),'count':total,'zero_count':zeros})
            return result
        finally:
            if values is not None:del values
    finally:
        scratch.unlink(missing_ok=True)


def compare(training_predictor,training_reference,evaluation_predictor,evaluation_reference,
            mask_path,output_dir,*,predictor_variable,reference_variable,
            predictor_group='input',reference_group='truth',quantiles=(.95,.999),
            windows=(1,3,5,9,15,25,41),absolute_thresholds=(20.,),scratch_dir=None):
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError("Choose a new output directory")
    if not windows or len(set(windows))!=len(windows) or any(not isinstance(w,(int,np.integer)) or w<1 or w%2!=1 for w in windows):
        raise ValueError("Windows must be distinct positive odd integers")
    if len(set(quantiles))!=len(quantiles):raise ValueError("Quantiles must be distinct")
    if len(set(absolute_thresholds))!=len(absolute_thresholds) or any(not np.isfinite(x) or x<0 for x in absolute_thresholds):
        raise ValueError("Absolute thresholds must be distinct finite nonnegative values")
    mask=np.load(mask_path,allow_pickle=False)
    if mask.dtype!=bool or mask.ndim!=2 or not mask.any():raise ValueError("Mask must be a nonempty 2D boolean array")
    with ExitStack() as stack:
        trp,trr,ep,er=[stack.enter_context(netCDF4.Dataset(x)) for x in
                      [training_predictor,training_reference,evaluation_predictor,evaluation_reference]]
        validate_grid(trr,trp);validate_grid(er,ep)
        calendars=[getattr(d['time'],'calendar','standard') for d in [trr,er]]
        calendars=['standard' if c=='gregorian' else c for c in calendars]
        if calendars[0]!=calendars[1]:raise ValueError("Training/evaluation calendars differ")
        training_times=timestamps(trr);evaluation_times=timestamps(er)
        if not training_times or not evaluation_times or max(training_times)>=min(evaluation_times):
            raise ValueError("Training must strictly precede evaluation")
        for d in [trr,trp,er,ep]:
            if d['lat'].shape!=mask.shape:raise ValueError("Mask grid differs from data")
        for coord in ['lat','lon']:
            if not np.array_equal(trr[coord][:],er[coord][:]):raise ValueError("Training/evaluation grids differ")
        def field(d,g,v,t):
            a=time_slice(d.groups[g][v],t)
            if a.shape!=(1,*mask.shape) or not np.isfinite(a[0,mask]).all() or (a[0,mask]<0).any():
                raise ValueError("Expected finite nonnegative deterministic rainfall on fixed mask")
            return a[0]
        with tempfile.TemporaryDirectory(dir=scratch_dir,prefix='training-quantiles-') as tmp:
            qpred=exact_quantiles((field(trp,predictor_group,predictor_variable,t)[mask] for t in range(len(training_times))),quantiles,Path(tmp)/'predictor.f64')
            qref=exact_quantiles((field(trr,reference_group,reference_variable,t)[mask] for t in range(len(training_times))),quantiles,Path(tmp)/'reference.f64')
        if any(x['threshold_mm']<=0 for x in [*qpred,*qref]):
            raise ValueError("A fitted rainfall quantile is zero: inclusive >= events would include every dry cell. Choose a higher quantile rather than report degenerate percentile FSS.")
        cases=[(f'absolute_{x:g}mm',float(x),float(x)) for x in absolute_thresholds]
        cases += [(f'quantile_{a["quantile"]:g}',a['threshold_mm'],b['threshold_mm']) for a,b in zip(qpred,qref)]
        if not cases:raise ValueError("No thresholds requested")
        totals={(name,w):[0.,0.] for name,_,_ in cases for w in windows}
        for t in range(len(evaluation_times)):
            o=field(er,reference_group,reference_variable,t);p=field(ep,predictor_group,predictor_variable,t)
            for name,pt,ot in cases:
                oe,pe=exceedance(o,ot),exceedance(p,pt)
                for w in windows:
                    of,ov=fractions(oe,mask,w,'valid','expanded');pf,pv=fractions(pe,mask,w,'valid','expanded')
                    use=ov&pv
                    totals[name,w][0]+=float(np.square(pf[use]-of[use]).sum())
                    totals[name,w][1]+=float((np.square(pf[use])+np.square(of[use])).sum())
        rows=[]
        for name,pt,ot in cases:
            for w in windows:
                num,den=totals[name,w]
                rows.append({'experiment':name,'window_cells':w,'predictor_threshold_mm':pt,'reference_threshold_mm':ot,
                             'FSS':1-num/den if den else None,'numerator':num,'denominator':den})
        audit={'training_predictor':str(training_predictor),'training_reference':str(training_reference),
               'evaluation_predictor':str(evaluation_predictor),'evaluation_reference':str(evaluation_reference),
               'training_first':min(training_times),'training_last':max(training_times),
               'evaluation_first':min(evaluation_times),'evaluation_last':max(evaluation_times),
               'training_count':len(training_times),'evaluation_count':len(evaluation_times),'mask_cells':int(mask.sum()),
               'predictor_quantiles':qpred,'reference_quantiles':qref,
               'quantile_definition':'Exact linear order statistics over pooled valid values including zeros',
               'FSS_definition':'Pooled time-space ratio; valid-support normalization; expanded valid neighborhood centers; float64 calculations',
               'interpretation':'Product-relative percentiles do not represent the same rainfall amount or establish a causal reason for errors.'}
    output_dir.mkdir(parents=True)
    with (output_dir/'fss.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output_dir/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['training-predictor','training-reference','evaluation-predictor','evaluation-reference','mask','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--predictor-variable',required=True);p.add_argument('--reference-variable',required=True)
    p.add_argument('--predictor-group',default='input');p.add_argument('--reference-group',default='truth')
    p.add_argument('--quantiles',type=float,nargs='+',default=[.95,.999])
    p.add_argument('--windows',type=int,nargs='+',default=[1,3,5,9,15,25,41])
    p.add_argument('--absolute-thresholds',type=float,nargs='*',default=[20.])
    p.add_argument('--scratch-dir',type=Path)
    a=vars(p.parse_args());a['mask_path']=a.pop('mask')
    compare(**a)


if __name__=='__main__':main()
