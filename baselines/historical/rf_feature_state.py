"""Persist RF training features so inference never fits held-out climatology.

Only trusted, locally produced model files should be loaded with joblib. Feature
state is embedded in new estimators; no model or climatology is distributed here.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


FLAGS=('add_coords','add_latlon','add_landmask','add_climatology','include_lead_time','use_physical')


def fingerprint(dataset,cfg):
    """Bind feature state to variable order, grid, mask policy and normalization."""
    variables={k:list(getattr(dataset,k)) for k in ('input_variables','output_variables')}
    policy={k:getattr(dataset,k,cfg.get(k)) for k in ('apply_scale_factors','clip_negative','coastal_buffer_pixels','mask_ocean_in_target','keep_ocean_as_nan','sanitize_inputs','precip_threshold')}
    policy['dataset_class']=type(dataset).__module__+'.'+type(dataset).__name__
    h=hashlib.sha256(json.dumps({'variables':variables,'policy':policy},sort_keys=True).encode())
    stats_path=cfg.get('stats_path')
    if stats_path:
        stats=json.loads(Path(stats_path).read_text())
        h.update(json.dumps(stats,sort_keys=True,separators=(',',':')).encode())
    for value in (dataset.latitude(),dataset.longitude()):
        a=np.asarray(value,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
    mask=getattr(dataset,'land_mask',None)
    if mask is not None:
        a=np.asarray(mask,dtype=bool);h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def normalized_times(times):
    """Naive UTC Gregorian labels; the whole supplied period is checked."""
    if times is None or len(times)==0:raise ValueError('Training and evaluation times must be explicit')
    for t in times:
        if getattr(t,'calendar','standard') not in ('standard','gregorian','proleptic_gregorian'):
            raise ValueError('RF feature-state times require a Gregorian calendar')
        if getattr(t,'tzinfo',None) is not None or re.search(r'(?:Z|[+-]\d{2}:\d{2})$',str(t)):
            raise ValueError('Convert RF feature-state timestamps explicitly to naive UTC')
    a=np.asarray([np.datetime64(str(t),'us') for t in times])
    if np.isnat(a).any():raise ValueError('Invalid feature-state time')
    return a


def make_state(args,dataset,cfg,normals,times,feature_count):
    t=normalized_times(times)
    if not cfg.get('train',True):raise ValueError('Feature state must be fitted on the training dataset')
    n=None if normals is None else np.asarray(normals,dtype=np.float32).copy()
    if args.add_climatology and (n is None or n.shape!=(367,len(dataset.input_variables)) or not np.isfinite(n).all()):
        raise ValueError('Training climatology must be finite with one row per day-of-year')
    return {'version':1,'flags':{k:bool(getattr(args,k)) for k in FLAGS},
            'lead_time_scale':float(args.lead_time_scale),'fingerprint':fingerprint(dataset,cfg),
            'input_variables':list(dataset.input_variables),'output_variables':list(dataset.output_variables),
            'training_first':str(t.min()),'training_last':str(t.max()),
            'feature_count':int(feature_count),'climatology':n}


def restore_state(model,args,dataset,cfg,times):
    """Restore saved training features, rejecting overlap and schema mismatches.

    Returns None only for old models without state. The caller must explicitly
    authorize legacy climatology recomputation in that case.
    """
    state=getattr(model,'downscaling_feature_state_',None)
    if state is None:return None
    if state.get('version')!=1:raise ValueError('Unsupported RF feature state version')
    if fingerprint(dataset,cfg)!=state['fingerprint']:raise ValueError('RF feature grid, variables, mask policy or normalization differ from training')
    t=normalized_times(times)
    if t.min()<=np.datetime64(state['training_last']):raise ValueError('RF evaluation must follow the saved training period')
    if int(model.n_features_in_)!=state['feature_count']:raise ValueError('Estimator feature count differs from saved schema')
    for key,value in state['flags'].items():
        if getattr(args,key) and not value:raise ValueError(f'Requested {key} was not used during training')
    n=state['climatology']
    if state['flags']['add_climatology']:
        n=np.asarray(n,dtype=np.float32)
        if n.shape!=(367,len(dataset.input_variables)) or not np.isfinite(n).all():raise ValueError('Invalid saved training climatology')
        n=n.copy()
    for key,value in state['flags'].items():setattr(args,key,value)
    args.lead_time_scale=state['lead_time_scale']
    return n


def check_feature_count(model,features):
    if features.ndim!=2 or features.shape[1]!=int(model.n_features_in_):
        raise ValueError(f'Feature schema mismatch: model expects {model.n_features_in_} columns, constructed shape {features.shape}. Supply the training feature flags; no prediction was run.')
    if not np.isfinite(features).all():raise ValueError('Nonfinite RF feature values')
