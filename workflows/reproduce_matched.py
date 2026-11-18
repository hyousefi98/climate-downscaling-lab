"""Run the November 18 matched comparison through its preserved packed adapter.

Use scripts/preprocess/matched_reference.py to reconstruct the external aligned
file first. This pathway retains historical grid labels and floating-point
transform order; workflows.matched_transfer instead consumes new physical stores.
Neither pathway fits statistics or repairs the historical trained weights.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from workflows._execution import guard_output

ROOT = Path(__file__).resolve().parents[1]


def read_times(path):
    times = json.loads(Path(path).read_text())
    if not isinstance(times, list) or len(times) != 200 or not all(isinstance(t, str) for t in times):
        raise ValueError('--times must contain the exact 200 ISO timestamp strings, in original order')
    if len(set(times)) != 200:
        raise ValueError('Matched timestamps must be unique')
    return times


def make_configs(args, times):
    from omegaconf import OmegaConf
    base = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs/aorc/generate.yaml'), resolve=False)
    configs = {}
    for source in ('synthetic', 'wrf'):
        cfg = copy.deepcopy(base)
        destination = args.run_dir / source
        cfg['dataset'] = {
            'type': 'paper_matched', 'input_source': source,
            'aligned_path': str(args.aligned), 'stats_path': str(args.normalization),
            'wrf_input_path': str(args.wrf_input), 'precip_threshold': .1,
            'coastal_buffer_pixels': 10,
        }
        cfg['generation'].update(times=times, times_range=None, random_samples=None,
                                 num_ensembles=4, seed_batch_size=1,
                                 residual_std_path=str(args.residual_scale))
        cfg['generation']['io'] = {
            'reg_ckpt_filename': str(args.regression), 'res_ckpt_filename': str(args.diffusion),
            'output_filename': str(destination / 'predictions.nc'), 'ema_ckpt_filename': None,
        }
        cfg['hydra'] = {'job': {'chdir': True, 'name': 'paper_matched_' + source},
                        'run': {'dir': str(destination)}}
        configs[source] = cfg
    return configs


def preflight(args, times, configs):
    import netCDF4 as nc
    import numpy as np
    from models.datasets.frozen_aorc import verify_interface
    from legacy.aorc.matched_packed_dataset import AlignedPairedDataset
    verify_interface(args.interface, {name: getattr(args, name) for name in
                     ('normalization', 'residual_scale', 'regression', 'diffusion')})
    with args.normalization.open() as handle:
        statistics = json.load(handle)
    if statistics.get('metadata', {}).get('numeric_profile') == 'physical':
        raise ValueError('New physical statistics cannot replace the frozen historical normalization')
    with nc.Dataset(args.aligned) as source:
        t = source['time']
        actual = [d.strftime('%Y-%m-%dT%H:%M:%S') for d in
                  nc.num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))]
        if actual != times:
            raise ValueError('Aligned file times differ from the requested 200 timestamps or their order')
    # The preserved adapter validates exact grid, time, and support identity.
    # Both inputs and the target must be finite before either inference starts.
    adapters = []
    try:
        for source in ('synthetic', 'wrf'):
            adapters.append(AlignedPairedDataset(**{k: v for k, v in configs[source]['dataset'].items() if k != 'type'}))
        if not np.array_equal(adapters[0].loss_mask, adapters[1].loss_mask):
            raise ValueError('The two arms have different verification support')
        for index in range(200):
            left, _ = adapters[0][index]
            right, _ = adapters[1][index]
            if not np.array_equal(left.numpy(), right.numpy(), equal_nan=True):
                raise ValueError(f'The two arms have different target values at index {index}')
        return {'passed': True, 'cases': 200, 'valid_cells': int(adapters[0].loss_mask.sum()),
                'input_target_finite_on_support': True, 'shared_target_exact': True}
    finally:
        for adapter in adapters:
            adapter.close()
        adapters.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aligned', 'wrf_input', 'times', 'normalization', 'residual_scale',
                 'regression', 'diffusion', 'interface', 'run_dir'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without reading artifacts or launching inference')
    parser.add_argument('--validate-only', action='store_true', help='Run the full CPU input/reference preflight; do not launch inference')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('The paired launcher requires one process; select its GPU with CUDA_VISIBLE_DEVICES')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    plan = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    plan.update(profile='paper_matched_2026_11_18', cases=200, ensembles=4, seeds=[0, 1, 2, 3],
                sampler_steps=18, output_humidity='10 times specific humidity; divide by 10 for kg/kg metrics',
                output_pressure='hPa', reference_policy='one shared ending-window AORC target',
                limitations=['Historical weights retain the original training-data defects.',
                             'WRF Q2 is interpreted as standard mixing ratio; original conversion history is incomplete.',
                             'WRF wind rotation and nonprecipitation temporal aggregation provenance remain unverified.'])
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    for name, value in vars(args).items():
        if isinstance(value, Path) and name != 'run_dir' and not value.exists():
            raise FileNotFoundError(value)
    guard_output(args.run_dir)
    if args.run_dir.exists() and (not args.run_dir.is_dir() or any(args.run_dir.iterdir())):
        raise FileExistsError('Choose an empty run directory')
    times = read_times(args.times)
    configs = make_configs(args, times)
    result = preflight(args, times, configs)
    if args.validate_only:
        print(json.dumps(dict(plan, preflight=result), indent=2)); return
    from omegaconf import OmegaConf
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'preflight.json').write_text(json.dumps(dict(plan, preflight=result), indent=2) + '\n')
    for source, cfg in configs.items():
        destination = args.run_dir / source
        destination.mkdir(exist_ok=False)
        config_path = destination / 'resolved.yaml'
        OmegaConf.save(OmegaConf.create(cfg), config_path)
        environment = {k: v for k, v in os.environ.items() if not k.startswith('CORRDIFF_GEN_DEBUG')}
        environment.update(DG_CACHE_ROOT=str(destination / 'cache'), WANDB_MODE='disabled')
        subprocess.run([sys.executable, '-m', 'workflows._run_config', '--config', str(config_path)],
                       check=True, env=environment)


if __name__ == '__main__':
    main()
