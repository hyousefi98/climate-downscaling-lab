"""Compare two physical input sources with one reference and frozen AORC models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from workflows._execution import guard_output

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description='Matched synthetic/WRF inference with common reference, times, mask, seeds and frozen artifact hashes. No retraining or normalization fitting.')
    for name in ('synthetic_input', 'wrf_input', 'reference', 'times', 'normalization', 'residual_scale', 'regression', 'diffusion', 'interface', 'landmask', 'run_dir'):
        p.add_argument('--' + name.replace('_', '-'), required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('The paired launcher requires one process; select its GPU with CUDA_VISIBLE_DEVICES')
    plan = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    plan.update(ensembles=4, seeds=[0, 1, 2, 3], model_coordinate='10 times specific humidity; pressure hPa', output_coordinate='physical specific humidity; pressure hPa', reference_policy='one shared AORC target', sampler_steps=18)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    from omegaconf import OmegaConf
    from models.datasets.frozen_aorc import FrozenAORCPairDataset, verify_interface
    verify_interface(args.interface, {name: getattr(args, name) for name in ('normalization', 'residual_scale', 'regression', 'diffusion')})
    with args.times.open() as handle: times = json.load(handle)
    if not isinstance(times, list) or not times: raise ValueError('--times must contain a nonempty JSON list of ISO timestamps')
    run = guard_output(args.run_dir)
    if run.exists() and any(run.iterdir()): raise FileExistsError('Choose an empty run directory')
    base = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs/aorc/generate.yaml'), resolve=False)
    dataset_common = {'type': 'frozen_aorc', 'target_data_path': str(args.reference.resolve()), 'stats_path': str(args.normalization.resolve()),
                      'checkpoint_interface_path': str(args.interface.resolve()), 'landmask_path': str(args.landmask.resolve()),
                      'train': False, 'valid_years': [2012, 2012], 'crop_shape': [288, 480],
                      'coastal_buffer_pixels': 10, 'precip_threshold': .1, 'keep_ocean_as_nan': True,
                      'requested_times': times, 'shared_support_paths': [str(args.synthetic_input.resolve()), str(args.wrf_input.resolve())]}
    configs, adapters = {}, []
    for source, path in [('synthetic', args.synthetic_input), ('wrf', args.wrf_input)]:
        cfg = json.loads(json.dumps(base))
        cfg['dataset'] = dict(dataset_common, input_source=source, input_data_path=str(path.resolve()))
        adapters.append(FrozenAORCPairDataset(**{k: v for k, v in cfg['dataset'].items() if k != 'type'}))
        destination = run / source
        cfg['generation'].update(num_ensembles=4, seed_batch_size=1, random_samples=None, times=times,
                                 inference_mode='all', hr_mean_conditioning=True, residual_std_path=str(args.residual_scale.resolve()))
        cfg['generation']['io'] = {'reg_ckpt_filename': str(args.regression.resolve()), 'res_ckpt_filename': str(args.diffusion.resolve()),
                                   'output_filename': str(destination / 'predictions.nc'), 'ema_ckpt_filename': None}
        cfg['hydra'] = {'job': {'chdir': True, 'name': 'matched_' + source}, 'run': {'dir': str(destination)}}
        configs[source] = cfg
    import numpy as np
    if adapters[0].time() != adapters[1].time() or not np.array_equal(adapters[0].loss_mask, adapters[1].loss_mask):
        raise ValueError('The two inference conditions do not have identical times and support')
    for source, cfg in configs.items():
        destination = run / source
        destination.mkdir(parents=True, exist_ok=False)
        path = destination / 'resolved.yaml'
        OmegaConf.save(OmegaConf.create(cfg), path)
        environment = dict(os.environ, DG_CACHE_ROOT=str(destination / 'cache'))
        subprocess.run([sys.executable, '-m', 'workflows._run_config', '--config', str(path)], check=True, env=environment)


if __name__ == '__main__': main()
