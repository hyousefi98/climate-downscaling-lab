"""Portable configuration and execution front end for preserved experiment code."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from workflows._execution import guard_output, initialize_run

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("aorc", "wrf", "era5_conus404")
PATH_KEYS = {"input_data_path", "target_data_path", "stats_path", "landmask_path",
             "residual_std_path", "regression_checkpoint_path", "reg_ckpt_filename",
             "res_ckpt_filename", "ema_ckpt_filename", "aligned_path", "wrf_input_path"}


def parser(action):
    p = argparse.ArgumentParser(description=f"{action.title()} the paper's U-Net/CorrDiff implementations with explicit data profiles.")
    p.add_argument("--experiment", choices=EXPERIMENTS, default="aorc")
    p.add_argument("--profile", choices=("corrected", "legacy"), default="corrected",
                   help="corrected uses physical units and new matching statistics/checkpoints; legacy uses the archived numerical convention")
    if action == "train": p.add_argument("--stage", choices=("regression", "diffusion"), default="regression")
    p.add_argument("--config", type=Path, help="Override the supplied YAML configuration")
    p.add_argument("--run-dir", type=Path, help="Output directory; required for execution, unnecessary for --dry-run")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override; repeat as needed")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved plan without reading data, importing the model runtime, or writing files")
    p.add_argument("--check-inputs", action="store_true", help="Check external artifact existence and declared source profiles")
    p.add_argument("--allow-invalid-legacy-data", action="store_true", help="Explicit historical replay only: retain original NaN-to-zero handling and contaminated values")
    p.add_argument("--resume", action="store_true", help="Allow reuse of an existing training run directory")
    return p


def absolute_paths(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PATH_KEYS and isinstance(value, str): node[key] = str(Path(value).expanduser().resolve())
            elif isinstance(value, (dict, list)): absolute_paths(value)
    elif isinstance(node, list):
        for item in node: absolute_paths(item)


def configuration(args, action):
    from omegaconf import OmegaConf
    filename = f"train_{args.stage}.yaml" if action == "train" else "generate.yaml"
    path = args.config or ROOT / "configs" / args.experiment / filename
    cfg = OmegaConf.load(path)
    if args.profile == "corrected":
        original = OmegaConf.to_container(cfg.dataset, resolve=False)
        allowed = {"input_data_path", "target_data_path", "stats_path", "input_variables", "output_variables",
                   "train", "train_years", "valid_years", "landmask_path", "coastal_buffer_pixels",
                   "precip_threshold", "keep_ocean_as_nan", "crop_to_multiple"}
        cfg.dataset = {k: v for k, v in original.items() if k in allowed}
        cfg.dataset.type = "physical_pair"
        cfg.dataset.crop_shape = [224, 352] if args.experiment == "era5_conus404" else [288, 480]
        if args.config is None:
            # New training/statistics/inference must share support. Historical
            # AORC inference used 10 pixels despite 5 in the training snapshot.
            cfg.dataset.coastal_buffer_pixels = 5 if args.experiment == "aorc" else 0
        cfg.dataset.stats_path = "${oc.env:DG_ARTIFACT_ROOT,./artifacts}/" + args.experiment + "_physical_stats.json"
        fresh_checkpoint = "${oc.env:DG_CHECKPOINT_ROOT,./checkpoints}/" + args.experiment + "_physical/"
        if action == "train" and args.stage == "diffusion":
            cfg.training.io.regression_checkpoint_path = fresh_checkpoint + "regression.mdlus"
        if action == "generate":
            cfg.generation.io.reg_ckpt_filename = fresh_checkpoint + "regression.mdlus"
            cfg.generation.io.res_ckpt_filename = fresh_checkpoint + "diffusion.mdlus"
        if args.experiment == "aorc" and (action == "generate" or args.stage == "diffusion"):
            section = cfg.generation if action == "generate" else cfg.training.hp
            section.residual_std_path = "${oc.env:DG_ARTIFACT_ROOT,./artifacts}/residual_std_aorc_physical.json"
    if args.allow_invalid_legacy_data and args.profile != "legacy":
        raise ValueError("--allow-invalid-legacy-data is only available with --profile legacy")
    cfg.dataset.quality_policy = "archive_replay" if args.allow_invalid_legacy_data else "strict"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    result = OmegaConf.to_container(cfg, resolve=True)
    absolute_paths(result)
    if args.profile == "corrected" and result["dataset"]["type"] != "physical_pair":
        raise ValueError("Corrected profile requires the strict physical_pair adapter")
    if not args.allow_invalid_legacy_data and result["dataset"].get("quality_policy") != "strict":
        raise ValueError("Changing the quality policy requires --allow-invalid-legacy-data")
    return path, result


def check_inputs(cfg, args, action):
    required = [cfg['dataset'][k] for k in ('input_data_path', 'target_data_path', 'stats_path')]
    if cfg['dataset'].get('landmask_path'): required.append(cfg['dataset']['landmask_path'])
    if action == 'generate':
        mode = cfg['generation']['inference_mode']
        required.append(cfg['generation']['io']['reg_ckpt_filename'])
        if mode != 'regression': required.append(cfg['generation']['io']['res_ckpt_filename'])
        if cfg['generation'].get('residual_std_path') and mode != 'regression': required.append(cfg['generation']['residual_std_path'])
    elif args.stage == 'diffusion':
        required.append(cfg['training']['io']['regression_checkpoint_path'])
        if cfg['training']['hp'].get('residual_std_path'): required.append(cfg['training']['hp']['residual_std_path'])
    missing = [str(x) for x in required if not x or not Path(x).exists()]
    if missing: raise FileNotFoundError('Missing external artifacts: ' + ', '.join(missing))
    import zarr
    for key in ('input_data_path', 'target_data_path'):
        profile = zarr.open_group(cfg['dataset'][key], mode='r').attrs.get('numeric_profile')
        if args.profile == 'corrected' and profile != 'physical':
            raise ValueError(f'{key} requires physical numeric_profile')
        if args.profile == 'legacy' and profile not in (None, 'legacy_packed'):
            raise ValueError(f'{key}: legacy scaling would misinterpret profile {profile!r}')
    if args.profile == 'corrected':
        with open(cfg['dataset']['stats_path']) as handle: metadata = json.load(handle).get('metadata', {})
        if metadata.get('numeric_profile') != 'physical': raise ValueError('Corrected statistics must declare physical numeric_profile')


def main(action):
    args = parser(action).parse_args()
    source, cfg = configuration(args, action)
    if args.check_inputs or not args.dry_run: check_inputs(cfg, args, action)
    plan = {'action': action, 'experiment': args.experiment, 'profile': args.profile,
            'source_config': str(source), 'run_dir': str(args.run_dir.resolve()) if args.run_dir else None,
            'runtime_module': f'legacy.{args.experiment}.{action}', 'configuration': cfg}
    if args.profile == 'legacy':
        plan['limitations'] = ['Historical AORC checkpoints used contaminated training values and a 10× specific-humidity convention.',
                               'WRF paper inference names a deleted v6 diffusion checkpoint directory; its exact training lineage is unverified.',
                               'These preserved source/configuration snapshots are not proof of exact regeneration of every published output.']
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if args.run_dir is None: raise ValueError('--run-dir is required for execution')
    run = guard_output(args.run_dir)
    if action == 'train': cfg['training']['io']['checkpoint_dir'] = str(run)
    else:
        output = Path(cfg['generation']['io']['output_filename'])
        if not output.is_absolute(): output = run / output
        cfg['generation']['io']['output_filename'] = str(guard_output(output))
    cfg['hydra'] = {'job': {'chdir': True, 'name': f'{args.experiment}_{action}'}, 'run': {'dir': str(run)}}
    initialize_run(run, cfg, action, resume=args.resume)
    os.environ.setdefault('DG_CACHE_ROOT', str(run / 'cache'))
    sys.argv = [plan['runtime_module'], '--config-path', str(run), '--config-name', 'resolved']
    module = importlib.import_module(plan['runtime_module'])
    module.main()
