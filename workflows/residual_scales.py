"""Compute external residual RMS scales with the regression checkpoint frozen."""
import argparse
import hashlib
import json
from pathlib import Path
from workflows._execution import guard_output


def main():
    p = argparse.ArgumentParser(description='Compute per-channel RMS of normalized target minus regression prediction, matching the historical residual-scale definition (not centered standard deviation).')
    p.add_argument('--config', required=True, type=Path, help='Resolved training YAML, for example saved by workflows.train')
    p.add_argument('--regression', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-samples', type=int, default=0, help='0 uses every training timestamp')
    args = p.parse_args()
    args.output = guard_output(args.output)
    if args.output.exists(): raise FileExistsError(args.output)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from physicsnemo import Module
    from models.datasets.factory import known_datasets
    from models.datasets.frozen_aorc import file_sha256
    from models.losses_aorc import apply_precip_threshold_normalized
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)['dataset']
    kind = cfg.pop('type'); quality = cfg.pop('quality_policy', 'strict')
    if not cfg.get('train', True): raise ValueError('Residual scales must use the training split')
    dataset = known_datasets[kind](**cfg); dataset.quality_policy = quality
    net = Module.from_checkpoint(str(args.regression), {'amp_mode': False}).to(args.device).eval().requires_grad_(False)
    support = dataset.get_loss_mask()
    if support is None: support = np.ones(dataset.image_shape(), dtype=bool)
    total = np.zeros(len(dataset.output_variables), np.float64)
    squared = total.copy(); count = 0
    n = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
    for i in range(n):
        target, inputs = dataset[i]
        target, inputs = target[None].to(args.device), inputs[None].to(args.device)
        with torch.inference_mode(): pred = net(torch.zeros_like(target), inputs)
        info = dataset.get_precip_threshold_info() if hasattr(dataset, 'get_precip_threshold_info') else None
        pred = apply_precip_threshold_normalized(pred, info)
        target = apply_precip_threshold_normalized(target, info)
        residual = (target - pred)[0].cpu().numpy()[:, support].astype(np.float64)
        if not np.isfinite(residual).all(): raise ValueError('Nonfinite residuals')
        total += residual.sum(axis=1); squared += np.square(residual).sum(axis=1); count += residual.shape[1]
    scale = np.sqrt(squared / count)
    if not np.isfinite(scale).all() or (scale <= 0).any(): raise ValueError('Invalid residual scale')
    result = {'output': dict(zip(dataset.output_variables, scale.tolist())), 'mean': dict(zip(dataset.output_variables, (total / count).tolist())),
              'num_samples': n, 'count_per_channel': count, 'definition': 'RMS normalized residual; no centering',
              'regression_sha256': file_sha256(args.regression), 'statistics_sha256': file_sha256(cfg['stats_path']),
              'numeric_profile': 'physical' if kind == 'physical_pair' else 'historical'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
