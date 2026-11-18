"""Bind external historical checkpoint artifacts without copying them."""
import argparse
import json
from pathlib import Path
from workflows._execution import guard_output


def main():
    p = argparse.ArgumentParser(description="Declare and hash the historical AORC 10q/hPa checkpoint interface. This declaration does not establish training provenance.")
    for name in ('regression', 'diffusion', 'normalization', 'residual_scale'):
        p.add_argument('--' + name.replace('_', '-'), required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output = guard_output(args.output)
    from models.datasets.frozen_aorc import file_sha256
    if args.output.exists(): raise FileExistsError(args.output)
    names = ('regression', 'diffusion', 'normalization', 'residual_scale')
    result = {'model_coordinate': 'aorc_legacy_10q_hpa',
              'declaration': 'User-declared historical AORC interface; hashes bind artifact identity, not training validity.',
              'sha256': {name: file_sha256(getattr(args, name)) for name in names},
              'normalization': 'Keep the original normalization and residual RMS scales with these checkpoints.',
              'limitation': 'Historical training included known source missing-value contamination. This interface does not repair trained weights.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__': main()
