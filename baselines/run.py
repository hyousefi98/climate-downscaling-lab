"""Run a historical baseline entry point without assuming a repository location.

python -m baselines.run ridge -- --train-config YOUR_TRAIN.yaml --gen-config YOUR_GEN.yaml
python -m baselines.run forest -- --config YOUR_TRAIN.yaml --model-out YOUR_MODEL.joblib
python -m baselines.run predict-forest -- --help

These guarded historical commands require the pinned PhysicsNeMo/PyTorch stack.
Defaults select configs/baselines/*_physical.yaml and the shared physical_pair
loader. Archived packed-number data require an explicit aorc_legacy configuration;
that loader rejects physical stores and normalization statistics.
New RF models persist training climatology and feature metadata; inference restores
them. Old climatology refitting is disabled unless explicitly requested. See
baselines/source_manifest.json for all intentional changes to archived code.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ENTRY_POINTS = {
    "ridge": "generate_multivariate_baseline_nc.py",
    "forest": "RF_3km.py",
    "predict-forest": "generate_rf_nc.py",
    "linear": "generate_linear_baseline_nc.py",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=ENTRY_POINTS)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a=p.parse_args()
    args=a.arguments[1:] if a.arguments[:1]==["--"] else a.arguments
    script=Path(__file__).resolve().parent / "historical" / ENTRY_POINTS[a.method]
    raise SystemExit(subprocess.call([sys.executable,str(script),*args]))


if __name__=="__main__":
    main()
