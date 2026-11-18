"""Internal process entry point for an already validated inference configuration."""
import argparse
import importlib
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    config = args.config.resolve()
    sys.argv = ['matched_aorc_generate', '--config-path', str(config.parent), '--config-name', config.stem]
    importlib.import_module('legacy.aorc.generate').main()


if __name__ == '__main__': main()
