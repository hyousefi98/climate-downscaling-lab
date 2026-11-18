#!/usr/bin/env python3
"""Ingest privately supplied WRF files from an explicit checksum manifest.

The paper's simulations were supplied by Hsin-I Chang. A public archive and the
original processing writer have not been verified; this command invents neither.
"""
import argparse
import json
from data.acquisition import ingest_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,help='JSON list of {source,path,sha256}; local paths or HTTPS')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    print(json.dumps(ingest_manifest(a.manifest,a.output_dir,dry_run=a.dry_run),indent=2))


if __name__=='__main__':main()
