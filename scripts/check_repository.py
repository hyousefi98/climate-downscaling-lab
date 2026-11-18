#!/usr/bin/env python3
"""Check tracked/staged files for accidental data, results, and credentials.

Does not print secret contents. Intended for local verification and CI.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".nc", ".nc4", ".h5", ".hdf5", ".pt", ".pth", ".ckpt",
                      ".mdlus", ".pkl", ".pickle", ".joblib", ".npy", ".npz",
                      ".csv", ".parquet", ".png", ".jpg", ".jpeg", ".pdf",
                      ".docx", ".xlsx", ".mp4", ".log", ".zip", ".tgz"}
SECRET_PATTERNS = [
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?:https?://)[^\s/@:]+:[^\s/@]+@"),
]


def inspect(paths: list[Path]) -> list[str]:
    problems = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            problems.append(f"Symlink is not portable: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(p.endswith(".zarr") for p in relative.parts):
            problems.append(f"Data/result artifact: {relative}")
        if path.name in {".env", ".netrc", ".cdsapirc", "credentials.json"}:
            problems.append(f"Credential file: {relative}")
        if path.stat().st_size > 5_000_000:
            problems.append(f"Unexpected large file: {relative}")
        payload = path.read_bytes()
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            problems.append(f"Possible secret (contents withheld): {relative}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if args.staged else ["git", "ls-files", "-z"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    names = [n.decode() for n in result.stdout.split(b"\0") if n]
    if not names:
        raise SystemExit("No tracked files to check. Stage the intended source files first.")
    problems = inspect([ROOT / name for name in names])
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(f"Checked {len(names)} source/configuration/documentation files; no result artifacts or matching secret patterns.")


if __name__ == "__main__":
    main()
