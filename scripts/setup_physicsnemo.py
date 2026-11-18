#!/usr/bin/env python3
"""Install or verify the exact PhysicsNeMo source plus the research patch.

Existing unrelated/dirty checkouts are refused. This script never changes the
original research checkout unless explicitly passed a verified patched tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True,
                            capture_output=capture)
    return result.stdout.strip() if capture else ""


def verify_tree(destination: Path, manifest: dict, patched: bool = True) -> list[str]:
    field = "patched_sha256" if patched else "base_sha256"
    errors = []
    for item in manifest["files"]:
        path = destination / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != item[field]:
            errors.append(item["path"])
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "deps" / "physicsnemo")
    parser.add_argument("--verify-only", action="store_true", help="Read hashes only; no clone, patch, or install")
    parser.add_argument("--no-install", action="store_true", help="Prepare the pinned checkout without pip installation")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "patches" / "physicsnemo-source.json").read_text())
    patch = ROOT / "patches" / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise SystemExit("Patch checksum mismatch; refusing to proceed.")
    destination = args.destination.expanduser().resolve()
    if args.verify_only:
        if (destination / ".git").exists():
            head = run("git", "rev-parse", "HEAD", cwd=destination, capture=True)
            if head != manifest["base_commit"]:
                raise SystemExit("PhysicsNeMo checkout does not match the pinned base commit.")
        errors = verify_tree(destination, manifest)
        if errors:
            raise SystemExit("PhysicsNeMo files differ from the audited environment: " + ", ".join(errors))
        print("All six modified PhysicsNeMo files match the audited source hashes.")
        return
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout",
            manifest["upstream_repository"], str(destination))
        run("git", "checkout", "--detach", manifest["base_commit"], cwd=destination)
    if not (destination / ".git").exists():
        raise SystemExit("Destination exists but is not a Git checkout; refusing to overwrite it.")
    head = run("git", "rev-parse", "HEAD", cwd=destination, capture=True)
    if head != manifest["base_commit"]:
        raise SystemExit("Destination is at a different commit; use a new destination.")
    changed = set(run("git", "diff", "HEAD", "--name-only", cwd=destination, capture=True).splitlines())
    expected_changed = {item["path"] for item in manifest["files"]}
    if changed - expected_changed:
        raise SystemExit("Destination contains changes outside the recorded patch; use a new destination.")
    if verify_tree(destination, manifest):
        if run("git", "status", "--porcelain", cwd=destination, capture=True):
            raise SystemExit("Destination has unrelated local changes; use a new destination.")
        if verify_tree(destination, manifest, patched=False):
            raise SystemExit("Base file checksums differ; refusing to apply the patch.")
        run("git", "apply", "--check", str(patch), cwd=destination)
        run("git", "apply", str(patch), cwd=destination)
    errors = verify_tree(destination, manifest)
    if errors:
        raise SystemExit("Patched source verification failed: " + ", ".join(errors))
    print("Pinned PhysicsNeMo source and research modifications verified.")
    if not args.no_install:
        run(sys.executable, "-m", "pip", "install", "-e", str(destination) + "[launch]")


if __name__ == "__main__":
    main()
