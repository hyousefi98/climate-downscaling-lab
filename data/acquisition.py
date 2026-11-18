"""Bounded spatial/time selection and checksum-verified source ingestion."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np


def subset(source, *, start, end, bbox, latitude="latitude", longitude="longitude"):
    """[start,end) selection; bbox west,south,east,north; no zero fill."""
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Expected non-wrapping west,south,east,north bounds")
    if not np.issubdtype(source.time.dtype, np.datetime64):
        raise ValueError("Expected decoded datetime64 time coordinate")
    if np.datetime64(start) >= np.datetime64(end):
        raise ValueError("start must precede end")
    source = source.sel(time=(source.time >= np.datetime64(start)) & (source.time < np.datetime64(end)))
    lat, lon = source[latitude], source[longitude]
    if lat.ndim == lon.ndim == 1:
        source = source.isel({lat.dims[0]: np.flatnonzero((lat >= south) & (lat <= north)),
                              lon.dims[0]: np.flatnonzero((lon >= west) & (lon <= east))})
    elif lat.ndim == lon.ndim == 2 and lat.dims == lon.dims:
        inside = np.asarray((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))
        rows, cols = np.where(inside)
        if not len(rows):
            raise ValueError("Bounding box does not overlap source")
        source = source.isel({lat.dims[0]: slice(rows.min(), rows.max()+1), lat.dims[1]: slice(cols.min(), cols.max()+1)})
    else:
        raise ValueError("Latitude and longitude must be paired 1D or 2D coordinates")
    if any(size == 0 for size in source.sizes.values()):
        raise ValueError("Requested subset is empty")
    source.attrs.update(request_start_inclusive=start, request_end_exclusive=end,
                        request_bbox_wsen=json.dumps(list(bbox)))
    return source


def ingest_manifest(manifest, destination, *, dry_run=False):
    """Copy supplied local files or HTTPS URLs, verifying mandatory SHA256.

    Each record contains source, relative destination path, sha256. Directories
    (including Zarr) are represented by individual files: no opaque recursive
    copy, archive extraction, private endpoint guessing, or overwritten output.
    """
    import requests
    root = Path(destination).expanduser().resolve()
    entries = json.loads(Path(manifest).read_text())
    paths = set()
    for entry in entries:
        relative = Path(entry["path"])
        target = (root / relative).resolve()
        digest = entry["sha256"].lower()
        if relative.is_absolute() or root not in target.parents or target in paths:
            raise ValueError("Manifest paths must be unique relative descendants")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Each entry requires a SHA256 digest")
        if target.exists() or target.with_name(target.name+".partial").exists():
            raise FileExistsError(target)
        paths.add(target)
    if dry_run:
        return {"files": len(entries), "destination": str(root), "sha256_required": True}
    for entry in entries:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name+".partial")
        source = entry["source"]
        parsed = urlparse(source)
        sha = hashlib.sha256()
        if parsed.scheme not in ("", "https"):
            raise ValueError("Use local paths or HTTPS URLs only")
        with partial.open("xb") as handle:
            if parsed.scheme == "https":
                with requests.get(source, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024*1024):
                        handle.write(chunk); sha.update(chunk)
            else:
                with Path(source).expanduser().open("rb") as stream:
                    while chunk := stream.read(1024*1024):
                        handle.write(chunk); sha.update(chunk)
        if sha.hexdigest() != entry["sha256"].lower():
            raise ValueError(f"Checksum mismatch for {entry['path']}; partial retained")
        partial.rename(target)
    return {"verified_files": len(entries), "destination": str(root)}
