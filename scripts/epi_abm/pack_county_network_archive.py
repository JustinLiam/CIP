"""Pack one county's EpiABM mobility pickles into contiguous tensor storage."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch


ARCHIVE_NAME = "tensor_archive_v1.pt"
FORMAT_VERSION = 1
NETWORK_DIRS = {
    "school": "schoolnets",
    "occ": "occnets",
    "rand": "randnets",
}


def _read_edges(path: Path) -> torch.Tensor:
    value = pd.read_pickle(path)
    if hasattr(value, "edges"):
        value = pd.DataFrame(value.edges(), columns=["node1", "node2"])
    edges = torch.tensor(value.values, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{path} produced invalid edge shape {tuple(edges.shape)}")
    return edges.contiguous()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _pack_sequence(
    paths: Iterable[Path],
    *,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    paths = list(paths)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tensors = list(pool.map(_read_edges, paths))
    else:
        tensors = [_read_edges(path) for path in paths]
    offsets = [0]
    for edges in tensors:
        offsets.append(offsets[-1] + int(edges.shape[0]))
    if tensors:
        packed = torch.cat(tensors, dim=0).contiguous()
    else:
        packed = torch.empty((0, 2), dtype=torch.long)
    return packed, torch.tensor(offsets, dtype=torch.long)


def _fingerprints(payload: Dict[str, object]) -> Dict[str, str]:
    keys = [
        "school_edges",
        "school_offsets",
        "occ_edges",
        "occ_offsets",
        "rand_edges",
        "rand_offsets",
        "household_edges",
    ]
    return {key: _tensor_digest(payload[key]) for key in keys}


def _validate_payload(payload: Dict[str, object], county: str, num_steps: int) -> None:
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"Unexpected archive format: {payload.get('format_version')!r}")
    if str(payload.get("county", "")).zfill(5) != county.zfill(5):
        raise ValueError(
            f"Archive county mismatch: {payload.get('county')!r} != {county!r}"
        )
    if int(payload.get("num_steps", -1)) != int(num_steps):
        raise ValueError(
            f"Archive num_steps mismatch: {payload.get('num_steps')!r} != {num_steps!r}"
        )
    for kind in NETWORK_DIRS:
        edges = payload[f"{kind}_edges"]
        offsets = payload[f"{kind}_offsets"]
        if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long:
            raise ValueError(f"{kind}_edges is not an int64 tensor")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"{kind}_edges has shape {tuple(edges.shape)}")
        if not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.long:
            raise ValueError(f"{kind}_offsets is not an int64 tensor")
        if offsets.tolist()[0] != 0 or offsets.tolist()[-1] != edges.shape[0]:
            raise ValueError(f"{kind}_offsets do not cover the packed edge tensor")
        if len(offsets) != int(num_steps) + 1:
            raise ValueError(
                f"{kind}_offsets has {len(offsets)} entries, expected {num_steps + 1}"
            )
    household = payload["household_edges"]
    if (
        not isinstance(household, torch.Tensor)
        or household.dtype != torch.long
        or household.ndim != 2
        or household.shape[1] != 2
    ):
        raise ValueError("household_edges is not an [N, 2] int64 tensor")


def pack_county(
    *,
    epi_root: Path,
    county: str,
    num_steps: int,
    force: bool,
    verify: bool,
    workers: int,
) -> dict:
    county = str(county).zfill(5)
    mobility_dir = (
        epi_root
        / "data"
        / "networks"
        / "covid_output_causal"
        / county
        / "mobility_networks"
    )
    if not mobility_dir.is_dir():
        raise FileNotFoundError(f"Missing county mobility directory: {mobility_dir}")

    archive_path = mobility_dir / ARCHIVE_NAME
    lock_path = mobility_dir / f".{ARCHIVE_NAME}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if archive_path.exists() and not force:
            payload = torch.load(archive_path, map_location="cpu")
            _validate_payload(payload, county, num_steps)
            if verify:
                actual = _fingerprints(payload)
                if actual != payload.get("tensor_sha256"):
                    raise ValueError(f"Archive fingerprint mismatch: {archive_path}")
            return {
                "event": "epi_abm_network_archive_reused",
                "county": county,
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "edge_counts": {
                    kind: int(payload[f"{kind}_edges"].shape[0])
                    for kind in NETWORK_DIRS
                },
                "household_edges": int(payload["household_edges"].shape[0]),
            }

        started = time.time()
        payload: Dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "county": county,
            "num_steps": int(num_steps),
        }
        source_files = []
        for kind, dirname in NETWORK_DIRS.items():
            paths = [mobility_dir / dirname / f"{step}.pkl" for step in range(num_steps)]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} {kind} network files; first={missing[0]}"
                )
            edges, offsets = _pack_sequence(paths, workers=workers)
            payload[f"{kind}_edges"] = edges
            payload[f"{kind}_offsets"] = offsets
            source_files.extend(paths)

        household_path = mobility_dir / "HOUSEHOLD_NETWORK.pkl"
        if not household_path.is_file():
            raise FileNotFoundError(f"Missing household network: {household_path}")
        payload["household_edges"] = _read_edges(household_path)
        source_files.append(household_path)
        payload["source_file_count"] = len(source_files)
        payload["source_total_bytes"] = sum(path.stat().st_size for path in source_files)
        payload["tensor_sha256"] = _fingerprints(payload)

        tmp_path = archive_path.with_name(
            f".{archive_path.name}.tmp.{os.getpid()}"
        )
        try:
            torch.save(payload, tmp_path)
            with tmp_path.open("rb") as tmp_file:
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, archive_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if verify:
            loaded = torch.load(archive_path, map_location="cpu")
            _validate_payload(loaded, county, num_steps)
            actual = _fingerprints(loaded)
            if actual != payload["tensor_sha256"]:
                raise ValueError(f"Post-write fingerprint mismatch: {archive_path}")

        return {
            "event": "epi_abm_network_archive_created",
            "county": county,
            "archive": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "source_file_count": len(source_files),
            "source_total_bytes": int(payload["source_total_bytes"]),
            "edge_counts": {
                kind: int(payload[f"{kind}_edges"].shape[0])
                for kind in NETWORK_DIRS
            },
            "household_edges": int(payload["household_edges"].shape[0]),
            "elapsed_sec": round(time.time() - started, 3),
            "verified": bool(verify),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epi-root", default="data_generation/epi_diff_abm")
    parser.add_argument("--counties", nargs="+", required=True)
    parser.add_argument("--num-steps", type=int, default=182)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--county-workers",
        type=int,
        default=1,
        help="Pack this many counties in isolated processes so intermediate memory is released.",
    )
    args = parser.parse_args()

    epi_root = Path(args.epi_root).expanduser().resolve()
    kwargs = [
        {
            "epi_root": epi_root,
            "county": county,
            "num_steps": int(args.num_steps),
            "force": bool(args.force),
            "verify": not bool(args.no_verify),
            "workers": max(1, int(args.workers)),
        }
        for county in args.counties
    ]
    county_workers = max(1, int(args.county_workers))
    if county_workers == 1:
        for item in kwargs:
            print(json.dumps(pack_county(**item), sort_keys=True), flush=True)
        return

    with ProcessPoolExecutor(max_workers=county_workers) as pool:
        futures = {
            pool.submit(pack_county, **item): item["county"]
            for item in kwargs
        }
        for future in as_completed(futures):
            county = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to pack county={county}") from exc
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
