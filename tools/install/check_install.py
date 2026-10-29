#!/usr/bin/env python3
"""Check a FollowBench 2.0 install: files, checksums and the Python env.

Every file the benchmark needs is listed in tools/install/files.tsv with its
size and SHA256. This script compares the files on disk against that list
and, optionally, checks that the active Python env can import the packages
and that its CARLA client matches the CARLA build.

Usage:
  python tools/install/check_install.py [options] [SCOPE ...]

SCOPE (what must pass for exit code 0; default: core):
  core       CARLA build, perception weights, YOLO11 -> every non-e2e baseline
  oa_vat     + OA-VAT weights, YOLOe and DINOv3 ViT-B/16  (PLANNER=oa_vat)
  trackvla   + TrackVLA weights and the DINOv3 ViT-S/16 snapshot  (PLANNER=trackvla)
  all        all of the above
Groups outside the scope are still listed, but do not affect the exit code.

Options:
  --deep           SHA256 every file (reads ~40 GB for `all`). Default: size for
                   every file, SHA256 only for files under 128 MiB.
  --carla-dir DIR  CARLA install (default: $CARLA_ROOT, dirname $CARLA_BIN, or
                   <repo>/CARLA_multi_robots).
  --no-python      Skip the Python env checks (imports, CARLA client version).
  -q, --quiet      Only print problems and the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "files.tsv"
SMALL_FILE = 128 * 1024 * 1024

SCOPES = {
    "core": ["carla", "weights", "yolo"],
    "oa_vat": ["oa_vat", "yoloe", "dinov3_oavat"],
    "trackvla": ["trackvla", "dinov3_trackvla"],
}
SCOPES["all"] = SCOPES["core"] + SCOPES["oa_vat"] + SCOPES["trackvla"]
GROUP_ORDER = SCOPES["all"] + ["optional"]

FIX = {
    "carla": "bash tools/install/install_assets.sh --download carla",
    "weights": "bash tools/install/install_assets.sh --download weights",
    "yolo": "bash tools/install/install_assets.sh --download weights   (fetches YOLO11 from Ultralytics)",
    "oa_vat": "bash tools/install/install_assets.sh --download oa_vat",
    "yoloe": "bash tools/install/install_assets.sh --download oa_vat    (fetches YOLOe from the upstream repo)",
    "trackvla": "bash tools/install/install_assets.sh --download trackvla",
    "dinov3_trackvla": "gated on Hugging Face; see docs/01_install.md, 'DINOv3 for TrackVLA'",
    "dinov3_oavat": "Meta request form; see docs/01_install.md, 'DINOv3 for OA-VAT'",
    "optional": "only for --oa-vat-dinov3 dinov3_vits16; see docs/01_install.md",
}


def default_carla_dir() -> Path:
    if os.environ.get("CARLA_ROOT"):
        return Path(os.environ["CARLA_ROOT"])
    if os.environ.get("CARLA_BIN"):
        return Path(os.environ["CARLA_BIN"]).parent
    return REPO_ROOT / "CARLA_multi_robots"


def read_manifest() -> list[dict]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        group, size, sha, path, source = line.split("\t")
        rows.append({"group": group, "size": size, "sha": sha, "path": path, "source": source})
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(row: dict, carla_dir: Path, deep: bool) -> tuple[str, str]:
    """Return (status, detail); status is ok | MISSING | BAD."""
    root = carla_dir if row["group"] == "carla" else REPO_ROOT
    path = root / row["path"]
    if not path.is_file():
        return "MISSING", str(path)
    size = path.stat().st_size
    if row["size"] != "-" and size != int(row["size"]):
        return "BAD", f"{path}: size {size}, expected {row['size']} (incomplete download or wrong file)"
    want = row["sha"]
    if want == "-":
        # torch.hub convention: name-<first 8 hex of sha256>.pth
        stem = path.name.rsplit(".", 1)[0]
        want_prefix = stem.rsplit("-", 1)[-1]
        got = sha256(path)
        if not got.startswith(want_prefix):
            return "BAD", f"{path}: sha256 {got[:12]}... does not start with {want_prefix} from the file name"
        return "ok", f"{path} (sha256 prefix)"
    if deep or size < SMALL_FILE:
        got = sha256(path)
        if got != want:
            return "BAD", f"{path}: sha256 {got[:12]}..., expected {want[:12]}..."
        return "ok", f"{path} (sha256)"
    return "ok", f"{path} (size)"


def check_python(carla_dir: Path, scopes: set[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def imp(name: str):
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            out.append(("BAD", f"import {name}: {type(e).__name__}: {e}"))
            return None

    out.append(("ok", f"python {sys.version.split()[0]} at {sys.executable}"))
    for pkg in ("followbench", "baselines"):
        mod = imp(pkg)
        if mod is not None:
            where = Path(mod.__file__).resolve()
            if REPO_ROOT not in where.parents:
                out.append(("BAD", f"{pkg} is imported from {where}, not from this repo; run `pip install -e .` here"))
            else:
                out.append(("ok", f"import {pkg} ({where.parent})"))

    carla = imp("carla")
    version_file = carla_dir / "VERSION"
    if carla is not None:
        client = carla.Client("127.0.0.1", 2000).get_client_version()
        server = version_file.read_text().strip() if version_file.is_file() else None
        if server is None:
            out.append(("BAD", f"carla client {client}; no {version_file} to compare with"))
        elif client != server:
            whl = next(iter(sorted((carla_dir / "PythonAPI/carla/dist").glob("carla-*-cp310-*.whl"))), None)
            out.append(("BAD", f"carla client {client} != CARLA build {server}; "
                               f"pip install --force-reinstall --no-deps {whl}"))
        else:
            out.append(("ok", f"carla client {client} matches the CARLA build"))

    torch = imp("torch")
    if torch is not None:
        if torch.cuda.is_available():
            out.append(("ok", f"torch {torch.__version__}, {torch.cuda.device_count()} CUDA device(s)"))
        else:
            out.append(("BAD", f"torch {torch.__version__} sees no CUDA device"))
    names = ["ultralytics", "cvxpy", "transformers"]
    if "oa_vat" in scopes:
        names.append("clip")  # YOLOe text prompts; see environment.multiRobots.yml
    for name in names:
        mod = imp(name)
        if mod is not None:
            out.append(("ok", f"import {name} {getattr(mod, '__version__', '')}".rstrip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scopes", nargs="*", default=["core"], choices=list(SCOPES), metavar="SCOPE")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--carla-dir", type=Path, default=None)
    ap.add_argument("--no-python", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    carla_dir = (args.carla_dir or default_carla_dir()).resolve()
    required = {g for s in args.scopes for g in SCOPES[s]}
    rows = read_manifest()

    print(f"[check] repo:  {REPO_ROOT}")
    print(f"[check] carla: {carla_dir}")
    print(f"[check] scope: {' '.join(args.scopes)}  ({'sha256 of every file' if args.deep else 'size, sha256 for small files'})")

    failed_required: list[str] = []
    missing_other: list[str] = []
    for group in GROUP_ORDER:
        results = [check_file(r, carla_dir, args.deep) for r in rows if r["group"] == group]
        n_ok = sum(1 for s, _ in results if s == "ok")
        is_req = group in required
        state = "ok" if n_ok == len(results) else ("FAIL" if is_req else "not installed")
        tag = "required" if is_req else "not required"
        print(f"\n[{group}] {n_ok}/{len(results)} files ok  -> {state} ({tag})")
        for status, detail in results:
            if status == "ok":
                show = not args.quiet and (is_req or n_ok == len(results))
            else:  # in quiet mode, missing files of groups you did not ask for are only counted
                show = is_req or status == "BAD" or not args.quiet
            if show:
                print(f"  {status:7} {detail}")
        if n_ok != len(results):
            print(f"  fix: {FIX[group]}")
            (failed_required if is_req else missing_other).append(group)

    if args.no_python:
        py_bad = []
    else:
        print("\n[python env]")
        py = check_python(carla_dir, {s for s in args.scopes} | ({'oa_vat'} if 'all' in args.scopes else set()))
        py_bad = [d for s, d in py if s != "ok"]
        for status, detail in py:
            if status != "ok" or not args.quiet:
                print(f"  {status:7} {detail}")

    print()
    if failed_required or py_bad:
        what = failed_required + (["python env"] if py_bad else [])
        print(f"[check] FAILED: {', '.join(what)}")
        return 1
    note = f"; not installed (not required): {', '.join(missing_other)}" if missing_other else ""
    print(f"[check] OK for scope: {' '.join(args.scopes)}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
