# -*- coding: utf-8 -*-
"""Orchestrate the new central, kappa, and OAT sensitivity runs.

The script is resumable and writes only the published namespace: ``newblk``,
``newblk_k*`` and ``sens_newblk_*``.  (``v3eubm_legind``/``eubm_k*``/``sens_eubm_*``
are the superseded historical calibration and are never written here.)
It excludes chemicals and uses
the endogenous EU CBAM benchmark with indirect electricity limited to cement
and fertilizers.  Use ``--dry-run`` to inspect the complete launch plan.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = ROOT / "sinretencion.py"
WELFARE = ROOT / "compute_welfare.py"
VALIDATOR = ROOT / "validate_batch_results.py"
CENTRAL_SUFFIX = "newblk"

CENTRAL_ENV: dict[str, str] = {
    "RAMSEY_ADJSH": "0.05",
    "RAMSEY_ARMINGTON_SCALE": "1",
    "RAMSEY_CARBON_PRICE_EUR": "100",
    "RAMSEY_CREDIBILITY_INDEX": "0",
    "RAMSEY_CBAM_EU_BENCHMARK_ENDOGENOUS": "1",
    "RAMSEY_CBAM_FLOOR_EPS": "1e-8",
    "RAMSEY_CBAM_INDIRECT_SECTORS": "CEM,FER",
    "RAMSEY_CBAM_PROCESS_EMISSIONS": "1",
    "RAMSEY_OUT_SUFFIX": CENTRAL_SUFFIX,
    "RAMSEY_RUN_LABEL": "central_newblk",
    "RAMSEY_PATH_TOL": "1e-7",
    "RAMSEY_G": "0.02",
    "RAMSEY_GAMMA": "2",
    "RAMSEY_RO": "0.05",
    "RAMSEY_RP": "0",
    "RAMSEY_RSS": "0.05",
    "RAMSEY_SCRAP_MARGIN": "0.02",
    "RAMSEY_SCRAP_MULT": "0",
    "RAMSEY_SIGMA_ELC": "3",
    "RAMSEY_SIGMA_EN": "0.5",
    "RAMSEY_SIGMA_VAE": "0.5",
    "RAMSEY_T_MAX": "30",
    "RAMSEY_T_STAR": "8",
    "RAMSEY_USE_BASE_TAX_WEDGES": "1",
    "RAMSEY_ITER_LIM": "10000",
    "RAMSEY_OUT_DIR": "results",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}

_POLICY_ENV = (
    "RAMSEY_ADJSH",
    "RAMSEY_ARMINGTON_SCALE",
    "RAMSEY_CARBON_PRICE_EUR",
    "RAMSEY_CBAM_PROCESS_EMISSIONS",
    "RAMSEY_CREDIBILITY_INDEX",
    "RAMSEY_CREDIBILITY_P",
    "RAMSEY_EXTRA_COVERED",
    "RAMSEY_EXCLUDE_SECTORS",
    "RAMSEY_CBAM_INDIRECT_SECTORS",
    "RAMSEY_CBAM_INDIRECT_ELECTRICITY",
    "RAMSEY_CBAM_EU_BENCHMARK_ENDOGENOUS",
    "RAMSEY_CBAM_FLOOR_EPS",
    "RAMSEY_G",
    "RAMSEY_GAMMA",
    "RAMSEY_RO",
    "RAMSEY_RP",
    "RAMSEY_RSS",
    "RAMSEY_SCRAP_MARGIN",
    "RAMSEY_SCRAP_MULT",
    "RAMSEY_SIGMA_ELC",
    "RAMSEY_SIGMA_EN",
    "RAMSEY_SIGMA_VAE",
    "RAMSEY_T_MAX",
    "RAMSEY_T_STAR",
    "RAMSEY_ITER_LIM",
    "RAMSEY_OUT_DIR",
    "RAMSEY_PATH_TOL",
    "RAMSEY_USE_BASE_TAX_WEDGES",
    "RAMSEY_CENTRAL_SUFFIX",
)


def expected_outputs(suffix: str) -> list[Path]:
    return [
        RESULTS / f"results_3d_{suffix}.parquet",
        RESULTS / f"results_2d_{suffix}.parquet",
        RESULTS / f"results_4d_{suffix}.parquet",
        RESULTS / f"parameters_{suffix}.parquet",
        RESULTS / f"sam_data_{suffix}.parquet",
    ]


def central_complete() -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in expected_outputs(CENTRAL_SUFFIX))


def welfare_output() -> Path:
    return RESULTS / f"welfare_{CENTRAL_SUFFIX}.csv"


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _POLICY_ENV:
        env.pop(name, None)
    return env


def run_central(*, force: bool, dry_run: bool) -> int:
    if central_complete() and not force:
        print(f"[skip] central outputs already exist ({CENTRAL_SUFFIX})")
        return 0
    if dry_run:
        print(f"[dry] central -> {CENTRAL_SUFFIX}; env={CENTRAL_ENV}")
        return 0
    env = clean_env()
    env.update(CENTRAL_ENV)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"run_{CENTRAL_SUFFIX}.log"
    print(f"[run] central -> {CENTRAL_SUFFIX}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            [str(PYTHON if PYTHON.exists() else Path(sys.executable)), str(MODEL)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode or not central_complete():
        print(f"[fail] central: return code {proc.returncode}; see {log_path}")
        return proc.returncode or 1
    print("[ok] central")
    return 0


def run_welfare(*, dry_run: bool) -> int:
    """Recompute welfare for the exact central suffix, with visible errors."""
    command = [
        str(PYTHON if PYTHON.exists() else Path(sys.executable)),
        str(WELFARE),
        "--results-dir",
        str(RESULTS),
        "--suffix",
        CENTRAL_SUFFIX,
    ]
    if dry_run:
        print("[dry] welfare -> " + " ".join(command))
        return 0
    if not central_complete():
        missing = [str(path) for path in expected_outputs(CENTRAL_SUFFIX) if not path.exists()]
        print(f"[fail] welfare: central inputs are incomplete ({', '.join(missing)})")
        return 1

    print(f"[run] welfare -> {welfare_output().name}")
    proc = subprocess.run(command, cwd=str(ROOT), env=clean_env())
    out_path = welfare_output()
    if proc.returncode:
        print(f"[fail] welfare: return code {proc.returncode}")
        return proc.returncode
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"[fail] welfare: expected output was not written ({out_path})")
        return 1
    newest_input = max(path.stat().st_mtime_ns for path in expected_outputs(CENTRAL_SUFFIX))
    if out_path.stat().st_mtime_ns < newest_input:
        print(f"[fail] welfare: output is older than the central inputs ({out_path})")
        return 1
    print("[ok] welfare")
    audit = subprocess.run(
        [str(PYTHON if PYTHON.exists() else Path(sys.executable)), str(VALIDATOR), "--suffix", CENTRAL_SUFFIX],
        cwd=str(ROOT),
        env=clean_env(),
        text=True,
    )
    if audit.returncode:
        print(f"[fail] central metadata/solver audit: return code {audit.returncode}")
        return audit.returncode
    print("[ok] central metadata/solver audit")
    return 0


def run_child(script: str, args: list[str], *, dry_run: bool) -> int:
    command = [str(PYTHON if PYTHON.exists() else Path(sys.executable)), str(ROOT / script), *args]
    if dry_run:
        command.append("--dry-run")
    print("[launch] " + " ".join(command))
    return subprocess.run(command, cwd=str(ROOT), env=clean_env()).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/resume the EU-benchmark recalculation pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print every planned run without solving.")
    parser.add_argument("--force", action="store_true", help="Rerun complete outputs as well.")
    parser.add_argument("--skip-central", action="store_true")
    parser.add_argument("--skip-welfare", action="store_true")
    parser.add_argument("--skip-kappa", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if not args.skip_central:
        rc = run_central(force=args.force, dry_run=args.dry_run)
        if rc:
            return rc
    if not args.skip_welfare:
        rc = run_welfare(dry_run=args.dry_run)
        if rc:
            return rc
    if not args.skip_kappa:
        child_args = ["--force"] if args.force else []
        rc = run_child("partial_cred_sweep_newblk.py", child_args, dry_run=args.dry_run)
        if rc:
            return rc
    if not args.skip_sensitivity:
        child_args = ["--plan", "paper", "--prefix", "sens_newblk", "--summary"]
        if args.force:
            child_args.append("--force")
        rc = run_child("sensitivity_experiments.py", child_args, dry_run=args.dry_run)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
