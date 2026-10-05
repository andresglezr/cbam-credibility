# -*- coding: utf-8 -*-
"""Run the deterministic perceived-stringency (kappa) sweep.

Only the three interior points require separate solves.  The paper pipeline
uses the central run's Surprise path at kappa=0 and its Credible path at
kappa=1, so this runner deliberately does not create redundant endpoint
files.  Outputs are written under new ``eubm_k*`` suffixes and never replace
the historical ``v3p*``/``v3ncp*`` results.

Examples
--------
Preview the three runs without solving::

    python partial_cred_sweep.py --dry-run

Run or resume the sweep::

    python partial_cred_sweep.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "sinretencion.py"
WELFARE = ROOT / "compute_welfare.py"
VALIDATOR = ROOT / "validate_batch_results.py"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# Kappa is a deterministic index of the perceived post-revelation policy
# path, not a probability.  Endpoints are read from the central run.
KAPPA_RUNS: tuple[tuple[float, str], ...] = (
    (0.25, "eubm_k25"),
    (0.50, "eubm_k50"),
    (0.75, "eubm_k75"),
)

BASE_ENV: dict[str, str] = {
    "RAMSEY_ADJSH": "0.05",
    "RAMSEY_ARMINGTON_SCALE": "1",
    "RAMSEY_CARBON_PRICE_EUR": "100",
    "RAMSEY_CBAM_EU_BENCHMARK_ENDOGENOUS": "1",
    "RAMSEY_CBAM_FLOOR_EPS": "1e-8",
    "RAMSEY_CBAM_PROCESS_EMISSIONS": "1",
    # Current-law central accounting: indirect electricity only for cement
    # and fertilizers.  The explicit list takes precedence over the legacy
    # all/none switch in the model.
    "RAMSEY_CBAM_INDIRECT_SECTORS": "CEM,FER",
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
    "RAMSEY_PATH_TOL": "1e-7",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}

_POLICY_ENV = (
    "RAMSEY_ADJSH",
    "RAMSEY_ARMINGTON_SCALE",
    "RAMSEY_CARBON_PRICE_EUR",
    "RAMSEY_CBAM_PROCESS_EMISSIONS",
    "RAMSEY_CREDIBILITY_INDEX",
    "RAMSEY_CREDIBILITY_P",  # historical alias; must not leak into a run
    "RAMSEY_EXTRA_COVERED",  # CHM is intentionally not part of this round
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
)


def expected_outputs(suffix: str) -> list[Path]:
    return [
        RESULTS / f"results_3d_{suffix}.parquet",
        RESULTS / f"results_2d_{suffix}.parquet",
        RESULTS / f"results_4d_{suffix}.parquet",
        RESULTS / f"parameters_{suffix}.parquet",
        RESULTS / f"sam_data_{suffix}.parquet",
    ]


def outputs_complete(suffix: str) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in expected_outputs(suffix))


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _POLICY_ENV:
        env.pop(name, None)
    env.update(BASE_ENV)
    return env


def compute_welfare(suffix: str, env: dict[str, str]) -> int:
    """Generate the published finite-horizon welfare file for this run."""
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    proc = subprocess.run(
        [str(python), str(WELFARE), "--suffix", suffix],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    if proc.returncode:
        print(f"[fail] welfare for {suffix}: return code {proc.returncode}")
    return proc.returncode


def validate_family(suffix: str, env: dict[str, str]) -> int:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    return subprocess.run(
        [str(python), str(VALIDATOR), "--suffix", suffix],
        cwd=str(ROOT),
        env=env,
        text=True,
    ).returncode


def run_point(kappa: float, suffix: str, *, force: bool, dry_run: bool) -> int:
    # A dry run must be side-effect free: return before touching existing
    # results or recomputing the welfare file.
    if dry_run:
        print(f"[dry] kappa={kappa:.2f} -> {suffix}")
        return 0
    if outputs_complete(suffix) and not force:
        print(f"[check] kappa={kappa:.2f}: candidate outputs already exist ({suffix})")
        env = clean_env()
        welfare_rc = compute_welfare(suffix, env)
        audit_rc = validate_family(suffix, env) if welfare_rc == 0 else 1
        if welfare_rc == 0 and audit_rc == 0:
            print(f"[skip] kappa={kappa:.2f}: existing family passed metadata and solver audit")
            return 0
        print(f"[rerun] kappa={kappa:.2f}: existing family failed welfare/audit checks")

    env = clean_env()
    env.update(
        {
            "RAMSEY_CREDIBILITY_INDEX": f"{kappa:g}",
            "RAMSEY_OUT_SUFFIX": suffix,
            "RAMSEY_RUN_LABEL": f"kappa_{kappa:.2f}",
        }
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"run_{suffix}.log"
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    print(f"[run] kappa={kappa:.2f} -> {suffix}")
    print(f"      log: {log_path}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            [str(python), str(MODEL)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0 or not outputs_complete(suffix):
        print(f"[fail] kappa={kappa:.2f}: return code {proc.returncode}; see {log_path}")
        return proc.returncode or 1
    print(f"[ok] kappa={kappa:.2f}")
    welfare_rc = compute_welfare(suffix, env)
    return welfare_rc or validate_family(suffix, env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic kappa sweep.")
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the runs without solving.")
    parser.add_argument(
        "--kappa",
        type=float,
        action="append",
        choices=[point[0] for point in KAPPA_RUNS],
        help="Run selected interior value(s); repeat the option as needed.",
    )
    args = parser.parse_args()

    selected = [point for point in KAPPA_RUNS if not args.kappa or point[0] in args.kappa]
    RESULTS.mkdir(exist_ok=True)
    for kappa, suffix in selected:
        rc = run_point(kappa, suffix, force=args.force, dry_run=args.dry_run)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
