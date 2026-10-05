"""Parametric sensitivity runner for the Ramsey-CBAM model.

This script solves ``sinretencion.py`` under alternative parameter values by
passing environment variables to the model. It is designed for one-at-a-time
(OAT) sensitivity around the central calibration, not a full factorial grid.

Why OAT here?
-------------
Each full model run is expensive. A full factorial grid over carbon prices,
energy substitution, Armington elasticities, revelation dates, and CBAM
accounting would quickly require dozens or hundreds of solves. OAT gives a
clean first robustness layer: each result can be attributed to one modeling
choice. Interactions can be added later once the central mechanism is stable.

Examples
--------
List experiments and plans:
    python sensitivity_experiments.py --list

Dry-run the core plan:
    python sensitivity_experiments.py --plan core --dry-run

Run the core plan, resuming already completed variants:
    python sensitivity_experiments.py --plan core

Run selected variants:
    python sensitivity_experiments.py sigma_en_low sigma_en_high carbon_125

Summarize generated sensitivity outputs:
    python sensitivity_experiments.py --summary-only --prefix sens_newblk

Write a validated manifest for existing outputs without recomputing welfare or
launching the model:
    python sensitivity_experiments.py --manifest-only --prefix sens_newblk
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "sinretencion.py"
WELFARE = ROOT / "compute_welfare.py"
VALIDATOR = ROOT / "validate_batch_results.py"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
# The paper's base row is fixed deliberately; it must not drift with a shell
# variable left over from an older experiment.
CENTRAL_SUFFIX = "newblk"

COVERED = ["STL", "CEM", "ALU", "FER"]

# Policy/accounting choices shared by the new central run and every OAT
# variant.  Keeping these values in the runner prevents a sensitivity result
# from silently reverting to the historical proportional-CBAM specification
# or to all-sector indirect-electricity coverage.
BASE_ENV: dict[str, str] = {
    "RAMSEY_ADJSH": "0.05",
    "RAMSEY_ARMINGTON_SCALE": "1",
    "RAMSEY_CARBON_PRICE_EUR": "100",
    "RAMSEY_CREDIBILITY_INDEX": "0",
    "RAMSEY_CBAM_EU_BENCHMARK_ENDOGENOUS": "1",
    "RAMSEY_CBAM_FLOOR_EPS": "1e-8",
    "RAMSEY_CBAM_INDIRECT_SECTORS": "CEM,FER",
    "RAMSEY_CBAM_PROCESS_EMISSIONS": "1",
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
}

# These old variants changed only the steady-state calibration rate (RSS),
# while leaving the Euler-equation discount rate (RO) unchanged.  They are not
# coherent patience experiments and are intentionally non-runnable.
INVALID_EXPERIMENTS: dict[str, str] = {
    "discount_low": "invalid: changes RAMSEY_RSS without RAMSEY_RO and joint recalibration",
    "discount_high": "invalid: changes RAMSEY_RSS without RAMSEY_RO and joint recalibration",
}


def manifest_path(prefix: str) -> Path:
    tag = prefix.removeprefix("sens_")
    return RESULTS / f"sensitivity_manifest_{tag}.csv"


def summary_path(prefix: str) -> Path:
    tag = prefix.removeprefix("sens_")
    return RESULTS / f"sensitivity_summary_{tag}.csv"


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    description: str
    env: dict[str, str]


EXPERIMENTS: dict[str, Experiment] = {
    "base": Experiment(
        "base",
        "benchmark",
        "Central calibration: endogenous EU benchmark and indirect electricity for CEM/FER only.",
        {},
    ),
    "carbon_75": Experiment(
        "carbon_75",
        "policy_level",
        "Lower carbon price: 75 EUR/tCO2.",
        {"RAMSEY_CARBON_PRICE_EUR": "75"},
    ),
    "carbon_125": Experiment(
        "carbon_125",
        "policy_level",
        "Higher carbon price: 125 EUR/tCO2.",
        {"RAMSEY_CARBON_PRICE_EUR": "125"},
    ),
    "sigma_en_low": Experiment(
        "sigma_en_low",
        "energy_substitution",
        "Lower ELC-REF substitution elasticity: sigma_EN=0.25.",
        {"RAMSEY_SIGMA_EN": "0.25"},
    ),
    "sigma_en_high": Experiment(
        "sigma_en_high",
        "energy_substitution",
        "Higher ELC-REF substitution elasticity: sigma_EN=0.75.",
        {"RAMSEY_SIGMA_EN": "0.75"},
    ),
    "sigma_vae_low": Experiment(
        "sigma_vae_low",
        "energy_substitution",
        "Lower VA-energy substitution elasticity: sigma_VAE=0.25.",
        {"RAMSEY_SIGMA_VAE": "0.25"},
    ),
    "sigma_vae_high": Experiment(
        "sigma_vae_high",
        "energy_substitution",
        "Higher VA-energy substitution elasticity: sigma_VAE=0.75.",
        {"RAMSEY_SIGMA_VAE": "0.75"},
    ),
    "sigma_elc_low": Experiment(
        "sigma_elc_low",
        "electricity_substitution",
        "Lower fossil-renewable electricity substitution: sigma_ELC=1.5.",
        {"RAMSEY_SIGMA_ELC": "1.5"},
    ),
    "sigma_elc_high": Experiment(
        "sigma_elc_high",
        "electricity_substitution",
        "Higher fossil-renewable electricity substitution: sigma_ELC=5.0.",
        {"RAMSEY_SIGMA_ELC": "5.0"},
    ),
    "armington_low": Experiment(
        "armington_low",
        "trade_substitution",
        "Lower all Armington elasticities by 25 percent.",
        {"RAMSEY_ARMINGTON_SCALE": "0.75"},
    ),
    "armington_high": Experiment(
        "armington_high",
        "trade_substitution",
        "Higher all Armington elasticities by 25 percent.",
        {"RAMSEY_ARMINGTON_SCALE": "1.25"},
    ),
    "cbam_direct_only": Experiment(
        "cbam_direct_only",
        "cbam_accounting",
        "CBAM base excludes benchmark indirect electricity emissions.",
        {
            "RAMSEY_CBAM_INDIRECT_SECTORS": "",
            "RAMSEY_CBAM_INDIRECT_ELECTRICITY": "0",
        },
    ),
    "cbam_no_process": Experiment(
        "cbam_no_process",
        "cbam_accounting",
        "CBAM base excludes process emissions. Domestic carbon wedges are unchanged.",
        {"RAMSEY_CBAM_PROCESS_EMISSIONS": "0"},
    ),
    "tstar_2028": Experiment(
        "tstar_2028",
        "timing",
        "Earlier revelation / phase-2 start: T*=7, year 2028.",
        {"RAMSEY_T_STAR": "7"},
    ),
    "tstar_2030": Experiment(
        "tstar_2030",
        "timing",
        "Later revelation / phase-2 start: T*=9, year 2030.",
        {"RAMSEY_T_STAR": "9"},
    ),
    "adj_cost_low": Experiment(
        "adj_cost_low",
        "dynamic_adjustment",
        "Lower benchmark adjustment-cost share: adjsh=0.015.",
        {"RAMSEY_ADJSH": "0.015"},
    ),
    "adj_cost_high": Experiment(
        "adj_cost_high",
        "dynamic_adjustment",
        "Higher benchmark adjustment-cost share: adjsh=0.06.",
        {"RAMSEY_ADJSH": "0.06"},
    ),
    "adj_cost_higher": Experiment(
        "adj_cost_higher",
        "dynamic_adjustment",
        "Steeper benchmark adjustment-cost share: adjsh=0.10.",
        {"RAMSEY_ADJSH": "0.10"},
    ),
    "gamma_log": Experiment(
        "gamma_log",
        "preferences",
        "Log utility: gamma=1.",
        {"RAMSEY_GAMMA": "1.0"},
    ),
    "gamma_high": Experiment(
        "gamma_high",
        "preferences",
        "Higher intertemporal risk aversion: gamma=4.",
        {"RAMSEY_GAMMA": "4.0"},
    ),
}

PLANS: dict[str, list[str]] = {
    "smoke": ["base"],
    "core": [
        "base",
        "sigma_en_low",
        "sigma_en_high",
        "armington_low",
        "armington_high",
        "carbon_75",
        "carbon_125",
        "cbam_direct_only",
        "tstar_2028",
        "tstar_2030",
    ],
    "mechanism": [
        "base",
        "sigma_en_low",
        "sigma_en_high",
        "sigma_vae_low",
        "sigma_vae_high",
        "sigma_elc_low",
        "sigma_elc_high",
        "adj_cost_low",
        "adj_cost_high",
        "gamma_log",
        "gamma_high",
    ],
    "cbam": ["base", "cbam_direct_only", "cbam_no_process"],
    "adjustment": ["base", "adj_cost_low", "adj_cost_high", "adj_cost_higher"],
    "paper": [
        "sigma_en_low",
        "sigma_en_high",
        "armington_low",
        "armington_high",
        "carbon_75",
        "carbon_125",
        "cbam_direct_only",
        "tstar_2028",
        "tstar_2030",
        "adj_cost_low",
        "adj_cost_high",
        "adj_cost_higher",
    ],
    "all": list(EXPERIMENTS.keys()),
}


def output_suffix(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


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


def compute_welfare_file(suffix: str, env: dict[str, str] | None = None) -> int:
    """Generate the finite-horizon welfare artifact for a completed variant."""
    proc = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else Path(sys.executable)),
            str(WELFARE),
            "--results-dir",
            str(RESULTS),
            "--suffix",
            suffix,
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    out = RESULTS / f"welfare_{suffix}.csv"
    if proc.returncode or not out.exists() or out.stat().st_size == 0:
        print(f"[fail] welfare for {suffix}: return code {proc.returncode}")
        return proc.returncode or 1
    return 0


#: Outcomes of a batch audit.  These are deliberately *not* integers: a skipped
#: audit used to share the success return code, which made it impossible to tell
#: "validated" from "never checked" downstream.
VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"
VALIDATION_SKIPPED = "skipped"

#: When true, a family outside the registered matrix is a hard error rather than
#: a warning.  Set by ``--certify`` for the final publication campaign.
CERTIFY_MODE = False


def validation_ok(result: str) -> bool:
    """A run may continue on a skip, but only a pass counts as validated."""
    return result == VALIDATION_PASSED


def validate_family(suffix: str, env: dict[str, str] | None = None) -> str:
    """Validate paper families; retain generic research variants outside the matrix."""
    from validate_batch_results import SPEC_BY_SUFFIX

    if suffix not in SPEC_BY_SUFFIX:
        # Not part of any registered matrix: the audit cannot run, so say so
        # loudly.  An exploratory run continues, but the family is recorded as
        # unvalidated and certification refuses it outright.
        if CERTIFY_MODE:
            print(f"[fail] {suffix}: not a registered family; refused under --certify")
            return VALIDATION_FAILED
        print(f"[warn] {suffix}: not a registered family; batch audit SKIPPED (not validated)")
        return VALIDATION_SKIPPED
    completed = subprocess.run(
        [
            str(PYTHON if PYTHON.exists() else Path(sys.executable)),
            str(VALIDATOR),
            "--suffix",
            suffix,
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    return VALIDATION_PASSED if completed.returncode == 0 else VALIDATION_FAILED


def append_manifest(row: dict[str, object], prefix: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = manifest_path(prefix)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    if "suffix" in df.columns:
        df = df.drop_duplicates(subset=["suffix"], keep="last")
    df.to_csv(path, index=False)


def write_existing_manifest(prefix: str, names: list[str]) -> pd.DataFrame:
    """Validate and inventory existing families without mutating their outputs.

    This deliberately does not call ``compute_welfare_file`` and has no route
    to ``run_experiment``.  It is therefore safe to use after an expensive
    batch has completed: a missing or invalid family aborts before the manifest
    is replaced, rather than silently relaunching a solve or truncating a log.
    """
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for name in names:
        exp = EXPERIMENTS[name]
        suffix = output_suffix(prefix, name)
        welfare = RESULTS / f"welfare_{suffix}.csv"
        if not outputs_complete(suffix):
            failures.append(f"{suffix}: incomplete model outputs")
            continue
        if not welfare.exists() or welfare.stat().st_size == 0:
            failures.append(f"{suffix}: missing welfare output")
            continue
        audit = validate_family(suffix)
        if audit == VALIDATION_FAILED:
            failures.append(f"{suffix}: validator failed")
            continue
        rows.append(
            {
                "experiment": exp.name,
                "group": exp.group,
                "suffix": suffix,
                "status": "exists_validated" if validation_ok(audit) else "exists_unvalidated",
                "return_code": 0,
                "seconds": 0.0,
                "log": str(LOGS / f"{suffix}.log"),
                "env": json.dumps({**BASE_ENV, **exp.env}, sort_keys=True),
            }
        )

    if failures:
        detail = "\n  - ".join(failures)
        raise RuntimeError(f"Cannot write sensitivity manifest:\n  - {detail}")

    df = pd.DataFrame(rows)
    path = manifest_path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"Wrote {path} ({len(df)} validated families; no outputs recomputed).")
    return df


def run_experiment(exp: Experiment, prefix: str, force: bool, dry_run: bool, stop_on_existing: bool) -> int:
    suffix = output_suffix(prefix, exp.name)
    log_path = LOGS / f"{suffix}.log"
    # A dry run must be side-effect free: return before touching existing
    # results, recomputing welfare or appending to the manifest.
    if dry_run:
        effective_env = {**BASE_ENV, **exp.env}
        print(f"[dry] {exp.name}: suffix={suffix}, env={effective_env}")
        return 0
    complete = outputs_complete(suffix)
    if complete and not force:
        print(f"[check] {exp.name}: candidate outputs already exist ({suffix})")
        welfare_rc = compute_welfare_file(suffix)
        audit = validate_family(suffix) if welfare_rc == 0 else VALIDATION_FAILED
        if welfare_rc == 0 and audit != VALIDATION_FAILED:
            status = "exists_validated" if validation_ok(audit) else "exists_unvalidated"
            append_manifest(
                {
                    "experiment": exp.name,
                    "group": exp.group,
                    "suffix": suffix,
                    "status": status,
                    "return_code": 0,
                    "seconds": 0.0,
                    "log": str(log_path),
                    "env": json.dumps({**BASE_ENV, **exp.env}, sort_keys=True),
                },
                prefix,
            )
            print(f"[skip] {exp.name}: existing family passed metadata and solver audit")
            return 0
        if stop_on_existing:
            return welfare_rc or audit_rc or 1
        print(f"[rerun] {exp.name}: existing family failed welfare/audit checks")

    env = os.environ.copy()
    # Policy-shaping env vars must not leak in from the caller's shell: a
    # variant is defined solely by exp.env, so any of these not set by the
    # variant is cleared to the model default. Otherwise a stray
    # RAMSEY_CREDIBILITY_INDEX=0.5 (or EXTRA_COVERED/EXCLUDE) left in the shell
    # would silently re-solve every variant under the wrong policy.
    for _leaky in (
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
    ):
        env.pop(_leaky, None)
    env.update(BASE_ENV)
    env.update(exp.env)
    env["RAMSEY_RUN_LABEL"] = exp.name
    env["RAMSEY_OUT_SUFFIX"] = suffix
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[run] {exp.name}: {exp.description}")
    print(f"      suffix={suffix}")
    print(f"      log={log_path}")
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            [str(PYTHON if PYTHON.exists() else Path(sys.executable)), str(MODEL)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    seconds = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 and outputs_complete(suffix) else "failed"
    welfare_rc = 0
    audit = VALIDATION_PASSED
    if status == "ok":
        welfare_rc = compute_welfare_file(suffix, env)
        if welfare_rc:
            status = "welfare_failed"
        else:
            audit = validate_family(suffix, env)
            if audit == VALIDATION_FAILED:
                status = "audit_failed"
            elif audit == VALIDATION_SKIPPED:
                # Solved fine, but nothing certified it: never report this as ok.
                status = "ok_unvalidated"
    print(f"[{status}] {exp.name}: {seconds/60:.1f} minutes")
    audit_rc = 1 if audit == VALIDATION_FAILED else 0
    effective_rc = proc.returncode or welfare_rc or audit_rc or (
        1 if status not in ("ok", "ok_unvalidated") else 0
    )
    append_manifest(
        {
            "experiment": exp.name,
            "group": exp.group,
            "suffix": suffix,
            "status": status,
            "return_code": effective_rc,
            "seconds": round(seconds, 3),
            "log": str(log_path),
            "env": json.dumps({**BASE_ENV, **exp.env}, sort_keys=True),
        },
        prefix,
    )
    return effective_rc


def scalar_param(params: pd.DataFrame, name: str, default: float | None = None) -> float:
    row = params.loc[params["param"].eq(name)]
    if row.empty:
        if default is None:
            raise KeyError(f"Missing scalar parameter {name}")
        return default
    return float(row.iloc[0]["value"])


def fa_schedule(params: pd.DataFrame) -> dict[int, float]:
    rows = params.loc[params["param"].eq("FA_schedule")]
    return {int(row["T"]): float(row["value"]) for _, row in rows.iterrows()}


def summarize_suffix(suffix: str) -> dict[str, object] | None:
    paths = expected_outputs(suffix)
    if not all(path.exists() for path in paths):
        return None

    r3 = pd.read_parquet(RESULTS / f"results_3d_{suffix}.parquet")
    params = pd.read_parquet(RESULTS / f"parameters_{suffix}.parquet")
    sam = pd.read_parquet(RESULTS / f"sam_data_{suffix}.parquet")

    g = scalar_param(params, "g", 0.02)
    sigma_en = scalar_param(params, "SIGMA_EN", 0.5)
    carbon_price = scalar_param(params, "CARBON_PRICE_EUR", 100.0) / scalar_param(params, "SCALE", 1000.0)
    t_star = int(scalar_param(params, "T_STAR", 8))
    fa = fa_schedule(params)
    regions = sorted(r3["reg"].unique())
    eu = [r for r in regions if str(r).startswith("EU")]

    def get_fa(t: int) -> float:
        return fa.get(int(t), 0.0)

    # Prefer the exact scenario policy paths exported by the solver. This is
    # essential for Anticipated Withdrawal/Eq4_Phase1: applying the legislated
    # wedge after T* to an equilibrium solved under withdrawal materially
    # mis-reconstructs refined-fuel demand and reverses the RN-AW decomposition.
    policy_rows = params.loc[params["param"].eq("scenario_fa_path")]
    policy_fa: dict[tuple[str, int], float] = {
        (str(row["partner"]), int(row["T"])): float(row["value"])
        for _, row in policy_rows.iterrows()
    }

    def scenario_fa(eq: str, t: int) -> float:
        exact = policy_fa.get((eq, int(t)))
        if exact is not None:
            return exact
        # Historical-output fallback, used only for pre-metadata parquets.
        if eq == "Eq1_NoPol":
            return 1.0
        if eq in ("Eq3_Reneg", "Eq4_Phase1") and int(t) >= t_star:
            return 1.0
        return get_fa(t)

    def tau_fuel_for(row: pd.Series) -> float:
        if row["reg"] in eu and row["sec"] in COVERED and row["eq"] != "Eq1_NoPol":
            phi = scenario_fa(str(row["eq"]), int(row["T"]))
            return (1.0 - phi) * carbon_price * row["emis_fuel_per_ref"]
        return 0.0

    # Benchmark of INVP is `inv` (gross capital formation of sector s), not the
    # SAM row `iz` (demand for good s as an investment good). The two differ by
    # the capital-output structure, so normalising INVP by iz mislabels the
    # percentage-point gap. `inv` reproduces the no-policy balanced-growth path
    # exactly (Eq1 INVP / (inv*(1+g)^(t-1)) == 1), which is the base the caption
    # claims and the base the sectoral tables in code/tables.py already use.
    base_inv = params.loc[params["param"].eq("inv"), ["reg", "sec", "value"]].rename(columns={"value": "base_inv"})
    inv = r3.loc[
        r3["var"].eq("INVP")
        & r3["eq"].isin(["Eq2_Credible", "Eq4_Surprise"])
        & r3["reg"].isin(eu)
        & r3["sec"].isin(COVERED)
        & r3["T"].isin([t_star - 1, t_star, t_star + 1]),
        ["eq", "reg", "sec", "T", "value"],
    ].merge(base_inv, on=["reg", "sec"], how="left")
    inv["inv_hat"] = inv["value"] / (inv["base_inv"] * (1.0 + g) ** (inv["T"] - 1))
    inv_avg = (
        inv.groupby(["eq", "T"])
        .apply(lambda x: np.average(x["inv_hat"], weights=x["base_inv"]), include_groups=False)
        .rename("inv_hat")
        .reset_index()
        .pivot(index="T", columns="eq", values="inv_hat")
    )
    inv_gaps = 100.0 * (inv_avg["Eq4_Surprise"] - inv_avg["Eq2_Credible"])

    param_frames = []
    for pname in ["emis_proc_intensity", "emis_ff_intensity", "emis_fuel_per_ref"]:
        param_frames.append(
            params.loc[params["param"].eq(pname), ["reg", "sec", "value"]]
            .rename(columns={"value": pname})
        )
    p = param_frames[0].merge(param_frames[1], on=["reg", "sec"]).merge(param_frames[2], on=["reg", "sec"])
    xd = sam.loc[sam["var"].eq("xd"), ["reg", "sec", "value"]].rename(columns={"value": "xd"})
    ref = sam.loc[sam["var"].eq("en_ref"), ["reg", "sec", "value"]].rename(columns={"value": "ref"})
    p = p.merge(xd, on=["reg", "sec"]).merge(ref, on=["reg", "sec"])

    wide = (
        r3.loc[
            r3["var"].isin(["H_XD", "H_EN", "H_PE"]),
            ["eq", "reg", "sec", "T", "var", "value"],
        ]
        .pivot_table(index=["eq", "reg", "sec", "T"], columns="var", values="value", aggfunc="first")
        .reset_index()
    )
    hp_ref = r3.loc[
        r3["var"].eq("H_P") & r3["sec"].eq("REF"),
        ["eq", "reg", "T", "value"],
    ].rename(columns={"value": "H_P_REF"})
    wide = wide.merge(hp_ref, on=["eq", "reg", "T"], how="left").merge(p, on=["reg", "sec"], how="left")
    wide["tau_fuel"] = wide.apply(tau_fuel_for, axis=1)
    wide["growth"] = (1.0 + g) ** (wide["T"] - 1)
    wide["ref_hat"] = wide["H_EN"] * (wide["H_PE"] / (wide["H_P_REF"] + wide["tau_fuel"])) ** sigma_en
    wide["emissions_exact"] = wide["growth"] * (
        wide["emis_proc_intensity"] * wide["xd"] * wide["H_XD"]
        + wide["emis_fuel_per_ref"] * wide["ref"] * wide["ref_hat"]
    )
    total_by_eq = wide.groupby("eq")["emissions_exact"].sum()
    emissions_si_minus_cc = total_by_eq.get("Eq4_Surprise", np.nan) - total_by_eq.get("Eq2_Credible", np.nan)
    emissions_rn_minus_aw = total_by_eq.get("Eq3_Reneg", np.nan) - total_by_eq.get("Eq4_Phase1", np.nan)
    emissions_aw_minus_cc = total_by_eq.get("Eq4_Phase1", np.nan) - total_by_eq.get("Eq2_Credible", np.nan)
    emissions_rn_minus_cc = total_by_eq.get("Eq3_Reneg", np.nan) - total_by_eq.get("Eq2_Credible", np.nan)
    decomposition_residual = emissions_rn_minus_cc - (emissions_rn_minus_aw + emissions_aw_minus_cc)

    base_xd = sam.loc[sam["var"].eq("xd"), ["reg", "sec", "value"]].rename(columns={"value": "base_xd"})
    out = r3.loc[
        r3["var"].eq("H_XD")
        & r3["eq"].isin(["Eq2_Credible", "Eq2_NoCBAM"])
        & r3["reg"].isin(eu)
        & r3["sec"].isin(COVERED)
        & r3["year"].eq(2034),
        ["eq", "reg", "sec", "value"],
    ].merge(base_xd, on=["reg", "sec"], how="left")
    if out.empty:
        cbam_effect = np.nan
    else:
        out_avg = (
            out.groupby("eq")
            .apply(lambda x: np.average(x["value"], weights=x["base_xd"]), include_groups=False)
            .rename("xd_hat")
        )
        cbam_effect = 100.0 * (out_avg.get("Eq2_Credible", np.nan) - out_avg.get("Eq2_NoCBAM", np.nan))

    param_lookup = {
        name: scalar_param(params, name, np.nan)
        for name in [
            "CARBON_PRICE_EUR",
            "SIGMA_EN",
            "SIGMA_VAE",
            "SIGMA_ELC",
            "ARMINGTON_SCALE",
            "T_STAR",
            "adjsh",
            "gamma",
            "CBAM_EU_BENCHMARK_ENDOGENOUS",
            "CREDIBILITY_INDEX",
            "CREDIBILITY_P",  # historical metadata fallback
        ]
    }
    return {
        "suffix": suffix,
        **param_lookup,
        "inv_gap_pre_pp": inv_gaps.get(t_star - 1, np.nan),
        "inv_gap_revelation_pp": inv_gaps.get(t_star, np.nan),
        "inv_gap_post_pp": inv_gaps.get(t_star + 1, np.nan),
        "emissions_si_minus_cc_mt": emissions_si_minus_cc,
        "emissions_rn_minus_aw_mt": emissions_rn_minus_aw,
        "emissions_aw_minus_cc_mt": emissions_aw_minus_cc,
        "emissions_rn_minus_cc_mt": emissions_rn_minus_cc,
        "emissions_decomposition_residual_mt": decomposition_residual,
        "cbam_effect_2034_pp": cbam_effect,
    }


def _assert_same_namespace(prefix: str, suffixes: list[str]) -> None:
    """Refuse to build a summary that mixes calibrations.

    The base row of the one-at-a-time table is copied from CENTRAL_SUFFIX while
    the remaining rows come from ``<prefix>_*``.  If those belong to different
    registered matrices the table silently blends two calibrations, which is
    exactly the failure this guard exists to prevent.
    """
    try:
        from validate_batch_results import PUBLISHED_SPECS, HISTORICAL_SPECS
    except ImportError:  # validator unavailable: nothing to check against
        return
    families = {
        "published": frozenset(spec.suffix for spec in PUBLISHED_SPECS),
        "historical": frozenset(spec.suffix for spec in HISTORICAL_SPECS),
    }

    def family_of(suffix: str) -> str | None:
        for name, members in families.items():
            if suffix in members:
                return name
        return None

    central_family = family_of(CENTRAL_SUFFIX)
    if central_family is None:
        return  # exploratory central: nothing to enforce
    offenders = [
        s for s in suffixes
        if family_of(s) is not None and family_of(s) != central_family
    ]
    if offenders:
        raise SystemExit(
            f"refusing to mix calibrations: central '{CENTRAL_SUFFIX}' is "
            f"{central_family}, but {offenders} belong to another matrix. "
            f"Use --prefix from the same family as CENTRAL_SUFFIX."
        )


def summarize(prefix: str, names: list[str] | None) -> pd.DataFrame:
    if names:
        suffixes = [output_suffix(prefix, name) for name in names]
    else:
        suffixes = sorted(
            path.name.removeprefix("parameters_").removesuffix(".parquet")
            for path in RESULTS.glob(f"parameters_{prefix}_*.parquet")
        )
    rows = []
    missing: list[str] = []
    _assert_same_namespace(prefix, suffixes)
    # The OAT base is the already-solved central run. Reuse it in the summary
    # under the conventional <prefix>_base row instead of paying for a
    # redundant thirteenth sensitivity solve or copying large parquet files.
    central = summarize_suffix(CENTRAL_SUFFIX)
    if central is None:
        missing.append(CENTRAL_SUFFIX)
    else:
        central["source_suffix"] = CENTRAL_SUFFIX
        central["suffix"] = output_suffix(prefix, "base")
        rows.append(central)
    for suffix in suffixes:
        if suffix == output_suffix(prefix, "base"):
            continue
        row = summarize_suffix(suffix)
        if row is None:
            missing.append(suffix)
        else:
            row["source_suffix"] = suffix
            rows.append(row)
    if missing:
        raise FileNotFoundError(
            "Cannot build a complete sensitivity summary; missing output families: "
            + ", ".join(missing)
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        path = summary_path(prefix)
        df.to_csv(path, index=False)
        print(f"Wrote {path}")
        cols = [
            "suffix",
            "CARBON_PRICE_EUR",
            "SIGMA_EN",
            "ARMINGTON_SCALE",
            "T_STAR",
            "inv_gap_pre_pp",
            "inv_gap_revelation_pp",
            "inv_gap_post_pp",
            "emissions_si_minus_cc_mt",
            "cbam_effect_2034_pp",
        ]
        print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
    else:
        print("No sensitivity outputs found to summarize.")
    return df


def choose_experiments(plan: str | None, names: list[str]) -> list[str]:
    selected: list[str] = []
    if plan:
        selected.extend(PLANS[plan])
    selected.extend(names)
    if not selected:
        selected = PLANS["paper"]
    invalid = [name for name in selected if name in INVALID_EXPERIMENTS]
    if invalid:
        detail = "; ".join(f"{name}: {INVALID_EXPERIMENTS[name]}" for name in invalid)
        raise SystemExit(f"Invalid sensitivity experiment(s): {detail}")
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments: {', '.join(unknown)}. Use --list.")
    deduped: list[str] = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and summarize Ramsey sensitivity experiments.")
    parser.add_argument("experiments", nargs="*", help="Experiment names. If omitted, --plan or core is used.")
    parser.add_argument("--plan", choices=sorted(PLANS), help="Predefined experiment plan.")
    parser.add_argument("--list", action="store_true", help="List experiments and plans, then exit.")
    parser.add_argument(
        "--prefix",
        default="sens_newblk",
        help="Output suffix prefix. Default: sens_newblk (the published namespace; "
             "does not overwrite historical sens_eubm_* runs).",
    )
    parser.add_argument(
        "--certify",
        action="store_true",
        help="Publication mode: a family outside the registered matrix is a hard error "
             "instead of a skipped audit.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun experiments even if outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runs without solving.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Keep running variants after one fails.")
    parser.add_argument("--summary", action="store_true", help="Summarize outputs after running.")
    parser.add_argument("--summary-only", action="store_true", help="Only summarize existing outputs.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate and inventory existing outputs without recomputing welfare or running the model.",
    )
    args = parser.parse_args()

    global CERTIFY_MODE
    CERTIFY_MODE = args.certify

    if args.summary_only and args.manifest_only:
        parser.error("--summary-only and --manifest-only are mutually exclusive")
    if args.manifest_only and args.dry_run:
        parser.error("--manifest-only validates existing files and cannot be combined with --dry-run")

    if args.list:
        print("Plans:")
        for name, exps in PLANS.items():
            print(f"  {name}: {', '.join(exps)}")
        print("\nExperiments:")
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:18s} [{exp.group}] {exp.description} env={exp.env}")
        print("\nInvalid historical experiments (not runnable):")
        for name, reason in INVALID_EXPERIMENTS.items():
            print(f"  {name:18s} {reason}")
        return 0

    names = choose_experiments(args.plan, args.experiments)

    if args.manifest_only:
        write_existing_manifest(args.prefix, names)
        return 0

    if args.summary_only:
        # ``choose_experiments`` defaults to the paper plan, so the no-argument
        # form is deliberately restricted to the twelve registered OAT runs.
        # Do not glob arbitrary experimental families into a paper summary.
        summarize(args.prefix, names)
        return 0

    RESULTS.mkdir(exist_ok=True)
    # Preserve any failure across the full batch.  With --continue-on-fail a
    # later successful run must not overwrite a non-zero status from an
    # earlier experiment.
    rc = 0
    print("Selected experiments:")
    for name in names:
        exp = EXPERIMENTS[name]
        print(f"  {name:18s} -> {output_suffix(args.prefix, name)}")
    for name in names:
        exp = EXPERIMENTS[name]
        run_rc = run_experiment(
            exp,
            prefix=args.prefix,
            force=args.force,
            dry_run=args.dry_run,
            stop_on_existing=False,
        )
        if run_rc != 0:
            rc = rc or run_rc
        if run_rc != 0 and not args.continue_on_fail:
            break

    if args.summary and not args.dry_run:
        if rc == 0:
            summarize(args.prefix, names)
        else:
            print("Skipping summary because at least one experiment failed.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
