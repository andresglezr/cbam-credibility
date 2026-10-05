"""Data-loading layer for paper assets.

All figure and table builders ingest data through this module so that the
expected schema is documented in a single place.  Each loader returns a
``pandas.DataFrame`` validated against an explicit list of required columns.
The functions are intentionally narrow: they only know how to open the files
under :data:`RESULTS_DIR`.  Anything that involves reshaping for a specific
chart lives in :mod:`figures` or :mod:`robustness`.

The directory ``RESULTS_DIR`` is configurable via the ``CBAM_RESULTS_DIR``
environment variable for off-tree reproduction; otherwise it defaults to the
repository's ``results/`` folder, located two levels above this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _HERE.parent.parent.parent / "results"
RESULTS_DIR = Path(os.environ.get("CBAM_RESULTS_DIR", _DEFAULT_RESULTS))


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing!r}")
    return df


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------


#: New central run: endogenous EU benchmark, indirect-electricity accounting
#: only for cement and fertilizers, and no chemicals extension.  Historical
#: runs remain readable through ``PAPER_RUN_SUFFIX`` (for example
#: ``PAPER_RUN_SUFFIX=v3eubm`` or ``v3nochm``), but the default intentionally
#: fails until the new central files exist instead of silently loading stale
#: unsuffixed outputs.
DEFAULT_RUN_SUFFIX = "newblk"
RUN_SUFFIX = os.environ.get("PAPER_RUN_SUFFIX", DEFAULT_RUN_SUFFIX)
DEFAULT_SENSITIVITY_SUMMARY = "sensitivity_summary_newblk.csv"


def _output_path(stem: str, suffix: str | None, extension: str = ".parquet") -> Path:
    """Resolve an exact output suffix, with one historical compatibility rule.

    New callers pass complete suffixes (``sens_newblk_base``).  Older paper code
    passed ``base`` and expected an implicit ``sens_`` prefix, so that path is
    tried only when the exact file does not exist.
    """
    selected = RUN_SUFFIX if suffix is None else suffix
    exact = RESULTS_DIR / f"{stem}_{selected}{extension}"
    if exact.exists() or suffix is None or str(suffix).startswith("sens_"):
        return exact
    historical = RESULTS_DIR / f"{stem}_sens_{suffix}{extension}"
    return historical if historical.exists() else exact


def load_results_3d(suffix: str | None = None) -> pd.DataFrame:
    """Load the (eq, var, reg, sec, year)-indexed results.

    Parameters
    ----------
    suffix
        Complete output suffix (for example ``"sens_newblk_armington_low"``).
        Historical short names such as ``"armington_low"`` are resolved to
        ``sens_armington_low`` when that old file exists. ``None`` selects
        :data:`RUN_SUFFIX`.
    """
    if suffix is None:
        path = _output_path("results_3d", None)
    else:
        path = _output_path("results_3d", suffix)
    if not path.exists():
        raise FileNotFoundError(f"Expected results file is missing: {path}")
    df = pd.read_parquet(path)
    return _require_columns(df, ["eq", "var", "reg", "sec", "year", "value"], path.name)


def load_results_2d(suffix: str | None = None) -> pd.DataFrame:
    """Load the (eq, var, reg, year)-indexed results (no sector dimension)."""
    if suffix is None:
        path = _output_path("results_2d", None)
    else:
        path = _output_path("results_2d", suffix)
    if not path.exists():
        raise FileNotFoundError(f"Expected results file is missing: {path}")
    df = pd.read_parquet(path)
    return _require_columns(df, ["eq", "var", "reg", "year", "value"], path.name)


def load_results_4d(suffix: str | None = None) -> pd.DataFrame:
    """Load the (eq, var, importer, sector, partner, year) results."""
    path = _output_path("results_4d", suffix)
    if not path.exists():
        raise FileNotFoundError(f"Expected results file is missing: {path}")
    df = pd.read_parquet(path)
    return _require_columns(
        df, ["eq", "var", "reg", "sec", "partner", "year", "value"], path.name
    )


def load_parameters(suffix: str | None = None) -> pd.DataFrame:
    """Load a run's parameter and policy-metadata table."""
    path = _output_path("parameters", suffix)
    if not path.exists():
        raise FileNotFoundError(f"Expected parameters file is missing: {path}")
    df = pd.read_parquet(path)
    return _require_columns(df, ["param", "value"], path.name)


def load_sam(suffix: str | None = None) -> pd.DataFrame:
    """Load the calibrated benchmark SAM used for output-weight aggregation."""
    path = _output_path("sam_data", suffix)
    if not path.exists():
        raise FileNotFoundError(f"Expected SAM file is missing: {path}")
    df = pd.read_parquet(path)
    return _require_columns(df, ["var", "reg", "sec", "value"], path.name)


def load_welfare(suffix: str | None = None) -> pd.DataFrame:
    """Load consumption-equivalent welfare deltas across regions and scopes."""
    path = _output_path("welfare", suffix, extension=".csv")
    if not path.exists():
        raise FileNotFoundError(f"Expected welfare file is missing: {path}")
    df = pd.read_csv(path)
    return _require_columns(
        df,
        ["scope", "reg", "eq", "finite_ce_vs_Eq2_pct"],
        path.name,
    )


def load_sensitivity_summary() -> pd.DataFrame:
    """Headline statistics across the one-at-a-time sensitivity grid."""
    configured = os.environ.get("PAPER_SENSITIVITY_SUMMARY")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = RESULTS_DIR / path
    elif RUN_SUFFIX in {"v3nochm", "v3eubm", "v3", "v2"}:
        path = RESULTS_DIR / "sensitivity_summary.csv"
    else:
        path = RESULTS_DIR / DEFAULT_SENSITIVITY_SUMMARY
    if not path.exists():
        raise FileNotFoundError(f"Expected sensitivity summary is missing: {path}")
    df = pd.read_csv(path)
    required = [
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
    return _require_columns(df, required, path.name)


# ---------------------------------------------------------------------------
# Convenience selectors
# ---------------------------------------------------------------------------


def scalar_parameter(params: pd.DataFrame, name: str, default: float | None = None) -> float:
    """Return the scalar value of a calibration parameter."""
    row = params.loc[params["param"].eq(name)]
    if row.empty:
        if default is None:
            raise KeyError(name)
        return float(default)
    return float(row.iloc[0]["value"])


def credibility_index(params: pd.DataFrame) -> float:
    """Return deterministic :math:`\\kappa`, reading historical metadata too."""
    for name in ("CREDIBILITY_INDEX", "CREDIBILITY_P"):
        row = params.loc[params["param"].eq(name)]
        if not row.empty:
            return float(row.iloc[0]["value"])
    return 0.0


AW_ALIASES = {"AW", "AnticipatedWithdrawal", "Eq4_AnticipatedWithdrawal"}


def resolve_equilibrium_key(
    df: pd.DataFrame,
    eq: str,
    params: pd.DataFrame | None = None,
) -> str:
    """Resolve publication aliases to keys stored in the result parquets.

    ``Eq4_Phase1`` is Anticipated Withdrawal only in the central
    :math:`\\kappa=0` calibration.  Interior-kappa phase-1 solves describe
    deterministic intermediate perceived paths and are never relabelled AW.
    """
    if eq not in AW_ALIASES:
        return eq
    if params is not None and abs(credibility_index(params)) > 1e-12:
        raise ValueError("AnticipatedWithdrawal is defined only for a kappa=0 run")
    candidates = ("Eq4_AnticipatedWithdrawal", "Eq4_Phase1")
    available = set(df["eq"].dropna().astype(str)) if "eq" in df else set()
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError("No AnticipatedWithdrawal/Eq4_Phase1 equilibrium in results")


def free_allocation_schedule(params: pd.DataFrame, t_max: int = 14) -> dict[int, float]:
    """Return the ``T -> phi`` mapping from the parameters table."""
    fa = params.loc[params["param"].eq("FA_schedule")]
    out = {int(row["T"]): float(row["value"]) for _, row in fa.iterrows() if pd.notna(row["T"])}
    # Pad to t_max with zero (post-2034 phase ends).
    last = max(out.keys()) if out else 0
    for t in range(last + 1, t_max + 1):
        out[t] = 0.0
    return out


def scenario_free_allocation_schedule(params: pd.DataFrame, eq: str) -> dict[int, float]:
    """Reconstruct the realized/perceived free-allocation path for a scenario.

    New parameter files carry explicit ``scenario_fa_path`` rows keyed by
    scenario in ``partner``.  The formula fallback keeps historical runs
    readable and mirrors the model's scenario definitions.
    """
    eq_key = "Eq4_Phase1" if eq in AW_ALIASES else eq
    meta = params.loc[params["param"].eq("scenario_fa_path")].copy()
    if not meta.empty and "partner" in meta:
        aliases = {eq_key, eq} | (AW_ALIASES if eq in AW_ALIASES else set())
        rows = meta.loc[meta["partner"].astype(str).isin(aliases)]
        if not rows.empty:
            return {int(row["T"]): float(row["value"]) for _, row in rows.iterrows()}

    t_max = int(scalar_parameter(params, "T_MAX", 30))
    t_star = int(scalar_parameter(params, "T_STAR", 8))
    legislated = free_allocation_schedule(params, t_max=t_max)
    kappa = credibility_index(params)
    if eq in AW_ALIASES and abs(kappa) > 1e-12:
        raise ValueError("AnticipatedWithdrawal is defined only for a kappa=0 run")

    path: dict[int, float] = {}
    for t in range(1, t_max + 1):
        phi = legislated.get(t, 0.0)
        if eq_key == "Eq1_NoPol":
            path[t] = 1.0
        elif eq_key in {"Eq3_Reneg", "Eq4_Phase1"} and t >= t_star:
            path[t] = 1.0 if eq_key == "Eq3_Reneg" else kappa * phi + (1.0 - kappa)
        else:
            path[t] = phi
    return path


def scenario_cbam_schedule(params: pd.DataFrame, eq: str) -> dict[int, float]:
    """Return the scenario's effective CBAM multiplier by model period.

    Core paths use zero/one.  Interior deterministic-kappa phase-1 paths use
    ``kappa`` after revelation so the border component is continuous at the
    no-policy endpoint.
    """
    eq_key = "Eq4_Phase1" if eq in AW_ALIASES else eq
    meta = params.loc[params["param"].eq("scenario_cbam_on")].copy()
    if not meta.empty and "partner" in meta:
        aliases = {eq_key, eq} | (AW_ALIASES if eq in AW_ALIASES else set())
        rows = meta.loc[meta["partner"].astype(str).isin(aliases)]
        if not rows.empty:
            return {int(row["T"]): float(row["value"]) for _, row in rows.iterrows()}
    t_star = int(scalar_parameter(params, "T_STAR", 8))
    kappa = credibility_index(params)
    out: dict[int, float] = {}
    for t, phi in scenario_free_allocation_schedule(params, eq).items():
        scale = float(eq_key != "Eq1_NoPol" and phi < 1.0 - 1e-12)
        if eq_key == "Eq4_Phase1" and t >= t_star:
            scale *= kappa
        out[t] = scale
    return out


def scenario_benchmark_weight_schedule(params: pd.DataFrame, eq: str) -> dict[int, float]:
    """Return the effective benchmark deduction weight ``cbam_on * phi``."""
    eq_key = "Eq4_Phase1" if eq in AW_ALIASES else eq
    meta = params.loc[params["param"].eq("scenario_cbam_benchmark_weight")].copy()
    if not meta.empty and "partner" in meta:
        aliases = {eq_key, eq} | (AW_ALIASES if eq in AW_ALIASES else set())
        rows = meta.loc[meta["partner"].astype(str).isin(aliases)]
        if not rows.empty:
            return {int(row["T"]): float(row["value"]) for _, row in rows.iterrows()}
    fa = scenario_free_allocation_schedule(params, eq)
    on = scenario_cbam_schedule(params, eq)
    return {t: on.get(t, 0.0) * phi for t, phi in fa.items()}


__all__ = [
    "RESULTS_DIR",
    "DEFAULT_RUN_SUFFIX",
    "RUN_SUFFIX",
    "load_results_3d",
    "load_results_2d",
    "load_results_4d",
    "load_parameters",
    "load_sam",
    "load_welfare",
    "load_sensitivity_summary",
    "scalar_parameter",
    "credibility_index",
    "resolve_equilibrium_key",
    "free_allocation_schedule",
    "scenario_free_allocation_schedule",
    "scenario_cbam_schedule",
    "scenario_benchmark_weight_schedule",
]
