"""Generate LaTeX tables for the paper.

The module emits 13 generated ``tex`` fragments under ``tables/`` that
the manuscript includes. The four-sector intensity table
``tab_intensity_intra_eu.tex`` is the one manually maintained fragment and is
not overwritten here. Each generated table is built from the same validated
parquet/CSV result namespaces used for the figures, so their numbers cannot
drift apart.

Conventions
-----------
*   We use ``booktabs`` (``\\toprule``, ``\\midrule``, ``\\bottomrule``)
    everywhere and never vertical rules.
*   Numeric columns use plain right-aligned ``r`` columns, with signs and
    decimal precision formatted explicitly by this module.
*   ``threeparttable`` carries footnotes that explain non-obvious columns.
*   Tables that travel together (sectoral disaggregations) share a numeric
    format so the reader can compare cells across them without reformatting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import style
import data_io
from figures import structural_annual_emissions
from data_io import (
    credibility_index,
    load_parameters,
    load_results_3d,
    load_sam,
    load_sensitivity_summary,
    load_welfare,
    scalar_parameter,
)


COVERED = ["STL", "CEM", "ALU", "FER"]
EU_BLOCKS = ["EUC", "EUM", "EUD"]
T_STAR = 2029
T_STAR_INTEGER_TOLERANCE = 1e-9


def _validated_t_star_year(value: object, *, family: str) -> int:
    """Convert a model-period ``T_STAR`` to a year without truncation.

    Sensitivity metadata should encode an integer model period.  Accept only
    floating-point noise around an integer and use ``round`` explicitly, so a
    malformed value such as 8.9 cannot silently select 2029 through ``int``.
    """
    try:
        raw = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{family}: T_STAR={value!r} is not numeric") from exc
    if not np.isfinite(raw):
        raise ValueError(f"{family}: T_STAR={raw!r} is not finite")
    rounded = round(raw)
    if abs(raw - rounded) > T_STAR_INTEGER_TOLERANCE:
        raise ValueError(
            f"{family}: T_STAR={raw!r} is not within "
            f"{T_STAR_INTEGER_TOLERANCE:g} of an integer model period"
        )
    return 2021 + int(rounded)


def _validated_eu_invp_sum(
    r3: pd.DataFrame,
    *,
    eq: str,
    sec: str,
    year: int,
    family: str,
) -> float:
    """Sum INVP only after proving one row exists for every EU block."""
    sub = r3[
        (r3["eq"] == eq)
        & (r3["var"] == "INVP")
        & r3["reg"].isin(EU_BLOCKS)
        & (r3["sec"] == sec)
        & (r3["year"] == year)
    ]
    counts = sub.groupby("reg", observed=True).size().to_dict()
    expected = {reg: 1 for reg in EU_BLOCKS}
    if counts != expected:
        observed = {reg: int(counts.get(reg, 0)) for reg in EU_BLOCKS}
        raise ValueError(
            f"{family}: expected exactly one INVP row per EU block for "
            f"eq={eq}, sec={sec}, year={year}; counts={observed}"
        )
    values = sub["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{family}: non-finite INVP value for eq={eq}, sec={sec}, year={year}"
        )
    return float(values.sum())


def _sectoral_gap_revelation(
    r3: pd.DataFrame,
    *,
    sec: str,
    t_star_year: int,
    family: str,
) -> float:
    """Sectoral SI-minus-CC investment gap used by the table."""
    sums = {
        eq: _validated_eu_invp_sum(
            r3,
            eq=eq,
            sec=sec,
            year=t_star_year,
            family=family,
        )
        for eq in ("Eq1_NoPol", "Eq2_Credible", "Eq4_Surprise")
    }
    base = sums["Eq1_NoPol"]
    if base == 0:
        raise ValueError(
            f"{family}: zero EU INVP base for sec={sec}, year={t_star_year}"
        )
    si_pp = 100.0 * (sums["Eq4_Surprise"] / base - 1.0)
    cc_pp = 100.0 * (sums["Eq2_Credible"] / base - 1.0)
    return si_pp - cc_pp


def _fmt_pct(v: float, digits: int = 1, sign: bool = True) -> str:
    if pd.isna(v):
        return "--"
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return f"${s}$"


def _fmt_mt(v: float, digits: int = 1) -> str:
    if pd.isna(v):
        return "--"
    return f"${v:.{digits}f}$"


def _write_table(out_dir: Path, name: str, tex: str) -> None:
    """Write the full table float and, next to it, its bare tabular as ``<name>_body.tex``.

    The body file lets a manuscript keep its own caption and notes while the
    numbers still come from the generated cells.
    """
    (out_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
    body = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}\n", tex, flags=re.S)
    if body is None:
        raise ValueError(f"{name}: no tabular environment to export")
    (out_dir / f"{name}_body.tex").write_text(body.group(0), encoding="utf-8")


def _stitched_value(r3: pd.DataFrame, eq: str, var: str, reg: Iterable[str],
                    sec: str, year: int) -> float:
    """Scalar from a complete reported path, stitching SI/RN when required.

    NP and CC are stored as complete paths. Surprise and Reneging select their
    belief equilibrium before revelation and their realized equilibrium from
    revelation onward. The AW alias resolves to the complete central-kappa-zero
    ``Eq4_Phase1`` path.
    """
    if eq == "Eq3_Reneg":
        belief_eq = "Eq2_Credible"
    elif eq == "Eq4_Surprise":
        belief_eq = "Eq4_Phase1"
    elif eq in data_io.AW_ALIASES:
        belief_eq = None
        eq = data_io.resolve_equilibrium_key(r3, eq)
    else:
        belief_eq = None
    mask = (r3["var"] == var) & r3["reg"].isin(list(reg)) & (r3["sec"] == sec) & (r3["year"] == year)
    if belief_eq is None or year >= T_STAR:
        eq_use = eq
    else:
        eq_use = belief_eq
    sub = r3.loc[mask & (r3["eq"] == eq_use), "value"]
    if sub.empty:
        return float("nan")
    return float(sub.sum() if var == "INVP" else sub.mean())


# ---------------------------------------------------------------------------
# Table — CBAM protection
# ---------------------------------------------------------------------------


def make_cbam_protection_table(out_dir: Path) -> None:
    r3 = load_results_3d()
    sam = load_sam()
    weights = sam.loc[(sam["var"] == "xd") & sam["reg"].isin(EU_BLOCKS), ["reg", "sec", "value"]]
    weights = weights.rename(columns={"value": "w"})

    rows = []
    for sec in COVERED:
        out = {}
        for eq in ["Eq2_Credible", "Eq2_NoCBAM"]:
            sub = r3[(r3["eq"] == eq) & (r3["var"] == "H_XD") &
                     r3["reg"].isin(EU_BLOCKS) & (r3["sec"] == sec) & (r3["year"] == 2034)]
            sub = sub.merge(weights[weights["sec"] == sec], on=["reg", "sec"], how="left")
            if sub["w"].sum() == 0 or sub.empty:
                out[eq] = float("nan")
            else:
                out[eq] = float(np.average(sub["value"], weights=sub["w"]))
        with_cbam = 100.0 * (out["Eq2_Credible"] - 1.0)
        without_cbam = 100.0 * (out["Eq2_NoCBAM"] - 1.0)
        protection = with_cbam - without_cbam
        rows.append((style.SECTOR_LABELS[sec], without_cbam, with_cbam, protection))

    body = "\n".join(
        f"{r[0]:<12} & {_fmt_pct(r[1])} & {_fmt_pct(r[2])} & {_fmt_pct(r[3])} \\\\"
        for r in rows
    )
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{EU covered-sector output in 2034 with and without the border charge. "
        "Output-weighted aggregation across the three EU blocks. Percentages relative to the "
        "no-policy balanced-growth path. Protection equals the gap between the CBAM-on and "
        "CBAM-off equilibria within a credibly announced phase-out.}\n"
        "\\label{tab:cbam_protection}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Sector & No CBAM & With CBAM & Protection (pp) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Output-weighted EU aggregate uses benchmark 2022 output as the weight. "
        "Protection is defined as covered-sector output with the CBAM minus output without "
        "the CBAM, in percentage points, computed from unrounded values.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_cbam_protection", tex)


# ---------------------------------------------------------------------------
# Table — Investment by sector around T*
# ---------------------------------------------------------------------------


def make_investment_table(out_dir: Path) -> None:
    r3 = load_results_3d()
    years = [2028, 2029, 2030, 2034]
    eqs = ["Eq2_Credible", "Eq4_Surprise", "Eq3_Reneg"]

    rows = []
    for sec in COVERED:
        cells: list[str] = [style.SECTOR_LABELS[sec]]
        # Base path
        base_y: dict[int, float] = {}
        for y in years:
            sub = r3[(r3["eq"] == "Eq1_NoPol") & (r3["var"] == "INVP")
                     & r3["reg"].isin(EU_BLOCKS) & (r3["sec"] == sec) & (r3["year"] == y)]
            base_y[y] = float(sub["value"].sum()) if not sub.empty else float("nan")
        for eq in eqs:
            for y in years:
                val = _stitched_value(r3, eq, "INVP", EU_BLOCKS, sec, y)
                base = base_y[y]
                if base and not pd.isna(val):
                    pct = 100.0 * (val / base - 1.0)
                    cells.append(_fmt_pct(pct))
                else:
                    cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")

    body = "\n".join(rows)
    header_year = "& \\multicolumn{4}{c}{Credible} & \\multicolumn{4}{c}{Surprise} & \\multicolumn{4}{c}{Reneging} \\\\"
    cmid = ("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
    subheader = "Sector & 2028 & 2029 & 2030 & 2034 & 2028 & 2029 & 2030 & 2034 & 2028 & 2029 & 2030 & 2034 \\\\"

    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{threeparttable}\n"
        "\\caption{EU investment in CBAM-covered sectors, percentage change relative to the no-policy "
        "balanced-growth path, by credibility regime. The revelation date $T^{\\ast}$ falls in 2029; "
        "the table reports the years immediately around revelation and 2034 without imposing a "
        "sign or timing pattern.}\n"
        "\\label{tab:investment}\n"
        "\\begin{tabular}{l rrrr rrrr rrrr}\n"
        "\\toprule\n"
        f"{header_year}\n"
        f"{cmid}\n"
        f"{subheader}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_investment", tex)


# ---------------------------------------------------------------------------
# Table — Intra-EU output by block in 2034
# ---------------------------------------------------------------------------


def make_intra_eu_table(out_dir: Path) -> None:
    r3 = load_results_3d()
    rows = []
    for sec in ["STL", "CEM", "ALU", "FER"]:
        cells = [style.SECTOR_LABELS[sec]]
        for reg in EU_BLOCKS:
            sub = r3[(r3["eq"] == "Eq2_Credible") & (r3["var"] == "H_XD")
                     & (r3["reg"] == reg) & (r3["sec"] == sec) & (r3["year"] == 2034)]
            if sub.empty:
                cells.append("--")
            else:
                pct = 100.0 * (float(sub["value"].iloc[0]) - 1.0)
                cells.append(_fmt_pct(pct))
        rows.append(" & ".join(cells) + " \\\\")

    body = "\n".join(rows)
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{Intra-EU output in 2034 under Credible commitment, percentage change relative to "
        "the no-policy balanced-growth path, by EU block and CBAM-covered sector. Signs and magnitudes "
        "are taken directly from the recalculated central equilibrium.}\n"
        "\\label{tab:intra_eu}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Sector & EU clean & EU middle & EU dirty \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    (out_dir / "tab_intra_eu.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table — Global reallocation in 2034
# ---------------------------------------------------------------------------


def make_global_realloc_table(out_dir: Path) -> None:
    r3 = load_results_3d()
    sectors = ["STL", "CEM", "ALU", "FER"]
    regions = list(style.REGION_LABELS.keys())
    rows = []
    for reg in regions:
        cells = [reg]
        for sec in sectors:
            sub = r3[(r3["eq"] == "Eq2_Credible") & (r3["var"] == "H_XD")
                     & (r3["reg"] == reg) & (r3["sec"] == sec) & (r3["year"] == 2034)]
            if sub.empty:
                cells.append("--")
            else:
                pct = 100.0 * (float(sub["value"].iloc[0]) - 1.0)
                cells.append(_fmt_pct(pct))
        rows.append(" & ".join(cells) + " \\\\")

    body = "\n".join(rows)
    sector_header = " & " + " & ".join(style.SECTOR_LABELS[s] for s in sectors) + " \\\\"
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{Global reallocation of output in 2034 under Credible commitment, percentage "
        "change relative to the no-policy balanced-growth path, by region and CBAM-covered sector. "
        "The table reports the recalculated regional and sectoral values without prespecifying their "
        "signs or ranking.}\n"
        "\\label{tab:global_realloc}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        f"Region {sector_header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    (out_dir / "tab_global_realloc.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table — Sectoral robustness of the core one-at-a-time panel
# ---------------------------------------------------------------------------


def make_sectoral_robustness_table(out_dir: Path) -> None:
    """Sectoral disaggregation across the core one-at-a-time grid.

    Cell ``(sector, variant)`` reports the EU-aggregate revelation-year
    investment gap between Surprise and Credible commitment in percentage
    points; no sign restriction is imposed across variants. The separate
    adjustment-cost grid is reported by its dedicated figure and table.
    """
    suffixes = [
        ("sens_newblk_base",               "Central"),
        ("sens_newblk_sigma_en_low",       "$\\sigma_{EN}=0.25$"),
        ("sens_newblk_sigma_en_high",      "$\\sigma_{EN}=0.75$"),
        ("sens_newblk_armington_low",      "Armington $\\times 0.75$"),
        ("sens_newblk_armington_high",     "Armington $\\times 1.25$"),
        ("sens_newblk_carbon_75",          "$p^{CO_2}=75$"),
        ("sens_newblk_carbon_125",         "$p^{CO_2}=125$"),
        ("sens_newblk_cbam_direct_only",   "CBAM direct"),
        ("sens_newblk_tstar_2028",         "$T^{\\ast}=2028$"),
        ("sens_newblk_tstar_2030",         "$T^{\\ast}=2030$"),
    ]

    ss = load_sensitivity_summary().set_index("suffix")
    if "source_suffix" not in ss.columns:
        raise ValueError(
            "sensitivity summary lacks source_suffix provenance required by "
            "the sectoral robustness table"
        )
    rows = []
    for suffix, label in suffixes:
        source_suffix = str(ss.at[suffix, "source_suffix"]).strip()
        if not source_suffix or source_suffix.lower() == "nan":
            raise ValueError(f"{suffix}: empty source_suffix in sensitivity summary")
        r3 = load_results_3d(suffix=source_suffix)
        t_star_year = _validated_t_star_year(
            ss.at[suffix, "T_STAR"], family=suffix
        )
        row_cells = [label]
        for sec in COVERED:
            gap = _sectoral_gap_revelation(
                r3,
                sec=sec,
                t_star_year=t_star_year,
                family=f"{suffix}<-{source_suffix}",
            )
            row_cells.append(_fmt_pct(gap, digits=2))
        rows.append(" & ".join(row_cells) + " \\\\")

    body = "\n".join(rows)
    secs_header = " & " + " & ".join(style.SECTOR_LABELS[s] for s in COVERED) + " \\\\"
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{threeparttable}\n"
        "\\caption{Sectoral robustness of the revelation-year investment gap, Surprise minus "
        "Credible commitment, percentage points relative to the no-policy balanced-growth path. "
        "Every cell is regenerated from the endogenous-EU-benchmark sensitivity runs; signs and "
        "magnitudes are reported without imposing the qualitative pattern from earlier runs.}\n"
        "\\label{tab:sectoral_robustness}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        f"Variant {secs_header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Each row comes from the complete output family underlying the five reported "
        "policy paths (NP, CC, SI, RN, and AW), with one calibration object perturbed relative "
        "to the central run. AW reuses the correctly anticipated phase-1 path rather than a "
        "redundant solve. All other parameters are held at their benchmark values. The "
        "revelation year is the variant-specific $T^{\\ast}$ (2029 in the central row).\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    (out_dir / "tab_sectoral_robustness.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table — Core one-at-a-time sensitivity
# ---------------------------------------------------------------------------


def make_sensitivity_core_table(out_dir: Path) -> None:
    """One-at-a-time sensitivity of the central credibility statistics.

    Auto-generated from ``sensitivity_summary_newblk.csv`` with validated pipeline
    statistics, so the table cannot drift from the figures (tornado, heatmap,
    adjustment sensitivity), which read from the same source. The
    cost-of-disbelief
    column reports the sensitivity pipeline's cumulative global emissions cost
    under the same structural (two-channel) accounting as the headline
    Table~\\ref{tab:emissions}: process emissions scale with output and
    combustion emissions with the realized refined-fuel demand. The central
    row is regenerated from the same current central result namespace.
    """
    ss = load_sensitivity_summary().set_index("suffix")
    rows_spec = [
        ("Central calibration",            "sens_newblk_base"),
        ("Armington elasticities $\\times 1.25$", "sens_newblk_armington_high"),
        ("Armington elasticities $\\times 0.75$", "sens_newblk_armington_low"),
        ("$\\sigma_{EN}=0.75$",            "sens_newblk_sigma_en_high"),
        ("$\\sigma_{EN}=0.25$",            "sens_newblk_sigma_en_low"),
        ("$p^{CO_2}=125$ EUR/tCO$_2$",     "sens_newblk_carbon_125"),
        ("$p^{CO_2}=75$ EUR/tCO$_2$",      "sens_newblk_carbon_75"),
        ("CBAM, direct emissions only",    "sens_newblk_cbam_direct_only"),
        ("$T^{\\ast}=2028$",               "sens_newblk_tstar_2028"),
        ("$T^{\\ast}=2030$",               "sens_newblk_tstar_2030"),
    ]
    rows = []
    for label, suffix in rows_spec:
        r = ss.loc[suffix]
        rows.append((
            label,
            float(r["inv_gap_pre_pp"]),
            float(r["inv_gap_revelation_pp"]),
            float(r["inv_gap_post_pp"]),
            float(r["emissions_si_minus_cc_mt"]),
            float(r["cbam_effect_2034_pp"]),
        ))

    body = "\n".join(
        f"{label:<38} & {_fmt_pct(a, 2)} & {_fmt_pct(b, 2)} & {_fmt_pct(c, 2)} & "
        f"{_fmt_mt(d, 1)} & {_fmt_pct(e, 2)} \\\\"
        for (label, a, b, c, d, e) in rows
    )
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{threeparttable}\n"
        "\\caption{One-at-a-time sensitivity of the central credibility statistics. Each row "
        "varies one calibration object relative to the central run while holding all others at "
        "their benchmark values. Columns 1--3 are the EU-aggregate investment gap between "
        "Surprise and Credible commitment, in percentage points relative to the no-policy "
        "balanced-growth path, in the year before $T^{\\ast}$, at $T^{\\ast}$, and one year "
        "after. Column 4 is the cumulative global emissions cost of disbelief, in megatonnes "
        "CO$_2$ equivalent. Column 5 is the 2034 effect of the modeled "
        "origin-intensity border charge on output in the EU represented industrial sectors.}\n"
        "\\label{tab:sensitivity_core}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "  & \\multicolumn{3}{c}{Inv.\\ gap (SI--CC), pp vs.\\ NP}"
        " & Cost of disbelief & CBAM eff. \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "Variant & $T^{\\ast}{-}1$ & $T^{\\ast}$ & $T^{\\ast}{+}1$"
        " & (global, Mt) & (2034, pp) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Central calibration: $\\sigma_{EN}=0.5$, $\\sigma_{VAE}=0.5$, $\\sigma_{ELC}=3.0$, "
        "sector-specific Armington elasticities at GTAP central values, $p^{CO_2}=100$ EUR/tCO$_2$, "
        "$T^{\\ast}=2029$, endogenous EU benchmark, and indirect-electricity emissions "
        "included only for cement and fertilizers.\n"
        "\\item Positive values for the cost of disbelief mean that Surprise generates more "
        "cumulative emissions than Credible commitment. The emissions column uses the same "
        "structural accounting as Table~\\ref{tab:emissions} and the headline results: "
        "process emissions scale with output and combustion emissions with actual "
        "refined-fuel demand at the realized fuel wedge.\n"
        "\\item Investment columns are indexed to the variant-specific $T^{\\ast}$.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_sensitivity_core", tex)


# ---------------------------------------------------------------------------
# Table — Adjustment-cost sensitivity of the credibility channel
# ---------------------------------------------------------------------------


def make_adjustment_sensitivity_table(out_dir: Path) -> None:
    """Response of the credibility statistics to the adjustment-cost share.

    Four points on the eta-bar grid: the dedicated low and high variants, the
    central run, and the steep high variant.  This is the numerical companion
    to ``fig_adjustment_sensitivity`` and is built from the same rows of the
    sensitivity summary, so the table and the figure cannot drift. The grid
    reports the response at four adjustment-cost shares without imposing a
    monotonicity restriction.
    """
    ss = load_sensitivity_summary().set_index("suffix")
    points = [
        ("sens_newblk_adj_cost_low",    "0.015", ""),
        ("sens_newblk_base",            "0.050", " (central)"),
        ("sens_newblk_adj_cost_high",   "0.060", ""),
        ("sens_newblk_adj_cost_higher", "0.100", ""),
    ]
    rows = []
    for suffix, eta, tag in points:
        r = ss.loc[suffix]
        label = f"${eta}${tag}"
        rows.append(
            f"{label:<22} & {_fmt_pct(float(r['inv_gap_pre_pp']), 2)} & "
            f"{_fmt_pct(float(r['inv_gap_revelation_pp']), 2)} & "
            f"{_fmt_pct(float(r['inv_gap_post_pp']), 2)} & "
            f"{_fmt_mt(float(r['emissions_si_minus_cc_mt']), 1)} & "
            f"{_fmt_pct(float(r['cbam_effect_2034_pp']), 2)} \\\\"
        )
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{Sensitivity of the credibility channel to the adjustment-cost "
        "share $\\bar\\eta$. Each row comes from the complete output family underlying "
        "the five reported policy paths (NP, CC, SI, RN, and AW), with $\\bar\\eta$ moved "
        "away from its central value of $0.05$. AW reuses the correctly anticipated "
        "phase-1 path rather than a redundant solve. The exogenous data and the parameters "
        "outside the experiment are held fixed, while the objects the calibration derives "
        "from $\\bar\\eta$ (the benchmark capital stock, the depreciation rate, the "
        "adjustment-cost intensities $\\psi_{rs}$, and the productivity normalizations that "
        "depend on the capital stock) are recomputed: each row is a recalibrated economy "
        "(online appendix). "
        "Columns 1--3 are the EU-aggregate investment gap between Surprise and "
        "Credible commitment, in percentage points relative to the no-policy "
        "balanced-growth path, in the year before $T^{\\ast}$, at $T^{\\ast}$, "
        "and one year after. Column 4 is the cumulative global emissions cost "
        "of disbelief. Column 5 is the 2034 effect of the modeled origin-intensity "
        "border charge on output in the EU represented industrial sectors.}\n"
        "\\label{tab:adjustment_sensitivity}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "  & \\multicolumn{3}{c}{Inv.\\ gap (SI--CC), pp vs.\\ NP}"
        " & Cost of disbelief & CBAM eff. \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "$\\bar\\eta$ & $T^{\\ast}{-}1$ & $T^{\\ast}$ & $T^{\\ast}{+}1$"
        " & (global, Mt) & (2034, pp) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item The adjustment-cost share sets benchmark adjustment costs as a "
        "fraction of benchmark gross investment in each sector-region cell; see "
        "equation \\eqref{eq:psi_calibration}. Each row reports the recomputed "
        "investment and emissions statistics at that value of $\\bar\\eta$; no "
        "monotonic response is assumed in advance.\n"
        "\\item The grid stops short of the degenerate limit $\\bar\\eta \\to 1$, "
        "at which benchmark adjustment costs would absorb the whole of gross "
        "investment and the calibrated shadow-value system loses economic "
        "content.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_adjustment_sensitivity", tex)


# ---------------------------------------------------------------------------
# Table — Welfare
# ---------------------------------------------------------------------------


def make_welfare_table(out_dir: Path) -> None:
    w = load_welfare()
    # We want the CE delta vs Credible commitment.  Sign convention: positive
    # means the scenario yields higher welfare than Credible commitment.
    # Published values are the finite-horizon (2022-2051) consumption
    # equivalents. The separately exported terminal-tail diagnostic is not
    # used in this table.
    target = w[
        w["eq"].isin(["Eq2_NoCBAM", "Eq3_Reneg", "Eq4_Surprise", "Eq4_Phase1"])
    ]
    target = target.pivot_table(index=["scope", "reg"], columns="eq",
                                 values="finite_ce_vs_Eq2_pct").reset_index()

    label_map = {
        ("EU", "EU"):           "EU (aggregate)",
        ("region", "EUC"):      "\\quad EU clean",
        ("region", "EUM"):      "\\quad EU middle",
        ("region", "EUD"):      "\\quad EU dirty",
        ("region", "GBR"):      "United Kingdom",
        ("region", "CHN"):      "China",
        ("region", "IND"):      "India",
        ("region", "RUS"):      "Russia",
        ("region", "TUR"):      "Turkey",
        ("region", "OCD"):      "Rest OECD",
        ("region", "ROW"):      "Rest of world",
        ("World", "World"):     "World",
    }
    order = list(label_map.keys())

    rows = []
    for key in order:
        scope, reg = key
        sub = target[(target["scope"] == scope) & (target["reg"] == reg)]
        if sub.empty:
            continue
        a = float(sub["Eq2_NoCBAM"].iloc[0])
        b = float(sub["Eq3_Reneg"].iloc[0])
        c = float(sub["Eq4_Surprise"].iloc[0])
        d = float(sub["Eq4_Phase1"].iloc[0])
        rows.append((label_map[key], a, b, c, d))

    body = "\n".join(
        f"{label:<24} & {_fmt_pct(a, 4)} & {_fmt_pct(b, 4)} & {_fmt_pct(c, 4)} & {_fmt_pct(d, 4)} \\\\"
        for (label, a, b, c, d) in rows
    )
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{Welfare relative to Credible commitment, by region and scenario, over the "
        "modeled horizon 2022--2051. Values are consumption-equivalent variations in percent: "
        "positive means the scenario yields higher welfare than Credible commitment, negative "
        "means lower. Anticipated Withdrawal (AW) reuses the $\\kappa=0$ phase-1 "
        "equilibrium and shares Reneging's realized post-2029 policy. The no-policy "
        "benchmark is omitted so that the columns compare each alternative directly with the "
        "credibly announced policy path.}\n"
        "\\label{tab:welfare}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Region & No CBAM & Reneging & Surprise & AW \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_welfare", tex)


# ---------------------------------------------------------------------------
# Table — Cumulative emissions and cost of disbelief (headline, Section 7)
# ---------------------------------------------------------------------------


def make_emissions_table(out_dir: Path) -> None:
    """Cumulative emission savings and scenario differences.

    The table uses the model's own two-channel accounting: process emissions
    scale with output and combustion emissions with the sector's actual
    refined-fuel demand at the realized fuel wedge. Surprise and Reneging read
    from the stitched belief/realised paths (``Eq4_Surprise``,
    ``Eq3_Reneg``).
    """
    from figures import structural_annual_emissions

    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()

    # Model's structural (two-channel, realized wedge) accounting throughout:
    # process emissions scale with output, combustion emissions with actual
    # refined-fuel demand at the realized fuel wedge.
    struct: dict[str, dict[str, float]] = {"EU": {}, "World": {}}
    for scope in ["EU", "World"]:
        for eq in ["Eq1_NoPol", "Eq2_Credible", "Eq4_Surprise", "Eq3_Reneg"]:
            struct[scope][eq] = structural_annual_emissions(r3, params, sam, eq, scope).sum()
    struct["Non-EU"] = {
        eq: struct["World"][eq] - struct["EU"][eq]
        for eq in struct["World"]
    }

    def panel_rows(scopes: dict[str, dict[str, float]]) -> str:
        rows = []
        for label in ["EU", "Non-EU", "World"]:
            t = scopes[label]
            benefit = t["Eq1_NoPol"] - t["Eq2_Credible"]
            si_loss = t["Eq4_Surprise"] - t["Eq2_Credible"]
            rn_loss = t["Eq3_Reneg"] - t["Eq2_Credible"]
            si_pct = 100.0 * si_loss / benefit
            rn_pct = 100.0 * rn_loss / benefit
            # Snap sub-0.05 magnitudes to zero so a tiny negative float never
            # renders as "-0.0" (a spurious "slightly negative" reading).
            si_pct = 0.0 if abs(si_pct) < 0.05 else si_pct
            rn_pct = 0.0 if abs(rn_pct) < 0.05 else rn_pct
            si_pct_txt = f"{si_pct:.1f}" if abs(si_pct) < 10 else f"{round(si_pct):d}"
            rn_pct_txt = f"{rn_pct:.1f}" if abs(rn_pct) < 10 else f"{round(rn_pct):d}"
            rows.append(
                f"{label} & {round(benefit):d} & {round(si_loss):+d} & {si_pct_txt} & "
                f"{round(rn_loss):+d} & {rn_pct_txt} \\\\"
            )
        return "\n".join(rows)

    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Cumulative CO$_2$ emission savings of credible commitment relative to no "
        "policy, and cumulative-emissions differences under each credibility regime, "
        "2022--2051, megatonnes. Emissions use the model's structural accounting: process "
        "emissions scale with output and combustion emissions with the sector's actual "
        "refined-fuel demand at the realized fuel wedge. Scenario differences are expressed "
        "as a percentage of the modeled policy benefit; positive values denote more emissions "
        "than under Credible commitment. These total scenario differences are "
        "descriptive: Reneging changes both pre-revelation expectations and the realized "
        "long-run policy. Table~\\ref{tab:withdrawal_decomposition} uses Anticipated "
        "Withdrawal to hold the realized policy fixed and splits the resulting difference "
        "at the revelation date.}\n"
        "\\label{tab:emissions}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{1}{c}{Modeled benefit} & \\multicolumn{2}{c}{Surprise}"
        " & \\multicolumn{2}{c}{Reneging} \\\\\n"
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Scope & (Credible vs.\\ NoPol, Mt) & $\\Delta E$ (Mt) & Share (\\%) & $\\Delta E$ (Mt) & "
        "Share (\\%) \\\\\n"
        "\\midrule\n"
        f"{panel_rows(struct)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    _write_table(out_dir, "tab_emissions", tex)


def make_withdrawal_decomposition_table(out_dir: Path) -> None:
    """Decompose reversal while distinguishing anticipation from inherited state."""
    from figures import withdrawal_decomposition_data

    annual = withdrawal_decomposition_data()
    params = load_parameters()
    t_star_year = _validated_t_star_year(
        scalar_parameter(params, "T_STAR", 8),
        family=data_io.RUN_SUFFIX,
    )
    totals = annual.groupby("scope", as_index=False)[
        ["rn_minus_aw_mt", "aw_minus_cc_mt", "rn_minus_cc_mt", "residual_mt"]
    ].sum()
    pre = (
        annual.loc[annual["year"].lt(t_star_year)]
        .groupby("scope", as_index=False)["rn_minus_aw_mt"]
        .sum()
        .rename(columns={"rn_minus_aw_mt": "rn_aw_pre_mt"})
    )
    post = (
        annual.loc[annual["year"].ge(t_star_year)]
        .groupby("scope", as_index=False)["rn_minus_aw_mt"]
        .sum()
        .rename(columns={"rn_minus_aw_mt": "rn_aw_post_mt"})
    )
    totals = totals.merge(pre, on="scope", validate="one_to_one").merge(
        post, on="scope", validate="one_to_one"
    )
    split_residual = (
        totals["rn_minus_aw_mt"] - totals["rn_aw_pre_mt"] - totals["rn_aw_post_mt"]
    ).abs()
    if float(split_residual.max()) > 1e-9:
        raise RuntimeError(
            "RN--AW pre/post split does not sum to the cumulative total: "
            f"max residual={float(split_residual.max()):.3e}"
        )
    rows = []
    for scope in ("EU", "World"):
        row = totals.loc[totals["scope"].eq(scope)].iloc[0]
        rows.append(
            f"{scope} & {row['rn_aw_pre_mt']:.1f} & {row['rn_aw_post_mt']:.1f} & "
            f"{row['rn_minus_aw_mt']:.1f} & {row['aw_minus_cc_mt']:.1f} & "
            f"{row['rn_minus_cc_mt']:.1f} & {row['residual_mt']:.2e} \\\\"
        )
    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Decomposition of cumulative emissions under policy reversal, "
        "2022--2051 (Mt CO$_2$). Anticipated Withdrawal (AW) has the same realized "
        "free-allocation and CBAM paths as Reneging (RN), but expectations differ before "
        f"$T^*={t_star_year}$. The pre-$T^*$ RN--AW column is the anticipatory difference "
        "before withdrawal. From $T^*$ onward, both paths face the same policy and installed "
        "capital is the only inherited state, so the post-$T^*$ column is the inherited-state "
        "component of the emissions difference. Their sum is the total same-realized-policy "
        "expectations effect. AW--CC compares two correctly anticipated policy paths. The "
        "identity RN--CC=(RN--AW)+(AW--CC) is accounting-exact for the cumulative totals.}\n"
        "\\label{tab:withdrawal_decomposition}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{3}{c}{RN--AW: same realized policy} & & & \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "Scope & Pre-$T^*$ & From $T^*$ & Total & AW--CC & RN--CC & Residual \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    (out_dir / "tab_withdrawal_decomposition.tex").write_text(tex, encoding="utf-8")
    annual.to_csv(out_dir / "withdrawal_decomposition_annual.csv", index=False)
    withdrawal_welfare_subperiod_data().to_csv(
        out_dir / "withdrawal_welfare_subperiods.csv", index=False
    )


def withdrawal_welfare_subperiod_data() -> pd.DataFrame:
    """RN relative to AW consumption equivalents before and after revelation.

    RN and AW share the same realized policy but have different beliefs before
    revelation. The full-horizon RN--AW consumption equivalent therefore mixes
    pre-revelation choices with the consequences of the capital inherited at
    T_STAR. This diagnostic applies the paper's exact finite-horizon CIES
    formula separately to 2022--T_STAR-1 and T_STAR--2051. The subperiod
    consumption equivalents are not additive.
    """
    from figures import _stitched

    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()
    t_star_year = _validated_t_star_year(
        scalar_parameter(params, "T_STAR", 8),
        family=data_io.RUN_SUFFIX,
    )
    g = scalar_parameter(params, "g", 0.02)
    ro = scalar_parameter(params, "ro", 0.05)
    gamma = scalar_parameter(params, "gamma", 2.0)
    adjsh = scalar_parameter(params, "adjsh", 0.05)
    beta = (1.0 + g) / (1.0 + ro)
    if not (np.isfinite(beta) and beta > 0 and np.isfinite(gamma) and gamma > 0):
        raise ValueError(f"Invalid welfare parameters beta={beta}, gamma={gamma}")

    alpha = params.loc[
        params["param"].eq("alphaH"), ["reg", "sec", "value"]
    ].rename(columns={"value": "alphaH"})
    alpha["alphaH"] = alpha["alphaH"].astype(float)
    alpha["alphaH_norm"] = alpha["alphaH"] / alpha.groupby("reg")["alphaH"].transform("sum")
    regs = sorted(alpha["reg"].unique())

    def consumption_index(eq: str) -> pd.DataFrame:
        h = _stitched(r3, eq, "H_C", regs, params=params)
        h = h.merge(alpha[["reg", "sec", "alphaH_norm"]], on=["reg", "sec"], how="left")
        if h["alphaH_norm"].isna().any() or (h["value"] <= 0).any():
            raise RuntimeError(f"{eq}: invalid consumption hats or alphaH weights")
        h["weighted_log"] = h["alphaH_norm"] * np.log(h["value"].astype(float))
        out = h.groupby(["reg", "year"], as_index=False)["weighted_log"].sum()
        out[eq] = np.exp(out["weighted_log"])
        return out[["reg", "year", eq]]

    paths = consumption_index("Eq3_Reneg").merge(
        consumption_index("AW"), on=["reg", "year"], validate="one_to_one"
    )

    c_raw = sam.loc[sam["var"].eq("c_raw"), ["reg", "sec", "value"]].rename(
        columns={"value": "c_raw"}
    )
    inv = sam.loc[sam["var"].eq("iz"), ["reg", "sec", "value"]].rename(
        columns={"value": "iz"}
    )
    cbase = c_raw.merge(inv, on=["reg", "sec"], how="outer").fillna(0.0)
    cbase["c_adj"] = cbase["c_raw"] - adjsh * cbase["iz"]
    reg_weights = cbase.groupby("reg", as_index=False)["c_adj"].sum()
    if not np.isfinite(reg_weights["c_adj"]).all() or (reg_weights["c_adj"] <= 0).any():
        raise RuntimeError("Invalid adjusted benchmark-consumption weights")

    def ce_factor(block: pd.DataFrame) -> float:
        c = block["Eq3_Reneg"].to_numpy(dtype=float)
        ref = block["AW"].to_numpy(dtype=float)
        w = beta ** (block["year"].to_numpy(dtype=int) - 2022)
        if abs(gamma - 1.0) < 1e-10:
            return float(np.exp(np.sum(w * (np.log(c) - np.log(ref))) / np.sum(w)))
        power = 1.0 - gamma
        ratio = np.sum(w * c**power) / np.sum(w * ref**power)
        return float(ratio ** (1.0 / power))

    periods = [
        ("pre_revelation", paths["year"].lt(t_star_year)),
        ("post_revelation", paths["year"].ge(t_star_year)),
        ("full_horizon", pd.Series(True, index=paths.index)),
    ]
    rows: list[dict[str, object]] = []
    for period, mask in periods:
        selected = paths.loc[mask]
        factors = selected.groupby("reg", sort=True).apply(
            ce_factor, include_groups=False
        ).rename("ce_factor").reset_index()
        for scope, scope_regs in (
            ("EU", [r for r in regs if str(r).startswith("EU")]),
            ("World", regs),
        ):
            tmp = factors.loc[factors["reg"].isin(scope_regs)].merge(
                reg_weights, on="reg", validate="one_to_one"
            )
            factor = float(np.average(tmp["ce_factor"], weights=tmp["c_adj"]))
            rows.append(
                {
                    "scope": scope,
                    "period": period,
                    "start_year": int(selected["year"].min()),
                    "end_year": int(selected["year"].max()),
                    "rn_vs_aw_ce_pct": 100.0 * (factor - 1.0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table — Sectoral investment timing of the credibility gap (central run)
# ---------------------------------------------------------------------------


def make_robust_investment_timing_table(out_dir: Path) -> None:
    """Per-sector EU-aggregate investment gap (Surprise minus Credible) at the
    three pivot years, relative to the no-policy balanced-growth path.

    Built from the central run so it cannot drift from the rest of the
    pipeline. The base is the no-policy ``INVP`` path, identical to the base
    used by :func:`make_investment_table` and
    :func:`make_sectoral_robustness_table`. The Surprise series is read through
    ``_stitched`` (never a raw equilibrium key), so the pre-revelation year
    2028 always carries the believed, not the realized, path -- the year where
    the two segments would differ if read without stitching.
    """
    from figures import _stitched

    r3 = load_results_3d()
    years = [2028, 2029, 2030]
    stitched = {
        eq: _stitched(r3, eq, "INVP", EU_BLOCKS)
        for eq in ["Eq1_NoPol", "Eq4_Surprise", "Eq2_Credible"]
    }

    def agg(eq: str, sec: str, year: int) -> float:
        d = stitched[eq]
        sub = d[(d["sec"] == sec) & (d["year"] == year)]
        return float(sub["value"].sum()) if not sub.empty else float("nan")

    rows = []
    for sec in COVERED:
        cells = [style.SECTOR_LABELS[sec]]
        for y in years:
            base = agg("Eq1_NoPol", sec, y)
            si = agg("Eq4_Surprise", sec, y)
            cc = agg("Eq2_Credible", sec, y)
            if base and not (pd.isna(si) or pd.isna(cc)):
                gap = 100.0 * (si / base - 1.0) - 100.0 * (cc / base - 1.0)
                cells.append(_fmt_pct(gap, digits=2))
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)

    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Investment timing of the credibility gap. Surprise minus credible "
        "commitment, EU-aggregate investment ($INVP$) in covered sectors, percentage points "
        "relative to the no-policy balanced-growth path. The columns report the year before "
        "revelation (2028), the revelation year (2029), and one year after (2030), without "
        "imposing a common sign pattern across sectors.}\n"
        "\\label{tab:robust_investment_timing}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Sector & 2028 & 2029 & 2030 \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    (out_dir / "tab_robust_investment_timing.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table — Cost of disbelief across the deterministic kappa sweep
# ---------------------------------------------------------------------------


def make_credibility_curve_table(out_dir: Path) -> None:
    """Numerical companion to ``fig_credibility_curve``.

    Same five-point sweep, same structural accounting; the table adds the
    SCC-200 climate cost in euros and realized steel investment in the
    revelation year.  The latter is conditioned on capital inherited from the
    corresponding pre-revelation perception, so the curve can be read in Mt,
    EUR, and the catch-up-investment mechanism without leaving the table.
    """
    import data_io
    import figures as figs

    df = figs.credibility_curve_data().sort_values("kappa")

    rows = []
    for _, row in df.iterrows():
        kappa = row["kappa"]
        suffix = str(row["suffix"])
        outcome_eq = str(row["outcome_eq"])
        r3 = pd.read_parquet(data_io.RESULTS_DIR / f"results_3d_{suffix}.parquet")
        params = pd.read_parquet(data_io.RESULTS_DIR / f"parameters_{suffix}.parquet")
        g = scalar_parameter(params, "g", 0.02)
        inv_b = params[params["param"] == "inv"][["reg", "sec", "value"]].rename(columns={"value": "b"})
        ip = r3[(r3["var"] == "INVP") & r3["reg"].isin(EU_BLOCKS) & (r3["sec"] == "STL") & (r3["year"] == 2029)]
        ip = ip.merge(inv_b, on=["reg", "sec"])
        ip["hat"] = ip["value"] / (ip["b"] * (1.0 + g) ** (2029 - 2022))
        si = ip[ip["eq"] == outcome_eq]
        npol = ip[ip["eq"] == "Eq1_NoPol"]
        si_w = (si["hat"] * si["b"]).sum() / si["b"].sum()
        np_w = (npol["hat"] * npol["b"]).sum() / npol["b"].sum()
        jump = 100.0 * (si_w / np_w - 1.0)
        def z(v: float) -> float:
            return 0.0 if abs(v) < 0.05 else v
        rows.append(
            f"${kappa:.2f}$ & {z(row['cost_EU_mt']):.1f} & {z(row['cost_World_mt']):.1f} & "
            f"{z(row['cost_EU_mt']*0.2):.1f} & {z(row['cost_World_mt']*0.2):.1f} & {jump:+.1f} \\\\"
        )
    body = "\n".join(rows)

    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Emissions along deterministic perceived-policy paths indexed by "
        "$\\kappa$. Values between zero and one interpolate the post-revelation "
        "free-allocation schedule deterministically; they are not policy-survival "
        "probabilities or expected stochastic outcomes. $\\kappa=0$ is full disbelief "
        "and $\\kappa=1$ reuses the credible equilibrium. "
        "Cumulative emission losses use the structural accounting of Table~\\ref{tab:emissions}; "
        "climate cost is evaluated at a social cost of carbon of EUR 200/tCO$_2$. "
        "The final column is EU steel investment in the revelation year (2029), "
        "after the full modeled continuation is revealed and relative to no policy; "
        "differences across $\\kappa$ reflect the capital inherited from each "
        "pre-revelation perceived path.}\n"
        "\\label{tab:credibility_curve}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c}{Cost of disbelief (Mt)} & \\multicolumn{2}{c}{Climate cost (EUR bn)} & Steel investment \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "$\\kappa$ & EU & World & EU & World & 2029 (\\%) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    (out_dir / "tab_credibility_curve.tex").write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table — Monetized social cost of credibility failures (SCC + CE welfare)
# ---------------------------------------------------------------------------


def make_scc_table(out_dir: Path) -> None:
    """Monetize the credibility costs: climate externality plus private CE.

    The climate component values the cumulative 2022--2051 structural
    emission change of each regime (relative to credible commitment) at a
    constant social cost of carbon.  A constant SCC is the value, at the date
    of emission, of the damages that tonne causes; summing across emission
    years therefore mixes valuation dates, so the table also reports the
    stream discounted to 2022 at rho.  The annual series are levels that
    already embed the (1+g)^t trend, so rho applies to them directly; this is
    the same convention as the private column, where the benchmark consumption
    base grows at g and is discounted at rho.
    The climate value of the "EU" scope is the global damage attributable to
    tonnes emitted in the EU, not the damage borne by EU residents.  The
    private component converts the finite-horizon (2022--2051)
    consumption-equivalent welfare change vs. credible into euros: annual
    flow on the benchmark consumption base, discounted at rho.
    CE welfare excludes climate damages by construction.  The two components
    differ in scope (global externality vs. residents' welfare) and, for the
    undiscounted climate column, in valuation date, so they are reported side
    by side and deliberately not summed.
    """
    from figures import structural_annual_emissions

    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()
    wel = load_welfare()

    g = scalar_parameter(params, "g", 0.02)
    ro = scalar_parameter(params, "ro", 0.05)
    adjsh = scalar_parameter(params, "adjsh", 0.05)
    t_max = int(scalar_parameter(params, "T_MAX", 30))
    beta = (1.0 + g) / (1.0 + ro)
    # Finite-horizon annuity over the modeled years, matching the finite CE.
    pv_factor = (1.0 - beta ** t_max) / (1.0 - beta)

    # Annual structural emission paths (Mt) and their cumulative totals.
    ann: dict[str, dict[str, pd.Series]] = {}
    tot: dict[str, dict[str, float]] = {}
    for scope in ["EU", "World"]:
        ann[scope] = {
            eq: structural_annual_emissions(r3, params, sam, eq, scope)
            for eq in ["Eq2_Credible", "Eq4_Surprise", "AW", "Eq3_Reneg"]
        }
        tot[scope] = {eq: float(ser.sum()) for eq, ser in ann[scope].items()}
    regime_eq = {
        "Disbelief (Surprise)": "Eq4_Surprise",
        "Anticipated withdrawal (AW)": "AW",
        "Reneging": "Eq3_Reneg",
    }
    d_emis = {
        (regime, scope): tot[scope][eq] - tot[scope]["Eq2_Credible"]
        for regime, eq in regime_eq.items()
        for scope in ["EU", "World"]
    }
    # The same differences discounted to 2022, tonne by tonne, at rho.  The
    # annual series are levels that already embed the (1+g)^t balanced-growth
    # trend, so the level flow is discounted at rho directly.  This is the
    # same convention as the private column, where C0 * beta^t is a flow
    # growing at g discounted at rho.  Applying beta to a level flow would
    # discount it at an effective (1+rho)/(1+g)-1 = 2.9 percent instead.
    d_emis_pv = {}
    for regime, eq in regime_eq.items():
        for scope in ["EU", "World"]:
            diff = ann[scope][eq] - ann[scope]["Eq2_Credible"]
            d_emis_pv[(regime, scope)] = float(
                sum(float(v) / (1.0 + ro) ** (int(year) - 2022) for year, v in diff.items())
            )

    # Benchmark consumption base in EUR bn (1 model unit = EUR 1 bn).
    c_raw = sam.loc[sam["var"] == "c_raw", ["reg", "sec", "value"]].rename(columns={"value": "c_raw"})
    iz = sam.loc[sam["var"] == "iz", ["reg", "sec", "value"]].rename(columns={"value": "iz"})
    base = c_raw.merge(iz, on=["reg", "sec"], how="outer").fillna(0.0)
    base["c_adj"] = base["c_raw"] - adjsh * base["iz"]
    by_reg = base.groupby("reg")["c_adj"].sum()
    cons_bn = {
        "EU": float(by_reg[[r for r in by_reg.index if str(r).startswith("EU")]].sum()),
        "World": float(by_reg.sum()),
    }

    # Private CE welfare vs credible commitment (positive pct = gain).
    agg = wel[wel["scope"].isin(["EU", "World"])]
    ce_eq = {
        "Disbelief (Surprise)": "Eq4_Surprise",
        "Anticipated withdrawal (AW)": "Eq4_Phase1",
        "Reneging": "Eq3_Reneg",
    }
    ce_pct = {
        (regime, scope): float(
            agg.loc[
                agg["reg"].eq(scope) & agg["eq"].eq(eq),
                "finite_ce_vs_Eq2_pct",
            ].iloc[0]
        )
        for regime, eq in ce_eq.items()
        for scope in ["EU", "World"]
    }

    SCC = [100.0, 200.0, 400.0]  # EUR per tCO2
    rows_tex = []
    rows_csv = []
    for regime in regime_eq:
        for scope in ["EU", "World"]:
            de = d_emis[(regime, scope)]                      # Mt vs credible
            climate = [de * scc / 1000.0 for scc in SCC]      # EUR bn, cumulative
            climate_pv = d_emis_pv[(regime, scope)] * SCC[1] / 1000.0  # EUR bn, 2022 PV at SCC 200
            priv_cost_bn = -(ce_pct[(regime, scope)] / 100.0) * cons_bn[scope] * pv_factor
            short = {"Disbelief (Surprise)": "SI", "Anticipated withdrawal (AW)": "AW", "Reneging": "RN"}[regime]
            rows_tex.append(
                f"{short} & {scope} & {de:+.0f} & "
                f"{climate[0]:.1f} & {climate[1]:.1f} & {climate[2]:.1f} & "
                f"{climate_pv:.1f} & {priv_cost_bn:+.1f} \\\\"
            )
            rows_csv.append(
                {
                    "regime": regime,
                    "scope": scope,
                    "d_emissions_mt_vs_cc": de,
                    "climate_cost_bn_scc100": climate[0],
                    "climate_cost_bn_scc200": climate[1],
                    "climate_cost_bn_scc400": climate[2],
                    "climate_cost_bn_scc200_pv2022": climate_pv,
                    "private_ce_cost_bn_pv": priv_cost_bn,
                }
            )
        if regime != "Reneging":
            rows_tex.append("\\addlinespace")

    body = "\n".join(rows_tex)
    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Climate value and private welfare cost relative to credible commitment, "
        "EUR bn. Climate columns: the cumulative 2022--2051 structural emission change valued at "
        "a constant SCC (each tonne at its own emission year), and the same stream discounted to "
        "2022 at $\\rho$ (SCC 200). The EU climate value is the global damage from tonnes emitted "
        "in the EU, not the damage borne by EU residents. Private CE: consumption-equivalent "
        "welfare change of the scope's residents as a 2022 present value at $\\rho$ on the "
        "benchmark consumption base, reported as a cost (positive = welfare loss); excludes "
        "climate damages. The components are not summed "
        "(Section~\\ref{subsec:scc}). SI: Surprise; AW: Anticipated withdrawal; RN: Reneging.}\n"
        "\\label{tab:scc}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llrrrrrr}\n"
        "\\toprule\n"
        " & & $\\Delta E$ & \\multicolumn{3}{c}{Climate value, cumulative} & "
        "Climate PV & Private CE \\\\\n"
        "\\cmidrule(lr){4-6}\n"
        "Regime & Scope & (Mt) & SCC 100 & SCC 200 & SCC 400 & "
        "2022, SCC 200 & PV \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    (out_dir / "tab_scc.tex").write_text(tex, encoding="utf-8")
    pd.DataFrame(rows_csv).to_csv(out_dir / "scc_summary.csv", index=False)


def leakage_data() -> dict[str, pd.Series]:
    """Cumulative 2022-2051 emission changes by zone and sector for the leakage table."""
    r3, params, sam = load_results_3d(), load_parameters(), load_sam()
    cum = {
        eq: structural_annual_emissions(r3, params, sam, eq, "World", by_zone_sector=True)
        for eq in ("Eq1_NoPol", "Eq2_NoCBAM", "Eq2_Credible", "Eq4_Surprise")
    }
    return {
        "nocbam": cum["Eq2_NoCBAM"] - cum["Eq1_NoPol"],
        "cc": cum["Eq2_Credible"] - cum["Eq1_NoPol"],
        "si": cum["Eq4_Surprise"] - cum["Eq2_Credible"],
    }


def make_leakage_table(out_dir: Path) -> None:
    d = leakage_data()
    cols = [d["nocbam"], d["cc"], d["si"]]

    def zone(s: pd.Series, z: str) -> float:
        return float(s.loc[z].sum())

    def secs(s: pd.Series, names: Sequence[str]) -> float:
        return float(s.loc["nonEU"].reindex(list(names)).fillna(0.0).sum())

    rest = [x for x in d["cc"].loc["nonEU"].index if x not in ("STL", "CEM", "ELF")]
    rows = [
        ("EU", [zone(c, "EU") for c in cols]),
        ("Non-EU", [zone(c, "nonEU") for c in cols]),
        ("\\quad steel", [secs(c, ["STL"]) for c in cols]),
        ("\\quad cement", [secs(c, ["CEM"]) for c in cols]),
        ("\\quad fossil electricity", [secs(c, ["ELF"]) for c in cols]),
        ("\\quad other sectors", [secs(c, rest) for c in cols]),
    ]
    body = "\n".join(f"{name} & " + " & ".join(f"${v:+.0f}$" for v in vals) + " \\\\" for name, vals in rows)
    rates = [100.0 * zone(c, "nonEU") / (-zone(c, "EU")) for c in cols[:2]]
    rate_row = "Leakage rate (\\%) & " + " & ".join(f"${r:+.0f}$" for r in rates) + " & -- \\\\"
    tex = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\begin{threeparttable}\n"
        "\\caption{Carbon leakage: cumulative change in emissions, 2022--2051 (Mt CO$_2$). The first two "
        "columns compare each policy with No policy, the third compares Surprise with Credible commitment. "
        "The leakage rate is the change in non-EU emissions divided by the reduction in EU emissions.}\n"
        "\\label{tab:leakage}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        " & Phase-out only & Phase-out and CBAM & Disbelief (SI$-$CC) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\midrule\n"
        f"{rate_row}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Structural emission accounting as in Table~\\ref{tab:emissions}. Phase-out only is the credible "
        "domestic phase-out without the border charge. The indented rows split the non-EU total by sector.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )
    _write_table(out_dir, "tab_leakage", tex)


def main(out_dir: Path | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing tables to: {out_dir}")
    make_cbam_protection_table(out_dir)
    print("  [ok] tab_cbam_protection")
    make_emissions_table(out_dir)
    print("  [ok] tab_emissions")
    make_withdrawal_decomposition_table(out_dir)
    print("  [ok] tab_withdrawal_decomposition")
    make_robust_investment_timing_table(out_dir)
    print("  [ok] tab_robust_investment_timing")
    make_intra_eu_table(out_dir)
    print("  [ok] tab_intra_eu")
    make_global_realloc_table(out_dir)
    print("  [ok] tab_global_realloc")
    make_investment_table(out_dir)
    print("  [ok] tab_investment")
    make_welfare_table(out_dir)
    print("  [ok] tab_welfare")
    make_sectoral_robustness_table(out_dir)
    print("  [ok] tab_sectoral_robustness")
    make_adjustment_sensitivity_table(out_dir)
    print("  [ok] tab_adjustment_sensitivity")
    make_scc_table(out_dir)
    print("  [ok] tab_scc")
    make_credibility_curve_table(out_dir)
    print("  [ok] tab_credibility_curve")
    make_sensitivity_core_table(out_dir)
    print("  [ok] tab_sensitivity_core")
    make_leakage_table(out_dir)
    print("  [ok] tab_leakage")


if __name__ == "__main__":
    main()
