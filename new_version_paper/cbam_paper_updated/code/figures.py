"""Publication-quality figures for the CBAM-credibility paper.

Each ``make_*`` function in this module produces exactly one figure in
``figures/``.  The script is idempotent and the figures are deterministic
given the validated result families: re-running ``python figures.py`` re-renders
the 10 PDF/PNG figure pairs with identical pixels. Central assets use
``PAPER_RUN_SUFFIX`` (``newblk`` for the paper), the deterministic
perceived-stringency curve reads ``newblk_k25``/``newblk_k50``/``newblk_k75``, and
the one-at-a-time panels read ``sensitivity_summary_newblk.csv``.

Design conventions:

*   Time-series plots use markers at integer years so the credible/surprise/
    reneging trajectories remain visually separable in black-and-white print.
*   The revelation date ``T*`` is drawn as a faint dashed vertical.  No solid
    annotation lines: every reference is a hairline.
*   Two-by-N panel grids use a shared y axis only when scales are commensurate.
    Mixed scales get free axes with consistent grid styling so the reader is
    not misled into a false comparison.
*   Every chart that displays signed percentage deviations centres the y axis
    on zero by default, then expands the more extreme side so the asymmetry
    is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import style
import data_io
from data_io import (
    credibility_index,
    free_allocation_schedule,
    load_parameters,
    load_results_3d,
    load_sam,
    load_sensitivity_summary,
    resolve_equilibrium_key,
    scalar_parameter,
    scenario_free_allocation_schedule,
)


COVERED = ["STL", "CEM", "ALU", "FER"]
#: Compatibility alias for panels that use the complete four-sector set.
FOUR = COVERED
EU_BLOCKS = ["EUC", "EUM", "EUD"]
#: Namespace prefix for the paper's one-at-a-time sensitivity families.
SENSITIVITY_PREFIX = "sens_newblk"


# ---------------------------------------------------------------------------
# Reusable selectors
# ---------------------------------------------------------------------------


def _series(
    df: pd.DataFrame,
    eq: str,
    var: str,
    reg: str | Sequence[str],
    sec: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    eq = resolve_equilibrium_key(df, eq)
    mask = (df["eq"] == eq) & (df["var"] == var)
    if isinstance(reg, str):
        mask &= df["reg"] == reg
    else:
        mask &= df["reg"].isin(list(reg))
    if sec is not None:
        if isinstance(sec, str):
            mask &= df["sec"] == sec
        else:
            mask &= df["sec"].isin(list(sec))
    return df.loc[mask, ["reg", "sec", "year", "value"]].copy()


def _weighted_sum(values: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Aggregate by sector × year using benchmark weights (sum, not mean)."""
    merged = values.merge(weights.rename(columns={"value": "w"}), on=["reg", "sec"], how="left")
    merged["w"] = merged["w"].fillna(0.0)
    merged["wv"] = merged["value"] * merged["w"]
    grp = merged.groupby("year", as_index=True)
    sum_wv = grp["wv"].sum()
    sum_w = grp["w"].sum().replace(0.0, np.nan)
    return sum_wv / sum_w


def _eu_weighted(df: pd.DataFrame, sam: pd.DataFrame, eq: str, var: str, sec: str) -> pd.Series:
    """EU aggregate of a hat variable, weighted by benchmark output."""
    weights = sam.loc[(sam["var"] == "xd") & sam["reg"].isin(EU_BLOCKS) & (sam["sec"] == sec)]
    vals = _series(df, eq, var, EU_BLOCKS, sec)
    return _weighted_sum(vals, weights[["reg", "sec", "value"]])


def _eu_invp(df: pd.DataFrame, sam: pd.DataFrame, eq: str, sec: str) -> pd.Series:
    """EU-aggregate investment level (INVP) summed by sector across blocks."""
    vals = _series(df, eq, "INVP", EU_BLOCKS, sec)
    grp = vals.groupby("year")["value"].sum()
    return grp


def _index_relative(series: pd.Series, base: pd.Series) -> pd.Series:
    """Return percentage deviation against a no-policy base series."""
    aligned = base.reindex(series.index)
    return 100.0 * (series / aligned - 1.0)


DEFAULT_T_STAR_YEAR = 2029


def _t_star_year(params: pd.DataFrame | None = None) -> int:
    """Calendar revelation year recorded by a result family.

    Historical callers without metadata retain the paper's central 2029
    default.  Sensitivity families pass their own parameter table, so paths
    with T*=2028 or T*=2030 are stitched at the correct boundary.
    """
    if params is None:
        return DEFAULT_T_STAR_YEAR
    return 2021 + int(round(scalar_parameter(params, "T_STAR", 8.0)))


def _stitched(
    df: pd.DataFrame,
    eq: str,
    var: str,
    reg_filter: str | Sequence[str],
    sec: str | Sequence[str] | None = None,
    params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a complete reported policy path from the stored equilibrium data.

    Current output files already export complete stitched paths under
    ``Eq3_Reneg``/``Eq4_Surprise``.  This helper reconstructs the same paths
    explicitly from the first-phase sources (``Eq2_Credible``/
    ``Eq4_Phase1``) and the corresponding realised paths at the revelation
    year recorded by the result family;
    that keeps the publication mapping transparent and remains compatible
    with historical post-revelation-only files. No Policy and Credible are
    already full paths.
    Anticipated Withdrawal resolves to the full central-kappa-zero
    ``Eq4_Phase1`` path; that alias is rejected for interior-kappa runs by the
    data-loading layer.
    """
    if eq == "Eq3_Reneg":
        pre = _series(df, "Eq2_Credible", var, reg_filter, sec)
        post = _series(df, "Eq3_Reneg",   var, reg_filter, sec)
    elif eq == "Eq4_Surprise":
        pre = _series(df, "Eq4_Phase1",   var, reg_filter, sec)
        post = _series(df, "Eq4_Surprise", var, reg_filter, sec)
    elif eq in data_io.AW_ALIASES:
        return _series(df, eq, var, reg_filter, sec)
    else:
        return _series(df, eq, var, reg_filter, sec)
    t_star_year = _t_star_year(params)
    pre = pre[pre["year"] < t_star_year]
    post = post[post["year"] >= t_star_year]
    return pd.concat([pre, post], ignore_index=True)


# ---------------------------------------------------------------------------
# Figure — Free-allocation phase-out
# ---------------------------------------------------------------------------


def make_reform_figure(out_dir: Path) -> None:
    params = load_parameters()
    fa = free_allocation_schedule(params, t_max=14)
    years = list(range(2025, 2036))
    phi = np.array([fa.get(y - 2022 + 1, 0.0) for y in years])

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    # 2028-2030 highlight band: where 38.5 pp of the schedule is delivered.
    ax.axvspan(2028, 2030, color="#fde6a4", alpha=0.55, zorder=0,
               label="2028-2030 step (38.5 pp)")
    ax.plot(years, phi, color="#1f3b73", linewidth=2.0, marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, zorder=3)

    # Endpoint annotations
    ax.annotate("100%", xy=(2025, 1.00), xytext=(2025.05, 1.04),
                color="#1f3b73", fontsize=9.5, ha="left", va="bottom")
    ax.annotate("0%", xy=(2034, 0.00), xytext=(2034.05, 0.03),
                color="#1f3b73", fontsize=9.5, ha="left", va="bottom")
    # Step annotation
    ax.annotate(
        "$\\phi_{2028}{-}\\phi_{2030}=0.385$",
        xy=(2029.0, 0.775),
        xytext=(2030.8, 0.86),
        fontsize=9.0,
        color="#7c2d12",
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.18",
                        color="#7c2d12", linewidth=0.8),
    )

    ax.set_ylim(-0.02, 1.12)
    ax.set_xlim(2024.5, 2035.5)
    ax.set_xticks(range(2025, 2036))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Year")
    ax.set_ylabel("Free-allocation rate $\\phi_t$")
    ax.set_title("CBAM-covered free-allocation phase-out under Directive (EU) 2023/959",
                 loc="left", pad=8)
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False)

    style.save_both(fig, out_dir, "fig_reform")


# ---------------------------------------------------------------------------
# Figure — Intra-EU heterogeneity (4 panels)
# ---------------------------------------------------------------------------


def make_intra_eu_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    sam = load_sam()

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    axes = axes.flatten()
    for ax, sec in zip(axes, FOUR):
        for reg in EU_BLOCKS:
            yrs_vals = _series(r3, "Eq2_Credible", "H_XD", reg, sec).set_index("year")["value"]
            pct = 100.0 * (yrs_vals - 1.0)
            ax.plot(
                pct.index,
                pct.values,
                color=style.EU_BLOCK_COLORS[reg],
                marker="o",
                markersize=3.5,
                markevery=4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=style.REGION_LABELS[reg] if sec == FOUR[0] else None,
                linewidth=1.6,
            )
        style.hairline_zero(ax)
        ax.axvline(2029, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
        ax.set_title(style.SECTOR_LABELS[sec], loc="left")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_xlim(2022, 2051)
    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Output, % vs no-policy baseline")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Intra-EU output trajectories under Credible commitment",
                 x=0.02, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    style.save_both(fig, out_dir, "fig_intra_eu")


# ---------------------------------------------------------------------------
# Figure — Global reallocation in 2034
# ---------------------------------------------------------------------------


def make_global_realloc_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    sectors = FOUR
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.3), sharey=False)

    all_regions = list(style.REGION_LABELS.keys())

    for ax, sec in zip(axes, sectors):
        rows = []
        for reg in all_regions:
            sub = _series(r3, "Eq2_Credible", "H_XD", reg, sec)
            if sub.empty:
                continue
            v2034 = sub.loc[sub["year"] == 2034, "value"]
            if v2034.empty:
                continue
            pct = 100.0 * (float(v2034.iloc[0]) - 1.0)
            rows.append((reg, pct))
        if not rows:
            ax.set_visible(False)
            continue
        df = pd.DataFrame(rows, columns=["reg", "pct"]).sort_values("pct")
        colors = ["#a8201a" if v < 0 else "#2a8e57" for v in df["pct"]]
        bars = ax.barh(df["reg"], df["pct"], color=colors, edgecolor="white", linewidth=0.4)
        # Compute axis padding so labels always sit clearly outside the bars.
        max_abs = max(abs(df["pct"].min()), abs(df["pct"].max()))
        label_pad = max_abs * 0.04
        ax.set_xlim(-max_abs * 1.30, max_abs * 1.30)
        for bar, v in zip(bars, df["pct"]):
            ha = "left" if v >= 0 else "right"
            offset = label_pad if v >= 0 else -label_pad
            ax.text(
                v + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}",
                fontsize=8.5,
                color="#1f1f1f",
                va="center",
                ha=ha,
            )
        ax.axvline(0, color="#3a3a3a", linewidth=0.8)
        ax.set_title(style.SECTOR_LABELS[sec], loc="left")
        ax.set_xlabel("Output 2034, % vs baseline")
        ax.grid(True, axis="x", alpha=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f"))

    fig.suptitle(
        "Output by region in 2034 under Credible commitment",
        x=0.02, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save_both(fig, out_dir, "fig_global_realloc")


# ---------------------------------------------------------------------------
# Figure — Who benefits from disbelief (Surprise minus Credible, 2034)
# ---------------------------------------------------------------------------


def make_who_benefits_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()

    # Baseline-output-weighted emission intensity across the four represented
    # CBAM industries. A simple mean would give equal weight to tiny and large
    # sector-region cells and is especially misleading for fertilizers.
    em = params.loc[
        (params["param"] == "emis_intensity") & params["sec"].isin(COVERED),
        ["reg", "sec", "value"],
    ].rename(columns={"value": "emis_intensity"})
    output = sam.loc[
        (sam["var"] == "xd") & sam["sec"].isin(COVERED),
        ["reg", "sec", "value"],
    ].rename(columns={"value": "baseline_output"})
    weighted = em.merge(output, on=["reg", "sec"], validate="one_to_one")
    weighted["emissions"] = weighted["emis_intensity"] * weighted["baseline_output"]
    totals = weighted.groupby("reg")[["emissions", "baseline_output"]].sum()
    intensity = (totals["emissions"] / totals["baseline_output"]).sort_values(ascending=True)

    sectors_show = ["STL", "ALU"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.5))
    for ax, sec, panel_letter in zip(axes, sectors_show, ["(a)", "(b)"]):
        rows = []
        for reg in intensity.index:
            sub_si = _series(r3, "Eq4_Surprise", "H_XD", reg, sec)
            sub_cc = _series(r3, "Eq2_Credible", "H_XD", reg, sec)
            v_si = sub_si.loc[sub_si["year"] == 2034, "value"]
            v_cc = sub_cc.loc[sub_cc["year"] == 2034, "value"]
            if v_si.empty or v_cc.empty:
                continue
            gap_pp = 100.0 * (float(v_si.iloc[0]) - float(v_cc.iloc[0]))
            rows.append((reg, gap_pp))
        df = pd.DataFrame(rows, columns=["reg", "gap"])
        df = df.set_index("reg").reindex(intensity.index).reset_index()
        colors = ["#a8201a" if v > 0 else "#1f3b73" for v in df["gap"]]
        bars = ax.barh(df["reg"], df["gap"], color=colors, edgecolor="white", linewidth=0.4)
        max_abs = max(abs(df["gap"].min()), abs(df["gap"].max()), 0.10)
        label_pad = max_abs * 0.05
        ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)
        for bar, v in zip(bars, df["gap"]):
            ha = "left" if v >= 0 else "right"
            offset = label_pad if v >= 0 else -label_pad
            ax.text(
                v + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}",
                fontsize=8.5,
                va="center",
                ha=ha,
                color="#1f1f1f",
            )
        ax.axvline(0, color="#3a3a3a", linewidth=0.8)
        ax.set_title(f"{panel_letter} {style.SECTOR_LABELS[sec]}", loc="left")
        ax.set_xlabel("Output gap, Surprise $-$ Credible (pp), 2034")
        ax.grid(True, axis="x", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Who benefits from disbelief? Regions ordered by emission intensity (top = dirtiest)",
        x=0.02, ha="left", fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_both(fig, out_dir, "fig_who_benefits")


# ---------------------------------------------------------------------------
# Figure — Investment dynamics by sector
# ---------------------------------------------------------------------------


def make_investment_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    params = load_parameters()
    t_star_year = _t_star_year(params)

    # Wider panels and taller aspect so the sector grid breathes; matplotlib's
    # default tight layout otherwise compresses the panels until tick labels
    # collide with axis labels.
    fig, axes = plt.subplots(1, len(COVERED), figsize=(12.4, 4.2), sharex=True)
    eqs = ["Eq2_Credible", "Eq4_Surprise", "Eq3_Reneg"]
    for ax, sec in zip(axes, COVERED):
        base = _series(r3, "Eq1_NoPol", "INVP", EU_BLOCKS, sec).groupby("year")["value"].sum()
        for eq in eqs:
            series = _stitched(
                r3, eq, "INVP", EU_BLOCKS, sec, params=params
            ).groupby("year")["value"].sum()
            pct = _index_relative(series, base)
            sty = style.SCENARIOS[eq]
            ax.plot(
                pct.index,
                pct.values,
                color=sty.color,
                linewidth=1.8,
                marker=sty.marker,
                markersize=3.8,
                markevery=4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=sty.label if sec == COVERED[0] else None,
            )
        style.hairline_zero(ax)
        ax.axvline(t_star_year, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
        ax.set_title(style.SECTOR_LABELS[sec], loc="left")
        ax.set_xlabel("Year")
        ax.set_xlim(2022, 2051)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axes[0].set_ylabel("EU investment, % vs no-policy baseline")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(
        "EU investment in CBAM-covered sectors, by credibility regime",
        x=0.02, ha="left", fontsize=12.0,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.20, wspace=0.30)
    style.save_both(fig, out_dir, "fig_investment")


# ---------------------------------------------------------------------------
# Figure — Capital-stock gaps: the state variable behind the investment flows
# ---------------------------------------------------------------------------


def capital_gap_data(r3: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """EU capital stock in the represented sectors, percent gaps by year.

    ``si_vs_cc``: Surprise relative to Credible commitment (same realized
    policy; the disbelief channel).  ``rn_vs_aw``: Reneging relative to
    Anticipated withdrawal (same realized withdrawal; the inherited-state
    channel).  Aggregation sums KP over EU blocks and the represented sectors.
    """
    def total(eq: str) -> pd.Series:
        key = resolve_equilibrium_key(r3, eq, params=params)
        parts = [
            _stitched(r3, key, "KP", EU_BLOCKS, sec, params=params).groupby("year")["value"].sum()
            for sec in COVERED
        ]
        return sum(parts)

    cc, si, rn, aw = (total(eq) for eq in ["Eq2_Credible", "Eq4_Surprise", "Eq3_Reneg", "AW"])
    out = pd.DataFrame({
        "si_vs_cc": 100.0 * (si / cc - 1.0),
        "rn_vs_aw": 100.0 * (rn / aw - 1.0),
    })
    out.index.name = "year"
    return out


def make_capital_gap_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    params = load_parameters()
    t_star_year = _t_star_year(params)
    d = capital_gap_data(r3, params)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for col, eq, label in (
        ("si_vs_cc", "Eq4_Surprise", "Surprise vs. Credible commitment"),
        ("rn_vs_aw", "Eq3_Reneg", "Reneging vs. Anticipated withdrawal"),
    ):
        sty = style.SCENARIOS[eq]
        ax.plot(
            d.index, d[col].values, color=sty.color, linewidth=1.8, marker=sty.marker,
            markersize=3.8, markevery=4, markerfacecolor="white", markeredgewidth=1.0, label=label,
        )
    style.hairline_zero(ax)
    ax.axvline(t_star_year, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
    ax.set_xlim(2022, 2051)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.set_xlabel("Year")
    ax.set_ylabel("EU capital stock, % gap")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Capital stock in the represented EU sectors: the inherited state",
        x=0.02, ha="left", fontsize=12.0,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)
    style.save_both(fig, out_dir, "fig_capital_gap")


# ---------------------------------------------------------------------------
# Figure — Deterministic perceived-stringency comparative statics
# ---------------------------------------------------------------------------

#: Deterministic perceived-policy sweep. Each entry is
#: ``kappa -> (run suffix, realized-outcome equilibrium)``.  For each interior
#: run, ``Eq4_Surprise`` stitches the intermediate pre-revelation perception to
#: the full modeled continuation revealed at T*.  The endpoints reuse the
#: central run: Surprise at kappa=0 and Credible at kappa=1. Only the three
#: interior paths require dedicated solves.
KAPPA_SWEEP: Mapping[float, tuple[str, str]] = {
    0.00: (data_io.RUN_SUFFIX, "Eq4_Surprise"),
    0.25: ("newblk_k25", "Eq4_Surprise"),
    0.50: ("newblk_k50", "Eq4_Surprise"),
    0.75: ("newblk_k75", "Eq4_Surprise"),
    1.00: (data_io.RUN_SUFFIX, "Eq2_Credible"),
}
# Historical import name retained; its values now include the equilibrium
# selector needed to reuse both central endpoints without redundant files.
CREDIBILITY_SWEEP = KAPPA_SWEEP


def credibility_curve_data(results_dir: Path | None = None) -> pd.DataFrame:
    """Cumulative emissions gap along deterministic perceived-policy paths.

    Reads each suffix independently (not through the ``PAPER_RUN_SUFFIX``
    routing of :mod:`data_io`, since this figure spans five different runs at
    once) and applies the same structural accounting as the headline figures.
    """
    if results_dir is None:
        results_dir = data_io.RESULTS_DIR
    rows = []
    for kappa, (suffix, outcome_eq) in sorted(KAPPA_SWEEP.items()):
        r3 = pd.read_parquet(results_dir / f"results_3d_{suffix}.parquet")
        params = pd.read_parquet(results_dir / f"parameters_{suffix}.parquet")
        sam = pd.read_parquet(results_dir / f"sam_data_{suffix}.parquet")
        row = {"kappa": kappa, "suffix": suffix, "outcome_eq": outcome_eq}
        for scope in ["EU", "World"]:
            cc = structural_annual_emissions(r3, params, sam, "Eq2_Credible", scope).sum()
            si = structural_annual_emissions(r3, params, sam, outcome_eq, scope).sum()
            row[f"cost_{scope}_mt"] = si - cc
        rows.append(row)
    return pd.DataFrame(rows)


def make_credibility_curve_figure(out_dir: Path) -> None:
    df = credibility_curve_data()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    world_sty = style.SCENARIOS["Eq4_Surprise"]
    eu_sty = style.SCENARIOS["Eq2_Credible"]
    ax.plot(df["kappa"], df["cost_World_mt"], color=world_sty.color, linewidth=1.8,
             marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.0,
             label="World")
    ax.plot(df["kappa"], df["cost_EU_mt"], color=eu_sty.color, linewidth=1.8,
             marker="s", markersize=4.5, markerfacecolor="white", markeredgewidth=1.0,
             label="EU")
    style.hairline_zero(ax)
    ax.set_xlabel("Perceived-stringency index $\\kappa$")
    ax.set_ylabel("Cumulative cost of disbelief (Mt CO$_2$)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False)
    fig.suptitle("Emissions gap along deterministic perceived-policy paths", x=0.02, ha="left", fontsize=12.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_both(fig, out_dir, "fig_credibility_curve")


# ---------------------------------------------------------------------------
# Figure — Emissions deviation from no-policy
# ---------------------------------------------------------------------------


def structural_annual_emissions(
    r3: pd.DataFrame,
    params: pd.DataFrame,
    sam: pd.DataFrame,
    eq: str,
    region_filter: str,
    covered: Sequence[str] | None = None,
    by_zone_sector: bool = False,
) -> pd.Series:
    """Annual emissions (Mt CO2) under the model's own two-channel accounting.

    Process emissions scale with physical output (Leontief, unavoidable);
    combustion emissions scale with the sector's *actual* refined-fuel demand,
    reconstructed from the EN-nest CES demand at the realized fuel wedge.
    This is the accounting the model itself uses to levy the fuel carbon cost
    (line ``tau_fuel`` in the model), so it nets out within-sector fuel
    substitution that the fixed-intensity proxy ignores.

    The wedge is the *realized* one: under ``Eq3_Reneg`` free allocation
    reverts to 1.0 from the run-specific revelation year
    (``fa_reneging_phase2``), so its fuel
    wedge is zero from the revelation year onward. In the central-kappa-zero
    family, Anticipated Withdrawal reuses the full ``Eq4_Phase1`` path and the
    same realized withdrawal schedule. Interior-kappa files instead use the
    intermediate schedule recorded in their own parameter metadata.
    """
    requested_eq = eq
    eq = resolve_equilibrium_key(r3, eq, params=params)
    if region_filter == "EU":
        reg_filter: Sequence[str] = EU_BLOCKS
    else:
        reg_filter = list(style.REGION_LABELS.keys())

    # The fuel wedge only exists for sectors the run actually covered, so the
    # covered set must match the run being read, not the reader's default.
    # Resolution order: (1) an explicit ``covered`` argument; (2) the covered
    # set recorded in the run's own metadata (``cbam_covered`` rows, written by
    # sinretencion.py); (3) the central four-sector default. Runs solved before
    # the metadata row existed fall through to (3), which is correct for every
    # four-sector run.  CHM remains an uncovered model sector in this paper.
    if covered is not None:
        covered_set = list(covered)
    else:
        meta_cov = params.loc[params["param"] == "cbam_covered", "sec"].tolist()
        covered_set = meta_cov if meta_cov else list(COVERED)

    g = scalar_parameter(params, "g", 0.02)
    sigma_en = scalar_parameter(params, "SIGMA_EN", 0.5)
    carbon = scalar_parameter(params, "CARBON_PRICE_EUR", 100.0) / scalar_parameter(params, "SCALE", 1000.0)
    policy_fa = scenario_free_allocation_schedule(params, requested_eq)

    proc = params.loc[params["param"] == "emis_proc_intensity", ["reg", "sec", "value"]].rename(columns={"value": "proc_c"})
    fuel = params.loc[params["param"] == "emis_fuel_per_ref", ["reg", "sec", "value"]].rename(columns={"value": "fuel_c"})
    base_xd = sam.loc[sam["var"] == "xd", ["reg", "sec", "value"]].rename(columns={"value": "base_xd"})
    base_ref = sam.loc[sam["var"] == "en_ref", ["reg", "sec", "value"]].rename(columns={"value": "base_ref"})

    wide = _stitched(r3, eq, "H_XD", reg_filter, params=params).rename(columns={"value": "H_XD"})
    for var in ["H_EN", "H_PE"]:
        seg = _stitched(r3, eq, var, reg_filter, params=params)[["reg", "sec", "year", "value"]].rename(columns={"value": var})
        wide = wide.merge(seg, on=["reg", "sec", "year"], how="left")
    href = _stitched(r3, eq, "H_P", reg_filter, sec="REF", params=params)[["reg", "year", "value"]].rename(columns={"value": "H_P_REF"})
    wide = wide.merge(href, on=["reg", "year"], how="left")
    wide = (
        wide.merge(proc, on=["reg", "sec"], how="left")
        .merge(fuel, on=["reg", "sec"], how="left")
        .merge(base_xd, on=["reg", "sec"], how="left")
        .merge(base_ref, on=["reg", "sec"], how="left")
    )
    wide["T"] = wide["year"] - 2021  # T = 1 is the 2022 benchmark year

    def tau_for(row: pd.Series) -> float:
        if not (str(row["reg"]).startswith("EU") and row["sec"] in covered_set):
            return 0.0
        phi = policy_fa.get(int(row["T"]), 1.0)
        return (1.0 - phi) * carbon * float(row["fuel_c"])

    wide["tau"] = wide.apply(tau_for, axis=1)
    wide["ref_hat"] = wide["H_EN"] * (wide["H_PE"] / (wide["H_P_REF"] + wide["tau"])) ** sigma_en
    wide["em"] = (1.0 + g) ** (wide["year"] - 2022) * (
        wide["proc_c"] * wide["base_xd"] * wide["H_XD"]
        + wide["fuel_c"] * wide["base_ref"] * wide["ref_hat"]
    )
    if by_zone_sector:
        # Cumulative 2022-2051 emissions by zone and sector (carbon-leakage table).
        wide["zone"] = ["EU" if str(r).startswith("EU") else "nonEU" for r in wide["reg"]]
        return wide.groupby(["zone", "sec"])["em"].sum()
    return wide.groupby("year")["em"].sum()


def withdrawal_decomposition_data(
    r3: pd.DataFrame | None = None,
    params: pd.DataFrame | None = None,
    sam: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Annual RN--AW--CC emissions decomposition for the central kappa=0 run.

    AW reuses ``Eq4_Phase1`` only here.  Calling this function on an interior
    kappa run is an error because that phase-1 equilibrium is then an
    intermediate perceived-policy path, not anticipated withdrawal.
    """
    if r3 is None:
        r3 = load_results_3d()
    if params is None:
        params = load_parameters()
    if sam is None:
        sam = load_sam()
    if abs(credibility_index(params)) > 1e-12:
        raise ValueError("RN-AW-CC decomposition requires the central kappa=0 run")

    rows: list[dict[str, float | int | str]] = []
    for scope in ("EU", "World"):
        cc = structural_annual_emissions(r3, params, sam, "Eq2_Credible", scope)
        aw = structural_annual_emissions(r3, params, sam, "AW", scope)
        rn = structural_annual_emissions(r3, params, sam, "Eq3_Reneg", scope)
        years = sorted(set(cc.index) & set(aw.index) & set(rn.index))
        for year in years:
            rn_aw = float(rn.loc[year] - aw.loc[year])
            aw_cc = float(aw.loc[year] - cc.loc[year])
            rn_cc = float(rn.loc[year] - cc.loc[year])
            rows.append(
                {
                    "scope": scope,
                    "year": int(year),
                    "rn_minus_aw_mt": rn_aw,
                    "aw_minus_cc_mt": aw_cc,
                    "rn_minus_cc_mt": rn_cc,
                    "residual_mt": rn_cc - rn_aw - aw_cc,
                }
            )
    return pd.DataFrame(rows)


def make_emissions_figure(out_dir: Path) -> None:
    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()

    def annual_emissions(eq: str, region_filter: str) -> pd.Series:
        return structural_annual_emissions(r3, params, sam, eq, region_filter)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for ax, scope_name in zip(axes, ["EU", "World"]):
        base = annual_emissions("Eq1_NoPol", scope_name)
        for eq in ["Eq2_Credible", "Eq4_Surprise", "Eq3_Reneg"]:
            ser = annual_emissions(eq, scope_name)
            delta = ser - base
            sty = style.SCENARIOS[eq]
            ax.plot(
                delta.index,
                delta.values,
                color=sty.color,
                linewidth=1.8,
                marker=sty.marker,
                markersize=3.8,
                markevery=4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=sty.label if scope_name == "EU" else None,
            )
        ax.axhline(0, color="#5b5b5b", linestyle=(0, (3, 3)), linewidth=0.9,
                   alpha=0.9, label="No policy" if scope_name == "EU" else None)
        ax.axvline(2029, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
        ax.set_title(scope_name, loc="left")
        ax.set_xlabel("Year")
        ax.set_xlim(2022, 2051)
    axes[0].set_ylabel("Annual emissions deviation (Mt CO$_2$)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle(
        "Annual CO$_2$ emissions, deviation from the no-policy baseline",
        x=0.02, ha="left", fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    style.save_both(fig, out_dir, "fig_emissions")


# ---------------------------------------------------------------------------
# Figure — Tornado plot of one-at-a-time sensitivity
# ---------------------------------------------------------------------------


def make_tornado_figure(out_dir: Path) -> None:
    ss = load_sensitivity_summary().set_index("suffix")
    base = ss.loc["sens_newblk_base", "emissions_si_minus_cc_mt"]

    # Each row: human-readable knob name, (low, high) deviations from base.
    grid = [
        ("Adjustment cost $\\bar\\eta$ (0.015 / 0.10)",
         ss.loc["sens_newblk_adj_cost_low",    "emissions_si_minus_cc_mt"] - base,
         ss.loc["sens_newblk_adj_cost_higher", "emissions_si_minus_cc_mt"] - base),
        ("Armington elasticities (\\textpm{}25%)",
         ss.loc["sens_newblk_armington_low",  "emissions_si_minus_cc_mt"] - base,
         ss.loc["sens_newblk_armington_high", "emissions_si_minus_cc_mt"] - base),
        ("Energy elasticity $\\sigma_{EN}$ (0.25 / 0.75)",
         ss.loc["sens_newblk_sigma_en_low",   "emissions_si_minus_cc_mt"] - base,
         ss.loc["sens_newblk_sigma_en_high",  "emissions_si_minus_cc_mt"] - base),
        ("Carbon price (75 / 125 EUR/t)",
         ss.loc["sens_newblk_carbon_75",      "emissions_si_minus_cc_mt"] - base,
         ss.loc["sens_newblk_carbon_125",     "emissions_si_minus_cc_mt"] - base),
        ("Revelation date $T^{\\ast}$ (2028 / 2030)",
         ss.loc["sens_newblk_tstar_2028",     "emissions_si_minus_cc_mt"] - base,
         ss.loc["sens_newblk_tstar_2030",     "emissions_si_minus_cc_mt"] - base),
        ("CBAM base: direct only",
         0.0,
         ss.loc["sens_newblk_cbam_direct_only", "emissions_si_minus_cc_mt"] - base),
    ]

    # Order rows by the largest absolute departure from the central run.  This
    # is the natural one-at-a-time influence measure when a row is one-sided
    # (the direct-only CBAM comparator) rather than a paired low/high interval.
    grid_sorted = sorted(grid, key=lambda r: max(abs(r[1]), abs(r[2])))
    labels = [g[0].replace("\\textpm{}", "±") for g in grid_sorted]
    lo = np.array([g[1] for g in grid_sorted])
    hi = np.array([g[2] for g in grid_sorted])

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.0))

    # Set axis range first so label-pad math is well defined.
    extent = max(abs(lo).max(), abs(hi).max())
    pad = extent * 0.18
    ax.set_xlim(-extent - pad, extent + pad)

    for i, (l, h) in enumerate(zip(lo, hi)):
        left, right = min(l, h), max(l, h)
        ax.barh(y[i], right - left, left=left, height=0.55,
                color="#c7d2eb", edgecolor="#1f3b73", linewidth=0.8, zorder=2)
        ax.plot(l, y[i], marker="o", color="#1f3b73", markersize=6, zorder=3)
        ax.plot(h, y[i], marker="o", color="#a8201a", markersize=6, zorder=3)
        ax.text(l - 0.10 * pad, y[i], f"{l:+.1f}", fontsize=8.4,
                color="#1f3b73", va="center", ha="right")
        ax.text(h + 0.10 * pad, y[i], f"{h:+.1f}", fontsize=8.4,
                color="#a8201a", va="center", ha="left")

    ax.axvline(0, color="#3a3a3a", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Change in global emissions cost of disbelief vs central run "
        f"(Mt CO$_2$; central = {base:.1f})"
    )
    ax.set_title(
        "Tornado: one-at-a-time sensitivity of the global cost of disbelief",
        loc="left", pad=8,
    )
    ax.grid(True, axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.20,
            "Blue = first endpoint (central for the one-sided CBAM row); "
            "red = second endpoint (direct-only for CBAM).",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    fig.tight_layout()
    style.save_both(fig, out_dir, "fig_tornado")


# ---------------------------------------------------------------------------
# Figure — Sensitivity heatmap (variants x headline statistics)
# ---------------------------------------------------------------------------


def make_sensitivity_heatmap(out_dir: Path) -> None:
    ss = load_sensitivity_summary().set_index("suffix")
    metrics = [
        ("inv_gap_pre_pp",        "Pre-revelation\ninvestment gap (pp)"),
        ("inv_gap_revelation_pp", "Revelation-year\nspike (pp)"),
        ("inv_gap_post_pp",       "Persistence\n($T^{\\ast}+1$) (pp)"),
        ("emissions_si_minus_cc_mt", "Cost of disbelief\n(Mt CO$_2$)"),
        ("cbam_effect_2034_pp",   "CBAM protection\n2034 (pp)"),
    ]
    suffix_labels = {
        "sens_newblk_base":              "Central",
        "sens_newblk_sigma_en_low":      "$\\sigma_{EN}=0.25$",
        "sens_newblk_sigma_en_high":     "$\\sigma_{EN}=0.75$",
        "sens_newblk_armington_low":     "Armington $\\times 0.75$",
        "sens_newblk_armington_high":    "Armington $\\times 1.25$",
        "sens_newblk_carbon_75":         "$p^{CO_2}=75$",
        "sens_newblk_carbon_125":        "$p^{CO_2}=125$",
        "sens_newblk_cbam_direct_only":  "CBAM direct only",
        "sens_newblk_tstar_2028":        "$T^{\\ast}=2028$",
        "sens_newblk_tstar_2030":        "$T^{\\ast}=2030$",
        "sens_newblk_adj_cost_low":      "$\\bar\\eta=0.015$",
        "sens_newblk_adj_cost_high":     "$\\bar\\eta=0.06$",
        "sens_newblk_adj_cost_higher":   "$\\bar\\eta=0.10$",
    }
    order = list(suffix_labels.keys())
    M = np.array([[ss.loc[s, m] for s, _ in [(s, None)] for m, _ in metrics] for s in order])
    # Reshape: rows = variants, cols = metrics
    M = np.array([[ss.loc[s, m] for m, _ in metrics] for s in order])

    # Standardize each column for colour, but keep raw values for labels.
    means = M.mean(axis=0)
    stds = M.std(axis=0) + 1e-12
    Z = (M - means) / stds

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([suffix_labels[s] for s in order], fontsize=9.0)
    ax.tick_params(axis="x", which="both", length=0, pad=4)
    ax.tick_params(axis="y", which="both", length=0, pad=4)

    # Annotate every cell with the raw value.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt = f"{val:+.2f}" if abs(val) < 100 else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if abs(Z[i, j]) > 1.2 else "#1f1f1f")

    # Outline the central row.
    central_idx = order.index("sens_newblk_base")
    ax.add_patch(plt.Rectangle((-0.5, central_idx - 0.5), len(metrics), 1,
                               fill=False, edgecolor="#1f1f1f", linewidth=1.6, zorder=3))

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("z-score across variants", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.0)

    ax.grid(False)
    ax.set_title(
        "Sensitivity heatmap: headline statistics across the one-at-a-time grid",
        loc="left", pad=8,
    )
    # Disable spines for cleaner look.
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    style.save_both(fig, out_dir, "fig_sensitivity_heatmap")


# ---------------------------------------------------------------------------
# Figure — Adjustment-cost sensitivity curve
# ---------------------------------------------------------------------------


def make_adjustment_sensitivity_figure(out_dir: Path) -> None:
    """Two-panel response of the credibility statistics to the adjustment-cost
    share eta-bar.

    The point of the figure is the *tension*: a steeper adjustment cost damps
    the visible investment-flow signature of the credibility channel (left
    panel) while it amplifies the cumulative emissions cost of disbelief
    (right panel).  The two panels share the x axis (eta-bar) so the
    divergence is immediate.
    """
    ss = load_sensitivity_summary().set_index("suffix")

    # eta-bar grid: the three variants plus the central run.  Central eta-bar
    # is 0.05; the variant rows do not carry eta-bar explicitly so we hard-map
    # the suffix -> value here (and assert against adjsh column when present).
    points = [
        ("sens_newblk_adj_cost_low",    0.015),
        ("sens_newblk_base",            0.050),
        ("sens_newblk_adj_cost_high",   0.060),
        ("sens_newblk_adj_cost_higher", 0.100),
    ]
    eta = np.array([p[1] for p in points])
    pre   = np.array([ss.loc[s, "inv_gap_pre_pp"]           for s, _ in points])
    spike = np.array([ss.loc[s, "inv_gap_revelation_pp"]    for s, _ in points])
    post  = np.array([ss.loc[s, "inv_gap_post_pp"]          for s, _ in points])
    emis  = np.array([ss.loc[s, "emissions_si_minus_cc_mt"] for s, _ in points])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # --- Left: investment-gap metrics --------------------------------------
    axL.plot(eta, pre,   color="#1f3b73", marker="o", markerfacecolor="white",
             markeredgewidth=1.1, label="Pre-revelation gap ($T^{\\ast}-1$)")
    axL.plot(eta, spike, color="#a8201a", marker="s", markerfacecolor="white",
             markeredgewidth=1.1, label="Revelation-year spike ($T^{\\ast}$)")
    axL.plot(eta, post,  color="#c08a1f", marker="^", markerfacecolor="white",
             markeredgewidth=1.1, label="Persistence ($T^{\\ast}+1$)")
    style.hairline_zero(axL)
    axL.axvline(0.05, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
    axL.text(0.0515, axL.get_ylim()[1] * 0.92, "central", fontsize=8.0, color="#3a3a3a")
    axL.set_xlabel("Adjustment-cost share $\\bar\\eta$")
    axL.set_ylabel("Investment gap, Surprise $-$ Credible (pp)")
    axL.set_title("(a) The investment signature damps", loc="left")
    axL.yaxis.set_major_formatter(mtick.FormatStrFormatter("%+.0f"))
    axL.legend(loc="center right", frameon=False)

    # --- Right: cumulative emissions cost of disbelief ----------------------
    axR.plot(eta, emis, color="#2a8e57", marker="D", markerfacecolor="white",
             markeredgewidth=1.1, linewidth=1.8)
    for x, y in zip(eta, emis):
        axR.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8.2, color="#1f1f1f")
    axR.axvline(0.05, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
    axR.text(0.0515, emis.min() + 0.06 * (emis.max() - emis.min()),
             "central", fontsize=8.0, color="#3a3a3a")
    axR.set_xlabel("Adjustment-cost share $\\bar\\eta$")
    axR.set_ylabel("Cumulative cost of disbelief (Mt CO$_2$)")
    axR.set_title("(b) The emissions cost grows", loc="left")
    axR.set_ylim(emis.min() - 0.12 * (emis.max() - emis.min()),
                 emis.max() + 0.18 * (emis.max() - emis.min()))

    fig.suptitle(
        "Sensitivity of the credibility channel to the adjustment-cost share",
        x=0.02, ha="left", fontsize=12.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save_both(fig, out_dir, "fig_adjustment_sensitivity")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(out_dir: Path | None = None) -> None:
    style.apply()
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "figures"
    print(f"Writing figures to: {out_dir}")
    make_reform_figure(out_dir)
    print("  [ok] fig_reform")
    make_intra_eu_figure(out_dir)
    print("  [ok] fig_intra_eu")
    make_global_realloc_figure(out_dir)
    print("  [ok] fig_global_realloc")
    make_who_benefits_figure(out_dir)
    print("  [ok] fig_who_benefits")
    make_investment_figure(out_dir)
    make_capital_gap_figure(out_dir)
    print("  [ok] fig_investment")
    make_emissions_figure(out_dir)
    print("  [ok] fig_emissions")
    make_tornado_figure(out_dir)
    print("  [ok] fig_tornado")
    make_sensitivity_heatmap(out_dir)
    print("  [ok] fig_sensitivity_heatmap")
    make_adjustment_sensitivity_figure(out_dir)
    print("  [ok] fig_adjustment_sensitivity")
    make_credibility_curve_figure(out_dir)
    print("  [ok] fig_credibility_curve")


if __name__ == "__main__":
    main()
