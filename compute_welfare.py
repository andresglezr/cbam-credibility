"""Compute consumption-equivalent welfare from exported Ramsey results.

The published metric is the finite-horizon (2022--2051) regional
consumption-equivalent variation.  Real consumption is reconstructed as the
Cobb-Douglas quantity index implied by ``alphaH`` and exported ``H_C`` hats.
An optional terminal balanced-growth continuation is also reported as a
separate diagnostic; it is not part of the published finite-horizon measure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_ORDER = [
    "Eq1_NoPol",
    "Eq2_Credible",
    "Eq2_NoCBAM",
    "Eq3_Reneg",
    "Eq4_Phase1",
    "Eq4_Surprise",
]


def scalar_param(params: pd.DataFrame, name: str, default: float) -> float:
    row = params[
        params["param"].eq(name)
        & params["reg"].fillna("").eq("")
        & params["sec"].fillna("").eq("")
    ]
    if len(row):
        return float(row.iloc[0]["value"])
    return default


def _validated_arrays(
    consumption: pd.Series | np.ndarray,
    reference: pd.Series | np.ndarray,
    weight: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite, positive and conformable CE inputs."""

    c = np.asarray(consumption, dtype=float)
    cref = np.asarray(reference, dtype=float)
    w = np.asarray(weight, dtype=float)
    if c.ndim != 1 or cref.ndim != 1 or w.ndim != 1 or not (len(c) == len(cref) == len(w)):
        raise ValueError("Consumption, reference, and welfare weights must be one-dimensional and equal-length.")
    if len(c) == 0:
        raise ValueError("At least one period is required to compute a consumption equivalent.")
    if not (np.isfinite(c).all() and np.isfinite(cref).all() and np.isfinite(w).all()):
        raise ValueError("Consumption-equivalent inputs must all be finite.")
    if (c <= 0).any() or (cref <= 0).any():
        raise ValueError("Consumption and reference consumption must be strictly positive.")
    if (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("Welfare weights must be nonnegative and have a positive sum.")
    return c, cref, w


def ce_factor_exact(
    consumption: pd.Series | np.ndarray,
    reference: pd.Series | np.ndarray,
    weight: pd.Series | np.ndarray,
    gamma: float,
) -> float:
    """Exact CIES consumption-equivalent factor for two consumption paths.

    For ``gamma != 1``, the factor is the ratio of the two discounted power
    sums, raised to ``1 / (1 - gamma)``.  This differs from taking a CES mean
    of period-by-period consumption ratios whenever reference consumption
    varies over time.  The log-utility limit is handled separately.
    """

    c, cref, w = _validated_arrays(consumption, reference, weight)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and strictly positive.")
    if abs(gamma - 1.0) < 1e-10:
        return float(np.exp(np.sum(w * (np.log(c) - np.log(cref))) / np.sum(w)))

    power = 1.0 - gamma
    utility_ratio = np.sum(w * c**power) / np.sum(w * cref**power)
    if not np.isfinite(utility_ratio) or utility_ratio <= 0:
        raise ValueError("The discounted CIES power-sum ratio must be finite and positive.")
    return float(utility_ratio ** (1.0 / power))


def ce_factor(hat: pd.Series | np.ndarray, weight: pd.Series | np.ndarray, gamma: float) -> float:
    """Compatibility helper when the reference path is identically one.

    Production welfare calculations use :func:`ce_factor_exact` with the
    scenario and reference paths separately.  Retaining this helper avoids
    breaking external imports that used the old normalized-path interface.
    """

    h = np.asarray(hat, dtype=float)
    return ce_factor_exact(h, np.ones_like(h), weight, gamma)


def run_formula_checks() -> None:
    """Fast analytical checks for the exact CIES formula."""

    reference = np.array([0.72, 1.05, 1.41], dtype=float)
    weights = np.array([1.0, 0.91, 0.83], dtype=float)
    scale = 1.037
    for gamma in (1.0, 2.0, 3.5):
        identity = ce_factor_exact(reference, reference, weights, gamma)
        scaled = ce_factor_exact(scale * reference, reference, weights, gamma)
        if not np.isclose(identity, 1.0, rtol=0.0, atol=1e-12):
            raise AssertionError(f"CIES identity check failed for gamma={gamma}: {identity}")
        if not np.isclose(scaled, scale, rtol=0.0, atol=1e-12):
            raise AssertionError(f"CIES scale check failed for gamma={gamma}: {scaled}")

    scenario = np.array([0.81, 0.96, 1.58], dtype=float)
    gamma = 2.0
    expected = (
        np.sum(weights * scenario ** (1.0 - gamma))
        / np.sum(weights * reference ** (1.0 - gamma))
    ) ** (1.0 / (1.0 - gamma))
    actual = ce_factor_exact(scenario, reference, weights, gamma)
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"CIES discounted-utility-ratio check failed: {actual} != {expected}")


def compute_welfare(
    results_dir: Path,
    suffix: str,
    gamma_default: float,
    terminal_tail: bool = True,
) -> pd.DataFrame:
    run_formula_checks()

    res3 = pd.read_parquet(results_dir / f"results_3d_{suffix}.parquet")
    params = pd.read_parquet(results_dir / f"parameters_{suffix}.parquet")
    sam = pd.read_parquet(results_dir / f"sam_data_{suffix}.parquet")

    g = scalar_param(params, "g", 0.02)
    ro = scalar_param(params, "ro", 0.05)
    gamma = scalar_param(params, "gamma", gamma_default)
    adjsh = scalar_param(params, "adjsh", 0.03)

    alpha = (
        params[params["param"].eq("alphaH")][["reg", "sec", "value"]]
        .rename(columns={"value": "alphaH"})
        .copy()
    )
    alpha["alphaH"] = alpha["alphaH"].astype(float)
    alpha_sum = alpha.groupby("reg")["alphaH"].transform("sum")
    alpha["alphaH_norm"] = np.where(alpha_sum.abs() > 1e-14, alpha["alphaH"] / alpha_sum, alpha["alphaH"])

    hc = res3[res3["var"].eq("H_C")][["eq", "reg", "sec", "T", "year", "value"]].copy()
    hc = hc.merge(alpha[["reg", "sec", "alphaH_norm"]], on=["reg", "sec"], how="left")
    if hc["alphaH_norm"].isna().any():
        missing = hc[hc["alphaH_norm"].isna()][["reg", "sec"]].drop_duplicates()
        raise RuntimeError(f"Missing alphaH rows: {missing.to_dict(orient='records')[:10]}")
    if (hc["value"] <= 0).any():
        bad = hc[hc["value"] <= 0].head()
        raise RuntimeError(f"Nonpositive H_C values: {bad.to_dict(orient='records')}")

    hc["weighted_log_hc"] = hc["alphaH_norm"] * np.log(hc["value"].astype(float))
    idx = hc.groupby(["eq", "reg", "T", "year"], as_index=False)["weighted_log_hc"].sum()
    idx["C_hat"] = np.exp(idx["weighted_log_hc"])

    # Welfare is lifetime utility over LEVEL consumption,
    #   V = sum_t delta^(t-1) C_t^(1-gamma)/(1-gamma),  delta = (1+g)^gamma/(1+ro),
    # the discount factor implied by the model's Euler (EQEULER in
    # sinretencion.py), where ro is the balanced-growth interest rate and the
    # implied pure rate of time preference is (1+ro)/(1+g)^gamma - 1. Writing
    # level consumption C_t = C_hat_t (1+g)^(t-1), the weight on the detrended
    # index C_hat is delta*(1+g)^(1-gamma) = (1+g)/(1+ro): discounting the
    # detrended hats by 1/(1+ro) alone would understate the weight on the
    # growing future and is inconsistent with the model's own preferences.
    beta = (1.0 + g) / (1.0 + ro)
    if not np.isfinite(beta) or beta <= 0:
        raise RuntimeError(f"Invalid detrended welfare discount weight beta={beta}.")
    if terminal_tail and beta >= 1:
        raise RuntimeError(
            "The optional infinite terminal tail requires beta=(1+g)/(1+ro) < 1; "
            f"got beta={beta}. Use terminal_tail=False for the finite-horizon measure."
        )
    t = idx["T"].astype(int)
    idx["w_t"] = beta ** (t - 1)

    ref_eq1 = idx[idx["eq"].eq("Eq1_NoPol")][["reg", "T", "C_hat"]].rename(columns={"C_hat": "C_ref_eq1"})
    ref_eq2 = idx[idx["eq"].eq("Eq2_Credible")][["reg", "T", "C_hat"]].rename(columns={"C_hat": "C_ref_eq2"})
    if ref_eq1.empty or ref_eq2.empty:
        raise RuntimeError("Both Eq1_NoPol and Eq2_Credible are required as welfare references.")
    work = idx.merge(ref_eq1, on=["reg", "T"], how="left").merge(ref_eq2, on=["reg", "T"], how="left")
    if work[["C_ref_eq1", "C_ref_eq2"]].isna().any().any():
        missing = work.loc[
            work[["C_ref_eq1", "C_ref_eq2"]].isna().any(axis=1),
            ["eq", "reg", "T"],
        ].head(10)
        raise RuntimeError(f"Incomplete welfare reference paths: {missing.to_dict(orient='records')}")

    rows = []
    for (eq, reg), gdf in work.groupby(["eq", "reg"]):
        gdf = gdf.sort_values("T")
        finite_1 = ce_factor_exact(gdf["C_hat"], gdf["C_ref_eq1"], gdf["w_t"], gamma)
        finite_2 = ce_factor_exact(gdf["C_hat"], gdf["C_ref_eq2"], gdf["w_t"], gamma)

        if terminal_tail:
            last = gdf.iloc[-1]
            last_t = int(last["T"])
            tail_weight = beta**last_t / (1.0 - beta)
            weights = np.append(gdf["w_t"].to_numpy(dtype=float), tail_weight)
            scenario_tail = np.append(gdf["C_hat"].to_numpy(dtype=float), float(last["C_hat"]))
            ref1_tail = np.append(gdf["C_ref_eq1"].to_numpy(dtype=float), float(last["C_ref_eq1"]))
            ref2_tail = np.append(gdf["C_ref_eq2"].to_numpy(dtype=float), float(last["C_ref_eq2"]))
            tail_1 = ce_factor_exact(scenario_tail, ref1_tail, weights, gamma)
            tail_2 = ce_factor_exact(scenario_tail, ref2_tail, weights, gamma)
            selected_1 = tail_1
            selected_2 = tail_2
            selected_measure = "finite_plus_terminal_tail"
        else:
            tail_1 = np.nan
            tail_2 = np.nan
            selected_1 = finite_1
            selected_2 = finite_2
            selected_measure = "finite_2022_2051"

        rows.append(
            {
                "scope": "region",
                "reg": reg,
                "eq": eq,
                # Backward-compatible aliases: these select the terminal-tail
                # diagnostic unless compute_welfare(..., terminal_tail=False)
                # is requested.  Published tables use the explicit finite_*
                # columns below.
                "ce_vs_Eq1_pct": (selected_1 - 1.0) * 100.0,
                "ce_vs_Eq2_pct": (selected_2 - 1.0) * 100.0,
                "ce_factor_vs_Eq1": selected_1,
                "ce_factor_vs_Eq2": selected_2,
                "ce_measure": selected_measure,
                "terminal_tail_included": bool(terminal_tail),
                "finite_ce_vs_Eq1_pct": (finite_1 - 1.0) * 100.0,
                "finite_ce_vs_Eq2_pct": (finite_2 - 1.0) * 100.0,
                "finite_ce_factor_vs_Eq1": finite_1,
                "finite_ce_factor_vs_Eq2": finite_2,
                "terminal_tail_ce_vs_Eq1_pct": (tail_1 - 1.0) * 100.0,
                "terminal_tail_ce_vs_Eq2_pct": (tail_2 - 1.0) * 100.0,
                "terminal_tail_ce_factor_vs_Eq1": tail_1,
                "terminal_tail_ce_factor_vs_Eq2": tail_2,
            }
        )
    region_df = pd.DataFrame(rows)

    c_raw = sam[sam["var"].eq("c_raw")][["reg", "sec", "value"]].rename(columns={"value": "c_raw"})
    inv = sam[sam["var"].eq("iz")][["reg", "sec", "value"]].rename(columns={"value": "iz"})
    cbase = c_raw.merge(inv, on=["reg", "sec"], how="outer").fillna(0.0)
    cbase["c_adj"] = cbase["c_raw"] - adjsh * cbase["iz"]
    weights_reg = cbase.groupby("reg", as_index=False)["c_adj"].sum().rename(columns={"c_adj": "base_consumption"})
    if not np.isfinite(weights_reg["base_consumption"]).all() or (weights_reg["base_consumption"] <= 0).any():
        bad = weights_reg.loc[
            ~np.isfinite(weights_reg["base_consumption"]) | (weights_reg["base_consumption"] <= 0)
        ]
        raise RuntimeError(f"Invalid adjusted benchmark-consumption weights: {bad.to_dict(orient='records')}")

    agg_rows = []
    groups = [
        ("EU", sorted(r for r in weights_reg["reg"].unique() if str(r).startswith("EU"))),
        ("World", sorted(weights_reg["reg"].unique())),
    ]
    for scope, regs in groups:
        weights = weights_reg[weights_reg["reg"].isin(regs)].copy()
        total = weights["base_consumption"].sum()
        if total <= 0:
            raise RuntimeError(f"Nonpositive adjusted benchmark consumption for aggregate {scope}.")
        for eq in sorted(region_df["eq"].unique()):
            tmp = region_df[region_df["eq"].eq(eq) & region_df["reg"].isin(regs)].merge(weights, on="reg")
            if set(tmp["reg"]) != set(regs):
                missing_regs = sorted(set(regs) - set(tmp["reg"]))
                raise RuntimeError(f"Missing regional welfare rows for {scope}, {eq}: {missing_regs}")
            finite_f1 = (tmp["finite_ce_factor_vs_Eq1"] * tmp["base_consumption"]).sum() / total
            finite_f2 = (tmp["finite_ce_factor_vs_Eq2"] * tmp["base_consumption"]).sum() / total
            if terminal_tail:
                tail_f1 = (tmp["terminal_tail_ce_factor_vs_Eq1"] * tmp["base_consumption"]).sum() / total
                tail_f2 = (tmp["terminal_tail_ce_factor_vs_Eq2"] * tmp["base_consumption"]).sum() / total
                selected_f1 = tail_f1
                selected_f2 = tail_f2
                selected_measure = "finite_plus_terminal_tail"
            else:
                tail_f1 = np.nan
                tail_f2 = np.nan
                selected_f1 = finite_f1
                selected_f2 = finite_f2
                selected_measure = "finite_2022_2051"
            agg_rows.append(
                {
                    "scope": scope,
                    "reg": scope,
                    "eq": eq,
                    "ce_vs_Eq1_pct": (selected_f1 - 1.0) * 100.0,
                    "ce_vs_Eq2_pct": (selected_f2 - 1.0) * 100.0,
                    "ce_factor_vs_Eq1": selected_f1,
                    "ce_factor_vs_Eq2": selected_f2,
                    "ce_measure": selected_measure,
                    "terminal_tail_included": bool(terminal_tail),
                    "finite_ce_vs_Eq1_pct": (finite_f1 - 1.0) * 100.0,
                    "finite_ce_vs_Eq2_pct": (finite_f2 - 1.0) * 100.0,
                    "finite_ce_factor_vs_Eq1": finite_f1,
                    "finite_ce_factor_vs_Eq2": finite_f2,
                    "terminal_tail_ce_vs_Eq1_pct": (tail_f1 - 1.0) * 100.0,
                    "terminal_tail_ce_vs_Eq2_pct": (tail_f2 - 1.0) * 100.0,
                    "terminal_tail_ce_factor_vs_Eq1": tail_f1,
                    "terminal_tail_ce_factor_vs_Eq2": tail_f2,
                }
            )

    return pd.concat([pd.DataFrame(agg_rows), region_df], ignore_index=True).sort_values(["scope", "reg", "eq"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--suffix", default="newblk",
                        help="Run suffix to load; default is the paper's published central "
                             "run. The historical family is v3eubm_legind.")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--no-terminal-tail",
        action="store_true",
        help=(
            "Do not calculate the optional terminal BGP continuation diagnostic. "
            "The published finite 2022-2051 columns are always calculated."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    welfare = compute_welfare(results_dir, args.suffix, args.gamma, terminal_tail=not args.no_terminal_tail)
    out_path = results_dir / f"welfare_{args.suffix}.csv"
    welfare.to_csv(out_path, index=False)

    agg = welfare[welfare["scope"].isin(["EU", "World"])]
    print(f"Wrote {out_path} ({len(welfare)} rows)")
    print("\nPublished finite-horizon CE (2022-2051), percent vs Eq1_NoPol:")
    print(agg.pivot(index="eq", columns="scope", values="finite_ce_vs_Eq1_pct").reindex(SCENARIO_ORDER).round(5).to_string())
    print("\nPublished finite-horizon CE (2022-2051), percent vs Eq2_Credible:")
    print(agg.pivot(index="eq", columns="scope", values="finite_ce_vs_Eq2_pct").reindex(SCENARIO_ORDER).round(5).to_string())
    if not args.no_terminal_tail:
        print("\nOptional terminal-tail diagnostic, percent vs Eq2_Credible:")
        print(
            agg.pivot(index="eq", columns="scope", values="terminal_tail_ce_vs_Eq2_pct")
            .reindex(SCENARIO_ORDER)
            .round(5)
            .to_string()
        )


if __name__ == "__main__":
    main()
