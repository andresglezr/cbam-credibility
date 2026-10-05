"""Graphical abstract for the Energy Economics submission.

Left panel: EU covered-sector aggregate capital stock under Credible
commitment vs. Surprise (stitched), indexed to the no-policy path. Right
panel: the RN--CC emissions gap decomposed into the same-realized-policy
RN--AW expectations component and the correctly anticipated AW--CC path
difference, using the
same structural accounting as the emissions tables.

Run with PAPER_RUN_SUFFIX pointing at the central run:
    PAPER_RUN_SUFFIX=newblk python graphical_abstract.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style
from data_io import load_parameters, load_results_3d, load_sam, scalar_parameter
from figures import COVERED, EU_BLOCKS, _stitched, structural_annual_emissions


def eu_covered_capital_index(r3: pd.DataFrame, eq: str) -> pd.Series:
    """EU covered-sector capital relative to the no-policy path (NP = 100)."""
    seg = _stitched(r3, eq, "KP", EU_BLOCKS)
    seg = seg[seg["sec"].isin(COVERED)]
    path = seg.groupby("year")["value"].sum()
    base = _stitched(r3, "Eq1_NoPol", "KP", EU_BLOCKS)
    base = base[base["sec"].isin(COVERED)].groupby("year")["value"].sum()
    return 100.0 * path / base


def main(out_dir: Path | None = None) -> None:
    style.apply()
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent

    r3 = load_results_3d()
    params = load_parameters()
    sam = load_sam()

    cc = eu_covered_capital_index(r3, "Eq2_Credible")
    si = eu_covered_capital_index(r3, "Eq4_Surprise")

    # EU cumulative emissions from the same structural accounting used by the
    # paper tables. AW is defined only for the central kappa=0 calibration.
    tot = {
        eq: structural_annual_emissions(r3, params, sam, eq, "EU").sum()
        for eq in ["Eq1_NoPol", "Eq2_Credible", "AW", "Eq3_Reneg"]
    }
    benefit = tot["Eq1_NoPol"] - tot["Eq2_Credible"]
    if abs(benefit) < 1e-12:
        raise ValueError("Cannot express the withdrawal decomposition as a share of a zero benefit")

    rn_minus_aw = tot["Eq3_Reneg"] - tot["AW"]
    aw_minus_cc = tot["AW"] - tot["Eq2_Credible"]
    rn_minus_cc = tot["Eq3_Reneg"] - tot["Eq2_Credible"]
    residual = rn_minus_cc - rn_minus_aw - aw_minus_cc
    scale = max(1.0, abs(rn_minus_cc), abs(rn_minus_aw), abs(aw_minus_cc))
    if abs(residual) > 1e-10 * scale:
        raise RuntimeError(f"RN-AW-CC emissions decomposition does not close: residual={residual}")

    rn_aw_pct = 100.0 * rn_minus_aw / benefit
    aw_cc_pct = 100.0 * aw_minus_cc / benefit
    rn_cc_pct = 100.0 * rn_minus_cc / benefit

    t_star_year = 2021 + int(scalar_parameter(params, "T_STAR", 8))

    cc_sty = style.SCENARIOS["Eq2_Credible"]
    si_sty = style.SCENARIOS["Eq4_Surprise"]
    rn_sty = style.SCENARIOS["Eq3_Reneg"]

    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(cc.index, cc.values, color=cc_sty.color, linewidth=2.6, label="Credible commitment")
    ax.plot(si.index, si.values, color=si_sty.color, linewidth=2.6, label="Surprise")
    ax.fill_between(cc.index, si.values, cc.values, where=cc.values >= si.values,
                    color=si_sty.fill, alpha=0.55, label="Cost of disbelief")
    ax.axvline(t_star_year, color="#3a3a3a", linestyle=(0, (1.4, 1.4)), linewidth=1.0)
    ax.text(t_star_year + 0.3, ax.get_ylim()[1], f"$T^*={t_star_year}$",
            fontsize=11, color="#3a3a3a", va="top")
    ax.set_title("Industrial capital stock under firm disbelief", fontsize=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Capital stock (no-policy = 100)", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, fontsize=11, loc="lower right")

    axr = fig.add_subplot(gs[0, 1])
    axr.axis("off")
    axr.text(0.02, 0.95, "Decomposing policy withdrawal", fontsize=18, fontweight="bold", va="top")
    axr.text(0.02, 0.80, "share of announced EU emission savings",
             fontsize=13, style="italic", color="#555555", va="top")
    axr.text(0.02, 0.60, "Same-policy expectations\n$RN-AW$", fontsize=13, fontweight="bold",
              color=rn_sty.color, va="center")
    axr.text(0.98, 0.60, f"${rn_aw_pct:+.1f}\\%$", fontsize=27, fontweight="bold",
              color=rn_sty.color, va="center", ha="right")
    axr.text(0.02, 0.35, "Correctly anticipated paths\n$AW-CC$", fontsize=13, fontweight="bold",
              color=cc_sty.color, va="center")
    axr.text(0.98, 0.35, f"${aw_cc_pct:+.1f}\\%$", fontsize=27, fontweight="bold",
              color=cc_sty.color, va="center", ha="right")
    axr.text(0.02, 0.10,
             f"Accounting identity: $RN-CC={rn_cc_pct:+.1f}\\%$\n"
             "$RN-CC=(RN-AW)+(AW-CC)$",
             fontsize=13, style="italic", color="#333333", va="center")

    fig.subplots_adjust(left=0.065, right=0.98, top=0.92, bottom=0.14, wspace=0.20)
    fig.savefig(out_dir / "graphical_abstract.pdf")
    fig.savefig(out_dir / "graphical_abstract.png", dpi=200)
    plt.close(fig)
    print(
        "graphical abstract: "
        f"RN-AW {rn_aw_pct:+.1f}%, AW-CC {aw_cc_pct:+.1f}%, "
        f"RN-CC {rn_cc_pct:+.1f}% (residual {residual:.3e} Mt)"
    )


if __name__ == "__main__":
    main()
