"""Shared matplotlib style and palette for the CBAM-credibility paper.

Implements a single source of truth for typography, colour, and layout across
the 10 generated paper figures and the graphical abstract. Importing
:mod:`style` triggers global rcParams configuration; explicit helper functions
are also exported for callers that want to layer additional touches on top.

The scenario palette is keyed by the stored equilibrium labels plotted by the
asset pipeline: No Policy, Credible, Surprise, Reneging, and the No-CBAM
diagnostic. Each entry has a primary colour, a desaturated companion for fill,
and a marker glyph. Anticipated Withdrawal is a publication alias for the
central-kappa-zero ``Eq4_Phase1`` path rather than a separate stored
equilibrium label, so callers assign its presentation in the context where it
is used. Region and sector palettes are categorical and colour-blind friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

# Match the body of the paper, which uses Computer Modern / Latin Modern via
# the default LaTeX setup.  Falling back to STIX Two Text keeps figures
# coherent when matplotlib's mathtext is rendered without TeX (e.g. when
# generating drafts without a full TeX install).
_FONT_STACK = ["STIX Two Text", "Source Serif Pro", "DejaVu Serif"]
_MATH_FONT_SET = "stix"


def apply() -> None:
    """Apply the paper-wide matplotlib style.

    Idempotent.  Call once per process before generating figures.  Returns
    None.  Rationale for each rcParam is intentionally explicit so the same
    knobs are easy to inspect from a referee report.
    """

    mpl.rcParams.update({
        # --- typography ---------------------------------------------------
        "font.family": "serif",
        "font.serif": _FONT_STACK,
        "mathtext.fontset": _MATH_FONT_SET,
        "font.size": 10.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "figure.titlesize": 12.0,
        # --- lines and markers -------------------------------------------
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.8,
        # --- axes ---------------------------------------------------------
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#3a3a3a",
        "axes.labelcolor": "#1f1f1f",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "axes.titlepad": 6.0,
        "axes.labelpad": 4.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        # --- grid ---------------------------------------------------------
        "grid.color": "#dadada",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        # --- ticks --------------------------------------------------------
        "xtick.color": "#3a3a3a",
        "ytick.color": "#3a3a3a",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        # --- legend -------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.2,
        # --- figure -------------------------------------------------------
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,  # embed TrueType, not Type-3 (publisher requirement)
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioStyle:
    """Visual recipe for a stored policy-path or diagnostic label."""

    color: str          # primary line colour
    fill: str           # paired soft colour for shading
    marker: str         # marker glyph for time series
    label: str          # display label for legends


SCENARIOS: Mapping[str, ScenarioStyle] = {
    "Eq2_Credible": ScenarioStyle(
        color="#1f3b73", fill="#c7d2eb", marker="o", label="Credible commitment"
    ),
    "Eq4_Surprise": ScenarioStyle(
        color="#a8201a", fill="#f1c4c0", marker="s", label="Surprise"
    ),
    "Eq3_Reneg": ScenarioStyle(
        color="#c08a1f", fill="#f3e1bd", marker="^", label="Reneging"
    ),
    "Eq1_NoPol": ScenarioStyle(
        color="#5b5b5b", fill="#dcdcdc", marker="D", label="No policy"
    ),
    "Eq2_NoCBAM": ScenarioStyle(
        color="#2f8f6f", fill="#c8e3d6", marker="v", label="Credible, no CBAM"
    ),
}


# Three-block EU palette, anchored on a green-purple-red sequential reading
# so the order EUC < EUM < EUD is visually obvious.
EU_BLOCK_COLORS: Mapping[str, str] = {
    "EUC": "#2a8e57",
    "EUM": "#6e6aa8",
    "EUD": "#b03a2e",
}


# Region palette for cross-region bar plots.  Chosen with the constraint that
# the EU triplet stays close to its colours in EU_BLOCK_COLORS while the rest
# of the world maps to a categorical viridis-like sweep.
REGION_COLORS: Mapping[str, str] = {
    "EUC": "#2a8e57",
    "EUM": "#6e6aa8",
    "EUD": "#b03a2e",
    "GBR": "#3a7fa1",
    "CHN": "#d97706",
    "IND": "#a16207",
    "RUS": "#7f1d1d",
    "TUR": "#9d174d",
    "OCD": "#1e293b",
    "ROW": "#475569",
}


# Sector palette, used in any chart that compares sectors side by side.
SECTOR_COLORS: Mapping[str, str] = {
    "STL": "#2b6cb0",
    "CEM": "#6b7280",
    "ALU": "#7c3aed",
    "FER": "#15803d",
    "CHM": "#b45309",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SECTOR_LABELS: Mapping[str, str] = {
    "STL": "Steel",
    "CEM": "Cement",
    "ALU": "Aluminium",
    "FER": "Fertilizers",
    "CHM": "Chemicals",
}

REGION_LABELS: Mapping[str, str] = {
    "EUC": "EU clean",
    "EUM": "EU middle",
    "EUD": "EU dirty",
    "GBR": "United Kingdom",
    "CHN": "China",
    "IND": "India",
    "RUS": "Russia",
    "TUR": "Turkey",
    "OCD": "Rest OECD",
    "ROW": "Rest of world",
}


def annotate_revelation(ax, x: float, label: str = "$T^{\\ast}=2029$", color: str = "#3a3a3a") -> None:
    """Draw the dashed revelation-date marker used in every time-series plot."""
    ax.axvline(x, color=color, linestyle=(0, (1.4, 1.4)), linewidth=0.9, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    ax.text(
        x + 0.15,
        ymax - 0.04 * (ymax - ymin),
        label,
        fontsize=8.0,
        color=color,
        ha="left",
        va="top",
    )


def hairline_zero(ax, color: str = "#9aa0a6", linewidth: float = 0.7) -> None:
    """Light horizontal zero reference, drawn behind data."""
    ax.axhline(0, color=color, linewidth=linewidth, alpha=0.8, zorder=0)


def save_both(fig, out_dir: Path, name: str, png_max_px: int = 1900) -> None:
    """Save figure as both PDF (for the paper) and PNG (for previews).

    The PDF is the vector asset the paper embeds.  The PNG is a raster preview;
    its dpi is capped so the longest side stays under ``png_max_px``.  Previews
    above ~2000 px are rejected by some image viewers, so the cap keeps every
    preview openable while never upscaling small figures past the 200-dpi base.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    longest_in = max(fig.get_size_inches())
    png_dpi = min(200.0, png_max_px / longest_in)
    fig.savefig(out_dir / f"{name}.png", dpi=png_dpi)
    plt.close(fig)


__all__ = [
    "apply",
    "ScenarioStyle",
    "SCENARIOS",
    "EU_BLOCK_COLORS",
    "REGION_COLORS",
    "SECTOR_COLORS",
    "SECTOR_LABELS",
    "REGION_LABELS",
    "annotate_revelation",
    "hairline_zero",
    "save_both",
]
