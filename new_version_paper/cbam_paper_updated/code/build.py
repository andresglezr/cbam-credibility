"""Top-level build driver for paper assets.

Run::

    python build.py

to regenerate every figure and every table from the parquet results.  The
script is intentionally a thin wrapper so that the heavy lifting stays in
:mod:`figures` and :mod:`tables`; this keeps the entry point readable from
the command line and from a Makefile.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import data_io
import figures
import graphical_abstract
import style
import tables


def main() -> int:
    style.apply()
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "figures"
    tab_dir = root / "tables"

    print("=" * 60)
    print("CBAM credibility paper — asset build")
    print(f"Central run: PAPER_RUN_SUFFIX={data_io.RUN_SUFFIX}")
    print("=" * 60)

    t0 = time.perf_counter()
    figures.main(out_dir=fig_dir)
    t_fig = time.perf_counter() - t0
    print(f"Figures finished in {t_fig:.1f}s")

    t0 = time.perf_counter()
    tables.main(out_dir=tab_dir)
    t_tab = time.perf_counter() - t0
    print(f"Tables finished in {t_tab:.1f}s")

    graphical_abstract.main(out_dir=root)
    print("Graphical abstract finished")

    print()
    print(f"All assets written under {root}.")
    print("Compile with the usual `pdflatex; bibtex; pdflatex; pdflatex` cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
