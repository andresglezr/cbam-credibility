# The Cost of (In)Credibility: replication package

Code and data for *The Cost of (In)Credibility: Anticipation, Surprise, and the
Investment Channel of the EU Carbon Border Adjustment Mechanism* (Andrés
González-Rodríguez, University of Santiago de Compostela).

The package contains the model code, the aggregated input data the model
runs on, and the scripts that regenerate every figure and table of the paper
from the solved equilibria.

## Contents

| Path | What it is |
|---|---|
| `sinretencion.py` | The forward-looking multi-region Ramsey CGE model (GAMSPy/PATH) |
| `run_campaign.sh` | One command that solves every equilibrium used in the paper and builds the assets |
| `recenter_compute.py`, `run_central_eul.py`, `sensitivity_experiments.py`, `partial_cred_sweep*.py` | Run drivers (central run, sensitivity grid, credibility index) |
| `validate_batch_results.py`, `compute_welfare.py` | Solver/accounting validation and welfare post-processing |
| `exiobase3_2022_10x9.xlsx` | Model input: 2022 input-output table, emissions and value added, aggregated to 10 regions x 9 sectors |
| `replication/workbooks/` | The same aggregate under the robustness choices (China reallocation 50/35 percent, published or rule-based EU blocks) |
| `replication/build_sam.py` | Rebuilds the aggregate from raw EXIOBASE 3, with the sector concordance and EU block mapping in `replication/` |
| `replication/requirements-*.txt` | Pinned Python environments for the model and for the asset pipeline |
| `new_version_paper/cbam_paper_updated/code/` | Figures and tables of the paper, built from the solved runs |

## Data and license

The input workbooks are derived aggregates of **EXIOBASE 3** (Stadler et al.,
2018, https://doi.org/10.5281/zenodo.5589597), licensed CC BY-SA 4.0. They are
redistributed here under the same license. The raw EXIOBASE tables are not
included. To rebuild the aggregate from them:

    python replication/build_sam.py --raw path/to/IOT_2022_ixi.zip

## Replication

**1. Solve the model.** Environment `replication/requirements-model.txt`
(Python 3.12, GAMSPy 1.20.0) plus a GAMS installation with a PATH solver
license. The full campaign, from the repository root:

    bash run_campaign.sh

It solves the central run, the credibility-index sweep, the sensitivity grid
and the robustness runs, writes them to `results/`, validates them and builds
the paper assets. A single central run is `python sinretencion.py`.

**2. Figures and tables only** (after step 1). Environment
`replication/requirements-pipeline.txt`:

    cd new_version_paper/cbam_paper_updated/code
    PAPER_RUN_SUFFIX=newblk python build.py

Figures are written to `new_version_paper/cbam_paper_updated/figures/` and
tables to `new_version_paper/cbam_paper_updated/tables/`.
The solved runs are read from `results/` (override with `CBAM_RESULTS_DIR`).
