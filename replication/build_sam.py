# -*- coding: utf-8 -*-
"""Build the model input workbook ``exiobase3_2022_10x9.xlsx`` from raw EXIOBASE 3.

Stages
------
1. Parse the raw 2022 industry-by-industry MRIO (49 regions x 163 industries)
   and aggregate industries to the paper's nine sectors. The refined-fuels
   sector REF is the complete fossil supply chain (coal mining, crude oil and
   natural gas extraction, liquefaction/regasification, coke ovens, petroleum
   refinery, nuclear fuel processing, gas manufacture/distribution) so that
   every fossil carrier whose combustion generates the emissions in GHG_FF
   flows through the REF row of the model's energy nest. Electricity
   generation is split into fossil (ELF: coal, gas, petroleum) and non-fossil
   (ELR); transmission and distribution are network services and go to OTH.
2. EU blocks. The input of record is the versioned ``eu_block_mapping.csv``
   (default ``--blocks csv``). The script also recomputes the statistical
   rule behind it -- per-country CO2 intensity (direct emissions per unit of
   gross output) of electricity generation, iron and steel, and cement;
   z-score each across member states; average into a composite; terciles
   (9/9/9) -- prints the full score table and reports every country on which
   the rule and the versioned mapping differ. ``--blocks rule`` aggregates
   with the rule's assignment instead; only ``--write-mapping`` (with
   ``--blocks rule``) overwrites the CSV, never implicitly.
3. Aggregate regions (blocks + GBR, CHN, IND, RUS, TUR, OCD, ROW), assemble
   the IOT sheet (bilateral Z, household and investment final demand, L/K
   value added, TAX row) and the emi sheet (GHG_FF combustion CO2eq with
   GWP100 AR5; GHG_PROC = non-combustion cement+lime CO2 plus PFC/SF6/HFC),
   apply the China balanced-growth investment rule (reallocate CHINA_FRAC of
   GFCF to consumption; default 0.5, see the calibration section of the
   paper), and write the workbook.

Usage
-----
    python build_sam.py --raw path/to/IOT_2022_ixi.zip [--out exiobase3_2022_10x9.xlsx]

Requires pymrio (see requirements-pipeline.txt) and ~16 GB RAM for the parse.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SECTORS = ["ALU", "CEM", "CHM", "ELF", "ELR", "FER", "OTH", "REF", "STL"]

NONEU = {"GB": "GBR", "CN": "CHN", "IN": "IND", "RU": "RUS", "TR": "TUR",
         "US": "OCD", "JP": "OCD", "CA": "OCD", "KR": "OCD", "AU": "OCD", "CH": "OCD", "NO": "OCD",
         "MX": "ROW", "BR": "ROW", "TW": "ROW", "ID": "ROW", "ZA": "ROW",
         "WA": "ROW", "WL": "ROW", "WE": "ROW", "WF": "ROW", "WM": "ROW"}
EU_ISO = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU",
          "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
ISO2NAME = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia",
            "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
            "FR": "France", "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
            "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
            "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden",
            "SI": "Slovenia", "SK": "Slovakia"}

CHINA_FRAC = float(os.environ.get("CHINA_FRAC", "0.5"))

GWP = {"CH4": 28.0, "N2O": 265.0}  # GWP100, IPCC AR5
FF_TERMS = [("CO2 - combustion - air", 1.0), ("CO2 - waste - fossil - air", 1.0),
            ("CH4 - combustion - air", GWP["CH4"]), ("N2O - combustion - air", GWP["N2O"])]
PROC_TERMS = [("CO2 - non combustion - Cement production - air", 1.0),
              ("CO2 - non combustion - Lime production - air", 1.0),
              ("PFC - air", 1.0), ("SF6 - air", 1.0), ("HFC - air", 1.0)]
KG_TO_MT = 1e-9

H_CATS = ["Final consumption expenditure by households",
          "Final consumption expenditure by non-profit organisations serving households (NPISH)",
          "Final consumption expenditure by government"]
I_CATS = ["Gross fixed capital formation", "Changes in inventories", "Changes in valuables"]


def norm(s: str) -> str:
    s = re.sub(r"\(\d+\)", "", str(s).lower().strip())
    return re.sub(r"\s+", " ", s).strip()


def sector_map(sec_names: list[str]) -> list[str]:
    conc = pd.read_csv(os.path.join(HERE, "sector_concordance_163_to_9.csv"))
    m = {norm(r["industry"]): r["sector"] for _, r in conc.iterrows()}
    out = []
    for s in sec_names:
        n = norm(s)
        tgt = m.get(n)
        if tgt is None:
            for k, v in m.items():
                if k in n or n in k:
                    tgt = v
                    break
        if tgt is None:
            raise KeyError(f"industry not in concordance: {s}")
        out.append(tgt)
    return out


def block_scores(directCO2: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Per-member-state marker intensities, z-scores, composite and tercile block."""
    def inten(c, secs):
        o = sum(x.get((c, s), 0.0) for s in secs)
        return sum(directCO2.get((c, s), 0.0) for s in secs) / o if o > 0 else 0.0
    feat = pd.DataFrame({
        "elec": {c: inten(c, ["ELF", "ELR"]) for c in EU_ISO},
        "steel": {c: inten(c, ["STL"]) for c in EU_ISO},
        "cement": {c: inten(c, ["CEM"]) for c in EU_ISO},
    })
    z = (feat - feat.mean()) / feat.std(ddof=0)
    out = feat.copy()
    out["composite"] = z.mean(axis=1)
    out = out.sort_values("composite")
    out["rank"] = range(1, len(out) + 1)
    out["block_rule"] = ["EUC" if i < 9 else "EUM" if i < 18 else "EUD" for i in range(len(out))]
    return out


def assign_blocks(directCO2: pd.Series, x: pd.Series) -> dict[str, str]:
    return block_scores(directCO2, x)["block_rule"].to_dict()


def read_mapping(path: str) -> dict[str, str]:
    name2iso = {v: k for k, v in ISO2NAME.items()}
    m = pd.read_csv(path)
    out = {name2iso[r["country"]]: r["block"] for _, r in m.iterrows()}
    missing = sorted(set(EU_ISO) - set(out))
    if missing:
        raise KeyError(f"{path}: no block for {missing}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to EXIOBASE IOT_2022_ixi.zip")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "exiobase3_2022_10x9.xlsx"))
    ap.add_argument("--blocks", choices=["csv", "rule"], default="csv",
                    help="EU block assignment used for aggregation: the versioned "
                         "eu_block_mapping.csv (input of record, default) or the z-score "
                         "tercile rule recomputed from the raw data.")
    ap.add_argument("--write-mapping", action="store_true",
                    help="With --blocks rule: overwrite eu_block_mapping.csv with the rule's "
                         "assignment. Never done implicitly.")
    args = ap.parse_args()

    import pymrio

    print("parsing raw EXIOBASE (heavy)...", flush=True)
    io = pymrio.parse_exiobase3(args.raw)
    io.calc_all()

    print("aggregating 163 -> 9 sectors...", flush=True)
    agg = sector_map(io.get_sectors().tolist())
    print("  sector counts:", dict(Counter(agg)), flush=True)
    io.aggregate(sector_agg=agg, inplace=True)

    ext = io.air_emissions
    co2_rows = [i for i in ext.F.index if isinstance(i, str) and i.startswith("CO2 ")
                and "bio" not in i.lower() and "peat" not in i.lower()]
    directCO2 = ext.F.loc[co2_rows].sum(axis=0)
    x = io.x["indout"] if getattr(io, "x", None) is not None else (io.Z.sum(1) + io.Y.sum(1))

    print("EU block rule (z-score composite of marker intensities, terciles)...", flush=True)
    scores = block_scores(directCO2, x)
    rule = scores["block_rule"].to_dict()
    csv_path = os.path.join(HERE, "eu_block_mapping.csv")
    versioned = read_mapping(csv_path)
    scores["block_csv"] = [versioned[c] for c in scores.index]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(scores.to_string(), flush=True)
    mismatch = sorted(c for c in EU_ISO if rule[c] != versioned[c])
    print(f"  rule vs versioned mapping: {len(mismatch)} difference(s) {mismatch}", flush=True)
    blocks = rule if args.blocks == "rule" else versioned
    print(f"  aggregating with --blocks {args.blocks}", flush=True)
    if args.write_mapping:
        if args.blocks != "rule":
            raise SystemExit("--write-mapping requires --blocks rule")
        mapping = pd.DataFrame(
            [(ISO2NAME[c], rule[c]) for c in sorted(EU_ISO, key=lambda k: (rule[k], k))],
            columns=["country", "block"])
        mapping.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}", flush=True)
    for b in ["EUC", "EUM", "EUD"]:
        print(f"  {b}: {sorted(c for c in EU_ISO if blocks[c] == b)}", flush=True)

    print("aggregating regions...", flush=True)
    rmap = {**blocks, **NONEU}
    io.aggregate(region_agg=[rmap[r] for r in io.get_regions()], inplace=True)
    REGS = sorted(io.get_regions())
    labels = [f"{r}_{s}" for r in REGS for s in SECTORS]

    Z = io.Z.copy()
    Z.index = [f"{r}_{s}" for r, s in Z.index]
    Z.columns = [f"{r}_{s}" for r, s in Z.columns]
    Z = Z.reindex(index=labels, columns=labels)

    Y = io.Y.copy()
    Y.index = [f"{r}_{s}" for r, s in Y.index]
    h_cols = {f"{r}_h": Y[[(r, c) for c in H_CATS]].sum(axis=1) for r in REGS}
    i_cols = {f"{r}_i": Y[[(r, c) for c in I_CATS]].sum(axis=1) for r in REGS}
    if CHINA_FRAC > 0 and "CHN_i" in i_cols:
        shift = CHINA_FRAC * i_cols["CHN_i"]
        h_cols["CHN_h"] = h_cols["CHN_h"] + shift
        i_cols["CHN_i"] = i_cols["CHN_i"] - shift
        print(f"  China I->C rule: moved {shift.sum():,.0f} ({CHINA_FRAC:.0%} of GFCF) to consumption", flush=True)
    FD = pd.DataFrame({**h_cols, **i_cols}).reindex(labels)

    Fv = io.factor_inputs.F.copy()
    Fv.columns = [f"{r}_{s}" for r, s in Fv.columns]
    L = Fv.loc[[r for r in Fv.index if r.startswith("Compensation of employees")]].sum(axis=0)
    K = Fv.loc[[r for r in Fv.index if r.startswith("Operating surplus")]].sum(axis=0)
    TAX = Fv.loc["Other net taxes on production"]

    col_order = labels + [f"{r}_h" for r in REGS] + [f"{r}_i" for r in REGS]
    iot = pd.DataFrame(0.0, index=labels + ["L", "K", "TAX"], columns=col_order)
    iot.loc[labels, labels] = Z.values
    iot.loc[labels, [f"{r}_h" for r in REGS] + [f"{r}_i" for r in REGS]] = FD.values
    iot.loc["L", labels] = L.reindex(labels).values
    iot.loc["K", labels] = K.reindex(labels).values
    iot.loc["TAX", labels] = TAX.reindex(labels).values

    Fe = io.air_emissions.F.copy()
    Fe.columns = [f"{r}_{s}" for r, s in Fe.columns]

    def basket(terms):
        acc = pd.Series(0.0, index=Fe.columns)
        for row, w in terms:
            if row in Fe.index:
                acc = acc + w * Fe.loc[row]
        return acc * KG_TO_MT

    ff = basket(FF_TERMS)
    proc = basket(PROC_TERMS)
    emi = pd.DataFrame({lbl: {"GHG_FF": ff.get(lbl, 0.0), "GHG_PROC": proc.get(lbl, 0.0)}
                        for lbl in labels})

    out = os.path.abspath(args.out)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        iot.to_excel(w, sheet_name="IOT")
        emi.to_excel(w, sheet_name="emi")
    print(f"wrote {out}: IOT {iot.shape}, emi {emi.shape}, regions {REGS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
