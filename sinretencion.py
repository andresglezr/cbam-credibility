"""
===============================================================================
  Dynamic Hat Algebra — Ramsey CGE — N REG + ARMINGTON + BOP + CARBONO
  v3-EN: Energy nesting (CES: VA vs Energy, CES: ELC vs REF)
  
  Production structure:
    XD = Leontief(Materials, VAE_bundle, TAX, tau_c)
      VAE = CES(VA, EN; σ_VAE=0.5)
        VA = K^αF × L^(1-αF)
        EN = CES(ELC, REF; σ_EN=0.5)
  
  Eliminadas por sustitución:
    H_E  + EQTRADE   → E(r,s,rp) = M(rp,s,r), sustituido en EQXDD
    H_PM + EQPM      → PM = PD(origen) + cbam, sustituido en EQPMAGG/EQIMPORT_BOT
    LAMP + EQFSTORD   → λ = PK + 2φ·PD·I/K, sustituido en EQLAMBDA
  
  PK: Leontief. Costes de ajuste: SÍ (φ > 0).
  Elasticidades Armington sector-específicas (GTAP ESUBD/ESUBM).
===============================================================================
"""

import gamspy as gp
from gamspy import (Container, Set, Alias, Parameter, Variable, Equation,
                    Model, Sum, Product)
from gamspy.math import sqrt as _gsqrt
import numpy as np
import hashlib
from datetime import datetime, timezone
import os
import pandas as pd
import sys

gp.set_options({"ALLOW_AMBIGUOUS_EQUATIONS": "yes"})

def env_str(name, default):
    val = os.environ.get(name)
    return default if val is None or val == "" else val

def env_float(name, default):
    val = os.environ.get(name)
    return default if val is None or val == "" else float(val)

def env_int(name, default):
    val = os.environ.get(name)
    return default if val is None or val == "" else int(val)

def env_bool(name, default):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on", "si", "sí"}

#: Economic-specification version of this model file.  Bump it whenever an
#: equation changes in a way that makes previously solved families incomparable.
#: 2 = Euler equation converts the numeraire return with p^C(t-1)/p^C(t).
MODEL_SPEC_VERSION = 2
MODEL_SPEC_LABEL = "euler-consumption-deflator"

RUN_LABEL = env_str("RAMSEY_RUN_LABEL", "baseline")
OUT_SUFFIX = env_str("RAMSEY_OUT_SUFFIX", "v2")
OUT_DIR = env_str("RAMSEY_OUT_DIR", "results")

print(f"Run label: {RUN_LABEL}")
print(f"Output target: {OUT_DIR}/*_{OUT_SUFFIX}.parquet")

# =============================================================================
# 1. SAM DATA
# =============================================================================
SAM_FILE = "exiobase3_2022_10x9.xlsx"
SAM_SHEET = "IOT"
EMIS_SHEET = "emi"
SCALE = 1000.0

SAM = pd.read_excel(SAM_FILE, sheet_name=SAM_SHEET, index_col=0)
print(f"SAM importada ({SAM_FILE}, hoja {SAM_SHEET}):")
print(SAM); print()

emis_df = pd.read_excel(SAM_FILE, sheet_name=EMIS_SHEET, index_col=0)
print(f"Emisiones importadas ({SAM_FILE}, hoja {EMIS_SHEET}):")
print(emis_df); print()

# Parse emissions: two rows — GHG_FF (fuel combustion) and GHG_PROC (process)
emis_ff_absolute = {}   # Fuel combustion emissions (Mt CO2eq), attributable to REF use
emis_proc_absolute = {} # Process emissions (Mt CO2eq), Leontief, unavoidable

for col in emis_df.columns:
    col_s = str(col).strip()
    # Column format: REG_SEC (e.g., EUD_STL, CHN_ELF)
    # Need to split into region and sector — sectors can have underscores (ELF)
    # Strategy: try matching known regions at the start
    matched = False
    for r_candidate in sorted([str(c).strip() for c in emis_df.columns], key=len, reverse=True):
        pass  # placeholder
    # Simpler: rsplit won't work for ELF. Use region prefix matching.
    found_r = None
    found_s = None
    for prefix_len in [3, 4, 5]:  # Try different region name lengths
        r_try = col_s[:prefix_len]
        s_try = col_s[prefix_len+1:] if len(col_s) > prefix_len and col_s[prefix_len] == '_' else None
        if s_try and r_try in col_s:
            found_r = r_try
            found_s = s_try
            break
    if found_r is None or found_s is None:
        continue
    r, s = found_r, found_s
    # Read GHG_FF row
    if "GHG_FF" in emis_df.index:
        try:
            emis_ff_absolute[(r,s)] = float(emis_df.loc["GHG_FF", col])
        except (ValueError, TypeError):
            emis_ff_absolute[(r,s)] = 0.0
    elif "GHG_FF (Mt CO2eq)" in emis_df.index:
        try:
            emis_ff_absolute[(r,s)] = float(emis_df.loc["GHG_FF (Mt CO2eq)", col])
        except (ValueError, TypeError):
            emis_ff_absolute[(r,s)] = 0.0
    # Read GHG_PROC row
    if "GHG_PROC" in emis_df.index:
        try:
            emis_proc_absolute[(r,s)] = float(emis_df.loc["GHG_PROC", col])
        except (ValueError, TypeError):
            emis_proc_absolute[(r,s)] = 0.0
    elif "GHG_PROC (Mt CO2eq)" in emis_df.index:
        try:
            emis_proc_absolute[(r,s)] = float(emis_df.loc["GHG_PROC (Mt CO2eq)", col])
        except (ValueError, TypeError):
            emis_proc_absolute[(r,s)] = 0.0

# Fallback: if no rows found, try reading by position (row 0 = FF, row 1 = PROC)
if len(emis_ff_absolute) == 0:
    print("  WARNING: GHG_FF row not found by name, trying positional read...")
    emis_df_noidx = pd.read_excel(SAM_FILE, sheet_name=EMIS_SHEET)
    for col in emis_df_noidx.columns:
        col_s = str(col).strip()
        found_r = None; found_s = None
        for prefix_len in [3, 4, 5]:
            r_try = col_s[:prefix_len]
            s_try = col_s[prefix_len+1:] if len(col_s) > prefix_len and col_s[prefix_len] == '_' else None
            if s_try:
                found_r = r_try; found_s = s_try; break
        if found_r is None: continue
        r, s = found_r, found_s
        try: emis_ff_absolute[(r,s)] = float(emis_df_noidx[col].iloc[0])
        except: emis_ff_absolute[(r,s)] = 0.0
        try: emis_proc_absolute[(r,s)] = float(emis_df_noidx[col].iloc[1])
        except: emis_proc_absolute[(r,s)] = 0.0

print(f"Emisiones FF leídas: {len(emis_ff_absolute)} sectores-región")
print(f"Emisiones PROC leídas: {len(emis_proc_absolute)} sectores-región")

# Combined for backward compatibility
emis_absolute = {}
for k in set(list(emis_ff_absolute.keys()) + list(emis_proc_absolute.keys())):
    emis_absolute[k] = emis_ff_absolute.get(k, 0.0) + emis_proc_absolute.get(k, 0.0)

FACTORS = ["L", "K"]

_h_cols = {}; _i_cols = {}; _act_cols = []
for lbl in [str(c).strip() for c in SAM.columns]:
    if lbl in FACTORS: continue
    if lbl.endswith("_h"):
        # Household column: everything before last "_h" is region name
        _h_cols[lbl[:-2]] = lbl
    elif lbl.endswith("_i"):
        # Investment column: everything before last "_i" is region name
        _i_cols[lbl[:-2]] = lbl
    else:
        _act_cols.append(lbl)

# Parse activity columns: REG_SEC where SEC may contain underscores (e.g., ELF)
# Strategy: first pass to identify region prefixes (3-letter codes before first _)
_region_candidates = set()
for lbl in _act_cols:
    idx = lbl.find("_")
    if idx > 0:
        _region_candidates.add(lbl[:idx])

_parsed = {}
for lbl in _act_cols:
    # Try each known region prefix
    matched = False
    for rc in sorted(_region_candidates, key=len, reverse=True):
        if lbl.startswith(rc + "_"):
            reg_name = rc
            sec_name = lbl[len(rc)+1:]
            if reg_name not in _parsed: _parsed[reg_name] = {}
            _parsed[reg_name][sec_name] = lbl
            matched = True
            break
    if not matched:
        print(f"  WARNING: Could not parse column '{lbl}'")

regions = sorted(_parsed.keys())
sectors = sorted(list(set(s for secs in _parsed.values() for s in secs.keys())))

REG_CFG = {}
for r in regions:
    a_labels = [_parsed[r][s] for s in sectors]
    REG_CFG[r] = {"a": a_labels, "h": _h_cols.get(r, f"{r}_h"), "i": _i_cols.get(r, f"{r}_i")}

NUMERAIRE_REG = regions[0]
print(f"Regiones: {regions}, Sectores: {sectors}, Numerario: {NUMERAIRE_REG}")

sam_int = {}; sam_c = {}; sam_i = {}; sam_f = {}
for r in regions:
    cfg = REG_CFG[r]
    for j, s in enumerate(sectors):
        col = cfg["a"][j]
        for r2 in regions:
            cfg2 = REG_CFG[r2]
            for i2, si in enumerate(sectors):
                row = cfg2["a"][i2]
                sam_int[(si, r2, s, r)] = float(SAM.loc[row, col])
        for fac in FACTORS:
            sam_f[(fac, s, r)] = float(SAM.loc[fac, col])
    for r2 in regions:
        cfg2 = REG_CFG[r2]
        for i2, si in enumerate(sectors):
            row = cfg2["a"][i2]
            sam_c[(si, r2, r)] = float(SAM.loc[row, cfg["h"]])
            sam_i[(si, r2, r)] = float(SAM.loc[row, cfg["i"]])

sam_tax = {}
if "TAX" in SAM.index:
    for r in regions:
        cfg = REG_CFG[r]
        for j, s in enumerate(sectors):
            col = cfg["a"][j]
            sam_tax[(s,r)] = float(SAM.loc["TAX", col])
    print(f"TAX row found: {len(sam_tax)} entries")
else:
    for r in regions:
        for s in sectors:
            sam_tax[(s,r)] = 0.0
    print("No TAX row in SAM — setting to zero")

USE_BASE_TAX_WEDGES = env_bool("RAMSEY_USE_BASE_TAX_WEDGES", True)
print(
    "Benchmark tax wedges: "
    + ("ON (TAX row enters production and household tax revenue)"
       if USE_BASE_TAX_WEDGES
       else "OFF (TAX row absorbed into primary factors; taxShare=0)")
)

# =============================================================================
# 2. DERIVAR DATOS
# =============================================================================
d = {}; int_dom = {}; int_imp = {}; trade_bil = {}; trade_m_bil = {}

for r in regions:
    partners = [rp for rp in regions if rp != r]
    for s in sectors:
        for si in sectors:
            int_dom[(r,si,s)] = sam_int[(si,r,s,r)] / SCALE
            imp = sum(sam_int[(si,rp,s,r)] for rp in partners) / SCALE
            int_imp[(r,si,s)] = imp
        l_val = sam_f[("L",s,r)]
        k_val = sam_f[("K",s,r)]
        tax_raw = sam_tax.get((s,r), 0.0)
        tax_val = tax_raw if USE_BASE_TAX_WEDGES else 0.0
        va = l_val + k_val
        tax_accounting = tax_raw
        xd_cost = sum(sam_int[(si,ro,s,r)] for si in sectors for ro in regions) + va + tax_accounting
        c_total = sum(sam_c[(s,ro,r)] for ro in regions)
        i_total = sum(sam_i[(s,ro,r)] for ro in regions)
        d[(r,s)] = {"xd": xd_cost/SCALE, "va": va/SCALE, "l": l_val/SCALE,
                    "ky": k_val/SCALE, "c_raw": c_total/SCALE, "iz": i_total/SCALE,
                    "tax": tax_val/SCALE, "tax_raw": tax_raw/SCALE}
    for s in sectors:
        for rp in partners:
            e_val = sum(sam_int[(s,r,sj,rp)] for sj in sectors) + sam_c[(s,r,rp)] + sam_i[(s,r,rp)]
            trade_bil[(r,s,rp)] = e_val / SCALE

for r in regions:
    for s in sectors:
        for rp in [x for x in regions if x != r]:
            trade_m_bil[(r,s,rp)] = trade_bil[(rp,s,r)]

trade_m_total = {}; trade_e_total = {}
for r in regions:
    for s in sectors:
        partners = [rp for rp in regions if rp != r]
        trade_m_total[(r,s)] = sum(trade_m_bil[(r,s,rp)] for rp in partners)
        trade_e_total[(r,s)] = sum(trade_bil[(r,s,rp)] for rp in partners)

vbop_val = {}
for r in regions:
    vbop_val[r] = sum(trade_e_total[(r,s)] for s in sectors) - sum(trade_m_total[(r,s)] for s in sectors)
print(f"VBOP: {', '.join(f'{r}={vbop_val[r]:.6f}' for r in regions)}, sum={sum(vbop_val.values()):.6f}")

io_val = {}
for r in regions:
    for si in sectors:
        for sj in sectors:
            io_val[(r,si,sj)] = (int_dom[(r,si,sj)] + int_imp[(r,si,sj)]) / d[(r,sj)]["xd"]

# --- Energy nesting: define energy sectors ---
# Three-level nesting: VAE = CES(VA, EN), EN = CES(ELC, REF), ELC = CES(ELF, ELR)
ENERGY_SECS = ["ELF", "ELR", "REF"]
ELC_SECS = ["ELF", "ELR"]  # Sub-sectors of electricity composite
MAT_SECS = [s for s in sectors if s not in ENERGY_SECS]
print(f"Energy sectors: {ENERGY_SECS}, ELC sub-sectors: {ELC_SECS}, Material sectors: {MAT_SECS}")

en_data = {}
for r in regions:
    for s in sectors:
        rs = (r,s)
        # Electricity inputs (fossil + renewable)
        elc_ff_input = (int_dom[(r,"ELF",s)] + int_imp[(r,"ELF",s)])
        elc_rn_input = (int_dom[(r,"ELR",s)] + int_imp[(r,"ELR",s)])
        elc_input = elc_ff_input + elc_rn_input  # ELC composite
        ref_input = (int_dom[(r,"REF",s)] + int_imp[(r,"REF",s)])
        en_total = elc_input + ref_input
        en_data[rs] = {
            "elc_ff": elc_ff_input, "elc_rn": elc_rn_input,
            "elc": elc_input, "ref": ref_input, "en_total": en_total,
            "en_share_xd": en_total / d[rs]["xd"] if d[rs]["xd"] > 1e-10 else 0.0,
        }
        # ELC composite shares (within EN bundle)
        if en_total > 1e-10:
            en_data[rs]["theta_elc"] = elc_input / en_total
            en_data[rs]["theta_ref"] = ref_input / en_total
        else:
            en_data[rs]["theta_elc"] = 0.5
            en_data[rs]["theta_ref"] = 0.5
        # ELF vs ELR shares (within ELC composite)
        if elc_input > 1e-10:
            en_data[rs]["theta_ff"] = elc_ff_input / elc_input
            en_data[rs]["theta_rn"] = elc_rn_input / elc_input
        else:
            en_data[rs]["theta_ff"] = 0.5
            en_data[rs]["theta_rn"] = 0.5

SIGMA_VAE = env_float("RAMSEY_SIGMA_VAE", 0.5)
SIGMA_EN  = env_float("RAMSEY_SIGMA_EN", 0.5)
SIGMA_ELC = env_float("RAMSEY_SIGMA_ELC", 3.0)   # ELF vs ELR: high substitutability (long-run)
print(f"CES elasticities: σ_VAE={SIGMA_VAE}, σ_EN={SIGMA_EN}, σ_ELC={SIGMA_ELC}")

# =============================================================================
# 3. CALIBRACION
# =============================================================================
T_MAX = env_int("RAMSEY_T_MAX", 30)
g_val = env_float("RAMSEY_G", 0.02)
rss_val = env_float("RAMSEY_RSS", 0.05)
ro_val = env_float("RAMSEY_RO", 0.05)
adjsh_val = env_float("RAMSEY_ADJSH", 0.05)
rp_val = env_float("RAMSEY_RP", 0.0)  # rp=0: no retained profits
gamma_val = env_float("RAMSEY_GAMMA", 2.0)  # CIES risk aversion (1=log, 2=standard)

ESUBD_DICT = {
    "STL": 2.95, "ALU": 3.95, "CEM": 2.9, "ELF": 2.8, "ELR": 2.8,
    "FER": 3.75, "CHM": 3.3,  "REF": 2.1, "OTH": 2.9,
}
ESUBM_DICT = {
    "STL": 5.9,  "ALU": 7.9,  "CEM": 5.8,  "ELF": 5.6, "ELR": 5.6,
    "FER": 7.5,  "CHM": 6.6,  "REF": 4.2,  "OTH": 5.8,
}
ESUBD_DEFAULT = 2.9
ESUBM_DEFAULT = 5.8

ARMINGTON_SCALE = env_float("RAMSEY_ARMINGTON_SCALE", 1.0)
sigmaA_dict = {s: ESUBD_DICT.get(s, ESUBD_DEFAULT) * ARMINGTON_SCALE for s in sectors}
sigmaM_dict = {s: ESUBM_DICT.get(s, ESUBM_DEFAULT) * ARMINGTON_SCALE for s in sectors}
print(f"Elasticidades Armington sector-específicas:")
print(f"  Armington scale={ARMINGTON_SCALE:.3f}")
for s in sectors:
    print(f"  {s}: σA={sigmaA_dict[s]:.2f}  σM={sigmaM_dict[s]:.2f}")

CARBON_PRICE = env_float("RAMSEY_CARBON_PRICE_EUR", 100.0) / SCALE
EU_REGIONS = [r for r in regions if r.startswith("EU")]
print(f"EU regions for carbon tax/CBAM: {EU_REGIONS}")

BASE_YEAR = 2022
T_STAR = env_int("RAMSEY_T_STAR", 8)
T_STAR_YEAR = BASE_YEAR + T_STAR - 1
T_MAX_P2 = T_MAX - T_STAR + 1  # Phase 2 periods so all scenarios end in same year

def model_year(t, phase2=False):
    return BASE_YEAR + ((T_STAR - 1) if phase2 else 0) + t - 1

print(
    f"Revelation / Phase 2 start: T*={T_STAR} ({T_STAR_YEAR}); "
    f"Phase 2 horizon T=1..{T_MAX_P2} ({T_STAR_YEAR}-{BASE_YEAR + T_MAX - 1})"
)

FA_SCHEDULE = {
    1: 1.000, 2: 1.000, 3: 1.000, 4: 1.000,
    5: 0.975, 6: 0.950, 7: 0.900,
    8: 0.775,
    9: 0.515, 10: 0.390, 11: 0.265, 12: 0.140,
}
def get_fa(t):
    return FA_SCHEDULE.get(t, 0.0)

print(f"\nFree allocation schedule (T=1..15):")
for t in range(1, 16):
    fa = get_fa(t)
    year = BASE_YEAR + t - 1
    print(f"  T={t:2d} ({year}): FA={fa:.3f}  CBAM_factor={1-fa:.3f}")

cal = {}
for r in regions:
    total_ky = sum(d[(r,s)]["ky"] for s in sectors)
    total_i  = sum(d[(r,s)]["iz"] for s in sectors)
    for s in sectors:
        rs = (r,s); c = {}
        c["alphaF"] = d[rs]["ky"] / d[rs]["va"]
        c["inv"] = (d[rs]["ky"]/total_ky) * total_i
        c["c_adj"] = d[rs]["c_raw"] - adjsh_val*c["inv"]
        c["lambdaz"] = 1.0 + 2*adjsh_val
        vaShare = d[rs]["va"]/d[rs]["xd"]
        taxShare = d[rs]["tax"]/d[rs]["xd"]
        enShare = en_data[rs]["en_share_xd"]
        sum_mat_io = sum(io_val[(r,si,s)] for si in MAT_SECS)
        vaeShare = vaShare + enShare
        vae_val = d[rs]["va"] + en_data[rs]["en_total"]
        c["vae"] = vae_val
        c["vaeShare"] = vaeShare
        c["enShare"] = enShare
        c["theta_va"] = d[rs]["va"] / vae_val if vae_val > 1e-10 else 0.99
        c["theta_en"] = en_data[rs]["en_total"] / vae_val if vae_val > 1e-10 else 0.01
        c["pva"] = (1.0 - sum_mat_io - enShare - taxShare) / vaShare
        c["pvae"] = (1.0 - sum_mat_io - taxShare) / vaeShare
        c["taxShare"] = taxShare
        c["vaShare"] = vaShare
        af = c["alphaF"]; lam = c["lambdaz"]; inv = c["inv"]
        va = d[rs]["va"]; pva = c["pva"]
        c["kz"] = (af*pva*va+adjsh_val*inv-lam*inv)/((rss_val-g_val)*lam)
        c["phi"] = adjsh_val*c["kz"]/inv
        c["delta"] = inv/c["kz"]-g_val
        c["aF"] = va/(c["kz"]**af*d[rs]["l"]**(1-af))
        c["div"] = pva*va-c["phi"]*inv**2/c["kz"]-d[rs]["l"]-rp_val*inv
        c["adj"] = adjsh_val*inv
        d[rs]["e_total"] = trade_e_total[rs]; d[rs]["m_total"] = trade_m_total[rs]
        d[rs]["xdd"] = d[rs]["xd"] - c["adj"] - d[rs]["e_total"]
        d[rs]["x"] = d[rs]["xdd"] + d[rs]["m_total"]
        cal[rs] = c

    # --- Bröcker-style income: Y = labor + capital_income - adj_costs - VBOP + TAXREV ---
    # Capital income per sector = αF × PD × XD (= ky in SAM, since PD=1)
    # Adjustment costs reduce available output (deadweight loss in production)
    # No dividends, no retained profits. Household receives full factor income net of adj.
    sz = total_i  # S = total investment (household finances all)
    total_ky = sum(d[(r,s)]["ky"] for s in sectors)
    total_l = sum(d[(r,s)]["l"] for s in sectors)
    total_adj = sum(cal[(r,s)]["adj"] for s in sectors)
    if USE_BASE_TAX_WEDGES:
        total_tax_rev = sum(cal[(r,s)]["taxShare"] * d[(r,s)]["xd"] for s in sectors)
    else:
        total_tax_rev = sum(d[(r,s)]["tax_raw"] for s in sectors)
    yz = total_ky + total_l - total_adj - vbop_val[r] + total_tax_rev
    c_total = sum(cal[(r,s)]["c_adj"] for s in sectors)
    print(f"  {r}: Y={yz:.4f}, Y-S={yz-sz:.4f}, C={c_total:.4f}, diff={yz-sz-c_total:.6f}")
    for s in sectors:
        cal[(r,s)]["alphaH"] = cal[(r,s)]["c_adj"]/(yz-sz)
        cal[(r,s)]["alphaI"] = d[(r,s)]["iz"]/total_i
        cal[(r,s)]["sz"] = sz; cal[(r,s)]["yz"] = yz; cal[(r,s)]["lsz"] = total_l
        cal[(r,s)]["ky_share"] = d[(r,s)]["ky"] / yz  # capital income share of Y

arm_top = {}; arm_bot = {}
for r in regions:
    partners = [rp for rp in regions if rp != r]
    for s in sectors:
        rs = (r,s); xdd = d[rs]["xdd"]; m_tot = d[rs]["m_total"]; x = d[rs]["x"]
        gammaT = m_tot / x if (x > 1e-10 and m_tot > 1e-10) else 0.001
        arm_top[rs] = {"gammaA": gammaT, "aA": 1.0}
        gammas = {}
        for rp in partners:
            m_val = trade_m_bil.get((r,s,rp), 0)
            gammas[rp] = m_val / m_tot if m_tot > 1e-10 else 1.0/len(partners)
        arm_bot[rs] = {"aM": 1.0, "gamma": gammas}

emis_intensity = {}      # Total (FF + PROC) per unit XD
emis_ff_intensity = {}   # Fuel combustion per unit XD
emis_proc_intensity = {} # Process per unit XD
emis_fuel_per_ref = {}   # Fuel combustion per unit of REF input (for tau_fuel surcharge)
cbam_direct_intensity = {}
cbam_indirect_elc_intensity = {}
cbam_embodied_intensity = {}
for r in regions:
    for s in sectors:
        xd = d[(r,s)]["xd"]
        eff = emis_ff_absolute.get((r,s), 0.0)
        epr = emis_proc_absolute.get((r,s), 0.0)
        emis_ff_intensity[(r,s)] = eff / xd if xd > 1e-10 else 0.0
        emis_proc_intensity[(r,s)] = epr / xd if xd > 1e-10 else 0.0
        emis_intensity[(r,s)] = (eff + epr) / xd if xd > 1e-10 else 0.0
        # Fuel emissions per unit of REF input to this sector
        ref_input = en_data[(r,s)]["ref"]
        emis_fuel_per_ref[(r,s)] = eff / ref_input if ref_input > 1e-10 else 0.0

# Foreign origin-sector process and indirect-electricity coefficients are fixed
# at benchmark technology.  The common EU benchmark below nevertheless updates
# fuel combustion with equilibrium REF demand in every scenario and year.
#
# The no-environment default is the paper's legal-coverage approximation:
# indirect electricity for CEM and FER only.  The historical boolean remains an
# explicit backward-compatibility override, while a sector list takes priority.
_cbam_indirect_legacy_env = os.environ.get("RAMSEY_CBAM_INDIRECT_ELECTRICITY")
_CBAM_INDIRECT_LEGACY_ENABLED = (
    env_bool("RAMSEY_CBAM_INDIRECT_ELECTRICITY", False)
    if _cbam_indirect_legacy_env is not None else False
)
_cbam_indirect_sector_env = os.environ.get("RAMSEY_CBAM_INDIRECT_SECTORS")
CBAM_INDIRECT_SECTOR_OVERRIDE = _cbam_indirect_sector_env is not None
if CBAM_INDIRECT_SECTOR_OVERRIDE:
    _requested_indirect_sectors = [
        token.strip().upper()
        for token in _cbam_indirect_sector_env.split(",")
        if token.strip()
    ]
    if len(_requested_indirect_sectors) == 1 and _requested_indirect_sectors[0] in {"ALL", "*"}:
        CBAM_INDIRECT_SECTORS = [s for s in sectors if s not in ELC_SECS]
    elif len(_requested_indirect_sectors) == 1 and _requested_indirect_sectors[0] in {"NONE", "OFF"}:
        CBAM_INDIRECT_SECTORS = []
    else:
        _unknown_indirect_sectors = sorted(set(_requested_indirect_sectors) - set(sectors))
        if _unknown_indirect_sectors:
            raise ValueError(
                "RAMSEY_CBAM_INDIRECT_SECTORS contains unknown sectors: "
                + ", ".join(_unknown_indirect_sectors)
            )
        _electric_indirect_sectors = sorted(set(_requested_indirect_sectors) & set(ELC_SECS))
        if _electric_indirect_sectors:
            raise ValueError(
                "Indirect electricity cannot be assigned to electricity sectors: "
                + ", ".join(_electric_indirect_sectors)
            )
        CBAM_INDIRECT_SECTORS = [
            s for s in sectors if s in set(_requested_indirect_sectors)
        ]
    CBAM_INDIRECT_COVERAGE_MODE = "explicit_sector_list"
elif _cbam_indirect_legacy_env is not None:
    CBAM_INDIRECT_SECTORS = (
        [s for s in sectors if s not in ELC_SECS]
        if _CBAM_INDIRECT_LEGACY_ENABLED else []
    )
    CBAM_INDIRECT_COVERAGE_MODE = (
        "legacy_all_non_electric" if _CBAM_INDIRECT_LEGACY_ENABLED else "legacy_none"
    )
else:
    CBAM_INDIRECT_SECTORS = [s for s in ("CEM", "FER") if s in sectors]
    CBAM_INDIRECT_COVERAGE_MODE = "central_default"

# Effective boolean retained for old diagnostics and metadata consumers.
CBAM_INCLUDE_INDIRECT_ELECTRICITY = bool(CBAM_INDIRECT_SECTORS)
CBAM_INCLUDE_PROCESS_EMISSIONS = env_bool("RAMSEY_CBAM_PROCESS_EMISSIONS", True)
CBAM_EU_BENCHMARK_ENDOGENOUS = env_bool("RAMSEY_CBAM_EU_BENCHMARK_ENDOGENOUS", True)
CBAM_FLOOR_EPS = env_float("RAMSEY_CBAM_FLOOR_EPS", 1e-8)
print(
    "CBAM indirect-electricity coverage: "
    f"mode={CBAM_INDIRECT_COVERAGE_MODE}, "
    f"sectors={CBAM_INDIRECT_SECTORS or ['none']}"
)
for r in regions:
    for s in sectors:
        xd = d[(r,s)]["xd"]
        direct = emis_ff_intensity.get((r,s), 0.0)
        if CBAM_INCLUDE_PROCESS_EMISSIONS:
            direct += emis_proc_intensity.get((r,s), 0.0)

        indirect_elc_abs = 0.0
        if s in CBAM_INDIRECT_SECTORS:
            for er in ELC_SECS:
                for ro in regions:
                    input_val = sam_int[(er, ro, s, r)] / SCALE
                    indirect_elc_abs += input_val * emis_intensity.get((ro, er), 0.0)

        cbam_direct_intensity[(r,s)] = direct
        cbam_indirect_elc_intensity[(r,s)] = indirect_elc_abs / xd if xd > 1e-10 else 0.0
        cbam_embodied_intensity[(r,s)] = (
            cbam_direct_intensity[(r,s)] + cbam_indirect_elc_intensity[(r,s)]
        )

print("\nIntensidades de emisión (total | fuel | process):")
for r in regions[:3]:  # Print first 3 regions
    print(f"  {r}:")
    for s in sectors:
        print(f"    {s}: total={emis_intensity[(r,s)]:.4f}  fuel={emis_ff_intensity[(r,s)]:.4f}  proc={emis_proc_intensity[(r,s)]:.4f}  fuel/REF={emis_fuel_per_ref[(r,s)]:.4f}")

# =============================================================================
# 4. GAMSPY CONTAINER
# =============================================================================
m = Container()

T    = Set(container=m, name="T",   records=[str(i) for i in range(1,T_MAX+1)])
reg  = Set(container=m, name="reg", records=regions)
sec  = Set(container=m, name="sec", records=sectors)
secc = Alias(container=m, name="secc", alias_with=sec)
regp = Alias(container=m, name="regp", alias_with=reg)
regpp = Alias(container=m, name="regpp", alias_with=reg)
TFIRST = Set(container=m, name="TFIRST", domain=T); TFIRST.setRecords(["1"])
TLAST  = Set(container=m, name="TLAST",  domain=T); TLAST.setRecords([str(T_MAX)])
TACTIVE = Set(container=m, name="TACTIVE", domain=T); TACTIVE.setRecords([str(i) for i in range(1, T_MAX+1)])
rr = Set(container=m, name="rr", domain=[reg,regp],
         records=[(r,rp) for r in regions for rp in regions if r != rp])
NUMERAIRE = NUMERAIRE_REG
RNNUM = Set(container=m, name="RNNUM", domain=[reg],
            records=[r for r in regions if r != NUMERAIRE])
EUREG = Set(container=m, name="EUREG", domain=[reg], records=EU_REGIONS)

# =============================================================================
# 5. PARAMETROS
# =============================================================================
_T = list(range(1,T_MAX+1))

g_p  = Parameter(container=m, name="g",  records=g_val)
ro_p = Parameter(container=m, name="ro", records=ro_val)
gamma_p = Parameter(container=m, name="gamma", records=gamma_val)
rp_p = Parameter(container=m, name="rp", records=rp_val)

# Material IO coefficients (excluding ELC, REF from rows)
io_mat_p = Parameter(container=m, name="io_mat", domain=[reg,sec,secc])
io_mat_recs = []
for r in regions:
    for si in sectors:
        for sj in sectors:
            if si in ENERGY_SECS:
                io_mat_recs.append((r, si, sj, 0.0))
            else:
                io_mat_recs.append((r, si, sj, io_val[(r,si,sj)]))
io_mat_p.setRecords(io_mat_recs)

# Full IO (kept for reference)
io_p = Parameter(container=m, name="io", domain=[reg,sec,secc])
io_p.setRecords([(r,si,sj,io_val[(r,si,sj)]) for r in regions for si in sectors for sj in sectors])

vaShare_p = Parameter(container=m, name="vaShare", domain=[reg,sec])
vaShare_p.setRecords([(r,s,cal[(r,s)]["vaShare"]) for r in regions for s in sectors])
vaeShare_p = Parameter(container=m, name="vaeShare", domain=[reg,sec])
vaeShare_p.setRecords([(r,s,cal[(r,s)]["vaeShare"]) for r in regions for s in sectors])
alphaF_p = Parameter(container=m, name="alphaF", domain=[reg,sec])
alphaF_p.setRecords([(r,s,cal[(r,s)]["alphaF"]) for r in regions for s in sectors])
phi_p = Parameter(container=m, name="phi", domain=[reg,sec])
phi_p.setRecords([(r,s,cal[(r,s)]["phi"]) for r in regions for s in sectors])
delta_p = Parameter(container=m, name="delta", domain=[reg,sec])
delta_p.setRecords([(r,s,cal[(r,s)]["delta"]) for r in regions for s in sectors])
alphaH_p = Parameter(container=m, name="alphaH", domain=[reg,sec])
alphaH_p.setRecords([(r,s,cal[(r,s)]["alphaH"]) for r in regions for s in sectors])
alphaI_p = Parameter(container=m, name="alphaI", domain=[reg,sec])
alphaI_p.setRecords([(r,s,cal[(r,s)]["alphaI"]) for r in regions for s in sectors])
taxShare_p = Parameter(container=m, name="taxShare", domain=[reg,sec])
taxShare_p.setRecords([(r,s,cal[(r,s)]["taxShare"]) for r in regions for s in sectors])

# --- Asymmetric (scrapping) adjustment costs — irreversibility extension ---
# Off by default: RAMSEY_SCRAP_MULT=0 reproduces the base model equations
# exactly (the extra terms are not built). When on, a gross investment rate
# x = I/K below the reference rate xbar = (1-margin)*(g+delta) incurs an
# extra convex penalty AC- = scrap_mult*phi * pD * K * pos(xbar - x)^2, with
# pos(z) = 0.5*(z + sqrt(z^2 + eps^2)) a smooth positive part (PATH-friendly).
# At the benchmark x = g+delta sits above xbar by margin*(g+delta), so the
# penalty and its derivatives are numerically zero there and the calibration
# is unchanged. Putty-clay motivation: heavy-industry capital is cheap to
# run at trend but expensive to run down (Dixit-Pindyck irreversibility).
SCRAP_MULT = env_float("RAMSEY_SCRAP_MULT", 0.0)
SCRAP_MARGIN = env_float("RAMSEY_SCRAP_MARGIN", 0.02)
POS_EPS2 = 1.0e-8  # eps^2 with eps = 1e-4 on the investment-rate scale
xbar_p = Parameter(container=m, name="xbar", domain=[reg,sec])
xbar_p.setRecords([(r, s, (1.0 - SCRAP_MARGIN) * (g_val + cal[(r,s)]["delta"]))
                   for r in regions for s in sectors])
sKZ_Y_p = Parameter(container=m, name="sKZ_Y", domain=[reg,sec])
sKZ_Y_p.setRecords([(r, s, cal[(r,s)]["kz"] / cal[(r,s)]["yz"]) for r in regions for s in sectors])
print(f"Scrap asymmetry: mult={SCRAP_MULT}, margin={SCRAP_MARGIN}" if SCRAP_MULT > 0
      else "Scrap asymmetry: OFF (symmetric quadratic adjustment costs)")

# Energy nesting parameters
sigmaVAE_p = Parameter(container=m, name="sigmaVAE", records=SIGMA_VAE)
sigmaEN_p  = Parameter(container=m, name="sigmaEN",  records=SIGMA_EN)
sigmaELC_p = Parameter(container=m, name="sigmaELC", records=SIGMA_ELC)

theta_elc_p = Parameter(container=m, name="theta_elc", domain=[reg,sec])
theta_elc_p.setRecords([(r,s, en_data[(r,s)]["theta_elc"]) for r in regions for s in sectors])
theta_ref_p = Parameter(container=m, name="theta_ref", domain=[reg,sec])
theta_ref_p.setRecords([(r,s, en_data[(r,s)]["theta_ref"]) for r in regions for s in sectors])

# ELC sub-nesting shares (ELF vs ELR within ELC composite)
theta_ff_p = Parameter(container=m, name="theta_ff", domain=[reg,sec])
theta_ff_p.setRecords([(r,s, en_data[(r,s)]["theta_ff"]) for r in regions for s in sectors])
theta_rn_p = Parameter(container=m, name="theta_rn", domain=[reg,sec])
theta_rn_p.setRecords([(r,s, en_data[(r,s)]["theta_rn"]) for r in regions for s in sectors])

pvaeVaeShare_p = Parameter(container=m, name="pvaeVaeShare", domain=[reg,sec])
pvaeVaeShare_p.setRecords([(r,s, cal[(r,s)]["pvae"]*cal[(r,s)]["vaeShare"]) 
                            for r in regions for s in sectors])

# CES price shares for EQPVAE (hat form)
# H_PVAE^(1-σ) = sVA_VAE × H_PVA^(1-σ) + sEN_VAE × H_PE^(1-σ)
# where sVA_VAE = θ_va × (pva/pvae)^(1-σ), sEN_VAE = θ_en × (1/pvae)^(1-σ)
sVA_VAE_p = Parameter(container=m, name="sVA_VAE", domain=[reg,sec])
sEN_VAE_p = Parameter(container=m, name="sEN_VAE", domain=[reg,sec])

# CES price-share records for all region-sector pairs.
_sva_recs = []; _sen_recs = []
for r in regions:
    for s in sectors:
        c = cal[(r,s)]
        pva = c["pva"]; pvae = c["pvae"]
        tv = c["theta_va"]; te = c["theta_en"]
        exp = 1.0 - SIGMA_VAE
        _sva_recs.append((r,s, tv * (pva/pvae)**exp if pvae > 1e-10 else 0.99))
        _sen_recs.append((r,s, te * (1.0/pvae)**exp if pvae > 1e-10 else 0.01))
sVA_VAE_p.setRecords(_sva_recs)
sEN_VAE_p.setRecords(_sen_recs)

# Armington parameters
gammaA_p = Parameter(container=m, name="gammaA", domain=[reg,sec])
gammaA_p.setRecords([(r,s,arm_top[(r,s)]["gammaA"]) for r in regions for s in sectors])
sigmaA_p = Parameter(container=m, name="sigmaA", domain=[sec])
sigmaA_p.setRecords([(s, sigmaA_dict[s]) for s in sectors])
sigmaM_p = Parameter(container=m, name="sigmaM", domain=[sec])
sigmaM_p.setRecords([(s, sigmaM_dict[s]) for s in sectors])
gammaM_p = Parameter(container=m, name="gammaM", domain=[reg,sec,regp])
gammaM_p.setRecords([(r,s,rp,arm_bot[(r,s)]["gamma"].get(rp,0))
                     for r in regions for s in sectors for rp in regions if rp != r])

vb_p = Parameter(container=m, name="vb", domain=[reg])
vb_p.setRecords([(r, vbop_val[r]) for r in regions])
B_VB = Parameter(container=m, name="B_VB", domain=[reg,T])
B_VB.setRecords([(r, str(t), vbop_val[r]*(1+g_val)**(t-1)) for r in regions for t in _T])

_taxrev_base = {}
for r in regions:
    if USE_BASE_TAX_WEDGES:
        _taxrev_base[r] = sum(cal[(r,s)]["taxShare"] * d[(r,s)]["xd"] for s in sectors)
    else:
        _taxrev_base[r] = sum(d[(r,s)]["tax_raw"] for s in sectors)
B_TAXREV = Parameter(container=m, name="B_TAXREV", domain=[reg,T])
B_TAXREV.setRecords([(r, str(t), _taxrev_base[r]*(1+g_val)**(t-1)) for r in regions for t in _T])

# =============================================================================
# 6. BASELINE PATHS
# =============================================================================
B_XD  = Parameter(container=m, name="B_XD",  domain=[reg,sec,T])
B_VA  = Parameter(container=m, name="B_VA",  domain=[reg,sec,T])
B_K   = Parameter(container=m, name="B_K",   domain=[reg,sec,T])
B_INV = Parameter(container=m, name="B_INV", domain=[reg,sec,T])
B_L   = Parameter(container=m, name="B_L",   domain=[reg,sec,T])
B_C   = Parameter(container=m, name="B_C",   domain=[reg,sec,T])
B_I   = Parameter(container=m, name="B_I",   domain=[reg,sec,T])
B_PD  = Parameter(container=m, name="B_PD",  domain=[reg,sec,T])
B_PVA  = Parameter(container=m, name="B_PVA",  domain=[reg,sec,T])
B_PVAE = Parameter(container=m, name="B_PVAE", domain=[reg,sec,T])
B_VAE  = Parameter(container=m, name="B_VAE",  domain=[reg,sec,T])
B_EN   = Parameter(container=m, name="B_EN",   domain=[reg,sec,T])
B_DIV = Parameter(container=m, name="B_DIV", domain=[reg,sec,T])
B_XDD = Parameter(container=m, name="B_XDD", domain=[reg,sec,T])
B_X   = Parameter(container=m, name="B_X",   domain=[reg,sec,T])
B_MAGG= Parameter(container=m, name="B_MAGG",domain=[reg,sec,T])
B_M   = Parameter(container=m, name="B_M",   domain=[reg,sec,regp,T])
B_E   = Parameter(container=m, name="B_E",   domain=[reg,sec,regp,T])
B_PK  = Parameter(container=m, name="B_PK",  domain=[reg,T])
B_PL  = Parameter(container=m, name="B_PL",  domain=[reg,T])
B_IR  = Parameter(container=m, name="B_IR",  domain=[reg,T])
B_Y   = Parameter(container=m, name="B_Y",   domain=[reg,T])
B_S   = Parameter(container=m, name="B_S",   domain=[reg,T])
B_LS  = Parameter(container=m, name="B_LS",  domain=[reg,T])

def build_baseline_recs(offset=0, t_max=None):
    """Build baseline path records with optional time offset for Phase 2.
    Always builds records for ALL T=1..T_MAX. For t > t_max, uses baseline (no offset growth)."""
    if t_max is None:
        t_max = T_MAX
    recs_3d = {n: [] for n in ["XD","VA","K","INV","L","C","I","PD","PVA","PVAE","DIV","XDD","X","MAGG","VAE","EN"]}
    recs_2d = {n: [] for n in ["PK","PL","IR","Y","S","LS"]}
    recs_4d_m = []; recs_4d_e = []
    for t in range(1, T_MAX + 1):
        ts = str(t)
        # Active periods use offset; frozen periods (t > t_max) use simple growth from 0
        if t <= t_max:
            rt = (1+g_val)**(offset + t - 1)
        else:
            rt = (1+g_val)**(t - 1)  # baseline growth without offset
        for r in regions:
            recs_2d["PK"].append((r,ts,1.0)); recs_2d["PL"].append((r,ts,1.0))
            recs_2d["IR"].append((r,ts,ro_val))
            recs_2d["Y"].append((r,ts,cal[(r,sectors[0])]["yz"]*rt))
            recs_2d["S"].append((r,ts,cal[(r,sectors[0])]["sz"]*rt))
            recs_2d["LS"].append((r,ts,cal[(r,sectors[0])]["lsz"]*rt))
            for s in sectors:
                rs = (r,s)
                recs_3d["XD"].append((r,s,ts,d[rs]["xd"]*rt))
                recs_3d["VA"].append((r,s,ts,d[rs]["va"]*rt))
                recs_3d["K"].append((r,s,ts,cal[rs]["kz"]*rt))
                recs_3d["INV"].append((r,s,ts,cal[rs]["inv"]*rt))
                recs_3d["L"].append((r,s,ts,d[rs]["l"]*rt))
                recs_3d["C"].append((r,s,ts,cal[rs]["c_adj"]*rt))
                recs_3d["I"].append((r,s,ts,d[rs]["iz"]*rt))
                recs_3d["PD"].append((r,s,ts,1.0))
                recs_3d["PVA"].append((r,s,ts,cal[rs]["pva"]))
                recs_3d["PVAE"].append((r,s,ts,cal[rs]["pvae"]))
                recs_3d["VAE"].append((r,s,ts,cal[rs]["vae"]*rt))
                recs_3d["EN"].append((r,s,ts,en_data[rs]["en_total"]*rt))
                recs_3d["DIV"].append((r,s,ts,cal[rs]["div"]*rt))
                recs_3d["XDD"].append((r,s,ts,d[rs]["xdd"]*rt))
                recs_3d["X"].append((r,s,ts,d[rs]["x"]*rt))
                recs_3d["MAGG"].append((r,s,ts,d[rs]["m_total"]*rt))
                for rp in [x for x in regions if x != r]:
                    recs_4d_m.append((r,s,rp,ts,trade_m_bil[(r,s,rp)]*rt))
                    recs_4d_e.append((r,s,rp,ts,trade_bil[(r,s,rp)]*rt))
    return recs_3d, recs_2d, recs_4d_m, recs_4d_e

def set_all_baselines(offset=0, t_max=None):
    """Set all B_* parameters from baseline records."""
    if t_max is None:
        t_max = T_MAX
    t_range = list(range(1, t_max + 1))
    recs_3d, recs_2d, recs_4d_m, recs_4d_e = build_baseline_recs(offset, t_max)
    B_XD.setRecords(recs_3d["XD"]); B_VA.setRecords(recs_3d["VA"])
    B_K.setRecords(recs_3d["K"]); B_INV.setRecords(recs_3d["INV"])
    B_L.setRecords(recs_3d["L"]); B_C.setRecords(recs_3d["C"])
    B_I.setRecords(recs_3d["I"]); B_PD.setRecords(recs_3d["PD"])
    B_PVA.setRecords(recs_3d["PVA"]); B_PVAE.setRecords(recs_3d["PVAE"])
    B_VAE.setRecords(recs_3d["VAE"]); B_EN.setRecords(recs_3d["EN"])
    B_DIV.setRecords(recs_3d["DIV"]); B_XDD.setRecords(recs_3d["XDD"])
    B_X.setRecords(recs_3d["X"]); B_MAGG.setRecords(recs_3d["MAGG"])
    B_M.setRecords(recs_4d_m); B_E.setRecords(recs_4d_e)
    B_PK.setRecords(recs_2d["PK"]); B_PL.setRecords(recs_2d["PL"])
    B_IR.setRecords(recs_2d["IR"]); B_Y.setRecords(recs_2d["Y"])
    B_S.setRecords(recs_2d["S"]); B_LS.setRecords(recs_2d["LS"])
    B_VB.setRecords([(r, str(t), vbop_val[r]*(1+g_val)**((offset if t<=t_max else 0)+t-1)) for r in regions for t in range(1, T_MAX+1)])
    B_TAXREV.setRecords([(r, str(t), _taxrev_base[r]*(1+g_val)**((offset if t<=t_max else 0)+t-1)) for r in regions for t in range(1, T_MAX+1)])

# Set initial baselines
set_all_baselines(offset=0)
# =============================================================================
# 7. SHOCK PARAMETERS + SCENARIO FUNCTIONS
# =============================================================================
emis_p = Parameter(container=m, name="emis", domain=[reg,sec])
emis_p.setRecords([(r,s, emis_intensity.get((r,s),0.0)) for r in regions for s in sectors])

# tau_proc: process emissions carbon cost → enters EQXD/EQPVAE_resid proportional to XD
tau_proc = Parameter(container=m, name="tau_proc", domain=[reg,sec,T])
tau_proc.setRecords([(r,s,str(t), 0.0) for r in regions for s in sectors for t in _T])

# tau_fuel: fuel combustion carbon cost → enters EQPE as surcharge on REF price
# Units: per unit of REF input to sector (r,s). Sector-specific because fuel intensity varies.
tau_fuel = Parameter(container=m, name="tau_fuel", domain=[reg,sec,T])
tau_fuel.setRecords([(r,s,str(t), 0.0) for r in regions for s in sectors for t in _T])

# cbam: border adjustment on imports using fixed embodied carbon coefficients.
cbam = Parameter(container=m, name="cbam", domain=[reg,sec,regp,T])
cbam.setRecords([(r,s,rp,str(t), 0.0)
                 for r in regions for s in sectors for rp in regions if rp != r for t in _T])

# Central CBAM approximation used in the paper.  It keeps each foreign origin's
# benchmark-technology embodied intensity and deducts FA times an endogenous
# common EU sector benchmark:
#
#   cbam = pCO2 * max(0, origin_intensity - FA * d_s * EU_benchmark_t)
#
# The EU benchmark is output-weighted across EUC/EUM/EUD and recomputed inside
# every equilibrium/year.  Direct fuel emissions therefore respond to the
# equilibrium REF demand; process and indirect-electricity coefficients remain
# fixed per unit of output.  Here d_s=1 for the four industrial sectors and
# d_ELF=0, so imported fossil electricity receives no free-allocation deduction.
fa_rate_p = Parameter(container=m, name="fa_rate", domain=[T])
fa_rate_p.setRecords([(str(t), 1.0) for t in _T])
cbam_on_p = Parameter(container=m, name="cbam_on", domain=[T])
cbam_on_p.setRecords([(str(t), 0.0) for t in _T])
cbam_eligible_p = Parameter(container=m, name="cbam_eligible", domain=[reg,sec,regp])
cbam_benchmark_deduction_p = Parameter(container=m, name="cbam_benchmark_deduction", domain=[sec])
cbam_benchmark_deduction_p.setRecords([
    (s, 0.0 if s == "ELF" else 1.0) for s in sectors
])
cbam_origin_intensity_p = Parameter(container=m, name="cbam_origin_intensity", domain=[reg,sec])
cbam_origin_intensity_p.setRecords([
    (r, s, cbam_embodied_intensity.get((r, s), 0.0)) for r in regions for s in sectors
])
cbam_proc_intensity_p = Parameter(container=m, name="cbam_proc_intensity", domain=[reg,sec])
cbam_proc_intensity_p.setRecords([
    (r, s, emis_proc_intensity.get((r, s), 0.0) if CBAM_INCLUDE_PROCESS_EMISSIONS else 0.0)
    for r in regions for s in sectors
])
cbam_fuel_per_ref_p = Parameter(container=m, name="cbam_fuel_per_ref", domain=[reg,sec])
cbam_fuel_per_ref_p.setRecords([
    (r, s, emis_fuel_per_ref.get((r, s), 0.0)) for r in regions for s in sectors
])
cbam_indirect_intensity_p = Parameter(container=m, name="cbam_indirect_intensity", domain=[reg,sec])
cbam_indirect_intensity_p.setRecords([
    (r, s, cbam_indirect_elc_intensity.get((r, s), 0.0)) for r in regions for s in sectors
])
base_xd0_p = Parameter(container=m, name="base_xd0", domain=[reg,sec])
base_xd0_p.setRecords([(r, s, d[(r, s)]["xd"]) for r in regions for s in sectors])
base_ref0_p = Parameter(container=m, name="base_ref0", domain=[reg,sec])
base_ref0_p.setRecords([(r, s, en_data[(r, s)]["ref"]) for r in regions for s in sectors])

_eu_cbam_bm0 = {}
for s in sectors:
    den = sum(d[(r, s)]["xd"] for r in EU_REGIONS)
    num = sum(cbam_embodied_intensity.get((r, s), 0.0) * d[(r, s)]["xd"] for r in EU_REGIONS)
    _eu_cbam_bm0[s] = num / den if den > 1e-12 else 0.0

"""
Policy design — Split fuel/process emissions:
  - tau_proc(r,s,t): (1-FA) × P_CO2 × proc_intensity(r,s) — Leontief, unavoidable
    Enters EQXD zero-profit as cost proportional to XD.
  - tau_fuel(r,s,t): (1-FA) × P_CO2 × fuel_per_ref(r,s) — CES, substitutable  
    Enters EQPE as surcharge on REF price. Sectors can substitute REF→ELC to avoid.
  - CBAM uses a benchmark embodied-carbon coefficient for the import origin:
    direct emissions plus fixed benchmark indirect electricity emissions.
  
  EU regions in CBAM_ACTIVE industrial sectors: both reduced-form domestic
  wedges apply once free allocation starts to decline in 2026.
  ELF/ELR: tau_proc=0, tau_fuel=0; the import-side electricity charge is
  handled separately by the CBAM wedge.
"""
# Central coverage is the four represented industrial sectors used in the
# paper: iron and steel, cement, aluminium, and fertilizers. This is not a
# claim that the aggregation reproduces the complete legal product list;
# fossil electricity is added on the import side and hydrogen is not separately
# resolved. CHM remains uncovered. Generic include/exclude environment switches
# are retained for researcher diagnostics but are not part of this round.
CBAM_INDUSTRY = ["STL", "CEM", "ALU", "FER"]
_extra = [s.strip() for s in os.environ.get("RAMSEY_EXTRA_COVERED", "").split(",") if s.strip()]
_excluded = [s.strip() for s in os.environ.get("RAMSEY_EXCLUDE_SECTORS", "").split(",") if s.strip()]
CBAM_ACTIVE = [s for s in (CBAM_INDUSTRY + _extra) if s not in _excluded]
CBAM_IMPORT_SECTORS = CBAM_ACTIVE + ["ELF"]
cbam_eligible_p.setRecords([
    (r, s, rp, 1.0)
    for r in regions for s in sectors for rp in regions
    if rp != r and r in EU_REGIONS and rp not in EU_REGIONS and s in CBAM_IMPORT_SECTORS
])
print(f"CBAM active sectors: {CBAM_ACTIVE}")
print(
    "CBAM basis: fixed origin-sector embodied carbon "
    f"(indirect electricity sectors={CBAM_INDIRECT_SECTORS or ['none']}, "
    f"process={CBAM_INCLUDE_PROCESS_EMISSIONS})"
)
print(
    "CBAM phase-in formula: "
    + ("origin intensity minus eligible industrial EU benchmark" if CBAM_EU_BENCHMARK_ENDOGENOUS
       else "(1-FA) times origin intensity")
)
if CBAM_EU_BENCHMARK_ENDOGENOUS:
    print("Initial output-weighted EU CBAM benchmarks:")
    for s in CBAM_IMPORT_SECTORS:
        print(f"  {s}: {_eu_cbam_bm0.get(s, 0.0):.6f}")

print("Import-weighted CBAM base: old differential vs new embodied coefficient")
for s in CBAM_IMPORT_SECTORS:
    den = 0.0
    old_num = 0.0
    new_num = 0.0
    for r in EU_REGIONS:
        for rp in regions:
            if rp in EU_REGIONS or rp == r:
                continue
            w = trade_m_bil.get((r, s, rp), 0.0)
            den += w
            old_num += w * max(0.0, emis_intensity.get((rp,s), 0.0) - emis_intensity.get((r,s), 0.0))
            new_num += w * cbam_embodied_intensity.get((rp,s), 0.0)
    if den > 1e-10:
        print(f"  {s}: old={old_num/den:.4f}  new={new_num/den:.4f}")

CBAM_START_YEAR = 2026

def _cbam_on_value(calendar_year, fa_rate, with_cbam=True):
    """Return the realized CBAM switch; a zero FA rate still means full CBAM."""
    return float(
        with_cbam
        and calendar_year >= CBAM_START_YEAR
        and fa_rate < 1.0 - 1e-12
    )

def set_shocks(
    fa_func,
    with_cbam=True,
    t_max=None,
    year_offset=0,
    cbam_scale_func=None,
):
    """Set tau_proc, tau_fuel and cbam parameters. Always populates all T=1..T_MAX.
    For t > t_max, shocks are zero. ``year_offset`` maps a chained phase-2
    period back to its calendar year, so the CBAM switch is never inferred from
    the local phase-2 index. ``cbam_scale_func`` optionally supplies a
    deterministic multiplier in [0,1] for perceived-policy experiments.  Core
    scenarios retain the binary legislated on/off rule."""
    if t_max is None:
        t_max = T_MAX
    tau_proc_recs = []; tau_fuel_recs = []; cbam_recs = []
    fa_recs = []; cbam_on_recs = []
    fa_by_t = {}; cbam_on_by_t = {}
    for t in range(1, T_MAX + 1):
        fa = fa_func(t) if t <= t_max else 1.0
        calendar_year = BASE_YEAR + year_offset + t - 1
        cbam_on = _cbam_on_value(calendar_year, fa, with_cbam and t <= t_max)
        if cbam_on and cbam_scale_func is not None:
            cbam_on *= float(cbam_scale_func(t))
        if not -1e-12 <= cbam_on <= 1.0 + 1e-12:
            raise ValueError(f"CBAM scale must lie in [0,1], got {cbam_on} at t={t}.")
        cbam_on = min(1.0, max(0.0, cbam_on))
        fa_by_t[t] = fa
        cbam_on_by_t[t] = cbam_on
        fa_recs.append((str(t), fa))
        cbam_on_recs.append((str(t), cbam_on))
    for r in regions:
        for s in sectors:
            for t in range(1, T_MAX + 1):
                if t <= t_max and r in EU_REGIONS and s in CBAM_ACTIVE:
                    # Deliberately retain the paper's domestic simplification:
                    # free allocation reduces the marginal carbon wedge by 1-FA.
                    cf = 1.0 - fa_by_t[t]
                    tp = cf * CARBON_PRICE * emis_proc_intensity.get((r,s), 0.0)
                    tf = cf * CARBON_PRICE * emis_fuel_per_ref.get((r,s), 0.0)
                else:
                    tp = 0.0; tf = 0.0
                tau_proc_recs.append((r, s, str(t), tp))
                tau_fuel_recs.append((r, s, str(t), tf))
                for rp in regions:
                    if rp == r: continue
                    if cbam_on_by_t[t] and r in EU_REGIONS and rp not in EU_REGIONS and s in CBAM_IMPORT_SECTORS:
                        cf = 1.0 - fa_by_t[t]
                        val = cf * CARBON_PRICE * cbam_embodied_intensity.get((rp,s), 0.0)
                    else:
                        val = 0.0
                    cbam_recs.append((r, s, rp, str(t), val))
    tau_proc.setRecords(tau_proc_recs)
    tau_fuel.setRecords(tau_fuel_recs)
    cbam.setRecords(cbam_recs)
    fa_rate_p.setRecords(fa_recs)
    cbam_on_p.setRecords(cbam_on_recs)

def set_shocks_zero():
    set_shocks(lambda t: 1.0)

# Deterministic perceived-stringency/credibility index.  This is not a
# probability and the model has no stochastic branches.  Each kappa value
# defines one deterministic phase-1 policy path after T*:
#
#   perceived_FA = kappa * legislated_FA + (1-kappa) * no_policy_FA
#
# kappa=0 reproduces full disbelief/anticipated withdrawal; kappa=1 reproduces
# the fully perceived legislated schedule.  The former RAMSEY_CREDIBILITY_P is
# accepted as a deprecated environment alias when the new name is absent.
_credibility_index_env = os.environ.get("RAMSEY_CREDIBILITY_INDEX")
_credibility_legacy_env = os.environ.get("RAMSEY_CREDIBILITY_P")
if _credibility_index_env is not None and _credibility_index_env != "":
    CREDIBILITY_INDEX = float(_credibility_index_env)
    CREDIBILITY_INDEX_FROM_LEGACY_ALIAS = False
    if _credibility_legacy_env not in {None, ""}:
        _legacy_index_value = float(_credibility_legacy_env)
        if abs(_legacy_index_value - CREDIBILITY_INDEX) > 1e-12:
            print(
                "WARNING: RAMSEY_CREDIBILITY_INDEX overrides conflicting "
                "RAMSEY_CREDIBILITY_P."
            )
else:
    CREDIBILITY_INDEX = env_float("RAMSEY_CREDIBILITY_P", 0.0)
    CREDIBILITY_INDEX_FROM_LEGACY_ALIAS = _credibility_legacy_env not in {None, ""}
if not 0.0 <= CREDIBILITY_INDEX <= 1.0:
    raise ValueError(
        f"RAMSEY_CREDIBILITY_INDEX must lie in [0,1], got {CREDIBILITY_INDEX}."
    )
print(
    f"Deterministic credibility index: kappa={CREDIBILITY_INDEX:.6g} "
    + ("(legacy RAMSEY_CREDIBILITY_P alias)" if CREDIBILITY_INDEX_FROM_LEGACY_ALIAS
       else "(RAMSEY_CREDIBILITY_INDEX/default)")
)

def fa_credible(t): return get_fa(t)
def fa_no_policy(t): return 1.0
def fa_reneging_phase2(t): return 1.0
def fa_perceived_policy_phase1(t):
    if t < T_STAR:
        return get_fa(t)
    return CREDIBILITY_INDEX * get_fa(t) + (1.0 - CREDIBILITY_INDEX) * 1.0
def cbam_scale_perceived_policy_phase1(t):
    """Continuous deterministic CBAM stringency for the kappa sweep.

    Before revelation, firms observe the legislated policy.  From T* onward,
    kappa scales the perceived CBAM charge together with the interpolated
    domestic/free-allocation path.  This avoids a discontinuous jump from no
    CBAM at kappa=0 to a full border mechanism for arbitrarily small kappa.
    """
    return 1.0 if t < T_STAR else CREDIBILITY_INDEX
def fa_legislated_phase2(t): return get_fa(t + T_STAR - 1)

def fa_reneging_full_path(t):
    """Realized full-horizon path: legislated through T*-1, withdrawn at T*."""
    return get_fa(t) if t < T_STAR else 1.0

def fa_anticipated_withdrawal_full_path(t):
    """Same realized policy as reneging, but anticipated in phase 1."""
    return get_fa(t) if t < T_STAR else 1.0

ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE = abs(CREDIBILITY_INDEX) <= 1e-12
if ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE:
    print("Scenario alias active: AnticipatedWithdrawal -> Eq4_Phase1")
    assert all(
        abs(fa_anticipated_withdrawal_full_path(t) - fa_reneging_full_path(t)) <= 1e-12
        for t in range(1, T_MAX + 1)
    )

# Fail early if a future schedule edit violates the legislated CBAM timing.
for _t in range(1, min(T_MAX, CBAM_START_YEAR - BASE_YEAR) + 1):
    _year = BASE_YEAR + _t - 1
    assert _cbam_on_value(_year, get_fa(_t), True) == 0.0
_cbam_start_t = CBAM_START_YEAR - BASE_YEAR + 1
if T_MAX >= _cbam_start_t:
    assert _cbam_on_value(CBAM_START_YEAR, get_fa(_cbam_start_t), True) == 1.0
for _t in range(1, T_MAX + 1):
    _fa = get_fa(_t)
    _on = _cbam_on_value(BASE_YEAR + _t - 1, _fa, True)
    if abs(_fa) <= 1e-12:
        # This is the exact multiplier on the EU benchmark deduction.
        assert abs(_on * _fa) <= 1e-12

# =============================================================================
# 8. VARIABLES
# =============================================================================
H_PD    = Variable(container=m, name="H_PD",    domain=[reg,sec,T])
H_PVA   = Variable(container=m, name="H_PVA",   domain=[reg,sec,T])
H_PVAE  = Variable(container=m, name="H_PVAE",  domain=[reg,sec,T])
H_VAE   = Variable(container=m, name="H_VAE",   domain=[reg,sec,T])
H_EN    = Variable(container=m, name="H_EN",    domain=[reg,sec,T])
H_PE    = Variable(container=m, name="H_PE",    domain=[reg,sec,T])
H_PELC  = Variable(container=m, name="H_PELC",  domain=[reg,sec,T])  # ELC composite price
H_ELC   = Variable(container=m, name="H_ELC",   domain=[reg,sec,T])  # ELC composite quantity
H_P     = Variable(container=m, name="H_P",     domain=[reg,sec,T])
H_PMAGG = Variable(container=m, name="H_PMAGG", domain=[reg,sec,T])
H_PC    = Variable(container=m, name="H_PC",    domain=[reg,T])
H_PL    = Variable(container=m, name="H_PL",    domain=[reg,T])
H_PK    = Variable(container=m, name="H_PK",    domain=[reg,T])
H_XD    = Variable(container=m, name="H_XD",    domain=[reg,sec,T])
H_VA    = Variable(container=m, name="H_VA",    domain=[reg,sec,T])
H_XDD   = Variable(container=m, name="H_XDD",   domain=[reg,sec,T])
H_X     = Variable(container=m, name="H_X",     domain=[reg,sec,T])
H_MAGG  = Variable(container=m, name="H_MAGG",  domain=[reg,sec,T])
H_M     = Variable(container=m, name="H_M",     domain=[reg,sec,regp,T])
H_L     = Variable(container=m, name="H_L",     domain=[reg,sec,T])
H_C     = Variable(container=m, name="H_C",     domain=[reg,sec,T])
H_I     = Variable(container=m, name="H_I",     domain=[reg,sec,T])
H_Y     = Variable(container=m, name="H_Y",     domain=[reg,T])
H_S     = Variable(container=m, name="H_S",     domain=[reg,T])
H_DIV   = Variable(container=m, name="H_DIV",   domain=[reg,sec,T])
KP   = Variable(container=m, name="KP",   domain=[reg,sec,T])
INVP = Variable(container=m, name="INVP", domain=[reg,sec,T])
IRP  = Variable(container=m, name="IRP",  domain=[reg,T])
EU_CBAM_BM = Variable(container=m, name="EU_CBAM_BM", domain=[sec,T])

# =============================================================================
# 9. ECUACIONES
# =============================================================================

# Endogenous EU benchmark for the paper's central CBAM adjustment.  Growth
# cancels between the numerator and denominator, so benchmark-year levels can be
# combined with hats in both the full-horizon and chained phase-2 solves.
EQEU_CBAM_BM = Equation(container=m, name="EQEU_CBAM_BM", domain=[sec,T])
EQEU_CBAM_BM[sec,T].where[TACTIVE[T]] = (
    EU_CBAM_BM[sec,T] * Sum(reg.where[EUREG[reg]], base_xd0_p[reg,sec] * H_XD[reg,sec,T])
    == Sum(reg.where[EUREG[reg]],
        cbam_proc_intensity_p[reg,sec] * base_xd0_p[reg,sec] * H_XD[reg,sec,T]
        + cbam_fuel_per_ref_p[reg,sec] * base_ref0_p[reg,sec]
          * H_EN[reg,sec,T]
          * (H_PE[reg,sec,T] / (H_P[reg,"REF",T] + tau_fuel[reg,sec,T]))**sigmaEN_p
        + cbam_indirect_intensity_p[reg,sec] * base_xd0_p[reg,sec] * H_XD[reg,sec,T]
    )
)

def _cbam_wedge(r, s, rp, t):
    if not CBAM_EU_BENCHMARK_ENDOGENOUS:
        return cbam[r,s,rp,t]
    gap = (
        cbam_origin_intensity_p[rp,s]
        - fa_rate_p[t] * cbam_benchmark_deduction_p[s] * EU_CBAM_BM[s,t]
    )
    # Smooth positive part: avoids a negative border subsidy for origins whose
    # intensity lies below the EU benchmark while keeping the MCP differentiable.
    positive_gap = 0.5 * (gap + _gsqrt(gap**2 + CBAM_FLOOR_EPS**2))
    return cbam_eligible_p[r,s,rp] * cbam_on_p[t] * CARBON_PRICE * positive_gap

# --- A) PRODUCTION with energy nesting ---
# EQVA: VA = K^αF × L^(1-αF) — determines H_VA
EQVA = Equation(container=m, name="EQVA", domain=[reg,sec,T])
EQVA[reg,sec,T].where[TACTIVE[T]] = H_VA[reg,sec,T] == (
    (KP[reg,sec,T]/B_K[reg,sec,T])**alphaF_p[reg,sec]
    * H_L[reg,sec,T]**(1-alphaF_p[reg,sec])
)

# EQPELC: CES price of electricity composite — determines H_PELC
# H_PELC^(1-σ_ELC) = θ_ff × H_P[ELF]^(1-σ_ELC) + θ_rn × H_P[ELR]^(1-σ_ELC)
EQPELC = Equation(container=m, name="EQPELC", domain=[reg,sec,T])
EQPELC[reg,sec,T].where[TACTIVE[T]] = H_PELC[reg,sec,T]**(1-sigmaELC_p) == (
    theta_ff_p[reg,sec] * H_P[reg,"ELF",T]**(1-sigmaELC_p)
    + theta_rn_p[reg,sec] * H_P[reg,"ELR",T]**(1-sigmaELC_p)
)

# EQPE: CES price of energy bundle — determines H_PE
# H_PE^(1-σ_EN) = θ_elc × H_PELC^(1-σ_EN) + θ_ref × (H_P[REF] + tau_fuel)^(1-σ_EN)
# tau_fuel enters as surcharge on REF price — sectors can substitute REF→ELC to avoid
EQPE = Equation(container=m, name="EQPE", domain=[reg,sec,T])
EQPE[reg,sec,T].where[TACTIVE[T]] = H_PE[reg,sec,T]**(1-sigmaEN_p) == (
    theta_elc_p[reg,sec] * H_PELC[reg,sec,T]**(1-sigmaEN_p)
    + theta_ref_p[reg,sec] * (H_P[reg,"REF",T] + tau_fuel[reg,sec,T])**(1-sigmaEN_p)
)

# EQPVAE: CES price of VAE bundle — determines H_PVA (given H_PVAE from residual, H_PE from above)
# H_PVAE^(1-σ) = sVA_VAE × H_PVA^(1-σ) + sEN_VAE × H_PE^(1-σ)
EQPVAE = Equation(container=m, name="EQPVAE", domain=[reg,sec,T])
EQPVAE[reg,sec,T].where[TACTIVE[T]] = H_PVAE[reg,sec,T]**(1-sigmaVAE_p) == (
    sVA_VAE_p[reg,sec] * H_PVA[reg,sec,T]**(1-sigmaVAE_p)
    + sEN_VAE_p[reg,sec] * H_PE[reg,sec,T]**(1-sigmaVAE_p)
)

# EQPVAE_resid: PVAE from zero-profit residual — determines H_PVAE
# Process emissions cost (tau_proc) enters here proportional to XD (Leontief)
EQPVAE_resid = Equation(container=m, name="EQPVAE_resid", domain=[reg,sec,T])
EQPVAE_resid[reg,sec,T].where[TACTIVE[T]] = (
    H_PVAE[reg,sec,T]*B_PVAE[reg,sec,T] * vaeShare_p[reg,sec]
    == H_PD[reg,sec,T] - Sum(secc, H_P[reg,secc,T]*io_mat_p[reg,secc,sec])
     - taxShare_p[reg,sec]
     - tau_proc[reg,sec,T]
)

# EQVAE_VA: CES demand for VA within VAE — determines H_VAE (inverted)
# H_VAE = H_VA × (H_PVA×B_PVA / (H_PVAE×B_PVAE))^σ_VAE
EQVAE_VA = Equation(container=m, name="EQVAE_VA", domain=[reg,sec,T])
EQVAE_VA[reg,sec,T].where[TACTIVE[T]] = H_VAE[reg,sec,T] == (
    H_VA[reg,sec,T]
    * (H_PVA[reg,sec,T]*B_PVA[reg,sec,T] / (H_PVAE[reg,sec,T]*B_PVAE[reg,sec,T]))**sigmaVAE_p
)

# EQVAE_EN: CES demand for EN within VAE — determines H_EN
# H_EN = H_VAE × (H_PVAE×B_PVAE / H_PE)^σ_VAE
EQVAE_EN = Equation(container=m, name="EQVAE_EN", domain=[reg,sec,T])
EQVAE_EN[reg,sec,T].where[TACTIVE[T]] = H_EN[reg,sec,T] == (
    H_VAE[reg,sec,T]
    * (H_PVAE[reg,sec,T]*B_PVAE[reg,sec,T] / H_PE[reg,sec,T])**sigmaVAE_p
)

# EQELC_DEMAND: CES demand for ELC composite within EN bundle — determines H_ELC
# H_ELC = H_EN × (H_PE / H_PELC)^σ_EN
EQELC_DEMAND = Equation(container=m, name="EQELC_DEMAND", domain=[reg,sec,T])
EQELC_DEMAND[reg,sec,T].where[TACTIVE[T]] = H_ELC[reg,sec,T] == (
    H_EN[reg,sec,T]
    * (H_PE[reg,sec,T] / H_PELC[reg,sec,T])**sigmaEN_p
)

# EQXD: Zero profit (primal) — determines H_XD
# PD×XD = XD×Σ(io_mat×P) + pvaeVaeShare×PVAE×VAE + taxShare×XD + tau_proc×XD
# Note: tau_fuel enters EQPE (surcharge on REF), not here.
EQXD = Equation(container=m, name="EQXD", domain=[reg,sec,T])
EQXD[reg,sec,T].where[TACTIVE[T]] = (
    H_PD[reg,sec,T] * H_XD[reg,sec,T]
    == H_XD[reg,sec,T] * Sum(secc, io_mat_p[reg,secc,sec]*H_P[reg,secc,T])
     + pvaeVaeShare_p[reg,sec] * H_PVAE[reg,sec,T] * H_VAE[reg,sec,T]
     + taxShare_p[reg,sec] * H_XD[reg,sec,T]
     + tau_proc[reg,sec,T] * H_XD[reg,sec,T]
)

# --- B) ARMINGTON ---
sXDD_p = Parameter(container=m, name="sXDD", domain=[reg,sec])
sXDD_p.setRecords([(r,s, d[(r,s)]["xdd"]/d[(r,s)]["x"] if d[(r,s)]["x"] > 1e-10 else 1.0)
                    for r in regions for s in sectors])
sMAGG_p = Parameter(container=m, name="sMAGG", domain=[reg,sec])
sMAGG_p.setRecords([(r,s, d[(r,s)]["m_total"]/d[(r,s)]["x"] if d[(r,s)]["x"] > 1e-10 else 0.0)
                     for r in regions for s in sectors])

EQARMP = Equation(container=m, name="EQARMP", domain=[reg,sec,T])
EQARMP[reg,sec,T].where[TACTIVE[T]] = (
    H_P[reg,sec,T] * H_X[reg,sec,T]
    == H_PD[reg,sec,T] * H_XDD[reg,sec,T] * sXDD_p[reg,sec]
     + H_PMAGG[reg,sec,T] * H_MAGG[reg,sec,T] * sMAGG_p[reg,sec]
)

EQIMPORT_TOP = Equation(container=m, name="EQIMPORT_TOP", domain=[reg,sec,T])
EQIMPORT_TOP[reg,sec,T].where[TACTIVE[T]] = H_MAGG[reg,sec,T] == (
    (H_P[reg,sec,T] / H_PMAGG[reg,sec,T])**sigmaA_p[sec] * H_X[reg,sec,T]
)

EQARMD = Equation(container=m, name="EQARMD", domain=[reg,sec,T])
EQARMD[reg,sec,T].where[TACTIVE[T]] = H_XDD[reg,sec,T] == (
    (H_P[reg,sec,T] / H_PD[reg,sec,T])**sigmaA_p[sec] * H_X[reg,sec,T]
)

EQPMAGG = Equation(container=m, name="EQPMAGG", domain=[reg,sec,T])
EQPMAGG[reg,sec,T].where[TACTIVE[T]] = H_PMAGG[reg,sec,T]**(1-sigmaM_p[sec]) == (
    Sum(regp.where[rr[reg,regp]],
        gammaM_p[reg,sec,regp] * (H_PD[regp,sec,T] + _cbam_wedge(reg,sec,regp,T))**(1-sigmaM_p[sec]))
)

EQIMPORT_BOT = Equation(container=m, name="EQIMPORT_BOT", domain=[reg,sec,regp,T])
EQIMPORT_BOT[reg,sec,regp,T].where[rr[reg,regp] & TACTIVE[T]] = H_M[reg,sec,regp,T] == (
    (H_PMAGG[reg,sec,T] / (H_PD[regp,sec,T] + _cbam_wedge(reg,sec,regp,T)))**sigmaM_p[sec]
    * H_MAGG[reg,sec,T]
)

# EQXDD
sXDD_XD_p = Parameter(container=m, name="sXDD_XD", domain=[reg,sec])
sXDD_XD_p.setRecords([(r,s, d[(r,s)]["xdd"]/d[(r,s)]["xd"]) for r in regions for s in sectors])
sE_p = Parameter(container=m, name="sE", domain=[reg,sec,regp])
sE_p.setRecords([(r,s,rp, trade_bil[(r,s,rp)]/d[(r,s)]["xd"] if d[(r,s)]["xd"]>1e-10 else 0.0)
                  for r in regions for s in sectors for rp in regions if rp != r])

EQXDD_eq = Equation(container=m, name="EQXDD", domain=[reg,sec,T])
EQXDD_eq[reg,sec,T].where[TACTIVE[T]] = (
    H_XD[reg,sec,T] - phi_p[reg,sec]*INVP[reg,sec,T]**2/(KP[reg,sec,T]*B_XD[reg,sec,T])
    == H_XDD[reg,sec,T]*sXDD_XD_p[reg,sec]
     + Sum(regp.where[rr[reg,regp]], H_M[regp,sec,reg,T]*sE_p[reg,sec,regp])
)

# --- C) FIRMA ---
# Helper expressions for the (optional) scrapping asymmetry. z = xbar - I/K;
# pos(z) = 0.5*(z + sqrt(z^2 + eps^2)); pos'(z) = 0.5*(1 + z/sqrt(z^2 + eps^2)).
# Marginal effects:  dAC-/dI = -2*phi_s*pD*pos*pos'   (enters q with minus)
#                    dAC-/dK = +phi_s*pD*(pos^2 + 2*pos*pos'*I/K)  (flow FOC)
def _scrap_z(Tx):
    return xbar_p[reg,sec] - INVP[reg,sec,Tx]/KP[reg,sec,Tx]

def _scrap_pos(Tx):
    z = _scrap_z(Tx)
    return 0.5*(z + _gsqrt(z**2 + POS_EPS2))

def _scrap_posp(Tx):
    z = _scrap_z(Tx)
    return 0.5*(1 + z/_gsqrt(z**2 + POS_EPS2))

def _q_expr(Tx):
    """Shadow value of installed capital, q = pK + dAC/dI at Tx."""
    q = H_PK[reg,Tx] + 2*phi_p[reg,sec]*H_PD[reg,sec,Tx]*INVP[reg,sec,Tx]/KP[reg,sec,Tx]
    if SCRAP_MULT > 0:
        q = q - 2*SCRAP_MULT*phi_p[reg,sec]*H_PD[reg,sec,Tx]*_scrap_pos(Tx)*_scrap_posp(Tx)
    return q

def _scrap_flow(Tx):
    """+dAC-/dK at Tx; subtracted in EQLAMBDA so the FOC carries -dAC-/dK."""
    if SCRAP_MULT <= 0:
        return 0.0
    return SCRAP_MULT*phi_p[reg,sec]*H_PD[reg,sec,Tx]*(
        _scrap_pos(Tx)**2
        + 2*_scrap_pos(Tx)*_scrap_posp(Tx)*INVP[reg,sec,Tx]/KP[reg,sec,Tx]
    )

EQLAMBDA = Equation(container=m, name="EQLAMBDA", domain=[reg,sec,T])
EQLAMBDA[reg,sec,T].where[~TFIRST[T] & TACTIVE[T]] = (
    alphaF_p[reg,sec]*H_PVA[reg,sec,T.lag(1)]*B_PVA[reg,sec,T.lag(1)]*H_VA[reg,sec,T.lag(1)]*B_VA[reg,sec,T.lag(1)]/KP[reg,sec,T.lag(1)]
    + phi_p[reg,sec]*H_PD[reg,sec,T.lag(1)]*(INVP[reg,sec,T.lag(1)]/KP[reg,sec,T.lag(1)])**2
    - _scrap_flow(T.lag(1))
    - (1+IRP[reg,T.lag(1)])*_q_expr(T.lag(1))
    + (1-delta_p[reg,sec])*_q_expr(T)
    == 0
)

EQK = Equation(container=m, name="EQK", domain=[reg,sec,T])
EQK[reg,sec,T].where[~TFIRST[T] & TACTIVE[T]] = (
    KP[reg,sec,T] == (1-delta_p[reg,sec])*KP[reg,sec,T.lag(1)] + INVP[reg,sec,T.lag(1)]
)

# --- D) CONSUMIDOR (CIES utility, gamma=risk aversion) ---
# Cobb-Douglas consumption price index. Intertemporal choice must smooth real
# consumption expenditure, not nominal expenditure.
EQPC = Equation(container=m, name="EQPC", domain=[reg,T])
EQPC[reg,T].where[TACTIVE[T]] = H_PC[reg,T] == Product(sec, H_P[reg,sec,T]**alphaH_p[reg,sec])

# Euler on real consumption ratio (includes BGP growth in B_Y, B_S).
# IRP(T-1) is the numeraire-denominated return earned between T-1 and T,
# matching EQLAMBDA (q and MPK are in numeraire units).  Real consumption is
# deflated by H_PC, so the return is converted into consumption units with
# the factor H_PC(T-1)/H_PC(T).  On the benchmark BGP H_PC is constant, the
# factor equals 1, and calibration and EQTC2 (IRP[TLAST]=ro) are unaffected.
# (real ratio)^gamma = (1+IR_{T-1}) * (PC_{T-1}/PC_T) * (1+g)^gamma / (1+ro)
EQEULER = Equation(container=m, name="EQEULER", domain=[reg,T])
EQEULER[reg,T].where[~TFIRST[T] & TACTIVE[T]] = (
    (((H_Y[reg,T]*B_Y[reg,T] - H_S[reg,T]*B_S[reg,T]) / H_PC[reg,T])
    / ((H_Y[reg,T.lag(1)]*B_Y[reg,T.lag(1)] - H_S[reg,T.lag(1)]*B_S[reg,T.lag(1)])
       / H_PC[reg,T.lag(1)])
    )**gamma_p
    == (1+IRP[reg,T.lag(1)])*(H_PC[reg,T.lag(1)]/H_PC[reg,T])*(1+g_p)**gamma_p/(1+ro_p)
)

# --- E) TERMINALES ---
EQTC = Equation(container=m, name="EQTC", domain=[reg,sec,T])
EQTC[reg,sec,TLAST] = INVP[reg,sec,TLAST] == (g_p+delta_p[reg,sec])*KP[reg,sec,TLAST]

EQTC2 = Equation(container=m, name="EQTC2", domain=[reg,T])
EQTC2[reg,TLAST] = IRP[reg,TLAST] == ro_p

# --- F) INTRATEMPORAL ---
EQFSTORL = Equation(container=m, name="EQFSTORL", domain=[reg,sec,T])
EQFSTORL[reg,sec,T].where[TACTIVE[T]] = (
    H_PL[reg,T] * H_L[reg,sec,T]
    == H_PVA[reg,sec,T]*B_PVA[reg,sec,T]*H_VA[reg,sec,T]
)

EQINV = Equation(container=m, name="EQINV", domain=[reg,sec,T])
EQINV[reg,sec,T].where[TACTIVE[T]] = (
    H_P[reg,sec,T] * H_I[reg,sec,T]*B_I[reg,sec,T]
    == alphaI_p[reg,sec] * Sum(secc, H_PK[reg,T]*INVP[reg,secc,T])
)

EQPK = Equation(container=m, name="EQPK", domain=[reg,T])
EQPK[reg,T].where[TACTIVE[T]] = H_PK[reg,T] == Sum(sec, alphaI_p[reg,sec] * H_P[reg,sec,T])

EQC = Equation(container=m, name="EQC", domain=[reg,sec,T])
EQC[reg,sec,T].where[TACTIVE[T]] = (
    H_P[reg,sec,T] * H_C[reg,sec,T]*B_C[reg,sec,T]
    == alphaH_p[reg,sec] * (H_Y[reg,T]*B_Y[reg,T] - H_S[reg,T]*B_S[reg,T])
)

# EQY — Bröcker-style: Y = labor + capital_income - adj_costs - VBOP + TAXREV + carbon_rev
# Capital income per sector in hat form: alphaF*H_PVA*H_VA (= H_PD*H_XD in baseline since pva*va=va)
# We use ky_share = ky/yz as the share parameter
sKY_Y_p = Parameter(container=m, name="sKY_Y", domain=[reg,sec])
sKY_Y_p.setRecords([(r,s, cal[(r,s)]["ky_share"]) for r in regions for s in sectors])
sLS_p = Parameter(container=m, name="sLS", domain=[reg])
sLS_p.setRecords([(r, cal[(r,sectors[0])]["lsz"]/cal[(r,sectors[0])]["yz"]) for r in regions])
sVB_p = Parameter(container=m, name="sVB", domain=[reg])
sVB_p.setRecords([(r, vbop_val[r]/cal[(r,sectors[0])]["yz"]) for r in regions])
sTAXREV_p = Parameter(container=m, name="sTAXREV", domain=[reg])
sTAXREV_p.setRecords([(r, _taxrev_base[r]/cal[(r,sectors[0])]["yz"]) for r in regions])
# Adjustment cost share: adj(s)/yz
sADJ_Y_p = Parameter(container=m, name="sADJ_Y", domain=[reg,sec])
sADJ_Y_p.setRecords([(r,s, cal[(r,s)]["adj"]/cal[(r,s)]["yz"]) for r in regions for s in sectors])
sXD_Y_p = Parameter(container=m, name="sXD_Y", domain=[reg,sec])
sXD_Y_p.setRecords([(r,s, d[(r,s)]["xd"]/cal[(r,s)]["yz"]) for r in regions for s in sectors])
sM_Y_p = Parameter(container=m, name="sM_Y", domain=[reg,sec,regp])
sM_Y_p.setRecords([(r,s,rp, trade_m_bil[(r,s,rp)]/cal[(r,s)]["yz"])
                    for r in regions for s in sectors for rp in regions if rp != r])

# tau_fuel revenue share: tau_fuel × ref_input / yz (benchmark units)
sREF_Y_p = Parameter(container=m, name="sREF_Y", domain=[reg,sec])
sREF_Y_p.setRecords([(r,s, en_data[(r,s)]["ref"]/cal[(r,s)]["yz"]) for r in regions for s in sectors])

EQY = Equation(container=m, name="EQY", domain=[reg,T])
EQY[reg,T].where[TACTIVE[T]] = H_Y[reg,T] == (
    # Capital income: sKY * hat(KY). hat(KY) = H_PVA * H_VA (since pva*va=va, ky=alphaF*va)
    Sum(sec, sKY_Y_p[reg,sec] * H_PVA[reg,sec,T] * H_VA[reg,sec,T])
    # Labor income
    + H_PL[reg,T]*sLS_p[reg]
    # Adjustment costs (hat form): adj_hat = H_PD * (INVP/B_INV)^2 / (KP/B_K)
    - Sum(sec, sADJ_Y_p[reg,sec] * H_PD[reg,sec,T]
          * (INVP[reg,sec,T]/B_INV[reg,sec,T])**2
          / (KP[reg,sec,T]/B_K[reg,sec,T]))
    # Scrapping asymmetry resource cost (zero when SCRAP_MULT=0):
    # AC-/yz_path = scrap_mult*phi * (kz/yz) * H_PD * (KP/B_K) * pos(xbar-I/K)^2
    - (Sum(sec, SCRAP_MULT*phi_p[reg,sec]*sKZ_Y_p[reg,sec]*H_PD[reg,sec,T]
            * (KP[reg,sec,T]/B_K[reg,sec,T])
            * (0.5*(xbar_p[reg,sec] - INVP[reg,sec,T]/KP[reg,sec,T]
                    + _gsqrt((xbar_p[reg,sec] - INVP[reg,sec,T]/KP[reg,sec,T])**2 + POS_EPS2)))**2)
       if SCRAP_MULT > 0 else 0.0)
    # BOP transfer
    - H_PK[NUMERAIRE,T]*sVB_p[reg]
    # Benchmark TAX-row revenue. With USE_BASE_TAX_WEDGES=True the SAM TAX row
    # enters production as a benchmark wedge and is returned to households.
    + sTAXREV_p[reg]
    # Carbon revenue: process emissions
    + Sum(sec, tau_proc[reg,sec,T] * H_XD[reg,sec,T]*sXD_Y_p[reg,sec])
    # Carbon revenue: fuel emissions
    + Sum(sec, tau_fuel[reg,sec,T]
          * H_EN[reg,sec,T]
          * (H_PE[reg,sec,T] / (H_P[reg,"REF",T] + tau_fuel[reg,sec,T]))**sigmaEN_p
          * sREF_Y_p[reg,sec])
    # CBAM revenue
    + Sum(sec, Sum(regp.where[rr[reg,regp]], _cbam_wedge(reg,sec,regp,T) * H_M[reg,sec,regp,T]*sM_Y_p[reg,sec,regp]))
)

# EQDIV — ELIMINATED (no dividends in Bröcker-style model)
# H_DIV is kept as a variable but not determined by an equation.
# It will be removed from the model equations list.

# EQXM — Market clearing with three-level energy nesting
# Material goods: Leontief demand from XD
# Energy goods: CES demand from energy bundles
#   REF demand from sector sj = (PE(sj)/P[REF])^σ_EN × EN(sj)
#   ELF demand from sector sj = (PELC(sj)/P[ELF])^σ_ELC × ELC(sj)
#   ELR demand from sector sj = (PELC(sj)/P[ELR])^σ_ELC × ELC(sj)

sIO_X_mat_p = Parameter(container=m, name="sIO_X_mat", domain=[reg,sec,secc])
sIO_X_mat_recs = []
for r in regions:
    for si in sectors:
        for sj in sectors:
            if si in ENERGY_SECS:
                sIO_X_mat_recs.append((r, si, sj, 0.0))
            else:
                val = io_val[(r,si,sj)]*d[(r,sj)]["xd"]/d[(r,si)]["x"] if d[(r,si)]["x"] > 1e-10 else 0.0
                sIO_X_mat_recs.append((r, si, sj, val))
sIO_X_mat_p.setRecords(sIO_X_mat_recs)

# REF demand shares: ref_input(sj) / X(REF)
sREF_X_p = Parameter(container=m, name="sREF_X", domain=[reg,secc])
sREF_X_p.setRecords([(r, sj, en_data[(r,sj)]["ref"] / d[(r,"REF")]["x"] if d[(r,"REF")]["x"] > 1e-10 else 0.0)
                      for r in regions for sj in sectors])

# ELF demand shares: elc_ff_input(sj) / X(ELF)
sELCFF_X_p = Parameter(container=m, name="sELCFF_X", domain=[reg,secc])
sELCFF_X_p.setRecords([(r, sj, en_data[(r,sj)]["elc_ff"] / d[(r,"ELF")]["x"] if d[(r,"ELF")]["x"] > 1e-10 else 0.0)
                        for r in regions for sj in sectors])

# ELR demand shares: elc_rn_input(sj) / X(ELR)
sELCRN_X_p = Parameter(container=m, name="sELCRN_X", domain=[reg,secc])
sELCRN_X_p.setRecords([(r, sj, en_data[(r,sj)]["elc_rn"] / d[(r,"ELR")]["x"] if d[(r,"ELR")]["x"] > 1e-10 else 0.0)
                        for r in regions for sj in sectors])

sC_X_p = Parameter(container=m, name="sC_X", domain=[reg,sec])
sC_X_p.setRecords([(r,s, cal[(r,s)]["c_adj"]/d[(r,s)]["x"] if d[(r,s)]["x"] > 1e-10 else 0.0)
                    for r in regions for s in sectors])
sI_X_p = Parameter(container=m, name="sI_X", domain=[reg,sec])
sI_X_p.setRecords([(r,s, d[(r,s)]["iz"]/d[(r,s)]["x"] if d[(r,s)]["x"] > 1e-10 else 0.0)
                    for r in regions for s in sectors])

# Market clearing: separate treatment per good type
# For material goods (MAT_SECS): intermediate = Leontief from XD
# For REF: intermediate = CES demand from EN bundles (PE/P_REF)^σ_EN × EN
# For ELF: intermediate = CES demand from ELC bundles (PELC/P_ELF)^σ_ELC × ELC
# For ELR: intermediate = CES demand from ELC bundles (PELC/P_ELR)^σ_ELC × ELC

# We need a conditional EQXM. Simplest: write one equation, use sIO_X_mat (0 for energy),
# sREF_X (0 for non-REF), sELCFF_X (0 for non-ELF), sELCRN_X (0 for non-ELR)

# Indicator parameters for sector type
is_ref_p = Parameter(container=m, name="is_ref", domain=[sec])
is_ref_p.setRecords([(s, 1.0 if s == "REF" else 0.0) for s in sectors])
is_elcff_p = Parameter(container=m, name="is_elcff", domain=[sec])
is_elcff_p.setRecords([(s, 1.0 if s == "ELF" else 0.0) for s in sectors])
is_elcrn_p = Parameter(container=m, name="is_elcrn", domain=[sec])
is_elcrn_p.setRecords([(s, 1.0 if s == "ELR" else 0.0) for s in sectors])

EQXM = Equation(container=m, name="EQXM", domain=[reg,sec,T])
EQXM[reg,sec,T].where[TACTIVE[T]] = H_X[reg,sec,T] == (
    # Material intermediate demand (Leontief from XD) — zero for energy goods
    Sum(secc, sIO_X_mat_p[reg,sec,secc]*H_XD[reg,secc,T])
    # REF intermediate demand from EN bundles
    + is_ref_p[sec] * Sum(secc, sREF_X_p[reg,secc]
          * (H_PE[reg,secc,T] / (H_P[reg,sec,T] + tau_fuel[reg,secc,T]))**sigmaEN_p
          * H_EN[reg,secc,T])
    # ELF intermediate demand from ELC composites
    + is_elcff_p[sec] * Sum(secc, sELCFF_X_p[reg,secc]
          * (H_PELC[reg,secc,T] / H_P[reg,sec,T])**sigmaELC_p
          * H_ELC[reg,secc,T])
    # ELR intermediate demand from ELC composites
    + is_elcrn_p[sec] * Sum(secc, sELCRN_X_p[reg,secc]
          * (H_PELC[reg,secc,T] / H_P[reg,sec,T])**sigmaELC_p
          * H_ELC[reg,secc,T])
    # Final demand
    + H_C[reg,sec,T]*sC_X_p[reg,sec] + H_I[reg,sec,T]*sI_X_p[reg,sec]
)

# EQS — Bröcker-style: household savings = total investment expenditure
EQS = Equation(container=m, name="EQS", domain=[reg,T])
EQS[reg,T].where[TACTIVE[T]] = (
    H_S[reg,T]*B_S[reg,T] == Sum(sec, H_PK[reg,T]*INVP[reg,sec,T])
)

# EQLM
sL_LS_p = Parameter(container=m, name="sL_LS", domain=[reg,sec])
sL_LS_p.setRecords([(r,s, d[(r,s)]["l"]/cal[(r,s)]["lsz"]) for r in regions for s in sectors])

# Walras closure: keep every household budget constraint (EQY) and use the
# numeraire labor market as the omitted market-clearing equation.
EQLM = Equation(container=m, name="EQLM", domain=[reg,T])
EQLM[reg,T].where[RNNUM[reg] & TACTIVE[T]] = 1.0 == Sum(sec, H_L[reg,sec,T]*sL_LS_p[reg,sec])

# =============================================================================
# 10. VALORES INICIALES
# =============================================================================
def set_initvals(t_max=None):
    """Set all variables to baseline. For Phase2 (t_max<T_MAX), fix hats for T>t_max."""
    if t_max is None:
        t_max = T_MAX
    all_t = list(range(1, T_MAX + 1))
    fixing = (t_max < T_MAX)
    
    def make_2d(val_fn, fix_overshoot=True):
        if fixing and fix_overshoot:
            return pd.DataFrame(
                [(r,str(t), val_fn(r,t), val_fn(r,t), val_fn(r,t)) if t > t_max
                 else (r,str(t), val_fn(r,t), 0.0, float('inf'))
                 for r in regions for t in all_t],
                columns=["uni_0","uni_1","level","lower","upper"])
        return pd.DataFrame(
            [(r,str(t), val_fn(r,t)) for r in regions for t in all_t],
            columns=["uni_0","uni_1","level"])
    
    def make_3d(val_fn, fix_overshoot=True):
        if fixing and fix_overshoot:
            return pd.DataFrame(
                [(r,s,str(t), val_fn(r,s,t), val_fn(r,s,t), val_fn(r,s,t)) if t > t_max
                 else (r,s,str(t), val_fn(r,s,t), 0.0, float('inf'))
                 for r in regions for s in sectors for t in all_t],
                columns=["uni_0","uni_1","uni_2","level","lower","upper"])
        return pd.DataFrame(
            [(r,s,str(t), val_fn(r,s,t)) for r in regions for s in sectors for t in all_t],
            columns=["uni_0","uni_1","uni_2","level"])
    
    def make_4d(fix_overshoot=True):
        if fixing and fix_overshoot:
            return pd.DataFrame(
                [(r,s,rp,str(t), 1.0, 1.0, 1.0) if t > t_max
                 else (r,s,rp,str(t), 1.0, 0.0, float('inf'))
                 for r in regions for s in sectors for rp in regions if rp != r for t in all_t],
                columns=["uni_0","uni_1","uni_2","uni_3","level","lower","upper"])
        return pd.DataFrame(
            [(r,s,rp,str(t),1.0) for r in regions for s in sectors
             for rp in regions if rp != r for t in all_t],
            columns=["uni_0","uni_1","uni_2","uni_3","level"])

    def make_sec2d(val_fn, fix_overshoot=True):
        if fixing and fix_overshoot:
            return pd.DataFrame(
                [(s,str(t), val_fn(s,t), val_fn(s,t), val_fn(s,t)) if t > t_max
                 else (s,str(t), val_fn(s,t), 0.0, float('inf'))
                 for s in sectors for t in all_t],
                columns=["uni_0","uni_1","level","lower","upper"])
        return pd.DataFrame(
            [(s,str(t), val_fn(s,t)) for s in sectors for t in all_t],
            columns=["uni_0","uni_1","level"])
    
    one2d = lambda r,t: 1.0
    one3d = lambda r,s,t: 1.0
    for var in [H_PC, H_PL, H_PK, H_Y, H_S]:
        var.setRecords(make_2d(one2d))
    for var in [H_PD, H_PVA, H_PVAE, H_P, H_PMAGG, H_PE, H_PELC, H_XD, H_VA, H_VAE, H_EN, H_ELC,
                H_XDD, H_X, H_MAGG, H_L, H_C, H_I]:
        var.setRecords(make_3d(one3d))
    H_M.setRecords(make_4d())
    EU_CBAM_BM.setRecords(make_sec2d(lambda s,t: _eu_cbam_bm0[s]))
    # IRP, KP, INVP: never fixed (let equations determine in frozen zone)
    IRP.setRecords(make_2d(lambda r,t: ro_val, fix_overshoot=False))
    KP.setRecords(make_3d(lambda r,s,t: cal[(r,s)]["kz"]*(1+g_val)**(t-1), fix_overshoot=False))
    INVP.setRecords(make_3d(lambda r,s,t: cal[(r,s)]["inv"]*(1+g_val)**(t-1), fix_overshoot=False))

set_initvals()

# =============================================================================
# 11. FIJAR EXOGENAS
# =============================================================================
def fix_exogenous(t_max=None):
    if t_max is None:
        t_max = T_MAX
    for r in regions:
        for s in sectors:
            KP.fx[(r,s,"1")] = cal[(r,s)]["kz"]
    # Set TLAST to correct terminal period
    TLAST.setRecords([str(t_max)])
    TACTIVE.setRecords([str(i) for i in range(1, t_max + 1)])
    H_PL.fx[(NUMERAIRE,T)] = 1.0

fix_exogenous()

def reset_initvals(t_max=None):
    set_initvals(t_max)
    fix_exogenous(t_max)

# =============================================================================
# 12. MODELO
# =============================================================================
HatMod = Model(
    container=m,
    name="HatMod",
    equations=[
        EQEU_CBAM_BM,
        EQVA, EQPELC, EQPE, EQPVAE, EQPVAE_resid, EQVAE_VA, EQVAE_EN, EQELC_DEMAND, EQXD,
        EQARMP, EQIMPORT_TOP, EQARMD,
        EQPMAGG, EQIMPORT_BOT,
        EQXDD_eq,
        EQLAMBDA, EQK,
        EQPC, EQEULER, EQTC, EQTC2,
        EQFSTORL, EQINV, EQPK,
        EQC, EQY,
        EQXM, EQS,
        EQLM,
    ],
    matches={EQEU_CBAM_BM: EU_CBAM_BM},
    problem="MCP",
)

# =============================================================================
# 13. SOLVE INFRASTRUCTURE
# =============================================================================
SOLVE_ITER_LIM = env_int("RAMSEY_ITER_LIM", 10000)
PATH_CONVERGENCE_TOL = env_float("RAMSEY_PATH_TOL", 1e-7)
SOLVE_OPTS = gp.Options(hold_fixed_variables=True, iteration_limit=SOLVE_ITER_LIM)
SOLVER_OPTS = {
    "crash_method": "none", "crash_perturb": "no",
    "proximal_perturbation": 0, "output_linear_model": "no",
    "convergence_tolerance": PATH_CONVERGENCE_TOL,
}
print(f"PATH options: iter_lim={SOLVE_ITER_LIM}, convergence_tolerance={PATH_CONVERGENCE_TOL:g}")

def x2(v):
    df = v.records.copy()
    return dict(zip(zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str)), df["level"]))

def x3(v):
    df = v.records.copy()
    return dict(zip(zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str), df.iloc[:,2].astype(str)), df["level"]))

def x4(v):
    df = v.records.copy()
    return dict(zip(zip(df.iloc[:,0].astype(str), df.iloc[:,1].astype(str),
                        df.iloc[:,2].astype(str), df.iloc[:,3].astype(str)), df["level"]))

def extract_kp_at_tstar(result):
    ts = str(T_STAR)
    kp = {}
    for r in regions:
        for s in sectors:
            kp[(r,s)] = result["KP"].get((r,s,ts), cal[(r,s)]["kz"]*(1+g_val)**(T_STAR-1))
    return kp

def warm_start_from_phase1(phase1_result, t_max=None):
    if t_max is None:
        t_max = T_MAX
    all_t = list(range(1, T_MAX + 1))
    ts = str(T_STAR); offset = T_STAR - 1
    hat3d_map = {
        "H_XD": H_XD, "H_PD": H_PD, "H_P": H_P, "H_VA": H_VA,
        "H_VAE": H_VAE, "H_EN": H_EN, "H_ELC": H_ELC,
        "H_L": H_L, "H_C": H_C, "H_XDD": H_XDD, "H_X": H_X, "H_MAGG": H_MAGG,
        "H_PVA": H_PVA, "H_PVAE": H_PVAE, "H_PE": H_PE, "H_PELC": H_PELC, "H_PMAGG": H_PMAGG,
        "H_I": H_I,
    }
    for vname, var in hat3d_map.items():
        recs = []
        for r in regions:
            for s in sectors:
                v0 = phase1_result[vname].get((r,s,ts), 1.0)
                for t in all_t:
                    recs.append((r, s, str(t), v0 if t <= t_max else 1.0))
        var.setRecords(pd.DataFrame(recs, columns=["uni_0","uni_1","uni_2","level"]))
    hat2d_map = {"H_Y": H_Y, "H_S": H_S, "H_PK": H_PK, "H_PC": H_PC, "H_PL": H_PL}
    for vname, var in hat2d_map.items():
        recs = []
        for r in regions:
            v0 = phase1_result[vname].get((r,ts), 1.0)
            for t in all_t:
                recs.append((r, str(t), v0 if t <= t_max else 1.0))
        var.setRecords(pd.DataFrame(recs, columns=["uni_0","uni_1","level"]))
    bm_recs = []
    for s in sectors:
        bm0 = phase1_result["EU_CBAM_BM"].get((s,ts), _eu_cbam_bm0[s])
        for t in all_t:
            if t <= t_max:
                bm_recs.append((s, str(t), bm0, 0.0, float("inf")))
            else:
                fixed_bm = _eu_cbam_bm0[s]
                bm_recs.append((s, str(t), fixed_bm, fixed_bm, fixed_bm))
    EU_CBAM_BM.setRecords(pd.DataFrame(
        bm_recs, columns=["uni_0","uni_1","level","lower","upper"]
    ))
    recs = []
    for r in regions:
        for s in sectors:
            for rp in regions:
                if rp == r: continue
                v0 = phase1_result["H_M"].get((r,s,rp,ts), 1.0)
                for t in all_t:
                    recs.append((r, s, rp, str(t), v0 if t <= t_max else 1.0))
    H_M.setRecords(pd.DataFrame(recs, columns=["uni_0","uni_1","uni_2","uni_3","level"]))
    recs = []
    for r in regions:
        ir0 = phase1_result["IRP"].get((r,ts), ro_val)
        for t in all_t:
            recs.append((r, str(t), ir0 if t <= t_max else ro_val))
    IRP.setRecords(pd.DataFrame(recs, columns=["uni_0","uni_1","level"]))
    kp_recs = []; invp_recs = []
    for r in regions:
        for s in sectors:
            k0 = phase1_result["KP"].get((r,s,ts), cal[(r,s)]["kz"]*(1+g_val)**offset)
            inv0 = phase1_result["INVP"].get((r,s,ts), cal[(r,s)]["inv"]*(1+g_val)**offset)
            for t in all_t:
                if t <= t_max:
                    kp_recs.append((r, s, str(t), k0*(1+g_val)**(t-1)))
                    invp_recs.append((r, s, str(t), inv0*(1+g_val)**(t-1)))
                else:
                    kp_recs.append((r, s, str(t), cal[(r,s)]["kz"]*(1+g_val)**(t-1)))
                    invp_recs.append((r, s, str(t), cal[(r,s)]["inv"]*(1+g_val)**(t-1)))
    KP.setRecords(pd.DataFrame(kp_recs, columns=["uni_0","uni_1","uni_2","level"]))
    INVP.setRecords(pd.DataFrame(invp_recs, columns=["uni_0","uni_1","uni_2","level"]))
    H_PL.fx[(NUMERAIRE,T)] = 1.0
    print(f"  Warm-started from Phase 1 solution at T*={T_STAR} (t_max={t_max})")

def solve_eq(
    name,
    fa_func,
    kp_init=None,
    phase2=False,
    phase1_result=None,
    with_cbam=True,
    cbam_scale_func=None,
):
    print(f"\n{'='*70}")
    print(f"  SOLVING: {name}")
    print(f"{'='*70}")
    t_max_solve = T_MAX_P2 if phase2 else T_MAX
    t_max_export = t_max_solve
    if phase2:
        set_all_baselines(offset=T_STAR-1, t_max=t_max_solve)
    else:
        set_all_baselines(offset=0, t_max=t_max_solve)
    reset_initvals(t_max=t_max_solve)
    set_shocks(
        fa_func,
        with_cbam=with_cbam,
        t_max=t_max_solve,
        year_offset=(T_STAR - 1) if phase2 else 0,
        cbam_scale_func=cbam_scale_func,
    )
    if phase2 and phase1_result is not None and kp_init is not None:
        warm_start_from_phase1(phase1_result, t_max=t_max_solve)
        for r in regions:
            for s in sectors:
                KP.fx[(r,s,"1")] = kp_init[(r,s)]
        print(f"  Initial capital chained from Phase 1 at T*={T_STAR} ({T_STAR_YEAR})")
        print(f"  Phase 2 solve horizon: T=1..{t_max_solve} (internal terminal year {model_year(t_max_solve, phase2=True)})")
        print(f"  Phase 2 export horizon: T=1..{t_max_export} (comparable years {T_STAR_YEAR}-{BASE_YEAR+T_MAX-1})")
    elif kp_init is not None:
        for r in regions:
            for s in sectors:
                KP.fx[(r,s,"1")] = kp_init[(r,s)]
    print(f"  Policy path (P_CO2 = {CARBON_PRICE*SCALE:.0f} EUR/tCO2, t_max={t_max_solve}):")
    for t in [1, 5, 7, 8, 9, 10, 13, t_max_solve]:
        if t > t_max_solve: continue
        fa = fa_func(t)
        cal_year = model_year(t, phase2=phase2)
        print(f"    T={t:2d} ({cal_year}): FA={fa:.3f}  effective_rate={1-fa:.3f}")
    HatMod.solve(options=SOLVE_OPTS, solver_options=SOLVER_OPTS, output=sys.stdout)
    status_text = str(HatMod.status)
    print(f"  Solve status: {status_text}")

    def _optional_model_diagnostic(attribute):
        try:
            value = getattr(HatMod, attribute, None)
            return None if value is None else float(value)
        except (AttributeError, TypeError, ValueError):
            return None

    max_infeasibility = _optional_model_diagnostic("max_infeasibility")
    mean_infeasibility = _optional_model_diagnostic("mean_infeasibility")
    num_infeasibilities = _optional_model_diagnostic("num_infeasibilities")
    print(
        "  Infeasibility diagnostics: "
        f"max={max_infeasibility}, mean={mean_infeasibility}, "
        f"count={num_infeasibilities}"
    )
    if "OptimalGlobal" not in status_text:
        raise RuntimeError(f"{name} failed: model status is {status_text}.")
    infeasibility_limit = max(PATH_CONVERGENCE_TOL * 1.05, PATH_CONVERGENCE_TOL + 1e-12)
    if (
        max_infeasibility is not None
        and np.isfinite(max_infeasibility)
        and max_infeasibility > infeasibility_limit
    ):
        raise RuntimeError(
            f"{name} failed: max infeasibility {max_infeasibility:.6g} exceeds "
            f"the validation limit {infeasibility_limit:.6g}."
        )
    result = {
        "name": name, "status": status_text, "phase2": phase2,
        "t_max_solve": t_max_solve,
        "t_max_export": t_max_export,
        "max_infeasibility": max_infeasibility,
        "mean_infeasibility": mean_infeasibility,
        "num_infeasibilities": num_infeasibilities,
        "H_XD": x3(H_XD), "H_PD": x3(H_PD), "H_P": x3(H_P),
        "H_VA": x3(H_VA), "H_VAE": x3(H_VAE), "H_EN": x3(H_EN),
        "H_L": x3(H_L), "H_C": x3(H_C),
        "H_XDD": x3(H_XDD), "H_X": x3(H_X), "H_MAGG": x3(H_MAGG),
        "H_PVA": x3(H_PVA), "H_PVAE": x3(H_PVAE), "H_PE": x3(H_PE),
        "H_PELC": x3(H_PELC), "H_ELC": x3(H_ELC),
        "H_PMAGG": x3(H_PMAGG), "H_I": x3(H_I),
        "H_Y": x2(H_Y), "H_S": x2(H_S), "H_PC": x2(H_PC),
        "IRP": x2(IRP), "KP": x3(KP), "INVP": x3(INVP),
        "H_PK": x2(H_PK), "H_PL": x2(H_PL), "H_M": x4(H_M),
        "EU_CBAM_BM": x2(EU_CBAM_BM),
    }
    # Restore TLAST to T_MAX for subsequent solves
    if phase2:
        TLAST.setRecords([str(T_MAX)])
    return result

# =============================================================================
# 14. SOLVE
# =============================================================================
print("\n"+"="*70)
print("  STEP 0: BASELINE VERIFICATION (hat=1, iteration_limit=0)")
print("="*70)
reset_initvals()
set_shocks_zero()
HatMod.solve(options=gp.Options(hold_fixed_variables=True, iteration_limit=0), output=sys.stdout)
print(f"  Baseline status: {HatMod.status}")

eq1 = solve_eq("Eq.1 NO POLICY (baseline)", fa_no_policy)

# --- Quick diagnostic: baseline should be all 1.0 ---
print("\n=== DIAGNOSTIC: Baseline (Eq1) ===")
hy = eq1["H_Y"]  # dict: (reg, T) -> value
max_dev = max(abs(v - 1.0) for (r,t), v in hy.items() if r == "EUD")
print(f"  EUD H_Y max deviation from 1.0: {max_dev:.2e}")
if max_dev > 1e-4:
    print("  WARNING: baseline not reproducing hat=1!")
else:
    print("  OK: baseline reproduces hat=1")
print("=== END BASELINE DIAGNOSTIC ===\n")

eq2 = solve_eq("Eq.2 CREDIBLE COMMITMENT (legislated schedule)", fa_credible)

# --- Quick diagnostic: check key variables for EU ---
print("\n=== DIAGNOSTIC: Credible (Eq2) key variables ===")
hy = eq2["H_Y"]; hs = eq2["H_S"]; hp = eq2["H_P"]
for r in ["EUD"]:
    print(f"  {r}:")
    for t in [1, 5, 8, 13, 20, 30]:
        ts = str(t)
        print(f"    T={t}: H_Y={hy.get((r,ts),1):.6f} H_S={hs.get((r,ts),1):.6f} H_P[OTH]={hp.get((r,'OTH',ts),1):.6f} H_P[CEM]={hp.get((r,'CEM',ts),1):.6f}")
print("=== END DIAGNOSTIC ===\n")

# --- NoCBAM: solve right after Credible (clean state) ---
eq2_nocbam = solve_eq("Eq.2 CREDIBLE (NO CBAM, tau_c only)", fa_credible, with_cbam=False)

kp_credible_tstar = extract_kp_at_tstar(eq2)
eq3 = solve_eq(
    f"Eq.3 RENEGING Phase 2 (FA reverts to 1.0 from {T_STAR_YEAR})",
    fa_reneging_phase2,
    kp_init=kp_credible_tstar, phase2=True, phase1_result=eq2
)

if abs(CREDIBILITY_INDEX) <= 1e-12:
    _eq4_phase1_name = (
        f"Eq.4 ANTICIPATED WITHDRAWAL full path (withdrawal from {T_STAR_YEAR})"
    )
else:
    _eq4_phase1_name = (
        "Eq.4 DETERMINISTIC PERCEIVED-POLICY Phase 1 "
        f"(kappa={CREDIBILITY_INDEX:.6g})"
    )
eq4_phase1 = solve_eq(
    _eq4_phase1_name,
    fa_perceived_policy_phase1,
    cbam_scale_func=cbam_scale_perceived_policy_phase1,
)

kp_perceived_tstar = extract_kp_at_tstar(eq4_phase1)
eq4 = solve_eq(
    f"Eq.4 LEGISLATED POLICY REALIZED Phase 2 (continues from {T_STAR_YEAR})",
    fa_legislated_phase2,
    kp_init=kp_perceived_tstar, phase2=True, phase1_result=eq4_phase1
)

def stitch_phase_paths(name, phase1_result, phase2_result):
    """Return a full 2022-2051 path: phase 1 through T*-1, phase 2 from T* onward."""
    _max_infeasibilities = [
        value for value in [
            phase1_result.get("max_infeasibility"),
            phase2_result.get("max_infeasibility"),
        ] if value is not None
    ]
    _mean_infeasibilities = [
        value for value in [
            phase1_result.get("mean_infeasibility"),
            phase2_result.get("mean_infeasibility"),
        ] if value is not None
    ]
    _num_infeasibilities = [
        value for value in [
            phase1_result.get("num_infeasibilities"),
            phase2_result.get("num_infeasibilities"),
        ] if value is not None
    ]
    stitched = {
        "name": name,
        "status": f"{phase1_result['status']} + {phase2_result['status']}",
        "phase2": False,
        "stitched": True,
        "t_max_solve": T_MAX,
        "t_max_export": T_MAX,
        "max_infeasibility": max(_max_infeasibilities) if _max_infeasibilities else None,
        "mean_infeasibility": max(_mean_infeasibilities) if _mean_infeasibilities else None,
        "num_infeasibilities": sum(_num_infeasibilities) if _num_infeasibilities else None,
    }
    for key, phase1_values in phase1_result.items():
        if not isinstance(phase1_values, dict):
            continue
        phase2_values = phase2_result.get(key, {})
        combined = {}
        for idx, val in phase1_values.items():
            *dims, ts = idx
            if int(ts) < T_STAR:
                combined[idx] = val
        for idx, val in phase2_values.items():
            *dims, ts = idx
            global_t = int(ts) + T_STAR - 1
            if global_t <= T_MAX:
                combined[tuple(dims + [str(global_t)])] = val
        stitched[key] = combined
    return stitched

eq3_full = stitch_phase_paths(
    "Eq.3 RENEGING full path (credible phase 1, reneging phase 2)",
    eq2,
    eq3,
)
eq4_full = stitch_phase_paths(
    "Eq.4 full path (deterministic perceived policy phase 1, legislated phase 2)",
    eq4_phase1,
    eq4,
)

# =============================================================================
# 15. RESULTS
# =============================================================================
print("\n" + "="*70)
print("  RESULTS COMPARISON")
print("="*70)

equilibria = {
    "Eq1_NoPol": eq1, "Eq2_Credible": eq2, "Eq2_NoCBAM": eq2_nocbam,
    "Eq3_Reneg": eq3_full,
    "Eq4_Phase1": eq4_phase1, "Eq4_Surprise": eq4_full,
}

_eu_avg_int = {s: np.mean([emis_intensity.get((r,s),0) for r in EU_REGIONS]) for s in sectors}
cbam_secs = sorted([s for s in sectors if _eu_avg_int[s] > 0.01],
                    key=lambda s: _eu_avg_int[s], reverse=True)[:4]

report_times = sorted({1, 5, T_STAR - 1, T_STAR, T_STAR + 1, 10, 13, 20, T_MAX})

for eq_name, eq_res in equilibria.items():
    print(f"\n  === {eq_name}: {eq_res['name']} (status: {eq_res['status']}) ===")
    r0 = EU_REGIONS[0]
    is_p2 = eq_res.get("phase2", False)
    hdr = f"  {'T':>3} {'Year':>5}"
    for s in cbam_secs:
        hdr += f" {'XD_'+s:>9}"
    hdr += f" {'IR':>9}"
    print(hdr)
    for t in report_times:
        if t > eq_res.get("t_max_export", T_MAX):
            continue
        ts = str(t)
        year = (BASE_YEAR + T_STAR - 1 + t - 1) if is_p2 else (BASE_YEAR + t - 1)
        line = f"  {t:3d} {year:5d}"
        for s in cbam_secs:
            xd = eq_res["H_XD"].get((r0, s, ts), 1.0)
            line += f" {(xd-1)*100:+8.3f}%"
        ir = eq_res["IRP"].get((r0, ts), ro_val)
        line += f" {ir:9.5f}"
        print(line)

print("\n  SOLVE SUMMARY:")
for eq_name, eq_res in equilibria.items():
    print(f"    {eq_name}: {eq_res['status']}")

# =============================================================================
# 16. EXPORT RESULTS TO PARQUET
# =============================================================================
print("\n" + "="*70)
print("  EXPORTING RESULTS TO PARQUET")
print("="*70)

os.makedirs(OUT_DIR, exist_ok=True)

vars_3d = ["H_XD", "H_PD", "H_P", "H_VA", "H_VAE", "H_EN", "H_ELC", "H_L", "H_C",
            "H_XDD", "H_X", "H_MAGG",
            "H_PVA", "H_PVAE", "H_PE", "H_PELC", "H_PMAGG", "H_I", "KP", "INVP"]
vars_2d = ["H_Y", "H_S", "IRP", "H_PK", "H_PC", "H_PL"]
vars_4d = ["H_M"]

def build_3d_rows(var_dict, eq_name, var_name, is_phase2=False, t_max=T_MAX):
    offset = (T_STAR - 1) if is_phase2 else 0
    rows = []
    for (r, s, ts), val in var_dict.items():
        t = int(ts)
        if t > t_max:
            continue
        rows.append({"eq": eq_name, "var": var_name, "reg": r, "sec": s,
                      "T": t, "year": BASE_YEAR + offset + t - 1, "value": val})
    return rows

def build_2d_rows(var_dict, eq_name, var_name, is_phase2=False, t_max=T_MAX):
    offset = (T_STAR - 1) if is_phase2 else 0
    rows = []
    for (r, ts), val in var_dict.items():
        t = int(ts)
        if t > t_max:
            continue
        rows.append({"eq": eq_name, "var": var_name, "reg": r, "sec": "",
                      "T": t, "year": BASE_YEAR + offset + t - 1, "value": val})
    return rows

def build_4d_rows(var_dict, eq_name, var_name, is_phase2=False, t_max=T_MAX):
    offset = (T_STAR - 1) if is_phase2 else 0
    rows = []
    for (r, s, rp, ts), val in var_dict.items():
        t = int(ts)
        if t > t_max:
            continue
        rows.append({"eq": eq_name, "var": var_name, "reg": r, "sec": s,
                      "partner": rp, "T": t, "year": BASE_YEAR + offset + t - 1, "value": val})
    return rows

def build_sec2d_rows(var_dict, eq_name, var_name, is_phase2=False, t_max=T_MAX):
    offset = (T_STAR - 1) if is_phase2 else 0
    rows = []
    for (s, ts), val in var_dict.items():
        t = int(ts)
        if t > t_max:
            continue
        rows.append({"eq": eq_name, "var": var_name, "reg": "EU", "sec": s,
                     "T": t, "year": BASE_YEAR + offset + t - 1, "value": val})
    return rows

all_rows_3d = []
all_rows_2d = []
all_rows_4d = []

for eq_name, eq_res in equilibria.items():
    is_p2 = eq_res.get("phase2", False)
    t_max_export = eq_res.get("t_max_export", T_MAX)
    for vname in vars_3d:
        if vname in eq_res:
            all_rows_3d.extend(build_3d_rows(eq_res[vname], eq_name, vname, is_p2, t_max_export))
    for vname in vars_2d:
        if vname in eq_res:
            all_rows_2d.extend(build_2d_rows(eq_res[vname], eq_name, vname, is_p2, t_max_export))
    for vname in vars_4d:
        if vname in eq_res:
            all_rows_4d.extend(build_4d_rows(eq_res[vname], eq_name, vname, is_p2, t_max_export))
    if "EU_CBAM_BM" in eq_res:
        all_rows_2d.extend(build_sec2d_rows(eq_res["EU_CBAM_BM"], eq_name, "EU_CBAM_BM", is_p2, t_max_export))

df_3d = pd.DataFrame(all_rows_3d)
df_2d = pd.DataFrame(all_rows_2d)
df_4d = pd.DataFrame(all_rows_4d)

f3 = os.path.join(OUT_DIR, f"results_3d_{OUT_SUFFIX}.parquet")
f2 = os.path.join(OUT_DIR, f"results_2d_{OUT_SUFFIX}.parquet")
f4 = os.path.join(OUT_DIR, f"results_4d_{OUT_SUFFIX}.parquet")

df_3d.to_parquet(f3, index=False)
df_2d.to_parquet(f2, index=False)
df_4d.to_parquet(f4, index=False)

print(f"  results_3d_{OUT_SUFFIX}.parquet: {len(df_3d):,} rows  ({os.path.getsize(f3)/1e6:.2f} MB)")
print(f"    Variables: {vars_3d}")
print(f"    Equilibria: {sorted(df_3d['eq'].unique())}")
print(f"  results_2d_{OUT_SUFFIX}.parquet: {len(df_2d):,} rows  ({os.path.getsize(f2)/1e6:.2f} MB)")
print(f"    Variables: {vars_2d}")
print(f"  results_4d_{OUT_SUFFIX}.parquet: {len(df_4d):,} rows  ({os.path.getsize(f4)/1e6:.2f} MB)")
print(f"    Variables: {vars_4d}")

# --- Parameters ---
meta_rows = []
for r in regions:
    for s in sectors:
        meta_rows.append({"param": "emis_intensity", "reg": r, "sec": s,
                          "partner": "", "value": emis_intensity.get((r,s), 0.0)})
        meta_rows.append({"param": "emis_ff_intensity", "reg": r, "sec": s,
                          "partner": "", "value": emis_ff_intensity.get((r,s), 0.0)})
        meta_rows.append({"param": "emis_proc_intensity", "reg": r, "sec": s,
                          "partner": "", "value": emis_proc_intensity.get((r,s), 0.0)})
        meta_rows.append({"param": "emis_fuel_per_ref", "reg": r, "sec": s,
                          "partner": "", "value": emis_fuel_per_ref.get((r,s), 0.0)})
        meta_rows.append({"param": "cbam_direct_intensity", "reg": r, "sec": s,
                          "partner": "", "value": cbam_direct_intensity.get((r,s), 0.0)})
        meta_rows.append({"param": "cbam_indirect_elc_intensity", "reg": r, "sec": s,
                          "partner": "", "value": cbam_indirect_elc_intensity.get((r,s), 0.0)})
        meta_rows.append({"param": "cbam_embodied_intensity", "reg": r, "sec": s,
                          "partner": "", "value": cbam_embodied_intensity.get((r,s), 0.0)})
for t in range(1, T_MAX+1):
    meta_rows.append({"param": "FA_schedule", "reg": "", "sec": "",
                      "partner": "", "T": t, "year": BASE_YEAR + t - 1,
                      "value": get_fa(t)})

# Export the exact policy path used by each reported equilibrium.  For the
# endogenous-benchmark formula, ``scenario_cbam_on`` multiplies the origin
# intensity and ``scenario_cbam_benchmark_weight`` (= cbam_on * FA) multiplies
# the EU benchmark deduction.  Hence the latter is identically zero once FA=0.
_scenario_policy_specs = {
    "Eq1_NoPol": (fa_no_policy, True, None),
    "Eq2_Credible": (fa_credible, True, None),
    "Eq2_NoCBAM": (fa_credible, False, None),
    "Eq3_Reneg": (fa_reneging_full_path, True, None),
    "Eq4_Phase1": (
        fa_perceived_policy_phase1,
        True,
        cbam_scale_perceived_policy_phase1,
    ),
    "Eq4_Surprise": (fa_credible, True, None),
}
if ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE:
    _scenario_policy_specs["AnticipatedWithdrawal"] = (
        fa_anticipated_withdrawal_full_path,
        True,
        cbam_scale_perceived_policy_phase1,
    )
    meta_rows.append({
        "param": "scenario_alias", "reg": "", "sec": "AnticipatedWithdrawal",
        "partner": "Eq4_Phase1", "value": 1.0,
    })
for _scenario, (_fa_func, _with_cbam, _cbam_scale_func) in _scenario_policy_specs.items():
    for t in range(1, T_MAX + 1):
        _year = BASE_YEAR + t - 1
        _fa = _fa_func(t)
        _cbam_on = _cbam_on_value(_year, _fa, _with_cbam)
        if _cbam_on and _cbam_scale_func is not None:
            _cbam_on *= float(_cbam_scale_func(t))
        for _param, _value in [
            ("scenario_fa_path", _fa),
            ("scenario_domestic_carbon_factor", 1.0 - _fa),
            ("scenario_cbam_on", _cbam_on),
            ("scenario_cbam_benchmark_weight", _cbam_on * _fa),
        ]:
            meta_rows.append({
                "param": _param, "reg": "", "sec": "", "partner": _scenario,
                "T": t, "year": _year, "value": _value,
            })

# Solver diagnostics are attached to scenario keys so long-run batch runners
# cannot silently publish a formally unsuccessful or infeasible equilibrium.
for _scenario, _result in equilibria.items():
    meta_rows.append({
        "param": "solve_optimal_global", "reg": "", "sec": "",
        "partner": _scenario,
        "value": 1.0 if "OptimalGlobal" in _result.get("status", "") else 0.0,
    })
    for _param in [
        "max_infeasibility", "mean_infeasibility", "num_infeasibilities"
    ]:
        _value = _result.get(_param)
        if _value is not None:
            meta_rows.append({
                "param": f"solve_{_param}", "reg": "", "sec": "",
                "partner": _scenario, "value": float(_value),
            })
for r in regions:
    for s in sectors:
        for pname, pval in [("alphaF", cal[(r,s)]["alphaF"]),
                             ("phi", cal[(r,s)]["phi"]),
                             ("delta", cal[(r,s)]["delta"]),
                             ("kz", cal[(r,s)]["kz"]),
                             ("inv", cal[(r,s)]["inv"]),
                             ("pva", cal[(r,s)]["pva"]),
                             ("pvae", cal[(r,s)]["pvae"]),
                             ("alphaH", cal[(r,s)]["alphaH"]),
                             ("alphaI", cal[(r,s)]["alphaI"]),
                             ("taxShare", cal[(r,s)]["taxShare"]),
                             ("div", cal[(r,s)]["div"]),
                             ("theta_va", cal[(r,s)]["theta_va"]),
                             ("theta_en", cal[(r,s)]["theta_en"])]:
            meta_rows.append({"param": pname, "reg": r, "sec": s,
                              "partner": "", "value": pval})
for s in sectors:
    meta_rows.append({"param": "sigmaA", "reg": "", "sec": s, "partner": "", "value": sigmaA_dict[s]})
    meta_rows.append({"param": "sigmaM", "reg": "", "sec": s, "partner": "", "value": sigmaM_dict[s]})
for r in regions:
    for s in sectors:
        meta_rows.append({"param": "gammaA", "reg": r, "sec": s, "partner": "",
                          "value": arm_top[(r,s)]["gammaA"]})
        for rp in regions:
            if rp == r: continue
            meta_rows.append({"param": "gammaM", "reg": r, "sec": s, "partner": rp,
                              "value": arm_bot[(r,s)]["gamma"].get(rp, 0.0)})
for r in regions:
    for si in sectors:
        for sj in sectors:
            meta_rows.append({"param": "io", "reg": r, "sec": si, "partner": sj,
                              "value": io_val[(r,si,sj)]})
for pname, pval in [("g", g_val), ("rss", rss_val), ("ro", ro_val),
                     ("adjsh", adjsh_val), ("rp", rp_val),
                     ("gamma", gamma_val),
                     ("ARMINGTON_SCALE", ARMINGTON_SCALE),
                     ("USE_BASE_TAX_WEDGES", 1.0 if USE_BASE_TAX_WEDGES else 0.0),
                     ("CBAM_INCLUDE_INDIRECT_ELECTRICITY", 1.0 if CBAM_INCLUDE_INDIRECT_ELECTRICITY else 0.0),
                     ("CBAM_INDIRECT_LEGACY_BOOLEAN", 1.0 if _CBAM_INDIRECT_LEGACY_ENABLED else 0.0),
                     ("CBAM_INDIRECT_SECTOR_OVERRIDE", 1.0 if CBAM_INDIRECT_SECTOR_OVERRIDE else 0.0),
                     ("CBAM_INDIRECT_SECTOR_COUNT", float(len(CBAM_INDIRECT_SECTORS))),
                     ("CBAM_INCLUDE_PROCESS_EMISSIONS", 1.0 if CBAM_INCLUDE_PROCESS_EMISSIONS else 0.0),
                     ("CBAM_EU_BENCHMARK_ENDOGENOUS", 1.0 if CBAM_EU_BENCHMARK_ENDOGENOUS else 0.0),
                     ("CBAM_FLOOR_EPS", CBAM_FLOOR_EPS),
                     ("CBAM_START_YEAR", CBAM_START_YEAR),
                     ("CBAM_START_T", CBAM_START_YEAR - BASE_YEAR + 1),
                     ("PATH_CONVERGENCE_TOL", PATH_CONVERGENCE_TOL),
                     ("SOLVE_ITER_LIM", SOLVE_ITER_LIM),
                     ("CARBON_PRICE_EUR", CARBON_PRICE*SCALE),
                     ("T_STAR", T_STAR), ("T_STAR_YEAR", T_STAR_YEAR),
                     ("T_MAX", T_MAX), ("T_MAX_P2", T_MAX_P2),
                     ("BASE_YEAR", BASE_YEAR), ("SCALE", SCALE),
                     ("SIGMA_VAE", SIGMA_VAE), ("SIGMA_EN", SIGMA_EN),
                     ("SIGMA_ELC", SIGMA_ELC),
                     ("SCRAP_MULT", SCRAP_MULT), ("SCRAP_MARGIN", SCRAP_MARGIN),
                     ("CREDIBILITY_INDEX", CREDIBILITY_INDEX),
                     # Deprecated numeric alias retained for old readers.
                     ("CREDIBILITY_P", CREDIBILITY_INDEX),
                     ("CREDIBILITY_INDEX_FROM_LEGACY_ALIAS", 1.0 if CREDIBILITY_INDEX_FROM_LEGACY_ALIAS else 0.0),
                     ("ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE", 1.0 if ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE else 0.0)]:
    meta_rows.append({"param": pname, "reg": "", "sec": "", "partner": "", "value": pval})

# Record the covered set in the run's own metadata so downstream accounting can
# recover which sectors carried the carbon wedge without the caller having to
# know the run's coverage (one row per covered sector; value = 1.0).
for _s in CBAM_ACTIVE:
    meta_rows.append({"param": "cbam_covered", "reg": "", "sec": _s, "partner": "", "value": 1.0})
for _s in sectors:
    meta_rows.append({
        "param": "cbam_indirect_covered", "reg": "", "sec": _s,
        "partner": "", "value": 1.0 if _s in CBAM_INDIRECT_SECTORS else 0.0,
    })

# --- solve provenance -------------------------------------------------------
# A completeness check ("the five parquets exist") cannot tell a family solved
# under the current economic specification from one solved before a formulation
# change.  Stamp the specification version, the model source hash and the input
# data hash so downstream code can refuse or regenerate stale families.
# Strings live in `partner` because `value` is a float column.
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


try:
    _model_sha = _sha256_file(os.path.abspath(__file__))
except Exception:
    _model_sha = "unavailable"
try:
    _input_sha = _sha256_file(SAM_FILE)
except Exception:
    _input_sha = "unavailable"
try:
    import gamspy as _gamspy
    _solver = f"gamspy {_gamspy.__version__}"
except Exception:
    _solver = "gamspy unknown"

meta_rows.append({"param": "provenance", "reg": "", "sec": "MODEL_SPEC_VERSION",
                  "partner": MODEL_SPEC_LABEL, "value": float(MODEL_SPEC_VERSION)})
meta_rows.append({"param": "provenance", "reg": "", "sec": "model_sha256",
                  "partner": _model_sha, "value": 0.0})
meta_rows.append({"param": "provenance", "reg": "", "sec": "input_sha256",
                  "partner": _input_sha, "value": 0.0})
meta_rows.append({"param": "provenance", "reg": "", "sec": "input_file",
                  "partner": os.path.basename(SAM_FILE), "value": 0.0})
meta_rows.append({"param": "provenance", "reg": "", "sec": "solved_utc",
                  "partner": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "value": 0.0})
meta_rows.append({"param": "provenance", "reg": "", "sec": "solver",
                  "partner": _solver, "value": 0.0})

df_meta = pd.DataFrame(meta_rows)
fm = os.path.join(OUT_DIR, f"parameters_{OUT_SUFFIX}.parquet")
df_meta.to_parquet(fm, index=False)
print(f"  parameters_{OUT_SUFFIX}.parquet: {len(df_meta):,} rows  ({os.path.getsize(fm)/1e6:.2f} MB)")

# --- SAM data ---
sam_rows = []
for r in regions:
    for s in sectors:
        rs = (r,s)
        for dname in ["xd","va","l","ky","c_raw","iz","tax","tax_raw","e_total","m_total","xdd","x"]:
            if dname in d[rs]:
                sam_rows.append({"var": dname, "reg": r, "sec": s, "partner": "", "value": d[rs][dname]})
        # Energy data
        for ename in ["elc_ff","elc_rn","elc","ref","en_total"]:
            if ename in en_data[rs]:
                sam_rows.append({"var": f"en_{ename}", "reg": r, "sec": s, "partner": "", "value": en_data[rs][ename]})
    sam_rows.append({"var": "vbop", "reg": r, "sec": "", "partner": "", "value": vbop_val[r]})
for r in regions:
    for s in sectors:
        for rp in regions:
            if rp == r: continue
            sam_rows.append({"var": "trade_bil", "reg": r, "sec": s, "partner": rp,
                             "value": trade_bil.get((r,s,rp), 0.0)})
            sam_rows.append({"var": "trade_m_bil", "reg": r, "sec": s, "partner": rp,
                             "value": trade_m_bil.get((r,s,rp), 0.0)})

df_sam = pd.DataFrame(sam_rows)
fs = os.path.join(OUT_DIR, f"sam_data_{OUT_SUFFIX}.parquet")
df_sam.to_parquet(fs, index=False)
print(f"  sam_data_{OUT_SUFFIX}.parquet: {len(df_sam):,} rows  ({os.path.getsize(fs)/1e6:.2f} MB)")

total_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".parquet")]
total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in total_files) / 1e6
print(f"\n  Total: {len(total_files)} files, {total_mb:.1f} MB in {OUT_DIR}/")
print("  Done.")
