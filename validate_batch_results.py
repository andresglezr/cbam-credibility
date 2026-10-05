# -*- coding: utf-8 -*-
"""Validate the complete EU-benchmark recalculation batch without solving.

The default invocation validates the central run, the three interior kappa
points and the twelve paper sensitivities.  It intentionally treats a missing
member of that matrix as an error.  ``--available`` is the resumable mode: it
validates only suffixes for which all five parquet files and welfare CSV exist
and are non-empty.

Examples
--------
Validate the full required batch::

    python validate_batch_results.py

Validate only complete families currently on disk::

    python validate_batch_results.py --available

Validate a named family::

    python validate_batch_results.py --suffix v3eubm_legind
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"

BASE_YEAR = 2022
T_MAX = 30
CBAM_START_YEAR = 2026
EXPECTED_T = tuple(range(1, T_MAX + 1))
EXPECTED_YEARS = tuple(range(BASE_YEAR, BASE_YEAR + T_MAX))
EQUILIBRIA = (
    "Eq1_NoPol",
    "Eq2_Credible",
    "Eq2_NoCBAM",
    "Eq3_Reneg",
    "Eq4_Phase1",
    "Eq4_Surprise",
)
EU_REGIONS = ("EUC", "EUD", "EUM")
COVERED_SECTORS = ("ALU", "CEM", "FER", "STL")
CENTRAL_INDIRECT_SECTORS = ("CEM", "FER")
ANTICIPATED_WITHDRAWAL = "AnticipatedWithdrawal"

FA_SCHEDULE = {
    1: 1.000,
    2: 1.000,
    3: 1.000,
    4: 1.000,
    5: 0.975,
    6: 0.950,
    7: 0.900,
    8: 0.775,
    9: 0.515,
    10: 0.390,
    11: 0.265,
    12: 0.140,
}

BASE_SIGMA_A = {
    "STL": 2.95,
    "ALU": 3.95,
    "CEM": 2.90,
    "ELF": 2.80,
    "ELR": 2.80,
    "FER": 3.75,
    "CHM": 3.30,
    "REF": 2.10,
    "OTH": 2.90,
}
BASE_SIGMA_M = {
    "STL": 5.90,
    "ALU": 7.90,
    "CEM": 5.80,
    "ELF": 5.60,
    "ELR": 5.60,
    "FER": 7.50,
    "CHM": 6.60,
    "REF": 4.20,
    "OTH": 5.80,
}

RESULT_FILES = {
    "results_3d": "results_3d_{suffix}.parquet",
    "results_2d": "results_2d_{suffix}.parquet",
    "results_4d": "results_4d_{suffix}.parquet",
    "parameters": "parameters_{suffix}.parquet",
    "sam_data": "sam_data_{suffix}.parquet",
    "welfare": "welfare_{suffix}.csv",
}

WELFARE_NUMERIC_COLUMNS = (
    "ce_vs_Eq1_pct",
    "ce_vs_Eq2_pct",
    "ce_factor_vs_Eq1",
    "ce_factor_vs_Eq2",
    "finite_ce_vs_Eq1_pct",
    "finite_ce_vs_Eq2_pct",
    "finite_ce_factor_vs_Eq1",
    "finite_ce_factor_vs_Eq2",
    "terminal_tail_ce_vs_Eq1_pct",
    "terminal_tail_ce_vs_Eq2_pct",
    "terminal_tail_ce_factor_vs_Eq1",
    "terminal_tail_ce_factor_vs_Eq2",
)

REQUIRED_COLUMNS = {
    "results_2d": ("eq", "var", "reg", "sec", "T", "year", "value"),
    "results_3d": ("eq", "var", "reg", "sec", "T", "year", "value"),
    "results_4d": ("eq", "var", "reg", "sec", "partner", "T", "year", "value"),
    "parameters": ("param", "reg", "sec", "partner", "T", "year", "value"),
    "sam_data": ("var", "reg", "sec", "partner", "value"),
    "welfare": ("scope", "reg", "eq", *WELFARE_NUMERIC_COLUMNS),
}


@dataclass(frozen=True)
class RunSpec:
    suffix: str
    kind: str
    kappa: float = 0.0
    scalar_overrides: Mapping[str, float] = field(default_factory=dict)
    direct_only: bool = False
    #: Economic-specification version this family must have been solved under.
    #: 2 = Euler equation deflated by p^C; 1 = the superseded formulation, which
    #: predates provenance stamping and therefore carries no stamp at all.
    spec_version: float = 2.0


# The published matrix and the historical one differ only in their namespace,
# so both are generated from a single definition: a runner and the validator
# can no longer drift apart on which families exist or what defines them.
_SENSITIVITY_VARIANTS: tuple[tuple[str, dict, bool], ...] = (
    ("sigma_en_low", {"SIGMA_EN": 0.25}, False),
    ("sigma_en_high", {"SIGMA_EN": 0.75}, False),
    ("armington_low", {"ARMINGTON_SCALE": 0.75}, False),
    ("armington_high", {"ARMINGTON_SCALE": 1.25}, False),
    ("carbon_75", {"CARBON_PRICE_EUR": 75.0}, False),
    ("carbon_125", {"CARBON_PRICE_EUR": 125.0}, False),
    ("cbam_direct_only", {}, True),
    ("tstar_2028", {"T_STAR": 7.0}, False),
    ("tstar_2030", {"T_STAR": 9.0}, False),
    ("adj_cost_low", {"adjsh": 0.015}, False),
    ("adj_cost_high", {"adjsh": 0.060}, False),
    ("adj_cost_higher", {"adjsh": 0.100}, False),
)


def _family(central: str, kappa_prefix: str, sens_prefix: str,
            spec_version: float = 2.0) -> tuple[RunSpec, ...]:
    """One complete 16-run matrix: central + three kappa points + 12 variants."""
    specs = [RunSpec(central, "central", spec_version=spec_version)]
    for k, tag in ((0.25, "k25"), (0.50, "k50"), (0.75, "k75")):
        specs.append(RunSpec(f"{kappa_prefix}_{tag}", "kappa", kappa=k, spec_version=spec_version))
    for name, overrides, direct_only in _SENSITIVITY_VARIANTS:
        specs.append(
            RunSpec(
                f"{sens_prefix}_{name}",
                "sensitivity",
                scalar_overrides=overrides,
                direct_only=direct_only,
                spec_version=spec_version,
            )
        )
    return tuple(specs)


#: The matrix the current manuscript reports.  Anything published must be here.
PUBLISHED_SPECS = _family("newblk", "newblk", "sens_newblk")

#: Superseded calibration, retained so old results stay auditable.  These must
#: never be mixed with PUBLISHED_SPECS in a summary or a paper asset.
HISTORICAL_SPECS = _family("v3eubm_legind", "eubm", "sens_eubm", spec_version=1.0)

RUN_SPECS = PUBLISHED_SPECS + HISTORICAL_SPECS
PUBLISHED_SUFFIXES = frozenset(spec.suffix for spec in PUBLISHED_SPECS)

SPEC_BY_SUFFIX = {spec.suffix: spec for spec in RUN_SPECS}


BASE_SCALARS = {
    "g": 0.02,
    "rss": 0.05,
    "ro": 0.05,
    "adjsh": 0.05,
    "rp": 0.0,
    "gamma": 2.0,
    "ARMINGTON_SCALE": 1.0,
    "USE_BASE_TAX_WEDGES": 1.0,
    "CBAM_INDIRECT_LEGACY_BOOLEAN": 0.0,
    "CBAM_INDIRECT_SECTOR_OVERRIDE": 1.0,
    "CBAM_INCLUDE_PROCESS_EMISSIONS": 1.0,
    "CBAM_EU_BENCHMARK_ENDOGENOUS": 1.0,
    "CBAM_FLOOR_EPS": 1e-8,
    "CBAM_START_YEAR": float(CBAM_START_YEAR),
    "CBAM_START_T": float(CBAM_START_YEAR - BASE_YEAR + 1),
    "PATH_CONVERGENCE_TOL": 1e-7,
    "SOLVE_ITER_LIM": 10_000.0,
    "CARBON_PRICE_EUR": 100.0,
    "T_STAR": 8.0,
    "T_MAX": float(T_MAX),
    "BASE_YEAR": float(BASE_YEAR),
    "SCALE": 1_000.0,
    "SIGMA_VAE": 0.5,
    "SIGMA_EN": 0.5,
    "SIGMA_ELC": 3.0,
    "SCRAP_MULT": 0.0,
    "SCRAP_MARGIN": 0.02,
    "CREDIBILITY_INDEX_FROM_LEGACY_ALIAS": 0.0,
}


class ValidationAbort(RuntimeError):
    """Stop a dependent check after its prerequisites have failed."""


#: Economic-specification version the published matrix must carry.  Must match
#: MODEL_SPEC_VERSION in sinretencion.py.  Bumping the model without bumping
#: this constant (or vice versa) is what this gate is here to catch.
PUBLISHED_MODEL_SPEC_VERSION = 2.0


class RunValidator:
    """Collect all validation failures for one result suffix."""

    def __init__(self, spec: RunSpec, results_dir: Path) -> None:
        self.spec = spec
        self.results_dir = results_dir
        self.errors: list[str] = []
        self.notes: list[str] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.coverage_ok = False
        self.benchmark_identity_ok = False
        self.deferred_legacy_boolean: tuple[float, float] | None = None
        self.paths = {
            name: results_dir / pattern.format(suffix=spec.suffix)
            for name, pattern in RESULT_FILES.items()
        }

    def error(self, message: str) -> None:
        self.errors.append(message)

    def expected_scalars(self) -> dict[str, float]:
        expected = dict(BASE_SCALARS)
        expected.update(self.spec.scalar_overrides)
        t_star = int(expected["T_STAR"])
        expected.update(
            {
                "T_STAR_YEAR": float(BASE_YEAR + t_star - 1),
                "T_MAX_P2": float(T_MAX - t_star + 1),
                "CREDIBILITY_INDEX": self.spec.kappa,
                "CREDIBILITY_P": self.spec.kappa,
                "ANTICIPATED_WITHDRAWAL_ALIAS_ACTIVE": (
                    1.0 if math.isclose(self.spec.kappa, 0.0, abs_tol=1e-12) else 0.0
                ),
                "CBAM_INCLUDE_INDIRECT_ELECTRICITY": 0.0 if self.spec.direct_only else 1.0,
                "CBAM_INDIRECT_SECTOR_COUNT": 0.0 if self.spec.direct_only else 2.0,
            }
        )
        return expected

    def complete_family(self) -> bool:
        return all(path.is_file() and path.stat().st_size > 0 for path in self.paths.values())

    def validate(self) -> bool:
        if not self._load_tables():
            return False
        self._check_integrity()
        # Dependent checks assume the declared schemas and numeric value
        # columns are usable.  A corrupt table should produce a validation
        # failure, never an unrelated traceback.
        if self.errors:
            return False
        self._check_result_horizons()
        self._check_welfare_equilibria()
        self._check_exact_parameters()
        self._check_solver_diagnostics()
        errors_before_coverage = len(self.errors)
        self._check_coverage()
        self.coverage_ok = len(self.errors) == errors_before_coverage
        self._check_policy_paths()
        self._check_endogenous_benchmark_identity()
        self._check_provenance()
        self._finalize_legacy_boolean()
        return not self.errors

    def _check_provenance(self) -> None:
        """Refuse a family solved under a different economic specification.

        Completeness ("the five parquets exist") cannot distinguish a family
        solved under the current formulation from one left over from an older
        one, which is exactly how a resumed campaign silently mixes vintages.
        """
        params = self.tables.get("parameters")
        if params is None:
            return
        expected = float(self.spec.spec_version)
        rows = params.loc[params["param"] == "provenance"]
        if rows.empty:
            if expected < 2.0:
                # The superseded matrix was solved before stamping existed; its
                # absence is the expected signature of that vintage.
                self.notes.append("sin sello (calibracion historica, anterior al sellado)")
                return
            self.error(
                "sin sello de procedencia (parameters no contiene filas 'provenance'): "
                "resuelto antes del sellado; vuelva a resolver la familia o rellene el sello"
            )
            return
        if expected < 2.0:
            self.error(
                "familia historica con sello de la especificacion actual: "
                "el namespace historico no debe re-resolverse con el modelo publicado"
            )
            return
        stamped = rows.loc[rows["sec"] == "MODEL_SPEC_VERSION", "value"]
        if stamped.empty:
            self.error("sello de procedencia incompleto: falta MODEL_SPEC_VERSION")
            return
        version = float(stamped.iloc[0])
        if not math.isclose(version, expected, abs_tol=1e-9):
            self.error(
                f"especificacion economica {version:g} != {expected:g} esperada: "
                "la familia procede de otra formulacion del modelo"
            )
            return

        def _text(key: str) -> str:
            hit = rows.loc[rows["sec"] == key, "partner"]
            return str(hit.iloc[0]) if not hit.empty else "?"

        note = f"spec {version:g} ({_text('MODEL_SPEC_VERSION')}), datos {_text('input_file')}"
        if not rows.loc[rows["sec"] == "backfilled"].empty:
            note += "; sello anadido tras el solve"
        self.notes.append(note)

    def _load_tables(self) -> bool:
        missing = [
            path.name
            for path in self.paths.values()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            self.error("faltan o están vacíos: " + ", ".join(missing))
            return False

        for name, path in self.paths.items():
            try:
                if path.suffix.lower() == ".csv":
                    table = pd.read_csv(path)
                else:
                    table = pd.read_parquet(path)
            except Exception as exc:  # pragma: no cover - depends on corrupt input
                self.error(f"no se puede leer {path.name}: {exc}")
                continue
            if table.empty:
                self.error(f"{path.name} no contiene filas")
            self.tables[name] = table
        return len(self.tables) == len(self.paths)

    def _check_integrity(self) -> None:
        for name, table in self.tables.items():
            required = REQUIRED_COLUMNS[name]
            missing = sorted(set(required) - set(table.columns))
            if missing:
                self.error(f"{name}: faltan columnas {missing}")
                continue

            if name == "welfare":
                key_columns = ["scope", "reg", "eq"]
                numeric_columns = list(WELFARE_NUMERIC_COLUMNS)
            else:
                key_columns = [column for column in required if column != "value"]
                numeric_columns = ["value"]

            normalized_keys = table[key_columns].copy()
            for column in key_columns:
                if pd.api.types.is_numeric_dtype(normalized_keys[column]):
                    normalized_keys[column] = normalized_keys[column].fillna(-9.87654321e99)
                else:
                    normalized_keys[column] = (
                        normalized_keys[column].fillna("<NA>").astype(str)
                    )
            duplicate_count = int(normalized_keys.duplicated(keep=False).sum())
            if duplicate_count:
                self.error(f"{name}: {duplicate_count} filas con claves duplicadas")

            for column in numeric_columns:
                values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
                bad_count = int((~np.isfinite(values)).sum())
                if bad_count:
                    self.error(f"{name}.{column}: {bad_count} valores no finitos")

    def _check_result_horizons(self) -> None:
        expected_eq = set(EQUILIBRIA)
        expected_t = set(EXPECTED_T)
        expected_years = set(EXPECTED_YEARS)
        for name in ("results_2d", "results_3d", "results_4d"):
            table = self.tables[name]
            if not set(REQUIRED_COLUMNS[name]).issubset(table.columns):
                continue

            actual_eq = set(table["eq"].astype(str).unique())
            if actual_eq != expected_eq:
                self.error(
                    f"{name}: equilibrios {sorted(actual_eq)}, esperados {sorted(expected_eq)}"
                )

            t_values = pd.to_numeric(table["T"], errors="coerce")
            year_values = pd.to_numeric(table["year"], errors="coerce")
            if t_values.isna().any() or not np.allclose(t_values, np.round(t_values)):
                self.error(f"{name}: T contiene valores ausentes o no enteros")
                continue
            if year_values.isna().any() or not np.allclose(year_values, np.round(year_values)):
                self.error(f"{name}: year contiene valores ausentes o no enteros")
                continue
            if set(t_values.astype(int).unique()) != expected_t:
                self.error(f"{name}: el horizonte T no es exactamente 1–{T_MAX}")
            if set(year_values.astype(int).unique()) != expected_years:
                self.error(
                    f"{name}: el horizonte anual no es exactamente {BASE_YEAR}–{BASE_YEAR + T_MAX - 1}"
                )
            mapping_bad = year_values.astype(int) != BASE_YEAR + t_values.astype(int) - 1
            if bool(mapping_bad.any()):
                self.error(f"{name}: hay {int(mapping_bad.sum())} filas con year != BASE_YEAR + T - 1")

            cell_columns = [
                column
                for column in REQUIRED_COLUMNS[name]
                if column not in {"T", "year", "value"}
            ]
            grouped = table.groupby(cell_columns, dropna=False, sort=False)["T"].agg(
                size="size", nunique="nunique", minimum="min", maximum="max"
            )
            bad = grouped[
                (grouped["size"] != T_MAX)
                | (grouped["nunique"] != T_MAX)
                | (grouped["minimum"] != 1)
                | (grouped["maximum"] != T_MAX)
            ]
            if not bad.empty:
                example = bad.index[0]
                self.error(
                    f"{name}: {len(bad)} celdas no tienen los {T_MAX} periodos; ejemplo={example}"
                )

    def _check_welfare_equilibria(self) -> None:
        welfare = self.tables["welfare"]
        if not {"scope", "reg", "eq"}.issubset(welfare.columns):
            return
        expected_eq = set(EQUILIBRIA)
        if set(welfare["eq"].astype(str).unique()) != expected_eq:
            self.error("welfare: el conjunto global de equilibrios no coincide con los seis requeridos")
        bad_groups: list[tuple[str, str]] = []
        for (scope, region), group in welfare.groupby(["scope", "reg"], dropna=False):
            if set(group["eq"].astype(str)) != expected_eq or len(group) != len(EQUILIBRIA):
                bad_groups.append((str(scope), str(region)))
        if bad_groups:
            self.error(
                f"welfare: {len(bad_groups)} agregados no contienen exactamente los seis equilibrios; "
                f"ejemplo={bad_groups[0]}"
            )

        sam_regions = set(
            self.tables["sam_data"].loc[
                self.tables["sam_data"]["var"].eq("xd"), "reg"
            ].astype(str)
        )
        welfare_regions = set(
            welfare.loc[welfare["scope"].eq("region"), "reg"].astype(str)
        )
        if welfare_regions != sam_regions:
            self.error(
                "welfare: las regiones individuales no coinciden con las regiones de sam_data"
            )

    def _scalar(self, name: str, *, required: bool = True) -> float | None:
        parameters = self.tables["parameters"]
        subset = parameters.loc[
            parameters["param"].eq(name)
            & parameters["reg"].fillna("").eq("")
            & parameters["sec"].fillna("").eq("")
            & parameters["partner"].fillna("").eq("")
            & parameters["T"].isna()
            & parameters["year"].isna()
        ]
        if len(subset) != 1:
            if required:
                self.error(f"parameters: {name} debe tener una única fila escalar; tiene {len(subset)}")
            return None
        return float(subset["value"].iloc[0])

    @staticmethod
    def _close(actual: float, expected: float, *, atol: float = 1e-12) -> bool:
        return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol)

    def _check_exact_parameters(self) -> None:
        expected = self.expected_scalars()
        for name, expected_value in expected.items():
            actual = self._scalar(name)
            if actual is not None and not self._close(actual, expected_value):
                if (
                    name == "CBAM_INDIRECT_LEGACY_BOOLEAN"
                    and self._close(actual, 1.0)
                    and self._close(expected_value, 0.0)
                    and not self.spec.direct_only
                ):
                    # The explicit sector list has priority in the model.  This
                    # mismatch can only be downgraded after coverage and the
                    # reconstructed benchmark both prove it was inert.
                    self.deferred_legacy_boolean = (actual, expected_value)
                    continue
                self.error(f"parameters: {name}={actual:.12g}; esperado {expected_value:.12g}")

        parameters = self.tables["parameters"]
        sectors = set(
            self.tables["sam_data"].loc[
                self.tables["sam_data"]["var"].eq("xd"), "sec"
            ].astype(str)
        )
        armington_scale = expected["ARMINGTON_SCALE"]
        for param, baseline in (("sigmaA", BASE_SIGMA_A), ("sigmaM", BASE_SIGMA_M)):
            rows = parameters.loc[parameters["param"].eq(param)]
            actual_sectors = set(rows["sec"].astype(str))
            if actual_sectors != sectors:
                self.error(f"parameters: {param} no contiene exactamente todos los sectores")
                continue
            for sector in sectors:
                if sector not in baseline:
                    self.error(f"parameters: no hay elasticidad base declarada para {sector}")
                    continue
                values = rows.loc[rows["sec"].eq(sector), "value"]
                if len(values) != 1:
                    self.error(f"parameters: {param}[{sector}] tiene {len(values)} filas")
                    continue
                expected_value = baseline[sector] * armington_scale
                actual = float(values.iloc[0])
                if not self._close(actual, expected_value):
                    self.error(
                        f"parameters: {param}[{sector}]={actual:.12g}; esperado {expected_value:.12g}"
                    )

    def _diagnostic_map(self, name: str, *, required: bool = True) -> dict[str, float] | None:
        parameters = self.tables["parameters"]
        rows = parameters.loc[parameters["param"].eq(name)]
        if rows.empty and not required:
            return None
        partners = set(rows["partner"].astype(str))
        if len(rows) != len(EQUILIBRIA) or partners != set(EQUILIBRIA):
            self.error(
                f"parameters: {name} debe tener una fila para cada uno de los seis equilibrios"
            )
            return None
        if (
            rows["reg"].fillna("").ne("").any()
            or rows["sec"].fillna("").ne("").any()
            or rows["T"].notna().any()
            or rows["year"].notna().any()
        ):
            self.error(f"parameters: {name} tiene dimensiones inesperadas")
        return dict(zip(rows["partner"].astype(str), rows["value"].astype(float)))

    def _check_solver_diagnostics(self) -> None:
        optimal = self._diagnostic_map("solve_optimal_global")
        maximum = self._diagnostic_map("solve_max_infeasibility")
        mean = self._diagnostic_map("solve_mean_infeasibility", required=False)
        count = self._diagnostic_map("solve_num_infeasibilities", required=False)
        path_tol = self._scalar("PATH_CONVERGENCE_TOL")
        if optimal is not None:
            bad = [eq for eq, value in optimal.items() if not self._close(value, 1.0)]
            if bad:
                self.error("solve_optimal_global != 1 en: " + ", ".join(sorted(bad)))
        if maximum is not None and path_tol is not None:
            bad = [
                (eq, value)
                for eq, value in maximum.items()
                if value < 0.0 or value > path_tol + 1e-15
            ]
            if bad:
                shown = ", ".join(f"{eq}={value:.3g}" for eq, value in bad)
                self.error(
                    f"solve_max_infeasibility supera PATH_CONVERGENCE_TOL={path_tol:.3g}: {shown}"
                )
        if mean is not None and maximum is not None:
            bad = [
                eq
                for eq in EQUILIBRIA
                if mean[eq] < 0.0 or mean[eq] > maximum[eq] + 1e-15
            ]
            if bad:
                self.error("solve_mean_infeasibility es incoherente en: " + ", ".join(bad))
        if count is not None:
            bad = [
                eq
                for eq, value in count.items()
                if value < 0.0 or not self._close(value, round(value))
            ]
            if bad:
                self.error("solve_num_infeasibilities no es entero/no negativo en: " + ", ".join(bad))

    def _check_coverage(self) -> None:
        parameters = self.tables["parameters"]
        covered = parameters.loc[parameters["param"].eq("cbam_covered")].copy()
        if "CHM" in set(covered["sec"].astype(str)):
            self.error("cbam_covered contiene CHM, excluido de esta ronda")
        actual_covered = set(covered.loc[covered["value"].gt(0.5), "sec"].astype(str))
        if actual_covered != set(COVERED_SECTORS) or len(covered) != len(COVERED_SECTORS):
            self.error(
                f"cbam_covered={sorted(actual_covered)}; esperado {sorted(COVERED_SECTORS)} sin filas extra"
            )
        if not np.allclose(covered["value"].astype(float), 1.0):
            self.error("cbam_covered contiene valores distintos de 1")

        sectors = set(
            self.tables["sam_data"].loc[
                self.tables["sam_data"]["var"].eq("xd"), "sec"
            ].astype(str)
        )
        indirect = parameters.loc[parameters["param"].eq("cbam_indirect_covered")]
        if len(indirect) != len(sectors) or set(indirect["sec"].astype(str)) != sectors:
            self.error("cbam_indirect_covered debe tener exactamente una fila por sector")
        expected_indirect = set() if self.spec.direct_only else set(CENTRAL_INDIRECT_SECTORS)
        actual_indirect = set(indirect.loc[indirect["value"].gt(0.5), "sec"].astype(str))
        if actual_indirect != expected_indirect:
            self.error(
                f"cbam_indirect_covered={sorted(actual_indirect)}; esperado {sorted(expected_indirect)}"
            )
        non_binary = ~np.isclose(indirect["value"].astype(float), 0.0) & ~np.isclose(
            indirect["value"].astype(float), 1.0
        )
        if bool(non_binary.any()):
            self.error("cbam_indirect_covered contiene valores no binarios")

        intensity = parameters.loc[parameters["param"].eq("cbam_indirect_elc_intensity")]
        nonzero_sectors = set(
            intensity.loc[intensity["value"].abs().gt(1e-14), "sec"].astype(str)
        )
        if not nonzero_sectors.issubset(expected_indirect):
            self.error(
                "cbam_indirect_elc_intensity es no nula fuera de la cobertura esperada: "
                + ", ".join(sorted(nonzero_sectors - expected_indirect))
            )
        if not self.spec.direct_only:
            missing_positive = expected_indirect - nonzero_sectors
            if missing_positive:
                self.error(
                    "cbam_indirect_elc_intensity no contiene intensidad positiva para: "
                    + ", ".join(sorted(missing_positive))
                )

        xd_rows = self.tables["sam_data"].loc[self.tables["sam_data"]["var"].eq("xd")]
        actual_eu = set(reg for reg in xd_rows["reg"].astype(str).unique() if reg.startswith("EU"))
        if actual_eu != set(EU_REGIONS):
            self.error(f"regiones UE={sorted(actual_eu)}; esperadas {sorted(EU_REGIONS)}")

    @staticmethod
    def _legislated_fa(period: int) -> float:
        return FA_SCHEDULE.get(period, 0.0)

    def _expected_policy(self, scenario: str, period: int) -> tuple[float, float]:
        t_star = int(self.expected_scalars()["T_STAR"])
        legislated = self._legislated_fa(period)
        with_cbam = True
        scale = 1.0

        if scenario == "Eq1_NoPol":
            fa = 1.0
        elif scenario in {"Eq2_Credible", "Eq4_Surprise"}:
            fa = legislated
        elif scenario == "Eq2_NoCBAM":
            fa = legislated
            with_cbam = False
        elif scenario in {"Eq3_Reneg", ANTICIPATED_WITHDRAWAL}:
            fa = legislated if period < t_star else 1.0
            if scenario == ANTICIPATED_WITHDRAWAL and period >= t_star:
                scale = self.spec.kappa
        elif scenario == "Eq4_Phase1":
            fa = (
                legislated
                if period < t_star
                else self.spec.kappa * legislated + (1.0 - self.spec.kappa)
            )
            if period >= t_star:
                scale = self.spec.kappa
        else:  # pragma: no cover - guarded by the expected scenario set
            raise KeyError(scenario)

        year = BASE_YEAR + period - 1
        cbam_on = float(with_cbam and year >= CBAM_START_YEAR and fa < 1.0 - 1e-12)
        if cbam_on:
            cbam_on *= scale
        return fa, cbam_on

    def _policy_series(self, parameter: str, partner: str) -> pd.Series | None:
        parameters = self.tables["parameters"]
        rows = parameters.loc[
            parameters["param"].eq(parameter) & parameters["partner"].eq(partner)
        ].copy()
        if len(rows) != T_MAX:
            self.error(f"{parameter}[{partner}] tiene {len(rows)} filas; esperadas {T_MAX}")
            return None
        t_values = pd.to_numeric(rows["T"], errors="coerce")
        year_values = pd.to_numeric(rows["year"], errors="coerce")
        if (
            t_values.isna().any()
            or year_values.isna().any()
            or not np.allclose(t_values, np.round(t_values))
            or not np.allclose(year_values, np.round(year_values))
        ):
            self.error(f"{parameter}[{partner}] tiene T/year ausentes o no enteros")
            return None
        if set(t_values.astype(int)) != set(EXPECTED_T):
            self.error(f"{parameter}[{partner}] no cubre exactamente T=1–{T_MAX}")
            return None
        if not np.array_equal(
            year_values.astype(int).to_numpy(),
            (BASE_YEAR + t_values.astype(int) - 1).to_numpy(),
        ):
            self.error(f"{parameter}[{partner}] tiene una correspondencia T/year incorrecta")
            return None
        if rows["reg"].fillna("").ne("").any() or rows["sec"].fillna("").ne("").any():
            self.error(f"{parameter}[{partner}] tiene dimensiones reg/sec inesperadas")
            return None
        if t_values.duplicated().any():
            self.error(f"{parameter}[{partner}] tiene periodos duplicados")
            return None
        rows["T"] = t_values.astype(int)
        return rows.set_index("T")["value"].astype(float).sort_index()

    def _check_policy_paths(self) -> None:
        parameters = self.tables["parameters"]
        schedule = parameters.loc[parameters["param"].eq("FA_schedule")].copy()
        if len(schedule) != T_MAX:
            self.error(f"FA_schedule tiene {len(schedule)} filas; esperadas {T_MAX}")
        else:
            t_values = pd.to_numeric(schedule["T"], errors="coerce")
            year_values = pd.to_numeric(schedule["year"], errors="coerce")
            valid_timing = (
                not t_values.isna().any()
                and not year_values.isna().any()
                and np.allclose(t_values, np.round(t_values))
                and np.allclose(year_values, np.round(year_values))
                and set(t_values.astype(int)) == set(EXPECTED_T)
                and np.array_equal(
                    year_values.astype(int).to_numpy(),
                    (BASE_YEAR + t_values.astype(int) - 1).to_numpy(),
                )
            )
            if not valid_timing:
                self.error("FA_schedule no cubre correctamente T=1–30 / 2022–2051")
            else:
                schedule["T"] = t_values.astype(int)
                schedule = schedule.set_index("T").sort_index()
                differences = [
                    abs(float(schedule.at[t, "value"]) - self._legislated_fa(t))
                    for t in EXPECTED_T
                ]
                if max(differences, default=0.0) > 1e-12:
                    self.error("FA_schedule no coincide con la senda legislada 2022–2051")

        policy_parameters = (
            "scenario_fa_path",
            "scenario_domestic_carbon_factor",
            "scenario_cbam_on",
            "scenario_cbam_benchmark_weight",
        )
        expected_partners = set(EQUILIBRIA)
        if math.isclose(self.spec.kappa, 0.0, abs_tol=1e-12):
            expected_partners.add(ANTICIPATED_WITHDRAWAL)
        for parameter in policy_parameters:
            rows = parameters.loc[parameters["param"].eq(parameter)]
            actual_partners = set(rows["partner"].astype(str))
            if actual_partners != expected_partners:
                self.error(
                    f"{parameter}: escenarios {sorted(actual_partners)}, esperados {sorted(expected_partners)}"
                )

        cached: dict[tuple[str, str], pd.Series] = {}
        for scenario in sorted(expected_partners):
            for parameter in policy_parameters:
                series = self._policy_series(parameter, scenario)
                if series is not None:
                    cached[(parameter, scenario)] = series
            if not all((parameter, scenario) in cached for parameter in policy_parameters):
                continue

            expected_fa = np.array(
                [self._expected_policy(scenario, t)[0] for t in EXPECTED_T], dtype=float
            )
            expected_on = np.array(
                [self._expected_policy(scenario, t)[1] for t in EXPECTED_T], dtype=float
            )
            actual_fa = cached[("scenario_fa_path", scenario)].reindex(EXPECTED_T).to_numpy()
            actual_domestic = cached[
                ("scenario_domestic_carbon_factor", scenario)
            ].reindex(EXPECTED_T).to_numpy()
            actual_on = cached[("scenario_cbam_on", scenario)].reindex(EXPECTED_T).to_numpy()
            actual_weight = cached[
                ("scenario_cbam_benchmark_weight", scenario)
            ].reindex(EXPECTED_T).to_numpy()

            checks = (
                ("FA", actual_fa, expected_fa),
                ("factor doméstico", actual_domestic, 1.0 - expected_fa),
                ("multiplicador CBAM", actual_on, expected_on),
                ("peso benchmark", actual_weight, actual_on * actual_fa),
            )
            for label, actual, expected in checks:
                differences = np.abs(actual - expected)
                if not np.isfinite(differences).all() or differences.max(initial=0.0) > 1e-12:
                    worst = int(np.nanargmax(differences)) if np.isfinite(differences).any() else 0
                    self.error(
                        f"política {scenario}: {label} no coincide en T={EXPECTED_T[worst]} "
                        f"(actual={actual[worst]:.12g}, esperado={expected[worst]:.12g})"
                    )

        alias_rows = parameters.loc[parameters["param"].eq("scenario_alias")]
        if math.isclose(self.spec.kappa, 0.0, abs_tol=1e-12):
            alias_ok = (
                len(alias_rows) == 1
                and str(alias_rows.iloc[0]["sec"]) == ANTICIPATED_WITHDRAWAL
                and str(alias_rows.iloc[0]["partner"]) == "Eq4_Phase1"
                and self._close(float(alias_rows.iloc[0]["value"]), 1.0)
            )
            if not alias_ok:
                self.error("scenario_alias debe mapear AnticipatedWithdrawal -> Eq4_Phase1 con kappa=0")
        elif not alias_rows.empty:
            self.error("scenario_alias no debe existir para valores interiores de kappa")

        # The stranded-capital comparison uses this realized-policy identity
        # only in the central kappa=0 run.  It must not compare outcome paths.
        if self.spec.kind == "central" and math.isclose(self.spec.kappa, 0.0, abs_tol=1e-12):
            for parameter in policy_parameters:
                reneging = cached.get((parameter, "Eq3_Reneg"))
                anticipated = cached.get((parameter, ANTICIPATED_WITHDRAWAL))
                if reneging is None or anticipated is None:
                    continue
                if not np.allclose(reneging.to_numpy(), anticipated.to_numpy(), rtol=0.0, atol=1e-12):
                    self.error(f"central: RN y AW no comparten la misma política realizada ({parameter})")

    def _indexed_series(
        self,
        table_name: str,
        selector_column: str,
        selector_value: str,
        index_columns: Iterable[str],
    ) -> pd.Series:
        table = self.tables[table_name]
        rows = table.loc[table[selector_column].eq(selector_value)]
        if rows.empty:
            self.error(f"{table_name}: falta {selector_column}={selector_value}")
            raise ValidationAbort
        series = rows.set_index(list(index_columns))["value"].astype(float)
        if series.index.has_duplicates:
            self.error(f"{table_name}: {selector_value} tiene claves duplicadas")
            raise ValidationAbort
        return series

    def _check_endogenous_benchmark_identity(self) -> None:
        if self._scalar("CBAM_EU_BENCHMARK_ENDOGENOUS") != 1.0:
            return
        try:
            benchmark = self._indexed_series(
                "results_2d", "var", "EU_CBAM_BM", ("eq", "sec", "T")
            )
            h_xd = self._indexed_series(
                "results_3d", "var", "H_XD", ("eq", "reg", "sec", "T")
            )
            h_en = self._indexed_series(
                "results_3d", "var", "H_EN", ("eq", "reg", "sec", "T")
            )
            h_pe = self._indexed_series(
                "results_3d", "var", "H_PE", ("eq", "reg", "sec", "T")
            )
            h_p = self._indexed_series(
                "results_3d", "var", "H_P", ("eq", "reg", "sec", "T")
            )
            process_intensity = self._indexed_series(
                "parameters", "param", "emis_proc_intensity", ("reg", "sec")
            )
            fuel_per_ref = self._indexed_series(
                "parameters", "param", "emis_fuel_per_ref", ("reg", "sec")
            )
            indirect_intensity = self._indexed_series(
                "parameters", "param", "cbam_indirect_elc_intensity", ("reg", "sec")
            )
            fa = self._indexed_series(
                "parameters", "param", "scenario_fa_path", ("partner", "T")
            )
            base_xd = self._indexed_series("sam_data", "var", "xd", ("reg", "sec"))
            base_ref = self._indexed_series(
                "sam_data", "var", "en_ref", ("reg", "sec")
            )
        except ValidationAbort:
            return

        sectors = sorted(set(base_xd.index.get_level_values("sec")))
        expected_index = pd.MultiIndex.from_product(
            [EQUILIBRIA, sectors, EXPECTED_T], names=["eq", "sec", "T"]
        )
        if len(benchmark) != len(expected_index) or set(benchmark.index) != set(expected_index):
            self.error("EU_CBAM_BM no contiene exactamente equilibrio × sector × 2022–2051")
            return

        sigma_en = self._scalar("SIGMA_EN")
        carbon_price_eur = self._scalar("CARBON_PRICE_EUR")
        scale = self._scalar("SCALE")
        path_tol = self._scalar("PATH_CONVERGENCE_TOL")
        if None in {sigma_en, carbon_price_eur, scale, path_tol}:
            return
        assert sigma_en is not None
        assert carbon_price_eur is not None
        assert scale is not None
        assert path_tol is not None
        carbon_price_internal = carbon_price_eur / scale
        covered = set(COVERED_SECTORS)

        worst_error = -1.0
        worst_key: tuple[str, str, int] | None = None
        worst_actual = math.nan
        worst_rebuilt = math.nan
        try:
            for (equilibrium, sector, period), actual_value in benchmark.items():
                period_int = int(period)
                numerator = 0.0
                denominator = 0.0
                for region in EU_REGIONS:
                    output = float(base_xd.at[(region, sector)]) * float(
                        h_xd.at[(equilibrium, region, sector, period)]
                    )
                    tau_fuel = 0.0
                    if sector in covered:
                        tau_fuel = (
                            (1.0 - float(fa.at[(equilibrium, float(period))]))
                            * carbon_price_internal
                            * float(fuel_per_ref.at[(region, sector)])
                        )
                    ref_price = float(h_p.at[(equilibrium, region, "REF", period)]) + tau_fuel
                    if ref_price <= 0.0:
                        self.error(
                            f"benchmark: precio REF no positivo en {(equilibrium, region, sector, period_int)}"
                        )
                        return
                    ref_hat = float(h_en.at[(equilibrium, region, sector, period)]) * (
                        float(h_pe.at[(equilibrium, region, sector, period)]) / ref_price
                    ) ** sigma_en
                    numerator += (
                        float(process_intensity.at[(region, sector)])
                        + float(indirect_intensity.at[(region, sector)])
                    ) * output
                    numerator += (
                        float(fuel_per_ref.at[(region, sector)])
                        * float(base_ref.at[(region, sector)])
                        * ref_hat
                    )
                    denominator += output
                if denominator <= 0.0:
                    self.error(f"benchmark: denominador UE no positivo para {(sector, period_int)}")
                    return
                rebuilt = numerator / denominator
                error = abs(float(actual_value) - rebuilt)
                if error > worst_error:
                    worst_error = error
                    worst_key = (str(equilibrium), str(sector), period_int)
                    worst_actual = float(actual_value)
                    worst_rebuilt = rebuilt
        except KeyError as exc:
            self.error(f"benchmark: falta una celda necesaria para reconstruir la identidad: {exc}")
            return

        benchmark_tolerance = max(1e-9, path_tol)
        if worst_error > benchmark_tolerance:
            self.error(
                "EU_CBAM_BM incumple la identidad endógena: "
                f"máx.|guardado-reconstruido|={worst_error:.3g} > {benchmark_tolerance:.3g} "
                f"en {worst_key} (guardado={worst_actual:.12g}, reconstruido={worst_rebuilt:.12g})"
            )
        else:
            self.benchmark_identity_ok = True
            self.notes.append(
                f"benchmark reconstruido (error máximo {worst_error:.3g}, tolerancia {benchmark_tolerance:.3g})"
            )

    def _finalize_legacy_boolean(self) -> None:
        if self.deferred_legacy_boolean is None:
            return
        actual, expected = self.deferred_legacy_boolean
        override = self._scalar("CBAM_INDIRECT_SECTOR_OVERRIDE", required=False)
        indirect_count = self._scalar("CBAM_INDIRECT_SECTOR_COUNT", required=False)
        explicit_cem_fer = (
            override is not None
            and indirect_count is not None
            and self._close(override, 1.0)
            and self._close(indirect_count, 2.0)
            and self.coverage_ok
        )
        if explicit_cem_fer and self.benchmark_identity_ok and not self.spec.direct_only:
            self.notes.append(
                "ADVERTENCIA: CBAM_INDIRECT_LEGACY_BOOLEAN=1 es inerte porque la lista "
                "explícita y efectiva es exactamente CEM,FER"
            )
            return
        self.error(
            "parameters: CBAM_INDIRECT_LEGACY_BOOLEAN="
            f"{actual:.12g}; esperado {expected:.12g}; no se demostró que fuera inerte"
        )


def requested_specs(suffixes: list[str] | None) -> list[RunSpec]:
    if not suffixes:
        return list(RUN_SPECS)
    return [SPEC_BY_SUFFIX[suffix] for suffix in suffixes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida central, kappa y sensibilidades del lote EU-benchmark sin lanzar el modelo."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directorio de resultados (por defecto: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--available",
        action="store_true",
        help="Valida solo familias con los cinco parquet y welfare presentes y no vacíos.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        choices=tuple(SPEC_BY_SUFFIX),
        help="Limita la validación a este sufijo; puede repetirse.",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir.resolve()
    specs = requested_specs(args.suffix)
    skipped: list[str] = []
    if args.available:
        available_specs: list[RunSpec] = []
        for spec in specs:
            validator = RunValidator(spec, results_dir)
            if validator.complete_family():
                available_specs.append(spec)
            else:
                skipped.append(spec.suffix)
        specs = available_specs
        if skipped:
            print("[SKIP] familias incompletas: " + ", ".join(skipped))
        if not specs:
            print("[FALLO] --available no encontró ninguna familia completa.")
            return 1

    failures = 0
    for spec in specs:
        validator = RunValidator(spec, results_dir)
        ok = validator.validate()
        if ok:
            detail = "; ".join(validator.notes)
            print(f"[OK] {spec.suffix}" + (f": {detail}" if detail else ""))
        else:
            failures += 1
            print(f"[FALLO] {spec.suffix}")
            for error in validator.errors:
                print(f"  - {error}")

    if failures:
        print(f"\nResultado: {failures} de {len(specs)} familias fallaron la validación.")
        return 1
    print(f"\nResultado: {len(specs)} familias validadas correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
