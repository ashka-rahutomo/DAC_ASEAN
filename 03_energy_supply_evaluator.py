from __future__ import annotations

"""
03_energy_supply_evaluator.py

Energy-supply evaluator for ASEAN DAC TVSA post-processing.

Purpose
-------
This script connects annual process-side DAC demand from:

    02.TEA_LCOD/02_ANNUAL_DYNAMIC_OPERATION/annual_results/
    annual_operation_summary_by_province_policy.csv

to electricity/heat supply scenarios for downstream CCS, emission, and LCOD modules.

It evaluates scenario-level annual energy supply for each province and dynamic-operation
policy, including:

S0_grid_HP
    100% grid electricity; regeneration heat supplied by heat pump.

S1_grid_PVwind_HP
    Onsite PV/wind used first; no battery; residual electricity from grid; heat pump.

S2_PVwind_battery_grid_HP
    Onsite PV/wind + battery dispatch; residual electricity from grid; heat pump.

S3_grid_geothermalHeat
    Process electricity from grid; regeneration heat from geothermal heat.
    Valid only for geothermal-eligible provinces.

S4_PVwind_battery_grid_geothermalHeat
    Process electricity supplied by PV/wind + battery + residual grid;
    regeneration heat from geothermal heat. Valid only for geothermal-eligible provinces.

Notes
-----
1. PV/wind hourly profiles are treated as variable renewable supply per unit capacity:
   - pv_energy_kWh_per_kWp
   - wind_energy_kWh_per_kW
   The installed PV/wind capacities for DAC are sizing variables.

2. If hourly DAC demand from module 02 is unavailable, this script uses a flat hourly
   demand profile derived from annual process electricity and heat demand. This is a
   transparent approximation and is flagged in the output as demand_profile_method.

3. This module is not the final LCOD calculator. It exports energy supply, capacity
   sizing, dispatch shares, operational energy emissions, validity flags, and downstream LCOD inputs.
   Full annualized CAPEX/OPEX should be computed in the LCOD module.

Default placement
-----------------
Script:
    D:/Ashka/5.DAC/06.PYTHON/02.TEA_LCOD/03_ENERGY_SUPPLY_EVALUATOR/03_energy_supply_evaluator.py

Outputs:
    D:/Ashka/5.DAC/06.PYTHON/02.TEA_LCOD/03_ENERGY_SUPPLY_EVALUATOR/
"""

from pathlib import Path
import argparse
import json
import math
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ID_COLS = ["country_code", "country_name", "province_id", "province_name"]
COORD_COLS = ["longitude", "latitude"]

SCENARIO_BASES = [
    "S0_grid_HP",
    "S1_grid_PVwind_HP",
    "S2_PVwind_battery_grid_HP",
    "S3_grid_geothermalHeat",
    "S4_PVwind_battery_grid_geothermalHeat",
]


# =============================================================================
# Utilities
# =============================================================================

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def to_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
        val = float(x)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV robustly, including files with broken quote characters.

    Some manually edited CSVs contain long source/notes fields with unescaped quotes and
    trailing semicolons in headers. Pandas' normal CSV parser may then shift columns or
    raise ParserError. This reader tests several parsers and returns the candidate with
    the strongest column match, rather than the first parser that happens to return a
    DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    read_attempts = [
        # Normal auto-detection for clean comma/semicolon CSVs.
        dict(sep=None, engine="python", encoding="utf-8-sig"),
        # Robust mode: disable quote handling so unescaped quotes in source/notes
        # do not break the comma structure.
        dict(sep=",", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        dict(sep=";", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        # Standard fallbacks.
        dict(sep=",", encoding="utf-8-sig", index_col=False),
        dict(sep=";", encoding="utf-8-sig", index_col=False),
        dict(sep=",", encoding="latin1", index_col=False),
        dict(sep=";", encoding="latin1", index_col=False),
    ]

    candidates = []
    for opts in read_attempts:
        opts2 = opts.copy()
        opts2.update(kwargs)
        try:
            with warnings.catch_warnings():
                # Some deliberately defensive parsing attempts trigger harmless
                # ParserWarnings before the best candidate is selected.
                warnings.simplefilter("ignore")
                df = pd.read_csv(path, **opts2)
            if df is None or df.empty and len(df.columns) == 0:
                continue
            # Clean broken trailing semicolons in headers here so scoring sees the real names.
            df.columns = [re.sub(r";+$", "", str(c).strip().replace("\ufeff", "")) for c in df.columns]
            candidates.append((opts, df))
        except Exception:
            continue

    if not candidates:
        # Last attempt: raw line split by comma with quote handling disabled.
        return pd.read_csv(path, sep=",", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="skip", **kwargs)

    expected_tokens = [
        "country_code", "country_name", "province_name", "province_id",
        "grid_emission_factor", "emission_factor", "electricity_price",
        "grid_electricity_price", "annual_CO2", "operation_policy",
        "pv_energy", "wind_energy", "technology", "resource_metric",
    ]

    def score_df(df: pd.DataFrame) -> float:
        cols = [str(c).strip() for c in df.columns]
        lower_cols = [c.lower() for c in cols]
        n_cols = len(cols)
        n_unnamed = sum(c.lower().startswith("unnamed") for c in cols)
        token_score = sum(any(tok.lower() in c for c in lower_cols) for tok in expected_tokens)
        # Prefer parsers that produce many meaningful columns and known headers.
        # Penalize one-column semicolon/comma failures and many unnamed columns.
        return token_score * 1000 + n_cols * 10 - n_unnamed * 25

    best = max(candidates, key=lambda item: score_df(item[1]))[1]
    return best


def find_latest(patterns: list[str], roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            candidates.extend(root.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(text: str, default: list[float]) -> list[float]:
    if text is None or str(text).strip() == "":
        return default
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals if vals else default


def canonicalize_annual_operation(df: pd.DataFrame) -> pd.DataFrame:
    """Map annual-operation module output columns to canonical demand columns."""
    out = df.copy()

    required = ["country_code", "province_id", "province_name", "operation_policy"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Annual operation CSV missing required columns: {missing}")

    if "country_name" not in out.columns:
        out["country_name"] = out["country_code"]
    if "longitude" not in out.columns:
        out["longitude"] = np.nan
    if "latitude" not in out.columns:
        out["latitude"] = np.nan

    alias_map = {
        "annual_CO2_t_per_1000kgads": [
            "annual_CO2_t_per_1000kgads",
            "annual_tCO2_per_1000kgads",
            "annual_CO2_t_1000kgads",
        ],
        "annual_H2O_t_per_1000kgads": [
            "annual_H2O_t_per_1000kgads",
            "annual_tH2O_per_1000kgads",
        ],
        "annual_process_electricity_MWhe": [
            "annual_electricity_MWhe_per_1000kgads",
            "annual_E_total_el_MWhe_per_1000kgads",
            "annual_process_electricity_MWhe",
        ],
        "annual_heat_demand_MWhth": [
            "annual_heat_MWhth_per_1000kgads",
            "annual_Q_heat_MWhth_per_1000kgads",
            "annual_heat_demand_MWhth",
        ],
        "annual_cooling_demand_MWhth": [
            "annual_cooling_MWhth_per_1000kgads",
            "annual_Q_cool_MWhth_per_1000kgads",
            "annual_cooling_demand_MWhth",
        ],
        "SEC_heat_MWhth_tCO2": ["SEC_heat_MWhth_tCO2"],
        "SEC_el_MWhe_tCO2": ["SEC_el_MWhe_tCO2"],
        "SEC_total_MWh_tCO2_before_compression": ["SEC_total_MWh_tCO2_before_compression"],
        "H2O_CO2_mass_ratio_tH2O_tCO2": ["H2O_CO2_mass_ratio_tH2O_tCO2"],
    }

    for canonical, aliases in alias_map.items():
        if canonical in out.columns:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
            continue
        found = None
        for a in aliases:
            if a in out.columns:
                found = a
                break
        if found is not None:
            out[canonical] = pd.to_numeric(out[found], errors="coerce")
        else:
            out[canonical] = np.nan

    out["annual_total_electricity_demand_MWhe_HP_pending"] = np.nan
    out["province_key"] = make_province_key(out)
    return out


def make_province_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["country_code"].astype(str).map(norm_text)
        + "|"
        + df["province_id"].astype(str).map(norm_text)
        + "|"
        + df["province_name"].astype(str).map(norm_text)
    )


def simple_key(country_code, province_id, province_name) -> str:
    return f"{norm_text(country_code)}|{norm_text(province_id)}|{norm_text(province_name)}"


def country_province_name_key(country_code, province_name) -> str:
    return f"{norm_text(country_code)}|{norm_text(province_name)}"


# =============================================================================
# Grid and geothermal resource loading
# =============================================================================

def _asean_country_code_from_name(name) -> str:
    """Map common ASEAN country names to ISO3 codes when CSV source uses names only."""
    n = norm_text(name)
    mapping = {
        "brunei": "BRN",
        "brunei_darussalam": "BRN",
        "cambodia": "KHM",
        "kampuchea": "KHM",
        "indonesia": "IDN",
        "laos": "LAO",
        "lao_pdr": "LAO",
        "lao_peoples_democratic_republic": "LAO",
        "malaysia": "MYS",
        "myanmar": "MMR",
        "burma": "MMR",
        "philippines": "PHL",
        "the_philippines": "PHL",
        "singapore": "SGP",
        "thailand": "THA",
        "viet_nam": "VNM",
        "vietnam": "VNM",
    }
    return mapping.get(n, "")


def _first_valid(series: pd.Series):
    for x in series:
        if pd.isna(x):
            continue
        if isinstance(x, str) and x.strip() == "":
            continue
        return x
    return np.nan


def _find_numeric_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    for a in aliases:
        if a in cols:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _canonicalize_grid_table(df: pd.DataFrame, path: Path, hint: str = "generic") -> pd.DataFrame:
    """Canonicalize grid input or source-registry CSV into the structure required by attach_grid().

    hint='emission' allows a generic numeric column named value/central_value to map to EF.
    hint='price' allows a generic numeric column named value/central_value to map to price.
    """
    df = df.copy()
    df.columns = [re.sub(r";+$", "", str(c).strip().replace("\ufeff", "")) for c in df.columns]

    # Canonical identity columns.
    id_aliases = {
        "country_code": ["country_code", "iso3", "ISO3", "country_iso3", "code", "country_id"],
        "country_name": ["country_name", "country", "COUNTRY", "name_0", "NAME_0", "market", "country_or_area"],
        "province_id": ["province_id", "prov_id", "adm1_id", "id", "GID_1", "gid_1"],
        "province_name": ["province_name", "prov_name", "province", "state", "adm1_name", "NAME_1", "name_1"],
    }
    out = pd.DataFrame(index=df.index)
    for c, aliases in id_aliases.items():
        found = _find_numeric_col(df, aliases) if False else None
        lower = {str(col).lower(): col for col in df.columns}
        for a in aliases:
            if a in df.columns:
                found = a
                break
            if a.lower() in lower:
                found = lower[a.lower()]
                break
        out[c] = df[found] if found is not None else np.nan

    # Clean country codes. If missing, derive from country_name for ASEAN.
    out["country_code"] = out["country_code"].map(
        lambda x: str(x).strip().strip('"').strip("'").upper() if not pd.isna(x) else ""
    )
    out["country_name"] = out["country_name"].map(
        lambda x: str(x).strip().strip('"').strip("'") if not pd.isna(x) else x
    )
    out["province_name"] = out["province_name"].map(
        lambda x: str(x).strip().strip('"').strip("'") if not pd.isna(x) else ""
    )
    out["province_id"] = out["province_id"].map(
        lambda x: str(x).strip().strip('"').strip("'") if not pd.isna(x) else ""
    )
    missing_cc = out["country_code"].astype(str).str.strip().eq("") | out["country_code"].astype(str).str.lower().isin(["nan", "none"])
    if missing_cc.any():
        out.loc[missing_cc, "country_code"] = out.loc[missing_cc, "country_name"].map(_asean_country_code_from_name)

    # If country_code accidentally contains full names, map those too.
    bad_code = ~out["country_code"].astype(str).str.match(r"^[A-Z]{3}$", na=False)
    if bad_code.any():
        mapped = out.loc[bad_code, "country_code"].map(_asean_country_code_from_name)
        out.loc[bad_code & mapped.astype(str).ne(""), "country_code"] = mapped[mapped.astype(str).ne("")]

    out["country_name"] = out["country_name"].where(out["country_name"].notna(), out["country_code"])
    out["province_id"] = out["province_id"].fillna("")
    out["province_name"] = out["province_name"].fillna("")

    # Numeric aliases.
    ef_aliases = [
        "grid_emission_factor_tCO2_MWh",
        "grid_emission_factor_tCO2_per_MWh",
        "grid_EF_tCO2_MWh",
        "EF_tCO2_MWh",
        "emission_factor_tCO2_MWh",
        "grid_emission_factor_kgCO2_kWh",
        "grid_emission_factor_kgCO2_per_kWh",
        "kgCO2_kWh",
        "tCO2_MWh",
    ]
    price_aliases = [
        "grid_electricity_price_USD_MWh",
        "electricity_price_USD_MWh",
        "industrial_electricity_price_USD_MWh",
        "industrial_tariff_USD_MWh",
        "tariff_USD_MWh",
        "price_USD_MWh",
        "usd_mwh",
        "USD_MWh",
    ]
    tariff_aliases = [
        "industrial_tariff_USD_MWh",
        "industrial_electricity_price_USD_MWh",
        "grid_electricity_price_USD_MWh",
        "tariff_USD_MWh",
        "price_USD_MWh",
    ]
    generic_numeric_aliases = ["value", "central_value", "mean_value", "grid_value", "numeric_value"]

    if hint == "emission":
        ef_aliases = ef_aliases + generic_numeric_aliases
    if hint == "price":
        price_aliases = price_aliases + generic_numeric_aliases
        tariff_aliases = tariff_aliases + generic_numeric_aliases

    numeric_targets = {
        "grid_emission_factor_tCO2_MWh": ef_aliases,
        "grid_electricity_price_USD_MWh": price_aliases,
        "industrial_tariff_USD_MWh": tariff_aliases,
        "lowcarbon_grid_emission_factor_tCO2_MWh": [
            "lowcarbon_grid_emission_factor_tCO2_MWh",
            "low_carbon_grid_emission_factor_tCO2_MWh",
            "ppa_emission_factor_tCO2_MWh",
        ],
        "solar_LCOE_USD_MWh": ["solar_LCOE_USD_MWh", "pv_LCOE_USD_MWh", "solar_pv_LCOE_USD_MWh"],
        "wind_LCOE_USD_MWh": ["wind_LCOE_USD_MWh", "onshore_wind_LCOE_USD_MWh"],
        "geothermal_heat_price_USD_MWhth": ["geothermal_heat_price_USD_MWhth", "heat_price_USD_MWhth"],
        "geothermal_heat_emission_factor_tCO2_MWhth": ["geothermal_heat_emission_factor_tCO2_MWhth", "heat_emission_factor_tCO2_MWhth"],
    }
    for target, aliases in numeric_targets.items():
        found = _find_numeric_col(df, aliases)
        if found is None:
            out[target] = np.nan
        else:
            out[target] = pd.to_numeric(df[found].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Text metadata.
    if "source_spatial_resolution" in df.columns:
        out["source_spatial_resolution"] = df["source_spatial_resolution"].fillna("unknown")
    elif "spatial_resolution" in df.columns:
        out["source_spatial_resolution"] = df["spatial_resolution"].fillna("unknown")
    else:
        # If province fields are empty, this is country-level data.
        is_country_level = out["province_name"].astype(str).str.strip().eq("") & out["province_id"].astype(str).str.strip().eq("")
        out["source_spatial_resolution"] = np.where(is_country_level, "country", "province_or_mixed")

    if "grid_data_confidence" in df.columns:
        out["grid_data_confidence"] = df["grid_data_confidence"].fillna("unknown")
    elif "data_confidence" in df.columns:
        out["grid_data_confidence"] = df["data_confidence"].fillna("unknown")
    elif "confidence" in df.columns:
        out["grid_data_confidence"] = df["confidence"].fillna("unknown")
    elif "confidence_level" in df.columns:
        out["grid_data_confidence"] = df["confidence_level"].fillna("unknown")
    else:
        out["grid_data_confidence"] = "unknown"

    out["source_file_grid"] = str(path)
    out["grid_table_hint"] = hint
    out["country_key"] = out["country_code"].map(norm_text)
    out["country_province_name_key"] = out.apply(lambda r: country_province_name_key(r["country_code"], r["province_name"]), axis=1)
    out["province_key"] = make_province_key(out)
    return out


def _select_grid_primary_file(grid_dir: Path | None, explicit_grid_csv: Path | None) -> Path | None:
    if explicit_grid_csv and explicit_grid_csv.exists():
        return explicit_grid_csv
    if explicit_grid_csv and not explicit_grid_csv.exists():
        raise FileNotFoundError(f"Explicit grid CSV not found: {explicit_grid_csv}")
    if grid_dir is None or not grid_dir.exists():
        return None

    # Prefer the merged file used by 03, then explicit final/updated variants, then starter.
    candidates = []
    preferred_names = [
        "province_energy_price_emission.csv",
        "province_energy_price_emission_FINAL.csv",
        "province_energy_price_emission_UPDATED.csv",
        "province_energy_price_emission_starter_UPDATED.csv",
        "province_energy_price_emission_starter.csv",
        "grid_energy_assumptions.csv",
    ]
    for name in preferred_names:
        p = grid_dir / name
        if p.exists():
            candidates.append(p)
    if candidates:
        return candidates[0]

    # Last fallback: newest plausible province-energy file.
    pats = ["*province*energy*price*emission*.csv", "*grid*assumption*.csv"]
    files = []
    for pat in pats:
        files.extend(grid_dir.glob(pat))
    files = [p for p in files if p.is_file()]
    if files:
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def load_grid_assumptions(grid_dir: Path | None, explicit_grid_csv: Path | None = None) -> pd.DataFrame:
    """Load province/country grid price and emission assumptions.

    Revised behavior:
    - Uses the merged province_energy_price_emission*.csv as the primary file.
    - Also reads grid_emission_factor_asean_sourced.csv and
      grid_electricity_price_asean_sourced.csv when present.
    - Fills missing country-level EF/price in the primary file from those source registries.
    - Keeps the canonical columns expected by attach_grid().
    """
    empty_cols = [
        "country_code", "country_name", "province_id", "province_name",
        "grid_emission_factor_tCO2_MWh", "grid_electricity_price_USD_MWh", "industrial_tariff_USD_MWh",
        "lowcarbon_grid_emission_factor_tCO2_MWh", "solar_LCOE_USD_MWh", "wind_LCOE_USD_MWh",
        "geothermal_heat_price_USD_MWhth", "geothermal_heat_emission_factor_tCO2_MWhth",
        "source_spatial_resolution", "grid_data_confidence", "source_file_grid",
        "country_key", "country_province_name_key", "province_key", "grid_table_hint",
    ]

    primary_path = _select_grid_primary_file(grid_dir, explicit_grid_csv)
    frames = []

    if primary_path is not None:
        df0 = read_csv_auto(primary_path)
        frames.append(_canonicalize_grid_table(df0, primary_path, hint="generic"))

    if grid_dir is not None and grid_dir.exists():
        ef_path = grid_dir / "grid_emission_factor_asean_sourced.csv"
        price_path = grid_dir / "grid_electricity_price_asean_sourced.csv"
        # Also accept UPDATED/FINAL names if the user keeps them without renaming.
        if not ef_path.exists():
            alt = find_latest(["*emission*factor*asean*sourced*.csv", "*grid*emission*factor*.csv"], [grid_dir])
            ef_path = alt if alt is not None else ef_path
        if not price_path.exists():
            alt = find_latest(["*electricity*price*asean*sourced*.csv", "*grid*electricity*price*.csv"], [grid_dir])
            price_path = alt if alt is not None else price_path

        if ef_path.exists():
            try:
                frames.append(_canonicalize_grid_table(read_csv_auto(ef_path), ef_path, hint="emission"))
            except Exception as exc:
                print(f"[WARNING] Could not read grid emission source file {ef_path}: {exc}")
        if price_path.exists():
            try:
                frames.append(_canonicalize_grid_table(read_csv_auto(price_path), price_path, hint="price"))
            except Exception as exc:
                print(f"[WARNING] Could not read grid price source file {price_path}: {exc}")

    if not frames:
        return pd.DataFrame(columns=empty_cols)

    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in empty_cols:
        if c not in out.columns:
            out[c] = np.nan

    # Remove rows without country code; they cannot be joined.
    out["country_code"] = out["country_code"].map(lambda x: str(x).strip().strip('"').strip("'").upper() if not pd.isna(x) else "")
    out = out[out["country_code"].astype(str).str.match(r"^[A-Z]{3}$", na=False)].copy()
    if out.empty:
        return pd.DataFrame(columns=empty_cols)

    # Fill missing numeric assumptions from any country-level source row.
    numeric_cols = [
        "grid_emission_factor_tCO2_MWh",
        "grid_electricity_price_USD_MWh",
        "industrial_tariff_USD_MWh",
        "lowcarbon_grid_emission_factor_tCO2_MWh",
        "solar_LCOE_USD_MWh",
        "wind_LCOE_USD_MWh",
        "geothermal_heat_price_USD_MWhth",
        "geothermal_heat_emission_factor_tCO2_MWhth",
    ]
    text_cols = ["country_name", "source_spatial_resolution", "grid_data_confidence", "source_file_grid"]

    # Prioritize rows with more available numeric values, but keep all rows for province/country joins.
    out["n_numeric_available"] = out[numeric_cols].notna().sum(axis=1)
    country_fill = (
        out.sort_values(["country_key", "n_numeric_available"], ascending=[True, False])
        .groupby("country_key", dropna=False)
        .agg({**{c: _first_valid for c in numeric_cols}, **{c: _first_valid for c in text_cols}})
        .reset_index()
    )

    for c in numeric_cols:
        fill_map = country_fill.set_index("country_key")[c]
        out[c] = out[c].combine_first(out["country_key"].map(fill_map))
    for c in ["country_name", "grid_data_confidence"]:
        fill_map = country_fill.set_index("country_key")[c]
        out[c] = out[c].combine_first(out["country_key"].map(fill_map))

    # If industrial tariff still missing but grid price exists, copy price as fallback.
    out["industrial_tariff_USD_MWh"] = out["industrial_tariff_USD_MWh"].combine_first(out["grid_electricity_price_USD_MWh"])

    # Keep most complete duplicates first so attach_grid country fallback does not select empty rows.
    out["n_numeric_available"] = out[numeric_cols].notna().sum(axis=1)
    out = out.sort_values(
        ["country_key", "province_key", "n_numeric_available"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Rebuild keys after cleanup.
    out["country_key"] = out["country_code"].map(norm_text)
    out["country_province_name_key"] = out.apply(lambda r: country_province_name_key(r["country_code"], r["province_name"]), axis=1)
    out["province_key"] = make_province_key(out)

    return out[empty_cols + ["n_numeric_available"]].copy()



def attach_grid(annual: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    out = annual.copy()
    if grid.empty:
        for col in [
            "grid_emission_factor_tCO2_MWh", "grid_electricity_price_USD_MWh", "industrial_tariff_USD_MWh",
            "lowcarbon_grid_emission_factor_tCO2_MWh", "solar_LCOE_USD_MWh", "wind_LCOE_USD_MWh",
            "geothermal_heat_price_USD_MWhth", "geothermal_heat_emission_factor_tCO2_MWhth",
            "source_spatial_resolution", "grid_data_confidence",
        ]:
            out[col] = np.nan if "factor" in col or "price" in col or "LCOE" in col or "tariff" in col else "missing"
        return out

    # Prefer province_id-level, then province-name-level, then country-level.
    grid_prov = grid[grid["province_id"].notna() & (grid["province_id"].astype(str).str.strip() != "")].copy()
    grid_name = grid[grid["province_name"].notna() & (grid["province_name"].astype(str).str.strip() != "")].copy()
    grid_country = grid.drop_duplicates("country_key", keep="first").copy()

    cols_to_add = [
        "grid_emission_factor_tCO2_MWh", "grid_electricity_price_USD_MWh", "industrial_tariff_USD_MWh",
        "lowcarbon_grid_emission_factor_tCO2_MWh", "solar_LCOE_USD_MWh", "wind_LCOE_USD_MWh",
        "geothermal_heat_price_USD_MWhth", "geothermal_heat_emission_factor_tCO2_MWhth",
        "source_spatial_resolution", "grid_data_confidence", "source_file_grid",
    ]

    out["country_key"] = out["country_code"].map(norm_text)
    out["country_province_name_key"] = out.apply(lambda r: country_province_name_key(r["country_code"], r["province_name"]), axis=1)

    if not grid_prov.empty:
        tmp = grid_prov.drop_duplicates("province_key", keep="first")[["province_key"] + cols_to_add]
        out = out.merge(tmp, on="province_key", how="left", suffixes=("", "_prov"))
    else:
        for c in cols_to_add:
            out[c] = np.nan

    # Fill missing from name-level.
    if not grid_name.empty:
        tmp = grid_name.drop_duplicates("country_province_name_key", keep="first")[["country_province_name_key"] + cols_to_add]
        name_add = out[["country_province_name_key"]].merge(tmp, on="country_province_name_key", how="left")
        for c in cols_to_add:
            out[c] = out[c].combine_first(name_add[c])

    # Fill missing from country-level.
    tmp = grid_country[["country_key"] + cols_to_add]
    country_add = out[["country_key"]].merge(tmp, on="country_key", how="left")
    for c in cols_to_add:
        out[c] = out[c].combine_first(country_add[c])

    return out


def load_geothermal(geo_csv: Path | None) -> pd.DataFrame:
    if geo_csv is None or not geo_csv.exists():
        return pd.DataFrame(columns=[
            "country_code", "province_name", "geothermal_operating_capacity_MW",
            "geothermal_heat_potential_score", "geothermal_heat_eligible_flag",
            "geothermal_data_confidence",
        ])

    df = read_csv_auto(geo_csv)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["country_code", "country_name", "province_id", "province_name", "province_or_state_name", "technology", "resource_metric", "value", "confidence_level"]:
        if c not in df.columns:
            df[c] = np.nan

    df["value_num"] = pd.to_numeric(df["value"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["country_key"] = df["country_code"].map(norm_text)
    df["province_name_use"] = df["province_name"].combine_first(df["province_or_state_name"])
    df["country_province_name_key"] = df.apply(lambda r: country_province_name_key(r["country_code"], r["province_name_use"]), axis=1)

    power = df[(df["technology"].astype(str).str.lower() == "geothermal_power") & (df["resource_metric"].astype(str).str.lower() == "operating_capacity")].copy()
    heat = df[(df["technology"].astype(str).str.lower() == "geothermal_heat") & (df["resource_metric"].astype(str).str.lower() == "heat_potential_score")].copy()

    power_sum = power.groupby(["country_key", "country_province_name_key"], dropna=False).agg(
        geothermal_operating_capacity_MW=("value_num", "sum"),
        country_code=("country_code", "first"),
        country_name=("country_name", "first"),
        province_name=("province_name_use", "first"),
        geothermal_data_confidence=("confidence_level", "first"),
    ).reset_index()

    def heat_score_rank(s: str) -> int:
        s = str(s).lower().strip()
        if s == "high":
            return 3
        if s == "medium":
            return 2
        if s == "low":
            return 1
        return 0

    if not heat.empty:
        heat["heat_rank"] = heat["value"].map(heat_score_rank)
        heat_agg = heat.sort_values("heat_rank", ascending=False).drop_duplicates("country_province_name_key")[[
            "country_key", "country_province_name_key", "value", "heat_rank", "confidence_level"
        ]].rename(columns={
            "value": "geothermal_heat_potential_score",
            "confidence_level": "geothermal_heat_confidence",
        })
    else:
        heat_agg = pd.DataFrame(columns=["country_key", "country_province_name_key", "geothermal_heat_potential_score", "heat_rank", "geothermal_heat_confidence"])

    if power_sum.empty:
        out = heat_agg.copy()
        out["country_code"] = out["country_key"]
        out["country_name"] = np.nan
        out["province_name"] = np.nan
        out["geothermal_operating_capacity_MW"] = np.nan
        out["geothermal_data_confidence"] = np.nan
    else:
        out = power_sum.merge(heat_agg, on=["country_key", "country_province_name_key"], how="outer")

    out["geothermal_operating_capacity_MW"] = pd.to_numeric(out["geothermal_operating_capacity_MW"], errors="coerce").fillna(0.0)
    out["heat_rank"] = pd.to_numeric(out["heat_rank"], errors="coerce").fillna(0).astype(int)
    out["geothermal_heat_potential_score"] = out["geothermal_heat_potential_score"].fillna("unknown")
    out["geothermal_heat_eligible_flag"] = (out["heat_rank"] >= 2) | (out["geothermal_operating_capacity_MW"] > 0)
    out["source_file_geothermal"] = str(geo_csv)
    return out


def attach_geothermal(annual: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    out = annual.copy()
    out["country_province_name_key"] = out.apply(lambda r: country_province_name_key(r["country_code"], r["province_name"]), axis=1)
    out["country_key"] = out["country_code"].map(norm_text)

    cols = [
        "geothermal_operating_capacity_MW", "geothermal_heat_potential_score", "heat_rank",
        "geothermal_heat_eligible_flag", "geothermal_data_confidence", "geothermal_heat_confidence", "source_file_geothermal",
    ]
    if geo.empty:
        for c in cols:
            out[c] = False if c == "geothermal_heat_eligible_flag" else np.nan
        out["geothermal_heat_potential_score"] = "unknown"
        return out

    g = geo.drop_duplicates("country_province_name_key", keep="first")[["country_province_name_key"] + cols]
    out = out.merge(g, on="country_province_name_key", how="left")
    out["geothermal_operating_capacity_MW"] = pd.to_numeric(out["geothermal_operating_capacity_MW"], errors="coerce").fillna(0.0)
    out["geothermal_heat_potential_score"] = out["geothermal_heat_potential_score"].fillna("unknown")
    out["heat_rank"] = pd.to_numeric(out["heat_rank"], errors="coerce").fillna(0).astype(int)
    out["geothermal_heat_eligible_flag"] = out["geothermal_heat_eligible_flag"].fillna(False).astype(bool)
    return out


# =============================================================================
# Renewable hourly loading and dispatch
# =============================================================================

def load_re_summary(re_summary_csv: Path | None, re_hourly_csv: Path) -> pd.DataFrame:
    if re_summary_csv is not None and re_summary_csv.exists():
        df = read_csv_auto(re_summary_csv)
    else:
        usecols = ["country_code", "country_name", "province_id", "province_name", "pv_energy_kWh_per_kWp", "wind_energy_kWh_per_kW", "GHI_allsky_W_m2", "WS50M_m_s"]
        parts = []
        for chunk in pd.read_csv(re_hourly_csv, usecols=usecols, chunksize=300_000):
            gp = chunk.groupby(ID_COLS, dropna=False).agg(
                pv_annual_kWh_per_kWp=("pv_energy_kWh_per_kWp", "sum"),
                wind_annual_kWh_per_kW=("wind_energy_kWh_per_kW", "sum"),
                GHI_allsky_mean=("GHI_allsky_W_m2", "mean"),
                WS50M_mean=("WS50M_m_s", "mean"),
                n_hours=("pv_energy_kWh_per_kWp", "count"),
            ).reset_index()
            parts.append(gp)
        df = pd.concat(parts, ignore_index=True).groupby(ID_COLS, dropna=False).agg(
            pv_annual_kWh_per_kWp=("pv_annual_kWh_per_kWp", "sum"),
            wind_annual_kWh_per_kW=("wind_annual_kWh_per_kW", "sum"),
            GHI_allsky_mean=("GHI_allsky_mean", "mean"),
            WS50M_mean=("WS50M_mean", "mean"),
            n_hours=("n_hours", "sum"),
        ).reset_index()

    # Alias cleanup.
    aliases = {
        "pv_annual_kWh_per_kWp": ["pv_annual_kWh_per_kWp", "pv_annual_kwh_per_kwp"],
        "wind_annual_kWh_per_kW": ["wind_annual_kWh_per_kW", "wind_annual_kwh_per_kw"],
    }
    for canonical, alist in aliases.items():
        if canonical not in df.columns:
            for a in alist:
                if a in df.columns:
                    df[canonical] = df[a]
                    break
        if canonical not in df.columns:
            df[canonical] = np.nan
        df[canonical] = pd.to_numeric(df[canonical], errors="coerce")

    df["province_key"] = make_province_key(df)
    return df


def load_re_hourly_profiles(re_hourly_csv: Path, max_provinces: int = 0) -> dict[str, dict[str, np.ndarray]]:
    usecols = [
        "country_code", "country_name", "province_id", "province_name",
        "pv_energy_kWh_per_kWp", "wind_energy_kWh_per_kW",
    ]
    df = pd.read_csv(re_hourly_csv, usecols=usecols)
    df["province_key"] = make_province_key(df)

    profiles = {}
    keys = df["province_key"].dropna().unique().tolist()
    if max_provinces and max_provinces > 0:
        keys = keys[:max_provinces]
        df = df[df["province_key"].isin(keys)].copy()

    for key, sub in df.groupby("province_key", sort=False):
        profiles[key] = {
            "pv": pd.to_numeric(sub["pv_energy_kWh_per_kWp"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
            "wind": pd.to_numeric(sub["wind_energy_kWh_per_kW"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
            "n_hours": len(sub),
        }
    return profiles




# =============================================================================
# Hourly DAC demand loading from module 02
# =============================================================================

def _find_col_case_insensitive(columns: list[str], aliases: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for a in aliases:
        if a in columns:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _numeric_array_from_chunk(df: pd.DataFrame, col: str | None, default: float = 0.0) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), default, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype="float64")


def load_hourly_operation_profiles(
    hourly_csv: Path | None,
    province_filter: set[str] | None = None,
    chunksize: int = 500_000,
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], pd.DataFrame]:
    """Load module-02 hourly operation profile as compact NumPy arrays.

    Returned key: (province_key, operation_policy). The hourly profile is used
    only for demand timing; annual totals are later scaled back to the annual
    summary to preserve compatibility with existing module 02 outputs.
    """
    diag_rows = []
    if hourly_csv is None or not Path(hourly_csv).exists():
        return {}, pd.DataFrame([{
            "check": "hourly_operation_profile_available",
            "value": False,
            "notes": f"File not found: {hourly_csv}",
        }])

    hourly_csv = Path(hourly_csv)
    header = pd.read_csv(hourly_csv, nrows=0, encoding="utf-8-sig")
    cols = [str(c).strip().replace("\ufeff", "") for c in header.columns]
    header.columns = cols

    id_alias = {
        "country_code": ["country_code", "iso3"],
        "country_name": ["country_name", "country"],
        "province_id": ["province_id", "GID_1", "gid_1"],
        "province_name": ["province_name", "NAME_1", "name_1", "province"],
        "operation_policy": ["operation_policy", "policy", "operation_policy_name"],
    }
    demand_alias = {
        "CO2_kg_h": ["CO2_kg_h", "co2_kg_h", "CO2_kg_per_h", "co2_kg_per_h", "CO2_kg_hour", "co2_kg_hour"],
        "Q_heat_kWhth_h": ["Q_heat_kWhth_h", "q_heat_kWhth_h", "Q_heat_kWh_h", "heat_kWhth_h", "heat_kWh_h", "Q_heat_kWhth_hour"],
        "E_total_el_kWhe_h": ["E_total_el_kWhe_h", "e_total_el_kWhe_h", "E_total_el_kWh_h", "process_electricity_kWh_h", "electricity_kWh_h"],
        "Q_cool_kWhth_h": ["Q_cool_kWhth_h", "q_cool_kWhth_h", "Q_cool_kWh_h", "cooling_kWhth_h", "cooling_kWh_h"],
    }
    optional_alias = {
        "datetime_local": ["datetime_local", "local_datetime", "time_local"],
        "datetime_utc": ["datetime_utc", "utc_datetime", "time_utc"],
    }

    selected = {}
    for k, aliases in {**id_alias, **demand_alias, **optional_alias}.items():
        selected[k] = _find_col_case_insensitive(cols, aliases)

    required_missing = [k for k in ["country_code", "province_id", "province_name", "operation_policy"] if selected.get(k) is None]
    if required_missing:
        return {}, pd.DataFrame([{
            "check": "hourly_operation_profile_available",
            "value": False,
            "notes": f"Missing required columns in hourly profile: {required_missing}",
        }])

    usecols = sorted({c for c in selected.values() if c is not None})
    store: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    n_rows = 0
    n_rows_kept = 0

    for chunk in pd.read_csv(hourly_csv, usecols=usecols, chunksize=chunksize, encoding="utf-8-sig"):
        chunk.columns = [str(c).strip().replace("\ufeff", "") for c in chunk.columns]
        n_rows += len(chunk)

        cc = selected["country_code"]
        cn = selected.get("country_name")
        pid = selected["province_id"]
        pn = selected["province_name"]
        pol = selected["operation_policy"]

        tmp = pd.DataFrame({
            "country_code": chunk[cc].astype(str).str.strip().str.upper(),
            "country_name": chunk[cn].astype(str).str.strip() if cn is not None and cn in chunk.columns else chunk[cc].astype(str).str.strip().str.upper(),
            "province_id": chunk[pid].astype(str).str.strip(),
            "province_name": chunk[pn].astype(str).str.strip(),
            "operation_policy": chunk[pol].astype(str).str.strip(),
            "CO2_kg_h": _numeric_array_from_chunk(chunk, selected.get("CO2_kg_h"), 0.0),
            "Q_heat_kWhth_h": _numeric_array_from_chunk(chunk, selected.get("Q_heat_kWhth_h"), 0.0),
            "E_total_el_kWhe_h": _numeric_array_from_chunk(chunk, selected.get("E_total_el_kWhe_h"), 0.0),
            "Q_cool_kWhth_h": _numeric_array_from_chunk(chunk, selected.get("Q_cool_kWhth_h"), 0.0),
        })
        tmp["province_key"] = make_province_key(tmp)
        if province_filter is not None:
            tmp = tmp[tmp["province_key"].isin(province_filter)].copy()
        if tmp.empty:
            continue
        n_rows_kept += len(tmp)

        for (pkey, policy), sub in tmp.groupby(["province_key", "operation_policy"], sort=False, dropna=False):
            key = (str(pkey), str(policy))
            rec = store.setdefault(key, {
                "CO2_kg_h": [], "Q_heat_kWhth_h": [], "E_total_el_kWhe_h": [], "Q_cool_kWhth_h": []
            })
            for c in rec:
                rec[c].append(sub[c].to_numpy(dtype="float64"))

    profiles: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, rec in store.items():
        profiles[key] = {c: np.concatenate(parts) if parts else np.array([], dtype="float64") for c, parts in rec.items()}
        profiles[key]["n_hours"] = len(profiles[key]["E_total_el_kWhe_h"])

    if profiles:
        lengths = pd.Series([v["n_hours"] for v in profiles.values()], dtype="int64")
        policies = pd.Series([k[1] for k in profiles.keys()]).value_counts().reset_index()
        policies.columns = ["operation_policy", "n_province_policy_profiles"]
    else:
        lengths = pd.Series([], dtype="float64")
        policies = pd.DataFrame(columns=["operation_policy", "n_province_policy_profiles"])

    diag_rows.extend([
        {"check": "hourly_operation_profile_available", "value": True, "notes": str(hourly_csv)},
        {"check": "hourly_operation_rows_read", "value": int(n_rows), "notes": "raw rows read from CSV"},
        {"check": "hourly_operation_rows_kept", "value": int(n_rows_kept), "notes": "rows after optional province filter"},
        {"check": "hourly_operation_profile_count", "value": int(len(profiles)), "notes": "province-policy profiles loaded"},
        {"check": "hourly_operation_min_hours", "value": int(lengths.min()) if len(lengths) else 0, "notes": "minimum loaded hours per province-policy"},
        {"check": "hourly_operation_median_hours", "value": float(lengths.median()) if len(lengths) else 0.0, "notes": "median loaded hours per province-policy"},
        {"check": "hourly_operation_max_hours", "value": int(lengths.max()) if len(lengths) else 0, "notes": "maximum loaded hours per province-policy"},
    ])
    diag = pd.DataFrame(diag_rows)
    if not policies.empty:
        policy_rows = policies.assign(check="profiles_by_policy", notes="count by operation policy").rename(columns={"operation_policy": "value", "n_province_policy_profiles": "extra_value"})
        diag = pd.concat([diag, policy_rows[["check", "value", "extra_value", "notes"]]], ignore_index=True, sort=False)
    return profiles, diag


def _resize_or_pad_profile(arr: np.ndarray, n_hours: int) -> np.ndarray:
    arr = np.asarray(arr, dtype="float64")
    if len(arr) == n_hours:
        return arr.copy()
    if len(arr) > n_hours:
        return arr[:n_hours].copy()
    if len(arr) > 0 and len(arr) < n_hours:
        out = np.zeros(n_hours, dtype="float64")
        out[:len(arr)] = arr
        return out
    return np.zeros(n_hours, dtype="float64")


def _scale_hourly_to_annual(arr: np.ndarray, annual_total_kwh: float) -> np.ndarray:
    arr = np.asarray(arr, dtype="float64")
    annual_total_kwh = to_float(annual_total_kwh, np.nan)
    s = float(np.nansum(arr))
    if np.isfinite(annual_total_kwh) and annual_total_kwh > 0:
        if s > 1e-12:
            return arr * annual_total_kwh / s
        return np.full(len(arr), annual_total_kwh / max(len(arr), 1), dtype="float64")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _profile_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(arr, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.quantile(arr, 0.95)), float(np.max(arr)), float(np.sum(arr))


def _design_value(arr: np.ndarray, basis: str) -> float:
    mean_v, p95_v, peak_v, _ = _profile_stats(arr)
    basis = str(basis).lower()
    if basis == "peak":
        return peak_v
    if basis == "average":
        return mean_v
    return p95_v


def make_demand_profiles_for_row(
    row: pd.Series,
    n_hours: int,
    hourly_profiles: dict[tuple[str, str], dict[str, np.ndarray]] | None,
    args,
) -> dict:
    """Return hourly process heat/electricity profiles and design-load metadata.

    Hourly profiles from module 02 are scaled to the annual totals in the annual
    operation summary. This preserves backwards compatibility while exposing the
    timing needed for renewable dispatch.
    """
    n_hours = int(n_hours) if n_hours and n_hours > 0 else 8760
    process_annual_kwh = to_float(row.get("annual_process_electricity_MWhe"), 0.0) * 1000.0
    heat_annual_kwh = to_float(row.get("annual_heat_demand_MWhth"), 0.0) * 1000.0
    cool_annual_kwh = to_float(row.get("annual_cooling_demand_MWhth"), 0.0) * 1000.0
    co2_annual_kg = to_float(row.get("annual_CO2_t_per_1000kgads"), 0.0) * 1000.0

    process = np.full(n_hours, process_annual_kwh / max(n_hours, 1), dtype="float64")
    heat = np.full(n_hours, heat_annual_kwh / max(n_hours, 1), dtype="float64")
    cooling = np.full(n_hours, cool_annual_kwh / max(n_hours, 1), dtype="float64")
    co2 = np.full(n_hours, co2_annual_kg / max(n_hours, 1), dtype="float64")
    method = "flat_annual_from_module02"
    used = False
    note = "No matching hourly operation profile; flat annual demand used."

    if hourly_profiles:
        key = (str(row.get("province_key")), str(row.get("operation_policy")))
        prof = hourly_profiles.get(key)
        if prof is not None and int(prof.get("n_hours", 0)) > 0:
            raw_process = _resize_or_pad_profile(prof.get("E_total_el_kWhe_h", np.array([])), n_hours)
            raw_heat = _resize_or_pad_profile(prof.get("Q_heat_kWhth_h", np.array([])), n_hours)
            raw_cool = _resize_or_pad_profile(prof.get("Q_cool_kWhth_h", np.array([])), n_hours)
            raw_co2 = _resize_or_pad_profile(prof.get("CO2_kg_h", np.array([])), n_hours)
            process = _scale_hourly_to_annual(raw_process, process_annual_kwh)
            heat = _scale_hourly_to_annual(raw_heat, heat_annual_kwh)
            cooling = _scale_hourly_to_annual(raw_cool, cool_annual_kwh)
            co2 = _scale_hourly_to_annual(raw_co2, co2_annual_kg)
            method = "hourly_operation_profile_from_module02_scaled_to_annual"
            used = True
            note = "Hourly profile matched by province_key and operation_policy, then scaled to annual summary totals."

    hp_el = heat / args.heat_pump_cop if args.heat_pump_cop > 0 else np.full(n_hours, np.nan)
    total_hp = process + hp_el
    total_geo = process.copy()

    proc_mean, proc_p95, proc_peak, proc_sum = _profile_stats(process)
    heat_mean, heat_p95, heat_peak, heat_sum = _profile_stats(heat)
    hp_mean, hp_p95, hp_peak, hp_sum = _profile_stats(hp_el)
    total_hp_mean, total_hp_p95, total_hp_peak, total_hp_sum = _profile_stats(total_hp)
    total_geo_mean, total_geo_p95, total_geo_peak, total_geo_sum = _profile_stats(total_geo)

    heatpump_capacity_design = _design_value(heat, getattr(args, "heat_pump_capacity_design_basis", "p95"))
    battery_power_hp = _design_value(total_hp, getattr(args, "battery_power_basis", "average"))
    battery_power_geo = _design_value(total_geo, getattr(args, "battery_power_basis", "average"))

    return {
        "process_el_kwh": process,
        "heat_kwh": heat,
        "cooling_kwh": cooling,
        "co2_kg": co2,
        "hp_el_kwh": hp_el,
        "total_hp_kwh": total_hp,
        "total_geo_kwh": total_geo,
        "process_el_MWhe": proc_sum / 1000.0,
        "heat_MWhth": heat_sum / 1000.0,
        "cooling_MWhth": _profile_stats(cooling)[3] / 1000.0,
        "co2_t": _profile_stats(co2)[3] / 1000.0,
        "hp_el_MWhe": hp_sum / 1000.0,
        "demand_profile_method": method,
        "hourly_profile_used_flag": bool(used),
        "hourly_profile_n_hours": int(n_hours),
        "demand_profile_note": note,
        "peak_process_electricity_kW": proc_peak,
        "p95_process_electricity_kW": proc_p95,
        "average_process_electricity_kW": proc_mean,
        "peak_heat_demand_kWth": heat_peak,
        "p95_heat_demand_kWth": heat_p95,
        "average_heat_demand_kWth": heat_mean,
        "heat_pump_capacity_kWth_design": heatpump_capacity_design,
        "heat_pump_capacity_design_basis": getattr(args, "heat_pump_capacity_design_basis", "p95"),
        "peak_heat_pump_electricity_kW": hp_peak,
        "p95_heat_pump_electricity_kW": hp_p95,
        "average_heat_pump_electricity_kW": hp_mean,
        "peak_total_electricity_demand_kW_HP": total_hp_peak,
        "p95_total_electricity_demand_kW_HP": total_hp_p95,
        "average_total_electricity_demand_kW_HP": total_hp_mean,
        "peak_total_electricity_demand_kW_geothermal": total_geo_peak,
        "p95_total_electricity_demand_kW_geothermal": total_geo_p95,
        "average_total_electricity_demand_kW_geothermal": total_geo_mean,
        "battery_power_kW_HP_design": battery_power_hp,
        "battery_power_kW_geothermal_design": battery_power_geo,
        "battery_power_basis": getattr(args, "battery_power_basis", "average"),
    }


def demand_metadata_for_output(demand: dict, heat_pump_enabled: bool) -> dict:
    """Metadata columns passed downstream to modules 04/05."""
    if heat_pump_enabled:
        peak_total = demand["peak_total_electricity_demand_kW_HP"]
        p95_total = demand["p95_total_electricity_demand_kW_HP"]
        avg_total = demand["average_total_electricity_demand_kW_HP"]
        hp_cap = demand["heat_pump_capacity_kWth_design"]
        batt_power = demand["battery_power_kW_HP_design"]
    else:
        peak_total = demand["peak_total_electricity_demand_kW_geothermal"]
        p95_total = demand["p95_total_electricity_demand_kW_geothermal"]
        avg_total = demand["average_total_electricity_demand_kW_geothermal"]
        hp_cap = 0.0
        batt_power = demand["battery_power_kW_geothermal_design"]
    return {
        "demand_profile_method": demand["demand_profile_method"],
        "hourly_profile_used_flag": demand["hourly_profile_used_flag"],
        "hourly_profile_n_hours": demand["hourly_profile_n_hours"],
        "demand_profile_note": demand["demand_profile_note"],
        "peak_process_electricity_kW": demand["peak_process_electricity_kW"],
        "p95_process_electricity_kW": demand["p95_process_electricity_kW"],
        "average_process_electricity_kW": demand["average_process_electricity_kW"],
        "peak_heat_demand_kWth": demand["peak_heat_demand_kWth"],
        "p95_heat_demand_kWth": demand["p95_heat_demand_kWth"],
        "average_heat_demand_kWth": demand["average_heat_demand_kWth"],
        "heat_pump_capacity_kWth_design": hp_cap,
        "heat_pump_capacity_design_basis": demand["heat_pump_capacity_design_basis"],
        "peak_heat_pump_electricity_kW": demand["peak_heat_pump_electricity_kW"] if heat_pump_enabled else 0.0,
        "p95_heat_pump_electricity_kW": demand["p95_heat_pump_electricity_kW"] if heat_pump_enabled else 0.0,
        "average_heat_pump_electricity_kW": demand["average_heat_pump_electricity_kW"] if heat_pump_enabled else 0.0,
        "peak_total_electricity_demand_kW": peak_total,
        "p95_total_electricity_demand_kW": p95_total,
        "average_total_electricity_demand_kW": avg_total,
        "battery_power_kW_design": batt_power,
        "battery_power_basis": demand["battery_power_basis"],
    }

def dispatch_no_battery(pv_gen: np.ndarray, wind_gen: np.ndarray, demand_kwh: np.ndarray) -> dict:
    gen = pv_gen + wind_gen
    used_total = np.minimum(gen, demand_kwh)
    curtailment = np.maximum(gen - demand_kwh, 0.0)
    grid = np.maximum(demand_kwh - gen, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pv_frac = np.where(gen > 0, pv_gen / gen, 0.0)
        wind_frac = np.where(gen > 0, wind_gen / gen, 0.0)

    pv_used = used_total * pv_frac
    wind_used = used_total * wind_frac

    return {
        "pv_generation_kWh": float(np.sum(pv_gen)),
        "wind_generation_kWh": float(np.sum(wind_gen)),
        "pv_used_kWh": float(np.sum(pv_used)),
        "wind_used_kWh": float(np.sum(wind_used)),
        "battery_charge_kWh": 0.0,
        "battery_discharge_kWh": 0.0,
        "battery_losses_kWh": 0.0,
        "grid_electricity_kWh": float(np.sum(grid)),
        "curtailed_RE_kWh": float(np.sum(curtailment)),
        "soc_initial_kWh": 0.0,
        "soc_final_kWh": 0.0,
        "soc_min_kWh": 0.0,
        "soc_max_kWh": 0.0,
    }


def dispatch_with_battery(
    pv_gen: np.ndarray,
    wind_gen: np.ndarray,
    demand_kwh: np.ndarray,
    battery_capacity_kwh: float,
    battery_power_kw: float,
    roundtrip_eff: float,
    initial_soc_frac: float,
) -> dict:
    eta = math.sqrt(max(min(roundtrip_eff, 1.0), 1e-9))
    soc = float(battery_capacity_kwh * initial_soc_frac)
    soc_min = soc
    soc_max = soc

    pv_used = 0.0
    wind_used = 0.0
    grid = 0.0
    curtailed = 0.0
    charge_input_total = 0.0
    discharge_output_total = 0.0
    battery_losses = 0.0

    pv_generation = float(np.sum(pv_gen))
    wind_generation = float(np.sum(wind_gen))

    for pv, wind, d in zip(pv_gen, wind_gen, demand_kwh):
        gen = pv + wind
        direct_used = min(gen, d)
        if gen > 0:
            pv_used += direct_used * (pv / gen)
            wind_used += direct_used * (wind / gen)

        surplus = max(gen - d, 0.0)
        deficit = max(d - gen, 0.0)

        if surplus > 0 and battery_capacity_kwh > 0 and battery_power_kw > 0:
            charge_input = min(surplus, battery_power_kw, max((battery_capacity_kwh - soc) / eta, 0.0))
            soc += charge_input * eta
            charge_input_total += charge_input
            curtailed += surplus - charge_input
            battery_losses += charge_input * (1.0 - eta)
        else:
            curtailed += surplus

        if deficit > 0 and battery_capacity_kwh > 0 and battery_power_kw > 0:
            discharge_output = min(deficit, battery_power_kw, soc * eta)
            soc -= discharge_output / eta
            discharge_output_total += discharge_output
            grid += deficit - discharge_output
            battery_losses += discharge_output * (1.0 / eta - 1.0)
        else:
            grid += deficit

        soc_min = min(soc_min, soc)
        soc_max = max(soc_max, soc)

    return {
        "pv_generation_kWh": pv_generation,
        "wind_generation_kWh": wind_generation,
        "pv_used_kWh": float(pv_used),
        "wind_used_kWh": float(wind_used),
        "battery_charge_kWh": float(charge_input_total),
        "battery_discharge_kWh": float(discharge_output_total),
        "battery_losses_kWh": float(battery_losses),
        "grid_electricity_kWh": float(grid),
        "curtailed_RE_kWh": float(curtailed),
        "soc_initial_kWh": float(battery_capacity_kwh * initial_soc_frac),
        "soc_final_kWh": float(soc),
        "soc_min_kWh": float(soc_min),
        "soc_max_kWh": float(soc_max),
    }


# =============================================================================
# Scenario evaluation
# =============================================================================

def calc_common_metrics(row: dict, args) -> dict:
    process_el = to_float(row.get("annual_process_electricity_MWhe"), 0.0)
    heat = to_float(row.get("annual_heat_demand_MWhth"), 0.0)
    co2 = to_float(row.get("annual_CO2_t_per_1000kgads"), np.nan)
    hp_el = heat / args.heat_pump_cop if args.heat_pump_cop > 0 else np.nan
    return {"process_el_MWhe": process_el, "heat_MWhth": heat, "co2_t": co2, "hp_el_MWhe": hp_el}


def finalize_energy_row(base: dict, args) -> dict:
    # Convert kWh dispatch to MWh.
    for k in [
        "pv_generation", "wind_generation", "pv_used", "wind_used", "battery_charge", "battery_discharge",
        "battery_losses", "grid_electricity", "curtailed_RE",
    ]:
        kkwh = f"{k}_kWh"
        kmwh = f"{k}_MWhe"
        if kkwh in base:
            base[kmwh] = base[kkwh] / 1000.0

    total_demand = to_float(base.get("annual_total_electricity_demand_MWhe"), np.nan)
    if not np.isfinite(total_demand) or total_demand <= 0:
        total_demand = to_float(base.get("grid_electricity_MWhe"), 0.0) + to_float(base.get("pv_used_MWhe"), 0.0) + to_float(base.get("wind_used_MWhe"), 0.0) + to_float(base.get("battery_discharge_MWhe"), 0.0)
        base["annual_total_electricity_demand_MWhe"] = total_demand

    pv_used = to_float(base.get("pv_used_MWhe"), 0.0)
    wind_used = to_float(base.get("wind_used_MWhe"), 0.0)
    batt_dis = to_float(base.get("battery_discharge_MWhe"), 0.0)
    grid_mwh = to_float(base.get("grid_electricity_MWhe"), 0.0)
    curtail = to_float(base.get("curtailed_RE_MWhe"), 0.0)
    pv_gen = to_float(base.get("pv_generation_MWhe"), 0.0)
    wind_gen = to_float(base.get("wind_generation_MWhe"), 0.0)

    if total_demand > 0:
        base["renewable_share_of_electricity"] = (pv_used + wind_used + batt_dis) / total_demand
        base["grid_backup_share"] = grid_mwh / total_demand
        base["pv_used_share_of_electricity"] = pv_used / total_demand
        base["wind_used_share_of_electricity"] = wind_used / total_demand
    else:
        base["renewable_share_of_electricity"] = np.nan
        base["grid_backup_share"] = np.nan
        base["pv_used_share_of_electricity"] = np.nan
        base["wind_used_share_of_electricity"] = np.nan

    total_re_gen = pv_gen + wind_gen
    base["curtailment_share_of_RE_generation"] = curtail / total_re_gen if total_re_gen > 0 else 0.0

    batt_cap = to_float(base.get("battery_capacity_kWh"), 0.0)
    batt_dis_kwh = to_float(base.get("battery_discharge_kWh"), 0.0)
    base["battery_equivalent_cycles"] = batt_dis_kwh / batt_cap if batt_cap > 0 else 0.0

    # Operational energy emissions. PV/wind lifecycle emissions are left for LCOD/net-removal module unless user supplies explicit factors.
    ef_grid = to_float(base.get("effective_grid_emission_factor_tCO2_MWh"), np.nan)
    ef_heat = to_float(base.get("geothermal_heat_emission_factor_tCO2_MWhth"), np.nan)
    geo_heat = to_float(base.get("geothermal_heat_used_MWhth"), 0.0)

    grid_em = grid_mwh * ef_grid if np.isfinite(ef_grid) else np.nan
    heat_em = geo_heat * ef_heat if np.isfinite(ef_heat) else 0.0
    base["energy_emissions_tCO2"] = grid_em + heat_em if np.isfinite(grid_em) else np.nan

    # Energy cost. PV/wind LCOE cost is optional; capacity CAPEX is intended for LCOD module.
    grid_price = to_float(base.get("effective_grid_electricity_price_USD_MWh"), np.nan)
    pv_lcoe = to_float(base.get("pv_LCOE_USD_MWh"), np.nan)
    wind_lcoe = to_float(base.get("wind_LCOE_USD_MWh"), np.nan)
    heat_price = to_float(base.get("geothermal_heat_price_USD_MWhth"), np.nan)

    grid_cost = grid_mwh * grid_price if np.isfinite(grid_price) else np.nan
    pv_cost = pv_used * pv_lcoe if np.isfinite(pv_lcoe) else np.nan
    wind_cost = wind_used * wind_lcoe if np.isfinite(wind_lcoe) else np.nan
    heat_cost = geo_heat * heat_price if np.isfinite(heat_price) else np.nan

    base["grid_electricity_cost_USD"] = grid_cost
    base["pv_electricity_cost_USD"] = pv_cost
    base["wind_electricity_cost_USD"] = wind_cost
    base["geothermal_heat_cost_USD"] = heat_cost

    # Strict energy_cost_USD only if all used components have prices.
    cost_terms = []
    missing_cost = []
    if grid_mwh > 1e-9:
        if np.isfinite(grid_cost): cost_terms.append(grid_cost)
        else: missing_cost.append("grid_price")
    if pv_used > 1e-9:
        if np.isfinite(pv_cost): cost_terms.append(pv_cost)
        else: missing_cost.append("pv_lcoe")
    if wind_used > 1e-9:
        if np.isfinite(wind_cost): cost_terms.append(wind_cost)
        else: missing_cost.append("wind_lcoe")
    if geo_heat > 1e-9:
        if np.isfinite(heat_cost): cost_terms.append(heat_cost)
        else: missing_cost.append("geothermal_heat_price")
    base["energy_cost_USD"] = float(np.sum(cost_terms)) if not missing_cost else np.nan
    base["missing_energy_cost_items"] = ";".join(sorted(set(missing_cost)))

    co2 = to_float(base.get("annual_CO2_t_per_1000kgads"), np.nan)
    if co2 > 0:
        base["energy_emission_intensity_tCO2_per_tCO2captured"] = base["energy_emissions_tCO2"] / co2 if np.isfinite(base["energy_emissions_tCO2"]) else np.nan
        base["energy_cost_USD_per_tCO2captured"] = base["energy_cost_USD"] / co2 if np.isfinite(base["energy_cost_USD"]) else np.nan
    else:
        base["energy_emission_intensity_tCO2_per_tCO2captured"] = np.nan
        base["energy_cost_USD_per_tCO2captured"] = np.nan

    return base


def make_base_output(row: pd.Series, scenario_id: str, scenario_base: str) -> dict:
    d = {
        "country_code": row.get("country_code"),
        "country_name": row.get("country_name"),
        "province_id": row.get("province_id"),
        "province_name": row.get("province_name"),
        "longitude": row.get("longitude", np.nan),
        "latitude": row.get("latitude", np.nan),
        "operation_policy": row.get("operation_policy"),
        "energy_scenario": scenario_id,
        "energy_scenario_base": scenario_base,
        "annual_CO2_t_per_1000kgads": row.get("annual_CO2_t_per_1000kgads", np.nan),
        "annual_H2O_t_per_1000kgads": row.get("annual_H2O_t_per_1000kgads", np.nan),
        "annual_process_electricity_MWhe": row.get("annual_process_electricity_MWhe", np.nan),
        "annual_heat_demand_MWhth": row.get("annual_heat_demand_MWhth", np.nan),
        "annual_cooling_demand_MWhth": row.get("annual_cooling_demand_MWhth", np.nan),
        "SEC_heat_MWhth_tCO2": row.get("SEC_heat_MWhth_tCO2", np.nan),
        "SEC_el_MWhe_tCO2": row.get("SEC_el_MWhe_tCO2", np.nan),
        "SEC_total_MWh_tCO2_before_compression": row.get("SEC_total_MWh_tCO2_before_compression", np.nan),
        "H2O_CO2_mass_ratio_tH2O_tCO2": row.get("H2O_CO2_mass_ratio_tH2O_tCO2", np.nan),
        "grid_emission_factor_tCO2_MWh": row.get("grid_emission_factor_tCO2_MWh", np.nan),
        "grid_electricity_price_USD_MWh": row.get("grid_electricity_price_USD_MWh", np.nan),
        "industrial_tariff_USD_MWh": row.get("industrial_tariff_USD_MWh", np.nan),
        "lowcarbon_grid_emission_factor_tCO2_MWh": row.get("lowcarbon_grid_emission_factor_tCO2_MWh", np.nan),
        "grid_data_confidence": row.get("grid_data_confidence", "unknown"),
        "geothermal_operating_capacity_MW": row.get("geothermal_operating_capacity_MW", 0.0),
        "geothermal_heat_potential_score": row.get("geothermal_heat_potential_score", "unknown"),
        "geothermal_heat_eligible_flag": bool(row.get("geothermal_heat_eligible_flag", False)),
        "geothermal_data_confidence": row.get("geothermal_data_confidence", "unknown"),
        "demand_profile_method": row.get("demand_profile_method", "flat_annual"),
        "scenario_valid_flag": True,
        "invalid_reason": "",
        "data_confidence": "starter_mixed",
    }
    return d




def apply_wind_capacity_constraint(d: dict, args) -> dict:
    """Flag implausible wind-heavy sizing for the 1000 kg adsorbent basis.

    The RE scenario generator includes PV/wind generation-share variants. In
    wind-poor provinces, pure-wind or wind-heavy variants can require absurdly
    large installed capacity. Keeping those rows as valid distorts LCOD and best
    scenario selection downstream. This function preserves the rows for audit,
    but marks them invalid when the capacity exceeds the user-defined limit.
    """
    limit = to_float(getattr(args, "max_wind_capacity_kw", np.nan), np.nan)
    wind_cap = to_float(d.get("wind_capacity_kW"), 0.0)
    d["wind_capacity_limit_kW_per_1000kgads"] = limit
    d["wind_capacity_feasible_flag"] = True
    d["wind_capacity_constraint_exceeded_flag"] = False

    if np.isfinite(limit) and limit > 0 and np.isfinite(wind_cap) and wind_cap > limit:
        d["wind_capacity_feasible_flag"] = False
        d["wind_capacity_constraint_exceeded_flag"] = True
        existing = str(d.get("invalid_reason", "") or "")
        if "wind_capacity_above_limit" not in existing:
            existing += "wind_capacity_above_limit;"
        d["invalid_reason"] = existing
        d["scenario_valid_flag"] = False
    return d

def evaluate_scenarios(
    annual: pd.DataFrame,
    re_summary: pd.DataFrame,
    re_profiles: dict[str, dict[str, np.ndarray]],
    hourly_operation_profiles: dict[tuple[str, str], dict[str, np.ndarray]] | None,
    args,
) -> pd.DataFrame:
    # Merge RE summary into annual rows.
    re_cols = ["province_key", "pv_annual_kWh_per_kWp", "wind_annual_kWh_per_kW"]
    for opt in ["GHI_allsky_mean", "WS50M_mean", "n_hours"]:
        if opt in re_summary.columns:
            re_cols.append(opt)
    ann = annual.merge(re_summary[re_cols].drop_duplicates("province_key"), on="province_key", how="left")

    rows = []
    re_oversize_factors = parse_float_list(args.re_oversize_factors, [1.0, 1.5, 2.0])
    pv_shares = parse_float_list(args.pv_target_shares, [1.0, 0.75, 0.5, 0.25, 0.0])
    battery_durations = parse_float_list(args.battery_durations_h, [4.0])

    if args.max_rows and args.max_rows > 0:
        ann = ann.head(args.max_rows).copy()

    for i, (_, r) in enumerate(ann.iterrows(), start=1):
        if i % 100 == 0:
            print(f"[EVAL] annual row {i:,}/{len(ann):,}")

        province_key = r["province_key"]
        prof = re_profiles.get(province_key)
        if prof is not None:
            pv_profile = prof["pv"]
            wind_profile = prof["wind"]
            n_hours = len(pv_profile)
        else:
            pv_profile = np.array([], dtype="float64")
            wind_profile = np.array([], dtype="float64")
            n_hours = 8760

        demand = make_demand_profiles_for_row(r, n_hours, hourly_operation_profiles, args)
        process_el = demand["process_el_MWhe"]
        heat = demand["heat_MWhth"]
        hp_el = demand["hp_el_MWhe"]
        co2 = demand["co2_t"]

        # Global per-row assumptions.
        grid_ef = to_float(r.get("grid_emission_factor_tCO2_MWh"), np.nan)
        grid_price = to_float(r.get("grid_electricity_price_USD_MWh"), np.nan)
        if not np.isfinite(grid_price):
            grid_price = to_float(r.get("industrial_tariff_USD_MWh"), np.nan)
        # PV/wind CAPEX is intended for module 05. To avoid double counting,
        # module 03 uses CLI PV/wind LCOE values only. Defaults are zero.
        pv_lcoe = args.pv_lcoe_usd_mwh
        wind_lcoe = args.wind_lcoe_usd_mwh

        geo_heat_price = to_float(r.get("geothermal_heat_price_USD_MWhth"), np.nan)
        if not np.isfinite(geo_heat_price):
            geo_heat_price = args.geothermal_heat_price_usd_mwhth
        geo_heat_ef = to_float(r.get("geothermal_heat_emission_factor_tCO2_MWhth"), np.nan)
        if not np.isfinite(geo_heat_ef):
            geo_heat_ef = args.geothermal_heat_ef_tco2_mwhth

        # S0 grid + heat pump.
        if "S0" in args.scenarios or "all" in args.scenarios:
            d = make_base_output(r, "S0_grid_HP", "S0_grid_HP")
            total_el = process_el + hp_el
            d.update(demand_metadata_for_output(demand, heat_pump_enabled=True))
            d.update({
                "heat_source": "heat_pump",
                "electricity_source": "grid",
                "battery_enabled": False,
                "geothermal_heat_enabled": False,
                "heat_pump_COP": args.heat_pump_cop,
                "annual_heat_pump_electricity_MWhe": hp_el,
                "annual_total_electricity_demand_MWhe": total_el,
                "annual_process_electricity_MWhe": process_el,
                "annual_heat_demand_MWhth": heat,
                "annual_cooling_demand_MWhth": demand["cooling_MWhth"],
                "annual_CO2_t_per_1000kgads": co2,
                "heat_from_heatpump_MWhth": heat,
                "geothermal_heat_used_MWhth": 0.0,
                "pv_capacity_kWp": 0.0,
                "wind_capacity_kW": 0.0,
                "battery_capacity_kWh": 0.0,
                "battery_power_kW": 0.0,
                "pv_generation_kWh": 0.0,
                "wind_generation_kWh": 0.0,
                "pv_used_kWh": 0.0,
                "wind_used_kWh": 0.0,
                "battery_charge_kWh": 0.0,
                "battery_discharge_kWh": 0.0,
                "battery_losses_kWh": 0.0,
                "grid_electricity_kWh": total_el * 1000.0,
                "curtailed_RE_kWh": 0.0,
                "effective_grid_emission_factor_tCO2_MWh": grid_ef,
                "effective_grid_electricity_price_USD_MWh": grid_price,
                "pv_LCOE_USD_MWh": pv_lcoe,
                "wind_LCOE_USD_MWh": wind_lcoe,
                "geothermal_heat_price_USD_MWhth": geo_heat_price,
                "geothermal_heat_emission_factor_tCO2_MWhth": geo_heat_ef,
            })
            if not np.isfinite(grid_ef):
                d["invalid_reason"] += "missing_grid_EF;"
            d["grid_price_available_flag"] = bool(np.isfinite(grid_price))
            d["scenario_valid_flag"] = d["invalid_reason"] == ""
            rows.append(finalize_energy_row(d, args))

        # S3 grid electricity + geothermal heat.
        if ("S3" in args.scenarios or "all" in args.scenarios) and bool(r.get("geothermal_heat_eligible_flag", False)):
            d = make_base_output(r, "S3_grid_geothermalHeat", "S3_grid_geothermalHeat")
            total_el = process_el
            d.update(demand_metadata_for_output(demand, heat_pump_enabled=False))
            d.update({
                "heat_source": "geothermal_heat",
                "electricity_source": "grid",
                "battery_enabled": False,
                "geothermal_heat_enabled": True,
                "heat_pump_COP": np.nan,
                "annual_heat_pump_electricity_MWhe": 0.0,
                "annual_total_electricity_demand_MWhe": total_el,
                "annual_process_electricity_MWhe": process_el,
                "annual_heat_demand_MWhth": heat,
                "annual_cooling_demand_MWhth": demand["cooling_MWhth"],
                "annual_CO2_t_per_1000kgads": co2,
                "heat_from_heatpump_MWhth": 0.0,
                "geothermal_heat_used_MWhth": heat,
                "pv_capacity_kWp": 0.0,
                "wind_capacity_kW": 0.0,
                "battery_capacity_kWh": 0.0,
                "battery_power_kW": 0.0,
                "pv_generation_kWh": 0.0,
                "wind_generation_kWh": 0.0,
                "pv_used_kWh": 0.0,
                "wind_used_kWh": 0.0,
                "battery_charge_kWh": 0.0,
                "battery_discharge_kWh": 0.0,
                "battery_losses_kWh": 0.0,
                "grid_electricity_kWh": total_el * 1000.0,
                "curtailed_RE_kWh": 0.0,
                "effective_grid_emission_factor_tCO2_MWh": grid_ef,
                "effective_grid_electricity_price_USD_MWh": grid_price,
                "pv_LCOE_USD_MWh": pv_lcoe,
                "wind_LCOE_USD_MWh": wind_lcoe,
                "geothermal_heat_price_USD_MWhth": geo_heat_price,
                "geothermal_heat_emission_factor_tCO2_MWhth": geo_heat_ef,
            })
            if not np.isfinite(grid_ef):
                d["invalid_reason"] += "missing_grid_EF;"
            d["grid_price_available_flag"] = bool(np.isfinite(grid_price))
            d["scenario_valid_flag"] = d["invalid_reason"] == ""
            rows.append(finalize_energy_row(d, args))

        # Variable RE scenarios use hourly RE profiles. If unavailable, S1/S2/S4 are skipped.
        if prof is None or n_hours == 0:
            continue

        pv_annual = to_float(r.get("pv_annual_kWh_per_kWp"), np.nansum(pv_profile))
        wind_annual = to_float(r.get("wind_annual_kWh_per_kW"), np.nansum(wind_profile))

        demand_hp_mwh = process_el + hp_el
        demand_hp_kwh_annual = demand_hp_mwh * 1000.0
        demand_hp_hourly = demand["total_hp_kwh"]
        demand_geo_mwh = process_el
        demand_geo_kwh_annual = demand_geo_mwh * 1000.0
        demand_geo_hourly = demand["total_geo_kwh"]

        avg_hp_kw = demand["battery_power_kW_HP_design"]
        avg_geo_kw = demand["battery_power_kW_geothermal_design"]

        for of in re_oversize_factors:
            for f_pv in pv_shares:
                f_wind = 1.0 - f_pv

                pv_cap_hp = (f_pv * of * demand_hp_kwh_annual / pv_annual) if pv_annual > 0 and demand_hp_kwh_annual > 0 else 0.0
                wind_cap_hp = (f_wind * of * demand_hp_kwh_annual / wind_annual) if wind_annual > 0 and demand_hp_kwh_annual > 0 else 0.0
                pv_gen_hp = pv_profile * pv_cap_hp
                wind_gen_hp = wind_profile * wind_cap_hp

                scenario_suffix = f"OF{of:g}_PV{f_pv:.2f}_W{f_wind:.2f}".replace(".", "p")

                # S1 no battery.
                if "S1" in args.scenarios or "all" in args.scenarios:
                    disp = dispatch_no_battery(pv_gen_hp, wind_gen_hp, demand_hp_hourly)
                    d = make_base_output(r, f"S1_grid_PVwind_HP_{scenario_suffix}", "S1_grid_PVwind_HP")
                    d.update(demand_metadata_for_output(demand, heat_pump_enabled=True))
                    d.update(disp)
                    d.update({
                        "heat_source": "heat_pump",
                        "electricity_source": "PV_wind_first_residual_grid",
                        "battery_enabled": False,
                        "geothermal_heat_enabled": False,
                        "heat_pump_COP": args.heat_pump_cop,
                        "annual_heat_pump_electricity_MWhe": hp_el,
                        "annual_total_electricity_demand_MWhe": demand_hp_mwh,
                        "annual_process_electricity_MWhe": process_el,
                        "annual_heat_demand_MWhth": heat,
                        "annual_cooling_demand_MWhth": demand["cooling_MWhth"],
                        "annual_CO2_t_per_1000kgads": co2,
                        "heat_from_heatpump_MWhth": heat,
                        "geothermal_heat_used_MWhth": 0.0,
                        "pv_capacity_kWp": pv_cap_hp,
                        "wind_capacity_kW": wind_cap_hp,
                        "battery_capacity_kWh": 0.0,
                        "battery_power_kW": 0.0,
                        "RE_oversize_factor": of,
                        "PV_target_generation_share": f_pv,
                        "wind_target_generation_share": f_wind,
                        "effective_grid_emission_factor_tCO2_MWh": grid_ef,
                        "effective_grid_electricity_price_USD_MWh": grid_price,
                        "pv_LCOE_USD_MWh": pv_lcoe,
                        "wind_LCOE_USD_MWh": wind_lcoe,
                        "geothermal_heat_price_USD_MWhth": geo_heat_price,
                        "geothermal_heat_emission_factor_tCO2_MWhth": geo_heat_ef,
                    })
                    if not np.isfinite(grid_ef):
                        d["invalid_reason"] += "missing_grid_EF;"
                    d["grid_price_available_flag"] = bool(np.isfinite(grid_price))
                    d["scenario_valid_flag"] = d["invalid_reason"] == ""
                    d = apply_wind_capacity_constraint(d, args)
                    rows.append(finalize_energy_row(d, args))

                # S2 battery.
                if "S2" in args.scenarios or "all" in args.scenarios:
                    for dur in battery_durations:
                        batt_power = avg_hp_kw
                        batt_cap = dur * batt_power
                        disp = dispatch_with_battery(
                            pv_gen_hp, wind_gen_hp, demand_hp_hourly,
                            battery_capacity_kwh=batt_cap,
                            battery_power_kw=batt_power,
                            roundtrip_eff=args.battery_roundtrip_efficiency,
                            initial_soc_frac=args.initial_soc_frac,
                        )
                        d = make_base_output(r, f"S2_PVwind_battery_grid_HP_{scenario_suffix}_B{dur:g}h".replace(".", "p"), "S2_PVwind_battery_grid_HP")
                        d.update(demand_metadata_for_output(demand, heat_pump_enabled=True))
                        d.update(disp)
                        d.update({
                            "heat_source": "heat_pump",
                            "electricity_source": "PV_wind_battery_residual_grid",
                            "battery_enabled": True,
                            "geothermal_heat_enabled": False,
                            "heat_pump_COP": args.heat_pump_cop,
                            "annual_heat_pump_electricity_MWhe": hp_el,
                            "annual_total_electricity_demand_MWhe": demand_hp_mwh,
                            "annual_process_electricity_MWhe": process_el,
                            "annual_heat_demand_MWhth": heat,
                            "annual_cooling_demand_MWhth": demand["cooling_MWhth"],
                            "annual_CO2_t_per_1000kgads": co2,
                            "heat_from_heatpump_MWhth": heat,
                            "geothermal_heat_used_MWhth": 0.0,
                            "pv_capacity_kWp": pv_cap_hp,
                            "wind_capacity_kW": wind_cap_hp,
                            "battery_capacity_kWh": batt_cap,
                            "battery_power_kW": batt_power,
                            "battery_duration_h": dur,
                            "battery_roundtrip_efficiency": args.battery_roundtrip_efficiency,
                            "RE_oversize_factor": of,
                            "PV_target_generation_share": f_pv,
                            "wind_target_generation_share": f_wind,
                            "effective_grid_emission_factor_tCO2_MWh": grid_ef,
                            "effective_grid_electricity_price_USD_MWh": grid_price,
                            "pv_LCOE_USD_MWh": pv_lcoe,
                            "wind_LCOE_USD_MWh": wind_lcoe,
                            "geothermal_heat_price_USD_MWhth": geo_heat_price,
                            "geothermal_heat_emission_factor_tCO2_MWhth": geo_heat_ef,
                        })
                        if not np.isfinite(grid_ef):
                            d["invalid_reason"] += "missing_grid_EF;"
                        d["grid_price_available_flag"] = bool(np.isfinite(grid_price))
                        d["scenario_valid_flag"] = d["invalid_reason"] == ""
                        d = apply_wind_capacity_constraint(d, args)
                        rows.append(finalize_energy_row(d, args))

                # S4 PV/wind + battery + geothermal heat.
                if ("S4" in args.scenarios or "all" in args.scenarios) and bool(r.get("geothermal_heat_eligible_flag", False)):
                    pv_cap_geo = (f_pv * of * demand_geo_kwh_annual / pv_annual) if pv_annual > 0 and demand_geo_kwh_annual > 0 else 0.0
                    wind_cap_geo = (f_wind * of * demand_geo_kwh_annual / wind_annual) if wind_annual > 0 and demand_geo_kwh_annual > 0 else 0.0
                    pv_gen_geo = pv_profile * pv_cap_geo
                    wind_gen_geo = wind_profile * wind_cap_geo
                    for dur in battery_durations:
                        batt_power = avg_geo_kw
                        batt_cap = dur * batt_power
                        disp = dispatch_with_battery(
                            pv_gen_geo, wind_gen_geo, demand_geo_hourly,
                            battery_capacity_kwh=batt_cap,
                            battery_power_kw=batt_power,
                            roundtrip_eff=args.battery_roundtrip_efficiency,
                            initial_soc_frac=args.initial_soc_frac,
                        )
                        d = make_base_output(r, f"S4_PVwind_battery_grid_geothermalHeat_{scenario_suffix}_B{dur:g}h".replace(".", "p"), "S4_PVwind_battery_grid_geothermalHeat")
                        d.update(demand_metadata_for_output(demand, heat_pump_enabled=False))
                        d.update(disp)
                        d.update({
                            "heat_source": "geothermal_heat",
                            "electricity_source": "PV_wind_battery_residual_grid",
                            "battery_enabled": True,
                            "geothermal_heat_enabled": True,
                            "heat_pump_COP": np.nan,
                            "annual_heat_pump_electricity_MWhe": 0.0,
                            "annual_total_electricity_demand_MWhe": demand_geo_mwh,
                            "annual_process_electricity_MWhe": process_el,
                            "annual_heat_demand_MWhth": heat,
                            "annual_cooling_demand_MWhth": demand["cooling_MWhth"],
                            "annual_CO2_t_per_1000kgads": co2,
                            "heat_from_heatpump_MWhth": 0.0,
                            "geothermal_heat_used_MWhth": heat,
                            "pv_capacity_kWp": pv_cap_geo,
                            "wind_capacity_kW": wind_cap_geo,
                            "battery_capacity_kWh": batt_cap,
                            "battery_power_kW": batt_power,
                            "battery_duration_h": dur,
                            "battery_roundtrip_efficiency": args.battery_roundtrip_efficiency,
                            "RE_oversize_factor": of,
                            "PV_target_generation_share": f_pv,
                            "wind_target_generation_share": f_wind,
                            "effective_grid_emission_factor_tCO2_MWh": grid_ef,
                            "effective_grid_electricity_price_USD_MWh": grid_price,
                            "pv_LCOE_USD_MWh": pv_lcoe,
                            "wind_LCOE_USD_MWh": wind_lcoe,
                            "geothermal_heat_price_USD_MWhth": geo_heat_price,
                            "geothermal_heat_emission_factor_tCO2_MWhth": geo_heat_ef,
                        })
                        if not np.isfinite(grid_ef):
                            d["invalid_reason"] += "missing_grid_EF;"
                        d["grid_price_available_flag"] = bool(np.isfinite(grid_price))
                        d["scenario_valid_flag"] = d["invalid_reason"] == ""
                        d = apply_wind_capacity_constraint(d, args)
                        rows.append(finalize_energy_row(d, args))

    return pd.DataFrame(rows)

def build_best_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = summary[summary["scenario_valid_flag"] == True].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["country_code", "province_id", "province_name", "operation_policy", "energy_scenario_base"]

    # Selection 1: lowest energy emissions if available, else lowest grid share then curtailment.
    tmp = valid.copy()
    tmp["sort_emissions"] = tmp["energy_emissions_tCO2"].fillna(np.inf)
    tmp["sort_grid"] = tmp["grid_backup_share"].fillna(np.inf)
    tmp["sort_curtail"] = tmp["curtailment_share_of_RE_generation"].fillna(np.inf)
    best_emission = tmp.sort_values(["sort_emissions", "sort_grid", "sort_curtail"]).groupby(group_cols, dropna=False).head(1).drop(columns=["sort_emissions", "sort_grid", "sort_curtail"])

    # Selection 2: lowest grid backup share, then lowest curtailment.
    tmp = valid.copy()
    tmp["sort_grid"] = tmp["grid_backup_share"].fillna(np.inf)
    tmp["sort_curtail"] = tmp["curtailment_share_of_RE_generation"].fillna(np.inf)
    best_grid = tmp.sort_values(["sort_grid", "sort_curtail"]).groupby(group_cols, dropna=False).head(1).drop(columns=["sort_grid", "sort_curtail"])

    return best_emission, best_grid



# =============================================================================
# Diagnostics and visualization
# =============================================================================

CONSULTING_PALETTE = {
    "navy": "#002060",
    "blue": "#1F77B4",
    "teal": "#00A3A1",
    "sky": "#6BAED6",
    "orange": "#F28E2B",
    "green": "#2CA25F",
    "gray": "#A6A6A6",
    "dark_gray": "#4D4D4D",
    "light_gray": "#E6E6E6",
    "very_light_gray": "#F7F7F7",
}

SCENARIO_COLOR = {
    "S0_grid_HP": CONSULTING_PALETTE["dark_gray"],
    "S1_grid_PVwind_HP": CONSULTING_PALETTE["sky"],
    "S2_PVwind_battery_grid_HP": CONSULTING_PALETTE["teal"],
    "S3_grid_geothermalHeat": CONSULTING_PALETTE["orange"],
    "S4_PVwind_battery_grid_geothermalHeat": CONSULTING_PALETTE["green"],
}

POLICY_COLOR_SEQUENCE = [
    CONSULTING_PALETTE["navy"],
    CONSULTING_PALETTE["teal"],
    CONSULTING_PALETTE["orange"],
    CONSULTING_PALETTE["sky"],
    CONSULTING_PALETTE["green"],
    CONSULTING_PALETTE["gray"],
]


def _scenario_label(s: str) -> str:
    labels = {
        "S0_grid_HP": "S0 Grid + HP",
        "S1_grid_PVwind_HP": "S1 PV/Wind + Grid + HP",
        "S2_PVwind_battery_grid_HP": "S2 PV/Wind + Battery + Grid + HP",
        "S3_grid_geothermalHeat": "S3 Grid + Geothermal Heat",
        "S4_PVwind_battery_grid_geothermalHeat": "S4 PV/Wind + Battery + Grid + Geothermal Heat",
    }
    return labels.get(str(s), str(s).replace("_", " "))


def _policy_label(s: str) -> str:
    return str(s).replace("_", " ").replace("-", " ").title()


def set_consulting_style() -> None:
    """Apply a clean consulting-style visual grammar."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans", "Liberation Sans"],
        "axes.edgecolor": CONSULTING_PALETTE["light_gray"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "grid.color": CONSULTING_PALETTE["light_gray"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
    })


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CONSULTING_PALETTE["light_gray"])
    ax.spines["bottom"].set_color(CONSULTING_PALETTE["light_gray"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-")
    ax.tick_params(axis="both", colors=CONSULTING_PALETTE["dark_gray"])


def _savefig(fig, path: Path, dpi: int = 260) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _safe_label_list(values) -> list[str]:
    return [_scenario_label(v) if str(v).startswith("S") else str(v).replace("_", "\n") for v in values]


def _boxplot_by_category(df: pd.DataFrame, category_col: str, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    data = df[[category_col, value_col]].dropna()
    if data.empty:
        return
    cats = data[category_col].dropna().drop_duplicates().tolist()
    arrays = []
    cats2 = []
    for c in cats:
        a = pd.to_numeric(data.loc[data[category_col] == c, value_col], errors="coerce").dropna().to_numpy()
        if len(a) > 0:
            arrays.append(a)
            cats2.append(c)
    if not arrays:
        return

    set_consulting_style()
    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(cats2)), 5.2))
    bp = ax.boxplot(arrays, tick_labels=_safe_label_list(cats2), showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(POLICY_COLOR_SEQUENCE[i % len(POLICY_COLOR_SEQUENCE)])
        patch.set_alpha(0.65)
        patch.set_edgecolor(CONSULTING_PALETTE["dark_gray"])
    for median in bp["medians"]:
        median.set_color(CONSULTING_PALETTE["navy"])
        median.set_linewidth(1.5)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    _style_axes(ax)
    _savefig(fig, out_path)


def write_geothermal_availability_diagnostics(annual: pd.DataFrame, out_dir: Path) -> None:
    diag_dir = out_dir / "diagnostics"
    fig_dir = out_dir / "figures"
    ensure_dir(diag_dir)
    ensure_dir(fig_dir)
    if annual is None or annual.empty or "geothermal_heat_eligible_flag" not in annual.columns:
        return

    prov_cols = ["country_code", "country_name", "province_id", "province_name", "geothermal_heat_eligible_flag",
                 "geothermal_operating_capacity_MW", "geothermal_heat_potential_score", "heat_rank"]
    prov_cols = [c for c in prov_cols if c in annual.columns]
    prov = annual[prov_cols].drop_duplicates(["country_code", "province_id", "province_name"]).copy()
    prov["geothermal_heat_eligible_flag"] = prov["geothermal_heat_eligible_flag"].astype(bool)
    prov.to_csv(diag_dir / "geothermal_eligible_provinces.csv", index=False, encoding="utf-8-sig")

    by_country = prov.groupby(["country_code", "country_name"], dropna=False).agg(
        n_provinces=("province_name", "count"),
        n_geothermal_eligible=("geothermal_heat_eligible_flag", "sum"),
    ).reset_index()
    by_country["geothermal_eligible_share"] = by_country["n_geothermal_eligible"] / by_country["n_provinces"].replace(0, np.nan)
    by_country.to_csv(diag_dir / "geothermal_eligibility_by_country.csv", index=False, encoding="utf-8-sig")

    set_consulting_style()
    plot = by_country.sort_values("n_geothermal_eligible", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [CONSULTING_PALETTE["orange"] if v > 0 else CONSULTING_PALETTE["light_gray"] for v in plot["n_geothermal_eligible"]]
    ax.bar(plot["country_code"], plot["n_geothermal_eligible"], color=colors)
    ax.set_title("Geothermal-eligible provinces by country", loc="left")
    ax.set_ylabel("Number of provinces")
    ax.set_xlabel("")
    _style_axes(ax)
    _savefig(fig, fig_dir / "geothermal_eligible_provinces_by_country.png")


def write_operation_policy_diagnostics(annual: pd.DataFrame, out_dir: Path) -> None:
    diag_dir = out_dir / "diagnostics"
    fig_dir = out_dir / "figures"
    ensure_dir(diag_dir)
    ensure_dir(fig_dir)
    if annual is None or annual.empty or "operation_policy" not in annual.columns:
        return

    metric_cols = [
        "annual_CO2_t_per_1000kgads",
        "annual_heat_demand_MWhth",
        "annual_process_electricity_MWhe",
        "SEC_total_MWh_tCO2_before_compression",
    ]
    metric_cols = [c for c in metric_cols if c in annual.columns]
    if not metric_cols:
        return

    op_summary = annual.groupby("operation_policy", dropna=False).agg({
        c: ["count", "mean", "median", "std", "min", "max"] for c in metric_cols
    })
    op_summary.columns = ["_".join(x).strip("_") for x in op_summary.columns]
    op_summary = op_summary.reset_index()
    op_summary.to_csv(diag_dir / "operation_policy_process_kpi_summary.csv", index=False, encoding="utf-8-sig")

    # Relative variation within each province across operation policies.
    var_records = []
    id_cols = ["country_code", "province_id", "province_name"]
    for metric in metric_cols:
        piv = annual.pivot_table(index=id_cols, columns="operation_policy", values=metric, aggfunc="mean")
        rel = (piv.max(axis=1) - piv.min(axis=1)) / piv.median(axis=1).replace(0, np.nan)
        tmp = rel.reset_index(name=f"{metric}_relative_range")
        var_records.append(tmp)
    variation = var_records[0]
    for tmp in var_records[1:]:
        variation = variation.merge(tmp, on=id_cols, how="outer")
    variation.to_csv(diag_dir / "operation_policy_within_province_relative_variation.csv", index=False, encoding="utf-8-sig")

    # Short warning text for interpretation.
    warnings_txt = []
    for metric in metric_cols:
        col = f"{metric}_relative_range"
        med = pd.to_numeric(variation[col], errors="coerce").median()
        if np.isfinite(med) and med < 0.03:
            warnings_txt.append(f"{metric}: median within-province policy variation = {med:.2%}; operation-policy effect is small in module 02 output.")
        elif np.isfinite(med):
            warnings_txt.append(f"{metric}: median within-province policy variation = {med:.2%}.")
    (diag_dir / "operation_policy_variation_interpretation.txt").write_text("\n".join(warnings_txt), encoding="utf-8")

    # Figure 1: normalized KPI means by operation policy.
    means = op_summary[["operation_policy"] + [f"{c}_mean" for c in metric_cols]].copy()
    long = means.melt(id_vars="operation_policy", var_name="metric", value_name="mean_value")
    long["metric"] = long["metric"].str.replace("_mean", "", regex=False)
    long["normalized_mean"] = long.groupby("metric")["mean_value"].transform(lambda s: s / s.mean() if s.mean() != 0 else np.nan)
    pivot = long.pivot(index="operation_policy", columns="metric", values="normalized_mean")

    set_consulting_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    pivot.plot(kind="bar", ax=ax, color=[POLICY_COLOR_SEQUENCE[i % len(POLICY_COLOR_SEQUENCE)] for i in range(len(pivot.columns))], width=0.78)
    ax.set_title("Operation-policy effect on annual process KPIs", loc="left")
    ax.set_ylabel("Normalized mean (mean across policies = 1)")
    ax.set_xlabel("")
    ax.set_xticklabels([_policy_label(x) for x in pivot.index], rotation=20, ha="right")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _style_axes(ax)
    _savefig(fig, fig_dir / "operation_policy_process_kpis_normalized.png")

    # Figure 2: distribution of relative variation across provinces.
    rel_cols = [c for c in variation.columns if c.endswith("_relative_range")]
    if rel_cols:
        rel_long = variation[rel_cols].melt(var_name="metric", value_name="relative_range").dropna()
        rel_long["metric"] = rel_long["metric"].str.replace("_relative_range", "", regex=False)
        set_consulting_style()
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        cats = rel_long["metric"].drop_duplicates().tolist()
        arrays = [rel_long.loc[rel_long["metric"] == c, "relative_range"].to_numpy() for c in cats]
        bp = ax.boxplot(arrays, tick_labels=[c.replace("_", "\n") for c in cats], showfliers=False, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(POLICY_COLOR_SEQUENCE[i % len(POLICY_COLOR_SEQUENCE)])
            patch.set_alpha(0.65)
            patch.set_edgecolor(CONSULTING_PALETTE["dark_gray"])
        for median in bp["medians"]:
            median.set_color(CONSULTING_PALETTE["navy"])
            median.set_linewidth(1.5)
        ax.set_title("Within-province variation across operation policies", loc="left")
        ax.set_ylabel("Relative range across policies")
        _style_axes(ax)
        _savefig(fig, fig_dir / "operation_policy_within_province_variation_boxplot.png")


def _country_code_from_name(name) -> str:
    return _asean_country_code_from_name(name)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _make_spatial_keys(gdf):
    gdf = gdf.copy()
    ccol = _pick_column(gdf, ["country_code", "iso3", "ISO3", "GID_0", "gid_0", "country_id", "ADM0_CODE", "name_0", "NAME_0", "country", "COUNTRY"])
    pcol = _pick_column(gdf, ["province_id", "GID_1", "gid_1", "adm1_id", "ADM1_CODE", "province_code", "id", "ID"])
    ncol = _pick_column(gdf, ["province_name", "NAME_1", "name_1", "adm1_name", "province", "state", "NAME_EN", "name"])

    if ccol is None:
        raise ValueError("Could not infer country column in spatial GPKG.")
    if ncol is None and pcol is None:
        raise ValueError("Could not infer province id/name column in spatial GPKG.")

    gdf["_map_country_code"] = gdf[ccol].map(lambda x: str(x).strip().upper() if not pd.isna(x) else "")
    bad = ~gdf["_map_country_code"].str.match(r"^[A-Z]{3}$", na=False)
    if bad.any():
        mapped = gdf.loc[bad, ccol].map(_country_code_from_name)
        gdf.loc[bad & mapped.astype(str).ne(""), "_map_country_code"] = mapped[mapped.astype(str).ne("")]

    gdf["_map_province_id"] = gdf[pcol].map(lambda x: str(x).strip()) if pcol is not None else ""
    gdf["_map_province_name"] = gdf[ncol].map(lambda x: str(x).strip()) if ncol is not None else ""
    gdf["_map_key_id"] = (
        gdf["_map_country_code"].map(norm_text) + "|" +
        gdf["_map_province_id"].map(norm_text) + "|" +
        gdf["_map_province_name"].map(norm_text)
    )
    gdf["_map_key_name"] = gdf["_map_country_code"].map(norm_text) + "|" + gdf["_map_province_name"].map(norm_text)
    return gdf


def _make_data_map_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["country_code", "province_id", "province_name"]:
        if c not in out.columns:
            out[c] = ""
    out["_map_key_id"] = (
        out["country_code"].map(norm_text) + "|" +
        out["province_id"].astype(str).map(norm_text) + "|" +
        out["province_name"].astype(str).map(norm_text)
    )
    out["_map_key_name"] = out["country_code"].map(norm_text) + "|" + out["province_name"].astype(str).map(norm_text)
    return out


def _join_spatial_data(gdf, data: pd.DataFrame):
    data = _make_data_map_keys(data)
    data["_data_present"] = True
    data_id = data.drop_duplicates("_map_key_id", keep="first")
    joined = gdf.merge(data_id, on="_map_key_id", how="left", suffixes=("", "_data"))
    matched_id = int(joined["_data_present"].fillna(False).astype(bool).sum()) if "_data_present" in joined.columns else 0

    # If id match is weak, use country+province_name fallback.
    if matched_id < max(5, 0.25 * len(data)):
        keep_cols = [c for c in data.columns if c not in ["_map_key_id"]]
        data_name = data[keep_cols].drop_duplicates("_map_key_name", keep="first")
        joined = gdf.merge(data_name, on="_map_key_name", how="left", suffixes=("", "_data"))
    return joined


def make_spatial_visualizations(summary: pd.DataFrame, annual_operation: pd.DataFrame | None, spatial_gpkg: Path | None, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    diag_dir = out_dir / "diagnostics"
    maps_dir = out_dir / "maps"
    ensure_dir(fig_dir)
    ensure_dir(diag_dir)
    ensure_dir(maps_dir)

    if spatial_gpkg is None or not Path(spatial_gpkg).exists():
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text(
            f"Spatial GPKG not found: {spatial_gpkg}. Point-map diagnostics were created instead.",
            encoding="utf-8",
        )
        return

    try:
        import geopandas as gpd
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text(
            f"geopandas or spatial plotting dependency is unavailable: {exc}",
            encoding="utf-8",
        )
        return

    try:
        gdf = gpd.read_file(spatial_gpkg)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf = _make_spatial_keys(gdf)
        pd.DataFrame([{
            "spatial_file": str(spatial_gpkg),
            "n_features": int(len(gdf)),
            "crs": str(gdf.crs),
            "columns": ";".join(map(str, gdf.columns)),
        }]).to_csv(diag_dir / "map_spatial_input_used.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text(
            f"Could not read or key spatial GPKG {spatial_gpkg}: {exc}",
            encoding="utf-8",
        )
        return

    if summary is None or summary.empty:
        return
    valid = summary[summary["scenario_valid_flag"] == True].copy()
    if valid.empty:
        return

    # Map 1: minimum energy-emission intensity by province.
    if "energy_emission_intensity_tCO2_per_tCO2captured" in valid.columns:
        best_map = valid.dropna(subset=["energy_emission_intensity_tCO2_per_tCO2captured"]).sort_values(
            "energy_emission_intensity_tCO2_per_tCO2captured"
        ).groupby(["country_code", "province_id", "province_name"], dropna=False).head(1)
        best_map.to_csv(diag_dir / "best_energy_emission_intensity_by_province.csv", index=False, encoding="utf-8-sig")
        joined = _join_spatial_data(gdf, best_map)
        matched = joined["energy_emission_intensity_tCO2_per_tCO2captured"].notna().sum()
        pd.DataFrame([{
            "map": "minimum_energy_emission_intensity",
            "spatial_file": str(spatial_gpkg),
            "n_spatial_features": len(joined),
            "n_data_rows": len(best_map),
            "n_matched_features": int(matched),
        }]).to_csv(diag_dir / "map_join_diagnostics_min_energy_emission.csv", index=False, encoding="utf-8-sig")

        if matched > 0:
            try:
                joined.to_file(maps_dir / "map_min_energy_emission_intensity_choropleth_data.geojson", driver="GeoJSON")
            except Exception as exc:
                (maps_dir / "map_min_energy_emission_intensity_choropleth_data_ERROR.txt").write_text(str(exc), encoding="utf-8")

            cmap = LinearSegmentedColormap.from_list(
                "consulting_blue_teal",
                [CONSULTING_PALETTE["very_light_gray"], "#B9DDE7", CONSULTING_PALETTE["teal"], CONSULTING_PALETTE["navy"]]
            )
            set_consulting_style()
            fig, ax = plt.subplots(figsize=(11, 8.5))
            joined.plot(
                column="energy_emission_intensity_tCO2_per_tCO2captured",
                ax=ax,
                cmap=cmap,
                legend=True,
                linewidth=0.18,
                edgecolor="white",
                missing_kwds={"color": CONSULTING_PALETTE["light_gray"], "edgecolor": "white", "hatch": "///", "label": "No data"},
                legend_kwds={"label": "tCO2 emitted per tCO2 captured", "shrink": 0.75},
            )
            joined.boundary.plot(ax=ax, linewidth=0.10, color="white")
            ax.set_title("Minimum energy-emission intensity by province", loc="left")
            ax.set_axis_off()
            _savefig(fig, fig_dir / "map_min_energy_emission_intensity_choropleth.png", dpi=280)

    # Map 2: best scenario by province.
    if "energy_scenario_base" in valid.columns and "energy_emission_intensity_tCO2_per_tCO2captured" in valid.columns:
        best_scen = valid.dropna(subset=["energy_emission_intensity_tCO2_per_tCO2captured"]).sort_values(
            "energy_emission_intensity_tCO2_per_tCO2captured"
        ).groupby(["country_code", "province_id", "province_name"], dropna=False).head(1)
        joined = _join_spatial_data(gdf, best_scen)
        try:
            joined.to_file(maps_dir / "map_best_energy_scenario_by_province_data.geojson", driver="GeoJSON")
        except Exception as exc:
            (maps_dir / "map_best_energy_scenario_by_province_data_ERROR.txt").write_text(str(exc), encoding="utf-8")
        set_consulting_style()
        fig, ax = plt.subplots(figsize=(11, 8.5))
        joined.boundary.plot(ax=ax, linewidth=0.20, color="white")
        # Background for no data.
        joined.plot(ax=ax, color=CONSULTING_PALETTE["light_gray"], edgecolor="white", linewidth=0.18)
        for scen, color in SCENARIO_COLOR.items():
            sub = joined[joined.get("energy_scenario_base", pd.Series(dtype=str)).astype(str) == scen]
            if not sub.empty:
                sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.18, label=_scenario_label(scen))
        ax.set_title("Best energy-supply scenario by province", loc="left")
        ax.set_axis_off()
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=1)
        _savefig(fig, fig_dir / "map_best_energy_scenario_by_province.png", dpi=280)

    # Map 3: geothermal eligibility.
    if annual_operation is not None and not annual_operation.empty and "geothermal_heat_eligible_flag" in annual_operation.columns:
        gdata = annual_operation.drop_duplicates(["country_code", "province_id", "province_name"]).copy()
        gdata["geothermal_heat_eligible_flag"] = gdata["geothermal_heat_eligible_flag"].astype(bool)
        joined = _join_spatial_data(gdf, gdata)
        try:
            joined.to_file(maps_dir / "map_geothermal_heat_eligibility_data.geojson", driver="GeoJSON")
        except Exception as exc:
            (maps_dir / "map_geothermal_heat_eligibility_data_ERROR.txt").write_text(str(exc), encoding="utf-8")
        set_consulting_style()
        fig, ax = plt.subplots(figsize=(11, 8.5))
        joined.boundary.plot(ax=ax, linewidth=0.20, color="white")
        joined.plot(ax=ax, color=CONSULTING_PALETTE["light_gray"], edgecolor="white", linewidth=0.18, label="Not geothermal eligible")
        sub = joined[joined.get("geothermal_heat_eligible_flag", pd.Series(dtype=bool)).fillna(False).astype(bool)]
        if not sub.empty:
            sub.plot(ax=ax, color=CONSULTING_PALETTE["orange"], edgecolor="white", linewidth=0.18, label="Geothermal eligible")
        ax.set_title("Geothermal heat eligibility used in S3/S4", loc="left")
        ax.set_axis_off()
        ax.legend(loc="lower left")
        _savefig(fig, fig_dir / "map_geothermal_heat_eligibility.png", dpi=280)

    # Map 4: minimum grid-backup share by province.
    if "grid_backup_share" in valid.columns:
        grid_map = valid.dropna(subset=["grid_backup_share"]).sort_values(
            "grid_backup_share"
        ).groupby(["country_code", "province_id", "province_name"], dropna=False).head(1)
        if not grid_map.empty:
            grid_map.to_csv(diag_dir / "best_grid_backup_share_by_province.csv", index=False, encoding="utf-8-sig")
            joined = _join_spatial_data(gdf, grid_map)
            matched = joined["grid_backup_share"].notna().sum()
            pd.DataFrame([{
                "map": "minimum_grid_backup_share",
                "spatial_file": str(spatial_gpkg),
                "n_spatial_features": len(joined),
                "n_data_rows": len(grid_map),
                "n_matched_features": int(matched),
            }]).to_csv(diag_dir / "map_join_diagnostics_min_grid_backup_share.csv", index=False, encoding="utf-8-sig")

            if matched > 0:
                try:
                    joined.to_file(maps_dir / "map_min_grid_backup_share_data.geojson", driver="GeoJSON")
                except Exception as exc:
                    (maps_dir / "map_min_grid_backup_share_data_ERROR.txt").write_text(str(exc), encoding="utf-8")

                cmap = LinearSegmentedColormap.from_list(
                    "consulting_grid_backup",
                    [CONSULTING_PALETTE["very_light_gray"], "#D7ECF2", CONSULTING_PALETTE["sky"], CONSULTING_PALETTE["navy"]]
                )
                set_consulting_style()
                fig, ax = plt.subplots(figsize=(11, 8.5))
                joined.plot(
                    column="grid_backup_share",
                    ax=ax,
                    cmap=cmap,
                    legend=True,
                    linewidth=0.18,
                    edgecolor="white",
                    missing_kwds={"color": CONSULTING_PALETTE["light_gray"], "edgecolor": "white", "hatch": "///", "label": "No data"},
                    legend_kwds={"label": "Minimum grid-backup share", "shrink": 0.75},
                )
                joined.boundary.plot(ax=ax, linewidth=0.10, color="white")
                ax.set_title("Minimum grid-backup share by province", loc="left")
                ax.set_axis_off()
                _savefig(fig, fig_dir / "map_min_grid_backup_share_choropleth.png", dpi=280)


def make_energy_supply_visualizations(
    summary: pd.DataFrame,
    best_emission: pd.DataFrame,
    best_grid: pd.DataFrame,
    out_dir: Path,
    spatial_gpkg: Path | None = None,
    annual_operation: pd.DataFrame | None = None,
) -> None:
    fig_dir = out_dir / "figures"
    diag_dir = out_dir / "diagnostics"
    ensure_dir(fig_dir)
    ensure_dir(diag_dir)
    set_consulting_style()

    if annual_operation is not None:
        write_operation_policy_diagnostics(annual_operation, out_dir)
        write_geothermal_availability_diagnostics(annual_operation, out_dir)

    if summary.empty:
        return

    valid = summary[summary["scenario_valid_flag"] == True].copy()
    if valid.empty:
        return

    # Summary table by scenario and operation policy.
    summary_cols = [
        "energy_emission_intensity_tCO2_per_tCO2captured",
        "grid_backup_share",
        "renewable_share_of_electricity",
        "curtailment_share_of_RE_generation",
        "pv_capacity_kWp",
        "wind_capacity_kW",
        "battery_capacity_kWh",
        "annual_total_electricity_demand_MWhe",
    ]
    agg_dict = {c: ["count", "mean", "median", "std", "min", "max"] for c in summary_cols if c in valid.columns}
    if agg_dict:
        sp = valid.groupby(["energy_scenario_base", "operation_policy"], dropna=False).agg(agg_dict)
        sp.columns = ["_".join(x).strip("_") for x in sp.columns]
        sp = sp.reset_index()
        sp.to_csv(diag_dir / "scenario_policy_energy_summary.csv", index=False, encoding="utf-8-sig")

    # 1. Energy emissions intensity by scenario and operation policy: median grouped bar.
    if "energy_emission_intensity_tCO2_per_tCO2captured" in valid.columns:
        med = valid.groupby(["energy_scenario_base", "operation_policy"], dropna=False)["energy_emission_intensity_tCO2_per_tCO2captured"].median().reset_index()
        if not med.empty:
            pivot = med.pivot(index="energy_scenario_base", columns="operation_policy", values="energy_emission_intensity_tCO2_per_tCO2captured")
            pivot.index = [_scenario_label(x) for x in pivot.index]
            fig, ax = plt.subplots(figsize=(13.2, 6.2))
            pivot.plot(kind="bar", ax=ax, width=0.82, color=[POLICY_COLOR_SEQUENCE[i % len(POLICY_COLOR_SEQUENCE)] for i in range(len(pivot.columns))])
            ax.set_title("Energy-emission intensity by scenario and operation policy", loc="left")
            ax.set_ylabel("tCO2 emitted per tCO2 captured")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=22)
            ax.legend(title="Operation policy", loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(5, len(pivot.columns)))
            _style_axes(ax)
            _savefig(fig, fig_dir / "energy_emission_intensity_by_scenario_policy.png")

        _boxplot_by_category(
            valid,
            "energy_scenario_base",
            "energy_emission_intensity_tCO2_per_tCO2captured",
            "Energy-emission intensity distribution by scenario",
            "tCO2 emitted per tCO2 captured",
            fig_dir / "energy_emission_intensity_boxplot_by_scenario.png",
        )

    # 2. Grid backup share by scenario.
    if "grid_backup_share" in valid.columns:
        _boxplot_by_category(
            valid,
            "energy_scenario_base",
            "grid_backup_share",
            "Grid backup share by energy scenario",
            "Grid backup share of electricity demand",
            fig_dir / "grid_backup_share_by_scenario.png",
        )

    # 3. Renewable used vs curtailment trade-off.
    if {"renewable_share_of_electricity", "curtailment_share_of_RE_generation"}.issubset(valid.columns):
        scat = valid.dropna(subset=["renewable_share_of_electricity", "curtailment_share_of_RE_generation"]).copy()
        if not scat.empty:
            size = pd.to_numeric(scat.get("battery_capacity_kWh", 0.0), errors="coerce").fillna(0.0)
            s = 18 + 70 * (size / max(float(size.max()), 1e-12)) if size.max() > 0 else 22
            colors = scat["energy_scenario_base"].map(SCENARIO_COLOR).fillna(CONSULTING_PALETTE["gray"])
            fig, ax = plt.subplots(figsize=(8.6, 6.2))
            ax.scatter(scat["renewable_share_of_electricity"], scat["curtailment_share_of_RE_generation"], s=s, alpha=0.30, c=colors)
            ax.set_title("Renewable share vs curtailment trade-off", loc="left")
            ax.set_xlabel("Renewable share of electricity demand")
            ax.set_ylabel("Curtailment share of RE generation")
            _style_axes(ax, grid_axis="both")
            _savefig(fig, fig_dir / "renewable_share_vs_curtailment_tradeoff.png")

    # 4. PV-wind-battery sizing scatter.
    if {"pv_capacity_kWp", "wind_capacity_kW", "battery_capacity_kWh"}.issubset(valid.columns):
        cap = valid[(pd.to_numeric(valid["pv_capacity_kWp"], errors="coerce").fillna(0) > 0) | (pd.to_numeric(valid["wind_capacity_kW"], errors="coerce").fillna(0) > 0)].copy()
        if not cap.empty:
            batt = pd.to_numeric(cap["battery_capacity_kWh"], errors="coerce").fillna(0.0)
            s = 18 + 90 * batt / max(float(batt.max()), 1e-12) if batt.max() > 0 else 22
            colors = cap["energy_scenario_base"].map(SCENARIO_COLOR).fillna(CONSULTING_PALETTE["gray"])
            fig, ax = plt.subplots(figsize=(8.6, 6.2))
            ax.scatter(cap["pv_capacity_kWp"], cap["wind_capacity_kW"], s=s, alpha=0.28, c=colors)
            ax.set_title("PV-wind-battery sizing candidates", loc="left")
            ax.set_xlabel("PV capacity (kWp)")
            ax.set_ylabel("Wind capacity (kW)")
            _style_axes(ax, grid_axis="both")
            _savefig(fig, fig_dir / "pv_wind_battery_sizing_scatter.png")

    # 5. Point map retained as fallback/diagnostic.
    if {"longitude", "latitude", "energy_emission_intensity_tCO2_per_tCO2captured"}.issubset(valid.columns):
        mapsrc = valid.dropna(subset=["longitude", "latitude", "energy_emission_intensity_tCO2_per_tCO2captured"]).copy()
        if not mapsrc.empty:
            best_map = mapsrc.sort_values("energy_emission_intensity_tCO2_per_tCO2captured").groupby(
                ["country_code", "province_id", "province_name"], dropna=False
            ).head(1)
            best_map.to_csv(diag_dir / "best_energy_emission_intensity_by_province.csv", index=False, encoding="utf-8-sig")
            fig, ax = plt.subplots(figsize=(8.2, 7.0))
            sc = ax.scatter(best_map["longitude"], best_map["latitude"], c=best_map["energy_emission_intensity_tCO2_per_tCO2captured"], s=28, alpha=0.85, cmap="YlGnBu_r")
            ax.set_title("Minimum energy-emission intensity by province (point diagnostic)", loc="left")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            _style_axes(ax, grid_axis="both")
            fig.colorbar(sc, ax=ax, label="tCO2 emitted per tCO2 captured")
            _savefig(fig, fig_dir / "asean_min_energy_emission_intensity_pointmap.png")

    # 6. Heat pump vs geothermal heat benefit.
    if {"energy_scenario_base", "annual_total_electricity_demand_MWhe", "energy_emission_intensity_tCO2_per_tCO2captured"}.issubset(valid.columns):
        pair = valid[valid["energy_scenario_base"].isin(["S0_grid_HP", "S2_PVwind_battery_grid_HP", "S3_grid_geothermalHeat", "S4_PVwind_battery_grid_geothermalHeat"])].copy()
        if not pair.empty:
            med = pair.groupby("energy_scenario_base", dropna=False).agg(
                annual_total_electricity_demand_MWhe=("annual_total_electricity_demand_MWhe", "median"),
                energy_emission_intensity_tCO2_per_tCO2captured=("energy_emission_intensity_tCO2_per_tCO2captured", "median"),
            ).reset_index()
            med.to_csv(diag_dir / "heatpump_vs_geothermal_heat_median_summary.csv", index=False, encoding="utf-8-sig")
            med["scenario_label"] = med["energy_scenario_base"].map(_scenario_label)
            fig, ax = plt.subplots(figsize=(11, 5.5))
            x = np.arange(len(med))
            width = 0.38
            ax.bar(x - width/2, med["annual_total_electricity_demand_MWhe"], width=width, color=CONSULTING_PALETTE["navy"], label="Median electricity demand")
            ax2 = ax.twinx()
            ax2.bar(x + width/2, med["energy_emission_intensity_tCO2_per_tCO2captured"], width=width, color=CONSULTING_PALETTE["teal"], label="Median emission intensity")
            ax.set_title("Heat-pump and geothermal-heat scenario comparison", loc="left")
            ax.set_ylabel("MWhe/year per 1000 kg ads")
            ax2.set_ylabel("tCO2/tCO2 captured")
            ax.set_xticks(x)
            ax.set_xticklabels(med["scenario_label"], rotation=20, ha="right")
            _style_axes(ax)
            ax2.spines["top"].set_visible(False)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
            _savefig(fig, fig_dir / "heatpump_vs_geothermal_heat_comparison.png")

    # 7. Scenario validity and row availability.
    if {"energy_scenario_base", "scenario_valid_flag"}.issubset(summary.columns):
        validity = summary.groupby(["energy_scenario_base", "scenario_valid_flag"], dropna=False).size().reset_index(name="n_rows")
        pivot = validity.pivot(index="energy_scenario_base", columns="scenario_valid_flag", values="n_rows").fillna(0)
        pivot.index = [_scenario_label(x) for x in pivot.index]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        pivot.plot(kind="bar", stacked=True, ax=ax, color=[CONSULTING_PALETTE["orange"], CONSULTING_PALETTE["teal"]])
        ax.set_title("Scenario row availability and validity", loc="left")
        ax.set_ylabel("Number of evaluated rows")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=22)
        ax.legend(["Invalid", "Valid"], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        _style_axes(ax)
        _savefig(fig, fig_dir / "scenario_validity_by_scenario.png")

        invalid = summary[summary["scenario_valid_flag"] == False].copy()
        if not invalid.empty and "invalid_reason" in invalid.columns:
            invalid_reason = invalid.groupby(["energy_scenario_base", "invalid_reason"], dropna=False).size().reset_index(name="n_rows")
            invalid_reason.to_csv(diag_dir / "invalid_reason_breakdown.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["energy_scenario_base", "invalid_reason", "n_rows"]).to_csv(
                diag_dir / "invalid_reason_breakdown.csv", index=False, encoding="utf-8-sig"
            )

    # 8. Spatial choropleth maps using the user's province map, when available.
    if spatial_gpkg is not None:
        make_spatial_visualizations(summary, annual_operation, spatial_gpkg, out_dir)


def make_grid_join_diagnostics(annual_with_grid: pd.DataFrame, grid: pd.DataFrame, out_dir: Path) -> None:
    """Write grid-load and grid-join diagnostics for troubleshooting."""
    diag_dir = out_dir / "diagnostics"
    ensure_dir(diag_dir)

    if grid is None or grid.empty:
        pd.DataFrame([{
            "metric": "grid_rows_loaded",
            "value": 0,
            "notes": "No grid table loaded. Check --grid-csv or 00.RESOURCES/04_GRID.",
        }]).to_csv(diag_dir / "grid_load_diagnostics.csv", index=False, encoding="utf-8-sig")
        return

    # Country-level grid table coverage.
    grid_country = grid.groupby("country_code", dropna=False).agg(
        n_grid_rows=("country_code", "size"),
        n_with_grid_EF=("grid_emission_factor_tCO2_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_with_grid_price=("grid_electricity_price_USD_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_with_industrial_tariff=("industrial_tariff_USD_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        source_files=("source_file_grid", lambda s: ";".join(sorted(set(str(x) for x in s.dropna())))),
    ).reset_index()
    grid_country.to_csv(diag_dir / "grid_load_diagnostics_by_country.csv", index=False, encoding="utf-8-sig")

    # Annual rows after join.
    ann_country = annual_with_grid.groupby("country_code", dropna=False).agg(
        n_annual_rows=("country_code", "size"),
        n_rows_with_grid_EF=("grid_emission_factor_tCO2_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_rows_with_grid_price=("grid_electricity_price_USD_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        n_rows_with_industrial_tariff=("industrial_tariff_USD_MWh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
    ).reset_index()
    ann_country["grid_EF_join_coverage"] = ann_country["n_rows_with_grid_EF"] / ann_country["n_annual_rows"].replace(0, np.nan)
    ann_country["grid_price_join_coverage"] = ann_country["n_rows_with_grid_price"] / ann_country["n_annual_rows"].replace(0, np.nan)
    ann_country.to_csv(diag_dir / "grid_join_diagnostics_by_country.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# Main build
# =============================================================================

def resolve_paths(args):
    project_dir = Path(args.project_dir)
    if args.tea_dir:
        tea_dir = Path(args.tea_dir)
    else:
        candidates = [project_dir / "02.TEA_LCOD", project_dir.parent / "02.TEA_LCOD"]
        tea_dir = first_existing(candidates) or candidates[0]

    root_5dac = project_dir.parent if project_dir.name.upper().endswith("PYTHON") else project_dir

    out_dir = Path(args.out_dir) if args.out_dir else tea_dir / "03_ENERGY_SUPPLY_EVALUATOR"
    ensure_dir(out_dir)
    for sub in ["inputs", "annual_results", "dispatch_samples", "diagnostics", "figures", "maps"]:
        ensure_dir(out_dir / sub)

    annual_csv = Path(args.annual_operation_csv) if args.annual_operation_csv else tea_dir / "02_ANNUAL_DYNAMIC_OPERATION" / "annual_results" / "annual_operation_summary_by_province_policy.csv"

    re_hourly_csv = Path(args.re_hourly_csv) if args.re_hourly_csv else root_5dac / "00.TEMPORAL_DATA" / "RE_HOURLY_SUPPLY" / f"RE_hourly_supply_profile_{args.year}.csv"
    re_summary_csv = Path(args.re_summary_csv) if args.re_summary_csv else root_5dac / "00.TEMPORAL_DATA" / "RE_HOURLY_SUPPLY" / "summary" / f"RE_supply_summary_by_province_{args.year}.csv"
    if not re_summary_csv.exists():
        re_summary_csv = None

    grid_dir = Path(args.grid_dir) if args.grid_dir else root_5dac / "00.RESOURCES" / "04_GRID"
    grid_csv = Path(args.grid_csv) if args.grid_csv else None

    geo_csv = Path(args.geothermal_csv) if args.geothermal_csv else root_5dac / "00.RESOURCES" / "02_RENEWABLES" / "renewable_resource_geothermal_joinable.csv"

    spatial_dir = Path(args.spatial_dir) if args.spatial_dir else root_5dac / "00.SPATIAL_MAP"
    # User's canonical ASEAN province boundary generated by check_asean_map.py.
    # This must be preferred over generic GPKG auto-discovery because other GPKGs
    # may exist under 00.SPATIAL_MAP or NASA_POWER/mean_maps.
    default_spatial_gpkg = spatial_dir / "ASEAN" / "ASEAN_PROVINCES_LEVEL1.gpkg"
    if args.spatial_gpkg:
        spatial_gpkg = Path(args.spatial_gpkg)
    elif default_spatial_gpkg.exists():
        spatial_gpkg = default_spatial_gpkg
    else:
        spatial_gpkg = find_latest(
            [
                "ASEAN_PROVINCES_LEVEL1.gpkg",
                "*ASEAN*PROVINCES*.gpkg",
                "*PROVINCES*LEVEL1*.gpkg",
                "*.gpkg",
            ],
            [spatial_dir / "ASEAN", spatial_dir],
        )

    hourly_operation_csv = Path(args.hourly_operation_profile_csv) if getattr(args, "hourly_operation_profile_csv", None) else tea_dir / "02_ANNUAL_DYNAMIC_OPERATION" / "predictions" / "hourly_operation_profile_by_province_policy.csv"
    if not hourly_operation_csv.exists():
        hourly_operation_csv = None

    return {
        "project_dir": project_dir,
        "tea_dir": tea_dir,
        "root_5dac": root_5dac,
        "out_dir": out_dir,
        "annual_csv": annual_csv,
        "hourly_operation_csv": hourly_operation_csv,
        "re_hourly_csv": re_hourly_csv,
        "re_summary_csv": re_summary_csv,
        "grid_dir": grid_dir,
        "grid_csv": grid_csv,
        "geo_csv": geo_csv,
        "spatial_dir": spatial_dir,
        "spatial_gpkg": spatial_gpkg,
    }


def build(args):
    paths = resolve_paths(args)
    out_dir = paths["out_dir"]

    print("=" * 100)
    print("03 ENERGY SUPPLY EVALUATOR")
    print("=" * 100)
    for k, v in paths.items():
        print(f"{k:20s}: {v}")
    print("=" * 100)

    if not paths["annual_csv"].exists():
        raise FileNotFoundError(f"Annual operation summary not found: {paths['annual_csv']}")
    if not paths["re_hourly_csv"].exists():
        raise FileNotFoundError(f"RE hourly profile not found: {paths['re_hourly_csv']}")

    print("[LOAD] Annual operation summary")
    annual_raw = read_csv_auto(paths["annual_csv"])
    annual = canonicalize_annual_operation(annual_raw)
    annual["demand_profile_method"] = "flat_annual_from_module02"

    print("[LOAD] Grid assumptions")
    grid = load_grid_assumptions(paths["grid_dir"], paths["grid_csv"])
    if not grid.empty and "source_file_grid" in grid.columns:
        grid_sources = sorted(set(str(x) for x in grid["source_file_grid"].dropna()))
        print("[INFO] Grid source files used:")
        for src in grid_sources:
            print(f"       - {src}")
        print(f"[INFO] Grid rows loaded: {len(grid):,}; countries: {grid['country_code'].nunique():,}")
    else:
        print("[WARNING] No grid assumptions were loaded.")
    annual = attach_grid(annual, grid)
    make_grid_join_diagnostics(annual, grid, out_dir)

    print("[LOAD] Geothermal resource eligibility")
    geo = load_geothermal(paths["geo_csv"])
    annual = attach_geothermal(annual, geo)
    write_geothermal_availability_diagnostics(annual, out_dir)

    print("[LOAD] RE annual summary")
    re_summary = load_re_summary(paths["re_summary_csv"], paths["re_hourly_csv"])

    print("[LOAD] RE hourly profiles")
    re_profiles = load_re_hourly_profiles(paths["re_hourly_csv"], max_provinces=args.max_provinces)
    print(f"[INFO] Loaded hourly RE profiles for {len(re_profiles):,} provinces")

    # If max_provinces was used, filter annual to those provinces.
    if args.max_provinces and args.max_provinces > 0:
        annual = annual[annual["province_key"].isin(re_profiles.keys())].copy()

    print("[LOAD] Hourly DAC operation profile from module 02")
    hourly_profiles = {}
    hourly_profile_diag = pd.DataFrame([{
        "check": "hourly_operation_profile_available",
        "value": False,
        "notes": "Disabled or file unavailable; flat annual demand will be used.",
    }])
    if not args.disable_hourly_operation_profile:
        hourly_profiles, hourly_profile_diag = load_hourly_operation_profiles(
            paths.get("hourly_operation_csv"),
            province_filter=set(annual["province_key"].dropna().astype(str).unique()),
            chunksize=args.hourly_profile_chunksize,
        )
    hourly_profile_diag.to_csv(out_dir / "diagnostics" / "hourly_operation_profile_diagnostics.csv", index=False, encoding="utf-8-sig")
    if hourly_profiles:
        print(f"[INFO] Loaded hourly operation profiles: {len(hourly_profiles):,} province-policy profiles")
    else:
        print("[INFO] Hourly operation profile not used; falling back to flat annual demand.")

    print("[EVAL] Energy supply scenarios")
    summary = evaluate_scenarios(annual, re_summary, re_profiles, hourly_profiles, args)

    print("[POST] Best variant tables")
    best_emission, best_grid = build_best_tables(summary)

    # Save input snapshots.
    annual.to_csv(out_dir / "inputs" / "energy_evaluator_input_annual_operation_with_resources.csv", index=False, encoding="utf-8-sig")
    re_summary.to_csv(out_dir / "inputs" / "renewable_supply_summary_used.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "inputs" / "grid_assumptions_used.csv", index=False, encoding="utf-8-sig")
    geo.to_csv(out_dir / "inputs" / "geothermal_eligibility_used.csv", index=False, encoding="utf-8-sig")

    # Save main outputs.
    summary_path = out_dir / "annual_results" / "energy_supply_summary_by_province_policy_scenario.csv"
    best_em_path = out_dir / "annual_results" / "energy_supply_best_by_min_energy_emissions.csv"
    best_grid_path = out_dir / "annual_results" / "energy_supply_best_by_min_grid_backup.csv"
    scenario_config_path = out_dir / "energy_scenario_config.csv"
    diag_path = out_dir / "diagnostics" / "energy_supply_diagnostics.csv"
    readme_path = out_dir / "README_03_ENERGY_SUPPLY_EVALUATOR.txt"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best_emission.to_csv(best_em_path, index=False, encoding="utf-8-sig")
    best_grid.to_csv(best_grid_path, index=False, encoding="utf-8-sig")

    # Scenario config.
    scenario_rows = []
    re_oversize_factors = parse_float_list(args.re_oversize_factors, [1.0, 1.5, 2.0])
    pv_shares = parse_float_list(args.pv_target_shares, [1.0, 0.75, 0.5, 0.25, 0.0])
    battery_durations = parse_float_list(args.battery_durations_h, [4.0])
    for s in SCENARIO_BASES:
        scenario_rows.append({
            "scenario_base": s,
            "heat_pump_COP": args.heat_pump_cop if "HP" in s else np.nan,
            "battery_roundtrip_efficiency": args.battery_roundtrip_efficiency if "battery" in s else np.nan,
            "RE_oversize_factors": ",".join(map(str, re_oversize_factors)) if "PVwind" in s else "",
            "PV_target_generation_shares": ",".join(map(str, pv_shares)) if "PVwind" in s else "",
            "max_wind_capacity_kW_per_1000kgads": args.max_wind_capacity_kw if "PVwind" in s else "",
            "battery_durations_h": ",".join(map(str, battery_durations)) if "battery" in s else "",
            "notes": "Generated by 03_energy_supply_evaluator.py",
        })
    pd.DataFrame(scenario_rows).to_csv(scenario_config_path, index=False, encoding="utf-8-sig")

    # Diagnostics.
    if not summary.empty:
        diag = summary.groupby(["energy_scenario_base", "scenario_valid_flag"], dropna=False).size().reset_index(name="n_rows")
        invalid = summary[summary["scenario_valid_flag"] == False].groupby(["energy_scenario_base", "invalid_reason"], dropna=False).size().reset_index(name="n_rows")
        # Append invalid breakdown below with compatible columns.
        invalid.insert(1, "scenario_valid_flag", False)
        invalid = invalid.rename(columns={"invalid_reason": "diagnostic_reason"})
        diag["diagnostic_reason"] = "validity_count"
        diag = pd.concat([diag, invalid[["energy_scenario_base", "scenario_valid_flag", "diagnostic_reason", "n_rows"]]], ignore_index=True)
    else:
        diag = pd.DataFrame()
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    # Wind-capacity feasibility diagnostics. Rows above the capacity limit remain
    # in the main CSV for transparency but are marked scenario_valid_flag=False.
    if not summary.empty and "wind_capacity_kW" in summary.columns:
        wind_diag_path = out_dir / "diagnostics" / "wind_capacity_constraint_diagnostics.csv"
        wind = summary.copy()
        wind["wind_capacity_kW"] = pd.to_numeric(wind["wind_capacity_kW"], errors="coerce")
        wind["wind_capacity_limit_kW_per_1000kgads"] = pd.to_numeric(wind.get("wind_capacity_limit_kW_per_1000kgads", np.nan), errors="coerce")
        wind["wind_capacity_constraint_exceeded_flag"] = wind.get("wind_capacity_constraint_exceeded_flag", False).astype(bool) if "wind_capacity_constraint_exceeded_flag" in wind.columns else False
        wind_stats = wind.groupby("energy_scenario_base", dropna=False).agg(
            n_rows=("energy_scenario_base", "size"),
            n_wind_capacity_above_limit=("wind_capacity_constraint_exceeded_flag", lambda x: int(pd.Series(x).astype(bool).sum())),
            median_wind_capacity_kW=("wind_capacity_kW", "median"),
            p95_wind_capacity_kW=("wind_capacity_kW", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.95))),
            p99_wind_capacity_kW=("wind_capacity_kW", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.99))),
            max_wind_capacity_kW=("wind_capacity_kW", "max"),
        ).reset_index()
        wind_stats.to_csv(wind_diag_path, index=False, encoding="utf-8-sig")

    print("[PLOT] Energy supply visualizations")
    make_energy_supply_visualizations(
        summary,
        best_emission,
        best_grid,
        out_dir,
        spatial_gpkg=None if args.disable_spatial_maps else paths.get("spatial_gpkg"),
        annual_operation=annual,
    )

    run_config = vars(args).copy()
    run_config.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **{k: str(v) for k, v in paths.items()},
        "n_annual_operation_rows": int(len(annual)),
        "n_energy_supply_rows": int(len(summary)),
        "demand_profile_method": "mixed_hourly_or_flat_from_module02",
        "hourly_operation_profile_used": bool(len(hourly_profiles) > 0),
    })
    (out_dir / "run_config_03_energy_supply.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    readme = f"""03_ENERGY_SUPPLY_EVALUATOR output

Main purpose:
This module evaluates electricity/heat supply scenarios for annual DAC demand from module 02.
It does not calculate final LCOD. It exports energy demand, supply shares, sizing variables,
operational energy emissions, and available energy-cost components for downstream LCOD and CCS modules.

Input annual operation summary:
{paths['annual_csv']}

Input RE hourly profile:
{paths['re_hourly_csv']}

Input RE summary:
{paths['re_summary_csv']}

Input grid assumptions:
{paths['grid_dir']}
{paths['grid_csv']}

Input geothermal resource table:
{paths['geo_csv']}

Input spatial map:
{paths.get('spatial_gpkg')}

Output folder:
{out_dir}

Demand profile method:
If available, the evaluator uses module-02 hourly_operation_profile_by_province_policy.csv for hourly DAC heat/electricity timing, scaled to the annual operation summary totals. If unavailable, it falls back to flat annual demand distributed over 8760 hours. Each output row is flagged with demand_profile_method and hourly_profile_used_flag.

Scenarios:
S0_grid_HP: 100% grid electricity + heat pump.
S1_grid_PVwind_HP: PV/wind used first, no battery, residual grid + heat pump.
S2_PVwind_battery_grid_HP: PV/wind + battery + residual grid + heat pump.
S3_grid_geothermalHeat: grid electricity + geothermal direct heat; evaluated only for geothermal-eligible provinces.
S4_PVwind_battery_grid_geothermalHeat: PV/wind + battery + residual grid + geothermal direct heat; evaluated only for geothermal-eligible provinces.

Main outputs:
{summary_path}
{best_em_path}
{best_grid_path}
{scenario_config_path}
{diag_path}

Important notes:
- PV/wind shares in the output are dispatch results, not assumed fixed shares.
- PV/wind capacities are sized using annual generation target sweep.
- Battery is represented with fixed-duration sizing and simple hourly dispatch.
- Geothermal heat is an eligibility/proxy layer from the geothermal resource table.
- Provinces without geothermal eligibility are omitted from S3/S4 rather than marked invalid, so comparison is made only where the scenario is physically available.
- Spatial figures use the province GPKG when available; otherwise point-map diagnostics are retained.
- PV/wind LCOE in this module defaults to zero to avoid double counting with PV/wind CAPEX in module 05.
- Missing grid price/emission data are retained as flags rather than filled with fabricated values.
- Low-carbon grid/PPA scenario is excluded from the main thesis workflow.
"""
    readme_path.write_text(readme, encoding="utf-8")

    print("=" * 100)
    print("03 ENERGY SUPPLY EVALUATOR COMPLETE")
    print("=" * 100)
    print(f"Main output : {summary_path}")
    print(f"Best by emissions: {best_em_path}")
    print(f"Best by grid backup: {best_grid_path}")
    print(f"Diagnostics : {diag_path}")
    print(f"Figures     : {out_dir / 'figures'}")
    if not diag.empty:
        print(diag.head(30).to_string(index=False))
    print("=" * 100)


def parse_args():
    p = argparse.ArgumentParser(description="Energy supply evaluator for ASEAN DAC annual operation results.")
    p.add_argument("--project-dir", default=r"D:/Ashka/5.DAC/06.PYTHON")
    p.add_argument("--tea-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--annual-operation-csv", default=None)
    p.add_argument("--hourly-operation-profile-csv", default=None, help="Optional module-02 hourly_operation_profile_by_province_policy.csv. If present, hourly demand is used for RE dispatch.")
    p.add_argument("--disable-hourly-operation-profile", action="store_true", help="Force fallback to flat annual demand even if hourly profile exists.")
    p.add_argument("--hourly-profile-chunksize", type=int, default=500000, help="Chunk size for reading module-02 hourly operation profile.")
    p.add_argument("--re-hourly-csv", default=None)
    p.add_argument("--re-summary-csv", default=None)
    p.add_argument("--grid-dir", default=None)
    p.add_argument("--grid-csv", default=None)
    p.add_argument("--geothermal-csv", default=None)
    p.add_argument("--spatial-dir", default=None)
    p.add_argument("--spatial-gpkg", default=None)
    p.add_argument("--disable-spatial-maps", action="store_true", help="Disable GPKG-based choropleth maps.")
    p.add_argument("--year", type=int, default=2025)

    p.add_argument("--scenarios", nargs="+", default=["all"], help="all or subset: S0 S1 S2 S3 S4")
    p.add_argument("--heat-pump-cop", type=float, default=2.5)
    p.add_argument("--battery-roundtrip-efficiency", type=float, default=0.85)
    p.add_argument("--battery-durations-h", default="4", help="Comma-separated battery durations, e.g. 4 or 4,8")
    p.add_argument("--initial-soc-frac", type=float, default=0.5)
    p.add_argument("--heat-pump-capacity-design-basis", choices=["average", "p95", "peak"], default="p95", help="Design basis for heat-pump capacity columns passed to module 05.")
    p.add_argument("--battery-power-basis", choices=["average", "p95", "peak"], default="average", help="Sizing basis for battery power when hourly DAC demand is available.")
    p.add_argument("--re-oversize-factors", default="1.0,1.5,2.0")
    p.add_argument("--pv-target-shares", default="1.0,0.75,0.5,0.25,0.0")
    p.add_argument("--max-wind-capacity-kw", type=float, default=1000.0, help="Maximum allowed installed wind capacity per 1000 kg adsorbent. Wind-heavy rows above this limit are kept but marked invalid for downstream LCOD/best-case selection. Use <=0 or NaN to disable.")

    # Optional energy-cost/emission defaults. NaN means leave missing and flag for LCOD module.
    p.add_argument("--pv-lcoe-usd-mwh", type=float, default=0.0)
    p.add_argument("--wind-lcoe-usd-mwh", type=float, default=0.0)
    p.add_argument("--geothermal-heat-price-usd-mwhth", type=float, default=54)
    p.add_argument("--geothermal-heat-ef-tco2-mwhth", type=float, default=0)

    # Debug/performance options.
    p.add_argument("--max-provinces", type=int, default=0, help="Debug only: limit number of RE provinces loaded.")
    p.add_argument("--max-rows", type=int, default=0, help="Debug only: limit number of annual operation rows evaluated.")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
