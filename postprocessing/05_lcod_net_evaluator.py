from __future__ import annotations

"""
05_lcod_net_evaluator.py

Final TEA / LCOD / net-removal evaluator for the ASEAN DACCS workflow.

Purpose
-------
This script takes the energy + CCS results from:

    5.DAC/02.TEA_LCOD/04_CCS_EVALUATOR/annual_results/
        energy_ccs_summary_by_province_policy_scenario.csv

and combines them with cost, WACC, CAPEX/OPEX, adsorbent replacement, and
Aspen Plus CO2-compression assumptions to calculate boundary-explicit cost and net-removal metrics:

    1. LCOD_gross_DAC_only_USD_tCO2
    2. LCOD_gross_DACCS_USD_tCO2cap
    3. LCOD_net_DACCS_GtG_USD_tCO2net
    4. LCOD_net_DACCS_CtG_energy_supply_USD_tCO2net
    5. LCOD_net_DACCS_CtG_full_system_USD_tCO2net

The emission accounting follows the captured-CO2-basis / boundary-separation
logic used by Yagihara et al. (2026): gross captured CO2, positive emissions,
net removed CO2, and LCOD denominators are kept explicit and not mixed across
Gate-to-Gate (GtG) and Cradle-to-Gate (CtG) boundaries.

The calculation basis is one DAC unit containing 1000 kg adsorbent, consistent
with the upstream annualized process/energy/CCS workflow.

Default project layout
----------------------
Root:
    D:/Ashka/5.DAC

Cost input:
    00.RESOURCES/01_COST/
        capex_opex_assumptions_master.csv
        recommended_tea_lcod_assumptions.csv
        wacc_country_technology.csv
        finance_cost_data_gap_confidence.csv
        wacc_capex_opex_source_registry.csv

Compression input:
    01.DAC_SYSTEM/04_ASPENPLUS_COMPRESSION/aspen_plus_compression*.csv

Upstream module input:
    02.TEA_LCOD/04_CCS_EVALUATOR/annual_results/
        energy_ccs_summary_by_province_policy_scenario.csv

Output:
    02.TEA_LCOD/05_LCOD_NET_EVALUATOR/
        annual_lcod_net_summary.csv
        cost_breakdown_by_province_policy_scenario.csv
        net_removal_breakdown.csv
        best_cases_by_lcod_net.csv
        best_cases_by_net_removal.csv
        best_cases_by_multiobjective_screening.csv
        diagnostics_lcod_net.csv
        inputs/*.csv
        config_05_LCOD_NET_EVALUATOR.json
        README_05_LCOD_NET_EVALUATOR.txt

Key modelling choices
---------------------
- Scale basis: 1000 kg adsorbent.
- DAC CAPEX default mode: actual annual CO2 capture basis,
  CAPEX_DAC = USD_per_tCO2yr * annual_CO2_captured_t.
- Optional fixed DAC nameplate mode is available using --dac-capacity-mode fixed.
- WACC: country-level central by default from WACC table.
- Adsorbent replacement: physical basis, default 1000 kg adsorbent replaced every
  2 years.
- Energy cost and energy emissions are taken from module 03/04.
- Transport and storage cost/emissions are taken from module 04.
- Compression electricity is taken from Aspen Plus compression CSV if available;
  otherwise the script falls back to the NETL compression value in the CCS cost file
  when available.
- Heat pump, PV, wind, and battery CAPEX are annualized here using capacities from
  module 03/04 outputs. If module 03 provides hourly-demand design capacity,
  heat-pump CAPEX uses that value instead of annual-average heat/8760.
"""

from pathlib import Path
import argparse
import json
import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    gpd = None
    _GPD_IMPORT_ERROR = exc
else:
    _GPD_IMPORT_ERROR = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_IMPORT_ERROR = exc
else:
    _MPL_IMPORT_ERROR = None


# =============================================================================
# Utility helpers
# =============================================================================

def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        val = float(x)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_optional(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_latest_csv(folder: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(folder.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def normalize_name(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def normalize_country_code(s: Any) -> str:
    return normalize_name(s).upper()


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def numeric_series(df: pd.DataFrame, col: str | None, default=np.nan) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def crf(wacc_percent: float, lifetime_years: float) -> float:
    r = safe_float(wacc_percent) / 100.0
    n = safe_float(lifetime_years)
    if not np.isfinite(r) or not np.isfinite(n) or n <= 0:
        return np.nan
    if abs(r) < 1e-12:
        return 1.0 / n
    return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)


def finite_or_zero(x: pd.Series | float) -> pd.Series | float:
    if isinstance(x, pd.Series):
        return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if x is None or not np.isfinite(x):
        return 0.0
    return x


def boolean_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """Return a robust boolean series for mixed True/False, 1/0, yes/no columns."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    txt = s.fillna(default).astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f", "nan", "none", ""}
    out = txt.map(lambda x: True if x in true_vals else (False if x in false_vals else default))
    return out.astype(bool)


# =============================================================================
# Path handling
# =============================================================================

def resolve_paths(args) -> dict[str, Path]:
    root = Path(args.root_dir)
    tea_dir = Path(args.tea_dir) if args.tea_dir else root / "02.TEA_LCOD"
    cost_dir = Path(args.cost_dir) if args.cost_dir else root / "00.RESOURCES" / "01_COST"
    compression_dir = Path(args.compression_dir) if args.compression_dir else root / "01.DAC_SYSTEM" / "04_ASPENPLUS_COMPRESSION"
    out_dir = Path(args.out_dir) if args.out_dir else tea_dir / "05_LCOD_NET_EVALUATOR"

    input_energy_ccs = Path(args.energy_ccs_csv) if args.energy_ccs_csv else tea_dir / "04_CCS_EVALUATOR" / "annual_results" / "energy_ccs_summary_by_province_policy_scenario.csv"

    spatial_dir = Path(args.spatial_dir) if args.spatial_dir else root / "00.SPATIAL_MAP"
    if args.spatial_gpkg:
        spatial_gpkg = Path(args.spatial_gpkg)
    else:
        spatial_gpkg = spatial_dir / "ASEAN" / "ASEAN_PROVINCES_LEVEL1.gpkg"

    paths = {
        "root": root,
        "tea_dir": tea_dir,
        "cost_dir": cost_dir,
        "compression_dir": compression_dir,
        "out_dir": out_dir,
        "energy_ccs_csv": input_energy_ccs,
        "capex_opex": cost_dir / "capex_opex_assumptions_master.csv",
        "recommended": cost_dir / "recommended_tea_lcod_assumptions.csv",
        "wacc": cost_dir / "wacc_country_technology.csv",
        "gap_confidence": cost_dir / "finance_cost_data_gap_confidence.csv",
        "source_registry": cost_dir / "wacc_capex_opex_source_registry.csv",
        "ccs_cost": root / "00.RESOURCES" / "03_CCS" / "ccs_cost_emission_assumptions.csv",
        "spatial_dir": spatial_dir,
        "spatial_gpkg": spatial_gpkg,
    }

    return paths


# =============================================================================
# Cost and finance assumption loading
# =============================================================================

def load_capex_master(path: Path) -> pd.DataFrame:
    df = read_csv_optional(path)
    if df.empty:
        warnings.warn(f"CAPEX/OPEX master not found or empty: {path}")
        return df
    for c in ["component", "technology", "subtechnology", "scenario", "currency", "CAPEX_unit", "fixed_OPEX_unit", "replacement_cost_unit"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def select_master_row(master: pd.DataFrame, *, component_contains: str | None = None, technology_contains: str | None = None, scenario: str = "central") -> pd.Series | None:
    if master.empty:
        return None
    m = pd.Series(True, index=master.index)
    if component_contains:
        m &= master.get("component", "").astype(str).str.lower().str.contains(component_contains.lower(), na=False)
    if technology_contains:
        m &= master.get("technology", "").astype(str).str.lower().str.contains(technology_contains.lower(), na=False)
    sub = master[m].copy()
    if sub.empty:
        return None
    if "scenario" in sub.columns:
        exact = sub[sub["scenario"].astype(str).str.lower() == scenario.lower()]
        if not exact.empty:
            sub = exact
        else:
            central = sub[sub["scenario"].astype(str).str.lower() == "central"]
            if not central.empty:
                sub = central
    return sub.iloc[0]


def get_master_value(master: pd.DataFrame, component: str, technology: str | None, value_col: str, scenario: str, default=np.nan) -> float:
    row = select_master_row(master, component_contains=component, technology_contains=technology, scenario=scenario)
    if row is None or value_col not in row.index:
        return default
    return safe_float(row[value_col], default=default)


def get_master_lifetime(master: pd.DataFrame, component: str, technology: str | None, scenario: str, default: float) -> float:
    val = get_master_value(master, component, technology, "lifetime_years", scenario, np.nan)
    return val if np.isfinite(val) and val > 0 else default


def load_wacc_table(path: Path, scenario: str) -> pd.DataFrame:
    df = read_csv_optional(path)
    if df.empty:
        warnings.warn(f"WACC table not found or empty: {path}")
        return pd.DataFrame(columns=["country_code", "WACC_percent"])
    df = df.copy()
    df["country_code"] = df["country_code"].map(normalize_country_code)
    if "scenario" in df.columns:
        preferred = df[df["scenario"].astype(str).str.lower() == scenario.lower()].copy()
        if preferred.empty and scenario.lower() != "central":
            preferred = df[df["scenario"].astype(str).str.lower() == "central"].copy()
        if preferred.empty:
            preferred = df.copy()
    else:
        preferred = df.copy()

    # Prefer clean_energy_proxy, then solar_pv, then any country value.
    tech_pref = {"clean_energy_proxy": 0, "solar_pv": 1, "renewable_power": 2, "battery_storage": 3, "hydropower": 4, "offshore_wind": 5}
    if "technology" in preferred.columns:
        preferred["_tech_rank"] = preferred["technology"].astype(str).str.lower().map(lambda x: tech_pref.get(x, 99))
    else:
        preferred["_tech_rank"] = 99

    # Use post-tax WACC first, else nominal.
    if "post_tax_WACC_percent" in preferred.columns:
        preferred["_wacc"] = pd.to_numeric(preferred["post_tax_WACC_percent"], errors="coerce")
    else:
        preferred["_wacc"] = np.nan
    if "nominal_WACC_percent" in preferred.columns:
        preferred["_wacc"] = preferred["_wacc"].fillna(pd.to_numeric(preferred["nominal_WACC_percent"], errors="coerce"))

    preferred = preferred.dropna(subset=["country_code", "_wacc"])
    preferred = preferred.sort_values(["country_code", "_tech_rank"])
    out = preferred.drop_duplicates("country_code", keep="first")[["country_code", "_wacc"]].rename(columns={"_wacc": "WACC_percent"})
    return out


def build_cost_parameters(master: pd.DataFrame, args) -> dict[str, float]:
    scenario = args.cost_scenario
    p: dict[str, float] = {}

    # DAC CAPEX/OPEX.
    p["dac_capex_usd_per_tco2yr"] = get_master_value(master, "full_DAC_plant", "solid_sorbent", "CAPEX_value", scenario, args.fallback_dac_capex_usd_per_tco2yr)
    p["dac_lifetime_years"] = get_master_lifetime(master, "full_DAC_plant", "solid_sorbent", scenario, args.project_lifetime_years)
    p["dac_fixed_opex_usd_per_tco2"] = get_master_value(master, "full_DAC_plant", "solid_sorbent", "fixed_OPEX_value", scenario, args.fallback_dac_fixed_opex_usd_per_tco2)

    # Sorbent cost is in USD/lb_sorbent in current master.
    p["sorbent_replacement_cost_usd_per_lb"] = get_master_value(master, "sorbent", "solid_sorbent", "replacement_cost_value", scenario, args.fallback_sorbent_cost_usd_per_lb)
    p["adsorbent_lifetime_years"] = args.adsorbent_lifetime_years
    p["adsorbent_mass_kg"] = args.adsorbent_mass_kg

    # Energy system CAPEX.
    p["pv_capex_usd_per_kw"] = get_master_value(master, "solar", "solar", "CAPEX_value", scenario, args.fallback_pv_capex_usd_per_kw)
    p["pv_lifetime_years"] = get_master_lifetime(master, "solar", "solar", scenario, args.pv_lifetime_years)

    p["wind_capex_usd_per_kw"] = get_master_value(master, "wind", "onshore", "CAPEX_value", scenario, args.fallback_wind_capex_usd_per_kw)
    p["wind_lifetime_years"] = get_master_lifetime(master, "wind", "onshore", scenario, args.wind_lifetime_years)

    p["battery_capex_usd_per_kwh"] = get_master_value(master, "battery", "battery", "CAPEX_value", scenario, args.fallback_battery_capex_usd_per_kwh)
    p["battery_lifetime_years"] = get_master_lifetime(master, "battery", "battery", scenario, args.battery_lifetime_years)

    p["heat_pump_capex_usd_per_kwth"] = args.heat_pump_capex_usd_per_kwth
    p["heat_pump_lifetime_years"] = args.heat_pump_lifetime_years
    # Purchased geothermal heat/LCOH is costed upstream in module 03. Keep geothermal heat system CAPEX
    # at zero unless the user explicitly wants to model owned geothermal heat infrastructure.

    p["geothermal_heat_capex_usd_per_kwth"] = args.geothermal_heat_capex_usd_per_kwth
    p["geothermal_heat_lifetime_years"] = args.geothermal_heat_lifetime_years

    return p


# =============================================================================
# Compression assumptions
# =============================================================================

def detect_compression_specific_energy(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    # Look for columns with kWh/tCO2 semantics.
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if ("kwh" in cl or "kwhe" in cl) and ("tco2" in cl or "ton" in cl or "tonne" in cl):
            candidates.append(c)
        elif "specific" in cl and "compression" in cl and ("electric" in cl or "energy" in cl):
            candidates.append(c)
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            # If unit looks like MWh/t, convert to kWh/t.
            val = float(vals.median())
            if "mwh" in c.lower() and "kwh" not in c.lower():
                val *= 1000.0
            return val
    # Look for a row-based parameter file.
    if {"parameter", "value"}.issubset(set(df.columns)):
        for _, row in df.iterrows():
            par = str(row.get("parameter", "")).lower()
            unit = str(row.get("unit", "")).lower()
            if "compression" in par and ("kwh" in unit or "mwh" in unit) and ("tco2" in unit or "ton" in unit):
                val = safe_float(row.get("value"))
                if np.isfinite(val):
                    if "mwh" in unit and "kwh" not in unit:
                        val *= 1000.0
                    return val
    return np.nan


def detect_compression_cost_per_t(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    for c in df.columns:
        cl = c.lower()
        if "cost" in cl and "usd" in cl and ("tco2" in cl or "ton" in cl):
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                return float(vals.median())
    if {"parameter", "value"}.issubset(set(df.columns)):
        for _, row in df.iterrows():
            par = str(row.get("parameter", "")).lower()
            unit = str(row.get("unit", "")).lower()
            if "compression" in par and "cost" in par and "usd" in unit and ("tco2" in unit or "ton" in unit):
                val = safe_float(row.get("value"))
                if np.isfinite(val):
                    return val
    return np.nan


def detect_compression_emission_per_t(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    for c in df.columns:
        cl = c.lower()
        if ("emission" in cl or "co2e" in cl) and ("tco2" in cl or "ton" in cl):
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                return float(vals.median())
    if {"parameter", "value"}.issubset(set(df.columns)):
        for _, row in df.iterrows():
            par = str(row.get("parameter", "")).lower()
            unit = str(row.get("unit", "")).lower()
            if "compression" in par and ("emission" in par or "co2e" in par) and ("tco2" in unit or "ton" in unit):
                val = safe_float(row.get("value"))
                if np.isfinite(val):
                    return val
    return np.nan


def fallback_compression_from_ccs_cost(ccs_cost_path: Path) -> float:
    df = read_csv_optional(ccs_cost_path)
    if df.empty:
        return np.nan
    m = df.get("component", pd.Series("", index=df.index)).astype(str).str.lower().str.contains("compression", na=False)
    sub = df[m].copy()
    if sub.empty:
        return np.nan
    for col in ["central_value", "energy_consumption_value", "cost_value"]:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.iloc[0])
    return np.nan


def load_compression_assumptions(compression_dir: Path, ccs_cost_path: Path, args) -> tuple[dict[str, Any], pd.DataFrame]:
    comp_file = Path(args.compression_csv) if args.compression_csv else find_latest_csv(compression_dir, ["aspen_plus_compression*.csv", "*compression*.csv", "*.csv"])
    comp_df = read_csv_optional(comp_file) if comp_file else pd.DataFrame()

    spec_kwh_t = detect_compression_specific_energy(comp_df)
    cost_usd_t = detect_compression_cost_per_t(comp_df)
    emis_t_t = detect_compression_emission_per_t(comp_df)

    source = str(comp_file) if comp_file else "not_found"
    if not np.isfinite(spec_kwh_t):
        spec_kwh_t = args.compression_kwh_per_tco2
        source = f"manual_arg_or_default:{source}"
    if not np.isfinite(spec_kwh_t):
        spec_kwh_t = fallback_compression_from_ccs_cost(ccs_cost_path)
        source = f"fallback_ccs_cost_file:{source}"

    assumptions = {
        "compression_source_file": source,
        "compression_kWh_per_tCO2": spec_kwh_t,
        "compression_cost_USD_per_tCO2_direct": cost_usd_t,
        "compression_emission_tCO2e_per_tCO2_direct": emis_t_t,
    }
    return assumptions, comp_df


# =============================================================================
# Input canonicalization
# =============================================================================

def canonicalize_energy_ccs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["country_code", "country_name", "province_id", "province_name", "operation_policy", "energy_scenario", "energy_scenario_base", "selected_transport_mode", "missing_energy_cost_items"]:
        if col not in out.columns:
            out[col] = ""
    out["country_code"] = out["country_code"].map(normalize_country_code)
    out["province_id"] = out["province_id"].astype(str).map(normalize_name)

    co2_col = first_existing_col(out, ["annual_CO2_for_CCS_t", "annual_CO2_t_per_1000kgads", "annual_CO2_t", "annual_CO2_captured_t", "annual_CO2_t_per_bed"])
    out["annual_CO2_captured_t"] = numeric_series(out, co2_col)

    # Energy module outputs.
    for target, candidates in {
        "annual_process_electricity_MWhe": ["annual_process_electricity_MWhe", "annual_E_total_el_MWhe_per_1000kgads", "annual_electricity_MWhe"],
        "annual_heat_demand_MWhth": ["annual_heat_demand_MWhth", "annual_Q_heat_MWhth_per_1000kgads", "annual_heat_MWhth"],
        "annual_heat_pump_electricity_MWhe": ["annual_heat_pump_electricity_MWhe"],
        "annual_total_electricity_demand_MWhe": ["annual_total_electricity_demand_MWhe", "annual_total_electricity_MWhe"],
        "energy_cost_USD": ["energy_cost_USD", "annual_energy_cost_USD"],
        "energy_emissions_tCO2e": ["energy_emissions_tCO2", "energy_emissions_tCO2e", "annual_energy_emissions_tCO2e"],
        "pv_capacity_kWp": ["pv_capacity_kWp", "PV_capacity_kWp"],
        "wind_capacity_kW": ["wind_capacity_kW", "wind_capacity_kw"],
        "battery_capacity_kWh": ["battery_capacity_kWh"],
        "battery_power_kW": ["battery_power_kW"],
        "geothermal_heat_used_MWhth": ["geothermal_heat_used_MWhth"],
        "heat_from_heatpump_MWhth": ["heat_from_heatpump_MWhth"],
        # Optional activity columns for boundary-explicit lifecycle/sensitivity emissions.
        "pv_generation_used_MWhe": [
            "pv_generation_used_MWhe", "annual_pv_generation_used_MWhe",
            "pv_used_MWhe", "annual_pv_used_MWhe", "annual_pv_generation_MWhe",
            "pv_generation_MWhe", "PV_generation_used_MWhe",
        ],
        "wind_generation_used_MWhe": [
            "wind_generation_used_MWhe", "annual_wind_generation_used_MWhe",
            "wind_used_MWhe", "annual_wind_used_MWhe", "annual_wind_generation_MWhe",
            "wind_generation_MWhe",
        ],
        "battery_throughput_MWhe": [
            "battery_throughput_MWhe", "annual_battery_throughput_MWhe",
            "battery_discharge_MWhe", "annual_battery_discharge_MWhe",
        ],
        "grid_electricity_MWhe": ["grid_electricity_MWhe"],
        "grid_emission_factor_tCO2_MWh": ["grid_emission_factor_tCO2_MWh"],
        "grid_electricity_price_USD_MWh": ["grid_electricity_price_USD_MWh", "industrial_tariff_USD_MWh"],
        # Hourly-demand / design-capacity columns propagated from module 03 via module 04.
        "hourly_profile_n_hours": ["hourly_profile_n_hours"],
        "peak_process_electricity_kW": ["peak_process_electricity_kW"],
        "p95_process_electricity_kW": ["p95_process_electricity_kW"],
        "peak_heat_demand_kWth": ["peak_heat_demand_kWth"],
        "p95_heat_demand_kWth": ["p95_heat_demand_kWth"],
        "heat_pump_capacity_kWth_design": ["heat_pump_capacity_kWth_design"],
        "peak_total_electricity_demand_kW": ["peak_total_electricity_demand_kW"],
        "p95_total_electricity_demand_kW": ["p95_total_electricity_demand_kW"],
        "battery_power_kW_design": ["battery_power_kW_design"],
    }.items():
        out[target] = numeric_series(out, first_existing_col(out, candidates), 0.0)

    # Non-numeric profile-method columns should be preserved for diagnostics and interpretation.
    for text_col in ["demand_profile_method", "heat_pump_capacity_design_basis", "battery_power_basis"]:
        if text_col not in out.columns:
            out[text_col] = ""
        out[text_col] = out[text_col].fillna("").astype(str)
    out["hourly_profile_used_flag"] = boolean_series(out, "hourly_profile_used_flag", default=False)
    out["module04_preserved_hourly_profile_columns_flag"] = boolean_series(
        out, "module04_preserved_hourly_profile_columns_flag", default=False
    )
    if "energy_ccs_valid_flag" not in out.columns:
        out["energy_ccs_valid_flag"] = True

    # CCS outputs.
    for target, candidates in {
        "annual_TandS_cost_USD": ["annual_TandS_cost_USD", "annual_total_CCS_cost_USD"],
        "annual_CCS_emissions_tCO2e": ["annual_total_CCS_emission_tCO2e", "annual_transport_emission_tCO2e"],
        "total_TandS_cost_USD_tCO2": ["total_TandS_cost_USD_tCO2", "selected_total_TandS_cost_USD_tCO2"],
        "selected_transport_cost_USD_tCO2": ["selected_transport_cost_USD_tCO2"],
        "storage_cost_USD_tCO2": ["storage_cost_USD_tCO2"],
        "mrv_cost_USD_tCO2": ["mrv_cost_USD_tCO2"],
        "distance_km": ["straight_distance_km", "distance_km", "pipeline_effective_distance_km"],
    }.items():
        out[target] = numeric_series(out, first_existing_col(out, candidates), 0.0)

    # If annual T&S is zero but unit cost is present, calculate.
    mask_ts_missing = (out["annual_TandS_cost_USD"].isna()) | (out["annual_TandS_cost_USD"] == 0)
    if "total_TandS_cost_USD_tCO2" in out.columns:
        out.loc[mask_ts_missing, "annual_TandS_cost_USD"] = out.loc[mask_ts_missing, "annual_CO2_captured_t"] * out.loc[mask_ts_missing, "total_TandS_cost_USD_tCO2"]

    # Scenario validity flags propagated if present.
    if "scenario_valid_flag" not in out.columns:
        out["scenario_valid_flag"] = True
    if "ccs_valid_flag" not in out.columns:
        out["ccs_valid_flag"] = True
    if "invalid_reason" not in out.columns:
        out["invalid_reason"] = ""

    return out



# =============================================================================
# Boundary-explicit emission accounting helpers
# =============================================================================

BOUNDARY_NAMES = ["GtG_operational", "CtG_energy_supply", "CtG_full_system"]


def _arg_float(args, name: str, default=np.nan) -> float:
    return safe_float(getattr(args, name, default), default)


def optional_emission_component(
    out: pd.DataFrame,
    *,
    activity_col: str,
    activity_fallback_col: str | None,
    ef_arg_name: str,
    output_col: str,
    missing_flag_col: str,
    status_col: str,
    active_threshold: float = 1e-12,
    scale: float = 1.0,
    args=None,
) -> pd.DataFrame:
    """Calculate optional lifecycle/sensitivity emissions without inventing missing factors.

    If a component is inactive, emissions are zero and status is not_applicable.
    If the component is active but the emission factor is missing, emissions are set to zero
    only as an EXCLUDED known subtotal, and a missing-source flag is raised. The boundary
    completeness flags then show that the corresponding CtG result is incomplete.
    """
    activity = pd.to_numeric(out.get(activity_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    if activity_fallback_col is not None:
        fallback = pd.to_numeric(out.get(activity_fallback_col, pd.Series(np.nan, index=out.index)), errors="coerce")
        activity = activity.where(np.isfinite(activity) & (activity > 0), fallback)
    activity = activity.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    active = activity > active_threshold
    ef = _arg_float(args, ef_arg_name, np.nan)

    if np.isfinite(ef):
        out[output_col] = activity * ef * scale
        out[missing_flag_col] = False
        out[status_col] = np.where(active, "included_from_user_supplied_or_sourced_EF", "not_applicable_no_activity")
    else:
        out[output_col] = 0.0
        out[missing_flag_col] = active
        out[status_col] = np.where(active, "excluded_missing_EF_needs_source", "not_applicable_no_activity")
    return out


def add_boundary_explicit_emissions(out: pd.DataFrame, annual_co2: pd.Series, args) -> pd.DataFrame:
    """Add GtG/CtG emission, net-removal, and LCOD denominator columns.

    This follows the boundary/functional-unit discipline used by Yagihara et al.:
    captured-CO2 basis is kept explicit; gross captured CO2 and net removed CO2 are
    reported separately; GtG operational and CtG/system-style emissions are not mixed.
    """
    out = out.copy()
    out["gross_CO2_captured_t"] = annual_co2
    out["CO2_storage_loss_fraction_assumed"] = _arg_float(args, "co2_storage_loss_fraction", 0.0)
    out["gross_CO2_stored_t"] = annual_co2 * (1.0 - out["CO2_storage_loss_fraction_assumed"])
    out["CO2_loss_assumption"] = np.where(
        out["CO2_storage_loss_fraction_assumed"].abs() < 1e-15,
        "no_storage_loss_assumed",
        "storage_loss_fraction_user_supplied",
    )

    # Component naming: keep inherited module-03/04 emissions as their own audited component.
    out["process_energy_supply_emissions_tCO2e"] = pd.to_numeric(out["energy_emissions_tCO2e"], errors="coerce")
    out["compression_electricity_emissions_tCO2e"] = pd.to_numeric(out["compression_emissions_tCO2e"], errors="coerce")
    out["transport_storage_operational_emissions_tCO2e"] = pd.to_numeric(out["annual_CCS_emissions_tCO2e"], errors="coerce")

    # GtG operational: known annual operating emissions inherited/calculated in modules 03-05.
    out["emissions_GtG_operational_tCO2e"] = (
        out["process_energy_supply_emissions_tCO2e"]
        + out["compression_electricity_emissions_tCO2e"]
        + out["transport_storage_operational_emissions_tCO2e"]
    )

    # Optional CtG energy-supply additions. Defaults do not invent values; missing active
    # factors are excluded from the known subtotal and flagged.
    out = optional_emission_component(
        out,
        activity_col="pv_generation_used_MWhe",
        activity_fallback_col=None,
        ef_arg_name="pv_lifecycle_ef_tco2e_mwh",
        output_col="pv_lifecycle_emissions_tCO2e",
        missing_flag_col="pv_lifecycle_EF_missing_flag",
        status_col="pv_lifecycle_emission_status",
        args=args,
    )
    out = optional_emission_component(
        out,
        activity_col="wind_generation_used_MWhe",
        activity_fallback_col=None,
        ef_arg_name="wind_lifecycle_ef_tco2e_mwh",
        output_col="wind_lifecycle_emissions_tCO2e",
        missing_flag_col="wind_lifecycle_EF_missing_flag",
        status_col="wind_lifecycle_emission_status",
        args=args,
    )
    out = optional_emission_component(
        out,
        activity_col="geothermal_heat_used_MWhth",
        activity_fallback_col=None,
        ef_arg_name="geothermal_heat_lifecycle_ef_tco2e_mwhth",
        output_col="geothermal_heat_lifecycle_emissions_tCO2e",
        missing_flag_col="geothermal_heat_lifecycle_EF_missing_flag",
        status_col="geothermal_heat_lifecycle_emission_status",
        args=args,
    )
    out = optional_emission_component(
        out,
        activity_col="battery_capacity_kWh",
        activity_fallback_col=None,
        ef_arg_name="battery_embodied_ef_tco2e_kwh",
        output_col="battery_lifecycle_emissions_tCO2e",
        missing_flag_col="battery_embodied_EF_missing_flag",
        status_col="battery_lifecycle_emission_status",
        scale=1.0 / max(_arg_float(args, "battery_lifetime_years", 15.0), 1e-12),
        args=args,
    )

    ctg_energy_missing_flags = [
        "pv_lifecycle_EF_missing_flag",
        "wind_lifecycle_EF_missing_flag",
        "geothermal_heat_lifecycle_EF_missing_flag",
        "battery_embodied_EF_missing_flag",
    ]
    out["CtG_energy_supply_missing_lifecycle_EF_flag"] = False
    for c in ctg_energy_missing_flags:
        if c in out.columns:
            out["CtG_energy_supply_missing_lifecycle_EF_flag"] |= out[c].astype(bool)

    out["emissions_CtG_energy_supply_known_tCO2e"] = (
        out["emissions_GtG_operational_tCO2e"]
        + out["pv_lifecycle_emissions_tCO2e"]
        + out["wind_lifecycle_emissions_tCO2e"]
        + out["geothermal_heat_lifecycle_emissions_tCO2e"]
        + out["battery_lifecycle_emissions_tCO2e"]
    )
    out["emissions_CtG_energy_supply_tCO2e"] = out["emissions_CtG_energy_supply_known_tCO2e"]
    out["CtG_energy_supply_complete_flag"] = ~out["CtG_energy_supply_missing_lifecycle_EF_flag"]
    out["CtG_energy_supply_scope_note"] = np.where(
        out["CtG_energy_supply_complete_flag"],
        "known operational plus supplied lifecycle energy-system factors",
        "known subtotal only; at least one active lifecycle energy-system EF is missing/needs source",
    )

    # Optional full-system sensitivity additions. These are not baseline unless factors are supplied.
    sorbent_ef = _arg_float(args, "sorbent_embodied_ef_tco2e_kg", np.nan)
    if np.isfinite(sorbent_ef):
        out["sorbent_replacement_emissions_tCO2e"] = (
            _arg_float(args, "adsorbent_mass_kg", 1000.0) * sorbent_ef / max(_arg_float(args, "adsorbent_lifetime_years", 2.0), 1e-12)
        )
        out["sorbent_embodied_EF_missing_flag"] = False
        out["sorbent_emission_status"] = "included_from_user_supplied_or_sourced_EF"
    else:
        out["sorbent_replacement_emissions_tCO2e"] = 0.0
        out["sorbent_embodied_EF_missing_flag"] = bool(getattr(args, "include_full_system_sensitivity", False))
        out["sorbent_emission_status"] = np.where(
            out["sorbent_embodied_EF_missing_flag"],
            "excluded_missing_EF_needs_source",
            "excluded_not_requested_for_baseline",
        )

    dac_infra_ef = _arg_float(args, "dac_infrastructure_ef_tco2e_tco2yr", np.nan)
    if np.isfinite(dac_infra_ef):
        out["DAC_infrastructure_emissions_tCO2e"] = (
            pd.to_numeric(out.get("DAC_capacity_basis_tCO2yr", annual_co2), errors="coerce").fillna(0.0)
            * dac_infra_ef
            / max(_arg_float(args, "project_lifetime_years", 30.0), 1e-12)
        )
        out["DAC_infrastructure_EF_missing_flag"] = False
        out["DAC_infrastructure_emission_status"] = "included_from_user_supplied_or_sourced_EF"
    else:
        out["DAC_infrastructure_emissions_tCO2e"] = 0.0
        out["DAC_infrastructure_EF_missing_flag"] = bool(getattr(args, "include_full_system_sensitivity", False))
        out["DAC_infrastructure_emission_status"] = np.where(
            out["DAC_infrastructure_EF_missing_flag"],
            "excluded_missing_EF_needs_source",
            "excluded_not_requested_for_baseline",
        )

    comp_infra_ef = _arg_float(args, "compression_infrastructure_ef_tco2e_tco2", np.nan)
    if np.isfinite(comp_infra_ef):
        out["compression_infrastructure_emissions_tCO2e"] = annual_co2 * comp_infra_ef
        out["compression_infrastructure_EF_missing_flag"] = False
        out["compression_infrastructure_emission_status"] = "included_from_user_supplied_or_sourced_EF"
    else:
        out["compression_infrastructure_emissions_tCO2e"] = 0.0
        out["compression_infrastructure_EF_missing_flag"] = False
        out["compression_infrastructure_emission_status"] = "excluded_no_EF_supplied"

    ccs_infra_ef = _arg_float(args, "ccs_infrastructure_ef_tco2e_tco2", np.nan)
    if np.isfinite(ccs_infra_ef):
        out["CCS_infrastructure_emissions_tCO2e"] = annual_co2 * ccs_infra_ef
        out["CCS_infrastructure_EF_missing_flag"] = False
        out["CCS_infrastructure_emission_status"] = "included_from_user_supplied_or_sourced_EF"
    else:
        out["CCS_infrastructure_emissions_tCO2e"] = 0.0
        out["CCS_infrastructure_EF_missing_flag"] = False
        out["CCS_infrastructure_emission_status"] = "excluded_no_EF_supplied"

    out["CtG_full_system_missing_EF_flag"] = (
        out["CtG_energy_supply_missing_lifecycle_EF_flag"].astype(bool)
        | out["sorbent_embodied_EF_missing_flag"].astype(bool)
        | out["DAC_infrastructure_EF_missing_flag"].astype(bool)
        | out["compression_infrastructure_EF_missing_flag"].astype(bool)
        | out["CCS_infrastructure_EF_missing_flag"].astype(bool)
    )
    out["emissions_CtG_full_system_known_tCO2e"] = (
        out["emissions_CtG_energy_supply_known_tCO2e"]
        + out["sorbent_replacement_emissions_tCO2e"]
        + out["DAC_infrastructure_emissions_tCO2e"]
        + out["compression_infrastructure_emissions_tCO2e"]
        + out["CCS_infrastructure_emissions_tCO2e"]
    )
    out["emissions_CtG_full_system_tCO2e"] = out["emissions_CtG_full_system_known_tCO2e"]
    out["CtG_full_system_complete_flag"] = ~out["CtG_full_system_missing_EF_flag"]
    out["CtG_full_system_scope_note"] = np.where(
        out["CtG_full_system_complete_flag"],
        "known operational, supplied energy-system lifecycle, and supplied infrastructure/sorbent factors",
        "known subtotal only; at least one active/requested full-system EF is missing/needs source",
    )

    for boundary, e_col in {
        "GtG": "emissions_GtG_operational_tCO2e",
        "CtG_energy_supply": "emissions_CtG_energy_supply_tCO2e",
        "CtG_full_system": "emissions_CtG_full_system_tCO2e",
    }.items():
        net_col = f"net_CO2_removed_{boundary}_t"
        eff_col = f"removal_efficiency_{boundary}"
        int_col = f"emission_intensity_{boundary}_tCO2e_tCO2cap"
        status_col = f"net_removal_status_{boundary}"
        out[net_col] = out["gross_CO2_stored_t"] - out[e_col]
        out[int_col] = out[e_col] / annual_co2.replace(0, np.nan)
        out[eff_col] = out[net_col] / annual_co2.replace(0, np.nan)
        out[status_col] = np.where(out[net_col] > 0, "net_removal_positive", "no_net_removal")

    return out

# =============================================================================
# LCOD calculation
# =============================================================================

def calculate_lcod(df: pd.DataFrame, wacc: pd.DataFrame, params: dict[str, float], comp: dict[str, Any], args) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(wacc, on="country_code", how="left")
    out["WACC_percent"] = pd.to_numeric(out["WACC_percent"], errors="coerce").fillna(args.fallback_wacc_percent)

    # CRF values.
    out["CRF_DAC"] = out["WACC_percent"].apply(lambda w: crf(w, params["dac_lifetime_years"]))
    out["CRF_PV"] = out["WACC_percent"].apply(lambda w: crf(w, params["pv_lifetime_years"]))
    out["CRF_wind"] = out["WACC_percent"].apply(lambda w: crf(w, params["wind_lifetime_years"]))
    out["CRF_battery"] = out["WACC_percent"].apply(lambda w: crf(w, params["battery_lifetime_years"]))
    out["CRF_heat_pump"] = out["WACC_percent"].apply(lambda w: crf(w, params["heat_pump_lifetime_years"]))
    out["CRF_geothermal_heat"] = out["WACC_percent"].apply(lambda w: crf(w, params["geothermal_heat_lifetime_years"]))

    annual_co2 = pd.to_numeric(out["annual_CO2_captured_t"], errors="coerce")

    # DAC nameplate / capacity basis.
    if args.dac_capacity_mode == "fixed":
        dac_capacity = pd.Series(args.dac_nameplate_tco2yr, index=out.index, dtype="float64")
    else:
        dac_capacity = annual_co2.copy()
    out["DAC_capacity_basis_tCO2yr"] = dac_capacity

    out["DAC_CAPEX_USD"] = params["dac_capex_usd_per_tco2yr"] * dac_capacity
    out["annualized_DAC_CAPEX_USD"] = out["DAC_CAPEX_USD"] * out["CRF_DAC"]

    # Fixed OPEX follows the agreed simple basis: USD/tCO2 applied to annual captured output.
    out["fixed_OPEX_DAC_USD"] = params["dac_fixed_opex_usd_per_tco2"] * annual_co2

    # Adsorbent replacement: physical basis, default 1000 kg adsorbent, replacement every 2 years.
    kg_to_lb = 2.20462262185
    out["adsorbent_mass_kg_basis"] = params["adsorbent_mass_kg"]
    out["adsorbent_replacement_lifetime_years"] = params["adsorbent_lifetime_years"]
    out["sorbent_replacement_cost_USD"] = (
        params["adsorbent_mass_kg"] * kg_to_lb * params["sorbent_replacement_cost_usd_per_lb"] / params["adsorbent_lifetime_years"]
        if params["adsorbent_lifetime_years"] > 0 else np.nan
    )

    # Energy-system CAPEX from module 03/04 capacities.
    out["PV_CAPEX_USD"] = finite_or_zero(out["pv_capacity_kWp"]) * params["pv_capex_usd_per_kw"]
    out["wind_CAPEX_USD"] = finite_or_zero(out["wind_capacity_kW"]) * params["wind_capex_usd_per_kw"]
    out["battery_CAPEX_USD"] = finite_or_zero(out["battery_capacity_kWh"]) * params["battery_capex_usd_per_kwh"]

    heatpump_heat = finite_or_zero(out["heat_from_heatpump_MWhth"])
    heatpump_active = pd.to_numeric(heatpump_heat, errors="coerce").fillna(0.0) > 1e-9

    avg_capacity = pd.to_numeric(heatpump_heat, errors="coerce").fillna(0.0) / 8760.0 * 1000.0
    design_capacity = pd.to_numeric(out.get("heat_pump_capacity_kWth_design", pd.Series(np.nan, index=out.index)), errors="coerce")
    p95_capacity = pd.to_numeric(out.get("p95_heat_demand_kWth", pd.Series(np.nan, index=out.index)), errors="coerce")
    peak_capacity = pd.to_numeric(out.get("peak_heat_demand_kWth", pd.Series(np.nan, index=out.index)), errors="coerce")

    basis = str(args.heat_pump_capacity_basis).strip().lower()
    if basis == "from_module03":
        used_capacity = design_capacity.copy()
        source = pd.Series("module03_heat_pump_capacity_kWth_design", index=out.index, dtype=object)
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, p95_capacity)
        source = source.where(~miss, "fallback_p95_heat_demand_kWth")
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, peak_capacity)
        source = source.where(~miss, "fallback_peak_heat_demand_kWth")
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, avg_capacity)
        source = source.where(~miss, "fallback_annual_average_heat_demand")
    elif basis == "p95":
        used_capacity = p95_capacity.copy()
        source = pd.Series("p95_heat_demand_kWth", index=out.index, dtype=object)
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, design_capacity)
        source = source.where(~miss, "fallback_module03_heat_pump_capacity_kWth_design")
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, avg_capacity)
        source = source.where(~miss, "fallback_annual_average_heat_demand")
    elif basis == "peak":
        used_capacity = peak_capacity.copy()
        source = pd.Series("peak_heat_demand_kWth", index=out.index, dtype=object)
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, design_capacity)
        source = source.where(~miss, "fallback_module03_heat_pump_capacity_kWth_design")
        miss = ~np.isfinite(used_capacity) | (used_capacity <= 0)
        used_capacity = used_capacity.where(~miss, avg_capacity)
        source = source.where(~miss, "fallback_annual_average_heat_demand")
    elif basis == "average":
        used_capacity = avg_capacity.copy()
        source = pd.Series("annual_average_heat_demand", index=out.index, dtype=object)
    else:
        raise ValueError("Unknown --heat-pump-capacity-basis. Use from_module03, p95, peak, or average.")

    # Geothermal-heat scenarios have no heat-pump heat load. Keep HP capacity and CAPEX at zero there.
    used_capacity = pd.to_numeric(used_capacity, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    used_capacity = used_capacity.where(heatpump_active, 0.0)
    source = source.where(heatpump_active, "no_heat_pump_in_scenario")

    out["heat_pump_capacity_kWth_average_proxy"] = avg_capacity.where(heatpump_active, 0.0)
    out["heat_pump_capacity_kWth_proxy"] = out["heat_pump_capacity_kWth_average_proxy"]  # backward-compatible name
    out["heat_pump_capacity_kWth_used"] = used_capacity
    out["heat_pump_capacity_source_05"] = source
    out["heat_pump_capacity_ratio_used_to_average"] = out["heat_pump_capacity_kWth_used"] / out["heat_pump_capacity_kWth_average_proxy"].replace(0, np.nan)
    out["heat_pump_CAPEX_USD"] = out["heat_pump_capacity_kWth_used"] * params["heat_pump_capex_usd_per_kwth"]

    geo_heat = finite_or_zero(out["geothermal_heat_used_MWhth"])
    out["geothermal_heat_capacity_kWth_proxy"] = geo_heat / 8760.0 * 1000.0
    out["geothermal_heat_CAPEX_USD"] = out["geothermal_heat_capacity_kWth_proxy"] * params["geothermal_heat_capex_usd_per_kwth"]

    out["annualized_PV_CAPEX_USD"] = out["PV_CAPEX_USD"] * out["CRF_PV"]
    out["annualized_wind_CAPEX_USD"] = out["wind_CAPEX_USD"] * out["CRF_wind"]
    out["annualized_battery_CAPEX_USD"] = out["battery_CAPEX_USD"] * out["CRF_battery"]
    out["annualized_heat_pump_CAPEX_USD"] = out["heat_pump_CAPEX_USD"] * out["CRF_heat_pump"]
    out["annualized_geothermal_heat_CAPEX_USD"] = out["geothermal_heat_CAPEX_USD"] * out["CRF_geothermal_heat"]

    out["annualized_energy_system_CAPEX_USD"] = (
        finite_or_zero(out["annualized_PV_CAPEX_USD"])
        + finite_or_zero(out["annualized_wind_CAPEX_USD"])
        + finite_or_zero(out["annualized_battery_CAPEX_USD"])
        + finite_or_zero(out["annualized_heat_pump_CAPEX_USD"])
        + finite_or_zero(out["annualized_geothermal_heat_CAPEX_USD"])
    )

    # Energy cost/emissions inherited from module 03. Do not silently convert missing cost to zero
    # unless the user explicitly requests a temporary lower-bound screening.
    out["energy_cost_USD_raw"] = pd.to_numeric(out["energy_cost_USD"], errors="coerce")
    out["energy_cost_missing_flag"] = out["energy_cost_USD_raw"].isna()
    out["energy_cost_missing_items"] = out.get("missing_energy_cost_items", "").fillna("").astype(str)
    out["energy_cost_missing_action"] = np.where(
        out["energy_cost_missing_flag"] & bool(args.allow_missing_energy_cost_as_zero),
        "set_to_zero_user_allowed_lower_bound",
        np.where(out["energy_cost_missing_flag"], "kept_missing_invalid_lcod", "not_missing"),
    )
    out["energy_cost_USD"] = out["energy_cost_USD_raw"].fillna(0.0) if args.allow_missing_energy_cost_as_zero else out["energy_cost_USD_raw"]

    out["energy_emissions_tCO2e_raw"] = pd.to_numeric(out["energy_emissions_tCO2e"], errors="coerce")
    out["energy_emissions_missing_flag"] = out["energy_emissions_tCO2e_raw"].isna()
    out["energy_emissions_tCO2e"] = out["energy_emissions_tCO2e_raw"]

    # Compression is skipped by default for the current workflow. Use --include-compression
    # when Aspen Plus or a manual kWh/tCO2 value is ready.
    if not args.include_compression:
        comp_kwh_t = 0.0
        out["compression_excluded_flag"] = True
        out["compression_scope"] = "pre_compression_or_compression_excluded"
        out["compression_kWh_per_tCO2"] = 0.0
        out["annual_compression_electricity_MWhe"] = 0.0
        out["compression_cost_USD"] = 0.0
        out["compression_cost_method"] = "compression_excluded_by_user"
        out["compression_emissions_tCO2e"] = 0.0
        out["compression_emission_method"] = "compression_excluded_by_user"
    else:
        comp_kwh_t = safe_float(comp.get("compression_kWh_per_tCO2"), np.nan)
        out["compression_excluded_flag"] = False
        out["compression_scope"] = "compression_included"
        out["compression_kWh_per_tCO2"] = comp_kwh_t
        out["annual_compression_electricity_MWhe"] = annual_co2 * comp_kwh_t / 1000.0 if np.isfinite(comp_kwh_t) else np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            out["effective_electricity_price_USD_MWh"] = np.where(
                out["annual_total_electricity_demand_MWhe"] > 0,
                out["energy_cost_USD"] / out["annual_total_electricity_demand_MWhe"],
                out["grid_electricity_price_USD_MWh"],
            )
            out["effective_electricity_EF_tCO2_MWh"] = np.where(
                out["annual_total_electricity_demand_MWhe"] > 0,
                out["energy_emissions_tCO2e"] / out["annual_total_electricity_demand_MWhe"],
                out["grid_emission_factor_tCO2_MWh"],
            )

        out["effective_electricity_price_USD_MWh"] = pd.to_numeric(out["effective_electricity_price_USD_MWh"], errors="coerce").fillna(args.fallback_electricity_price_USD_MWh)
        out["effective_electricity_EF_tCO2_MWh"] = pd.to_numeric(out["effective_electricity_EF_tCO2_MWh"], errors="coerce").fillna(args.fallback_electricity_EF_tCO2_MWh)

        comp_direct_cost = safe_float(comp.get("compression_cost_USD_per_tCO2_direct"), np.nan)
        if np.isfinite(comp_direct_cost):
            out["compression_cost_USD"] = annual_co2 * comp_direct_cost
            out["compression_cost_method"] = "direct_USD_per_tCO2_from_compression_csv"
        else:
            out["compression_cost_USD"] = out["annual_compression_electricity_MWhe"] * out["effective_electricity_price_USD_MWh"]
            out["compression_cost_method"] = "compression_electricity_times_effective_energy_price"

        comp_direct_ef = safe_float(comp.get("compression_emission_tCO2e_per_tCO2_direct"), np.nan)
        if np.isfinite(comp_direct_ef):
            out["compression_emissions_tCO2e"] = annual_co2 * comp_direct_ef
            out["compression_emission_method"] = "direct_tCO2e_per_tCO2_from_compression_csv"
        else:
            out["compression_emissions_tCO2e"] = out["annual_compression_electricity_MWhe"] * out["effective_electricity_EF_tCO2_MWh"]
            out["compression_emission_method"] = "compression_electricity_times_effective_energy_EF"

    # If compression is excluded, still define effective price/EF for downstream diagnostics.
    if "effective_electricity_price_USD_MWh" not in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["effective_electricity_price_USD_MWh"] = np.where(
                out["annual_total_electricity_demand_MWhe"] > 0,
                out["energy_cost_USD"] / out["annual_total_electricity_demand_MWhe"],
                out["grid_electricity_price_USD_MWh"],
            )
            out["effective_electricity_EF_tCO2_MWh"] = np.where(
                out["annual_total_electricity_demand_MWhe"] > 0,
                out["energy_emissions_tCO2e"] / out["annual_total_electricity_demand_MWhe"],
                out["grid_emission_factor_tCO2_MWh"],
            )

    out["annual_TandS_cost_USD"] = pd.to_numeric(out["annual_TandS_cost_USD"], errors="coerce")
    out["annual_CCS_emissions_tCO2e"] = pd.to_numeric(out["annual_CCS_emissions_tCO2e"], errors="coerce")

    # Costs. Energy and compression are intentionally allowed to remain NaN if missing and not explicitly skipped/allowed.
    out["annual_total_cost_DAC_only_USD"] = (
        finite_or_zero(out["annualized_DAC_CAPEX_USD"])
        + finite_or_zero(out["annualized_energy_system_CAPEX_USD"])
        + finite_or_zero(out["fixed_OPEX_DAC_USD"])
        + finite_or_zero(out["sorbent_replacement_cost_USD"])
        + out["energy_cost_USD"]
        + out["compression_cost_USD"]
    )
    out["annual_total_cost_DACCS_USD"] = out["annual_total_cost_DAC_only_USD"] + out["annual_TandS_cost_USD"]

    # Boundary-explicit emissions and net removal.
    out = add_boundary_explicit_emissions(out, annual_co2, args)

    # Backward-compatible legacy columns. These now point to the boundary-explicit
    # known CtG-energy-supply accounting so downstream 06/ranking can still use
    # the original column names without losing boundary information.
    out["annual_total_emissions_DAC_only_tCO2e"] = out["energy_emissions_tCO2e"] + out["compression_emissions_tCO2e"]
    out["annual_total_emissions_DACCS_tCO2e"] = out["emissions_CtG_energy_supply_tCO2e"]
    out["annual_net_CO2_removed_DAC_only_t"] = annual_co2 - out["annual_total_emissions_DAC_only_tCO2e"]
    out["annual_net_CO2_removed_DACCS_t"] = out["net_CO2_removed_CtG_energy_supply_t"]
    out["net_removal_efficiency_DAC_only"] = out["annual_net_CO2_removed_DAC_only_t"] / annual_co2.replace(0, np.nan)
    out["net_removal_efficiency_DACCS"] = out["removal_efficiency_CtG_energy_supply"]

    # LCOD values. Gross LCOD uses gross captured CO2. Net LCOD is reported per
    # boundary to avoid mixing functional units and system boundaries.
    out["LCOD_gross_DAC_only_USD_tCO2"] = out["annual_total_cost_DAC_only_USD"] / annual_co2.replace(0, np.nan)
    out["LCOD_gross_DACCS_USD_tCO2cap"] = out["annual_total_cost_DACCS_USD"] / annual_co2.replace(0, np.nan)
    out["LCOD_net_DAC_only_USD_tCO2"] = out["annual_total_cost_DAC_only_USD"] / out["annual_net_CO2_removed_DAC_only_t"].replace(0, np.nan)
    out["LCOD_net_DACCS_GtG_USD_tCO2net"] = out["annual_total_cost_DACCS_USD"] / out["net_CO2_removed_GtG_t"].replace(0, np.nan)
    out["LCOD_net_DACCS_CtG_energy_supply_USD_tCO2net"] = out["annual_total_cost_DACCS_USD"] / out["net_CO2_removed_CtG_energy_supply_t"].replace(0, np.nan)
    out["LCOD_net_DACCS_CtG_full_system_USD_tCO2net"] = out["annual_total_cost_DACCS_USD"] / out["net_CO2_removed_CtG_full_system_t"].replace(0, np.nan)
    out["LCOD_net_DACCS_USD_tCO2"] = out["LCOD_net_DACCS_CtG_energy_supply_USD_tCO2net"]

    # Methodology flags for transparent interpretation.
    out["heat_pump_CAPEX_excluded_flag"] = params["heat_pump_capex_usd_per_kwth"] == 0
    out["geothermal_heat_CAPEX_excluded_flag"] = params["geothermal_heat_capex_usd_per_kwth"] == 0
    out["LCOD_scope_note"] = np.where(
        out["compression_excluded_flag"],
        "Boundary-explicit LCOD; compression cost/emissions excluded by user, so DACCS result is pre-compression or incomplete for storage-ready CO2",
        "Boundary-explicit LCOD; compression included according to supplied Aspen/manual/fallback assumptions",
    )
    out["main_net_removal_boundary"] = "CtG_energy_supply_known"
    out["main_LCOD_boundary"] = "CtG_energy_supply_known"

    # Validity.
    invalid_reasons = []
    for idx, row in out.iterrows():
        reasons = []
        if not np.isfinite(row.get("annual_CO2_captured_t", np.nan)) or row.get("annual_CO2_captured_t", 0) <= 0:
            reasons.append("nonpositive_annual_CO2_captured")
        if row.get("energy_cost_missing_flag", False) and not args.allow_missing_energy_cost_as_zero:
            reasons.append("missing_energy_cost")
        if row.get("energy_emissions_missing_flag", False):
            reasons.append("missing_energy_emissions")
        if args.include_compression and (not np.isfinite(row.get("compression_kWh_per_tCO2", np.nan))):
            reasons.append("missing_compression_kWh_per_tCO2")
        if not np.isfinite(row.get("annual_total_cost_DACCS_USD", np.nan)):
            reasons.append("missing_or_invalid_total_cost_DACCS")
        if not np.isfinite(row.get("annual_net_CO2_removed_DACCS_t", np.nan)) or row.get("annual_net_CO2_removed_DACCS_t", 0) <= 0:
            reasons.append("nonpositive_net_CO2_removed_DACCS")
        if not np.isfinite(row.get("WACC_percent", np.nan)):
            reasons.append("missing_WACC")
        if row.get("scenario_valid_flag", True) in [False, "False", "false", 0, "0"]:
            reasons.append("energy_scenario_invalid")
        if row.get("ccs_valid_flag", True) in [False, "False", "false", 0, "0"]:
            reasons.append("ccs_invalid")
        if row.get("energy_ccs_valid_flag", True) in [False, "False", "false", 0, "0"]:
            reasons.append("energy_ccs_combined_invalid")
        invalid_reasons.append(";".join(reasons))
    out["lcod_invalid_reason"] = invalid_reasons
    out["valid_lcod_flag"] = out["lcod_invalid_reason"].eq("")

    # Metadata.
    out["cost_scenario"] = args.cost_scenario
    out["dac_capacity_mode"] = args.dac_capacity_mode
    out["project_lifetime_years_default"] = args.project_lifetime_years
    out["compression_source_file"] = comp.get("compression_source_file", "compression_excluded")

    return out


# =============================================================================
# Output tables
# =============================================================================

def build_breakdown_tables(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_cols = [c for c in [
        "country_code", "country_name", "province_id", "province_name", "operation_policy", "energy_scenario", "energy_scenario_base", "selected_transport_mode", "cost_scenario"
    ] if c in res.columns]

    cost_cols = id_cols + [
        "annual_CO2_captured_t",
        "DAC_CAPEX_USD", "annualized_DAC_CAPEX_USD",
        "PV_CAPEX_USD", "annualized_PV_CAPEX_USD",
        "wind_CAPEX_USD", "annualized_wind_CAPEX_USD",
        "battery_CAPEX_USD", "annualized_battery_CAPEX_USD",
        "heat_pump_capacity_kWth_design", "peak_heat_demand_kWth", "p95_heat_demand_kWth",
        "heat_pump_capacity_kWth_average_proxy", "heat_pump_capacity_kWth_used",
        "heat_pump_capacity_source_05", "heat_pump_capacity_ratio_used_to_average",
        "heat_pump_CAPEX_USD", "annualized_heat_pump_CAPEX_USD",
        "geothermal_heat_CAPEX_USD", "annualized_geothermal_heat_CAPEX_USD",
        "annualized_energy_system_CAPEX_USD",
        "fixed_OPEX_DAC_USD", "sorbent_replacement_cost_USD", "energy_cost_USD", "energy_cost_USD_raw", "energy_cost_missing_flag", "energy_cost_missing_items", "compression_cost_USD", "annual_TandS_cost_USD",
        "annual_total_cost_DAC_only_USD", "annual_total_cost_DACCS_USD",
    ]
    cost_breakdown = res[[c for c in cost_cols if c in res.columns]].copy()

    net_cols = id_cols + [
        "annual_CO2_captured_t", "gross_CO2_captured_t", "gross_CO2_stored_t", "CO2_loss_assumption",
        "energy_emissions_tCO2e", "process_energy_supply_emissions_tCO2e",
        "compression_emissions_tCO2e", "compression_electricity_emissions_tCO2e",
        "annual_CCS_emissions_tCO2e", "transport_storage_operational_emissions_tCO2e",
        "pv_lifecycle_emissions_tCO2e", "wind_lifecycle_emissions_tCO2e",
        "geothermal_heat_lifecycle_emissions_tCO2e", "battery_lifecycle_emissions_tCO2e",
        "sorbent_replacement_emissions_tCO2e", "DAC_infrastructure_emissions_tCO2e",
        "compression_infrastructure_emissions_tCO2e", "CCS_infrastructure_emissions_tCO2e",
        "emissions_GtG_operational_tCO2e", "emissions_CtG_energy_supply_tCO2e",
        "emissions_CtG_full_system_tCO2e",
        "annual_total_emissions_DAC_only_tCO2e", "annual_total_emissions_DACCS_tCO2e",
        "annual_net_CO2_removed_DAC_only_t", "annual_net_CO2_removed_DACCS_t",
        "net_CO2_removed_GtG_t", "net_CO2_removed_CtG_energy_supply_t",
        "net_CO2_removed_CtG_full_system_t",
        "net_removal_efficiency_DAC_only", "net_removal_efficiency_DACCS",
        "removal_efficiency_GtG", "removal_efficiency_CtG_energy_supply",
        "removal_efficiency_CtG_full_system",
        "emission_intensity_GtG_tCO2e_tCO2cap",
        "emission_intensity_CtG_energy_supply_tCO2e_tCO2cap",
        "emission_intensity_CtG_full_system_tCO2e_tCO2cap",
        "CtG_energy_supply_complete_flag", "CtG_energy_supply_scope_note",
        "CtG_full_system_complete_flag", "CtG_full_system_scope_note",
        "compression_excluded_flag", "LCOD_scope_note",
    ]
    net_breakdown = res[[c for c in net_cols if c in res.columns]].copy()

    summary_cols = id_cols + [
        "annual_CO2_captured_t", "gross_CO2_captured_t", "gross_CO2_stored_t",
        "annual_net_CO2_removed_DAC_only_t", "annual_net_CO2_removed_DACCS_t",
        "net_CO2_removed_GtG_t", "net_CO2_removed_CtG_energy_supply_t",
        "net_CO2_removed_CtG_full_system_t",
        "net_removal_efficiency_DACCS", "removal_efficiency_GtG",
        "removal_efficiency_CtG_energy_supply", "removal_efficiency_CtG_full_system",
        "emissions_GtG_operational_tCO2e", "emissions_CtG_energy_supply_tCO2e",
        "emissions_CtG_full_system_tCO2e",
        "annual_total_cost_DAC_only_USD", "annual_total_cost_DACCS_USD",
        "demand_profile_method", "hourly_profile_used_flag", "heat_pump_capacity_kWth_used",
        "heat_pump_capacity_source_05", "heat_pump_capacity_ratio_used_to_average",
        "LCOD_gross_DAC_only_USD_tCO2", "LCOD_gross_DACCS_USD_tCO2cap",
        "LCOD_net_DAC_only_USD_tCO2", "LCOD_net_DACCS_GtG_USD_tCO2net",
        "LCOD_net_DACCS_CtG_energy_supply_USD_tCO2net",
        "LCOD_net_DACCS_CtG_full_system_USD_tCO2net", "LCOD_net_DACCS_USD_tCO2",
        "WACC_percent", "CRF_DAC", "DAC_capacity_basis_tCO2yr", "valid_lcod_flag", "lcod_invalid_reason",
        "energy_cost_missing_flag", "energy_cost_missing_items",
        "CtG_energy_supply_complete_flag", "CtG_energy_supply_scope_note",
        "CtG_full_system_complete_flag", "CtG_full_system_scope_note",
        "main_net_removal_boundary", "main_LCOD_boundary",
        "compression_excluded_flag", "LCOD_scope_note",
    ]
    summary = res[[c for c in summary_cols if c in res.columns]].copy()

    diagnostics_rows = []
    diagnostics_rows.append({"check": "n_rows", "value": len(res)})
    diagnostics_rows.append({"check": "n_valid_lcod", "value": int(res.get("valid_lcod_flag", pd.Series(False, index=res.index)).sum())})
    diagnostics_rows.append({"check": "n_missing_energy_cost", "value": int(res.get("energy_cost_missing_flag", pd.Series(False, index=res.index)).sum())})
    diagnostics_rows.append({"check": "n_missing_geothermal_heat_price", "value": int(res.get("energy_cost_missing_items", pd.Series("", index=res.index)).astype(str).str.contains("geothermal_heat_price", na=False).sum())})
    diagnostics_rows.append({"check": "n_compression_excluded", "value": int(res.get("compression_excluded_flag", pd.Series(False, index=res.index)).sum())})
    diagnostics_rows.append({"check": "n_nonpositive_net_DACCS", "value": int((res.get("annual_net_CO2_removed_DACCS_t", pd.Series(np.nan, index=res.index)) <= 0).sum())})
    diagnostics_rows.append({"check": "n_nonpositive_net_GtG", "value": int((res.get("net_CO2_removed_GtG_t", pd.Series(np.nan, index=res.index)) <= 0).sum())})
    diagnostics_rows.append({"check": "n_nonpositive_net_CtG_energy_supply", "value": int((res.get("net_CO2_removed_CtG_energy_supply_t", pd.Series(np.nan, index=res.index)) <= 0).sum())})
    diagnostics_rows.append({"check": "n_nonpositive_net_CtG_full_system", "value": int((res.get("net_CO2_removed_CtG_full_system_t", pd.Series(np.nan, index=res.index)) <= 0).sum())})
    diagnostics_rows.append({"check": "n_CtG_energy_supply_incomplete_missing_lifecycle_EF", "value": int(res.get("CtG_energy_supply_missing_lifecycle_EF_flag", pd.Series(False, index=res.index)).astype(bool).sum())})
    diagnostics_rows.append({"check": "n_CtG_full_system_incomplete_missing_EF", "value": int(res.get("CtG_full_system_missing_EF_flag", pd.Series(False, index=res.index)).astype(bool).sum())})
    diagnostics_rows.append({"check": "n_missing_WACC", "value": int(res.get("WACC_percent", pd.Series(np.nan, index=res.index)).isna().sum())})
    diagnostics_rows.append({"check": "n_missing_compression_kWh_per_tCO2", "value": int(res.get("compression_kWh_per_tCO2", pd.Series(np.nan, index=res.index)).isna().sum())})
    wind_capacity = pd.to_numeric(res.get("wind_capacity_kW", pd.Series(0, index=res.index)), errors="coerce").fillna(0.0)
    diagnostics_rows.append({"check": "n_wind_capacity_gt_1000kW", "value": int((wind_capacity > 1_000).sum())})
    scenario_invalid = res.get("scenario_valid_flag", pd.Series(True, index=res.index)).astype(str).str.lower().isin(["false", "0", "no"])
    diagnostics_rows.append({"check": "n_energy_scenario_invalid", "value": int(scenario_invalid.sum())})
    upstream_invalid_reason = res.get("invalid_reason", pd.Series("", index=res.index)).fillna("").astype(str)
    lcod_invalid_reason = res.get("lcod_invalid_reason", pd.Series("", index=res.index)).fillna("").astype(str)
    diagnostics_rows.append({"check": "n_invalid_due_to_wind_capacity", "value": int((upstream_invalid_reason.str.contains("wind_capacity", case=False, na=False) | lcod_invalid_reason.str.contains("wind_capacity", case=False, na=False)).sum())})
    diagnostics_rows.append({"check": "n_hourly_profile_used_rows", "value": int(boolean_series(res, "hourly_profile_used_flag", default=False).sum())})
    hp_design = pd.to_numeric(res.get("heat_pump_capacity_kWth_design", pd.Series(np.nan, index=res.index)), errors="coerce")
    hp_used = pd.to_numeric(res.get("heat_pump_capacity_kWth_used", pd.Series(np.nan, index=res.index)), errors="coerce")
    hp_avg = pd.to_numeric(res.get("heat_pump_capacity_kWth_average_proxy", pd.Series(np.nan, index=res.index)), errors="coerce")
    diagnostics_rows.append({"check": "n_heat_pump_design_capacity_available", "value": int((hp_design > 0).sum())})
    diagnostics_rows.append({"check": "median_heat_pump_capacity_kWth_used", "value": float(hp_used[hp_used > 0].median()) if (hp_used > 0).any() else 0.0})
    diagnostics_rows.append({"check": "median_heat_pump_capacity_kWth_average_proxy", "value": float(hp_avg[hp_avg > 0].median()) if (hp_avg > 0).any() else 0.0})
    diagnostics_rows.append({"check": "median_heat_pump_used_to_average_ratio", "value": float((hp_used / hp_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).median())})
    diagnostics_rows.append({"check": "median_LCOD_net_DACCS_valid", "value": float(res.loc[res.get("valid_lcod_flag", False), "LCOD_net_DACCS_USD_tCO2"].median()) if "LCOD_net_DACCS_USD_tCO2" in res else np.nan})
    diagnostics = pd.DataFrame(diagnostics_rows)

    return summary, cost_breakdown, net_breakdown, diagnostics



def build_emission_breakdown_table(res: pd.DataFrame, args) -> pd.DataFrame:
    """Long-format emission component table for audit and method writing."""
    id_cols = [c for c in [
        "country_code", "country_name", "province_id", "province_name",
        "operation_policy", "energy_scenario", "energy_scenario_base",
        "selected_transport_mode", "cost_scenario",
    ] if c in res.columns]

    components = [
        {
            "boundary": "GtG_operational",
            "component": "process_energy_supply",
            "activity_col": "annual_total_electricity_demand_MWhe",
            "activity_unit": "MWhe_or_mixed_energy_basis_from_module03",
            "ef_col": "effective_electricity_EF_tCO2_MWh",
            "ef_arg": None,
            "ef_unit": "tCO2e/MWh_effective",
            "emission_col": "process_energy_supply_emissions_tCO2e",
            "source_note": "inherited from module 03/04 energy-supply evaluator",
        },
        {
            "boundary": "GtG_operational",
            "component": "compression_electricity",
            "activity_col": "annual_compression_electricity_MWhe",
            "activity_unit": "MWhe",
            "ef_col": "effective_electricity_EF_tCO2_MWh",
            "ef_arg": None,
            "ef_unit": "tCO2e/MWh",
            "emission_col": "compression_electricity_emissions_tCO2e",
            "source_note": "Aspen/manual/fallback compression electricity times effective electricity EF, unless direct EF supplied",
        },
        {
            "boundary": "GtG_operational",
            "component": "transport_storage_operational",
            "activity_col": "gross_CO2_stored_t",
            "activity_unit": "tCO2 stored/transported",
            "ef_col": None,
            "ef_arg": None,
            "ef_unit": "various inherited from module04",
            "emission_col": "transport_storage_operational_emissions_tCO2e",
            "source_note": "inherited from module 04 CCS evaluator",
        },
        {
            "boundary": "CtG_energy_supply",
            "component": "pv_lifecycle_sensitivity",
            "activity_col": "pv_generation_used_MWhe",
            "activity_unit": "MWhe",
            "ef_col": None,
            "ef_arg": "pv_lifecycle_ef_tco2e_mwh",
            "ef_unit": "tCO2e/MWh",
            "emission_col": "pv_lifecycle_emissions_tCO2e",
            "source_note": "optional lifecycle EF supplied by user/source; excluded and flagged if missing",
        },
        {
            "boundary": "CtG_energy_supply",
            "component": "wind_lifecycle_sensitivity",
            "activity_col": "wind_generation_used_MWhe",
            "activity_unit": "MWhe",
            "ef_col": None,
            "ef_arg": "wind_lifecycle_ef_tco2e_mwh",
            "ef_unit": "tCO2e/MWh",
            "emission_col": "wind_lifecycle_emissions_tCO2e",
            "source_note": "optional lifecycle EF supplied by user/source; excluded and flagged if missing",
        },
        {
            "boundary": "CtG_energy_supply",
            "component": "geothermal_heat_lifecycle_sensitivity",
            "activity_col": "geothermal_heat_used_MWhth",
            "activity_unit": "MWhth",
            "ef_col": None,
            "ef_arg": "geothermal_heat_lifecycle_ef_tco2e_mwhth",
            "ef_unit": "tCO2e/MWhth",
            "emission_col": "geothermal_heat_lifecycle_emissions_tCO2e",
            "source_note": "optional lifecycle EF supplied by user/source; excluded and flagged if missing",
        },
        {
            "boundary": "CtG_energy_supply",
            "component": "battery_embodied_sensitivity",
            "activity_col": "battery_capacity_kWh",
            "activity_unit": "kWh capacity",
            "ef_col": None,
            "ef_arg": "battery_embodied_ef_tco2e_kwh",
            "ef_unit": "tCO2e/kWh capacity, annualized by battery lifetime",
            "emission_col": "battery_lifecycle_emissions_tCO2e",
            "source_note": "optional embodied EF supplied by user/source; excluded and flagged if missing",
        },
        {
            "boundary": "CtG_full_system",
            "component": "sorbent_replacement_sensitivity",
            "activity_col": "adsorbent_mass_kg_basis",
            "activity_unit": "kg adsorbent basis",
            "ef_col": None,
            "ef_arg": "sorbent_embodied_ef_tco2e_kg",
            "ef_unit": "tCO2e/kg adsorbent, annualized by adsorbent lifetime",
            "emission_col": "sorbent_replacement_emissions_tCO2e",
            "source_note": "optional full-system sensitivity EF supplied by user/source",
        },
        {
            "boundary": "CtG_full_system",
            "component": "DAC_infrastructure_sensitivity",
            "activity_col": "DAC_capacity_basis_tCO2yr",
            "activity_unit": "tCO2/yr capacity basis",
            "ef_col": None,
            "ef_arg": "dac_infrastructure_ef_tco2e_tco2yr",
            "ef_unit": "tCO2e/(tCO2/yr capacity), annualized by project lifetime",
            "emission_col": "DAC_infrastructure_emissions_tCO2e",
            "source_note": "optional full-system sensitivity EF supplied by user/source",
        },
        {
            "boundary": "CtG_full_system",
            "component": "compression_infrastructure_sensitivity",
            "activity_col": "gross_CO2_captured_t",
            "activity_unit": "tCO2 captured",
            "ef_col": None,
            "ef_arg": "compression_infrastructure_ef_tco2e_tco2",
            "ef_unit": "tCO2e/tCO2 captured",
            "emission_col": "compression_infrastructure_emissions_tCO2e",
            "source_note": "optional full-system sensitivity EF supplied by user/source",
        },
        {
            "boundary": "CtG_full_system",
            "component": "CCS_infrastructure_sensitivity",
            "activity_col": "gross_CO2_stored_t",
            "activity_unit": "tCO2 stored",
            "ef_col": None,
            "ef_arg": "ccs_infrastructure_ef_tco2e_tco2",
            "ef_unit": "tCO2e/tCO2 stored",
            "emission_col": "CCS_infrastructure_emissions_tCO2e",
            "source_note": "optional full-system sensitivity EF supplied by user/source",
        },
    ]

    frames = []
    for comp in components:
        tmp = res[id_cols].copy() if id_cols else pd.DataFrame(index=res.index)
        tmp["boundary"] = comp["boundary"]
        tmp["emission_component"] = comp["component"]
        activity_col = comp["activity_col"]
        tmp["activity_value"] = pd.to_numeric(res.get(activity_col, pd.Series(np.nan, index=res.index)), errors="coerce")
        tmp["activity_unit"] = comp["activity_unit"]
        if comp["ef_col"] and comp["ef_col"] in res.columns:
            tmp["emission_factor"] = pd.to_numeric(res[comp["ef_col"]], errors="coerce")
            tmp["emission_factor_source"] = "row_value_from_upstream_or_effective_calculation"
        elif comp["ef_arg"] is not None:
            val = _arg_float(args, comp["ef_arg"], np.nan)
            tmp["emission_factor"] = val
            tmp["emission_factor_source"] = getattr(args, f"{comp['ef_arg']}_source", "user_arg_or_missing")
        else:
            tmp["emission_factor"] = np.nan
            tmp["emission_factor_source"] = comp["source_note"]
        tmp["emission_factor_unit"] = comp["ef_unit"]
        tmp["emissions_tCO2e"] = pd.to_numeric(res.get(comp["emission_col"], pd.Series(np.nan, index=res.index)), errors="coerce")
        tmp["source_note"] = comp["source_note"]
        tmp["data_quality"] = np.where(
            tmp["emissions_tCO2e"].notna() & (tmp["emissions_tCO2e"].abs() > 0),
            "included_nonzero",
            np.where(tmp["activity_value"].fillna(0.0).abs() > 0, "zero_or_excluded_check_status_flags", "not_applicable_no_activity"),
        )
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_best_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = summary.copy()
    if "valid_lcod_flag" in valid.columns:
        valid = valid[valid["valid_lcod_flag"] == True].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)

    group_cols = [c for c in ["country_code", "province_id", "province_name"] if c in valid.columns]
    if not group_cols or valid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    best_lcod = valid.dropna(subset=["LCOD_net_DACCS_USD_tCO2"]).sort_values("LCOD_net_DACCS_USD_tCO2").groupby(group_cols, as_index=False).head(1)
    best_net = valid.dropna(subset=["annual_net_CO2_removed_DACCS_t"]).sort_values("annual_net_CO2_removed_DACCS_t", ascending=False).groupby(group_cols, as_index=False).head(1)

    # Simple multiobjective score: percentile of low LCOD + high net removal + high net efficiency.
    mo = valid.copy()
    if not mo.empty:
        lcod = pd.to_numeric(mo["LCOD_net_DACCS_USD_tCO2"], errors="coerce")
        net = pd.to_numeric(mo["annual_net_CO2_removed_DACCS_t"], errors="coerce")
        eff = pd.to_numeric(mo["net_removal_efficiency_DACCS"], errors="coerce")
        mo["score_low_lcod"] = 1.0 - lcod.rank(pct=True)
        mo["score_high_net"] = net.rank(pct=True)
        mo["score_high_efficiency"] = eff.rank(pct=True)
        mo["multiobjective_screening_score"] = 0.50 * mo["score_low_lcod"] + 0.30 * mo["score_high_net"] + 0.20 * mo["score_high_efficiency"]
        best_mo = mo.dropna(subset=["multiobjective_screening_score"]).sort_values("multiobjective_screening_score", ascending=False).groupby(group_cols, as_index=False).head(1)
    else:
        best_mo = pd.DataFrame()

    return best_lcod, best_net, best_mo


def write_best_scenario_diagnostics(summary: pd.DataFrame, best_lcod: pd.DataFrame, out_dir: Path) -> None:
    """Write compact diagnostics showing whether one energy scenario dominates LCOD ranking."""
    diag_dir = ensure_dir(out_dir / "diagnostics")

    if best_lcod is not None and not best_lcod.empty:
        scenario_count = (
            best_lcod.get("energy_scenario_base", pd.Series("unknown", index=best_lcod.index))
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        scenario_count.columns = ["energy_scenario_base", "n_best_provinces"]
        scenario_count["share_best_provinces"] = scenario_count["n_best_provinces"] / max(float(scenario_count["n_best_provinces"].sum()), 1.0)
        scenario_count.to_csv(diag_dir / "best_lcod_scenario_count.csv", index=False, encoding="utf-8-sig")

        if {"country_code", "energy_scenario_base"}.issubset(best_lcod.columns):
            by_country = (
                best_lcod.groupby(["country_code", "energy_scenario_base"], dropna=False)
                .size()
                .reset_index(name="n_best_provinces")
                .sort_values(["country_code", "n_best_provinces"], ascending=[True, False])
            )
            by_country.to_csv(diag_dir / "best_lcod_by_country_scenario_count.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["country_code", "energy_scenario_base", "n_best_provinces"]).to_csv(
                diag_dir / "best_lcod_by_country_scenario_count.csv", index=False, encoding="utf-8-sig"
            )
    else:
        pd.DataFrame(columns=["energy_scenario_base", "n_best_provinces", "share_best_provinces"]).to_csv(
            diag_dir / "best_lcod_scenario_count.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(columns=["country_code", "energy_scenario_base", "n_best_provinces"]).to_csv(
            diag_dir / "best_lcod_by_country_scenario_count.csv", index=False, encoding="utf-8-sig"
        )

    valid = summary.copy()
    if "valid_lcod_flag" in valid.columns:
        valid = valid[valid["valid_lcod_flag"] == True].copy()
    if not valid.empty and {"energy_scenario_base", "LCOD_net_DACCS_USD_tCO2"}.issubset(valid.columns):
        med = (
            valid.assign(LCOD_net_DACCS_USD_tCO2=pd.to_numeric(valid["LCOD_net_DACCS_USD_tCO2"], errors="coerce"))
            .groupby("energy_scenario_base", dropna=False)
            .agg(
                n_valid_rows=("LCOD_net_DACCS_USD_tCO2", "count"),
                median_LCOD_net_DACCS_USD_tCO2=("LCOD_net_DACCS_USD_tCO2", "median"),
                p10_LCOD_net_DACCS_USD_tCO2=("LCOD_net_DACCS_USD_tCO2", lambda s: s.quantile(0.10)),
                p90_LCOD_net_DACCS_USD_tCO2=("LCOD_net_DACCS_USD_tCO2", lambda s: s.quantile(0.90)),
            )
            .reset_index()
            .sort_values("median_LCOD_net_DACCS_USD_tCO2")
        )
    else:
        med = pd.DataFrame(columns=[
            "energy_scenario_base", "n_valid_rows", "median_LCOD_net_DACCS_USD_tCO2",
            "p10_LCOD_net_DACCS_USD_tCO2", "p90_LCOD_net_DACCS_USD_tCO2",
        ])
    med.to_csv(diag_dir / "median_lcod_by_energy_scenario_base.csv", index=False, encoding="utf-8-sig")


def write_heat_pump_capacity_diagnostics(result: pd.DataFrame, out_dir: Path) -> None:
    """Write diagnostics for design-capacity based heat-pump CAPEX sizing."""
    diag_dir = ensure_dir(out_dir / "diagnostics")
    if result.empty:
        pd.DataFrame().to_csv(diag_dir / "heat_pump_capacity_diagnostics.csv", index=False, encoding="utf-8-sig")
        return

    hp_rows = result[pd.to_numeric(result.get("heat_from_heatpump_MWhth", pd.Series(0, index=result.index)), errors="coerce").fillna(0.0) > 1e-9].copy()
    if hp_rows.empty:
        summary = pd.DataFrame([{
            "group": "all",
            "n_rows": 0,
            "median_heat_pump_capacity_used_kWth": 0.0,
            "median_heat_pump_capacity_average_proxy_kWth": 0.0,
            "median_used_to_average_ratio": np.nan,
        }])
    else:
        hp_rows["heat_pump_capacity_kWth_used"] = pd.to_numeric(hp_rows.get("heat_pump_capacity_kWth_used"), errors="coerce")
        hp_rows["heat_pump_capacity_kWth_average_proxy"] = pd.to_numeric(hp_rows.get("heat_pump_capacity_kWth_average_proxy"), errors="coerce")
        hp_rows["heat_pump_capacity_ratio_used_to_average"] = pd.to_numeric(hp_rows.get("heat_pump_capacity_ratio_used_to_average"), errors="coerce")
        group_cols = [c for c in ["energy_scenario_base", "operation_policy", "heat_pump_capacity_source_05"] if c in hp_rows.columns]
        summary = (
            hp_rows.groupby(group_cols, dropna=False)
            .agg(
                n_rows=("heat_pump_capacity_kWth_used", "size"),
                median_heat_pump_capacity_used_kWth=("heat_pump_capacity_kWth_used", "median"),
                p95_heat_pump_capacity_used_kWth=("heat_pump_capacity_kWth_used", lambda s: s.quantile(0.95)),
                median_heat_pump_capacity_average_proxy_kWth=("heat_pump_capacity_kWth_average_proxy", "median"),
                median_used_to_average_ratio=("heat_pump_capacity_ratio_used_to_average", "median"),
            )
            .reset_index()
        )
    summary.to_csv(diag_dir / "heat_pump_capacity_diagnostics.csv", index=False, encoding="utf-8-sig")



# =============================================================================
# Visualization and spatial maps
# =============================================================================

def set_consulting_style() -> dict[str, str]:
    palette = {
        "navy": "#0B1F3A",
        "blue": "#1F77B4",
        "sky": "#8ECAE6",
        "teal": "#2A9D8F",
        "orange": "#E76F51",
        "yellow": "#E9C46A",
        "gray": "#9AA0A6",
        "light_gray": "#EEF1F4",
        "dark_gray": "#343A40",
        "white": "#FFFFFF",
    }
    if plt is not None:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans", "Liberation Sans"],
            "figure.facecolor": palette["white"],
            "axes.facecolor": palette["white"],
            "savefig.facecolor": palette["white"],
            "axes.edgecolor": palette["dark_gray"],
            "axes.labelcolor": palette["dark_gray"],
            "axes.titlecolor": palette["navy"],
            "xtick.color": palette["dark_gray"],
            "ytick.color": palette["dark_gray"],
            "axes.grid": True,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        })
    return palette


def simple_scenario_base(x: Any) -> str:
    s = str(x)
    if s.startswith("S0"):
        return "S0 Grid+HP"
    if s.startswith("S1"):
        return "S1 PV/Wind+HP"
    if s.startswith("S2"):
        return "S2 PV/Wind+Battery+HP"
    if s.startswith("S3"):
        return "S3 Grid+Geothermal heat"
    if s.startswith("S4"):
        return "S4 PV/Wind+Battery+Geothermal"
    return s[:40]


def find_spatial_gpkg(paths: dict[str, Path]) -> Path | None:
    explicit = paths.get("spatial_gpkg")
    if explicit is not None and explicit.exists():
        return explicit
    spatial_dir = paths.get("spatial_dir")
    if spatial_dir is None or not spatial_dir.exists():
        return None
    preferred = spatial_dir / "ASEAN" / "ASEAN_PROVINCES_LEVEL1.gpkg"
    if preferred.exists():
        return preferred
    candidates = []
    for pat in ["**/*ASEAN*PROVINCES*LEVEL1*.gpkg", "**/*PROVINCES*LEVEL1*.gpkg", "**/*.gpkg"]:
        candidates.extend(spatial_dir.glob(pat))
    if not candidates:
        return None
    def score(p: Path) -> int:
        n = p.name.lower()
        return int("asean" in n) * 20 + int("level1" in n) * 10 + int("province" in n) * 5
    return sorted(candidates, key=score, reverse=True)[0]


def _prep_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["country_code", "province_id", "province_name"]:
        if c not in out.columns:
            out[c] = ""
    out["country_code_join"] = out["country_code"].astype(str).str.strip().str.upper()
    out["province_id_join"] = out["province_id"].astype(str).str.strip()
    out["province_name_join"] = out["province_name"].astype(str).str.strip().str.lower()
    return out


def join_best_to_map(gdf, best_df: pd.DataFrame):
    g = _prep_join_keys(gdf)
    b = _prep_join_keys(best_df)
    keep_cols = [
        "country_code_join", "province_id_join", "province_name_join",
        "LCOD_net_DACCS_USD_tCO2", "annual_net_CO2_removed_DACCS_t", "net_removal_efficiency_DACCS",
        "operation_policy", "energy_scenario", "energy_scenario_base", "selected_transport_mode", "valid_lcod_flag",
    ]
    keep_cols = [c for c in keep_cols if c in b.columns]
    b_id = b[keep_cols].drop_duplicates(["country_code_join", "province_id_join", "province_name_join"], keep="first")
    joined = g.merge(b_id, on=["country_code_join", "province_id_join", "province_name_join"], how="left")
    if joined.get("LCOD_net_DACCS_USD_tCO2", pd.Series(index=joined.index, dtype=float)).notna().sum() == 0:
        b_name = b[[c for c in keep_cols if c != "province_id_join"]].drop_duplicates(["country_code_join", "province_name_join"], keep="first")
        joined = g.merge(b_name, on=["country_code_join", "province_name_join"], how="left")
    return joined


def finish_map(ax, title: str, palette: dict[str, str]) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(palette["gray"])
    ax.spines["bottom"].set_color(palette["gray"])


def plot_outputs(result: pd.DataFrame, summary: pd.DataFrame, best_lcod: pd.DataFrame, out_dir: Path, paths: dict[str, Path], args) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    map_dir = ensure_dir(out_dir / "maps")
    diag_dir = ensure_dir(out_dir / "diagnostics")
    palette = set_consulting_style()

    scenario_colors = {
        "S0 Grid+HP": palette["gray"],
        "S1 PV/Wind+HP": palette["blue"],
        "S2 PV/Wind+Battery+HP": palette["teal"],
        "S3 Grid+Geothermal heat": palette["yellow"],
        "S4 PV/Wind+Battery+Geothermal": palette["orange"],
        "No valid LCOD": palette["light_gray"],
    }
    scenario_order_master = [
        "S0 Grid+HP",
        "S1 PV/Wind+HP",
        "S2 PV/Wind+Battery+HP",
        "S3 Grid+Geothermal heat",
        "S4 PV/Wind+Battery+Geothermal",
    ]
    lcod_col = "LCOD_net_DACCS_USD_tCO2" if "LCOD_net_DACCS_USD_tCO2" in result.columns else "LCOD_net_DACCS_CtG_energy_supply_USD_tCO2net"

    def _valid_mask(df: pd.DataFrame) -> pd.Series:
        if "valid_lcod_flag" not in df.columns:
            return pd.Series(True, index=df.index)
        s = df["valid_lcod_flag"]
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        return s.astype(str).str.lower().isin(["true", "1", "yes"])

    def _scenario_order(df: pd.DataFrame) -> list[str]:
        available = [s for s in scenario_order_master if s in df["scenario_group"].dropna().unique().tolist()]
        others = [s for s in df["scenario_group"].dropna().unique().tolist() if s not in available]
        return available + sorted(others)

    def _select_extremes(df: pd.DataFrame, metric_col: str, n: int = 10) -> pd.DataFrame:
        work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_col]).sort_values(metric_col).copy()
        if work.empty:
            return work
        low = work.head(n).copy()
        low["extreme_group"] = "Lowest LCOD"
        low["extreme_rank"] = [f"L{i+1:02d}" for i in range(len(low))]

        high = work.sort_values(metric_col, ascending=False).head(n).copy()
        high = high.loc[~high.index.isin(low.index)].copy()
        high["extreme_group"] = "Highest LCOD"
        high["extreme_rank"] = [f"H{i+1:02d}" for i in range(len(high))]

        comb = pd.concat([low, high], axis=0)
        if comb.empty:
            return comb
        comb["case_label"] = (
            comb["extreme_rank"].astype(str)
            + " | "
            + comb.get("country_code", pd.Series("", index=comb.index)).astype(str)
            + "-"
            + comb.get("province_name", pd.Series("", index=comb.index)).astype(str).str.slice(0, 16)
        )
        comb["case_label"] = comb["case_label"].str.rstrip("-")
        comb["sort_group"] = np.where(comb["extreme_group"] == "Lowest LCOD", 0, 1)
        comb["sort_rank_num"] = comb["extreme_rank"].str.extract(r"(\d+)").astype(float)
        comb = comb.sort_values(["sort_group", "sort_rank_num"], ascending=[True, True]).copy()
        return comb

    def _stacked_bar_extremes(df: pd.DataFrame, metric_col: str, denom_col: str, components: list[tuple[str, str, str]],
                              title: str, ylabel: str, filename: str, figsize=(16, 7)) -> None:
        ext = _select_extremes(df, metric_col, n=10)
        if ext.empty:
            return
        denom = pd.to_numeric(ext.get(denom_col, np.nan), errors="coerce").replace(0, np.nan)
        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(ext), dtype=float)
        labels = ext["case_label"].tolist()
        for comp_label, comp_col, comp_color in components:
            vals = pd.to_numeric(ext.get(comp_col, 0.0), errors="coerce").fillna(0.0) / denom
            vals = vals.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ax.bar(labels, vals, bottom=bottom, label=comp_label, color=comp_color)
            bottom += vals.to_numpy(dtype=float)
        split_at = (ext["extreme_group"] == "Lowest LCOD").sum() - 0.5
        if split_at > 0 and split_at < len(ext) - 0.5:
            ax.axvline(split_at, color=palette["navy"], linestyle="--", linewidth=1.0)
            ymax = float(np.nanmax(bottom)) if len(bottom) else 0.0
            ytext = ymax * 1.02 if np.isfinite(ymax) and ymax > 0 else 0.02
            ax.text(split_at - 4.5, ytext, "10 lowest", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(split_at + 5.0, ytext, "10 highest", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y")
        ax.legend(ncol=3, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=280, bbox_inches="tight")
        plt.close(fig)

    if plt is not None:
        # 1. LCOD distribution by scenario base with fixed scenario-color mapping.
        try:
            valid = summary[_valid_mask(summary)].copy()
            if not valid.empty and lcod_col in valid.columns:
                valid["scenario_group"] = valid.get("energy_scenario_base", valid.get("energy_scenario", "")).map(simple_scenario_base)
                order = _scenario_order(valid)
                data = [pd.to_numeric(valid.loc[valid["scenario_group"] == s, lcod_col], errors="coerce").dropna().values for s in order]
                data = [d for d in data if len(d) > 0]
                order = [s for s in order if len(pd.to_numeric(valid.loc[valid["scenario_group"] == s, lcod_col], errors="coerce").dropna()) > 0]
                if data:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
                    for patch, scen in zip(bp["boxes"], order):
                        patch.set_facecolor(scenario_colors.get(scen, palette["light_gray"]))
                        patch.set_alpha(0.82)
                        patch.set_edgecolor(palette["navy"])
                    for med in bp["medians"]:
                        med.set_color(palette["navy"])
                        med.set_linewidth(1.5)
                    ax.set_ylabel("LCOD net DACCS (USD/tCO₂ net removed)")
                    ax.set_title("LCOD distribution by energy-supply scenario", loc="left", fontweight="bold")
                    ax.tick_params(axis="x", rotation=25)
                    ax.grid(axis="y")
                    fig.tight_layout()
                    fig.savefig(fig_dir / "lcod_net_DACCS_boxplot_by_energy_scenario.png", dpi=280, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            (fig_dir / "lcod_net_DACCS_boxplot_by_energy_scenario_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # 2. Cost breakdown of the 10 lowest and 10 highest LCOD cases.
        try:
            valid_full = result[_valid_mask(result)].copy()
            valid_full = valid_full.replace([np.inf, -np.inf], np.nan).dropna(subset=[lcod_col])
            if not valid_full.empty:
                cost_components = [
                    ("DAC CAPEX", "annualized_DAC_CAPEX_USD", palette["navy"]),
                    ("Energy system CAPEX", "annualized_energy_system_CAPEX_USD", palette["blue"]),
                    ("Fixed OPEX", "fixed_OPEX_DAC_USD", palette["teal"]),
                    ("Sorbent", "sorbent_replacement_cost_USD", palette["sky"]),
                    ("Energy", "energy_cost_USD", palette["yellow"]),
                    ("Compression", "compression_cost_USD", palette["gray"]),
                    ("T&S", "annual_TandS_cost_USD", palette["orange"]),
                ]
                _stacked_bar_extremes(
                    valid_full,
                    metric_col=lcod_col,
                    denom_col="annual_net_CO2_removed_DACCS_t",
                    components=cost_components,
                    title="Cost composition of the 10 lowest and 10 highest LCOD cases",
                    ylabel="USD/tCO₂ net removed",
                    filename="cost_breakdown_top10_lowest_lcod.png",
                )
        except Exception as exc:
            (fig_dir / "cost_breakdown_top10_lowest_lcod_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # 3. Emission intensity by energy scenario.
        try:
            valid = summary[_valid_mask(summary)].copy()
            col = "emission_intensity_CtG_energy_supply_tCO2e_tCO2cap"
            if not valid.empty and col in valid.columns:
                valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
                valid["scenario_group"] = valid.get("energy_scenario_base", valid.get("energy_scenario", "")).map(simple_scenario_base)
                order = _scenario_order(valid)
                data = [pd.to_numeric(valid.loc[valid["scenario_group"] == s, col], errors="coerce").dropna().values for s in order]
                data = [d for d in data if len(d) > 0]
                order = [s for s in order if len(pd.to_numeric(valid.loc[valid["scenario_group"] == s, col], errors="coerce").dropna()) > 0]
                if data:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
                    for patch, scen in zip(bp["boxes"], order):
                        patch.set_facecolor(scenario_colors.get(scen, palette["light_gray"]))
                        patch.set_alpha(0.82)
                        patch.set_edgecolor(palette["navy"])
                    for med in bp["medians"]:
                        med.set_color(palette["navy"])
                        med.set_linewidth(1.5)
                    ax.set_ylabel("Emission intensity (tCO₂e/tCO₂ captured)")
                    ax.set_title("CtG energy-supply emission intensity by energy-supply scenario", loc="left", fontweight="bold")
                    ax.tick_params(axis="x", rotation=25)
                    ax.grid(axis="y")
                    fig.tight_layout()
                    fig.savefig(fig_dir / "emission_intensity_CtG_energy_supply_by_energy_scenario.png", dpi=280, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            (fig_dir / "emission_intensity_CtG_energy_supply_by_energy_scenario_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # 4. Removal efficiency by energy scenario.
        try:
            valid = summary[_valid_mask(summary)].copy()
            col = "removal_efficiency_CtG_energy_supply"
            if not valid.empty and col in valid.columns:
                valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
                valid["scenario_group"] = valid.get("energy_scenario_base", valid.get("energy_scenario", "")).map(simple_scenario_base)
                order = _scenario_order(valid)
                data = [pd.to_numeric(valid.loc[valid["scenario_group"] == s, col], errors="coerce").dropna().values for s in order]
                data = [d for d in data if len(d) > 0]
                order = [s for s in order if len(pd.to_numeric(valid.loc[valid["scenario_group"] == s, col], errors="coerce").dropna()) > 0]
                if data:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
                    for patch, scen in zip(bp["boxes"], order):
                        patch.set_facecolor(scenario_colors.get(scen, palette["light_gray"]))
                        patch.set_alpha(0.82)
                        patch.set_edgecolor(palette["navy"])
                    for med in bp["medians"]:
                        med.set_color(palette["navy"])
                        med.set_linewidth(1.5)
                    ax.axhline(0.0, color=palette["gray"], linewidth=1.0, linestyle="--")
                    ax.axhline(0.5, color=palette["light_gray"], linewidth=0.8, linestyle=":")
                    ax.axhline(0.9, color=palette["light_gray"], linewidth=0.8, linestyle=":")
                    ax.set_ylabel("Removal efficiency (-)")
                    ax.set_title("CtG energy-supply removal efficiency by energy-supply scenario", loc="left", fontweight="bold")
                    ax.tick_params(axis="x", rotation=25)
                    ax.grid(axis="y")
                    fig.tight_layout()
                    fig.savefig(fig_dir / "removal_efficiency_CtG_energy_supply_by_energy_scenario.png", dpi=280, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            (fig_dir / "removal_efficiency_CtG_energy_supply_by_energy_scenario_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # 5. Emission breakdown of the 10 lowest and 10 highest LCOD cases.
        try:
            valid_full = result[_valid_mask(result)].copy()
            valid_full = valid_full.replace([np.inf, -np.inf], np.nan).dropna(subset=[lcod_col])
            if not valid_full.empty:
                emission_components = [
                    ("Process energy", "process_energy_supply_emissions_tCO2e", palette["navy"]),
                    ("Compression", "compression_electricity_emissions_tCO2e", palette["gray"]),
                    ("Transport+storage", "transport_storage_operational_emissions_tCO2e", palette["orange"]),
                    ("PV lifecycle", "pv_lifecycle_emissions_tCO2e", palette["blue"]),
                    ("Wind lifecycle", "wind_lifecycle_emissions_tCO2e", palette["teal"]),
                    ("Geothermal lifecycle", "geothermal_heat_lifecycle_emissions_tCO2e", palette["yellow"]),
                    ("Battery lifecycle", "battery_lifecycle_emissions_tCO2e", palette["sky"]),
                    ("Sorbent", "sorbent_replacement_emissions_tCO2e", palette["light_gray"]),
                    ("DAC infrastructure", "DAC_infrastructure_emissions_tCO2e", palette["navy"]),
                    ("Compression infrastructure", "compression_infrastructure_emissions_tCO2e", palette["blue"]),
                    ("CCS infrastructure", "CCS_infrastructure_emissions_tCO2e", palette["teal"]),
                ]
                _stacked_bar_extremes(
                    valid_full,
                    metric_col=lcod_col,
                    denom_col="gross_CO2_captured_t",
                    components=emission_components,
                    title="Emission composition of the 10 lowest and 10 highest LCOD cases",
                    ylabel="tCO₂e/tCO₂ captured",
                    filename="emission_breakdown_top10_lowest_highest_lcod.png",
                    figsize=(18, 7),
                )
        except Exception as exc:
            (fig_dir / "emission_breakdown_top10_lowest_highest_lcod_ERROR.txt").write_text(str(exc), encoding="utf-8")

        # 6. Trade-off scatter: LCOD vs emission intensity.
        try:
            valid = summary[_valid_mask(summary)].copy()
            xcol = lcod_col
            ycol = "emission_intensity_CtG_energy_supply_tCO2e_tCO2cap"
            size_col = "annual_net_CO2_removed_DACCS_t"
            if not valid.empty and xcol in valid.columns and ycol in valid.columns:
                valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol])
                valid["scenario_group"] = valid.get("energy_scenario_base", valid.get("energy_scenario", "")).map(simple_scenario_base)
                fig, ax = plt.subplots(figsize=(11, 7))
                sizes = pd.to_numeric(valid.get(size_col, 1.0), errors="coerce").fillna(0.0)
                smin = float(sizes.min()) if len(sizes) else 0.0
                smax = float(sizes.max()) if len(sizes) else 1.0
                size_scaled = 50.0 + 250.0 * (sizes - smin) / max(smax - smin, 1e-12)
                for scen in _scenario_order(valid):
                    sub = valid[valid["scenario_group"] == scen]
                    if sub.empty:
                        continue
                    ax.scatter(
                        pd.to_numeric(sub[xcol], errors="coerce"),
                        pd.to_numeric(sub[ycol], errors="coerce"),
                        s=size_scaled.loc[sub.index],
                        alpha=0.75,
                        label=scen,
                        color=scenario_colors.get(scen, palette["light_gray"]),
                        edgecolors=palette["navy"],
                        linewidths=0.4,
                    )
                ax.set_xlabel("LCOD net DACCS (USD/tCO₂ net removed)")
                ax.set_ylabel("Emission intensity (tCO₂e/tCO₂ captured)")
                ax.set_title("Trade-off between LCOD and emission intensity", loc="left", fontweight="bold")
                ax.grid(True)
                ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                fig.savefig(fig_dir / "lcod_vs_emission_intensity_tradeoff.png", dpi=280, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            (fig_dir / "lcod_vs_emission_intensity_tradeoff_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Spatial maps.
    if args.disable_spatial_maps:
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text("Spatial maps disabled by --disable-spatial-maps", encoding="utf-8")
        return
    if gpd is None or plt is None:
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text(
            f"geopandas or spatial plotting dependency is unavailable: geopandas={_GPD_IMPORT_ERROR}; matplotlib={_MPL_IMPORT_ERROR}",
            encoding="utf-8",
        )
        return
    spatial = find_spatial_gpkg(paths)
    if spatial is None or not spatial.exists():
        (fig_dir / "SPATIAL_MAP_NOT_CREATED.txt").write_text("ASEAN spatial GPKG not found.", encoding="utf-8")
        return

    try:
        from matplotlib.patches import Patch

        gdf = gpd.read_file(spatial)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        joined = join_best_to_map(gdf, best_lcod)
        joined.to_file(map_dir / "best_lcod_net_DACCS_map_data.geojson", driver="GeoJSON")
        joined.drop(columns="geometry", errors="ignore").to_csv(map_dir / "best_lcod_net_DACCS_map_join_table.csv", index=False, encoding="utf-8-sig")

        diag = pd.DataFrame([
            {"check": "spatial_file", "value": str(spatial)},
            {"check": "n_polygons", "value": len(joined)},
            {"check": "n_joined_best_lcod", "value": int(joined.get(lcod_col, pd.Series(index=joined.index)).notna().sum())},
        ])
        diag.to_csv(diag_dir / "map_join_diagnostics_lcod.csv", index=False, encoding="utf-8-sig")

        # Map 1: best LCOD.
        fig, ax = plt.subplots(figsize=(12, 8))
        joined.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
        vals = pd.to_numeric(joined.get(lcod_col), errors="coerce")
        if vals.notna().any():
            joined.assign(_val=vals).plot(
                column="_val", ax=ax, cmap="YlGnBu", legend=True,
                edgecolor="white", linewidth=0.20,
                legend_kwds={"label": "Best LCOD net DACCS (USD/tCO₂)", "shrink": 0.72},
            )
        finish_map(ax, "Lowest net-DACCS LCOD by province", palette)
        fig.tight_layout()
        fig.savefig(fig_dir / "map_best_LCOD_net_DACCS_by_province.png", dpi=280, bbox_inches="tight")
        plt.close(fig)

        # Map 2: best net removal.
        fig, ax = plt.subplots(figsize=(12, 8))
        joined.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
        vals = pd.to_numeric(joined.get("annual_net_CO2_removed_DACCS_t"), errors="coerce")
        if vals.notna().any():
            joined.assign(_val=vals).plot(
                column="_val", ax=ax, cmap="PuBuGn", legend=True,
                edgecolor="white", linewidth=0.20,
                legend_kwds={"label": "Net CO₂ removed (tCO₂/yr per 1000 kg ads)", "shrink": 0.72},
            )
        finish_map(ax, "Net CO₂ removal of lowest-LCOD case by province", palette)
        fig.tight_layout()
        fig.savefig(fig_dir / "map_net_CO2_removed_for_best_LCOD_case.png", dpi=280, bbox_inches="tight")
        plt.close(fig)

        # Map 3: best energy scenario category, using the same colors as the boxplot.
        if "energy_scenario" in joined.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            joined.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
            mode = joined["energy_scenario"].map(simple_scenario_base).fillna("No valid LCOD")
            plot_gdf = joined.assign(_scenario=mode)
            handles = []
            for label in scenario_order_master + ["No valid LCOD"]:
                color = scenario_colors.get(label, palette["light_gray"])
                sub = plot_gdf[plot_gdf["_scenario"] == label]
                if not sub.empty:
                    sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.20)
                handles.append(Patch(facecolor=color, edgecolor="white", label=label))
            finish_map(ax, "Energy scenario selected by lowest net-DACCS LCOD", palette)
            ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, title="Scenario")
            fig.tight_layout()
            fig.savefig(fig_dir / "map_best_energy_scenario_by_LCOD.png", dpi=280, bbox_inches="tight")
            plt.close(fig)

        # Map 4: removal efficiency for the best-LCOD case.
        if "removal_efficiency_CtG_energy_supply" in joined.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            joined.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
            vals = pd.to_numeric(joined.get("removal_efficiency_CtG_energy_supply"), errors="coerce")
            if vals.notna().any():
                joined.assign(_val=vals).plot(
                    column="_val", ax=ax, cmap="YlGn", legend=True,
                    edgecolor="white", linewidth=0.20,
                    legend_kwds={"label": "Removal efficiency (-)", "shrink": 0.72},
                )
            finish_map(ax, "CtG energy-supply removal efficiency of lowest-LCOD case", palette)
            fig.tight_layout()
            fig.savefig(fig_dir / "map_removal_efficiency_CtG_energy_supply_best_LCOD_case.png", dpi=280, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        (fig_dir / "SPATIAL_MAP_ERROR.txt").write_text(str(exc), encoding="utf-8")


# =============================================================================
# Main execution
# =============================================================================

def run(args) -> None:
    paths = resolve_paths(args)
    out_dir = ensure_dir(paths["out_dir"])
    for sub in ["inputs", "diagnostics", "best_cases", "figures", "maps"]:
        ensure_dir(out_dir / sub)

    print("=" * 100)
    print("05 LCOD NET EVALUATOR")
    print("=" * 100)
    for k, pth in paths.items():
        print(f"{k:22s}: {pth}")
    print(f"include_compression   : {args.include_compression}")
    print(f"heat_pump_capacity_basis: {args.heat_pump_capacity_basis}")
    print(f"allow_missing_energy_cost_as_zero: {args.allow_missing_energy_cost_as_zero}")
    print("=" * 100)

    if not paths["energy_ccs_csv"].exists():
        raise FileNotFoundError(f"Input from 04_CCS_EVALUATOR not found: {paths['energy_ccs_csv']}")

    print("[LOAD] Energy + CCS summary")
    energy_ccs_raw = read_csv_optional(paths["energy_ccs_csv"])
    energy_ccs = canonicalize_energy_ccs(energy_ccs_raw)

    print("[LOAD] Cost master and WACC")
    master = load_capex_master(paths["capex_opex"])
    recommended = read_csv_optional(paths["recommended"])
    wacc_table = load_wacc_table(paths["wacc"], args.cost_scenario)
    gap = read_csv_optional(paths["gap_confidence"])
    registry = read_csv_optional(paths["source_registry"])

    params = build_cost_parameters(master, args)

    if args.include_compression:
        print("[LOAD] Compression assumptions")
        comp, comp_df = load_compression_assumptions(paths["compression_dir"], paths["ccs_cost"], args)
    else:
        print("[SKIP] Compression excluded by default/current workflow")
        comp = {
            "compression_source_file": "compression_excluded_by_user",
            "compression_kWh_per_tCO2": 0.0,
            "compression_cost_USD_per_tCO2_direct": 0.0,
            "compression_emission_tCO2e_per_tCO2_direct": 0.0,
        }
        comp_df = pd.DataFrame()

    print("[CALC] LCOD and net removal")
    result = calculate_lcod(energy_ccs, wacc_table, params, comp, args)

    summary, cost_breakdown, net_breakdown, diagnostics = build_breakdown_tables(result)
    emission_breakdown = build_emission_breakdown_table(result, args)
    best_lcod, best_net, best_mo = build_best_tables(summary)
    write_best_scenario_diagnostics(summary, best_lcod, out_dir)
    write_heat_pump_capacity_diagnostics(result, out_dir)

    # Save input snapshots.
    energy_ccs_raw.to_csv(out_dir / "inputs" / "input_energy_ccs_summary_raw.csv", index=False, encoding="utf-8-sig")
    energy_ccs.to_csv(out_dir / "inputs" / "input_energy_ccs_summary_canonical.csv", index=False, encoding="utf-8-sig")
    if not master.empty:
        master.to_csv(out_dir / "inputs" / "capex_opex_assumptions_master_used.csv", index=False, encoding="utf-8-sig")
    if not recommended.empty:
        recommended.to_csv(out_dir / "inputs" / "recommended_tea_lcod_assumptions_used.csv", index=False, encoding="utf-8-sig")
    if not wacc_table.empty:
        wacc_table.to_csv(out_dir / "inputs" / "wacc_country_used.csv", index=False, encoding="utf-8-sig")
    if not gap.empty:
        gap.to_csv(out_dir / "inputs" / "finance_cost_data_gap_confidence_used.csv", index=False, encoding="utf-8-sig")
    if not registry.empty:
        registry.to_csv(out_dir / "inputs" / "wacc_capex_opex_source_registry_used.csv", index=False, encoding="utf-8-sig")
    if not comp_df.empty:
        comp_df.to_csv(out_dir / "inputs" / "aspen_plus_compression_input_used.csv", index=False, encoding="utf-8-sig")

    # Save main outputs.
    result.to_csv(out_dir / "lcod_net_full_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "annual_lcod_net_summary.csv", index=False, encoding="utf-8-sig")
    cost_breakdown.to_csv(out_dir / "cost_breakdown_by_province_policy_scenario.csv", index=False, encoding="utf-8-sig")
    net_breakdown.to_csv(out_dir / "net_removal_breakdown.csv", index=False, encoding="utf-8-sig")
    emission_breakdown.to_csv(out_dir / "emission_breakdown_by_boundary.csv", index=False, encoding="utf-8-sig")
    best_lcod.to_csv(out_dir / "best_cases" / "best_cases_by_lcod_net.csv", index=False, encoding="utf-8-sig")
    best_net.to_csv(out_dir / "best_cases" / "best_cases_by_net_removal.csv", index=False, encoding="utf-8-sig")
    best_mo.to_csv(out_dir / "best_cases" / "best_cases_by_multiobjective_screening.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / "diagnostics_lcod_net.csv", index=False, encoding="utf-8-sig")

    # Additional diagnostic breakdowns.
    if "lcod_invalid_reason" in result.columns:
        invalid_breakdown = result["lcod_invalid_reason"].fillna("").replace("", "valid").value_counts().reset_index()
        invalid_breakdown.columns = ["lcod_invalid_reason", "n_rows"]
        invalid_breakdown.to_csv(out_dir / "diagnostics" / "lcod_invalid_reason_breakdown.csv", index=False, encoding="utf-8-sig")
    if "energy_cost_missing_items" in result.columns:
        missing_energy = result.loc[result.get("energy_cost_missing_flag", False), ["country_code", "province_id", "province_name", "operation_policy", "energy_scenario", "energy_cost_missing_items"]].copy()
        missing_energy.to_csv(out_dir / "diagnostics" / "rows_with_missing_energy_cost.csv", index=False, encoding="utf-8-sig")

    # Save assumptions/config.
    assumption_rows = []
    for k, v in params.items():
        assumption_rows.append({"parameter": k, "value": v})
    for k, v in comp.items():
        assumption_rows.append({"parameter": k, "value": v})
    assumption_rows.append({"parameter": "heat_pump_capacity_basis", "value": args.heat_pump_capacity_basis})
    for k in [
        "co2_storage_loss_fraction",
        "pv_lifecycle_ef_tco2e_mwh",
        "wind_lifecycle_ef_tco2e_mwh",
        "geothermal_heat_lifecycle_ef_tco2e_mwhth",
        "battery_embodied_ef_tco2e_kwh",
        "sorbent_embodied_ef_tco2e_kg",
        "dac_infrastructure_ef_tco2e_tco2yr",
        "compression_infrastructure_ef_tco2e_tco2",
        "ccs_infrastructure_ef_tco2e_tco2",
        "include_full_system_sensitivity",
    ]:
        assumption_rows.append({"parameter": k, "value": getattr(args, k)})
    assumptions_df = pd.DataFrame(assumption_rows)
    assumptions_df.to_csv(out_dir / "cost_parameters_used.csv", index=False, encoding="utf-8-sig")

    print("[PLOT] LCOD visualizations and maps")
    plot_outputs(result, summary, best_lcod, out_dir, paths, args)

    config = {
        "scale_basis": "1000 kg adsorbent",
        "cost_scenario": args.cost_scenario,
        "dac_capacity_mode": args.dac_capacity_mode,
        "dac_nameplate_tco2yr": args.dac_nameplate_tco2yr,
        "adsorbent_mass_kg": args.adsorbent_mass_kg,
        "adsorbent_lifetime_years": args.adsorbent_lifetime_years,
        "project_lifetime_years_default": args.project_lifetime_years,
        "include_compression": args.include_compression,
        "allow_missing_energy_cost_as_zero": args.allow_missing_energy_cost_as_zero,
        "fallback_wacc_percent": args.fallback_wacc_percent,
        "fallback_electricity_price_USD_MWh": args.fallback_electricity_price_USD_MWh,
        "fallback_electricity_EF_tCO2_MWh": args.fallback_electricity_EF_tCO2_MWh,
        "heat_pump_capex_usd_per_kwth": args.heat_pump_capex_usd_per_kwth,
        "heat_pump_capacity_basis": args.heat_pump_capacity_basis,
        "heat_pump_capacity_reference_note": "Uses module 03 hourly-demand design capacity when available; fallback is p95/peak/annual-average heat demand depending on --heat-pump-capacity-basis.",
        "heat_pump_capex_reference_note": "Default 500 USD/kWth, selected as a screening central value within the 250-800 EUR/kWth installed large-scale heat-pump range reported in literature; NREL HTHP cases use 150-300 USD/kW as low/high cases for district-heating temperatures.",
        "geothermal_heat_capex_usd_per_kwth": args.geothermal_heat_capex_usd_per_kwth,
        "geothermal_heat_capex_note": "Kept at zero when geothermal heat is treated as purchased heat/LCOH costed in module 03.",
        "emission_accounting_method": "boundary_explicit_captured_CO2_basis_adapted_from_Yagihara_2026",
        "main_emission_boundary": "CtG_energy_supply_known",
        "co2_storage_loss_fraction": args.co2_storage_loss_fraction,
        "pv_lifecycle_ef_tco2e_mwh": args.pv_lifecycle_ef_tco2e_mwh,
        "wind_lifecycle_ef_tco2e_mwh": args.wind_lifecycle_ef_tco2e_mwh,
        "geothermal_heat_lifecycle_ef_tco2e_mwhth": args.geothermal_heat_lifecycle_ef_tco2e_mwhth,
        "battery_embodied_ef_tco2e_kwh": args.battery_embodied_ef_tco2e_kwh,
        "sorbent_embodied_ef_tco2e_kg": args.sorbent_embodied_ef_tco2e_kg,
        "dac_infrastructure_ef_tco2e_tco2yr": args.dac_infrastructure_ef_tco2e_tco2yr,
        "compression_infrastructure_ef_tco2e_tco2": args.compression_infrastructure_ef_tco2e_tco2,
        "ccs_infrastructure_ef_tco2e_tco2": args.ccs_infrastructure_ef_tco2e_tco2,
        "include_full_system_sensitivity": args.include_full_system_sensitivity,
        "paths": {k: str(v) for k, v in paths.items()},
        "compression_assumptions": comp,
        "notes": [
            "LCOD_gross_DAC_only uses annual captured CO2 as denominator.",
            "LCOD_net_DAC_only uses annual captured CO2 minus energy and compression emissions as denominator.",
            "LCOD_net_DACCS adds transport/storage cost and subtracts CCS transport/storage emissions.",
            "Energy cost/emissions are inherited from module 03/04 to avoid double counting.",
            "Heat-pump CAPEX uses hourly design capacity from module 03 when available, not annual-average heat/8760.",
            "Transport/storage cost/emissions are inherited from module 04 to avoid double counting.",
            "Compression is excluded by default in this version unless --include-compression is supplied.",
            "Missing energy cost is not converted to zero unless --allow-missing-energy-cost-as-zero is supplied.",
        ],
    }
    (out_dir / "config_05_LCOD_NET_EVALUATOR.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    readme = f"""05_LCOD_NET_EVALUATOR output

Input:
{paths['energy_ccs_csv']}

Output directory:
{out_dir}

Calculation basis:
- 1000 kg adsorbent basis.
- Three LCOD indicators are calculated:
  1. LCOD_gross_DAC_only_USD_tCO2
  2. LCOD_net_DAC_only_USD_tCO2
  3. LCOD_net_DACCS_USD_tCO2

Main equations:
CRF = WACC(1+WACC)^N / ((1+WACC)^N - 1)
Annualized CAPEX = CAPEX * CRF
Annual net CO2 removed = annual CO2 captured - annual emissions
LCOD = annual total cost / annual CO2 basis

Key assumptions:
- WACC is country-level and selected using scenario='{args.cost_scenario}'.
- DAC CAPEX mode: {args.dac_capacity_mode}
- Adsorbent replacement: {args.adsorbent_mass_kg} kg adsorbent replaced every {args.adsorbent_lifetime_years} years.
- Compression included: {args.include_compression}. If False, LCOD excludes CO2 compression cost and emissions.
- Missing energy cost set to zero: {args.allow_missing_energy_cost_as_zero}. If False, rows with missing energy cost are invalid for LCOD.
- Energy cost/emissions are inherited from module 03/04.
- T&S cost/emissions are inherited from module 04.
- Heat pump CAPEX: {args.heat_pump_capex_usd_per_kwth} USD/kWth.
- Heat pump capacity basis: {args.heat_pump_capacity_basis}. If module 03 hourly design capacity is available, 05 uses it instead of annual heat/8760.
- Geothermal heat CAPEX: {args.geothermal_heat_capex_usd_per_kwth} USD/kWth. Keep at 0 when geothermal heat is treated as purchased heat/LCOH from module 03.

Main output files:
- lcod_net_full_results.csv
- annual_lcod_net_summary.csv
- cost_breakdown_by_province_policy_scenario.csv
- net_removal_breakdown.csv
- best_cases/best_cases_by_lcod_net.csv
- best_cases/best_cases_by_net_removal.csv
- best_cases/best_cases_by_multiobjective_screening.csv
- diagnostics_lcod_net.csv
- diagnostics/best_lcod_scenario_count.csv
- diagnostics/best_lcod_by_country_scenario_count.csv
- diagnostics/median_lcod_by_energy_scenario_base.csv
- diagnostics/heat_pump_capacity_diagnostics.csv
- cost_parameters_used.csv
- config_05_LCOD_NET_EVALUATOR.json

Figures:
- figures/lcod_net_DACCS_boxplot_by_energy_scenario.png
- figures/cost_breakdown_top10_lowest_lcod.png
- figures/map_best_LCOD_net_DACCS_by_province.png
- figures/map_net_CO2_removed_for_best_LCOD_case.png
- figures/map_best_energy_scenario_by_LCOD.png
"""
    (out_dir / "README_05_LCOD_NET_EVALUATOR.txt").write_text(readme, encoding="utf-8")

    print("=" * 100)
    print("05 LCOD NET EVALUATOR COMPLETE")
    print("=" * 100)
    print(f"Rows processed      : {len(result):,}")
    print(f"Valid LCOD rows     : {int(result['valid_lcod_flag'].sum()):,}")
    print(f"Rows missing energy cost: {int(result['energy_cost_missing_flag'].sum()):,}")
    print(f"Compression included: {args.include_compression}")
    print(f"Output directory    : {out_dir}")
    print("Main outputs:")
    print(f"- {out_dir / 'annual_lcod_net_summary.csv'}")
    print(f"- {out_dir / 'lcod_net_full_results.csv'}")
    print(f"- {out_dir / 'diagnostics_lcod_net.csv'}")
    print("=" * 100)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Calculate LCOD and net CO2 removal from energy + CCS outputs.")

    p.add_argument("--root-dir", default=r"D:/Ashka/5.DAC", help="Project root folder.")
    p.add_argument("--tea-dir", default=None, help="Path to 02.TEA_LCOD. Default: root/02.TEA_LCOD.")
    p.add_argument("--cost-dir", default=None, help="Path to 00.RESOURCES/01_COST. Default: root/00.RESOURCES/01_COST.")
    p.add_argument("--compression-dir", default=None, help="Path to Aspen Plus compression folder. Default: root/01.DAC_SYSTEM/04_ASPENPLUS_COMPRESSION.")
    p.add_argument("--compression-csv", default=None, help="Optional explicit Aspen Plus compression CSV path.")
    p.add_argument("--energy-ccs-csv", default=None, help="Optional explicit input CSV from 04_CCS_EVALUATOR.")
    p.add_argument("--out-dir", default=None, help="Output dir. Default: tea_dir/05_LCOD_NET_EVALUATOR.")
    p.add_argument("--spatial-dir", default=None, help="Path to 00.SPATIAL_MAP. Default: root/00.SPATIAL_MAP.")
    p.add_argument("--spatial-gpkg", default=None, help="Explicit ASEAN province GPKG path. Default: root/00.SPATIAL_MAP/ASEAN/ASEAN_PROVINCES_LEVEL1.gpkg.")
    p.add_argument("--disable-spatial-maps", action="store_true", help="Disable GeoPackage map outputs.")

    p.add_argument("--cost-scenario", default="central", choices=["low", "central", "high"], help="Cost/WACC scenario.")
    p.add_argument("--project-lifetime-years", type=float, default=30.0)
    p.add_argument("--fallback-wacc-percent", type=float, default=10.0)

    p.add_argument("--adsorbent-mass-kg", type=float, default=1000.0)
    p.add_argument("--adsorbent-lifetime-years", type=float, default=2.0)

    p.add_argument("--dac-capacity-mode", default="actual", choices=["actual", "fixed"], help="actual: DAC CAPEX scales with each row annual CO2; fixed: use --dac-nameplate-tco2yr.")
    p.add_argument("--dac-nameplate-tco2yr", type=float, default=150.0, help="Used only when --dac-capacity-mode fixed.")

    # Fallback costs if cost files are missing or incomplete.
    p.add_argument("--fallback-dac-capex-usd-per-tco2yr", type=float, default=4523.0)
    p.add_argument("--fallback-dac-fixed-opex-usd-per-tco2", type=float, default=153.0)
    p.add_argument("--fallback-sorbent-cost-usd-per-lb", type=float, default=0.09)
    p.add_argument("--fallback-pv-capex-usd-per-kw", type=float, default=691.0)
    p.add_argument("--fallback-wind-capex-usd-per-kw", type=float, default=1041.0)
    p.add_argument("--fallback-battery-capex-usd-per-kwh", type=float, default=192.0)

    p.add_argument("--pv-lifetime-years", type=float, default=25.0)
    p.add_argument("--wind-lifetime-years", type=float, default=25.0)
    p.add_argument("--battery-lifetime-years", type=float, default=15.0)
    p.add_argument("--heat-pump-capex-usd-per-kwth", type=float, default=500.0, help="Heat-pump CAPEX in USD/kWth. Default 500 USD/kWth, screening central value based on large-scale/industrial heat-pump literature; override for sensitivity.")
    p.add_argument("--heat-pump-capacity-basis", default="from_module03", choices=["from_module03", "p95", "peak", "average"], help="Heat-pump capacity basis for CAPEX. Default uses module 03 design capacity, with fallback to p95/peak/annual-average heat demand.")
    p.add_argument("--heat-pump-lifetime-years", type=float, default=20.0)
    p.add_argument("--geothermal-heat-capex-usd-per-kwth", type=float, default=0.0, help="Default 0 because direct geothermal heat system CAPEX is not yet sourced.")
    p.add_argument("--geothermal-heat-lifetime-years", type=float, default=30.0)

    # Current workflow: compression is intentionally skipped unless user opts in.
    p.add_argument("--include-compression", action="store_true", help="Include CO2 compression cost and emissions. Default is skipped for current thesis workflow.")
    p.add_argument("--compression-kwh-per-tco2", type=float, default=np.nan, help="Manual fallback if compression is included and Aspen Plus compression CSV is unavailable.")

    # Missing-cost handling.
    p.add_argument("--allow-missing-energy-cost-as-zero", action="store_true", help="Temporary lower-bound screening only. If omitted, missing energy cost makes LCOD invalid.")
    p.add_argument("--fallback-electricity-price-USD-MWh", type=float, default=np.nan, help="Used only if module 03 effective energy price is missing and compression is included.")
    p.add_argument("--fallback-electricity-EF-tCO2-MWh", type=float, default=np.nan, help="Used only if module 03 effective energy EF is missing and compression is included.")

    # Boundary-explicit emission accounting. Defaults are NaN to avoid inventing
    # lifecycle/infrastructure factors. Active components with missing factors are
    # excluded from known CtG subtotals and flagged as missing/needs-source.
    p.add_argument("--co2-storage-loss-fraction", type=float, default=0.0, help="CO2 storage/transport loss fraction. Default 0 = no loss assumed.")
    p.add_argument("--pv-lifecycle-ef-tco2e-mwh", type=float, default=np.nan, help="Optional PV lifecycle emission factor in tCO2e/MWh used. Missing values are not invented.")
    p.add_argument("--wind-lifecycle-ef-tco2e-mwh", type=float, default=np.nan, help="Optional wind lifecycle emission factor in tCO2e/MWh used. Missing values are not invented.")
    p.add_argument("--geothermal-heat-lifecycle-ef-tco2e-mwhth", type=float, default=np.nan, help="Optional geothermal heat lifecycle emission factor in tCO2e/MWhth. Missing values are not invented.")
    p.add_argument("--battery-embodied-ef-tco2e-kwh", type=float, default=np.nan, help="Optional battery embodied emission factor in tCO2e/kWh capacity, annualized by battery lifetime.")
    p.add_argument("--sorbent-embodied-ef-tco2e-kg", type=float, default=np.nan, help="Optional sorbent embodied emission factor in tCO2e/kg sorbent for full-system sensitivity.")
    p.add_argument("--dac-infrastructure-ef-tco2e-tco2yr", type=float, default=np.nan, help="Optional DAC infrastructure embodied EF in tCO2e/(tCO2/yr capacity), annualized by project lifetime.")
    p.add_argument("--compression-infrastructure-ef-tco2e-tco2", type=float, default=np.nan, help="Optional compression infrastructure EF in tCO2e/tCO2 captured for full-system sensitivity.")
    p.add_argument("--ccs-infrastructure-ef-tco2e-tco2", type=float, default=np.nan, help="Optional CCS infrastructure EF in tCO2e/tCO2 stored for full-system sensitivity.")
    p.add_argument("--include-full-system-sensitivity", action="store_true", help="Flag full-system embodied/sorbent boundary as requested. Missing full-system factors are then marked as missing/needs-source.")

    # Optional free-text source labels for emission factors. Use these when supplying
    # factor values via CLI so the output registry remains auditable.
    p.add_argument("--pv-lifecycle-ef-tco2e-mwh-source", default="missing_or_user_arg", help="Source label for --pv-lifecycle-ef-tco2e-mwh.")
    p.add_argument("--wind-lifecycle-ef-tco2e-mwh-source", default="missing_or_user_arg", help="Source label for --wind-lifecycle-ef-tco2e-mwh.")
    p.add_argument("--geothermal-heat-lifecycle-ef-tco2e-mwhth-source", default="missing_or_user_arg", help="Source label for --geothermal-heat-lifecycle-ef-tco2e-mwhth.")
    p.add_argument("--battery-embodied-ef-tco2e-kwh-source", default="missing_or_user_arg", help="Source label for --battery-embodied-ef-tco2e-kwh.")
    p.add_argument("--sorbent-embodied-ef-tco2e-kg-source", default="missing_or_user_arg", help="Source label for --sorbent-embodied-ef-tco2e-kg.")
    p.add_argument("--dac-infrastructure-ef-tco2e-tco2yr-source", default="missing_or_user_arg", help="Source label for --dac-infrastructure-ef-tco2e-tco2yr.")
    p.add_argument("--compression-infrastructure-ef-tco2e-tco2-source", default="missing_or_user_arg", help="Source label for --compression-infrastructure-ef-tco2e-tco2.")
    p.add_argument("--ccs-infrastructure-ef-tco2e-tco2-source", default="missing_or_user_arg", help="Source label for --ccs-infrastructure-ef-tco2e-tco2.")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
