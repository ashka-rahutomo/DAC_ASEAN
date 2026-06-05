from __future__ import annotations

"""
04_ccs_evaluator.py

CCS / CO2 transport-storage evaluator for ASEAN DACCS post-processing.

Purpose
-------
This script adds a screening-level CCS transport and storage layer after:

    03_ENERGY_SUPPLY_EVALUATOR

The CCS method is intentionally simple and transparent for thesis-stage analysis:

1. DAC location is represented by each province representative point.
2. Storage location is represented by storage-site coordinates or basin/project centroid.
3. Domestic usable storage is prioritized.
4. Cross-border ASEAN storage is allowed only if a province's country has no usable domestic
   storage site coordinates.
5. Island-aware transport rule is used:
   - same island between DAC representative point and storage node -> pipeline
   - different island -> compare direct pipeline and ship/hybrid route, then choose the lower-cost option when both costs are available.
6. Hybrid shipping can use port data:
   DAC representative point -> nearest origin commercial port -> nearest destination commercial port -> storage node.
   If port data are unavailable, a proxy straight-line ship distance is retained and explicitly flagged.
7. Domestic usable storage is prioritized. Cross-border storage is allowed only if the country has no usable domestic storage coordinate.

Default output folder:
    D:/Ashka/5.DAC/02.TEA_LCOD/04_CCS_EVALUATOR/

Default inputs:
    D:/Ashka/5.DAC/00.RESOURCES/03_CCS/ccs_storage_sites_asean.csv
    D:/Ashka/5.DAC/00.RESOURCES/03_CCS/ccs_cost_emission_assumptions.csv
    D:/Ashka/5.DAC/00.RESOURCES/03_CCS/ccs_transport_assets_asean.csv
    D:/Ashka/5.DAC/00.SPATIAL_MAP/**/ASEAN_PROVINCES_LEVEL1.gpkg
    D:/Ashka/5.DAC/02.TEA_LCOD/03_ENERGY_SUPPLY_EVALUATOR/annual_results/energy_supply_summary_by_province_policy_scenario.csv

Notes
-----
- The straight line is not an engineered pipeline or shipping route.
- If storage-site latitude/longitude are missing, that site is not usable for distance calculation.
- Ship mode without port data is marked as proxy_no_port.
- This script does not compute LCOD; it prepares CCS cost and emission inputs for the next LCOD/net-removal module.
"""

from pathlib import Path
import argparse
import json
import math
import warnings
import re

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception as exc:  # pragma: no cover
    gpd = None
    LineString = None
    _GPD_IMPORT_ERROR = exc
else:
    _GPD_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_IMPORT_ERROR = exc
else:
    _MPL_IMPORT_ERROR = None


# =============================================================================
# Canonical / candidate column helpers
# =============================================================================

PROVINCE_ID_COLS = ["province_id", "prov_id", "adm1_id", "id", "GID_1", "gid_1"]
PROVINCE_NAME_COLS = ["province_name", "prov_name", "name_1", "NAME_1", "adm1_name", "ADM1_NAME"]
COUNTRY_CODE_COLS = ["country_code", "iso3", "ISO3", "GID_0", "country_iso3"]
COUNTRY_NAME_COLS = ["country_name", "name_0", "NAME_0", "country", "COUNTRY"]

ANNUAL_CO2_COLS = [
    "annual_CO2_t_per_1000kgads",
    "annual_CO2_t",
    "annual_gross_CO2_t",
    "annual_CO2_t_per_bed",
    "annual_tCO2_per_1000kgads",
    "annual_tCO2_per_bed",
]

ENERGY_SUMMARY_FILENAME = "energy_supply_summary_by_province_policy_scenario.csv"

PORT_FILENAME_DEFAULT = "ccs_ports_asean.csv"
PORT_COLS = [
    "port_id", "port_name", "country_code", "country_name", "province_id", "province_name",
    "island_name", "latitude", "longitude", "port_type", "port_status", "source_id", "notes",
]

# New columns written by module 03 after the hourly-demand revision.
# Module 04 should not reinterpret them; it should preserve them for module 05.
ENERGY_PROFILE_PASSTHROUGH_COLS = [
    "demand_profile_method",
    "hourly_profile_used_flag",
    "hourly_profile_n_hours",
    "peak_process_electricity_kW",
    "p95_process_electricity_kW",
    "peak_heat_demand_kWth",
    "p95_heat_demand_kWth",
    "heat_pump_capacity_kWth_design",
    "heat_pump_capacity_design_basis",
    "peak_total_electricity_demand_kW",
    "p95_total_electricity_demand_kW",
    "battery_power_kW_design",
    "battery_power_basis",
]


# Primary boundary file produced by check_asean_map.py and reused by the NASA POWER/DAC pipeline.
# This is the map source that should be used by modules 03 and 04 unless the user passes --spatial-file.
SPATIAL_GPKG_DEFAULT = Path(r"D:/Ashka/5.DAC/00.SPATIAL_MAP/ASEAN/ASEAN_PROVINCES_LEVEL1.gpkg")


# =============================================================================
# Generic utilities
# =============================================================================

def find_existing_file(base_dir: Path, preferred_name: str, patterns: list[str] | None = None) -> Path | None:
    """Find the best matching input file.

    The workflow often keeps both original and *_UPDATED / *_EXPANDED_UPDATED files in the
    same resource folder. To avoid accidentally reading an old starter file, this helper
    ranks UPDATED/EXPANDED files above old exact-name files, while still allowing users to
    force a specific file by passing an explicit filename that exists and is the only best match.
    """
    candidates: list[Path] = []
    exact = base_dir / preferred_name
    if exact.exists():
        candidates.append(exact)
    if patterns:
        for pat in patterns:
            candidates.extend(sorted(base_dir.glob(pat)))

    # Unique existing CSV-like files only.
    uniq = []
    seen = set()
    for c in candidates:
        if c.exists() and c.is_file() and c not in seen:
            uniq.append(c)
            seen.add(c)
    if not uniq:
        return None

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        s = 0
        if path.name == preferred_name:
            s += 10
        if "updated" in name:
            s += 40
        if "expanded" in name:
            s += 20
        if "final" in name:
            s += 15
        if "starter" in name:
            s -= 20
        if "template" in name:
            s -= 100
        if "readme" in name:
            s -= 100
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (s, mtime)

    return sorted(uniq, key=score, reverse=True)[0]


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True, label: str = "column") -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Could not find {label}. Tried: {candidates}. Available columns: {cols}")
    return None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read resource CSVs robustly, including files with broken quote characters.

    Several resource CSVs are edited manually and contain long source/notes fields with
    unescaped quotes, commas, semicolons, and trailing separators. Standard pandas CSV
    parsing can then fail or shift columns. This function tries multiple readers and
    selects the candidate with the most useful headers. The preferred robust path disables
    quote handling so that commas remain the only column separator.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    attempts = [
        dict(sep=None, engine="python", encoding="utf-8-sig"),
        dict(sep=",", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        dict(sep=",", encoding="utf-8-sig", index_col=False),
        dict(sep=";", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        dict(sep="\t", engine="python", encoding="utf-8-sig", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        dict(sep=",", engine="python", encoding="latin1", quotechar="\x07", index_col=False, on_bad_lines="warn"),
        dict(sep=";", engine="python", encoding="latin1", quotechar="\x07", index_col=False, on_bad_lines="warn"),
    ]

    candidates = []
    for opts in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(path, **opts)
            if df is None or (df.empty and len(df.columns) == 0):
                continue
            df.columns = [
                re.sub(r";+$", "", str(c).lstrip("\ufeff").strip().strip('"').strip("'").strip())
                for c in df.columns
            ]
            candidates.append((opts, df))
        except Exception:
            continue

    if not candidates:
        # Last-resort mode: keep whatever rows can be read rather than stopping the pipeline.
        df = pd.read_csv(
            path,
            sep=",",
            engine="python",
            encoding="utf-8-sig",
            quotechar="\x07",
            index_col=False,
            on_bad_lines="skip",
        )
        df.columns = [
            re.sub(r";+$", "", str(c).lstrip("\ufeff").strip().strip('"').strip("'").strip())
            for c in df.columns
        ]
        return df

    expected_tokens = [
        "storage_site_id", "storage_site_name", "latitude", "longitude", "offshore_or_onshore",
        "island_name", "component", "central_value", "cost_unit", "emission_factor",
        "port_id", "port_name", "country_code", "province_name", "source_url",
    ]

    def score(df: pd.DataFrame) -> float:
        cols = [str(c).strip() for c in df.columns]
        lower = [c.lower() for c in cols]
        token_score = sum(any(tok in c for c in lower) for tok in expected_tokens)
        unnamed_penalty = sum(c.lower().startswith("unnamed") for c in cols)
        one_col_penalty = 100 if len(cols) <= 2 else 0
        row_score = min(len(df), 200) / 10
        return token_score * 1000 + len(cols) * 10 + row_score - unnamed_penalty * 25 - one_col_penalty

    best = max(candidates, key=lambda item: score(item[1]))[1].copy()
    best.columns = [
        re.sub(r";+$", "", str(c).lstrip("\ufeff").strip().strip('"').strip("'").strip())
        for c in best.columns
    ]
    return best


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        val = float(x)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def truthy_text(value) -> bool:
    s = str(value).strip().lower()
    return s in {"yes", "true", "1", "y", "valid", "available", "operating", "planned"}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two WGS84 points in km."""
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def clean_country_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def clean_name(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_island(x) -> str:
    """Normalize island / coastal-region labels for same-island routing.

    The CSV may describe offshore storage as "Java Sea", "offshore Java", "Gulf of
    Thailand", "Sarawak", etc. For screening, those are mapped to the nearest
    island/mainland transport region so that the same-island rule is usable.
    """
    if pd.isna(x):
        return ""
    raw = str(x).strip().lower()
    if not raw:
        return ""
    s = re.sub(r"[^a-z0-9]+", "_", raw)
    s = re.sub(r"_+", "_", s).strip("_")

    alias_rules = [
        (["java_sea", "northwest_java", "west_java_sea", "sunda_asri", "offshore_java", "java"], "java"),
        (["sumatra", "central_sumatra", "south_sumatra", "north_sumatra", "arun", "aceh"], "sumatra"),
        (["natuna", "east_natuna", "west_natuna", "riau_islands", "riau_archipelago"], "natuna"),
        (["kalimantan", "borneo", "east_borneo", "kutai", "tarakan", "barito", "sarawak", "sabah", "brunei"], "borneo"),
        (["papua", "bintuni", "tangguh"], "papua"),
        (["maluku", "arafura", "masela", "abadi"], "maluku"),
        (["sulawesi", "banggai"], "sulawesi"),
        (["peninsular_malaysia", "malay_peninsula", "west_malaysia", "penyu", "malay_basin"], "peninsular_malaysia"),
        (["thailand_mainland", "mainland_thailand", "gulf_of_thailand", "pattani", "arthit", "mae_moh"], "thailand_mainland"),
        (["vietnam_mainland", "vietnam", "cuu_long", "nam_con_son", "bach_ho"], "vietnam_mainland"),
        (["palawan", "north_palawan", "malampaya"], "palawan"),
        (["luzon", "manila", "batangas"], "luzon"),
        (["visayas", "leyte", "cebu", "negros"], "visayas"),
        (["mindanao", "davao"], "mindanao"),
        (["singapore"], "singapore"),
        (["cambodia_mainland", "cambodia"], "cambodia_mainland"),
        (["myanmar_mainland", "myanmar"], "myanmar_mainland"),
    ]
    for keys, canonical in alias_rules:
        if any(k in s for k in keys):
            return canonical
    return s


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Cost and emission assumptions
# =============================================================================

def extract_cost_assumptions(
    cost_df: pd.DataFrame | None,
    eur_to_usd: float,
    offshore_pipeline_multiplier: float,
    ship_cost_usd_tco2_km: float | None,
    ship_terminal_cost_usd_tco2: float | None,
    ship_emission_tco2e_tco2_km: float | None,
    ship_terminal_emission_tco2e_tco2: float,
) -> tuple[dict, pd.DataFrame]:
    """Extract CCS cost assumptions from CSV where available, with explicit CLI fallback for ship."""
    assumptions = {
        "eur_to_usd": eur_to_usd,
        "offshore_pipeline_multiplier": offshore_pipeline_multiplier,
        "pipeline_cost_usd_tco2_km": np.nan,
        "pipeline_cost_original_value": np.nan,
        "pipeline_cost_original_currency": None,
        "pipeline_cost_source_id": None,
        "pipeline_emission_tco2e_tco2_km": np.nan,
        "pipeline_emission_original_value": np.nan,
        "storage_cost_usd_tco2": np.nan,
        "storage_cost_original_value": np.nan,
        "storage_cost_original_currency": None,
        "storage_cost_source_id": None,
        "mrv_cost_usd_tco2": np.nan,
        "mrv_cost_original_value": np.nan,
        "mrv_cost_original_currency": None,
        "mrv_cost_source_id": None,
        "ship_cost_usd_tco2_km": np.nan,
        "ship_terminal_cost_usd_tco2": np.nan,
        "ship_emission_tco2e_tco2_km": np.nan,
        "ship_terminal_emission_tco2e_tco2": ship_terminal_emission_tco2e_tco2,
        "ship_assumption_quality": "missing_ship_cost_assumption",
    }

    records = []

    def add_record(parameter, value, unit, source, notes):
        records.append({
            "parameter": parameter,
            "value": value,
            "unit": unit,
            "source": source,
            "notes": notes,
        })

    if cost_df is not None and not cost_df.empty:
        df = cost_df.copy()
        for col in ["component", "assumption_id", "currency", "source_id", "unit_final", "cost_unit", "emission_factor_unit"]:
            if col not in df.columns:
                df[col] = np.nan

        # Pipeline cost and emission
        pipe_rows = df[df["component"].astype(str).str.lower().str.contains("pipeline_transport", na=False)].copy()
        if not pipe_rows.empty:
            row = pipe_rows.iloc[0]
            val = safe_float(row.get("central_value"), safe_float(row.get("cost_value")))
            cur = str(row.get("currency") or "").strip().upper()
            if np.isfinite(val):
                assumptions["pipeline_cost_original_value"] = val
                assumptions["pipeline_cost_original_currency"] = cur
                assumptions["pipeline_cost_source_id"] = row.get("source_id")
                if cur == "EUR":
                    assumptions["pipeline_cost_usd_tco2_km"] = val * eur_to_usd
                elif cur == "USD":
                    assumptions["pipeline_cost_usd_tco2_km"] = val
                else:
                    # If currency is missing but unit_final mentions EUR, assume EUR but keep note.
                    unit_text = f"{row.get('unit_final')} {row.get('cost_unit')}".lower()
                    if "eur" in unit_text:
                        assumptions["pipeline_cost_usd_tco2_km"] = val * eur_to_usd
                        assumptions["pipeline_cost_original_currency"] = "EUR_inferred_from_unit"
                    elif "usd" in unit_text:
                        assumptions["pipeline_cost_usd_tco2_km"] = val
                        assumptions["pipeline_cost_original_currency"] = "USD_inferred_from_unit"

            ef = safe_float(row.get("emission_factor_value"))
            ef_unit = str(row.get("emission_factor_unit") or "").lower()
            if np.isfinite(ef):
                assumptions["pipeline_emission_original_value"] = ef
                if "gco2" in ef_unit or "gco2e" in ef_unit:
                    assumptions["pipeline_emission_tco2e_tco2_km"] = ef / 1_000_000.0
                elif "kgco2" in ef_unit or "kgco2e" in ef_unit:
                    assumptions["pipeline_emission_tco2e_tco2_km"] = ef / 1000.0
                else:
                    assumptions["pipeline_emission_tco2e_tco2_km"] = ef

        # Storage cost
        storage_rows = df[df["component"].astype(str).str.lower().eq("storage")].copy()
        if storage_rows.empty:
            storage_rows = df[df["component"].astype(str).str.lower().str.contains("storage", na=False)].copy()
        if not storage_rows.empty:
            # Prefer Terlouw median if present, otherwise first storage row.
            mask_terlouw = storage_rows["assumption_id"].astype(str).str.lower().str.contains("terlouw", na=False)
            row = storage_rows[mask_terlouw].iloc[0] if mask_terlouw.any() else storage_rows.iloc[0]
            val = safe_float(row.get("central_value"), safe_float(row.get("cost_value")))
            cur = str(row.get("currency") or "").strip().upper()
            if np.isfinite(val):
                assumptions["storage_cost_original_value"] = val
                assumptions["storage_cost_original_currency"] = cur
                assumptions["storage_cost_source_id"] = row.get("source_id")
                if cur == "EUR":
                    assumptions["storage_cost_usd_tco2"] = val * eur_to_usd
                elif cur == "USD":
                    assumptions["storage_cost_usd_tco2"] = val
                else:
                    unit_text = f"{row.get('unit_final')} {row.get('cost_unit')}".lower()
                    if "eur" in unit_text:
                        assumptions["storage_cost_usd_tco2"] = val * eur_to_usd
                        assumptions["storage_cost_original_currency"] = "EUR_inferred_from_unit"
                    elif "usd" in unit_text:
                        assumptions["storage_cost_usd_tco2"] = val
                        assumptions["storage_cost_original_currency"] = "USD_inferred_from_unit"

        # MRV cost: if central missing, use midpoint of low/high if available.
        mrv_rows = df[df["component"].astype(str).str.lower().str.contains("mrv|monitor", regex=True, na=False)].copy()
        if not mrv_rows.empty:
            row = mrv_rows.iloc[0]
            val = safe_float(row.get("central_value"), np.nan)
            if not np.isfinite(val):
                low = safe_float(row.get("low_value"))
                high = safe_float(row.get("high_value"))
                if np.isfinite(low) and np.isfinite(high):
                    val = 0.5 * (low + high)
            cur = str(row.get("currency") or "").strip().upper()
            if np.isfinite(val):
                assumptions["mrv_cost_original_value"] = val
                assumptions["mrv_cost_original_currency"] = cur
                assumptions["mrv_cost_source_id"] = row.get("source_id")
                if cur == "EUR":
                    assumptions["mrv_cost_usd_tco2"] = val * eur_to_usd
                elif cur == "USD":
                    assumptions["mrv_cost_usd_tco2"] = val
                else:
                    unit_text = f"{row.get('unit_final')} {row.get('cost_unit')}".lower()
                    if "eur" in unit_text:
                        assumptions["mrv_cost_usd_tco2"] = val * eur_to_usd
                        assumptions["mrv_cost_original_currency"] = "EUR_inferred_from_unit"
                    elif "usd" in unit_text:
                        assumptions["mrv_cost_usd_tco2"] = val
                        assumptions["mrv_cost_original_currency"] = "USD_inferred_from_unit"

        # Ship transport and terminal cost from file if present.
        # These are separated intentionally because ship/hybrid comparison needs:
        #   sea-leg cost per tCO2-km + terminal/loading-unloading cost per tCO2.
        if "technology_or_route_type" not in df.columns:
            df["technology_or_route_type"] = ""

        comp_lower = df["component"].astype(str).str.lower()
        route_lower = df["technology_or_route_type"].astype(str).str.lower()

        ship_transport_rows = df[
            (
                comp_lower.str.contains("ship_transport|shipping_transport|marine_transport", regex=True, na=False)
                | route_lower.str.contains("ship_hybrid|shipping|marine", regex=True, na=False)
            )
            & ~comp_lower.str.contains("terminal|loading|unloading", regex=True, na=False)
        ].copy()

        ship_terminal_rows = df[
            comp_lower.str.contains("ship_terminal|terminal|loading|unloading", regex=True, na=False)
            | route_lower.str.contains("terminal|loading|unloading", regex=True, na=False)
        ].copy()

        def convert_cost_value(row, per: str) -> tuple[float, str]:
            val = safe_float(row.get("central_value"), safe_float(row.get("cost_value")))
            cur = str(row.get("currency") or "").strip().upper()
            if not np.isfinite(val):
                return np.nan, cur
            unit_text = f"{row.get('unit_final')} {row.get('cost_unit')}".lower()
            if cur == "EUR" or ("eur" in unit_text and cur not in {"USD"}):
                return val * eur_to_usd, cur or "EUR_inferred_from_unit"
            if cur == "USD" or ("usd" in unit_text and cur not in {"EUR"}):
                return val, cur or "USD_inferred_from_unit"
            # Keep as NaN if currency cannot be interpreted; avoid silently using an ambiguous value.
            return np.nan, cur or "currency_missing"

        if not ship_transport_rows.empty:
            # Prefer explicit ship_transport component.
            explicit = ship_transport_rows[ship_transport_rows["component"].astype(str).str.lower().str.contains("ship_transport|shipping_transport", regex=True, na=False)]
            row = explicit.iloc[0] if not explicit.empty else ship_transport_rows.iloc[0]
            val_usd, cur_used = convert_cost_value(row, per="km")
            if np.isfinite(val_usd):
                assumptions["ship_cost_usd_tco2_km"] = val_usd
                assumptions["ship_assumption_quality"] = f"ship_transport_from_cost_csv:{row.get('source_id')}"
            ef = safe_float(row.get("emission_factor_value"))
            ef_unit = str(row.get("emission_factor_unit") or "").lower()
            if np.isfinite(ef):
                if "gco2" in ef_unit or "gco2e" in ef_unit:
                    assumptions["ship_emission_tco2e_tco2_km"] = ef / 1_000_000.0
                elif "kgco2" in ef_unit or "kgco2e" in ef_unit:
                    assumptions["ship_emission_tco2e_tco2_km"] = ef / 1000.0
                else:
                    assumptions["ship_emission_tco2e_tco2_km"] = ef

        if not ship_terminal_rows.empty:
            explicit = ship_terminal_rows[ship_terminal_rows["component"].astype(str).str.lower().str.contains("ship_terminal|terminal", regex=True, na=False)]
            row = explicit.iloc[0] if not explicit.empty else ship_terminal_rows.iloc[0]
            val_usd, cur_used = convert_cost_value(row, per="tonne")
            if np.isfinite(val_usd):
                assumptions["ship_terminal_cost_usd_tco2"] = val_usd
                if assumptions["ship_assumption_quality"] == "missing_ship_cost_assumption":
                    assumptions["ship_assumption_quality"] = f"ship_terminal_from_cost_csv:{row.get('source_id')}"
                else:
                    assumptions["ship_assumption_quality"] += f"+ship_terminal_from_cost_csv:{row.get('source_id')}"
            ef = safe_float(row.get("emission_factor_value"))
            ef_unit = str(row.get("emission_factor_unit") or "").lower()
            if np.isfinite(ef):
                if "gco2" in ef_unit or "gco2e" in ef_unit:
                    assumptions["ship_terminal_emission_tco2e_tco2"] = ef / 1_000_000.0
                elif "kgco2" in ef_unit or "kgco2e" in ef_unit:
                    assumptions["ship_terminal_emission_tco2e_tco2"] = ef / 1000.0
                else:
                    assumptions["ship_terminal_emission_tco2e_tco2"] = ef

    # CLI overrides for ship.
    if ship_cost_usd_tco2_km is not None and np.isfinite(ship_cost_usd_tco2_km):
        assumptions["ship_cost_usd_tco2_km"] = float(ship_cost_usd_tco2_km)
        assumptions["ship_assumption_quality"] = "cli_user_supplied_or_placeholder"
    if ship_terminal_cost_usd_tco2 is not None and np.isfinite(ship_terminal_cost_usd_tco2):
        assumptions["ship_terminal_cost_usd_tco2"] = float(ship_terminal_cost_usd_tco2)
        if assumptions["ship_assumption_quality"] == "missing_ship_cost_assumption":
            assumptions["ship_assumption_quality"] = "partial_cli_terminal_only"
    if ship_emission_tco2e_tco2_km is not None and np.isfinite(ship_emission_tco2e_tco2_km):
        assumptions["ship_emission_tco2e_tco2_km"] = float(ship_emission_tco2e_tco2_km)

    # Missing defaults: do not invent costs; leave NaN.
    if not np.isfinite(assumptions["ship_terminal_cost_usd_tco2"]):
        assumptions["ship_terminal_cost_usd_tco2"] = np.nan

    if np.isfinite(assumptions["ship_cost_usd_tco2_km"]) and np.isfinite(assumptions["ship_terminal_cost_usd_tco2"]):
        if assumptions["ship_assumption_quality"] == "missing_ship_cost_assumption":
            assumptions["ship_assumption_quality"] = "complete_ship_cost_from_cli_or_csv"
        elif "complete" not in str(assumptions["ship_assumption_quality"]).lower():
            assumptions["ship_assumption_quality"] = "complete:" + str(assumptions["ship_assumption_quality"])

    # If pipeline emission missing, keep NaN; no invented fallback.
    add_record("eur_to_usd", eur_to_usd, "USD/EUR", "CLI", "Used only for cost rows reported in EUR.")
    add_record("offshore_pipeline_multiplier", offshore_pipeline_multiplier, "dimensionless", "CLI", "Applied to pipeline effective distance for offshore storage sites.")
    add_record("pipeline_cost_usd_tco2_km", assumptions["pipeline_cost_usd_tco2_km"], "USD/tCO2/km", assumptions["pipeline_cost_source_id"], "Pipeline cost after currency conversion.")
    add_record("pipeline_emission_tco2e_tco2_km", assumptions["pipeline_emission_tco2e_tco2_km"], "tCO2e/tCO2/km", assumptions["pipeline_cost_source_id"], "Pipeline transport emission factor after unit conversion.")
    add_record("storage_cost_usd_tco2", assumptions["storage_cost_usd_tco2"], "USD/tCO2", assumptions["storage_cost_source_id"], "Storage cost after currency conversion.")
    add_record("mrv_cost_usd_tco2", assumptions["mrv_cost_usd_tco2"], "USD/tCO2", assumptions["mrv_cost_source_id"], "Monitoring/MRV cost after currency conversion, if available.")
    add_record("ship_cost_usd_tco2_km", assumptions["ship_cost_usd_tco2_km"], "USD/tCO2/km", assumptions["ship_assumption_quality"], "Ship proxy cost; NaN if not supplied or absent.")
    add_record("ship_terminal_cost_usd_tco2", assumptions["ship_terminal_cost_usd_tco2"], "USD/tCO2", assumptions["ship_assumption_quality"], "Ship loading/unloading/terminal proxy cost; NaN if not supplied or absent.")
    add_record("ship_emission_tco2e_tco2_km", assumptions["ship_emission_tco2e_tco2_km"], "tCO2e/tCO2/km", assumptions["ship_assumption_quality"], "Ship proxy emission factor; NaN if not supplied or absent.")
    add_record("ship_terminal_emission_tco2e_tco2", assumptions["ship_terminal_emission_tco2e_tco2"], "tCO2e/tCO2", "CLI", "Terminal emission assumption, default 0 unless supplied.")

    return assumptions, pd.DataFrame(records)


# =============================================================================
# Input loaders
# =============================================================================

def load_storage_sites(path: Path) -> pd.DataFrame:
    df = read_csv_flexible(path)
    required = ["storage_site_id", "storage_site_name", "country_code", "country_name", "latitude", "longitude"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df = df.copy()
    df["storage_site_id"] = df["storage_site_id"].astype(str).map(clean_name)
    df["storage_site_name"] = df["storage_site_name"].map(clean_name)
    df["country_code"] = df["country_code"].map(clean_country_code)
    df["country_name"] = df["country_name"].map(clean_name)
    df["latitude"] = to_numeric(df["latitude"])
    df["longitude"] = to_numeric(df["longitude"])
    df["coordinate_valid"] = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    for col in ["coordinate_quality", "storage_type", "offshore_or_onshore", "status", "basin_name", "field_name", "source_id", "island_name"]:
        if col not in df.columns:
            df[col] = ""
    df["island_key"] = df["island_name"].map(clean_island)
    df["offshore_or_onshore_clean"] = df["offshore_or_onshore"].astype(str).str.lower().str.strip()
    df["is_offshore"] = df["offshore_or_onshore_clean"].str.contains("offshore", na=False)
    return df


def classify_storage_maturity(row: pd.Series) -> str:
    """Infer storage maturity from existing CSV text fields without changing the CSV.

    The classifier is intentionally transparent and conservative. It does not create
    new data; it only interprets status/type/quality text that already exists in the
    storage CSV. The class is used by --storage-filter-mode.
    """
    fields = [
        "storage_site_id", "storage_site_name", "storage_type", "status",
        "coordinate_quality", "capacity_category", "basin_name", "field_name",
        "source_id", "notes", "notes_on_data_quality", "primary_source_url",
    ]
    txt = " ".join(str(row.get(c, "")) for c in fields if c in row.index).lower()
    txt_norm = re.sub(r"[^a-z0-9]+", "_", txt)

    # Project/planned cases: specific CCS/CCUS hub/project or development status.
    if re.search(r"operating|operation|under_construction|construction|fid|approved|final_investment|planned|project|hub|pilot|demonstration|injection", txt_norm):
        return "project_or_planned"

    # Specific field candidates are more mature than generic basin screening.
    if re.search(r"depleted|gas_field|oil_field|oilfield|field|eor|egr|reservoir|malampaya|bach_ho|tangguh|arun|kasawari|lang_lebah", txt_norm):
        return "field_candidate"

    # Basin/saline/fairway assessments are usable for broad screening but less mature.
    if re.search(r"basin|saline|aquifer|fairway|formation|assessment|prospective|p10|p50|p90|storage_capacity|regional", txt_norm):
        return "basin_assessment"

    return "screening_or_speculative"


def storage_high_confidence_flag(row: pd.Series) -> bool:
    """Infer whether a storage entry is high-confidence from existing text fields."""
    txt = " ".join(str(row.get(c, "")) for c in [
        "coordinate_quality", "status", "storage_type", "capacity_category", "field_name",
        "notes", "notes_on_data_quality", "source_id",
    ] if c in row.index).lower()
    txt_norm = re.sub(r"[^a-z0-9]+", "_", txt)
    negative = bool(re.search(r"proxy|screening|speculative|unknown|rough|regional_centroid|country_centroid", txt_norm))
    positive = bool(re.search(r"exact|reported|site|field|project|hub|planned|operating|depleted|high|medium_high|p50|official", txt_norm))
    return bool(positive and not negative)


def apply_storage_filter(storage_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Apply an internal maturity filter while keeping the original CSV unchanged.

    mode options:
    - all_assessed_storage: all valid-coordinate storage rows are usable.
    - project_or_high_confidence_storage_only: project/planned, field candidates, or
      high-confidence basin/project entries are usable.
    - project_only: project/planned rows only.
    """
    df = storage_df.copy()
    if "storage_maturity_class" not in df.columns:
        df["storage_maturity_class"] = df.apply(classify_storage_maturity, axis=1)
    if "storage_high_confidence_flag" not in df.columns:
        df["storage_high_confidence_flag"] = df.apply(storage_high_confidence_flag, axis=1)

    mode = str(mode or "all_assessed_storage").strip().lower()
    if mode == "all_assessed_storage":
        usable = df["coordinate_valid"].fillna(False).astype(bool)
    elif mode == "project_or_high_confidence_storage_only":
        usable = (
            df["coordinate_valid"].fillna(False).astype(bool)
            & (
                df["storage_maturity_class"].isin(["project_or_planned", "field_candidate"])
                | df["storage_high_confidence_flag"].fillna(False).astype(bool)
            )
        )
    elif mode == "project_only":
        usable = (
            df["coordinate_valid"].fillna(False).astype(bool)
            & df["storage_maturity_class"].eq("project_or_planned")
        )
    else:
        raise ValueError(
            "Unknown --storage-filter-mode. Use one of: "
            "all_assessed_storage, project_or_high_confidence_storage_only, project_only"
        )

    df["storage_filter_mode"] = mode
    df["storage_filter_usable_flag"] = usable
    return df


def summarize_storage_filter(storage_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = apply_storage_filter(storage_df, mode)
    maturity_summary = (
        df.groupby(["storage_maturity_class", "storage_high_confidence_flag"], dropna=False)
        .agg(
            n_sites=("storage_site_id", "count"),
            n_coordinate_valid=("coordinate_valid", "sum"),
            n_filter_usable=("storage_filter_usable_flag", "sum"),
        )
        .reset_index()
    )
    by_country = (
        df.groupby(["country_code", "country_name", "storage_maturity_class"], dropna=False)
        .agg(
            n_sites=("storage_site_id", "count"),
            n_coordinate_valid=("coordinate_valid", "sum"),
            n_filter_usable=("storage_filter_usable_flag", "sum"),
        )
        .reset_index()
    )
    return maturity_summary, by_country


def load_ports(path: Path | None) -> pd.DataFrame:
    """Load commercial/industrial port coordinates for hybrid ship routing.

    The script accepts a minimal ccs_ports_asean.csv with PORT_COLS.
    Missing file is allowed; ship routing will fall back to proxy_no_port.
    """
    if path is None or not path.exists():
        return pd.DataFrame(columns=PORT_COLS + ["coordinate_valid", "country_key", "island_key"])

    df = read_csv_flexible(path)
    df.columns = [str(c).strip().rstrip(';') for c in df.columns]
    for col in PORT_COLS:
        if col not in df.columns:
            df[col] = np.nan if col in {"latitude", "longitude"} else ""
    df = df.copy()
    df["country_code"] = df["country_code"].map(clean_country_code)
    df["country_name"] = df["country_name"].map(clean_name)
    df["province_id"] = df["province_id"].astype(str).map(clean_name)
    df["province_name"] = df["province_name"].map(clean_name)
    df["port_id"] = df["port_id"].astype(str).map(clean_name)
    df["port_name"] = df["port_name"].map(clean_name)
    df["latitude"] = to_numeric(df["latitude"])
    df["longitude"] = to_numeric(df["longitude"])
    df["coordinate_valid"] = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    df["country_key"] = df["country_code"].map(clean_country_code)
    df["island_key"] = df["island_name"].map(clean_island)
    # Keep only ports with valid coordinates; retain all commercial/general ports by default.
    return df


def nearest_port(lat: float, lon: float, country_code: str, ports_df: pd.DataFrame) -> dict:
    """Find nearest port in the same country. If no same-country port is available, fall back to all ports."""
    empty = {
        "port_id": "", "port_name": "", "country_code": "", "country_name": "",
        "province_id": "", "province_name": "", "island_name": "", "island_key": "",
        "latitude": np.nan, "longitude": np.nan, "distance_km": np.nan,
        "port_match_quality": "no_port_data",
    }
    if ports_df is None or ports_df.empty or not np.isfinite(lat) or not np.isfinite(lon):
        return empty

    valid = ports_df[ports_df["coordinate_valid"]].copy()
    if valid.empty:
        return empty

    cc = clean_country_code(country_code)
    same_country = valid[valid["country_code"] == cc].copy()
    if not same_country.empty:
        candidates = same_country
        quality = "nearest_same_country_port"
    else:
        candidates = valid
        quality = "nearest_any_country_port_fallback"

    distances = [haversine_km(lat, lon, r["latitude"], r["longitude"]) for _, r in candidates.iterrows()]
    candidates = candidates.copy()
    candidates["distance_km"] = distances
    candidates = candidates[np.isfinite(candidates["distance_km"])]
    if candidates.empty:
        return empty

    r = candidates.sort_values("distance_km").iloc[0]
    return {
        "port_id": r.get("port_id", ""),
        "port_name": r.get("port_name", ""),
        "country_code": r.get("country_code", ""),
        "country_name": r.get("country_name", ""),
        "province_id": r.get("province_id", ""),
        "province_name": r.get("province_name", ""),
        "island_name": r.get("island_name", ""),
        "island_key": r.get("island_key", ""),
        "latitude": safe_float(r.get("latitude")),
        "longitude": safe_float(r.get("longitude")),
        "distance_km": safe_float(r.get("distance_km")),
        "port_match_quality": quality,
    }


def find_spatial_file(spatial_dir: Path, explicit_path: Path | None = None) -> Path:
    """Return the ASEAN level-1 province GeoPackage used by the NASA POWER/DAC pipeline.

    Priority:
    1. User-supplied --spatial-file.
    2. spatial_dir/ASEAN/ASEAN_PROVINCES_LEVEL1.gpkg.
    3. D:/Ashka/5.DAC/00.SPATIAL_MAP/ASEAN/ASEAN_PROVINCES_LEVEL1.gpkg.
    4. Fallback glob search.

    This avoids accidentally plotting against NASA output maps or unrelated GPKG files.
    """
    if explicit_path is not None and explicit_path.exists():
        return explicit_path
    if explicit_path is not None and not explicit_path.exists():
        raise FileNotFoundError(f"Spatial file not found: {explicit_path}")

    priority_candidates = [
        spatial_dir / "ASEAN" / "ASEAN_PROVINCES_LEVEL1.gpkg",
        spatial_dir / "ASEAN_PROVINCES_LEVEL1.gpkg",
        SPATIAL_GPKG_DEFAULT,
    ]
    for candidate in priority_candidates:
        if candidate.exists():
            return candidate

    candidates = []
    for pat in ["**/ASEAN_PROVINCES_LEVEL1.gpkg", "**/*ASEAN*PROVINCES*LEVEL1*.gpkg", "**/*PROVINCES*LEVEL1*.gpkg", "**/*.gpkg"]:
        candidates.extend(spatial_dir.glob(pat))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No GPKG file found in {spatial_dir}. Expected primary file: "
            f"{spatial_dir / 'ASEAN' / 'ASEAN_PROVINCES_LEVEL1.gpkg'}"
        )

    def score(p: Path) -> tuple[int, float]:
        name = p.name.lower()
        parent = str(p.parent).lower()
        s = 0
        if name == "asean_provinces_level1.gpkg":
            s += 100
        if "asean" in parent:
            s += 20
        if "asean" in name:
            s += 10
        if "level1" in name:
            s += 8
        if "province" in name or "provinces" in name:
            s += 5
        if "mean_maps" in parent or "nasa_power" in parent:
            s -= 50
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (s, mtime)

    return sorted(candidates, key=score, reverse=True)[0]


def load_province_representative_points(spatial_file: Path) -> tuple[pd.DataFrame, "gpd.GeoDataFrame"]:
    if gpd is None:
        raise ImportError(f"geopandas is required to read the spatial map. Original import error: {_GPD_IMPORT_ERROR}")

    gdf = gpd.read_file(spatial_file)
    if gdf.empty:
        raise ValueError(f"Spatial map is empty: {spatial_file}")

    country_code_col = find_col(gdf, COUNTRY_CODE_COLS, required=True, label="country code column")
    country_name_col = find_col(gdf, COUNTRY_NAME_COLS, required=False, label="country name column")
    province_id_col = find_col(gdf, PROVINCE_ID_COLS, required=False, label="province id column")
    province_name_col = find_col(gdf, PROVINCE_NAME_COLS, required=True, label="province name column")

    if gdf.crs is None:
        warnings.warn("Spatial map CRS is missing. Assuming EPSG:4326.")
        gdf = gdf.set_crs(epsg=4326)

    # Representative point should be created in native CRS for geometric validity, then converted to WGS84.
    rep = gdf.copy()
    rep["geometry"] = rep.geometry.representative_point()
    rep_wgs = rep.to_crs(epsg=4326)

    province_points = pd.DataFrame({
        "country_code": rep_wgs[country_code_col].map(clean_country_code),
        "country_name": rep_wgs[country_name_col].map(clean_name) if country_name_col else "",
        "province_id": rep_wgs[province_id_col].astype(str).map(clean_name) if province_id_col else rep_wgs.index.astype(str),
        "province_name": rep_wgs[province_name_col].map(clean_name),
        "province_rep_lon": rep_wgs.geometry.x,
        "province_rep_lat": rep_wgs.geometry.y,
    })

    # Standardized polygon layer for plotting and joins.
    gdf_plot = gdf.to_crs(epsg=4326).copy()
    gdf_plot["country_code"] = gdf_plot[country_code_col].map(clean_country_code)
    gdf_plot["province_id"] = gdf_plot[province_id_col].astype(str).map(clean_name) if province_id_col else gdf_plot.index.astype(str)
    gdf_plot["province_name"] = gdf_plot[province_name_col].map(clean_name)
    if country_name_col:
        gdf_plot["country_name"] = gdf_plot[country_name_col].map(clean_name)
    else:
        gdf_plot["country_name"] = ""

    return province_points, gdf_plot


def load_energy_summary(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["country_code", "country_name", "province_id", "province_name"]:
        if col not in df.columns:
            df[col] = ""
    df["country_code"] = df["country_code"].map(clean_country_code)
    df["province_id"] = df["province_id"].astype(str).map(clean_name)
    df["province_name"] = df["province_name"].map(clean_name)
    return df


# =============================================================================
# Storage selection and transport calculation
# =============================================================================



def calculate_candidate_route(
    province: pd.Series,
    storage: pd.Series,
    assumptions: dict,
    ports_df: pd.DataFrame,
    ship_distance_multiplier: float,
    pipeline_default_if_ship_missing: bool = True,
    different_island_pipeline_multiplier: float = 1.0,
    max_different_island_pipeline_km: float = 0.0,
    force_ship_for_different_island: bool = False,
    transport_sensitivity_case: str = "baseline",
) -> dict:
    """Calculate direct-pipeline and ship/hybrid route metrics for one province-storage candidate.

    Changes relative to the first screening version:
    - same-island still forces pipeline;
    - different-island direct pipeline receives an additional multiplier;
    - different-island direct pipeline can be marked infeasible above a distance cap;
    - a forced-ship sensitivity can be evaluated without changing the resource CSVs.
    """
    p_country = clean_country_code(province["country_code"])
    p_lat = safe_float(province["province_rep_lat"])
    p_lon = safe_float(province["province_rep_lon"])
    s_country = clean_country_code(storage.get("country_code", ""))
    s_lat = safe_float(storage.get("latitude"))
    s_lon = safe_float(storage.get("longitude"))

    straight_km = haversine_km(p_lat, p_lon, s_lat, s_lon)
    is_offshore = bool(storage.get("is_offshore", False))

    origin_port = nearest_port(p_lat, p_lon, p_country, ports_df)
    dest_port = nearest_port(s_lat, s_lon, s_country, ports_df)

    p_island_key = clean_island(province.get("island_name", "")) or origin_port.get("island_key", "")
    p_island_name = province.get("island_name", "") if clean_island(province.get("island_name", "")) else origin_port.get("island_name", "")
    s_island_key = clean_island(storage.get("island_name", "")) or dest_port.get("island_key", "")
    s_island_name = storage.get("island_name", "") if clean_island(storage.get("island_name", "")) else dest_port.get("island_name", "")

    same_island = bool(p_island_key and s_island_key and p_island_key == s_island_key and p_country == s_country)
    different_island = not same_island
    island_rule_quality = "port_or_storage_island_match" if p_island_key and s_island_key else "island_unknown_proxy"

    # Direct pipeline screening distance.
    offshore_mult = assumptions["offshore_pipeline_multiplier"] if is_offshore else 1.0
    interisland_mult = float(different_island_pipeline_multiplier) if different_island else 1.0
    pipeline_mult = offshore_mult * interisland_mult
    pipeline_eff_km = straight_km * pipeline_mult if np.isfinite(straight_km) else np.nan

    pipeline_feasible = bool(np.isfinite(pipeline_eff_km))
    pipeline_infeasible_reason = ""
    max_cap = safe_float(max_different_island_pipeline_km, 0.0)
    if different_island and max_cap > 0 and np.isfinite(straight_km) and straight_km > max_cap:
        pipeline_feasible = False
        pipeline_infeasible_reason = f"different_island_distance_above_cap_{max_cap:g}km"
    if different_island and bool(force_ship_for_different_island):
        pipeline_feasible = False
        pipeline_infeasible_reason = (pipeline_infeasible_reason + ";" if pipeline_infeasible_reason else "") + "forced_ship_sensitivity"

    pipeline_cost_rate = assumptions["pipeline_cost_usd_tco2_km"]
    pipeline_cost = pipeline_eff_km * pipeline_cost_rate if pipeline_feasible and np.isfinite(pipeline_eff_km) and np.isfinite(pipeline_cost_rate) else np.nan
    pipeline_ef = assumptions["pipeline_emission_tco2e_tco2_km"]
    pipeline_emis = pipeline_eff_km * pipeline_ef if pipeline_feasible and np.isfinite(pipeline_eff_km) and np.isfinite(pipeline_ef) else np.nan

    # Hybrid route: DAC -> origin port (land), origin port -> destination port (sea), destination port -> storage (land).
    ports_available = all(np.isfinite(v) for v in [
        origin_port.get("latitude", np.nan), origin_port.get("longitude", np.nan),
        dest_port.get("latitude", np.nan), dest_port.get("longitude", np.nan),
    ])

    if ports_available:
        land_origin_km = haversine_km(p_lat, p_lon, origin_port["latitude"], origin_port["longitude"])
        sea_km = haversine_km(origin_port["latitude"], origin_port["longitude"], dest_port["latitude"], dest_port["longitude"]) * ship_distance_multiplier
        land_storage_km = haversine_km(dest_port["latitude"], dest_port["longitude"], s_lat, s_lon)
        ship_eff_km = land_origin_km + sea_km + land_storage_km
        ship_route_quality = "port_to_port_straightline_with_land_legs"
    else:
        land_origin_km = np.nan
        sea_km = straight_km * ship_distance_multiplier if np.isfinite(straight_km) else np.nan
        land_storage_km = np.nan
        ship_eff_km = sea_km
        ship_route_quality = "proxy_no_port_straightline"

    ship_cost_rate = assumptions["ship_cost_usd_tco2_km"]
    ship_terminal_cost = assumptions["ship_terminal_cost_usd_tco2"]

    land_total_km = 0.0
    if np.isfinite(land_origin_km):
        land_total_km += land_origin_km
    if np.isfinite(land_storage_km):
        land_total_km += land_storage_km
    land_cost = land_total_km * pipeline_cost_rate if np.isfinite(pipeline_cost_rate) and land_total_km > 0 else 0.0

    if np.isfinite(ship_cost_rate) and np.isfinite(ship_terminal_cost) and np.isfinite(sea_km):
        ship_cost = land_cost + ship_terminal_cost + sea_km * ship_cost_rate
        ship_cost_available = True
    else:
        ship_cost = np.nan
        ship_cost_available = False

    ship_ef = assumptions["ship_emission_tco2e_tco2_km"]
    ship_terminal_ef = assumptions["ship_terminal_emission_tco2e_tco2"]
    land_emis = land_total_km * pipeline_ef if np.isfinite(pipeline_ef) and land_total_km > 0 else 0.0
    if np.isfinite(ship_ef) and np.isfinite(sea_km):
        ship_emis = land_emis + ship_terminal_ef + sea_km * ship_ef
    else:
        ship_emis = np.nan

    storage_cost = assumptions["storage_cost_usd_tco2"]
    mrv_cost = assumptions["mrv_cost_usd_tco2"]

    if same_island:
        selected_mode = "pipeline_same_island_rule"
        selected_transport_cost = pipeline_cost
        selected_transport_emis = pipeline_emis
        selected_total_distance = pipeline_eff_km
        mode_selection_rule = "same_island_forced_pipeline"
    else:
        mode_selection_rule = "different_island_compare_pipeline_vs_ship_hybrid_with_feasibility_screen"
        if ship_cost_available and np.isfinite(pipeline_cost):
            if ship_cost < pipeline_cost:
                selected_mode = "ship_hybrid"
                selected_transport_cost = ship_cost
                selected_transport_emis = ship_emis
                selected_total_distance = ship_eff_km
            else:
                selected_mode = "pipeline_different_island_cheaper"
                selected_transport_cost = pipeline_cost
                selected_transport_emis = pipeline_emis
                selected_total_distance = pipeline_eff_km
        elif ship_cost_available and not np.isfinite(pipeline_cost):
            selected_mode = "ship_hybrid"
            selected_transport_cost = ship_cost
            selected_transport_emis = ship_emis
            selected_total_distance = ship_eff_km
        elif np.isfinite(pipeline_cost) and pipeline_default_if_ship_missing:
            selected_mode = "pipeline_different_island_ship_cost_missing"
            selected_transport_cost = pipeline_cost
            selected_transport_emis = pipeline_emis
            selected_total_distance = pipeline_eff_km
        else:
            selected_mode = "transport_cost_unavailable"
            selected_transport_cost = np.nan
            selected_transport_emis = np.nan
            selected_total_distance = np.nan

    total_ts = selected_transport_cost
    if np.isfinite(total_ts) and np.isfinite(storage_cost):
        total_ts += storage_cost
    if np.isfinite(total_ts) and np.isfinite(mrv_cost):
        total_ts += mrv_cost

    cross_border_used = bool(s_country != p_country)
    cost_gap = ship_cost - pipeline_cost if np.isfinite(ship_cost) and np.isfinite(pipeline_cost) else np.nan

    return {
        "transport_sensitivity_case": transport_sensitivity_case,
        "country_code": p_country,
        "country_name": province.get("country_name", ""),
        "province_id": province.get("province_id", ""),
        "province_name": province.get("province_name", ""),
        "province_rep_lat": p_lat,
        "province_rep_lon": p_lon,
        "province_island_name": p_island_name,
        "province_island_key": p_island_key,
        "storage_site_id": storage.get("storage_site_id", ""),
        "storage_site_name": storage.get("storage_site_name", ""),
        "storage_country_code": s_country,
        "storage_country_name": storage.get("country_name", ""),
        "storage_lat": s_lat,
        "storage_lon": s_lon,
        "storage_island_name": s_island_name,
        "storage_island_key": s_island_key,
        "storage_type": storage.get("storage_type", ""),
        "storage_maturity_class": storage.get("storage_maturity_class", ""),
        "storage_high_confidence_flag": storage.get("storage_high_confidence_flag", np.nan),
        "storage_filter_mode": storage.get("storage_filter_mode", ""),
        "storage_filter_usable_flag": storage.get("storage_filter_usable_flag", np.nan),
        "basin_name": storage.get("basin_name", ""),
        "field_name": storage.get("field_name", ""),
        "offshore_or_onshore": storage.get("offshore_or_onshore", ""),
        "storage_status": storage.get("status", ""),
        "coordinate_quality": storage.get("coordinate_quality", ""),
        "is_offshore_storage": is_offshore,
        "same_island_flag": same_island,
        "different_island_flag": different_island,
        "island_rule_quality": island_rule_quality,
        "cross_border_used": cross_border_used,
        "straight_distance_km": straight_km,
        "pipeline_effective_distance_km": pipeline_eff_km,
        "offshore_pipeline_multiplier_applied": offshore_mult,
        "different_island_pipeline_multiplier_applied": interisland_mult,
        "pipeline_feasible_flag": pipeline_feasible,
        "pipeline_infeasible_reason": pipeline_infeasible_reason,
        "max_different_island_pipeline_km": max_cap,
        "origin_port_id": origin_port.get("port_id", ""),
        "origin_port_name": origin_port.get("port_name", ""),
        "origin_port_lat": origin_port.get("latitude", np.nan),
        "origin_port_lon": origin_port.get("longitude", np.nan),
        "origin_port_distance_km": origin_port.get("distance_km", np.nan),
        "origin_port_match_quality": origin_port.get("port_match_quality", ""),
        "destination_port_id": dest_port.get("port_id", ""),
        "destination_port_name": dest_port.get("port_name", ""),
        "destination_port_lat": dest_port.get("latitude", np.nan),
        "destination_port_lon": dest_port.get("longitude", np.nan),
        "destination_port_distance_km": dest_port.get("distance_km", np.nan),
        "destination_port_match_quality": dest_port.get("port_match_quality", ""),
        "land_distance_origin_km": land_origin_km,
        "sea_distance_km": sea_km,
        "land_distance_storage_km": land_storage_km,
        "ship_hybrid_effective_distance_km": ship_eff_km,
        "ship_route_quality": ship_route_quality,
        "pipeline_transport_cost_USD_tCO2": pipeline_cost,
        "ship_hybrid_transport_cost_USD_tCO2": ship_cost,
        "ship_minus_pipeline_cost_USD_tCO2": cost_gap,
        "ship_cost_available": bool(ship_cost_available),
        "ship_assumption_quality": assumptions["ship_assumption_quality"],
        "selected_transport_mode": selected_mode,
        "selected_total_distance_km": selected_total_distance,
        "selected_transport_cost_USD_tCO2": selected_transport_cost,
        "storage_cost_USD_tCO2": storage_cost,
        "mrv_cost_USD_tCO2": mrv_cost,
        "total_TandS_cost_USD_tCO2": total_ts,
        "pipeline_transport_emission_tCO2e_tCO2": pipeline_emis,
        "ship_hybrid_transport_emission_tCO2e_tCO2": ship_emis,
        "selected_transport_emission_tCO2e_tCO2": selected_transport_emis,
        "mode_selection_rule": mode_selection_rule,
        "pipeline_route_quality": (
            "interisland_direct_pipeline_screening" if different_island else
            ("straight_line_screening_with_offshore_multiplier" if is_offshore else "straight_line_screening")
        ),
        "ccs_valid_flag": bool(np.isfinite(total_ts) and np.isfinite(selected_total_distance)),
        "invalid_reason": "" if np.isfinite(total_ts) else "missing_transport_or_storage_cost_assumption",
    }


def build_province_to_storage(
    province_points: pd.DataFrame,
    storage_df: pd.DataFrame,
    assumptions: dict,
    ship_distance_multiplier: float,
    ports_df: pd.DataFrame | None = None,
    storage_filter_mode: str = "all_assessed_storage",
    different_island_pipeline_multiplier: float = 1.0,
    max_different_island_pipeline_km: float = 0.0,
    force_ship_for_different_island: bool = False,
    transport_sensitivity_case: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create all-candidate and selected province-to-storage CCS tables.

    Domestic usable storage is now interpreted as:
        valid coordinate + passes --storage-filter-mode.
    Cross-border storage is allowed only when no domestic storage is usable under
    that selected filter mode.
    """
    if ports_df is None:
        ports_df = pd.DataFrame(columns=PORT_COLS)

    storage_work = apply_storage_filter(storage_df, storage_filter_mode)
    all_records = []
    selected_records = []
    usable_storage = storage_work[storage_work["coordinate_valid"] & storage_work["storage_filter_usable_flag"]].copy()

    for _, prov in province_points.iterrows():
        p_country = clean_country_code(prov["country_code"])
        domestic_all = storage_work[storage_work["country_code"] == p_country].copy()
        domestic_storage_available = len(domestic_all) > 0
        domestic_coordinate_valid = domestic_all[domestic_all["coordinate_valid"]].copy()
        domestic_storage_coordinate_valid = len(domestic_coordinate_valid) > 0
        domestic_usable = domestic_all[domestic_all["coordinate_valid"] & domestic_all["storage_filter_usable_flag"]].copy()
        domestic_storage_usable = len(domestic_usable) > 0
        domestic_storage_coordinate_missing = domestic_storage_available and not domestic_storage_coordinate_valid
        domestic_storage_filtered_out = domestic_storage_coordinate_valid and not domestic_storage_usable

        if domestic_storage_usable:
            candidates = domestic_usable.copy()
            cross_border_allowed = False
            selection_rule = f"domestic_storage_only_filter_{storage_filter_mode}"
        else:
            candidates = usable_storage.copy()
            cross_border_allowed = True
            selection_rule = f"asean_fallback_no_domestic_storage_usable_under_{storage_filter_mode}"

        if candidates.empty:
            base = {
                "transport_sensitivity_case": transport_sensitivity_case,
                "country_code": p_country,
                "country_name": prov.get("country_name", ""),
                "province_id": prov.get("province_id", ""),
                "province_name": prov.get("province_name", ""),
                "province_rep_lat": safe_float(prov.get("province_rep_lat")),
                "province_rep_lon": safe_float(prov.get("province_rep_lon")),
                "storage_filter_mode": storage_filter_mode,
                "domestic_storage_available": bool(domestic_storage_available),
                "domestic_storage_coordinate_valid": bool(domestic_storage_coordinate_valid),
                "domestic_storage_usable": bool(domestic_storage_usable),
                "domestic_storage_coordinate_missing": bool(domestic_storage_coordinate_missing),
                "domestic_storage_filtered_out": bool(domestic_storage_filtered_out),
                "cross_border_allowed": bool(cross_border_allowed),
                "selection_rule": selection_rule,
                "ccs_valid_flag": False,
                "invalid_reason": "no_usable_storage_under_filter_mode",
            }
            selected_records.append(base)
            all_records.append(base.copy())
            continue

        candidate_rows = []
        for _, s in candidates.iterrows():
            row = calculate_candidate_route(
                province=prov,
                storage=s,
                assumptions=assumptions,
                ports_df=ports_df,
                ship_distance_multiplier=ship_distance_multiplier,
                different_island_pipeline_multiplier=different_island_pipeline_multiplier,
                max_different_island_pipeline_km=max_different_island_pipeline_km,
                force_ship_for_different_island=force_ship_for_different_island,
                transport_sensitivity_case=transport_sensitivity_case,
            )
            row.update({
                "domestic_storage_available": bool(domestic_storage_available),
                "domestic_storage_coordinate_valid": bool(domestic_storage_coordinate_valid),
                "domestic_storage_usable": bool(domestic_storage_usable),
                "domestic_storage_coordinate_missing": bool(domestic_storage_coordinate_missing),
                "domestic_storage_filtered_out": bool(domestic_storage_filtered_out),
                "cross_border_allowed": bool(cross_border_allowed),
                "selection_rule": selection_rule,
            })
            candidate_rows.append(row)
            all_records.append(row)

        cand = pd.DataFrame(candidate_rows)
        if not cand.empty:
            valid_cost = cand[np.isfinite(pd.to_numeric(cand["total_TandS_cost_USD_tCO2"], errors="coerce"))].copy()
            if not valid_cost.empty:
                selected = valid_cost.sort_values(["total_TandS_cost_USD_tCO2", "selected_total_distance_km"], ascending=[True, True]).iloc[0].to_dict()
            else:
                selected = cand.sort_values("selected_total_distance_km", ascending=True, na_position="last").iloc[0].to_dict()
            selected["nearest_storage_site_id"] = selected.get("storage_site_id", "")
            selected["nearest_storage_site_name"] = selected.get("storage_site_name", "")
            selected["nearest_storage_country_code"] = selected.get("storage_country_code", "")
            selected["nearest_storage_country_name"] = selected.get("storage_country_name", "")
            selected["nearest_storage_lat"] = selected.get("storage_lat", np.nan)
            selected["nearest_storage_lon"] = selected.get("storage_lon", np.nan)
            selected_records.append(selected)

    all_candidates = pd.DataFrame(all_records)
    selected_df = pd.DataFrame(selected_records)
    return all_candidates, selected_df


def build_transport_sensitivity_outputs(
    province_points: pd.DataFrame,
    storage_df: pd.DataFrame,
    assumptions: dict,
    ship_distance_multiplier: float,
    ports_df: pd.DataFrame,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run screening sensitivities without changing the main output files."""
    cases = [
        {
            "case": "baseline_selected_settings",
            "storage_filter_mode": args.storage_filter_mode,
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "force_ship_for_different_island": False,
        },
        {
            "case": "high_interisland_pipeline_penalty",
            "storage_filter_mode": args.storage_filter_mode,
            "different_island_pipeline_multiplier": args.sensitivity_high_interisland_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "force_ship_for_different_island": False,
        },
        {
            "case": "distance_cap_500km",
            "storage_filter_mode": args.storage_filter_mode,
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": 500.0,
            "force_ship_for_different_island": False,
        },
        {
            "case": "forced_ship_different_island",
            "storage_filter_mode": args.storage_filter_mode,
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "force_ship_for_different_island": True,
        },
        {
            "case": "project_or_high_confidence_storage_only",
            "storage_filter_mode": "project_or_high_confidence_storage_only",
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "force_ship_for_different_island": False,
        },
        {
            "case": "project_only_storage",
            "storage_filter_mode": "project_only",
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "force_ship_for_different_island": False,
        },
    ]

    selected_frames = []
    summary_rows = []
    for c in cases:
        _, sel = build_province_to_storage(
            province_points=province_points,
            storage_df=storage_df,
            assumptions=assumptions,
            ship_distance_multiplier=ship_distance_multiplier,
            ports_df=ports_df,
            storage_filter_mode=c["storage_filter_mode"],
            different_island_pipeline_multiplier=c["different_island_pipeline_multiplier"],
            max_different_island_pipeline_km=c["max_different_island_pipeline_km"],
            force_ship_for_different_island=c["force_ship_for_different_island"],
            transport_sensitivity_case=c["case"],
        )
        if sel.empty:
            continue
        sel["transport_sensitivity_case"] = c["case"]
        selected_frames.append(sel)
        mode_counts = sel.groupby("selected_transport_mode", dropna=False).size().to_dict() if "selected_transport_mode" in sel.columns else {}
        summary_rows.append({
            "transport_sensitivity_case": c["case"],
            "storage_filter_mode": c["storage_filter_mode"],
            "different_island_pipeline_multiplier": c["different_island_pipeline_multiplier"],
            "max_different_island_pipeline_km": c["max_different_island_pipeline_km"],
            "force_ship_for_different_island": c["force_ship_for_different_island"],
            "n_provinces": len(sel),
            "n_valid": int(sel.get("ccs_valid_flag", pd.Series(False, index=sel.index)).fillna(False).sum()),
            "n_cross_border": int(sel.get("cross_border_used", pd.Series(False, index=sel.index)).fillna(False).sum()) if "cross_border_used" in sel.columns else 0,
            "n_ship_hybrid": int(mode_counts.get("ship_hybrid", 0)),
            "n_pipeline_same_island": int(mode_counts.get("pipeline_same_island_rule", 0)),
            "n_pipeline_different_island": int(sum(v for k, v in mode_counts.items() if str(k).startswith("pipeline_different_island"))),
            "median_TandS_cost_USD_tCO2": float(pd.to_numeric(sel.get("total_TandS_cost_USD_tCO2"), errors="coerce").median()) if "total_TandS_cost_USD_tCO2" in sel.columns else np.nan,
            "median_selected_distance_km": float(pd.to_numeric(sel.get("selected_total_distance_km"), errors="coerce").median()) if "selected_total_distance_km" in sel.columns else np.nan,
        })

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return selected_all, summary

# =============================================================================
# Merge with energy summary
# =============================================================================

def find_annual_co2_column(energy_df: pd.DataFrame) -> str | None:
    for col in ANNUAL_CO2_COLS:
        if col in energy_df.columns:
            return col
    # fallback search
    candidates = [c for c in energy_df.columns if "co2" in c.lower() and "annual" in c.lower() and ("t" in c.lower() or "ton" in c.lower())]
    return candidates[0] if candidates else None


def merge_energy_ccs(energy_df: pd.DataFrame, province_ccs: pd.DataFrame) -> pd.DataFrame:
    """Merge province-level CCS routing results into every scenario row from module 03.

    Important compatibility rule:
    - Keep every column from the 03 energy summary, including the new hourly-demand
      diagnostics/design-capacity columns.
    - Add CCS columns using suffixes only when names collide.
    - Add explicit combined validity flags for module 05 and diagnostics.
    """
    if energy_df is None or energy_df.empty:
        return pd.DataFrame()

    df = energy_df.copy()
    for col in ["country_code", "province_id", "province_name"]:
        if col not in df.columns:
            df[col] = ""
    df["country_code"] = df["country_code"].map(clean_country_code)
    df["province_id"] = df["province_id"].astype(str).map(clean_name)
    df["province_name"] = df["province_name"].map(clean_name)

    # Ensure new hourly-profile columns exist so downstream scripts can rely on a stable schema.
    # They are filled only if module 03 created them; otherwise NaN/False fallback is transparent.
    for col in ENERGY_PROFILE_PASSTHROUGH_COLS:
        if col not in df.columns:
            if col == "hourly_profile_used_flag":
                df[col] = False
            elif col == "demand_profile_method":
                df[col] = "not_reported_by_module03"
            else:
                df[col] = np.nan

    ccs = province_ccs.copy()
    ccs["country_code"] = ccs["country_code"].map(clean_country_code)
    ccs["province_id"] = ccs["province_id"].astype(str).map(clean_name)
    ccs["province_name"] = ccs["province_name"].map(clean_name)

    # Prefer country_code + province_id + province_name. This keeps province-name
    # ambiguity low while preserving all scenario rows from module 03.
    ccs_cols = [c for c in ccs.columns if c not in {"country_name"}]
    merged = df.merge(
        ccs[ccs_cols],
        on=["country_code", "province_id", "province_name"],
        how="left",
        suffixes=("", "_ccs"),
    )

    # Standardize CCS invalid-reason name after merge. Energy scenarios already use
    # invalid_reason; CCS invalid reason may appear as invalid_reason_ccs.
    if "invalid_reason_ccs" in merged.columns and "ccs_invalid_reason" not in merged.columns:
        merged["ccs_invalid_reason"] = merged["invalid_reason_ccs"]
    elif "ccs_invalid_reason" not in merged.columns:
        merged["ccs_invalid_reason"] = ""

    if "ccs_valid_flag" in merged.columns:
        merged["ccs_valid_flag"] = merged["ccs_valid_flag"].fillna(False).astype(bool)
    else:
        merged["ccs_valid_flag"] = False

    if "scenario_valid_flag" in merged.columns:
        merged["energy_scenario_valid_flag"] = merged["scenario_valid_flag"].fillna(False).astype(bool)
    else:
        merged["energy_scenario_valid_flag"] = True

    merged["energy_ccs_valid_flag"] = merged["energy_scenario_valid_flag"] & merged["ccs_valid_flag"]
    merged["ccs_join_match_flag"] = pd.to_numeric(
        merged.get("total_TandS_cost_USD_tCO2"), errors="coerce"
    ).notna()

    annual_co2_col = find_annual_co2_column(merged)
    if annual_co2_col is not None:
        annual_co2 = pd.to_numeric(merged[annual_co2_col], errors="coerce")
    else:
        annual_co2 = pd.Series(np.nan, index=merged.index)
        merged["annual_CO2_for_CCS_t"] = np.nan

    merged["annual_CO2_for_CCS_t"] = annual_co2
    merged["annual_TandS_cost_USD"] = annual_co2 * pd.to_numeric(merged.get("total_TandS_cost_USD_tCO2"), errors="coerce")
    merged["annual_transport_emission_tCO2e"] = annual_co2 * pd.to_numeric(merged.get("selected_transport_emission_tCO2e_tCO2"), errors="coerce")

    # Storage emission not included unless a future dataset adds it.
    merged["annual_storage_emission_tCO2e"] = np.nan
    merged["annual_total_CCS_emission_tCO2e"] = merged["annual_transport_emission_tCO2e"]
    merged["annual_total_CCS_cost_USD"] = merged["annual_TandS_cost_USD"]

    # Convenience flags for downstream auditing.
    merged["module04_preserved_hourly_profile_columns_flag"] = all(
        c in merged.columns for c in ENERGY_PROFILE_PASSTHROUGH_COLS
    )

    return merged


def make_energy_profile_passthrough_diagnostics(energy_df: pd.DataFrame | None, energy_ccs: pd.DataFrame) -> pd.DataFrame:
    """Create diagnostics showing whether module 03 hourly-demand columns survived module 04."""
    records = []
    source_cols = set(energy_df.columns) if energy_df is not None and not energy_df.empty else set()
    out_cols = set(energy_ccs.columns) if energy_ccs is not None and not energy_ccs.empty else set()
    records.append({"metric": "energy_input_rows", "value": 0 if energy_df is None else len(energy_df)})
    records.append({"metric": "energy_ccs_output_rows", "value": 0 if energy_ccs is None else len(energy_ccs)})
    for col in ENERGY_PROFILE_PASSTHROUGH_COLS:
        records.append({
            "metric": f"column_present_in_energy_input::{col}",
            "value": bool(col in source_cols),
        })
        records.append({
            "metric": f"column_present_in_energy_ccs_output::{col}",
            "value": bool(col in out_cols),
        })

    if energy_ccs is not None and not energy_ccs.empty:
        if "demand_profile_method" in energy_ccs.columns:
            vc = energy_ccs["demand_profile_method"].astype(str).value_counts(dropna=False)
            for k, v in vc.items():
                records.append({"metric": f"demand_profile_method_rows::{k}", "value": int(v)})
        if "hourly_profile_used_flag" in energy_ccs.columns:
            vc = energy_ccs["hourly_profile_used_flag"].fillna(False).astype(bool).value_counts(dropna=False)
            for k, v in vc.items():
                records.append({"metric": f"hourly_profile_used_flag_rows::{k}", "value": int(v)})
        if "energy_ccs_valid_flag" in energy_ccs.columns:
            records.append({
                "metric": "energy_ccs_valid_rows",
                "value": int(energy_ccs["energy_ccs_valid_flag"].fillna(False).astype(bool).sum()),
            })
        if "ccs_join_match_flag" in energy_ccs.columns:
            records.append({
                "metric": "ccs_join_matched_rows",
                "value": int(energy_ccs["ccs_join_match_flag"].fillna(False).astype(bool).sum()),
            })
        for col in ["heat_pump_capacity_kWth_design", "p95_heat_demand_kWth", "peak_heat_demand_kWth"]:
            if col in energy_ccs.columns:
                s = pd.to_numeric(energy_ccs[col], errors="coerce")
                records.extend([
                    {"metric": f"{col}_nonmissing_rows", "value": int(s.notna().sum())},
                    {"metric": f"{col}_median", "value": float(s.median()) if s.notna().any() else np.nan},
                    {"metric": f"{col}_p95", "value": float(s.quantile(0.95)) if s.notna().any() else np.nan},
                    {"metric": f"{col}_max", "value": float(s.max()) if s.notna().any() else np.nan},
                ])

    return pd.DataFrame(records)


# =============================================================================
# Figures
# =============================================================================


def set_consulting_style() -> dict:
    """Apply clean consulting-style matplotlib defaults and return palette."""
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
            "axes.edgecolor": palette["dark_gray"],
            "axes.labelcolor": palette["dark_gray"],
            "axes.titlecolor": palette["navy"],
            "xtick.color": palette["dark_gray"],
            "ytick.color": palette["dark_gray"],
            "figure.facecolor": palette["white"],
            "axes.facecolor": palette["white"],
            "savefig.facecolor": palette["white"],
            "axes.grid": False,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        })
    return palette


def _safe_to_file(gdf_obj, path: Path) -> None:
    try:
        gdf_obj.to_file(path, driver="GeoJSON")
    except Exception as exc:
        path.with_suffix(path.suffix + ".ERROR.txt").write_text(str(exc), encoding="utf-8")


def _finish_map(ax, title: str, palette: dict) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(palette["gray"])
    ax.spines["bottom"].set_color(palette["gray"])


def _merge_selection_to_polygons(provinces_gdf, selection_df: pd.DataFrame):
    g = provinces_gdf.copy()
    sel = selection_df.copy()
    for df in [g, sel]:
        for col in ["country_code", "province_id", "province_name"]:
            if col not in df.columns:
                df[col] = ""
        df["country_code_clean"] = df["country_code"].map(clean_country_code)
        df["province_id_clean"] = df["province_id"].astype(str).map(clean_name)
        df["province_name_clean"] = df["province_name"].map(clean_name)

    value_cols = [
        "country_code_clean", "province_id_clean", "province_name_clean",
        "selected_transport_mode", "selected_total_distance_km", "total_TandS_cost_USD_tCO2",
        "selected_transport_cost_USD_tCO2", "storage_cost_USD_tCO2", "mrv_cost_USD_tCO2",
        "cross_border_used", "same_island_flag", "ccs_valid_flag", "nearest_storage_site_name",
        "nearest_storage_country_code", "selected_transport_emission_tCO2e_tCO2",
    ]
    keep = [c for c in value_cols if c in sel.columns]
    sel_id = sel[keep].drop_duplicates(["country_code_clean", "province_id_clean", "province_name_clean"], keep="first")
    merged = g.merge(sel_id, on=["country_code_clean", "province_id_clean", "province_name_clean"], how="left", suffixes=("", "_sel"))
    if merged["ccs_valid_flag"].notna().sum() == 0:
        # Fallback when province IDs differ: country + province name only.
        keep2 = [c for c in keep if c != "province_id_clean"]
        sel_name = sel[keep2].drop_duplicates(["country_code_clean", "province_name_clean"], keep="first")
        drop_cols = [c for c in keep if c in merged.columns and c not in ["country_code_clean", "province_id_clean", "province_name_clean"]]
        merged = g.drop(columns=[c for c in drop_cols if c in g.columns], errors="ignore").merge(
            sel_name,
            on=["country_code_clean", "province_name_clean"],
            how="left",
            suffixes=("", "_sel"),
        )
    return merged


def make_lines_gdf(selection_df: pd.DataFrame, max_lines: int = 30):
    if gpd is None or LineString is None:
        return None
    valid = selection_df[
        selection_df["province_rep_lat"].notna()
        & selection_df["province_rep_lon"].notna()
        & selection_df["nearest_storage_lat"].notna()
        & selection_df["nearest_storage_lon"].notna()
    ].copy()
    if valid.empty:
        return None

    if "selected_total_distance_km" in valid.columns:
        valid_sorted = valid.sort_values("selected_total_distance_km")
    else:
        valid_sorted = valid.sort_values("straight_distance_km")
    shortest = valid_sorted.head(10)
    longest = valid_sorted.tail(10)
    remaining = valid_sorted.iloc[10:-10] if len(valid_sorted) > 20 else valid_sorted.iloc[0:0]
    middle = remaining.sample(n=min(10, len(remaining)), random_state=42) if len(remaining) > 0 else valid_sorted.iloc[0:0]
    sample = pd.concat([shortest, middle, longest], ignore_index=True).drop_duplicates(subset=["country_code", "province_id"])
    if len(sample) > max_lines:
        sample = sample.head(max_lines)

    lines = []
    for _, row in sample.iterrows():
        mode = str(row.get("selected_transport_mode", "")).lower()
        has_ports = all(pd.notna(row.get(c)) for c in ["origin_port_lon", "origin_port_lat", "destination_port_lon", "destination_port_lat"])
        if "ship" in mode and has_ports:
            coords = [
                (float(row["province_rep_lon"]), float(row["province_rep_lat"])),
                (float(row["origin_port_lon"]), float(row["origin_port_lat"])),
                (float(row["destination_port_lon"]), float(row["destination_port_lat"])),
                (float(row["nearest_storage_lon"]), float(row["nearest_storage_lat"])),
            ]
        else:
            coords = [
                (float(row["province_rep_lon"]), float(row["province_rep_lat"])),
                (float(row["nearest_storage_lon"]), float(row["nearest_storage_lat"])),
            ]
        lines.append(LineString(coords))
    return gpd.GeoDataFrame(sample, geometry=lines, crs="EPSG:4326")


def make_figures(
    out_dir: Path,
    provinces_gdf,
    province_selection: pd.DataFrame,
    storage_df: pd.DataFrame,
    ports_df: pd.DataFrame | None,
    max_lines: int,
) -> None:
    fig_dir = out_dir / "figures"
    maps_dir = out_dir / "maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    if plt is None or gpd is None:
        (fig_dir / "FIGURES_NOT_CREATED.txt").write_text(
            f"Figures not created because matplotlib/geopandas is unavailable. matplotlib error={_MPL_IMPORT_ERROR}; geopandas error={_GPD_IMPORT_ERROR}",
            encoding="utf-8",
        )
        return

    palette = set_consulting_style()

    valid_storage = storage_df[storage_df["coordinate_valid"]].copy()
    storage_gdf = None
    if not valid_storage.empty:
        storage_gdf = gpd.GeoDataFrame(
            valid_storage,
            geometry=gpd.points_from_xy(valid_storage["longitude"], valid_storage["latitude"]),
            crs="EPSG:4326",
        )
        _safe_to_file(storage_gdf, maps_dir / "ccs_sites_points.geojson")

    ports_gdf = None
    if ports_df is not None and not ports_df.empty and "coordinate_valid" in ports_df.columns:
        valid_ports = ports_df[ports_df["coordinate_valid"]].copy()
        if not valid_ports.empty:
            ports_gdf = gpd.GeoDataFrame(
                valid_ports,
                geometry=gpd.points_from_xy(valid_ports["longitude"], valid_ports["latitude"]),
                crs="EPSG:4326",
            )
            _safe_to_file(ports_gdf, maps_dir / "ports_points.geojson")

    # Polygon data for choropleth maps.
    try:
        poly = _merge_selection_to_polygons(provinces_gdf, province_selection)

        # Save both tabular and spatial versions of the joined map layer for inspection.
        poly.drop(columns="geometry", errors="ignore").to_csv(
            maps_dir / "province_ccs_map_join_table.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _safe_to_file(poly, maps_dir / "province_ccs_map_joined.geojson")

        diagnostics_dir = out_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        join_diag = pd.DataFrame([
            {"metric": "spatial_polygon_rows", "value": len(poly)},
            {"metric": "ccs_joined_rows", "value": int(poly.get("ccs_valid_flag", pd.Series(dtype=object)).notna().sum()) if "ccs_valid_flag" in poly.columns else 0},
            {"metric": "ccs_valid_rows", "value": int(poly.get("ccs_valid_flag", pd.Series(dtype=bool)).fillna(False).sum()) if "ccs_valid_flag" in poly.columns else 0},
            {"metric": "missing_total_TandS_cost_rows", "value": int(pd.to_numeric(poly.get("total_TandS_cost_USD_tCO2", pd.Series(index=poly.index)), errors="coerce").isna().sum()) if "total_TandS_cost_USD_tCO2" in poly.columns else len(poly)},
            {"metric": "missing_selected_distance_rows", "value": int(pd.to_numeric(poly.get("selected_total_distance_km", pd.Series(index=poly.index)), errors="coerce").isna().sum()) if "selected_total_distance_km" in poly.columns else len(poly)},
        ])
        join_diag.to_csv(diagnostics_dir / "map_join_diagnostics_ccs.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        poly = None
        (fig_dir / "map_polygon_join_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Choropleth 1: total transport and storage cost.
    if poly is not None and "total_TandS_cost_USD_tCO2" in poly.columns:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            poly.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
            vals = pd.to_numeric(poly["total_TandS_cost_USD_tCO2"], errors="coerce")
            if vals.notna().any():
                poly.assign(_val=vals).plot(
                    column="_val", ax=ax, cmap="YlGnBu", legend=True,
                    edgecolor="white", linewidth=0.20,
                    legend_kwds={"label": "T&S cost (USD/tCO₂)", "shrink": 0.72},
                )
            if storage_gdf is not None and not storage_gdf.empty:
                storage_gdf.plot(ax=ax, markersize=34, marker="^", color=palette["orange"], edgecolor=palette["navy"], linewidth=0.3, label="Storage node")
            _finish_map(ax, "CCS transport and storage cost by province", palette)
            # Colorbar already explains the numeric layer; storage nodes are retained as context.
            fig.tight_layout()
            fig.savefig(fig_dir / "map_ccs_total_TandS_cost_choropleth.png", dpi=280, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            (fig_dir / "map_ccs_total_TandS_cost_choropleth_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Choropleth 2: selected route distance.
    if poly is not None and "selected_total_distance_km" in poly.columns:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            poly.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
            vals = pd.to_numeric(poly["selected_total_distance_km"], errors="coerce")
            if vals.notna().any():
                poly.assign(_val=vals).plot(
                    column="_val", ax=ax, cmap="PuBuGn", legend=True,
                    edgecolor="white", linewidth=0.20,
                    legend_kwds={"label": "Selected route distance (km)", "shrink": 0.72},
                )
            if storage_gdf is not None and not storage_gdf.empty:
                storage_gdf.plot(ax=ax, markersize=34, marker="^", color=palette["orange"], edgecolor=palette["navy"], linewidth=0.3, label="Storage node")
            _finish_map(ax, "Selected CCS route distance by province", palette)
            # Colorbar already explains the numeric layer; storage nodes are retained as context.
            fig.tight_layout()
            fig.savefig(fig_dir / "map_ccs_selected_route_distance_choropleth.png", dpi=280, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            (fig_dir / "map_ccs_selected_route_distance_choropleth_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Choropleth 3: selected transport mode.
    if poly is not None and "selected_transport_mode" in poly.columns:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            poly.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
            mode = poly["selected_transport_mode"].fillna("no_valid_ccs").astype(str)
            mode_simple = np.where(mode.str.contains("ship", case=False, na=False), "Ship/hybrid",
                           np.where(mode.str.contains("pipeline", case=False, na=False), "Pipeline", "No valid CCS"))
            plot_poly = poly.assign(_mode=mode_simple)
            color_map = {"Pipeline": palette["blue"], "Ship/hybrid": palette["orange"], "No valid CCS": palette["light_gray"]}
            for label, color in color_map.items():
                sub = plot_poly[plot_poly["_mode"] == label]
                if not sub.empty:
                    sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.20, label=label)
            if storage_gdf is not None and not storage_gdf.empty:
                storage_gdf.plot(ax=ax, markersize=32, marker="^", color=palette["navy"], edgecolor="white", linewidth=0.3, label="Storage node")
            _finish_map(ax, "Selected CCS transport mode by province", palette)
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "map_selected_transport_mode_by_province.png", dpi=280, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            (fig_dir / "map_selected_transport_mode_by_province_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Map 4: CCS storage sites only.
    try:
        fig, ax = plt.subplots(figsize=(11, 8))
        provinces_gdf.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
        if storage_gdf is not None and not storage_gdf.empty:
            storage_gdf.plot(ax=ax, markersize=44, marker="^", color=palette["orange"], edgecolor=palette["navy"], linewidth=0.4, label="CCS storage node")
        if ports_gdf is not None and not ports_gdf.empty:
            ports_gdf.plot(ax=ax, markersize=10, marker="s", color=palette["dark_gray"], alpha=0.45, label="Port")
        _finish_map(ax, "ASEAN CCS storage and port nodes", palette)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "map_ccs_sites_only.png", dpi=280, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        (fig_dir / "map_ccs_sites_only_ERROR.txt").write_text(str(exc), encoding="utf-8")

    # Map 5: selected DAC-to-CCS / DAC-port-port-CCS lines.
    try:
        line_gdf = make_lines_gdf(province_selection, max_lines=max_lines)
        if line_gdf is None or line_gdf.empty:
            (fig_dir / "NO_VALID_DAC_TO_CCS_LINES.txt").write_text(
                "No valid lines could be drawn. Check storage coordinates and province representative points.",
                encoding="utf-8",
            )
            return
        _safe_to_file(line_gdf, maps_dir / "dac_to_ccs_lines_selected.geojson")

        point_gdf = gpd.GeoDataFrame(
            line_gdf.drop(columns="geometry"),
            geometry=gpd.points_from_xy(line_gdf["province_rep_lon"], line_gdf["province_rep_lat"]),
            crs="EPSG:4326",
        )
        _safe_to_file(point_gdf, maps_dir / "dac_points_selected.geojson")

        fig, ax = plt.subplots(figsize=(12, 8))
        provinces_gdf.plot(ax=ax, color=palette["light_gray"], edgecolor="white", linewidth=0.25)
        if "selected_transport_mode" in line_gdf.columns:
            pipe = line_gdf[~line_gdf["selected_transport_mode"].astype(str).str.lower().str.contains("ship", na=False)]
            ship = line_gdf[line_gdf["selected_transport_mode"].astype(str).str.lower().str.contains("ship", na=False)]
            if not pipe.empty:
                pipe.plot(ax=ax, linewidth=1.0, alpha=0.78, color=palette["blue"], label="Pipeline route")
            if not ship.empty:
                ship.plot(ax=ax, linewidth=1.1, alpha=0.82, color=palette["orange"], label="Ship/hybrid route")
        else:
            line_gdf.plot(ax=ax, linewidth=0.9, alpha=0.7, color=palette["navy"], label="Selected route")

        point_gdf.plot(ax=ax, markersize=13, color=palette["blue"], edgecolor="white", linewidth=0.2, label="DAC representative point")
        if storage_gdf is not None and not storage_gdf.empty:
            storage_gdf.plot(ax=ax, markersize=42, marker="^", color=palette["orange"], edgecolor=palette["navy"], linewidth=0.35, label="CCS storage node")
        if ports_gdf is not None and not ports_gdf.empty:
            ports_gdf.plot(ax=ax, markersize=14, marker="s", color=palette["dark_gray"], alpha=0.55, label="Port")
        _finish_map(ax, "Selected DAC-to-CCS screening routes", palette)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "map_dac_to_ccs_selected_routes.png", dpi=280, bbox_inches="tight")
        plt.close(fig)

        line_gdf.drop(columns="geometry").to_csv(fig_dir / "map_dac_to_ccs_selected_route_records.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        (fig_dir / "map_dac_to_ccs_selected_routes_ERROR.txt").write_text(str(exc), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def build_outputs(args) -> None:
    root_dir = Path(args.root_dir)
    tea_dir = Path(args.tea_dir) if args.tea_dir else root_dir / "02.TEA_LCOD"
    out_dir = Path(args.out_dir) if args.out_dir else tea_dir / "04_CCS_EVALUATOR"
    resource_dir = Path(args.ccs_resource_dir) if args.ccs_resource_dir else root_dir / "00.RESOURCES" / "03_CCS"
    spatial_dir = Path(args.spatial_dir) if args.spatial_dir else root_dir / "00.SPATIAL_MAP"

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["inputs", "distances", "annual_results", "diagnostics", "figures", "maps"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    storage_file = find_existing_file(resource_dir, args.storage_filename, patterns=["*storage*EXPANDED*UPDATED*.csv", "*storage*UPDATED*.csv", "*storage*asean*.csv", "*storage*.csv"])
    cost_file = find_existing_file(resource_dir, args.cost_filename, patterns=["*cost*emission*UPDATED*.csv", "*cost*UPDATED*.csv", "*cost*emission*.csv", "*cost*.csv"])
    transport_asset_file = find_existing_file(resource_dir, args.transport_assets_filename, patterns=["*transport*assets*.csv", "*transport*.csv"])
    port_file = find_existing_file(resource_dir, args.port_filename, patterns=["*port*UPDATED*.csv", "*port*asean*.csv", "*ports*.csv", "*port*.csv"])
    if storage_file is None:
        raise FileNotFoundError(f"Could not find storage CSV in {resource_dir}")
    if cost_file is None:
        warnings.warn(f"Could not find CCS cost CSV in {resource_dir}. Cost columns may be NaN.")

    spatial_file = find_spatial_file(spatial_dir, Path(args.spatial_file) if args.spatial_file else None)

    if args.energy_summary_file:
        energy_file = Path(args.energy_summary_file)
    else:
        energy_file = tea_dir / "03_ENERGY_SUPPLY_EVALUATOR" / "annual_results" / ENERGY_SUMMARY_FILENAME
        if not energy_file.exists():
            # fallback if previous scripts were stored under 06.PYTHON/02.TEA_LCOD by mistake
            alt = root_dir / "06.PYTHON" / "02.TEA_LCOD" / "03_ENERGY_SUPPLY_EVALUATOR" / "annual_results" / ENERGY_SUMMARY_FILENAME
            if alt.exists():
                energy_file = alt

    print("=" * 100)
    print("04 CCS EVALUATOR - ROUTE, COST, AND MAP LAYER")
    print("=" * 100)
    print(f"Resource dir                       : {resource_dir}")
    print(f"Storage file                       : {storage_file}")
    print(f"Cost file                          : {cost_file if cost_file else 'not found'}")
    print(f"Port file                          : {port_file if port_file else 'not found'}")
    print(f"Transport asset file               : {transport_asset_file if transport_asset_file else 'not used / not found'}")
    print(f"Spatial file                       : {spatial_file}")
    print(f"Energy summary                     : {energy_file if energy_file.exists() else 'not found'}")
    print("=" * 100)

    storage_df = apply_storage_filter(load_storage_sites(storage_file), args.storage_filter_mode)
    cost_df = read_csv_flexible(cost_file) if cost_file and cost_file.exists() else None
    transport_assets_df = read_csv_flexible(transport_asset_file) if transport_asset_file and transport_asset_file.exists() else pd.DataFrame()
    ports_df = load_ports(port_file) if port_file is not None else load_ports(None)
    province_points, provinces_gdf = load_province_representative_points(spatial_file)
    # Province island may be supplied in the spatial file in future; otherwise inferred from nearest origin port.
    if "island_name" not in province_points.columns:
        province_points["island_name"] = ""
    energy_df = load_energy_summary(energy_file) if energy_file.exists() else None

    assumptions, assumptions_used = extract_cost_assumptions(
        cost_df=cost_df,
        eur_to_usd=args.eur_to_usd,
        offshore_pipeline_multiplier=args.offshore_pipeline_multiplier,
        ship_cost_usd_tco2_km=args.ship_cost_usd_tco2_km,
        ship_terminal_cost_usd_tco2=args.ship_terminal_cost_usd_tco2,
        ship_emission_tco2e_tco2_km=args.ship_emission_tco2e_tco2_km,
        ship_terminal_emission_tco2e_tco2=args.ship_terminal_emission_tco2e_tco2,
    )

    extra_assumptions = pd.DataFrame([
        {"parameter": "storage_filter_mode", "value": args.storage_filter_mode, "unit": "mode", "source": "CLI/default", "notes": "Internal maturity filter applied without changing storage CSV."},
        {"parameter": "different_island_pipeline_multiplier", "value": args.different_island_pipeline_multiplier, "unit": "dimensionless", "source": "CLI/default", "notes": "Additional multiplier for direct pipeline when DAC and storage are not on the same island/region."},
        {"parameter": "max_different_island_pipeline_km", "value": args.max_different_island_pipeline_km, "unit": "km", "source": "CLI/default", "notes": "If >0, direct different-island pipeline is infeasible above this straight-line distance."},
        {"parameter": "run_transport_sensitivities", "value": args.run_transport_sensitivities, "unit": "boolean", "source": "CLI/default", "notes": "Whether additional transport/storage sensitivity tables are written."},
    ])
    assumptions_used = pd.concat([assumptions_used, extra_assumptions], ignore_index=True)

    # Minimum data checks before route calculation.
    if int(storage_df["coordinate_valid"].sum()) == 0:
        raise ValueError("No CCS storage site has valid latitude/longitude. Check ccs_storage_sites_asean.csv.")
    if storage_df["island_key"].replace("", np.nan).isna().all():
        warnings.warn("All storage island_name values are missing. Same-island routing will rely only on port/storage fallbacks.")
    if ports_df.empty or int(ports_df.get("coordinate_valid", pd.Series(dtype=bool)).sum()) == 0:
        warnings.warn("No valid port coordinates found. Ship/hybrid routes will fall back to proxy_no_port_straightline.")

    province_candidates, province_selection = build_province_to_storage(
        province_points=province_points,
        storage_df=storage_df,
        assumptions=assumptions,
        ship_distance_multiplier=args.ship_distance_multiplier,
        ports_df=ports_df,
        storage_filter_mode=args.storage_filter_mode,
        different_island_pipeline_multiplier=args.different_island_pipeline_multiplier,
        max_different_island_pipeline_km=args.max_different_island_pipeline_km,
        force_ship_for_different_island=False,
        transport_sensitivity_case="main",
    )

    sensitivity_selection = pd.DataFrame()
    sensitivity_summary = pd.DataFrame()
    if args.run_transport_sensitivities:
        sensitivity_selection, sensitivity_summary = build_transport_sensitivity_outputs(
            province_points=province_points,
            storage_df=storage_df,
            assumptions=assumptions,
            ship_distance_multiplier=args.ship_distance_multiplier,
            ports_df=ports_df,
            args=args,
        )

    energy_ccs = merge_energy_ccs(energy_df, province_selection) if energy_df is not None else pd.DataFrame()

    # Transport mode comparison per province.
    comparison_cols = [
        "country_code", "country_name", "province_id", "province_name",
        "nearest_storage_site_id", "nearest_storage_site_name", "nearest_storage_country_code",
        "same_island_flag", "province_island_name", "storage_island_name",
        "straight_distance_km", "pipeline_effective_distance_km", "ship_hybrid_effective_distance_km",
        "land_distance_origin_km", "sea_distance_km", "land_distance_storage_km",
        "pipeline_transport_cost_USD_tCO2", "ship_hybrid_transport_cost_USD_tCO2",
        "selected_transport_mode", "selected_transport_cost_USD_tCO2", "selected_total_distance_km",
        "storage_cost_USD_tCO2", "mrv_cost_USD_tCO2", "total_TandS_cost_USD_tCO2",
        "cross_border_used", "is_offshore_storage", "storage_maturity_class", "storage_filter_mode",
        "pipeline_feasible_flag", "pipeline_infeasible_reason", "different_island_pipeline_multiplier_applied",
        "ship_minus_pipeline_cost_USD_tCO2", "ccs_valid_flag", "invalid_reason",
    ]
    transport_comparison = province_selection[[c for c in comparison_cols if c in province_selection.columns]].copy()
    if "ship_hybrid_transport_cost_USD_tCO2" in transport_comparison.columns and "pipeline_transport_cost_USD_tCO2" in transport_comparison.columns:
        transport_comparison["ship_minus_pipeline_cost_USD_tCO2"] = (
            transport_comparison["ship_hybrid_transport_cost_USD_tCO2"] - transport_comparison["pipeline_transport_cost_USD_tCO2"]
        )

    # Diagnostics.
    diagnostics = pd.DataFrame([
        {"metric": "spatial_file_primary_expected", "value": str(SPATIAL_GPKG_DEFAULT)},
        {"metric": "spatial_file_used", "value": str(spatial_file)},
        {"metric": "storage_file", "value": str(storage_file)},
        {"metric": "cost_file", "value": str(cost_file) if cost_file else "not_found"},
        {"metric": "transport_asset_file", "value": str(transport_asset_file) if transport_asset_file else "not_found"},
        {"metric": "port_file", "value": str(port_file) if port_file else "not_found"},
        {"metric": "spatial_file", "value": str(spatial_file)},
        {"metric": "energy_summary_file", "value": str(energy_file) if energy_file.exists() else "not_found"},
        {"metric": "n_storage_sites_total", "value": len(storage_df)},
        {"metric": "n_storage_sites_with_valid_coordinates", "value": int(storage_df["coordinate_valid"].sum())},
        {"metric": "storage_filter_mode", "value": args.storage_filter_mode},
        {"metric": "n_storage_sites_usable_under_filter", "value": int(storage_df["storage_filter_usable_flag"].fillna(False).sum())},
        {"metric": "different_island_pipeline_multiplier", "value": args.different_island_pipeline_multiplier},
        {"metric": "max_different_island_pipeline_km", "value": args.max_different_island_pipeline_km},
        {"metric": "n_ports_total", "value": len(ports_df)},
        {"metric": "n_ports_with_valid_coordinates", "value": int(ports_df["coordinate_valid"].sum()) if not ports_df.empty else 0},
        {"metric": "n_provinces", "value": len(province_points)},
        {"metric": "n_provinces_ccs_valid", "value": int(province_selection["ccs_valid_flag"].fillna(False).sum())},
        {"metric": "n_provinces_cross_border_used", "value": int(province_selection["cross_border_used"].fillna(False).sum())},
        {"metric": "n_provinces_domestic_storage_available", "value": int(province_selection["domestic_storage_available"].fillna(False).sum())},
        {"metric": "n_provinces_domestic_storage_usable", "value": int(province_selection["domestic_storage_usable"].fillna(False).sum())},
        {"metric": "ship_cost_available", "value": bool(np.isfinite(assumptions["ship_cost_usd_tco2_km"]) and np.isfinite(assumptions["ship_terminal_cost_usd_tco2"]))},
        {"metric": "ship_assumption_quality", "value": assumptions["ship_assumption_quality"]},
    ])

    # Diagnostics for module 03 hourly-demand passthrough.
    energy_profile_diag = make_energy_profile_passthrough_diagnostics(energy_df, energy_ccs)
    if not energy_profile_diag.empty:
        extra_diag = energy_profile_diag.copy()
        extra_diag["metric"] = "energy_profile_passthrough::" + extra_diag["metric"].astype(str)
        diagnostics = pd.concat([diagnostics, extra_diag], ignore_index=True)

    # Save inputs and outputs.
    storage_df.to_csv(out_dir / "inputs" / "ccs_storage_sites_used.csv", index=False, encoding="utf-8-sig")
    if cost_df is not None:
        cost_df.to_csv(out_dir / "inputs" / "ccs_cost_emission_assumptions_input.csv", index=False, encoding="utf-8-sig")
    if not transport_assets_df.empty:
        transport_assets_df.to_csv(out_dir / "inputs" / "ccs_transport_assets_input.csv", index=False, encoding="utf-8-sig")
    if not ports_df.empty:
        ports_df.to_csv(out_dir / "inputs" / "ports_used.csv", index=False, encoding="utf-8-sig")
        ports_df.to_csv(out_dir / "maps" / "ports_points.csv", index=False, encoding="utf-8-sig")
    province_points.to_csv(out_dir / "inputs" / "province_representative_points.csv", index=False, encoding="utf-8-sig")
    province_points.to_csv(out_dir / "maps" / "dac_points.csv", index=False, encoding="utf-8-sig")
    storage_df.to_csv(out_dir / "maps" / "ccs_sites_points.csv", index=False, encoding="utf-8-sig")
    assumptions_used.to_csv(out_dir / "inputs" / "ccs_cost_assumptions_used.csv", index=False, encoding="utf-8-sig")
    province_candidates.to_csv(out_dir / "distances" / "province_to_storage_all_candidates.csv", index=False, encoding="utf-8-sig")
    province_selection.to_csv(out_dir / "distances" / "province_to_storage_selected.csv", index=False, encoding="utf-8-sig")
    # Backward-compatible filename for older downstream scripts.
    province_selection.to_csv(out_dir / "distances" / "province_to_storage_selection.csv", index=False, encoding="utf-8-sig")
    transport_comparison.to_csv(out_dir / "distances" / "transport_mode_comparison.csv", index=False, encoding="utf-8-sig")
    if not energy_ccs.empty:
        energy_ccs.to_csv(out_dir / "annual_results" / "energy_ccs_summary_by_province_policy_scenario.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / "diagnostics" / "ccs_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not energy_profile_diag.empty:
        energy_profile_diag.to_csv(out_dir / "diagnostics" / "energy_profile_passthrough_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not province_selection.empty:
        province_selection.groupby(["selected_transport_mode"], dropna=False).size().reset_index(name="n_provinces").to_csv(
            out_dir / "diagnostics" / "transport_mode_summary.csv", index=False, encoding="utf-8-sig"
        )
        province_selection.groupby(["country_code", "selected_transport_mode"], dropna=False).size().reset_index(name="n_provinces").to_csv(
            out_dir / "diagnostics" / "transport_mode_summary_by_country.csv", index=False, encoding="utf-8-sig"
        )
        province_selection.groupby(["same_island_flag"], dropna=False).size().reset_index(name="n_provinces").to_csv(
            out_dir / "diagnostics" / "same_island_rule_summary.csv", index=False, encoding="utf-8-sig"
        )
        province_selection.groupby(["cross_border_used"], dropna=False).size().reset_index(name="n_provinces").to_csv(
            out_dir / "diagnostics" / "crossborder_summary.csv", index=False, encoding="utf-8-sig"
        )
        if "ship_minus_pipeline_cost_USD_tCO2" in province_selection.columns:
            province_selection[[c for c in [
                "country_code", "country_name", "province_id", "province_name", "selected_transport_mode",
                "same_island_flag", "different_island_flag", "pipeline_feasible_flag", "pipeline_infeasible_reason",
                "pipeline_transport_cost_USD_tCO2", "ship_hybrid_transport_cost_USD_tCO2",
                "ship_minus_pipeline_cost_USD_tCO2", "straight_distance_km", "selected_total_distance_km",
            ] if c in province_selection.columns]].to_csv(
                out_dir / "diagnostics" / "ship_pipeline_cost_gap_by_province.csv", index=False, encoding="utf-8-sig"
            )

    maturity_summary, filter_country_summary = summarize_storage_filter(storage_df, args.storage_filter_mode)
    maturity_summary.to_csv(out_dir / "diagnostics" / "storage_maturity_summary.csv", index=False, encoding="utf-8-sig")
    filter_country_summary.to_csv(out_dir / "diagnostics" / "storage_filter_summary_by_country.csv", index=False, encoding="utf-8-sig")

    if not sensitivity_selection.empty:
        sensitivity_selection.to_csv(out_dir / "distances" / "province_to_storage_selected_sensitivity.csv", index=False, encoding="utf-8-sig")
        sens_cols = [c for c in comparison_cols + ["transport_sensitivity_case", "storage_filter_mode", "pipeline_feasible_flag", "pipeline_infeasible_reason", "ship_minus_pipeline_cost_USD_tCO2"] if c in sensitivity_selection.columns]
        sensitivity_selection[sens_cols].to_csv(out_dir / "distances" / "transport_mode_comparison_sensitivity.csv", index=False, encoding="utf-8-sig")
    if not sensitivity_summary.empty:
        sensitivity_summary.to_csv(out_dir / "diagnostics" / "transport_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "root_dir": str(root_dir),
        "tea_dir": str(tea_dir),
        "out_dir": str(out_dir),
        "resource_dir": str(resource_dir),
        "spatial_dir": str(spatial_dir),
        "storage_file": str(storage_file),
        "cost_file": str(cost_file) if cost_file else None,
        "transport_asset_file": str(transport_asset_file) if transport_asset_file else None,
        "port_file": str(port_file) if port_file else None,
        "spatial_file": str(spatial_file),
        "energy_summary_file": str(energy_file) if energy_file.exists() else None,
        "method": {
            "dac_point": "province representative point",
            "storage_point": "storage site coordinate / basin centroid / project coordinate from CSV",
            "domestic_rule": "domestic usable storage first under selected storage_filter_mode; cross-border ASEAN fallback only if no domestic storage is usable under that filter",
            "pipeline_distance": "straight-line geodesic with offshore multiplier and additional different-island pipeline multiplier/optional distance cap",
            "ship_distance": "DAC-to-origin-port + straight port-to-port sea distance + destination-port-to-storage; proxy straight-line if port data missing",
            "mode_selection": "same island forced pipeline; different island compares feasible direct pipeline vs ship/hybrid when costs are available",
            "storage_filter_mode": args.storage_filter_mode,
            "different_island_pipeline_multiplier": args.different_island_pipeline_multiplier,
            "max_different_island_pipeline_km": args.max_different_island_pipeline_km,
            "run_transport_sensitivities": args.run_transport_sensitivities,
            "module03_hourly_profile_columns_preserved": True,
            "module03_hourly_profile_column_list": ENERGY_PROFILE_PASSTHROUGH_COLS,
        },
        "assumptions": assumptions,
        "ship_distance_multiplier": args.ship_distance_multiplier,
        "figure_max_lines": args.figure_max_lines,
    }
    write_json(out_dir / "ccs_config_used.json", config)

    # Figures.
    if args.make_figures:
        make_figures(out_dir, provinces_gdf, province_selection, storage_df, ports_df, max_lines=args.figure_max_lines)

    # README.
    readme = f"""04_CCS_EVALUATOR output

Purpose:
Add screening-level CO2 transport and geological storage distance/cost/emission metrics to the output of 03_ENERGY_SUPPLY_EVALUATOR.

Method:
- DAC point: province representative point.
- Storage point: latitude/longitude from storage CSV.
- Domestic usable storage is prioritized.
- Cross-border ASEAN storage is allowed only when no usable domestic storage coordinate exists.
- Same island: pipeline is forced.
- Different island: direct pipeline and ship/hybrid are compared when cost assumptions are available.
- Pipeline: straight-line geodesic distance, with offshore multiplier = {args.offshore_pipeline_multiplier}.
- Different-island pipeline: additional multiplier = {args.different_island_pipeline_multiplier}; max distance cap = {args.max_different_island_pipeline_km} km where >0.
- Storage filter mode: {args.storage_filter_mode}. This is inferred internally from existing CSV text fields; the CSV is not modified.
- Module 03 hourly-demand/profile columns are preserved and passed through to annual_results/energy_ccs_summary_by_province_policy_scenario.csv for module 05.
- Ship/hybrid: DAC point -> nearest origin commercial port -> nearest destination commercial port -> storage point. If ports are unavailable, proxy/no-port mode is flagged.

Key outputs:
- distances/province_to_storage_all_candidates.csv
- distances/province_to_storage_selected.csv
- distances/province_to_storage_selection.csv
- distances/transport_mode_comparison.csv
- annual_results/energy_ccs_summary_by_province_policy_scenario.csv
- diagnostics/ccs_diagnostics.csv
- diagnostics/energy_profile_passthrough_diagnostics.csv
- inputs/ccs_cost_assumptions_used.csv
- figures/map_ccs_sites_only.png and figures/map_dac_to_ccs_selected_routes.png, if coordinates are available

Important limitation:
Straight-line links are screening distances, not engineered pipeline or shipping routes. Port-aware ship/hybrid routing uses nearest-port and straight port-to-port distances for screening.
"""
    (out_dir / "README_04_CCS_EVALUATOR.txt").write_text(readme, encoding="utf-8")

    print("=" * 92)
    print("04 CCS EVALUATOR COMPLETE")
    print("=" * 92)
    print(f"Output dir                         : {out_dir}")
    print(f"Storage file                       : {storage_file}")
    print(f"Spatial file                       : {spatial_file}")
    print(f"Port file                          : {port_file if port_file else 'not found'}")
    print(f"Energy summary                     : {energy_file if energy_file.exists() else 'not found'}")
    print(f"Storage sites total                : {len(storage_df)}")
    print(f"Storage sites with valid coords    : {int(storage_df['coordinate_valid'].sum())}")
    print(f"Storage sites usable under filter  : {int(storage_df['storage_filter_usable_flag'].fillna(False).sum())}")
    print(f"Storage filter mode                : {args.storage_filter_mode}")
    print(f"Different-island pipeline mult     : {args.different_island_pipeline_multiplier}")
    print(f"Different-island pipeline cap km   : {args.max_different_island_pipeline_km}")
    print(f"Provinces                          : {len(province_points)}")
    print(f"Valid province CCS rows            : {int(province_selection['ccs_valid_flag'].fillna(False).sum())}")
    print(f"Cross-border fallback used         : {int(province_selection['cross_border_used'].fillna(False).sum())}")
    print(f"Ship assumption quality            : {assumptions['ship_assumption_quality']}")
    print("-" * 92)
    print(f"Saved: {out_dir / 'distances' / 'province_to_storage_all_candidates.csv'}")
    print(f"Saved: {out_dir / 'distances' / 'province_to_storage_selected.csv'}")
    print(f"Saved: {out_dir / 'distances' / 'province_to_storage_selection.csv'}")
    print(f"Saved: {out_dir / 'distances' / 'transport_mode_comparison.csv'}")
    if not energy_ccs.empty:
        print(f"Saved: {out_dir / 'annual_results' / 'energy_ccs_summary_by_province_policy_scenario.csv'}")
        print(f"Saved: {out_dir / 'diagnostics' / 'energy_profile_passthrough_diagnostics.csv'}")
    else:
        print("Annual energy+CCS output not created because 03 energy summary was not found or empty.")
    print("=" * 92)


def parse_args():
    parser = argparse.ArgumentParser(description="Screening-level CCS transport and storage evaluator for ASEAN DACCS.")
    parser.add_argument("--root-dir", default=r"D:/Ashka/5.DAC", help="Root project folder, e.g. D:/Ashka/5.DAC")
    parser.add_argument("--tea-dir", default=None, help="Path to 02.TEA_LCOD. Default: root-dir/02.TEA_LCOD")
    parser.add_argument("--out-dir", default=None, help="Output folder. Default: tea-dir/04_CCS_EVALUATOR")
    parser.add_argument("--ccs-resource-dir", default=None, help="CCS resource folder. Default: root-dir/00.RESOURCES/03_CCS")
    parser.add_argument("--spatial-dir", default=None, help="Spatial map folder. Default: root-dir/00.SPATIAL_MAP")
    parser.add_argument("--spatial-file", default=None, help="Explicit province GPKG path. Optional.")
    parser.add_argument("--energy-summary-file", default=None, help="Explicit output CSV from 03_ENERGY_SUPPLY_EVALUATOR. Optional.")

    parser.add_argument("--storage-filename", default="ccs_storage_sites_asean.csv")
    parser.add_argument("--cost-filename", default="ccs_cost_emission_assumptions.csv")
    parser.add_argument("--transport-assets-filename", default="ccs_transport_assets_asean.csv")
    parser.add_argument("--port-filename", default=PORT_FILENAME_DEFAULT, help="Port CSV filename in CCS resource folder.")

    parser.add_argument("--eur-to-usd", type=float, default=1.08, help="EUR-to-USD conversion used for EUR-denominated CCS cost rows.")
    parser.add_argument("--offshore-pipeline-multiplier", type=float, default=1.25, help="Multiplier applied to straight-line pipeline distance if storage site is offshore.")
    parser.add_argument("--ship-distance-multiplier", type=float, default=1.0, help="Multiplier applied to straight-line ship proxy distance.")

    # Ship values are optional because no ship cost data may exist yet.
    parser.add_argument("--ship-cost-usd-tco2-km", type=float, default=None, help="Optional ship transport cost in USD/tCO2/km. If omitted, ship cost is NaN unless present in CCS cost CSV.")
    parser.add_argument("--ship-terminal-cost-usd-tco2", type=float, default=None, help="Optional ship loading/unloading/terminal cost in USD/tCO2. Required for cost comparison if ship cost is used.")
    parser.add_argument("--ship-emission-tco2e-tco2-km", type=float, default=None, help="Optional ship transport emission factor in tCO2e/tCO2/km.")
    parser.add_argument("--ship-terminal-emission-tco2e-tco2", type=float, default=0.0, help="Optional terminal emission in tCO2e/tCO2.")

    parser.add_argument("--storage-filter-mode", default="all_assessed_storage", choices=["all_assessed_storage", "project_or_high_confidence_storage_only", "project_only"], help="Internal storage maturity filter inferred from existing storage CSV text fields; CSV is not modified.")
    parser.add_argument("--different-island-pipeline-multiplier", type=float, default=2.5, help="Additional multiplier applied to direct pipeline distance when DAC and storage are not on the same island/region.")
    parser.add_argument("--max-different-island-pipeline-km", type=float, default=750.0, help="If >0, direct different-island pipeline is infeasible above this straight-line distance. Ship/hybrid can still be selected.")
    parser.add_argument("--run-transport-sensitivities", action="store_true", default=True, help="Write additional sensitivity tables for ship/pipeline and storage maturity filters.")
    parser.add_argument("--no-transport-sensitivities", dest="run_transport_sensitivities", action="store_false", help="Disable additional transport/storage sensitivity tables.")
    parser.add_argument("--sensitivity-high-interisland-pipeline-multiplier", type=float, default=5.0, help="Inter-island pipeline multiplier used in the high-penalty sensitivity case.")

    parser.add_argument("--make-figures", action="store_true", default=True, help="Create simple map figures if geopandas/matplotlib and coordinates are available.")
    parser.add_argument("--no-figures", dest="make_figures", action="store_false", help="Disable figure creation.")
    parser.add_argument("--figure-max-lines", type=int, default=30, help="Maximum example province-to-storage lines to draw.")

    return parser.parse_args()


if __name__ == "__main__":
    build_outputs(parse_args())
