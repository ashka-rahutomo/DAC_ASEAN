from __future__ import annotations

"""
02_annual_dynamic_operation.py

Annual dynamic-operation evaluator for the DAC TVSA cycle surrogate.

Purpose
-------
This script uses the cycle-level ANN surrogate trained in:

    02.TEA_LCOD/01_CYCLE_SURROGATE/

and hourly/province DAC climate input to estimate annual process-side DAC
performance under several operation policies:

    O0_Avg_static_reference : one fixed operation per province-year
    O1_lookup_table_LT      : global climate-bin lookup table
    O2_day_night            : one operation for day and one for night per province
    O3_seasonal_wet_dry     : one operation for wet-like and one for dry-like season
    O4_adaptive_upper_bound : province-specific climate-bin adaptive upper bound

This script only evaluates process-side annual demand and productivity:

    annual CO2, H2O, heat, cooling, electricity, SEC, and selected operations.

It does NOT perform renewable dispatch, CCS, LCOD, or net-removal calculations.
Those should be handled downstream in 03_ENERGY_SUPPLY_EVALUATOR and later modules.

Default folder layout expected by the user:

    D:/Ashka/5.DAC/06.PYTHON/
    └─ 02.TEA_LCOD/
       ├─ 00_CYCLE_KPI/
       ├─ 01_CYCLE_SURROGATE/
       └─ 02_ANNUAL_DYNAMIC_OPERATION/

The script is intentionally robust: if fewer than 3000 KPI rows exist or if only
one ANN run is available, it still runs.
"""

from pathlib import Path
import argparse
import json
import math
import pickle
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as exc:  # pragma: no cover
    tf = None
    load_model = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


# =============================================================================
# Canonical columns
# =============================================================================

ID_COLS = ["country_code", "country_name", "province_id", "province_name"]
COORD_COLS = ["longitude", "latitude"]
TIME_COLS = ["datetime_utc", "year", "month", "day", "hour_utc"]

CANONICAL_FEATURES = [
    "T_ads_K",
    "RH_frac",
    "P_Pa",
    "p_H2O_Pa",
    "p_CO2_Pa",
    "q_H2O_GAB_mol_kg",
    "q_CO2_WADST_mol_kg",
    "adsorption_time_s",
    "heating_desorption_time_s",
    "T_des_K",
    "T_coolant_K",
]

CANONICAL_TARGETS = [
    "kg_CO2_cycle_corrected",
    "kg_H2O_cycle_corrected",
    "Q_heat_kWhth_cycle",
    "E_total_el_kWhe_cycle",
    "Q_cool_kWhth_cycle",
    "E_fan_kWhe_cycle",
    "E_vacuum_kWhe_cycle",
    "E_repress_kWhe_cycle",
    "E_chiller_kWhe_cycle",
]

# Output names sometimes saved by 05a before corrected renaming.
TARGET_ALIASES = {
    "kg_CO2_cycle_corrected": ["kg_CO2_cycle_corrected", "kg_CO2_cycle"],
    "kg_H2O_cycle_corrected": ["kg_H2O_cycle_corrected", "kg_H2O_cycle"],
    "Q_heat_kWhth_cycle": ["Q_heat_kWhth_cycle"],
    "E_total_el_kWhe_cycle": ["E_total_el_kWhe_cycle"],
    "Q_cool_kWhth_cycle": ["Q_cool_kWhth_cycle"],
    "E_fan_kWhe_cycle": ["E_fan_kWhe_cycle"],
    "E_vacuum_kWhe_cycle": ["E_vacuum_kWhe_cycle"],
    "E_repress_kWhe_cycle": ["E_repress_kWhe_cycle"],
    "E_chiller_kWhe_cycle": ["E_chiller_kWhe_cycle"],
}

OPERATION_COLS = [
    "adsorption_time_s",
    "heating_desorption_time_s",
    "T_des_K",
    "T_coolant_K",
]

OPERATION_ALIASES = {
    "adsorption_time_s": [
        "adsorption_time_s", "adsorption_time_s_x", "adsorption_time_s_y",
        "t_ads_s", "ads_time_s", "ads_time", "t_ads", "adsorption_time",
        "adsorption_s", "t_adsorption_s", "t_adsorption",
    ],
    "heating_desorption_time_s": [
        "heating_desorption_time_s", "heating_desorption_time_s_x", "heating_desorption_time_s_y",
        "desorption_time_s", "desorption_time_s_x", "desorption_time_s_y",
        "t_des_s", "heating_time_s", "des_time_s", "des_time", "t_des",
        "desorption_time", "desorption_s", "heating_desorption_s", "t_heating_s",
    ],
    "T_des_K": [
        "T_des_K", "T_des_K_x", "T_des_K_y",
        "desorption_temperature_K", "T_heating_K", "T_hot_K", "T_regen_K",
    ],
    "T_coolant_K": [
        "T_coolant_K", "T_coolant_K_x", "T_coolant_K_y",
        "cooling_temperature_K", "T_cool_K", "T_coolant", "T_cooling_K",
    ],
}

# Some DAC hourly and 05a KPI files use these already; keep suffix aliases for robustness.
FEATURE_ALIASES = {
    "T_ads_K": ["T_ads_K", "T_ads_K_x", "T_ads_K_y", "T_K", "T_amb_K", "T_amb_K_x", "T_amb_K_y"],
    "RH_frac": ["RH_frac", "RH_frac_x", "RH_frac_y", "RH", "relative_humidity_frac"],
    "P_Pa": ["P_Pa", "P_Pa_x", "P_Pa_y", "P_amb_Pa", "P_amb_Pa_x", "P_amb_Pa_y", "pressure_Pa"],
    "p_H2O_Pa": ["p_H2O_Pa", "p_H2O_Pa_x", "p_H2O_Pa_y", "pH2O_Pa"],
    "p_CO2_Pa": ["p_CO2_Pa", "p_CO2_Pa_x", "p_CO2_Pa_y", "pCO2_Pa"],
    "q_H2O_GAB_mol_kg": ["q_H2O_GAB_mol_kg", "q_H2O_GAB_mol_kg_x", "q_H2O_GAB_mol_kg_y", "q_H2O_mol_kg"],
    "q_CO2_WADST_mol_kg": ["q_CO2_WADST_mol_kg", "q_CO2_WADST_mol_kg_x", "q_CO2_WADST_mol_kg_y", "q_CO2_mol_kg"],
}


# =============================================================================
# Utility functions
# =============================================================================

def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        val = float(x)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def sanitize_tag(value: float | int | str) -> str:
    s = str(value)
    s = s.replace(".", "p").replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.strip("_")


def choose_existing_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_tea_dir(project_dir: Path, user_tea_dir: str | None) -> Path:
    if user_tea_dir:
        return Path(user_tea_dir)

    candidates = [
        project_dir / "02.TEA_LCOD",
        project_dir.parent / "02.TEA_LCOD",
    ]
    found = choose_existing_dir(candidates)
    return found if found is not None else candidates[0]


def find_latest_file(folder: Path, patterns: list[str]) -> Path:
    """Return the newest suitable file from prioritized patterns.

    Excludes diagnostics/summary files so the annual-operation module does not
    accidentally use 05a_ann_ready_column_diagnostics or other non-KPI tables.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    banned_tokens = [
        "diagnostics",
        "column_diagnostics",
        "kpi_summary",
        "status_summary",
        "row_status",
        "missing_or_not_success",
        "design_csv_candidates",
        "case_inventory",
        "readme",
    ]

    searched = []
    for pat in patterns:
        matches = []
        for p in folder.glob(pat):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(tok in name for tok in banned_tokens):
                continue
            matches.append(p)

        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        searched.append(pat)

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No suitable KPI file found in {folder}. "
        f"Searched patterns: {searched}. Diagnostics/summary files were excluded."
    )

def find_latest_surrogate_run(surrogate_dir: Path) -> Path:
    """Return a folder containing model + scalers + metadata."""
    if not surrogate_dir.exists():
        raise FileNotFoundError(f"Surrogate dir not found: {surrogate_dir}")

    def has_required(p: Path) -> bool:
        return (
            (p / "surrogate_metadata.json").exists()
            and (p / "x_scaler.pkl").exists()
            and (p / "y_scaler.pkl").exists()
            and ((p / "best_ann_model.keras").exists() or (p / "best_ann_model.h5").exists())
        )

    if has_required(surrogate_dir):
        return surrogate_dir

    candidates = [p for p in surrogate_dir.iterdir() if p.is_dir() and has_required(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No ANN surrogate run folder found in {surrogate_dir}. "
            "Expected surrogate_metadata.json, x_scaler.pkl, y_scaler.pkl, and best_ann_model.keras/h5."
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_pickle_or_joblib(path: Path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_column(df: pd.DataFrame, canonical: str, aliases: dict[str, list[str]]) -> str | None:
    """Return the existing alias with the highest non-null numeric count.

    This avoids selecting sparse *_x columns when *_y or fallback columns are complete.
    """
    candidates = [c for c in aliases.get(canonical, [canonical]) if c in df.columns]
    if not candidates:
        return None
    counts = {}
    for c in candidates:
        counts[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
    order = {c: i for i, c in enumerate(candidates)}
    return sorted(candidates, key=lambda c: (-counts[c], order[c]))[0]


def canonicalize_operation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OPERATION_COLS:
        source = resolve_column(out, col, OPERATION_ALIASES)
        if source is None:
            raise ValueError(
                f"Operation column '{col}' was not found. Tried aliases: {OPERATION_ALIASES.get(col)}"
            )
        out[col] = pd.to_numeric(out[source], errors="coerce")
    return out


def canonicalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CANONICAL_FEATURES:
        if col in OPERATION_COLS:
            continue
        source = resolve_column(out, col, FEATURE_ALIASES)
        if source is None:
            raise ValueError(
                f"Feature column '{col}' was not found. Tried aliases: {FEATURE_ALIASES.get(col)}"
            )
        out[col] = pd.to_numeric(out[source], errors="coerce")
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.sum(vals[mask] * w[mask]) / np.sum(w[mask]))


def weighted_aggregate(df: pd.DataFrame, group_cols: list[str], value_cols: list[str], weight_col: str = "n_hours") -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec[weight_col] = float(pd.to_numeric(sub[weight_col], errors="coerce").fillna(0).sum())
        for col in value_cols:
            rec[col] = weighted_mean(sub[col], sub[weight_col])
        rows.append(rec)
    return pd.DataFrame(rows)


def infer_dac_hourly_csv(project_dir: Path, year: int) -> Path:
    candidates = [
        project_dir.parent / "00.TEMPORAL_DATA" / "DAC_HOURLY_INPUT",
        project_dir / "00.TEMPORAL_DATA" / "DAC_HOURLY_INPUT",
        project_dir.parent.parent / "00.TEMPORAL_DATA" / "DAC_HOURLY_INPUT",
    ]
    for folder in candidates:
        if folder.exists():
            files = sorted(folder.glob(f"DAC_hourly_input_WADST_GAB_CO2_*ppm_{year}.csv"))
            if files:
                return files[-1]
    raise FileNotFoundError(
        "Could not infer DAC hourly input CSV. Pass --dac-hourly-csv explicitly."
    )


# =============================================================================
# Surrogate wrapper
# =============================================================================

class CycleSurrogate:
    def __init__(self, run_dir: Path):
        if load_model is None:
            raise ImportError(
                "TensorFlow/Keras could not be imported. Install tensorflow first. "
                f"Original error: {_TF_IMPORT_ERROR}"
            )
        self.run_dir = run_dir
        self.metadata = json.loads((run_dir / "surrogate_metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = self.metadata.get("feature_columns") or self.metadata.get("input_features") or CANONICAL_FEATURES
        self.target_columns = self.metadata.get("target_columns") or self.metadata.get("output_features") or CANONICAL_TARGETS
        self.eps = float(self.metadata.get("target_log_epsilon", 1e-12))

        model_path = run_dir / "best_ann_model.keras"
        if not model_path.exists():
            model_path = run_dir / "best_ann_model.h5"
        self.model_path = model_path
        self.model = load_model(model_path, compile=False)
        self.x_scaler = load_pickle_or_joblib(run_dir / "x_scaler.pkl")
        self.y_scaler = load_pickle_or_joblib(run_dir / "y_scaler.pkl")

    def predict(self, df: pd.DataFrame, batch_size: int = 8192) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Surrogate input is missing feature columns: {missing}")
        X = df[self.feature_columns].astype("float64").to_numpy()
        Xs = self.x_scaler.transform(X)
        y_scaled = self.model.predict(Xs, batch_size=batch_size, verbose=0)
        y_log = self.y_scaler.inverse_transform(y_scaled)
        y = np.exp(y_log) - self.eps
        y = np.maximum(y, 0.0)
        out = pd.DataFrame(y, columns=self.target_columns, index=df.index)
        return out


# =============================================================================
# Candidate operation loading
# =============================================================================

def load_operation_candidates(kpi_csv: Path, max_candidates: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(kpi_csv)
    try:
        df = canonicalize_operation_columns(df)
    except ValueError as exc:
        similar = [c for c in df.columns if any(k in c.lower() for k in ["ads", "des", "time", "cool", "heat", "design"])]
        raise ValueError(
            f"Failed to read operation candidates from KPI CSV: {kpi_csv}\n"
            "This script needs the 05a SUCCESS KPI WITH DESIGN file, not the KPI summary file.\n"
            f"Original error: {exc}\n"
            f"Similar columns found: {similar}"
        ) from exc

    # Drop rows without complete operation values.
    ops = df[OPERATION_COLS].copy()
    for col in OPERATION_COLS:
        ops[col] = pd.to_numeric(ops[col], errors="coerce")
    valid = ops[OPERATION_COLS].notna().all(axis=1)
    df = df.loc[valid].copy()
    ops = ops.loc[valid].copy()

    # Keep unique operation settings.
    ops = ops.drop_duplicates().reset_index(drop=True)

    if len(ops) == 0:
        raise ValueError("No valid operation candidates found from KPI CSV.")

    if max_candidates is None or max_candidates <= 0 or len(ops) <= max_candidates:
        ops = ops.copy()
        ops.insert(0, "operation_id", np.arange(1, len(ops) + 1))
        return ops

    # If KPI metrics are available, prioritize globally useful candidates but keep diversity.
    df_unique = df.drop_duplicates(subset=OPERATION_COLS).copy()
    if len(df_unique) > len(ops):
        df_unique = df_unique.iloc[: len(ops)].copy()

    selected_idx: set[int] = set()
    n_each = max(1, max_candidates // 4)

    def add_top(metric: str, ascending: bool, n: int):
        if metric in df_unique.columns:
            s = pd.to_numeric(df_unique[metric], errors="coerce")
            ranked = s.sort_values(ascending=ascending).dropna().index.tolist()
            for idx in ranked[:n]:
                selected_idx.add(int(idx))

    add_top("productivity_kgCO2_kgads_year", ascending=False, n=n_each)
    add_top("annual_tCO2_per_1000kgads", ascending=False, n=n_each)
    add_top("specific_total_bed_MWh_tCO2_before_compression", ascending=True, n=n_each)
    add_top("H2O_CO2_mass_ratio_tH2O_tCO2", ascending=True, n=n_each)

    rng = np.random.default_rng(random_state)
    remaining = [i for i in df_unique.index.tolist() if int(i) not in selected_idx]
    if len(selected_idx) < max_candidates and remaining:
        add_n = min(max_candidates - len(selected_idx), len(remaining))
        selected_idx.update([int(i) for i in rng.choice(remaining, size=add_n, replace=False)])

    selected_idx = sorted(list(selected_idx))[:max_candidates]
    selected_ops = df_unique.loc[selected_idx, OPERATION_COLS].drop_duplicates().reset_index(drop=True)

    # If drop_duplicates reduced count too much, random fill from ops.
    if len(selected_ops) < max_candidates:
        merged = ops.merge(selected_ops, on=OPERATION_COLS, how="left", indicator=True)
        pool = merged[merged["_merge"] == "left_only"][OPERATION_COLS]
        if len(pool) > 0:
            fill_n = min(max_candidates - len(selected_ops), len(pool))
            fill = pool.sample(n=fill_n, random_state=random_state)
            selected_ops = pd.concat([selected_ops, fill], ignore_index=True).drop_duplicates().reset_index(drop=True)

    selected_ops.insert(0, "operation_id", np.arange(1, len(selected_ops) + 1))
    return selected_ops


# =============================================================================
# Climate binning
# =============================================================================

def create_bins_for_chunk(chunk: pd.DataFrame, args) -> pd.DataFrame:
    df = chunk.copy()
    df = canonicalize_feature_columns(df)

    # Numeric safety.
    for col in [*CANONICAL_FEATURES[:7], "longitude", "latitude", "month", "hour_utc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "T_C" in df.columns:
        T_C = pd.to_numeric(df["T_C"], errors="coerce")
    else:
        T_C = pd.to_numeric(df["T_ads_K"], errors="coerce") - 273.15
    RH_frac = pd.to_numeric(df["RH_frac"], errors="coerce")
    P_kPa = pd.to_numeric(df["P_Pa"], errors="coerce") / 1000.0

    df["T_bin"] = np.floor(T_C / args.t_bin_c) * args.t_bin_c
    df["RH_bin"] = np.floor(RH_frac / args.rh_bin_frac) * args.rh_bin_frac
    df["P_bin_kPa"] = np.floor(P_kPa / args.p_bin_kpa) * args.p_bin_kpa

    # Clip RH bin to [0, 1] range for robust labels.
    df["RH_bin"] = df["RH_bin"].clip(lower=0.0, upper=1.0)

    local_hour = (pd.to_numeric(df["hour_utc"], errors="coerce") + pd.to_numeric(df["longitude"], errors="coerce") / 15.0) % 24.0
    df["daynight"] = np.where((local_hour >= args.day_start_hour) & (local_hour < args.day_end_hour), "day", "night")

    # Keep only needed columns for aggregation.
    agg_value_cols = CANONICAL_FEATURES[:7]
    keep_cols = ID_COLS + COORD_COLS + ["month", "daynight", "T_bin", "RH_bin", "P_bin_kPa"] + agg_value_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DAC hourly CSV is missing required columns after canonicalization: {missing}")

    df = df[keep_cols].dropna(subset=agg_value_cols + ["T_bin", "RH_bin", "P_bin_kPa"])

    group_cols = ID_COLS + COORD_COLS + ["month", "daynight", "T_bin", "RH_bin", "P_bin_kPa"]
    grp = df.groupby(group_cols, dropna=False, sort=False).agg(
        n_hours=("T_ads_K", "count"),
        **{col: (col, "mean") for col in agg_value_cols},
    ).reset_index()
    return grp


def build_climate_bins(dac_hourly_csv: Path, args, out_dir: Path) -> pd.DataFrame:
    cache_path = out_dir / f"climate_bins_by_province_{args.year}_T{sanitize_tag(args.t_bin_c)}_RH{sanitize_tag(args.rh_bin_frac)}_P{sanitize_tag(args.p_bin_kpa)}.csv"
    if cache_path.exists() and not args.rebuild_climate_bins:
        print(f"[LOAD] Existing climate-bin cache: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[READ] DAC hourly input: {dac_hourly_csv}")
    usecols = None
    parts = []
    for i, chunk in enumerate(pd.read_csv(dac_hourly_csv, chunksize=args.chunksize, usecols=usecols), start=1):
        print(f"[BIN] Chunk {i:,} rows={len(chunk):,}")
        parts.append(create_bins_for_chunk(chunk, args))

    concat = pd.concat(parts, ignore_index=True)
    group_cols = ID_COLS + COORD_COLS + ["month", "daynight", "T_bin", "RH_bin", "P_bin_kPa"]
    value_cols = CANONICAL_FEATURES[:7]
    climate_bins = weighted_aggregate(concat, group_cols=group_cols, value_cols=value_cols, weight_col="n_hours")

    # Assign wet/dry season based on province-specific monthly mean RH.
    monthly_rh = weighted_aggregate(
        climate_bins,
        group_cols=ID_COLS + ["month"],
        value_cols=["RH_frac"],
        weight_col="n_hours",
    ).rename(columns={"RH_frac": "monthly_RH_frac"})

    med = monthly_rh.groupby(ID_COLS, dropna=False)["monthly_RH_frac"].median().reset_index().rename(
        columns={"monthly_RH_frac": "province_median_monthly_RH_frac"}
    )
    monthly_rh = monthly_rh.merge(med, on=ID_COLS, how="left")
    monthly_rh["season_wetdry"] = np.where(
        monthly_rh["monthly_RH_frac"] >= monthly_rh["province_median_monthly_RH_frac"],
        "wet_like",
        "dry_like",
    )

    climate_bins = climate_bins.merge(
        monthly_rh[ID_COLS + ["month", "monthly_RH_frac", "province_median_monthly_RH_frac", "season_wetdry"]],
        on=ID_COLS + ["month"],
        how="left",
    )

    climate_bins["climate_bin_id"] = (
        "T" + climate_bins["T_bin"].round(3).astype(str)
        + "_RH" + climate_bins["RH_bin"].round(3).astype(str)
        + "_P" + climate_bins["P_bin_kPa"].round(3).astype(str)
    )

    climate_bins.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Climate bins: {cache_path} rows={len(climate_bins):,}")
    return climate_bins


# =============================================================================
# Prediction and annualization
# =============================================================================

def add_operation_features(climate_groups: pd.DataFrame, op: pd.Series) -> pd.DataFrame:
    df = climate_groups.copy()
    for col in OPERATION_COLS:
        df[col] = float(op[col])
    return df


def cycle_time_from_operation(df: pd.DataFrame, args) -> np.ndarray:
    return (
        pd.to_numeric(df["adsorption_time_s"], errors="coerce").to_numpy(dtype="float64")
        + args.evacuation_time_s
        + pd.to_numeric(df["heating_desorption_time_s"], errors="coerce").to_numpy(dtype="float64")
        + args.cooling_time_s
        + args.repressurization_time_s
    )


def append_cycle_metrics(pred: pd.DataFrame, op_df: pd.DataFrame, args) -> pd.DataFrame:
    out = pred.copy()
    for col in OPERATION_COLS:
        out[col] = op_df[col].to_numpy()

    cycle_time = cycle_time_from_operation(op_df, args)
    out["cycle_time_s"] = cycle_time

    kg_co2 = out["kg_CO2_cycle_corrected"].to_numpy(dtype="float64")
    kg_h2o = out["kg_H2O_cycle_corrected"].to_numpy(dtype="float64")
    q_heat = out["Q_heat_kWhth_cycle"].to_numpy(dtype="float64")
    q_cool = out["Q_cool_kWhth_cycle"].to_numpy(dtype="float64")
    e_el = out["E_total_el_kWhe_cycle"].to_numpy(dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["co2_kg_per_h_per_bed"] = kg_co2 * 3600.0 / cycle_time
        out["productivity_kgCO2_kgads_year_if_full_year"] = (kg_co2 * 8760.0 * 3600.0 / cycle_time) / args.m_ads_kg
        out["specific_heat_MWhth_tCO2"] = q_heat / kg_co2
        out["specific_cooling_MWhth_tCO2"] = q_cool / kg_co2
        out["specific_electricity_MWhe_tCO2"] = e_el / kg_co2
        out["specific_total_MWh_tCO2_before_compression"] = (q_heat + e_el) / kg_co2
        out["H2O_CO2_mass_ratio_tH2O_tCO2"] = kg_h2o / kg_co2

    return out


def predict_for_groups_and_ops(
    surrogate: CycleSurrogate,
    groups: pd.DataFrame,
    ops: pd.DataFrame,
    args,
) -> pd.DataFrame:
    """Cross product groups x operations and predict cycle outputs.

    Revised for speed:
    - Previous version called model.predict once per operation candidate.
    - This version builds a larger cross-product batch and calls model.predict once
      per operation chunk, reducing TensorFlow overhead substantially.
    """
    groups = groups.reset_index(drop=True).copy()
    ops = ops.reset_index(drop=True).copy()
    if len(groups) == 0 or len(ops) == 0:
        return pd.DataFrame()

    op_batch_size = int(getattr(args, "operation_batch_size", 100) or 100)
    op_batch_size = max(1, op_batch_size)
    records = []

    group_repeats_cache = {}
    n_groups = len(groups)

    for start in range(0, len(ops), op_batch_size):
        op_chunk = ops.iloc[start:start + op_batch_size].reset_index(drop=True)
        n_ops = len(op_chunk)

        # Order: op1 with all groups, op2 with all groups, ...
        if n_ops not in group_repeats_cache:
            group_repeats_cache[n_ops] = pd.concat([groups] * n_ops, ignore_index=True)
        x = group_repeats_cache[n_ops].copy()

        op_rep = op_chunk.loc[op_chunk.index.repeat(n_groups)].reset_index(drop=True)
        for col in OPERATION_COLS:
            x[col] = pd.to_numeric(op_rep[col], errors="coerce").to_numpy()
        x["operation_id"] = pd.to_numeric(op_rep["operation_id"], errors="coerce").astype(int).to_numpy()

        pred = surrogate.predict(x[surrogate.feature_columns], batch_size=args.predict_batch_size)

        # Ensure canonical target names exist. If metadata targets are canonical, no change needed.
        for target in CANONICAL_TARGETS:
            if target not in pred.columns:
                found = None
                for alias in TARGET_ALIASES.get(target, [target]):
                    if alias in pred.columns:
                        found = alias
                        break
                if found is None:
                    raise ValueError(f"Surrogate prediction is missing target '{target}'. Predicted columns: {list(pred.columns)}")
                pred[target] = pred[found]

        out = pd.concat([x.reset_index(drop=True), append_cycle_metrics(pred[CANONICAL_TARGETS], x, args)], axis=1)
        # pd.concat can duplicate operation columns; keep the first occurrence of each duplicate name.
        out = out.loc[:, ~out.columns.duplicated()]
        records.append(out)

    return pd.concat(records, ignore_index=True)


def annualize_prediction_rows(pred_rows: pd.DataFrame, args, extra_group_cols: list[str]) -> pd.DataFrame:
    df = pred_rows.copy()
    n_hours = pd.to_numeric(df["n_hours"], errors="coerce").to_numpy(dtype="float64")
    cycle_time = pd.to_numeric(df["cycle_time_s"], errors="coerce").to_numpy(dtype="float64")
    cycles = n_hours * 3600.0 / cycle_time
    df["n_cycles"] = cycles

    # Per-bed annual values within the represented hours.
    df["CO2_kg_per_bed"] = df["kg_CO2_cycle_corrected"] * df["n_cycles"]
    df["H2O_kg_per_bed"] = df["kg_H2O_cycle_corrected"] * df["n_cycles"]
    df["Q_heat_kWhth_per_bed"] = df["Q_heat_kWhth_cycle"] * df["n_cycles"]
    df["Q_cool_kWhth_per_bed"] = df["Q_cool_kWhth_cycle"] * df["n_cycles"]
    df["E_total_el_kWhe_per_bed"] = df["E_total_el_kWhe_cycle"] * df["n_cycles"]
    for comp in ["E_fan_kWhe_cycle", "E_vacuum_kWhe_cycle", "E_repress_kWhe_cycle", "E_chiller_kWhe_cycle"]:
        if comp in df.columns:
            out_comp = comp.replace("_cycle", "_per_bed")
            df[out_comp] = df[comp] * df["n_cycles"]

    # For province-level annualization, group by ID/coordinate columns.
    # For global lookup diagnostics, those columns are absent; in that case
    # aggregate the whole dataframe into one diagnostic row instead of failing
    # with KeyError: country_code.
    requested_group_cols = ID_COLS + COORD_COLS + extra_group_cols
    group_cols = [c for c in requested_group_cols if c in df.columns]

    agg_dict = {
        "n_hours": "sum",
        "n_cycles": "sum",
        "CO2_kg_per_bed": "sum",
        "H2O_kg_per_bed": "sum",
        "Q_heat_kWhth_per_bed": "sum",
        "Q_cool_kWhth_per_bed": "sum",
        "E_total_el_kWhe_per_bed": "sum",
        "cycle_time_s": "mean",
    }
    for comp in ["E_fan_kWhe_per_bed", "E_vacuum_kWhe_per_bed", "E_repress_kWhe_per_bed", "E_chiller_kWhe_per_bed"]:
        if comp in df.columns:
            agg_dict[comp] = "sum"

    if group_cols:
        annual = df.groupby(group_cols, dropna=False, sort=False).agg(agg_dict).reset_index()
    else:
        rec = {}
        for col, how in agg_dict.items():
            if col not in df.columns:
                continue
            if how == "sum":
                rec[col] = pd.to_numeric(df[col], errors="coerce").sum()
            elif how == "mean":
                rec[col] = pd.to_numeric(df[col], errors="coerce").mean()
        rec["aggregation_scope"] = "global_diagnostic"
        annual = pd.DataFrame([rec])

    scale_1000 = 1000.0 / args.m_ads_kg

    annual["annual_CO2_t_per_bed"] = annual["CO2_kg_per_bed"] / 1000.0
    annual["annual_H2O_t_per_bed"] = annual["H2O_kg_per_bed"] / 1000.0
    annual["annual_CO2_t_per_1000kgads"] = annual["annual_CO2_t_per_bed"] * scale_1000
    annual["annual_H2O_t_per_1000kgads"] = annual["annual_H2O_t_per_bed"] * scale_1000

    annual["annual_heat_MWhth_per_bed"] = annual["Q_heat_kWhth_per_bed"] / 1000.0
    annual["annual_cooling_MWhth_per_bed"] = annual["Q_cool_kWhth_per_bed"] / 1000.0
    annual["annual_electricity_MWhe_per_bed"] = annual["E_total_el_kWhe_per_bed"] / 1000.0

    annual["annual_heat_MWhth_per_1000kgads"] = annual["annual_heat_MWhth_per_bed"] * scale_1000
    annual["annual_cooling_MWhth_per_1000kgads"] = annual["annual_cooling_MWhth_per_bed"] * scale_1000
    annual["annual_electricity_MWhe_per_1000kgads"] = annual["annual_electricity_MWhe_per_bed"] * scale_1000

    for comp in ["E_fan_kWhe_per_bed", "E_vacuum_kWhe_per_bed", "E_repress_kWhe_per_bed", "E_chiller_kWhe_per_bed"]:
        if comp in annual.columns:
            annual[comp.replace("kWhe_per_bed", "MWhe_per_bed")] = annual[comp] / 1000.0
            annual[comp.replace("kWhe_per_bed", "MWhe_per_1000kgads")] = annual[comp] / 1000.0 * scale_1000

    with np.errstate(divide="ignore", invalid="ignore"):
        annual["productivity_kgCO2_kgads_year"] = annual["CO2_kg_per_bed"] / args.m_ads_kg
        annual["SEC_heat_MWhth_tCO2"] = annual["Q_heat_kWhth_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_cooling_MWhth_tCO2"] = annual["Q_cool_kWhth_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_el_MWhe_tCO2"] = annual["E_total_el_kWhe_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_total_MWh_tCO2_before_compression"] = (
            annual["Q_heat_kWhth_per_bed"] + annual["E_total_el_kWhe_per_bed"]
        ) / annual["CO2_kg_per_bed"]
        annual["H2O_CO2_mass_ratio_tH2O_tCO2"] = annual["H2O_kg_per_bed"] / annual["CO2_kg_per_bed"]

    return annual


def choose_best_candidate(annual_by_candidate: pd.DataFrame, args) -> pd.DataFrame:
    df = annual_by_candidate.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["productivity_kgCO2_kgads_year", "SEC_total_MWh_tCO2_before_compression", "annual_CO2_t_per_bed"])
    df = df[df["annual_CO2_t_per_bed"] > 0]
    if df.empty:
        return annual_by_candidate.sort_values("annual_CO2_t_per_bed", ascending=False).head(1).copy()

    max_prod = df["productivity_kgCO2_kgads_year"].max()
    eligible = df[df["productivity_kgCO2_kgads_year"] >= args.min_productivity_frac * max_prod].copy()
    eligible = eligible[eligible["SEC_total_MWh_tCO2_before_compression"] <= args.max_sec_total]
    eligible = eligible[eligible["H2O_CO2_mass_ratio_tH2O_tCO2"] <= args.max_h2o_co2_ratio]
    if eligible.empty:
        eligible = df[df["productivity_kgCO2_kgads_year"] >= args.min_productivity_frac * max_prod].copy()
    if eligible.empty:
        eligible = df.copy()

    best = eligible.sort_values(
        ["SEC_total_MWh_tCO2_before_compression", "productivity_kgCO2_kgads_year"],
        ascending=[True, False],
    ).head(1).copy()
    return best


def select_best_per_segment(
    surrogate: CycleSurrogate,
    groups: pd.DataFrame,
    ops: pd.DataFrame,
    segment_cols: list[str],
    args,
    policy_name: str = "segment_policy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one operation per segment and return annual summary + selected ops."""
    annual_results = []
    selected_records = []

    segment_group_cols = ID_COLS + COORD_COLS + segment_cols
    grouped = list(groups.groupby(segment_group_cols, dropna=False, sort=False))
    n_total = len(grouped)
    print(f"[{policy_name}] segments/provinces to evaluate: {n_total:,}; operation candidates: {len(ops):,}", flush=True)

    for counter, (keys, sub) in enumerate(grouped, start=1):
        if not isinstance(keys, tuple):
            keys = (keys,)
        seg_info = dict(zip(segment_group_cols, keys))
        province_name = seg_info.get("province_name", "")
        country_code = seg_info.get("country_code", "")

        if counter == 1 or counter == n_total or counter % int(getattr(args, "progress_every", 10)) == 0:
            print(
                f"[{policy_name}] {counter:,}/{n_total:,} | {country_code} - {province_name} | "
                f"bins={len(sub):,} | candidates={len(ops):,}",
                flush=True,
            )

        pred_all = predict_for_groups_and_ops(surrogate, sub, ops, args)
        annual_by_op = annualize_prediction_rows(pred_all, args, extra_group_cols=segment_cols + ["operation_id"])
        best = choose_best_candidate(annual_by_op, args)
        best["selection_scope"] = "+".join(segment_cols) if segment_cols else "province_year"
        annual_results.append(best)

        op_id = int(best["operation_id"].iloc[0])
        op_row = ops[ops["operation_id"] == op_id].iloc[0].to_dict()
        selected = {**seg_info, **op_row}
        selected["operation_policy_internal"] = "+".join(segment_cols) if segment_cols else "static"
        selected["selected_SEC_total_MWh_tCO2"] = float(best["SEC_total_MWh_tCO2_before_compression"].iloc[0])
        selected["selected_productivity_kgCO2_kgads_year"] = float(best["productivity_kgCO2_kgads_year"].iloc[0])
        selected_records.append(selected)

    annual = pd.concat(annual_results, ignore_index=True) if annual_results else pd.DataFrame()
    selected = pd.DataFrame(selected_records)
    return annual, selected


def select_best_per_bin(
    surrogate: CycleSurrogate,
    groups: pd.DataFrame,
    ops: pd.DataFrame,
    group_cols_for_bin: list[str],
    args,
    policy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one operation independently for each bin/group, then aggregate to province annual."""
    chosen_rows = []
    selected_records = []

    grouped = list(groups.groupby(group_cols_for_bin, dropna=False, sort=False))
    n_total = len(grouped)
    print(f"[{policy_name}] bins/groups to evaluate: {n_total:,}; operation candidates: {len(ops):,}", flush=True)

    for counter, (keys, sub) in enumerate(grouped, start=1):
        if not isinstance(keys, tuple):
            keys = (keys,)
        bin_info = dict(zip(group_cols_for_bin, keys))
        if counter == 1 or counter == n_total or counter % int(getattr(args, "progress_every", 1000)) == 0:
            prov = bin_info.get("province_name", "global")
            ctry = bin_info.get("country_code", "")
            print(f"[{policy_name}] {counter:,}/{n_total:,} | {ctry} - {prov} | rows={len(sub):,}", flush=True)

        # sub normally has one row; can have more for global grouping.
        pred_all = predict_for_groups_and_ops(surrogate, sub, ops, args)

        # For bin-level choice, use hourly equivalent productivity and SEC.
        valid = pred_all.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["co2_kg_per_h_per_bed", "specific_total_MWh_tCO2_before_compression", "H2O_CO2_mass_ratio_tH2O_tCO2"]
        )
        valid = valid[valid["kg_CO2_cycle_corrected"] > 0]
        if valid.empty:
            best = pred_all.sort_values("kg_CO2_cycle_corrected", ascending=False).head(1).copy()
        else:
            max_rate = valid["co2_kg_per_h_per_bed"].max()
            eligible = valid[valid["co2_kg_per_h_per_bed"] >= args.min_bin_productivity_frac * max_rate].copy()
            eligible = eligible[eligible["specific_total_MWh_tCO2_before_compression"] <= args.max_sec_total]
            eligible = eligible[eligible["H2O_CO2_mass_ratio_tH2O_tCO2"] <= args.max_h2o_co2_ratio]
            if eligible.empty:
                eligible = valid[valid["co2_kg_per_h_per_bed"] >= args.min_bin_productivity_frac * max_rate].copy()
            if eligible.empty:
                eligible = valid.copy()
            best = eligible.sort_values(
                ["specific_total_MWh_tCO2_before_compression", "co2_kg_per_h_per_bed"],
                ascending=[True, False],
            ).head(1).copy()

        chosen_rows.append(best)
        op_id = int(best["operation_id"].iloc[0])
        op_row = ops[ops["operation_id"] == op_id].iloc[0].to_dict()
        selected = {**bin_info, **op_row}
        selected["operation_policy_internal"] = policy_name
        selected["selected_specific_total_MWh_tCO2"] = float(best["specific_total_MWh_tCO2_before_compression"].iloc[0])
        selected["selected_co2_kg_per_h_per_bed"] = float(best["co2_kg_per_h_per_bed"].iloc[0])
        selected_records.append(selected)

    chosen = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame()
    selected = pd.DataFrame(selected_records)

    annual = annualize_prediction_rows(chosen, args, extra_group_cols=[])
    return annual, selected


# =============================================================================
# Policy runners
# =============================================================================

def prepare_base_groupings(climate_bins: pd.DataFrame, args) -> dict[str, pd.DataFrame]:
    value_cols = CANONICAL_FEATURES[:7]
    bin_cols = ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]

    # Province climate bins, no month/daynight distinction.
    province_bins = weighted_aggregate(
        climate_bins,
        group_cols=ID_COLS + COORD_COLS + bin_cols,
        value_cols=value_cols,
        weight_col="n_hours",
    )

    # Province day/night bins.
    daynight_bins = weighted_aggregate(
        climate_bins,
        group_cols=ID_COLS + COORD_COLS + ["daynight"] + bin_cols,
        value_cols=value_cols,
        weight_col="n_hours",
    )

    # Province wet/dry bins.
    seasonal_bins = weighted_aggregate(
        climate_bins,
        group_cols=ID_COLS + COORD_COLS + ["season_wetdry"] + bin_cols,
        value_cols=value_cols,
        weight_col="n_hours",
    )

    # Global lookup bins: one row per climate bin across ASEAN.
    global_bins = weighted_aggregate(
        climate_bins,
        group_cols=bin_cols,
        value_cols=value_cols,
        weight_col="n_hours",
    )

    return {
        "province_bins": province_bins,
        "daynight_bins": daynight_bins,
        "seasonal_bins": seasonal_bins,
        "global_bins": global_bins,
    }


def apply_global_lookup(
    surrogate: CycleSurrogate,
    province_bins: pd.DataFrame,
    global_lookup: pd.DataFrame,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup_cols = ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]
    op_cols_with_id = ["operation_id"] + OPERATION_COLS
    table = global_lookup[lookup_cols + op_cols_with_id].drop_duplicates(subset=lookup_cols)

    merged = province_bins.merge(table, on=lookup_cols, how="left", validate="many_to_one")
    if merged["operation_id"].isna().any():
        n_missing = int(merged["operation_id"].isna().sum())
        warnings.warn(f"O4 global lookup missing operation for {n_missing} province bins. Dropping those bins.")
        merged = merged.dropna(subset=["operation_id"])
    merged["operation_id"] = merged["operation_id"].astype(int)

    # Predict one row per province-bin with its selected op.
    pred = surrogate.predict(merged[surrogate.feature_columns], batch_size=args.predict_batch_size)
    for target in CANONICAL_TARGETS:
        if target not in pred.columns:
            for alias in TARGET_ALIASES.get(target, [target]):
                if alias in pred.columns:
                    pred[target] = pred[alias]
                    break
    pred_rows = pd.concat([merged.reset_index(drop=True), append_cycle_metrics(pred[CANONICAL_TARGETS], merged, args)], axis=1)
    annual = annualize_prediction_rows(pred_rows, args, extra_group_cols=[])
    selected = merged[ID_COLS + COORD_COLS + lookup_cols + op_cols_with_id].copy()
    selected["operation_policy_internal"] = "global_lookup_table"
    return annual, selected


# =============================================================================
# Domain check
# =============================================================================

def make_domain_check(kpi_csv: Path, climate_bins: pd.DataFrame, operation_candidates: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    records = []
    try:
        kpi = pd.read_csv(kpi_csv)
        kpi = canonicalize_feature_columns(canonicalize_operation_columns(kpi))
        train_features = kpi[CANONICAL_FEATURES].copy()
    except Exception:
        train_features = pd.DataFrame()

    eval_base = climate_bins.copy()
    # operation candidates not included directly; create min/max for climate and operation separately.
    for col in CANONICAL_FEATURES[:7]:
        rec = {"feature": col, "source": "annual_climate_bins"}
        vals = pd.to_numeric(eval_base[col], errors="coerce")
        rec.update({"min": vals.min(), "max": vals.max(), "mean": vals.mean()})
        if not train_features.empty and col in train_features:
            tr = pd.to_numeric(train_features[col], errors="coerce")
            rec["training_min"] = tr.min()
            rec["training_max"] = tr.max()
            rec["outside_training_domain"] = bool((vals.min() < tr.min()) or (vals.max() > tr.max()))
        records.append(rec)

    for col in OPERATION_COLS:
        rec = {"feature": col, "source": "operation_candidates"}
        vals = pd.to_numeric(operation_candidates[col], errors="coerce")
        rec.update({"min": vals.min(), "max": vals.max(), "mean": vals.mean()})
        if not train_features.empty and col in train_features:
            tr = pd.to_numeric(train_features[col], errors="coerce")
            rec["training_min"] = tr.min()
            rec["training_max"] = tr.max()
            rec["outside_training_domain"] = bool((vals.min() < tr.min()) or (vals.max() > tr.max()))
        records.append(rec)

    domain = pd.DataFrame(records)
    path = out_dir / "diagnostics" / "surrogate_domain_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    domain.to_csv(path, index=False, encoding="utf-8-sig")
    return domain


# =============================================================================
# Main build
# =============================================================================


# =============================================================================
# Additional helpers for revised annual operation workflow and figures
# =============================================================================

POLICY_LABELS = {
    "O0": "O0_Avg_static_reference",
    "O1": "O1_lookup_table_LT",
    "O2": "O2_day_night_simple",
    "O3": "O3_seasonal_wet_dry_simple",
    "O4": "O4_adaptive_upper_bound",
}


def aggregate_segment_annual(annual_segments: pd.DataFrame, args) -> pd.DataFrame:
    group_cols = ID_COLS + COORD_COLS
    sum_cols = [
        "n_hours", "n_cycles", "CO2_kg_per_bed", "H2O_kg_per_bed",
        "Q_heat_kWhth_per_bed", "Q_cool_kWhth_per_bed", "E_total_el_kWhe_per_bed",
    ]
    available_sum = [c for c in sum_cols if c in annual_segments.columns]
    base = annual_segments.groupby(group_cols, dropna=False, sort=False)[available_sum].sum().reset_index()
    if "cycle_time_s" in annual_segments.columns:
        base["cycle_time_s"] = annual_segments.groupby(group_cols, dropna=False, sort=False)["cycle_time_s"].mean().values
    scale_1000 = 1000.0 / args.m_ads_kg
    base["annual_CO2_t_per_bed"] = base["CO2_kg_per_bed"] / 1000.0
    base["annual_H2O_t_per_bed"] = base["H2O_kg_per_bed"] / 1000.0
    base["annual_CO2_t_per_1000kgads"] = base["annual_CO2_t_per_bed"] * scale_1000
    base["annual_H2O_t_per_1000kgads"] = base["annual_H2O_t_per_bed"] * scale_1000
    base["annual_heat_MWhth_per_bed"] = base["Q_heat_kWhth_per_bed"] / 1000.0
    base["annual_cooling_MWhth_per_bed"] = base["Q_cool_kWhth_per_bed"] / 1000.0
    base["annual_electricity_MWhe_per_bed"] = base["E_total_el_kWhe_per_bed"] / 1000.0
    base["annual_heat_MWhth_per_1000kgads"] = base["annual_heat_MWhth_per_bed"] * scale_1000
    base["annual_cooling_MWhth_per_1000kgads"] = base["annual_cooling_MWhth_per_bed"] * scale_1000
    base["annual_electricity_MWhe_per_1000kgads"] = base["annual_electricity_MWhe_per_bed"] * scale_1000
    with np.errstate(divide="ignore", invalid="ignore"):
        base["productivity_kgCO2_kgads_year"] = base["CO2_kg_per_bed"] / args.m_ads_kg
        base["SEC_heat_MWhth_tCO2"] = base["Q_heat_kWhth_per_bed"] / base["CO2_kg_per_bed"]
        base["SEC_cooling_MWhth_tCO2"] = base["Q_cool_kWhth_per_bed"] / base["CO2_kg_per_bed"]
        base["SEC_el_MWhe_tCO2"] = base["E_total_el_kWhe_per_bed"] / base["CO2_kg_per_bed"]
        base["SEC_total_MWh_tCO2_before_compression"] = (base["Q_heat_kWhth_per_bed"] + base["E_total_el_kWhe_per_bed"]) / base["CO2_kg_per_bed"]
        base["H2O_CO2_mass_ratio_tH2O_tCO2"] = base["H2O_kg_per_bed"] / base["CO2_kg_per_bed"]
    return base


def choose_best_candidate_with_price_factor(annual_by_candidate: pd.DataFrame, args, price_factor: float) -> pd.DataFrame:
    df = annual_by_candidate.copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["productivity_kgCO2_kgads_year", "SEC_total_MWh_tCO2_before_compression", "annual_CO2_t_per_bed"])
    df = df[df["annual_CO2_t_per_bed"] > 0]
    if df.empty:
        return annual_by_candidate.sort_values("annual_CO2_t_per_bed", ascending=False).head(1).copy()
    prod = df["productivity_kgCO2_kgads_year"].to_numpy(dtype=float)
    sec = df["SEC_total_MWh_tCO2_before_compression"].to_numpy(dtype=float)
    prod_norm = (prod - np.nanmin(prod)) / max(np.nanmax(prod) - np.nanmin(prod), 1e-12)
    sec_norm = (sec - np.nanmin(sec)) / max(np.nanmax(sec) - np.nanmin(sec), 1e-12)
    score = price_factor * sec_norm - prod_norm
    df = df.assign(_score=score, _prod_norm=prod_norm, _sec_norm=sec_norm)
    return df.sort_values(["_score", "SEC_total_MWh_tCO2_before_compression", "productivity_kgCO2_kgads_year"], ascending=[True, True, False]).head(1).copy()


def load_hourly_rows_with_bins(dac_hourly_csv: Path, args) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(dac_hourly_csv, chunksize=args.chunksize):
        df = chunk.copy()
        df = canonicalize_feature_columns(df)
        for col in [*CANONICAL_FEATURES[:7], "longitude", "latitude", "month", "hour_utc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "T_C" in df.columns:
            df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
        else:
            df["T_C"] = pd.to_numeric(df["T_ads_K"], errors="coerce") - 273.15
        df["RH_frac"] = pd.to_numeric(df["RH_frac"], errors="coerce")
        df["P_bin_kPa"] = np.floor((pd.to_numeric(df["P_Pa"], errors="coerce") / 1000.0) / args.p_bin_kpa) * args.p_bin_kpa
        df["T_bin"] = np.floor(df["T_C"] / args.t_bin_c) * args.t_bin_c
        df["RH_bin"] = (np.floor(df["RH_frac"] / args.rh_bin_frac) * args.rh_bin_frac).clip(0.0, 1.0)
        local_hour = (pd.to_numeric(df["hour_utc"], errors="coerce") + pd.to_numeric(df["longitude"], errors="coerce") / 15.0) % 24.0
        df["local_hour"] = local_hour
        df["daynight"] = np.where((local_hour >= args.day_start_hour) & (local_hour < args.day_end_hour), "day", "night")
        chunks.append(df)
    hourly = pd.concat(chunks, ignore_index=True)
    monthly_rh = hourly.groupby(ID_COLS + ["month"], dropna=False, sort=False)["RH_frac"].mean().reset_index(name="monthly_RH_frac")
    med = monthly_rh.groupby(ID_COLS, dropna=False)["monthly_RH_frac"].median().reset_index(name="province_median_monthly_RH_frac")
    monthly_rh = monthly_rh.merge(med, on=ID_COLS, how="left")
    monthly_rh["season_wetdry"] = np.where(monthly_rh["monthly_RH_frac"] >= monthly_rh["province_median_monthly_RH_frac"], "wet_like", "dry_like")
    hourly = hourly.merge(monthly_rh[ID_COLS + ["month", "season_wetdry"]], on=ID_COLS + ["month"], how="left")
    hourly["climate_bin_id"] = "T" + hourly["T_bin"].round(3).astype(str) + "_RH" + hourly["RH_bin"].round(3).astype(str) + "_P" + hourly["P_bin_kPa"].round(3).astype(str)
    if "datetime_local" in hourly.columns:
        hourly["datetime_local"] = pd.to_datetime(hourly["datetime_local"], errors="coerce")
    elif "datetime_utc" in hourly.columns:
        utc = pd.to_datetime(hourly["datetime_utc"], errors="coerce")
        hourly["datetime_local"] = utc + pd.to_timedelta(hourly["longitude"] / 15.0, unit="h")
    else:
        hourly["datetime_local"] = pd.NaT
    return hourly


def find_province_match(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = str(query).strip().lower()
    mask = df["province_name"].astype(str).str.lower().str.contains(q, na=False)
    if not mask.any() and "jawa barat" in q:
        mask = df["province_name"].astype(str).str.lower().str.contains("west java|jawa barat", na=False)
    if not mask.any() and "west java" in q:
        mask = df["province_name"].astype(str).str.lower().str.contains("west java|jawa barat", na=False)
    return df.loc[mask].copy()


def choose_representative_provinces(hourly: pd.DataFrame) -> list[str]:
    stats = hourly.groupby(ID_COLS, dropna=False).agg(
        mean_T_C=("T_C", "mean"),
        mean_RH=("RH_frac", "mean"),
        std_T_C=("T_C", "std"),
        std_RH=("RH_frac", "std"),
    ).reset_index()
    if stats.empty:
        return []
    hot_humid = stats.sort_values(["mean_RH", "mean_T_C"], ascending=[False, False]).head(1)
    drier = stats.sort_values(["mean_RH", "mean_T_C"], ascending=[True, False]).head(1)
    cooler = stats.sort_values(["mean_T_C", "mean_RH"], ascending=[True, False]).head(1)
    names = pd.concat([hot_humid, drier, cooler], ignore_index=True)["province_name"].drop_duplicates().tolist()
    return names[:3]


def global_max_productivity_surface(groupings: dict[str, pd.DataFrame], surrogate: CycleSurrogate, ops: pd.DataFrame, args, out_dir: Path) -> pd.DataFrame:
    global_bins = groupings["global_bins"].copy()
    pred_all = predict_for_groups_and_ops(surrogate, global_bins, ops, args)
    best = pred_all.sort_values(["co2_kg_per_h_per_bed", "specific_total_MWh_tCO2_before_compression"], ascending=[False, True])
    best = best.groupby(["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"], dropna=False, sort=False).head(1).copy()
    surf = weighted_aggregate(
        best,
        group_cols=["T_bin", "RH_bin"],
        value_cols=["productivity_kgCO2_kgads_year_if_full_year", "specific_total_MWh_tCO2_before_compression"],
        weight_col="n_hours",
    )
    surf = surf.rename(columns={
        "productivity_kgCO2_kgads_year_if_full_year": "productivity_full_year_equiv_kgCO2_kgads_year",
        "specific_total_MWh_tCO2_before_compression": "specific_total_MWh_tCO2",
    })
    surf.to_csv(out_dir / "diagnostics" / "global_max_productivity_surface.csv", index=False, encoding="utf-8-sig")
    return surf


def pivot_surface(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    piv = df.pivot_table(index="RH_bin", columns="T_bin", values=value_col, aggfunc="mean")
    x = piv.columns.to_numpy(dtype=float)
    y = piv.index.to_numpy(dtype=float)
    z = piv.to_numpy(dtype=float)
    return x, y, z


def classify_ambient_zone(row: pd.Series) -> str:
    t = float(row["T_bin"])
    rh = float(row["RH_bin"])
    t_tag = "hot" if t >= 28 else ("warm" if t >= 22 else "cool")
    rh_tag = "humid" if rh >= 0.75 else ("moderateRH" if rh >= 0.55 else "dry")
    return f"{t_tag}_{rh_tag}"


def make_surface_and_normalized_figures(surface_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if surface_df.empty:
        return
    x1, y1, z1 = pivot_surface(surface_df, "productivity_full_year_equiv_kgCO2_kgads_year")
    x2, y2, z2 = pivot_surface(surface_df, "specific_total_MWh_tCO2")
    X1, Y1 = np.meshgrid(x1, y1)
    X2, Y2 = np.meshgrid(x2, y2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    cf1 = axes[0].contourf(X1, Y1, z1, levels=12)
    axes[0].contour(X1, Y1, z1, levels=12, linewidths=0.6, colors="k", alpha=0.6)
    axes[0].set_title("Productivity for maximal-productivity optimization")
    axes[0].set_xlabel("Temperature bin (°C)")
    axes[0].set_ylabel("Relative humidity bin (-)")
    fig.colorbar(cf1, ax=axes[0], label="kgCO2/kgads/year (full-year equivalent)")
    cf2 = axes[1].contourf(X2, Y2, z2, levels=12)
    axes[1].contour(X2, Y2, z2, levels=12, linewidths=0.6, colors="k", alpha=0.6)
    axes[1].set_title("Specific energy for maximal-productivity optimization")
    axes[1].set_xlabel("Temperature bin (°C)")
    axes[1].set_ylabel("Relative humidity bin (-)")
    fig.colorbar(cf2, ax=axes[1], label="MWh/tCO2")
    fig.savefig(fig_dir / "surface_productivity_and_specific_energy_maxprod.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    surf = surface_df.copy()
    prod = surf["productivity_full_year_equiv_kgCO2_kgads_year"].astype(float)
    sec = surf["specific_total_MWh_tCO2"].astype(float)
    surf["normalized_productivity"] = prod / max(prod.max(), 1e-12)
    surf["normalized_specific_energy"] = (sec - sec.min()) / max(sec.max() - sec.min(), 1e-12)
    surf["ambient_zone"] = surf.apply(classify_ambient_zone, axis=1)
    surf.to_csv(fig_dir / "normalized_surface_metrics.csv", index=False, encoding="utf-8-sig")

    x3, y3, z3 = pivot_surface(surf, "normalized_productivity")
    x4, y4, z4 = pivot_surface(surf, "normalized_specific_energy")
    X3, Y3 = np.meshgrid(x3, y3)
    X4, Y4 = np.meshgrid(x4, y4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    cf3 = axes[0].contourf(X3, Y3, z3, levels=12, vmin=0, vmax=1)
    axes[0].contour(X3, Y3, z3, levels=12, linewidths=0.6, colors="k", alpha=0.6)
    axes[0].set_title("Normalized productivity at different ambient conditions")
    axes[0].set_xlabel("Temperature bin (°C)")
    axes[0].set_ylabel("Relative humidity bin (-)")
    fig.colorbar(cf3, ax=axes[0], label="normalized productivity")
    cf4 = axes[1].contourf(X4, Y4, z4, levels=12, vmin=0, vmax=1)
    axes[1].contour(X4, Y4, z4, levels=12, linewidths=0.6, colors="k", alpha=0.6)
    axes[1].set_title("Normalized specific energy at different ambient conditions")
    axes[1].set_xlabel("Temperature bin (°C)")
    axes[1].set_ylabel("Relative humidity bin (-)")
    fig.colorbar(cf4, ax=axes[1], label="normalized specific energy")
    fig.savefig(fig_dir / "normalized_productivity_and_specific_energy_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    zone = surf.groupby("ambient_zone", dropna=False).agg(
        normalized_productivity=("normalized_productivity", "mean"),
        normalized_specific_energy=("normalized_specific_energy", "mean"),
        n=("ambient_zone", "size"),
    ).reset_index().sort_values("ambient_zone")
    zone.to_csv(fig_dir / "normalized_metrics_by_ambient_zone.csv", index=False, encoding="utf-8-sig")
    if not zone.empty:
        xpos = np.arange(len(zone))
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.bar(xpos - width / 2, zone["normalized_productivity"], width=width, label="normalized productivity")
        ax.bar(xpos + width / 2, zone["normalized_specific_energy"], width=width, label="normalized specific energy")
        ax.set_xticks(xpos)
        ax.set_xticklabels(zone["ambient_zone"], rotation=25, ha="right")
        ax.set_ylabel("Normalized value")
        ax.set_title("Normalized productivity and specific energy by ambient-condition class")
        ax.legend()
        fig.savefig(fig_dir / "normalized_productivity_and_specific_energy_bar_chart.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def build_hourly_policy_assignments(hourly_df: pd.DataFrame, selected_operations: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    sub = selected_operations[selected_operations["operation_policy"] == policy_name].copy()
    if sub.empty:
        return pd.DataFrame()
    hourly = hourly_df.copy()
    if policy_name == POLICY_LABELS["O0"]:
        keys = ID_COLS
    elif policy_name == POLICY_LABELS["O1"]:
        keys = ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]
    elif policy_name == POLICY_LABELS["O2"]:
        keys = ID_COLS + ["daynight"]
    elif policy_name == POLICY_LABELS["O3"]:
        keys = ID_COLS + ["season_wetdry"]
    elif policy_name == POLICY_LABELS["O4"]:
        keys = ID_COLS + COORD_COLS + ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]
    else:
        return pd.DataFrame()
    keep = [c for c in keys + ["operation_id", *OPERATION_COLS] if c in sub.columns]
    # Avoid many-to-many hourly expansion when the same climate bin appears in many provinces.
    table = sub[keep].drop_duplicates(subset=keys)
    merged = hourly.merge(table, on=keys, how="left")
    return merged


def make_west_java_operation_plots(hourly_df: pd.DataFrame, selected_operations: pd.DataFrame, out_dir: Path, preferred_policy: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    west = find_province_match(hourly_df, "west java")
    if west.empty:
        west = find_province_match(hourly_df, "jawa barat")
    if west.empty:
        return
    assigned = build_hourly_policy_assignments(west, selected_operations, preferred_policy)
    if assigned.empty:
        return
    assigned = assigned.sort_values("datetime_local").reset_index(drop=True)
    assigned["day_index"] = np.arange(len(assigned)) / 24.0
    monthly = assigned.groupby("month", dropna=False).agg(
        T_C=("T_C", "mean"),
        RH_frac=("RH_frac", "mean"),
        adsorption_time_h=("adsorption_time_s", lambda s: np.nanmean(pd.to_numeric(s, errors="coerce") / 3600.0)),
        desorption_time_h=("heating_desorption_time_s", lambda s: np.nanmean(pd.to_numeric(s, errors="coerce") / 3600.0)),
        T_des_K=("T_des_K", "mean"),
        T_coolant_K=("T_coolant_K", "mean"),
    ).reset_index()
    monthly.to_csv(fig_dir / "west_java_monthly_operation_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(monthly["month"], monthly["T_C"], label="T")
    axes[0].plot(monthly["month"], monthly["RH_frac"] * 100.0, label="RH (%)")
    axes[0].set_ylabel("Climate")
    axes[0].legend()
    axes[0].set_title(f"DAC operation in West Java over the year ({preferred_policy})")
    axes[1].plot(monthly["month"], monthly["adsorption_time_h"], label="adsorption time")
    axes[1].plot(monthly["month"], monthly["desorption_time_h"], label="desorption time")
    axes[1].set_ylabel("Time (h)")
    axes[1].legend()
    axes[2].plot(monthly["month"], monthly["T_des_K"], label="T_des")
    axes[2].plot(monthly["month"], monthly["T_coolant_K"], label="T_coolant")
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_xlabel("Month")
    axes[2].legend()
    fig.savefig(fig_dir / "west_java_operation_over_year.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Three representative weeks: lowest, median, highest weekly RH.
    tmp = assigned.dropna(subset=["datetime_local"]).copy()
    if tmp.empty:
        return
    tmp["week_start"] = tmp["datetime_local"].dt.to_period("W").apply(lambda x: x.start_time)
    weekly = tmp.groupby("week_start", dropna=False).agg(T_C=("T_C", "mean"), RH_frac=("RH_frac", "mean")).reset_index()
    weekly = weekly.sort_values("RH_frac")
    if len(weekly) >= 3:
        picks = pd.concat([weekly.head(1), weekly.iloc[[len(weekly)//2]], weekly.tail(1)], ignore_index=True)
    else:
        picks = weekly.head(3)
    weeks = picks["week_start"].tolist()
    fig, axes = plt.subplots(len(weeks), 2, figsize=(14, 4 * max(len(weeks), 1)), constrained_layout=True)
    if len(weeks) == 1:
        axes = np.array([axes])
    for i, wk in enumerate(weeks):
        wdf = tmp[(tmp["week_start"] == wk)].copy().sort_values("datetime_local")
        axes[i, 0].plot(wdf["datetime_local"], wdf["T_C"], label="T")
        axes[i, 0].plot(wdf["datetime_local"], wdf["RH_frac"] * 100.0, label="RH (%)")
        axes[i, 0].legend()
        axes[i, 0].set_title(f"Week starting {pd.Timestamp(wk).date()} - climate")
        axes[i, 1].plot(wdf["datetime_local"], pd.to_numeric(wdf["adsorption_time_s"], errors="coerce") / 3600.0, label="adsorption time")
        axes[i, 1].plot(wdf["datetime_local"], pd.to_numeric(wdf["heating_desorption_time_s"], errors="coerce") / 3600.0, label="desorption time")
        axes[i, 1].legend()
        axes[i, 1].set_title(f"Week starting {pd.Timestamp(wk).date()} - selected operation")
    fig.savefig(fig_dir / "west_java_operation_three_sample_weeks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_price_factor_plot(climate_bins: pd.DataFrame, surrogate: CycleSurrogate, ops: pd.DataFrame, args, out_dir: Path, sample_provinces: list[str]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not sample_provinces:
        return
    pfactors = [float(x) for x in str(args.price_factors).split(",") if str(x).strip()]
    records = []
    for pname in sample_provinces:
        sub = climate_bins[climate_bins["province_name"].astype(str) == str(pname)].copy()
        if sub.empty:
            continue
        pred_all = predict_for_groups_and_ops(surrogate, sub, ops, args)
        annual_by_op = annualize_prediction_rows(pred_all, args, extra_group_cols=["operation_id"])
        for pf in pfactors:
            best = choose_best_candidate_with_price_factor(annual_by_op, args, pf)
            records.append({
                "province_name": pname,
                "price_factor": pf,
                "productivity_kgCO2_kgads_year": float(best["productivity_kgCO2_kgads_year"].iloc[0]),
                "SEC_total_MWh_tCO2_before_compression": float(best["SEC_total_MWh_tCO2_before_compression"].iloc[0]),
            })
    out = pd.DataFrame(records)
    out.to_csv(fig_dir / "price_factor_sensitivity_three_locations.csv", index=False, encoding="utf-8-sig")
    if out.empty:
        return
    provinces = out["province_name"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for pname in provinces:
        s = out[out["province_name"] == pname].sort_values("price_factor")
        axes[0].plot(s["price_factor"], s["productivity_kgCO2_kgads_year"], marker="o", label=pname)
        axes[1].plot(s["price_factor"], s["SEC_total_MWh_tCO2_before_compression"], marker="o", label=pname)
    axes[0].set_title("Productivity for different price factors")
    axes[0].set_xlabel("Price factor")
    axes[0].set_ylabel("kgCO2/kgads/year")
    axes[1].set_title("Specific energy demand for different price factors")
    axes[1].set_xlabel("Price factor")
    axes[1].set_ylabel("MWh/tCO2")
    for ax in axes:
        ax.legend()
    fig.savefig(fig_dir / "price_factor_sensitivity_three_locations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_additional_figures(out_dir: Path, climate_bins: pd.DataFrame, groupings: dict[str, pd.DataFrame], selected_operations: pd.DataFrame, surrogate: CycleSurrogate, ops: pd.DataFrame, dac_hourly_csv: Path, args) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        surface = global_max_productivity_surface(groupings, surrogate, ops, args, out_dir)
        make_surface_and_normalized_figures(surface, out_dir)
    except Exception as exc:
        (fig_dir / "_warning_surface.txt").write_text(f"Surface plots failed: {exc}", encoding="utf-8")
    try:
        hourly = load_hourly_rows_with_bins(dac_hourly_csv, args)
        make_west_java_operation_plots(hourly, selected_operations, out_dir, preferred_policy=POLICY_LABELS["O1"])
        provinces = choose_representative_provinces(hourly)
        make_price_factor_plot(climate_bins, surrogate, ops, args, out_dir, provinces)
    except Exception as exc:
        (fig_dir / "_warning_hourly_plots.txt").write_text(f"Hourly-derived plots failed: {exc}", encoding="utf-8")




# =============================================================================
# McKinsey-style plotting and revised static-reference helpers
# =============================================================================

def set_consulting_style() -> dict:
    palette = {
        "navy": "#0B1F3A",
        "blue": "#1F77B4",
        "teal": "#2A9D8F",
        "orange": "#E76F51",
        "gold": "#E9C46A",
        "gray": "#9AA0A6",
        "light_gray": "#EEF1F4",
        "dark_gray": "#343A40",
        "white": "#FFFFFF",
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans", "Liberation Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": palette["dark_gray"],
        "axes.labelcolor": palette["dark_gray"],
        "axes.titlecolor": palette["navy"],
        "xtick.color": palette["dark_gray"],
        "ytick.color": palette["dark_gray"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E8EB",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "legend.frameon": False,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    return palette


def apply_selected_operations_to_groups(
    surrogate: CycleSurrogate,
    groups: pd.DataFrame,
    selected_ops: pd.DataFrame,
    args,
) -> pd.DataFrame:
    """Apply a selected province-level static operation to all climate bins.

    This is used to make O0 a true annual-average static reference: the operation
    is selected at the annual-average climate point, but annual performance is
    evaluated over the actual annual climate-bin distribution. Output columns are
    intentionally kept identical to the existing annual summary.
    """
    if selected_ops.empty or groups.empty:
        return pd.DataFrame()
    keys = ID_COLS + COORD_COLS
    table_cols = [c for c in keys + ["operation_id", *OPERATION_COLS] if c in selected_ops.columns]
    table = selected_ops[table_cols].drop_duplicates(subset=keys)
    merged = groups.merge(table, on=keys, how="left", validate="many_to_one")
    if merged["operation_id"].isna().any():
        missing = int(merged["operation_id"].isna().sum())
        warnings.warn(f"Static reference selected operation missing for {missing} climate-bin rows; dropping them.")
        merged = merged.dropna(subset=["operation_id"])
    if merged.empty:
        return pd.DataFrame()
    merged["operation_id"] = merged["operation_id"].astype(int)
    pred = surrogate.predict(merged[surrogate.feature_columns], batch_size=args.predict_batch_size)
    for target in CANONICAL_TARGETS:
        if target not in pred.columns:
            for alias in TARGET_ALIASES.get(target, [target]):
                if alias in pred.columns:
                    pred[target] = pred[alias]
                    break
    pred_rows = pd.concat([merged.reset_index(drop=True), append_cycle_metrics(pred[CANONICAL_TARGETS], merged, args)], axis=1)
    pred_rows = pred_rows.loc[:, ~pred_rows.columns.duplicated()]
    annual = annualize_prediction_rows(pred_rows, args, extra_group_cols=[])
    annual["selected_operation_count"] = 1
    return annual


def select_static_annual_mean_reference(
    surrogate: CycleSurrogate,
    climate_bins: pd.DataFrame,
    province_bins: pd.DataFrame,
    ops: pd.DataFrame,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O0: select operation from one annual-average climate point per province.

    This follows the spirit of the annual-average reference in weather-adaptive DAC
    studies. It prevents the static baseline from being optimized directly across
    all climate bins, which previously made O0 too strong and too close to the
    lookup/adaptive policies.
    """
    value_cols = CANONICAL_FEATURES[:7]
    annual_mean = weighted_aggregate(
        climate_bins,
        group_cols=ID_COLS + COORD_COLS,
        value_cols=value_cols,
        weight_col="n_hours",
    )
    annual_mean["T_bin"] = np.floor((annual_mean["T_ads_K"] - 273.15) / args.t_bin_c) * args.t_bin_c
    annual_mean["RH_bin"] = (np.floor(annual_mean["RH_frac"] / args.rh_bin_frac) * args.rh_bin_frac).clip(0.0, 1.0)
    annual_mean["P_bin_kPa"] = np.floor((annual_mean["P_Pa"] / 1000.0) / args.p_bin_kpa) * args.p_bin_kpa
    annual_mean["climate_bin_id"] = (
        "T" + annual_mean["T_bin"].round(3).astype(str)
        + "_RH" + annual_mean["RH_bin"].round(3).astype(str)
        + "_P" + annual_mean["P_bin_kPa"].round(3).astype(str)
    )
    annual_at_mean, selected = select_best_per_segment(
        surrogate,
        annual_mean,
        ops,
        segment_cols=[],
        args=args,
        policy_name=POLICY_LABELS["O0"],
    )
    selected["operation_policy_internal"] = "annual_mean_static_reference"
    selected["selection_note"] = "operation selected from one annual-average climate point; performance evaluated over actual annual climate bins"
    annual_actual = apply_selected_operations_to_groups(surrogate, province_bins, selected, args)
    return annual_actual, selected


def _finish_axis(ax, title: str | None = None):
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_surface_and_normalized_figures(surface_df: pd.DataFrame, out_dir: Path) -> None:
    """Reference-style contour figures; no ambient-zone bar chart.

    The output keeps the original filenames used downstream/README except the
    bar chart is intentionally not created, because the study focuses on ASEAN
    hot-humid operation rather than hot/cool/dry class comparison.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = set_consulting_style()
    if surface_df.empty:
        return
    x1, y1, z1 = pivot_surface(surface_df, "productivity_full_year_equiv_kgCO2_kgads_year")
    x2, y2, z2 = pivot_surface(surface_df, "specific_total_MWh_tCO2")
    X1, Y1 = np.meshgrid(x1, y1)
    X2, Y2 = np.meshgrid(x2, y2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    cf1 = axes[0].contourf(X1, Y1 * 100.0, z1, levels=14, cmap="YlGnBu")
    c1 = axes[0].contour(X1, Y1 * 100.0, z1, levels=10, colors=palette["white"], linewidths=0.45, alpha=0.75)
    axes[0].clabel(c1, inline=True, fontsize=7, fmt="%.1f")
    _finish_axis(axes[0], "Productivity for maximum-productivity operation")
    axes[0].set_xlabel("Ambient air temperature (°C)")
    axes[0].set_ylabel("Relative humidity (%)")
    cb1 = fig.colorbar(cf1, ax=axes[0], shrink=0.86)
    cb1.set_label("kgCO₂ kgads⁻¹ year⁻¹")

    cf2 = axes[1].contourf(X2, Y2 * 100.0, z2, levels=14, cmap="YlOrBr")
    c2 = axes[1].contour(X2, Y2 * 100.0, z2, levels=10, colors=palette["dark_gray"], linewidths=0.45, alpha=0.70)
    axes[1].clabel(c2, inline=True, fontsize=7, fmt="%.1f")
    _finish_axis(axes[1], "Specific energy for maximum-productivity operation")
    axes[1].set_xlabel("Ambient air temperature (°C)")
    axes[1].set_ylabel("Relative humidity (%)")
    cb2 = fig.colorbar(cf2, ax=axes[1], shrink=0.86)
    cb2.set_label("MWh tCO₂⁻¹")
    fig.savefig(fig_dir / "surface_productivity_and_specific_energy_maxprod.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    surf = surface_df.copy()
    prod = pd.to_numeric(surf["productivity_full_year_equiv_kgCO2_kgads_year"], errors="coerce")
    sec = pd.to_numeric(surf["specific_total_MWh_tCO2"], errors="coerce")
    surf["normalized_productivity"] = prod / max(prod.max(), 1e-12)
    surf["normalized_specific_energy"] = (sec - sec.min()) / max(sec.max() - sec.min(), 1e-12)
    surf.to_csv(fig_dir / "normalized_surface_metrics.csv", index=False, encoding="utf-8-sig")

    x3, y3, z3 = pivot_surface(surf, "normalized_productivity")
    x4, y4, z4 = pivot_surface(surf, "normalized_specific_energy")
    X3, Y3 = np.meshgrid(x3, y3)
    X4, Y4 = np.meshgrid(x4, y4)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    cf3 = axes[0].contourf(X3, Y3 * 100.0, z3, levels=np.linspace(0, 1, 15), cmap="YlGnBu")
    c3 = axes[0].contour(X3, Y3 * 100.0, z3, levels=np.linspace(0, 1, 9), colors=palette["white"], linewidths=0.45, alpha=0.75)
    axes[0].clabel(c3, inline=True, fontsize=7, fmt="%.2f")
    _finish_axis(axes[0], "Normalized productivity")
    axes[0].set_xlabel("Ambient air temperature (°C)")
    axes[0].set_ylabel("Relative humidity (%)")
    cb3 = fig.colorbar(cf3, ax=axes[0], shrink=0.86)
    cb3.set_label("Normalized value")
    cf4 = axes[1].contourf(X4, Y4 * 100.0, z4, levels=np.linspace(0, 1, 15), cmap="YlOrBr")
    c4 = axes[1].contour(X4, Y4 * 100.0, z4, levels=np.linspace(0, 1, 9), colors=palette["dark_gray"], linewidths=0.45, alpha=0.70)
    axes[1].clabel(c4, inline=True, fontsize=7, fmt="%.2f")
    _finish_axis(axes[1], "Normalized specific energy")
    axes[1].set_xlabel("Ambient air temperature (°C)")
    axes[1].set_ylabel("Relative humidity (%)")
    cb4 = fig.colorbar(cf4, ax=axes[1], shrink=0.86)
    cb4.set_label("Normalized value")
    fig.savefig(fig_dir / "normalized_productivity_and_specific_energy_heatmaps.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    # Remove stale old bar chart if it exists from a previous run.
    stale = fig_dir / "normalized_productivity_and_specific_energy_bar_chart.png"
    if stale.exists():
        try:
            stale.unlink()
        except Exception:
            pass


def _select_hourly_plot_policy(selected_operations: pd.DataFrame, preferred_policy: str) -> str:
    available = set(selected_operations.get("operation_policy", pd.Series(dtype=str)).dropna().astype(str))
    for p in [preferred_policy, POLICY_LABELS["O4"], POLICY_LABELS["O1"], POLICY_LABELS["O2"], POLICY_LABELS["O0"]]:
        if p in available:
            return p
    return preferred_policy


def make_west_java_operation_plots(hourly_df: pd.DataFrame, selected_operations: pd.DataFrame, out_dir: Path, preferred_policy: str) -> None:
    """West Java plots styled after reference: T/RH plus ads/des time only."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = set_consulting_style()
    west = find_province_match(hourly_df, "west java")
    if west.empty:
        west = find_province_match(hourly_df, "jawa barat")
    if west.empty:
        return
    policy = _select_hourly_plot_policy(selected_operations, preferred_policy)
    assigned = build_hourly_policy_assignments(west, selected_operations, policy)
    if assigned.empty:
        return
    assigned = assigned.sort_values("datetime_local").reset_index(drop=True)
    assigned["adsorption_time_h"] = pd.to_numeric(assigned["adsorption_time_s"], errors="coerce") / 3600.0
    assigned["desorption_time_h"] = pd.to_numeric(assigned["heating_desorption_time_s"], errors="coerce") / 3600.0
    assigned["RH_percent"] = pd.to_numeric(assigned["RH_frac"], errors="coerce") * 100.0
    assigned["T_C"] = pd.to_numeric(assigned["T_C"], errors="coerce")

    monthly = assigned.groupby("month", dropna=False).agg(
        T_C=("T_C", "mean"),
        RH_percent=("RH_percent", "mean"),
        adsorption_time_h=("adsorption_time_h", "mean"),
        desorption_time_h=("desorption_time_h", "mean"),
    ).reset_index()
    monthly.to_csv(fig_dir / "west_java_monthly_operation_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True, constrained_layout=True)
    ax0 = axes[0]
    ax0.plot(monthly["month"], monthly["T_C"], color=palette["navy"], lw=2.2, label="Ambient temperature")
    ax0.set_ylabel("Temperature (°C)")
    ax0b = ax0.twinx()
    ax0b.plot(monthly["month"], monthly["RH_percent"], color=palette["gray"], lw=2.0, ls="--", label="Relative humidity")
    ax0b.set_ylabel("Relative humidity (%)")
    _finish_axis(ax0, f"West Java climate and DAC operation over the year ({policy})")
    ax1 = axes[1]
    ax1.plot(monthly["month"], monthly["adsorption_time_h"], color=palette["blue"], lw=2.2, label="Adsorption time")
    ax1.plot(monthly["month"], monthly["desorption_time_h"], color=palette["orange"], lw=2.2, label="Desorption time")
    ax1.set_ylabel("Time (h)")
    ax1.set_xlabel("Month")
    ax1.set_xticks(range(1, 13))
    _finish_axis(ax1, "Selected operation schedule")
    lines = ax0.get_lines() + ax0b.get_lines()
    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc="upper left", ncol=2)
    ax1.legend(loc="upper left", ncol=2)
    fig.savefig(fig_dir / "west_java_operation_over_year.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    tmp = assigned.dropna(subset=["datetime_local"]).copy()
    if tmp.empty:
        return
    tmp["week_start"] = tmp["datetime_local"].dt.to_period("W").apply(lambda x: x.start_time)
    weekly = tmp.groupby("week_start", dropna=False).agg(T_C=("T_C", "mean"), RH_percent=("RH_percent", "mean")).reset_index()
    weekly = weekly.sort_values("RH_percent")
    if len(weekly) >= 3:
        picks = pd.concat([weekly.head(1), weekly.iloc[[len(weekly)//2]], weekly.tail(1)], ignore_index=True)
    else:
        picks = weekly.head(3)
    weeks = picks["week_start"].tolist()
    fig, axes = plt.subplots(len(weeks), 2, figsize=(13, 3.2 * max(len(weeks), 1)), constrained_layout=True)
    if len(weeks) == 1:
        axes = np.array([axes])
    for r, ws in enumerate(weeks):
        week = tmp[(tmp["datetime_local"] >= ws) & (tmp["datetime_local"] < ws + pd.Timedelta(days=7))].copy()
        if week.empty:
            continue
        t = (week["datetime_local"] - week["datetime_local"].min()).dt.total_seconds() / 3600.0
        ax = axes[r, 0]
        ax.plot(t, week["T_C"], color=palette["navy"], lw=1.8, label="T")
        ax.set_ylabel("T (°C)")
        axb = ax.twinx()
        axb.plot(t, week["RH_percent"], color=palette["gray"], lw=1.6, ls="--", label="RH")
        axb.set_ylabel("RH (%)")
        _finish_axis(ax, f"Week starting {pd.to_datetime(ws).date()}: climate")
        ax.set_xlabel("Time (h)")
        lines = ax.get_lines() + axb.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left", ncol=2)

        ax2 = axes[r, 1]
        ax2.step(t, week["adsorption_time_h"], where="post", color=palette["blue"], lw=1.9, label="Adsorption time")
        ax2.step(t, week["desorption_time_h"], where="post", color=palette["orange"], lw=1.9, label="Desorption time")
        ax2.set_ylabel("Time (h)")
        ax2.set_xlabel("Time (h)")
        _finish_axis(ax2, "Selected operation")
        ax2.legend(loc="upper left", ncol=2)
    fig.savefig(fig_dir / "west_java_operation_three_sample_weeks.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

def save_policy_checkpoint(out_dir: Path, annual_all: list[pd.DataFrame], selected_all: list[pd.DataFrame], tag: str) -> None:
    """Save partial policy results so completed policies are not lost if a later policy fails."""
    ckpt_dir = out_dir / "diagnostics" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if annual_all:
        pd.concat(annual_all, ignore_index=True).to_csv(
            ckpt_dir / f"annual_partial_after_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if selected_all:
        pd.concat(selected_all, ignore_index=True).to_csv(
            ckpt_dir / f"selected_partial_after_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )



def build(args) -> None:
    project_dir = Path(args.project_dir)
    tea_dir = resolve_tea_dir(project_dir, args.tea_dir)
    data_dir = Path(args.data_dir) if args.data_dir else tea_dir / "00_CYCLE_KPI"
    surrogate_dir = Path(args.surrogate_dir) if args.surrogate_dir else tea_dir / "01_CYCLE_SURROGATE"
    out_dir = Path(args.out_dir) if args.out_dir else tea_dir / "02_ANNUAL_DYNAMIC_OPERATION"

    for sub in ["inputs", "predictions", "annual_results", "diagnostics", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    kpi_csv = Path(args.kpi_csv) if args.kpi_csv else find_latest_file(
        data_dir,
        [
            "05a_ann_ready_kpi_with_design_rows_*.csv",
            "05a_ann_ready_kpi_with_design_*.csv",
            "05a_success_kpi_with_design_rows_*.csv",
            "05a_success_kpi_with_design_*.csv",
            "*success*kpi*with*design*.csv",
        ],
    )
    dac_hourly_csv = Path(args.dac_hourly_csv) if args.dac_hourly_csv else infer_dac_hourly_csv(project_dir, args.year)
    surrogate_run = Path(args.surrogate_run) if args.surrogate_run else find_latest_surrogate_run(surrogate_dir)

    print("=" * 100)
    print("02 ANNUAL DYNAMIC OPERATION EVALUATOR - REVISED")
    print("=" * 100)
    print(f"Project dir     : {project_dir}")
    print(f"TEA dir         : {tea_dir}")
    print(f"KPI CSV         : {kpi_csv}")
    print(f"DAC hourly CSV  : {dac_hourly_csv}")
    print(f"Surrogate run   : {surrogate_run}")
    print(f"Output dir      : {out_dir}")
    print("=" * 100)

    surrogate = CycleSurrogate(surrogate_run)
    ops = load_operation_candidates(kpi_csv, max_candidates=args.max_operation_candidates, random_state=args.random_state)
    ops_path = out_dir / "inputs" / "operation_candidate_grid.csv"
    ops.to_csv(ops_path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] Operation candidate grid: {ops_path} rows={len(ops):,}")

    climate_bins = build_climate_bins(dac_hourly_csv, args, out_dir / "inputs")
    groupings = prepare_base_groupings(climate_bins, args)
    make_domain_check(kpi_csv, climate_bins, ops, out_dir)

    annual_all = []
    selected_all = []

    run_policies = set(args.policies)
    if "all" in run_policies:
        run_policies = {"O0", "O1", "O2", "O3", "O4"}

    if "O0" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O0']} - annual-mean static reference")
        annual_o0, selected_o0 = select_static_annual_mean_reference(
            surrogate,
            climate_bins,
            groupings["province_bins"],
            ops,
            args,
        )
        annual_o0["operation_policy"] = POLICY_LABELS["O0"]
        selected_o0["operation_policy"] = POLICY_LABELS["O0"]
        annual_all.append(annual_o0)
        selected_all.append(selected_o0)
        save_policy_checkpoint(out_dir, annual_all, selected_all, "O0")

    # O1 = global lookup table LT (main implementable policy)
    if "O1" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O1']}")
        annual_lookup_global, global_lookup_selected = select_best_per_bin(
            surrogate,
            groupings["global_bins"],
            ops,
            group_cols_for_bin=["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"],
            args=args,
            policy_name="global_lookup_table",
        )
        annual_o1, selected_o1 = apply_global_lookup(surrogate, groupings["province_bins"], global_lookup_selected, args)
        annual_o1["operation_policy"] = POLICY_LABELS["O1"]
        selected_o1["operation_policy"] = POLICY_LABELS["O1"]
        annual_o1["selected_operation_count"] = np.nan
        annual_all.append(annual_o1)
        selected_all.append(selected_o1)
        save_policy_checkpoint(out_dir, annual_all, selected_all, "O1")
        global_lookup_selected.to_csv(out_dir / "predictions" / "global_climate_bin_lookup_table.csv", index=False, encoding="utf-8-sig")
        annual_lookup_global.to_csv(out_dir / "diagnostics" / "global_lookup_table_diagnostic_annualized.csv", index=False, encoding="utf-8-sig")

    if "O2" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O2']}")
        annual_segments, selected_o2 = select_best_per_segment(surrogate, groupings["daynight_bins"], ops, segment_cols=["daynight"], args=args, policy_name=POLICY_LABELS["O2"])
        annual_o2 = aggregate_segment_annual(annual_segments, args)
        annual_o2["operation_policy"] = POLICY_LABELS["O2"]
        annual_o2["selected_operation_count"] = 2
        selected_o2["operation_policy"] = POLICY_LABELS["O2"]
        annual_all.append(annual_o2)
        selected_all.append(selected_o2)
        save_policy_checkpoint(out_dir, annual_all, selected_all, "O2")
        annual_segments.to_csv(out_dir / "annual_results" / "annual_operation_O2_day_night_segments.csv", index=False, encoding="utf-8-sig")

    if "O3" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O3']}")
        annual_segments, selected_o3 = select_best_per_segment(surrogate, groupings["seasonal_bins"], ops, segment_cols=["season_wetdry"], args=args, policy_name=POLICY_LABELS["O3"])
        annual_o3 = aggregate_segment_annual(annual_segments, args)
        annual_o3["operation_policy"] = POLICY_LABELS["O3"]
        annual_o3["selected_operation_count"] = 2
        selected_o3["operation_policy"] = POLICY_LABELS["O3"]
        annual_all.append(annual_o3)
        selected_all.append(selected_o3)
        save_policy_checkpoint(out_dir, annual_all, selected_all, "O3")
        annual_segments.to_csv(out_dir / "annual_results" / "annual_operation_O3_seasonal_segments.csv", index=False, encoding="utf-8-sig")

    # O4 = province-specific adaptive upper bound
    if "O4" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O4']}")
        annual_o4, selected_o4 = select_best_per_bin(
            surrogate,
            groupings["province_bins"],
            ops,
            group_cols_for_bin=ID_COLS + COORD_COLS + ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"],
            args=args,
            policy_name="province_bin_adaptive_upper",
        )
        annual_o4["operation_policy"] = POLICY_LABELS["O4"]
        selected_o4["operation_policy"] = POLICY_LABELS["O4"]
        annual_all.append(annual_o4)
        selected_all.append(selected_o4)
        save_policy_checkpoint(out_dir, annual_all, selected_all, "O4")

    annual_summary = pd.concat(annual_all, ignore_index=True) if annual_all else pd.DataFrame()
    selected_operations = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()

    if not selected_operations.empty and not annual_summary.empty:
        counts = selected_operations.groupby(ID_COLS + ["operation_policy"], dropna=False).size().reset_index(name="selected_operation_count_calc")
        annual_summary = annual_summary.merge(counts, on=ID_COLS + ["operation_policy"], how="left")
        if "selected_operation_count" not in annual_summary.columns:
            annual_summary["selected_operation_count"] = annual_summary["selected_operation_count_calc"]
        else:
            annual_summary["selected_operation_count"] = annual_summary["selected_operation_count"].fillna(annual_summary["selected_operation_count_calc"])
        annual_summary = annual_summary.drop(columns=["selected_operation_count_calc"], errors="ignore")

    annual_path = out_dir / "annual_results" / "annual_operation_summary_by_province_policy.csv"
    selected_path = out_dir / "predictions" / "selected_operations_by_policy.csv"
    comparison_path = out_dir / "diagnostics" / "operation_policy_comparison.csv"
    config_path = out_dir / "policy_config.json"
    readme_path = out_dir / "README_02_ANNUAL_DYNAMIC_OPERATION.txt"

    annual_summary.to_csv(annual_path, index=False, encoding="utf-8-sig")
    selected_operations.to_csv(selected_path, index=False, encoding="utf-8-sig")

    if not annual_summary.empty:
        comp_cols = [
            "operation_policy", "annual_CO2_t_per_1000kgads", "productivity_kgCO2_kgads_year",
            "SEC_heat_MWhth_tCO2", "SEC_el_MWhe_tCO2", "SEC_total_MWh_tCO2_before_compression",
            "H2O_CO2_mass_ratio_tH2O_tCO2",
        ]
        comp = annual_summary[comp_cols].groupby("operation_policy").agg(["count", "mean", "median", "std", "min", "max"])
        comp.columns = ["_".join(col).strip() for col in comp.columns.values]
        comp = comp.reset_index()
        comp.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(comparison_path, index=False)

    run_config = vars(args).copy()
    run_config.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir),
        "tea_dir": str(tea_dir),
        "kpi_csv": str(kpi_csv),
        "dac_hourly_csv": str(dac_hourly_csv),
        "surrogate_run": str(surrogate_run),
        "out_dir": str(out_dir),
        "surrogate_feature_columns": surrogate.feature_columns,
        "surrogate_target_columns": surrogate.target_columns,
        "policy_labels": POLICY_LABELS,
    })
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    generate_additional_figures(out_dir, climate_bins, groupings, selected_operations, surrogate, ops, dac_hourly_csv, args)

    readme = f"""02_ANNUAL_DYNAMIC_OPERATION output\n\nThis folder contains annual process-side DAC operation results from the cycle surrogate.\n\nInput KPI CSV:\n{kpi_csv}\n\nInput DAC hourly climate CSV:\n{dac_hourly_csv}\n\nSurrogate run:\n{surrogate_run}\n\nPolicies evaluated:\n{sorted(run_policies)}\n\nOperation policies:\n{POLICY_LABELS['O0']} = one fixed operation per province-year (static reference)\n{POLICY_LABELS['O1']} = global climate-bin lookup table (main LT policy)\n{POLICY_LABELS['O2']} = one operation for day and one for night per province\n{POLICY_LABELS['O3']} = one operation for wet-like and one for dry-like season per province\n{POLICY_LABELS['O4']} = one operation per province climate bin; adaptive upper bound\n\nAdditional figures generated:\n- surface_productivity_and_specific_energy_maxprod.png\n- normalized_productivity_and_specific_energy_heatmaps.png\n- normalized_productivity_and_specific_energy_bar_chart.png\n- west_java_operation_over_year.png\n- west_java_operation_three_sample_weeks.png\n- price_factor_sensitivity_three_locations.png\n\nMain output files:\n{annual_path}\n{selected_path}\n{comparison_path}\n{config_path}\n"""
    readme_path.write_text(readme, encoding="utf-8")

    print("=" * 100)
    print("02 ANNUAL DYNAMIC OPERATION COMPLETE - REVISED")
    print("=" * 100)
    print(f"Saved annual summary      : {annual_path}")
    print(f"Saved selected operations : {selected_path}")
    print(f"Saved comparison summary  : {comparison_path}")
    print(f"Saved config              : {config_path}")
    print(f"Saved figures dir         : {out_dir / 'figures'}")
    print("=" * 100)
    if not annual_summary.empty:
        print(annual_summary.groupby("operation_policy")["annual_CO2_t_per_1000kgads"].describe().to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="Annual dynamic operation evaluator for DAC TVSA cycle surrogate (revised).")
    parser.add_argument("--project-dir", default=r"D:/Ashka/5.DAC/06.PYTHON")
    parser.add_argument("--tea-dir", default=None)
    parser.add_argument("--data-dir", default=None, help="Default: tea_dir/00_CYCLE_KPI")
    parser.add_argument("--surrogate-dir", default=None, help="Default: tea_dir/01_CYCLE_SURROGATE")
    parser.add_argument("--surrogate-run", default=None, help="Specific ANN run folder. If omitted, latest valid run is used.")
    parser.add_argument("--out-dir", default=None, help="Default: tea_dir/02_ANNUAL_DYNAMIC_OPERATION")
    parser.add_argument("--kpi-csv", default=None, help="Corrected 05a success KPI with design CSV. If omitted, latest is used.")
    parser.add_argument("--dac-hourly-csv", default=None, help="DAC hourly input CSV. If omitted, inferred from 00.TEMPORAL_DATA.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--rebuild-climate-bins", action="store_true")
    parser.add_argument("--m-ads-kg", type=float, default=0.13823)
    parser.add_argument("--evacuation-time-s", type=float, default=60.0)
    parser.add_argument("--cooling-time-s", type=float, default=600.0)
    parser.add_argument("--repressurization-time-s", type=float, default=180.0)
    parser.add_argument("--t-bin-c", type=float, default=1.0)
    parser.add_argument("--rh-bin-frac", type=float, default=0.05)
    parser.add_argument("--p-bin-kpa", type=float, default=5.0)
    parser.add_argument("--day-start-hour", type=float, default=6.0)
    parser.add_argument("--day-end-hour", type=float, default=18.0)
    parser.add_argument("--policies", nargs="+", default=["all"], help="Policies to run: all, O0, O1, O2, O3, O4")
    parser.add_argument("--max-operation-candidates", type=int, default=300, help="0 or negative means use all unique operation candidates. Default increased for operation diversity.")
    parser.add_argument("--operation-batch-size", type=int, default=100, help="Number of operation candidates predicted in one ANN cross-product batch.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N segments for segment policies; bin policies use this approximately.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-productivity-frac", type=float, default=0.80, help="For segment/year optimization.")
    parser.add_argument("--min-bin-productivity-frac", type=float, default=0.80, help="For bin-level optimization.")
    parser.add_argument("--max-sec-total", type=float, default=50.0)
    parser.add_argument("--max-h2o-co2-ratio", type=float, default=20.0)
    parser.add_argument("--price-factors", default="0.5,1.0,1.5,2.0,3.0", help="Comma-separated price-factor multipliers for figure-only sensitivity.")
    return parser.parse_args()


# =============================================================================
# USER-REQUESTED OVERRIDE v8: hourly/cycle-resolved policy workflow
# =============================================================================

# New operation policy set. Output CSV filenames are kept unchanged for downstream
# module compatibility; only operation_policy labels and added diagnostic/profile
# outputs change.
POLICY_LABELS = {
    "O0": "O0_static_fixed_reference",
    "O1": "O1_day_night",
    "O2": "O2_monthly_lookup",
    "O3": "O3_climate_lookup_table",
    "O4": "O4_continuous_adapt_each_cycle",
}


def _numeric_norm(s: pd.Series, higher_is_better: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("float64")
    finite = x.replace([np.inf, -np.inf], np.nan)
    lo = finite.min()
    hi = finite.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        out = pd.Series(0.0, index=s.index)
    else:
        out = (finite - lo) / (hi - lo)
    if higher_is_better:
        out = 1.0 - out
    return out.fillna(1.0)


def _domain_penalty_for_rows(df: pd.DataFrame, args) -> pd.Series:
    ranges = getattr(args, "_training_feature_ranges", None)
    if not ranges:
        return pd.Series(0.0, index=df.index)
    penalty = pd.Series(0.0, index=df.index, dtype="float64")
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue
        lo, hi = bounds
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        low_excess = ((lo - x) / (hi - lo)).clip(lower=0.0)
        high_excess = ((x - hi) / (hi - lo)).clip(lower=0.0)
        penalty = penalty + low_excess.fillna(0.0) + high_excess.fillna(0.0)
    return penalty


def _score_candidate_table(df: pd.DataFrame, args, mode: str, hourly_like: bool = False) -> pd.Series:
    """Lower score is better."""
    mode = str(mode or "balanced_TE_proxy").lower()
    if hourly_like:
        prod_col = "co2_kg_per_h_per_bed"
        sec_total_col = "specific_total_MWh_tCO2_before_compression"
        sec_heat_col = "specific_heat_MWhth_tCO2"
        sec_el_col = "specific_electricity_MWhe_tCO2"
    else:
        prod_col = "productivity_kgCO2_kgads_year"
        sec_total_col = "SEC_total_MWh_tCO2_before_compression"
        sec_heat_col = "SEC_heat_MWhth_tCO2"
        sec_el_col = "SEC_el_MWhe_tCO2"
    water_col = "H2O_CO2_mass_ratio_tH2O_tCO2"

    if mode == "min_sec":
        score = _numeric_norm(df.get(sec_total_col, pd.Series(np.nan, index=df.index)))
    elif mode == "max_productivity":
        score = _numeric_norm(df.get(prod_col, pd.Series(np.nan, index=df.index)), higher_is_better=True)
    elif mode == "cost_proxy":
        score = (
            args.weight_heat * _numeric_norm(df.get(sec_heat_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + args.weight_el * _numeric_norm(df.get(sec_el_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + args.weight_water * _numeric_norm(df.get(water_col, pd.Series(np.nan, index=df.index)))
        )
    elif mode == "emission_proxy":
        score = (
            0.25 * _numeric_norm(df.get(sec_heat_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + 0.55 * _numeric_norm(df.get(sec_el_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + 0.10 * _numeric_norm(df.get(water_col, pd.Series(np.nan, index=df.index)))
            + 0.10 * _numeric_norm(df.get(prod_col, pd.Series(np.nan, index=df.index)), higher_is_better=True)
        )
    else:
        score = (
            args.weight_heat * _numeric_norm(df.get(sec_heat_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + args.weight_el * _numeric_norm(df.get(sec_el_col, df.get(sec_total_col, pd.Series(np.nan, index=df.index))))
            + args.weight_water * _numeric_norm(df.get(water_col, pd.Series(np.nan, index=df.index)))
            + args.weight_prod * _numeric_norm(df.get(prod_col, pd.Series(np.nan, index=df.index)), higher_is_better=True)
        )
    if float(getattr(args, "domain_penalty_weight", 0.0) or 0.0) > 0:
        score = score + float(args.domain_penalty_weight) * _domain_penalty_for_rows(df, args)
    return score.replace([np.inf, -np.inf], np.nan).fillna(1e9)


def choose_best_candidate(annual_by_candidate: pd.DataFrame, args) -> pd.DataFrame:
    """Revised selector using configurable objective and domain penalty."""
    df = annual_by_candidate.copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["productivity_kgCO2_kgads_year", "SEC_total_MWh_tCO2_before_compression", "annual_CO2_t_per_bed"])
    df = df[df["annual_CO2_t_per_bed"] > 0]
    if df.empty:
        return annual_by_candidate.sort_values("annual_CO2_t_per_bed", ascending=False).head(1).copy()

    max_prod = df["productivity_kgCO2_kgads_year"].max()
    eligible = df[df["productivity_kgCO2_kgads_year"] >= args.min_productivity_frac * max_prod].copy()
    eligible = eligible[eligible["SEC_total_MWh_tCO2_before_compression"] <= args.max_sec_total]
    eligible = eligible[eligible["H2O_CO2_mass_ratio_tH2O_tCO2"] <= args.max_h2o_co2_ratio]
    if eligible.empty:
        eligible = df[df["productivity_kgCO2_kgads_year"] >= args.min_productivity_frac * max_prod].copy()
    if eligible.empty:
        eligible = df.copy()
    eligible["_objective_score"] = _score_candidate_table(eligible, args, args.operation_objective, hourly_like=False)
    return eligible.sort_values(
        ["_objective_score", "SEC_total_MWh_tCO2_before_compression", "productivity_kgCO2_kgads_year"],
        ascending=[True, True, False],
    ).head(1).copy()


def load_operation_candidates(kpi_csv: Path, max_candidates: int, random_state: int) -> pd.DataFrame:
    """Stratified operation-candidate selection.

    The previous selector over-sampled globally high-performing points. This version first
    samples the operation space itself, then fills with top performers. This increases the
    chance that day/night, monthly, lookup, and continuous policies select visibly different
    adsorption/desorption settings.
    """
    df = pd.read_csv(kpi_csv)
    try:
        df = canonicalize_operation_columns(df)
    except ValueError as exc:
        similar = [c for c in df.columns if any(k in c.lower() for k in ["ads", "des", "time", "cool", "heat", "design"])]
        raise ValueError(
            f"Failed to read operation candidates from KPI CSV: {kpi_csv}\n"
            "This script needs the 05a SUCCESS KPI WITH DESIGN file, not the KPI summary file.\n"
            f"Original error: {exc}\nSimilar columns found: {similar}"
        ) from exc

    for col in OPERATION_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=OPERATION_COLS).drop_duplicates(subset=OPERATION_COLS).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid operation candidates found from KPI CSV.")
    if max_candidates is None or max_candidates <= 0 or len(df) <= max_candidates:
        ops = df[OPERATION_COLS].drop_duplicates().reset_index(drop=True)
        ops.insert(0, "operation_id", np.arange(1, len(ops) + 1))
        return ops

    rng = np.random.default_rng(random_state)
    selected_idx: set[int] = set()

    # 1) Stratified sample across operation variables.
    strat = df[OPERATION_COLS].copy()
    for col in OPERATION_COLS:
        q = min(5, max(2, strat[col].nunique()))
        try:
            strat[f"_{col}_bin"] = pd.qcut(strat[col], q=q, labels=False, duplicates="drop")
        except Exception:
            strat[f"_{col}_bin"] = 0
    bin_cols = [f"_{c}_bin" for c in OPERATION_COLS]
    for _, sub in strat.groupby(bin_cols, dropna=False, sort=False):
        selected_idx.add(int(rng.choice(sub.index.to_numpy())))

    # 2) Add top performers from available KPI metrics.
    n_top_each = max(10, max_candidates // 12)
    metric_rules = [
        ("productivity_kgCO2_kgads_year", False),
        ("annual_tCO2_per_1000kgads", False),
        ("specific_total_bed_MWh_tCO2_before_compression", True),
        ("specific_heat_MWhth_tCO2", True),
        ("specific_electricity_MWhe_tCO2", True),
        ("H2O_CO2_mass_ratio_tH2O_tCO2", True),
    ]
    for metric, ascending in metric_rules:
        if metric in df.columns:
            s = pd.to_numeric(df[metric], errors="coerce")
            ranked = s.sort_values(ascending=ascending).dropna().index.tolist()
            for idx in ranked[:n_top_each]:
                selected_idx.add(int(idx))

    # 3) Random fill to exact maximum.
    remaining = [int(i) for i in df.index if int(i) not in selected_idx]
    if len(selected_idx) < max_candidates and remaining:
        add_n = min(max_candidates - len(selected_idx), len(remaining))
        selected_idx.update([int(i) for i in rng.choice(remaining, size=add_n, replace=False)])
    if len(selected_idx) > max_candidates:
        # Keep stratified/top order approximately by sorting, then sample down for diversity.
        selected_idx = set(rng.choice(sorted(selected_idx), size=max_candidates, replace=False).astype(int).tolist())

    selected_ops = df.loc[sorted(selected_idx), OPERATION_COLS].drop_duplicates().reset_index(drop=True)
    selected_ops.insert(0, "operation_id", np.arange(1, len(selected_ops) + 1))
    return selected_ops


def prepare_base_groupings(climate_bins: pd.DataFrame, args) -> dict[str, pd.DataFrame]:
    value_cols = CANONICAL_FEATURES[:7]
    bin_cols = ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]

    province_bins = weighted_aggregate(climate_bins, ID_COLS + COORD_COLS + bin_cols, value_cols, weight_col="n_hours")
    daynight_bins = weighted_aggregate(climate_bins, ID_COLS + COORD_COLS + ["daynight"] + bin_cols, value_cols, weight_col="n_hours")
    monthly_bins = weighted_aggregate(climate_bins, ID_COLS + COORD_COLS + ["month"] + bin_cols, value_cols, weight_col="n_hours")
    global_bins = weighted_aggregate(climate_bins, bin_cols, value_cols, weight_col="n_hours")
    return {
        "province_bins": province_bins,
        "daynight_bins": daynight_bins,
        "monthly_bins": monthly_bins,
        "global_bins": global_bins,
    }


def select_best_per_bin(
    surrogate: CycleSurrogate,
    groups: pd.DataFrame,
    ops: pd.DataFrame,
    group_cols_for_bin: list[str],
    args,
    policy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen_rows = []
    selected_records = []
    grouped = list(groups.groupby(group_cols_for_bin, dropna=False, sort=False))
    n_total = len(grouped)
    print(f"[{policy_name}] bins/groups to evaluate: {n_total:,}; operation candidates: {len(ops):,}", flush=True)
    for counter, (keys, sub) in enumerate(grouped, start=1):
        if not isinstance(keys, tuple):
            keys = (keys,)
        bin_info = dict(zip(group_cols_for_bin, keys))
        if counter == 1 or counter == n_total or counter % int(getattr(args, "progress_every", 1000)) == 0:
            prov = bin_info.get("province_name", "global")
            ctry = bin_info.get("country_code", "")
            print(f"[{policy_name}] {counter:,}/{n_total:,} | {ctry} - {prov} | rows={len(sub):,}", flush=True)

        pred_all = predict_for_groups_and_ops(surrogate, sub, ops, args)
        valid = pred_all.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["co2_kg_per_h_per_bed", "specific_total_MWh_tCO2_before_compression", "H2O_CO2_mass_ratio_tH2O_tCO2"]
        )
        valid = valid[valid["kg_CO2_cycle_corrected"] > 0]
        if valid.empty:
            best = pred_all.sort_values("kg_CO2_cycle_corrected", ascending=False).head(1).copy()
        else:
            max_rate = valid["co2_kg_per_h_per_bed"].max()
            eligible = valid[valid["co2_kg_per_h_per_bed"] >= args.min_bin_productivity_frac * max_rate].copy()
            eligible = eligible[eligible["specific_total_MWh_tCO2_before_compression"] <= args.max_sec_total]
            eligible = eligible[eligible["H2O_CO2_mass_ratio_tH2O_tCO2"] <= args.max_h2o_co2_ratio]
            if eligible.empty:
                eligible = valid[valid["co2_kg_per_h_per_bed"] >= args.min_bin_productivity_frac * max_rate].copy()
            if eligible.empty:
                eligible = valid.copy()
            eligible["_objective_score"] = _score_candidate_table(eligible, args, args.operation_objective, hourly_like=True)
            best = eligible.sort_values(
                ["_objective_score", "specific_total_MWh_tCO2_before_compression", "co2_kg_per_h_per_bed"],
                ascending=[True, True, False],
            ).head(1).copy()

        chosen_rows.append(best)
        op_id = int(best["operation_id"].iloc[0])
        op_row = ops[ops["operation_id"] == op_id].iloc[0].to_dict()
        selected = {**bin_info, **op_row}
        selected["operation_policy_internal"] = policy_name
        selected["selected_specific_total_MWh_tCO2"] = float(best["specific_total_MWh_tCO2_before_compression"].iloc[0])
        selected["selected_co2_kg_per_h_per_bed"] = float(best["co2_kg_per_h_per_bed"].iloc[0])
        selected_records.append(selected)
    chosen = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame()
    selected = pd.DataFrame(selected_records)
    annual = annualize_prediction_rows(chosen, args, extra_group_cols=[])
    return annual, selected


def make_fixed_reference_selected(climate_bins: pd.DataFrame, args) -> pd.DataFrame:
    provinces = climate_bins[ID_COLS + COORD_COLS].drop_duplicates().reset_index(drop=True).copy()
    provinces["operation_id"] = 0
    provinces["adsorption_time_s"] = float(args.static_adsorption_time_s)
    provinces["heating_desorption_time_s"] = float(args.static_heating_desorption_time_s)
    provinces["T_des_K"] = float(args.static_T_des_K)
    provinces["T_coolant_K"] = float(args.static_T_coolant_K)
    provinces["operation_policy_internal"] = "global_fixed_reference"
    provinces["selection_note"] = "fixed global reference operation applied to every province and every hour"
    provinces["operation_policy"] = POLICY_LABELS["O0"]
    return provinces


def build_hourly_policy_assignments(hourly_df: pd.DataFrame, selected_operations: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    sub = selected_operations[selected_operations["operation_policy"] == policy_name].copy()
    if sub.empty:
        return pd.DataFrame()
    hourly = hourly_df.copy()
    bin_cols = ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"]
    if policy_name == POLICY_LABELS["O0"]:
        # Per-province rows all contain the same fixed operation.
        keys = ID_COLS
    elif policy_name == POLICY_LABELS["O1"]:
        keys = ID_COLS + ["daynight"]
    elif policy_name == POLICY_LABELS["O2"]:
        keys = ID_COLS + ["month"]
    elif policy_name == POLICY_LABELS["O3"]:
        keys = bin_cols
    elif policy_name == POLICY_LABELS["O4"]:
        keys = ID_COLS + bin_cols
    else:
        return pd.DataFrame()
    keep = [c for c in keys + ["operation_id", *OPERATION_COLS] if c in sub.columns]
    table = sub[keep].drop_duplicates(subset=keys)
    merged = hourly.merge(table, on=keys, how="left")
    if merged["operation_id"].isna().any() and policy_name == POLICY_LABELS["O4"]:
        # Fallback: if a province-specific bin is absent, use global lookup if available.
        global_sub = selected_operations[selected_operations["operation_policy"] == POLICY_LABELS["O3"]].copy()
        if not global_sub.empty:
            missing = merged["operation_id"].isna()
            table_g = global_sub[[c for c in bin_cols + ["operation_id", *OPERATION_COLS] if c in global_sub.columns]].drop_duplicates(subset=bin_cols)
            fallback = hourly.loc[missing].merge(table_g, on=bin_cols, how="left")
            for c in ["operation_id", *OPERATION_COLS]:
                if c in fallback.columns:
                    merged.loc[missing, c] = fallback[c].to_numpy()
    return merged


def _profile_rows_from_prediction(chunk: pd.DataFrame, pred: pd.DataFrame, args, policy_name: str) -> pd.DataFrame:
    data = chunk.reset_index(drop=True).copy()
    metrics = append_cycle_metrics(pred[CANONICAL_TARGETS], data, args)
    metrics = metrics.loc[:, ~metrics.columns.duplicated()]
    cycle_time = pd.to_numeric(metrics["cycle_time_s"], errors="coerce")
    n_cycles_hour = 3600.0 / cycle_time
    out = data[[c for c in [*ID_COLS, *COORD_COLS, "datetime_utc", "datetime_local", "year", "month", "day", "hour_utc", "local_hour", "T_C", "T_ads_K", "RH_frac", "P_Pa", "p_H2O_Pa", "p_CO2_Pa", "q_H2O_GAB_mol_kg", "q_CO2_WADST_mol_kg", "T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id", "daynight", "operation_id", *OPERATION_COLS] if c in data.columns]].copy()
    out["operation_policy"] = policy_name
    out["cycle_time_s"] = cycle_time.to_numpy()
    out["n_cycles_hour"] = n_cycles_hour.to_numpy()
    for target in CANONICAL_TARGETS:
        out[target] = pd.to_numeric(pred[target], errors="coerce").to_numpy()
    out["CO2_kg_h"] = out["kg_CO2_cycle_corrected"] * out["n_cycles_hour"]
    out["H2O_kg_h"] = out["kg_H2O_cycle_corrected"] * out["n_cycles_hour"]
    out["Q_heat_kWhth_h"] = out["Q_heat_kWhth_cycle"] * out["n_cycles_hour"]
    out["Q_cool_kWhth_h"] = out["Q_cool_kWhth_cycle"] * out["n_cycles_hour"]
    out["E_total_el_kWhe_h"] = out["E_total_el_kWhe_cycle"] * out["n_cycles_hour"]
    out["E_fan_kWhe_h"] = out["E_fan_kWhe_cycle"] * out["n_cycles_hour"]
    out["E_vacuum_kWhe_h"] = out["E_vacuum_kWhe_cycle"] * out["n_cycles_hour"]
    out["E_repress_kWhe_h"] = out["E_repress_kWhe_cycle"] * out["n_cycles_hour"]
    out["E_chiller_kWhe_h"] = out["E_chiller_kWhe_cycle"] * out["n_cycles_hour"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["SEC_total_MWh_tCO2_hour"] = (out["Q_heat_kWhth_h"] + out["E_total_el_kWhe_h"]) / out["CO2_kg_h"]
        out["SEC_heat_MWhth_tCO2_hour"] = out["Q_heat_kWhth_h"] / out["CO2_kg_h"]
        out["SEC_el_MWhe_tCO2_hour"] = out["E_total_el_kWhe_h"] / out["CO2_kg_h"]
        out["H2O_CO2_mass_ratio_tH2O_tCO2_hour"] = out["H2O_kg_h"] / out["CO2_kg_h"]
    return out


def _annual_from_hourly_chunks(chunks: list[pd.DataFrame], args) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    sums = pd.concat(chunks, ignore_index=True)
    group_cols = ID_COLS + COORD_COLS + ["operation_policy"]
    annual = sums.groupby(group_cols, dropna=False, sort=False).sum(numeric_only=True).reset_index()
    scale_1000 = 1000.0 / args.m_ads_kg
    annual["CO2_kg_per_bed"] = annual["CO2_kg_h"]
    annual["H2O_kg_per_bed"] = annual["H2O_kg_h"]
    annual["Q_heat_kWhth_per_bed"] = annual["Q_heat_kWhth_h"]
    annual["Q_cool_kWhth_per_bed"] = annual["Q_cool_kWhth_h"]
    annual["E_total_el_kWhe_per_bed"] = annual["E_total_el_kWhe_h"]
    annual["n_hours"] = annual["n_profile_hours"]
    annual["n_cycles"] = annual["n_cycles_hour"]
    annual["annual_CO2_t_per_bed"] = annual["CO2_kg_per_bed"] / 1000.0
    annual["annual_H2O_t_per_bed"] = annual["H2O_kg_per_bed"] / 1000.0
    annual["annual_CO2_t_per_1000kgads"] = annual["annual_CO2_t_per_bed"] * scale_1000
    annual["annual_H2O_t_per_1000kgads"] = annual["annual_H2O_t_per_bed"] * scale_1000
    annual["annual_heat_MWhth_per_bed"] = annual["Q_heat_kWhth_per_bed"] / 1000.0
    annual["annual_cooling_MWhth_per_bed"] = annual["Q_cool_kWhth_per_bed"] / 1000.0
    annual["annual_electricity_MWhe_per_bed"] = annual["E_total_el_kWhe_per_bed"] / 1000.0
    annual["annual_heat_MWhth_per_1000kgads"] = annual["annual_heat_MWhth_per_bed"] * scale_1000
    annual["annual_cooling_MWhth_per_1000kgads"] = annual["annual_cooling_MWhth_per_bed"] * scale_1000
    annual["annual_electricity_MWhe_per_1000kgads"] = annual["annual_electricity_MWhe_per_bed"] * scale_1000
    for comp in ["E_fan_kWhe_h", "E_vacuum_kWhe_h", "E_repress_kWhe_h", "E_chiller_kWhe_h"]:
        if comp in annual.columns:
            annual[comp.replace("_h", "_per_bed")] = annual[comp]
            annual[comp.replace("kWhe_h", "MWhe_per_bed")] = annual[comp] / 1000.0
            annual[comp.replace("kWhe_h", "MWhe_per_1000kgads")] = annual[comp] / 1000.0 * scale_1000
    with np.errstate(divide="ignore", invalid="ignore"):
        annual["productivity_kgCO2_kgads_year"] = annual["CO2_kg_per_bed"] / args.m_ads_kg
        annual["SEC_heat_MWhth_tCO2"] = annual["Q_heat_kWhth_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_cooling_MWhth_tCO2"] = annual["Q_cool_kWhth_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_el_MWhe_tCO2"] = annual["E_total_el_kWhe_per_bed"] / annual["CO2_kg_per_bed"]
        annual["SEC_total_MWh_tCO2_before_compression"] = (annual["Q_heat_kWhth_per_bed"] + annual["E_total_el_kWhe_per_bed"]) / annual["CO2_kg_per_bed"]
        annual["H2O_CO2_mass_ratio_tH2O_tCO2"] = annual["H2O_kg_per_bed"] / annual["CO2_kg_per_bed"]
    return annual


def _domain_violation_summary(profile: pd.DataFrame, args, policy_name: str) -> dict:
    ranges = getattr(args, "_training_feature_ranges", None) or {}
    rec = {"operation_policy": policy_name, "n_rows": len(profile)}
    any_mask = pd.Series(False, index=profile.index)
    for col, bounds in ranges.items():
        if col not in profile.columns:
            continue
        lo, hi = bounds
        x = pd.to_numeric(profile[col], errors="coerce")
        mask = (x < lo) | (x > hi)
        rec[f"outside_{col}_rows"] = int(mask.sum())
        any_mask = any_mask | mask.fillna(False)
    rec["outside_any_training_feature_rows"] = int(any_mask.sum())
    rec["outside_any_training_feature_frac"] = float(any_mask.mean()) if len(any_mask) else np.nan
    return rec


def evaluate_hourly_profiles(
    hourly_df: pd.DataFrame,
    selected_operations: pd.DataFrame,
    surrogate: CycleSurrogate,
    args,
    out_dir: Path,
    run_policies: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = pred_dir / "hourly_operation_profile_by_province_policy.csv"
    cycle_path = pred_dir / "cycle_like_operation_profile_by_province_policy.csv"
    for p in [hourly_path, cycle_path]:
        if p.exists():
            p.unlink()
    header_written = False
    trans_header_written = False
    annual_chunks = []
    domain_records = []

    policy_order = ["O0", "O1", "O2", "O3", "O4"]
    for pkey in policy_order:
        if pkey not in run_policies:
            continue
        policy_name = POLICY_LABELS[pkey]
        print(f"[HOURLY] Building hourly profile for {policy_name}", flush=True)
        assigned = build_hourly_policy_assignments(hourly_df, selected_operations, policy_name)
        if assigned.empty:
            warnings.warn(f"No hourly assignment for {policy_name}; skipping hourly profile.")
            continue
        missing = int(assigned["operation_id"].isna().sum())
        if missing:
            warnings.warn(f"{policy_name}: {missing:,} hourly rows have no assigned operation and will be dropped.")
            assigned = assigned.dropna(subset=["operation_id"])
        assigned["operation_id"] = pd.to_numeric(assigned["operation_id"], errors="coerce").astype(int)

        # Transition/cycle-like table: only rows when selected operation changes.
        trans_cols = [c for c in [*ID_COLS, "datetime_local", "T_C", "RH_frac", "operation_id", *OPERATION_COLS] if c in assigned.columns]
        tmp = assigned.sort_values([*ID_COLS, "datetime_local"] if "datetime_local" in assigned.columns else ID_COLS).copy()
        tmp["_prev_op"] = tmp.groupby(ID_COLS, dropna=False)["operation_id"].shift(1)
        trans = tmp[(tmp["operation_id"] != tmp["_prev_op"]) | tmp["_prev_op"].isna()][trans_cols].copy()
        trans["operation_policy"] = policy_name
        trans.to_csv(cycle_path, index=False, encoding="utf-8-sig", mode="a", header=not trans_header_written)
        trans_header_written = True

        agg_parts = []
        n = len(assigned)
        step = int(args.hourly_eval_chunksize)
        for start in range(0, n, step):
            end = min(start + step, n)
            if start == 0 or end == n or ((start // step) + 1) % 10 == 0:
                print(f"[HOURLY] {policy_name}: rows {start+1:,}-{end:,}/{n:,}", flush=True)
            chunk = assigned.iloc[start:end].copy()
            pred = surrogate.predict(chunk[surrogate.feature_columns], batch_size=args.predict_batch_size)
            for target in CANONICAL_TARGETS:
                if target not in pred.columns:
                    for alias in TARGET_ALIASES.get(target, [target]):
                        if alias in pred.columns:
                            pred[target] = pred[alias]
                            break
            profile = _profile_rows_from_prediction(chunk, pred, args, policy_name)
            if args.write_hourly_profile:
                profile.to_csv(hourly_path, index=False, encoding="utf-8-sig", mode="a", header=not header_written)
                header_written = True
            agg = profile.groupby([*ID_COLS, *COORD_COLS, "operation_policy"], dropna=False, sort=False).agg(
                n_profile_hours=("CO2_kg_h", "count"),
                n_cycles_hour=("n_cycles_hour", "sum"),
                CO2_kg_h=("CO2_kg_h", "sum"),
                H2O_kg_h=("H2O_kg_h", "sum"),
                Q_heat_kWhth_h=("Q_heat_kWhth_h", "sum"),
                Q_cool_kWhth_h=("Q_cool_kWhth_h", "sum"),
                E_total_el_kWhe_h=("E_total_el_kWhe_h", "sum"),
                E_fan_kWhe_h=("E_fan_kWhe_h", "sum"),
                E_vacuum_kWhe_h=("E_vacuum_kWhe_h", "sum"),
                E_repress_kWhe_h=("E_repress_kWhe_h", "sum"),
                E_chiller_kWhe_h=("E_chiller_kWhe_h", "sum"),
            ).reset_index()
            agg_parts.append(agg)
            domain_records.append(_domain_violation_summary(profile, args, policy_name))
        annual_chunks.append(pd.concat(agg_parts, ignore_index=True))
    annual = _annual_from_hourly_chunks(annual_chunks, args)
    domain = pd.DataFrame(domain_records)
    if not domain.empty:
        domain = domain.groupby("operation_policy", dropna=False).agg(
            {c: ("sum" if c.endswith("rows") or c == "n_rows" else "mean") for c in domain.columns if c != "operation_policy"}
        ).reset_index()
    return annual, domain


def write_operation_diversity_diagnostics(selected_operations: pd.DataFrame, annual_summary: pd.DataFrame, out_dir: Path) -> None:
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    if selected_operations.empty:
        return
    unique_policy = selected_operations.groupby("operation_policy", dropna=False).agg(
        selected_rows=("operation_id", "size"),
        unique_operation_ids=("operation_id", "nunique"),
        unique_adsorption_times=("adsorption_time_s", "nunique"),
        unique_desorption_times=("heating_desorption_time_s", "nunique"),
    ).reset_index()
    unique_policy.to_csv(diag_dir / "unique_operation_count_by_policy.csv", index=False, encoding="utf-8-sig")
    share = selected_operations.groupby(["operation_policy", "operation_id"], dropna=False).size().reset_index(name="n_selected_rows")
    share["share_within_policy"] = share["n_selected_rows"] / share.groupby("operation_policy")["n_selected_rows"].transform("sum")
    share = share.sort_values(["operation_policy", "share_within_policy"], ascending=[True, False])
    share.to_csv(diag_dir / "operation_id_share_by_policy.csv", index=False, encoding="utf-8-sig")
    if all(c in selected_operations.columns for c in ID_COLS):
        div = selected_operations.groupby(ID_COLS + ["operation_policy"], dropna=False).agg(
            selected_rows=("operation_id", "size"),
            unique_operation_ids=("operation_id", "nunique"),
            unique_adsorption_times=("adsorption_time_s", "nunique"),
            unique_desorption_times=("heating_desorption_time_s", "nunique"),
        ).reset_index()
        div.to_csv(diag_dir / "operation_diversity_by_province.csv", index=False, encoding="utf-8-sig")


def make_policy_profile_plots(hourly_df: pd.DataFrame, selected_operations: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = set_consulting_style()
    west = find_province_match(hourly_df, "west java")
    if west.empty:
        west = find_province_match(hourly_df, "jawa barat")
    if west.empty:
        west = hourly_df[hourly_df["province_name"].astype(str).str.len() > 0].head(8760).copy()
    west = west.sort_values("datetime_local").copy()
    if west.empty:
        return
    policies = [POLICY_LABELS[k] for k in ["O0", "O1", "O2", "O3", "O4"] if POLICY_LABELS[k] in set(selected_operations["operation_policy"].astype(str))]
    colors = [palette["gray"], palette["blue"], palette["teal"], palette["orange"], palette["navy"]]

    assigned_list = []
    for pol in policies:
        a = build_hourly_policy_assignments(west, selected_operations, pol)
        if a.empty:
            continue
        a = a.sort_values("datetime_local").copy()
        a["adsorption_time_h"] = pd.to_numeric(a["adsorption_time_s"], errors="coerce") / 3600.0
        a["desorption_time_h"] = pd.to_numeric(a["heating_desorption_time_s"], errors="coerce") / 3600.0
        a["operation_policy"] = pol
        assigned_list.append(a)
    if not assigned_list:
        return
    assigned = pd.concat(assigned_list, ignore_index=True)
    assigned.to_csv(fig_dir / "west_java_hourly_operation_assignments_for_plot.csv", index=False, encoding="utf-8-sig")

    # Annual view, daily mean for readability.
    base = west.set_index("datetime_local").sort_index()
    climate_daily = base[["T_C", "RH_frac"]].resample("D").mean().reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(climate_daily["datetime_local"], climate_daily["T_C"], color=palette["navy"], lw=1.8, label="Temperature")
    ax0b = axes[0].twinx()
    ax0b.plot(climate_daily["datetime_local"], climate_daily["RH_frac"] * 100, color=palette["gray"], lw=1.5, ls="--", label="Relative humidity")
    axes[0].set_ylabel("Temperature (°C)")
    ax0b.set_ylabel("RH (%)")
    axes[0].set_title("West Java: climate and selected DAC cycle settings over the year", loc="left", fontweight="bold")
    for pol, color in zip(policies, colors):
        sub = assigned[assigned["operation_policy"] == pol].set_index("datetime_local").sort_index()
        day = sub[["adsorption_time_h", "desorption_time_h"]].resample("D").mean().reset_index()
        axes[1].step(day["datetime_local"], day["adsorption_time_h"], where="post", lw=1.4, color=color, label=pol)
        axes[2].step(day["datetime_local"], day["desorption_time_h"], where="post", lw=1.4, color=color, label=pol)
    axes[1].set_ylabel("Adsorption time (h)")
    axes[2].set_ylabel("Desorption time (h)")
    axes[2].set_xlabel("Date")
    axes[1].legend(ncol=2, fontsize=8, loc="upper left")
    for ax in axes:
        _finish_axis(ax)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "west_java_operation_over_year.png", dpi=280, bbox_inches="tight")
    plt.close(fig)

    # Three sample weeks: dry-ish, typical, wet-ish.
    tmp = west.dropna(subset=["datetime_local"]).copy()
    tmp["week_start"] = tmp["datetime_local"].dt.to_period("W").apply(lambda x: x.start_time)
    weekly = tmp.groupby("week_start", dropna=False).agg(T_C=("T_C", "mean"), RH_frac=("RH_frac", "mean")).reset_index().sort_values("RH_frac")
    if len(weekly) >= 3:
        picks = pd.concat([weekly.head(1), weekly.iloc[[len(weekly)//2]], weekly.tail(1)], ignore_index=True)
    else:
        picks = weekly.head(3)
    weeks = picks["week_start"].tolist()
    if not weeks:
        return
    fig, axes = plt.subplots(len(weeks) * 3, 1, figsize=(13, 3.0 * len(weeks) * 1.15), sharex=False, constrained_layout=True)
    if len(weeks) == 1:
        axes = np.array([axes]).ravel()
    for i, wk in enumerate(weeks):
        start = pd.Timestamp(wk)
        end = start + pd.Timedelta(days=7)
        clim = west[(west["datetime_local"] >= start) & (west["datetime_local"] < end)]
        axc, axa, axd = axes[i*3], axes[i*3+1], axes[i*3+2]
        axc.plot(clim["datetime_local"], clim["T_C"], color=palette["navy"], lw=1.6, label="Temperature")
        axcb = axc.twinx()
        axcb.plot(clim["datetime_local"], clim["RH_frac"] * 100, color=palette["gray"], lw=1.3, ls="--", label="RH")
        axc.set_ylabel("T (°C)")
        axcb.set_ylabel("RH (%)")
        axc.set_title(f"Sample week starting {start.date()}: climate and cycle-setting changes", loc="left", fontweight="bold")
        for pol, color in zip(policies, colors):
            sub = assigned[(assigned["operation_policy"] == pol) & (assigned["datetime_local"] >= start) & (assigned["datetime_local"] < end)]
            axa.step(sub["datetime_local"], sub["adsorption_time_h"], where="post", lw=1.35, color=color, label=pol)
            axd.step(sub["datetime_local"], sub["desorption_time_h"], where="post", lw=1.35, color=color, label=pol)
        axa.set_ylabel("Ads time (h)")
        axd.set_ylabel("Des time (h)")
        axa.legend(ncol=3, fontsize=7, loc="upper left")
        for ax in [axc, axa, axd]:
            _finish_axis(ax)
            ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "west_java_operation_three_sample_weeks.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def make_lookup_decision_heatmaps(selected_operations: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = set_consulting_style()
    for pol in [POLICY_LABELS["O3"], POLICY_LABELS["O4"]]:
        sub = selected_operations[selected_operations["operation_policy"] == pol].copy()
        if sub.empty or not {"T_bin", "RH_bin"}.issubset(sub.columns):
            continue
        # For O4, average selected settings across provinces for a compact T-RH decision map.
        agg = sub.groupby(["T_bin", "RH_bin"], dropna=False).agg(
            adsorption_time_s=("adsorption_time_s", "mean"),
            heating_desorption_time_s=("heating_desorption_time_s", "mean"),
        ).reset_index()
        for value_col, label, fname_tag, cmap in [
            ("adsorption_time_s", "Adsorption time (h)", "adsorption_time", "YlGnBu"),
            ("heating_desorption_time_s", "Desorption time (h)", "desorption_time", "YlOrBr"),
        ]:
            table = agg.copy()
            table[value_col + "_h"] = table[value_col] / 3600.0
            piv = table.pivot_table(index="RH_bin", columns="T_bin", values=value_col + "_h", aggfunc="mean")
            if piv.empty:
                continue
            x = piv.columns.to_numpy(dtype=float)
            y = piv.index.to_numpy(dtype=float) * 100.0
            X, Y = np.meshgrid(x, y)
            Z = piv.to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
            cf = ax.contourf(X, Y, Z, levels=12, cmap=cmap)
            c = ax.contour(X, Y, Z, levels=8, colors=palette["dark_gray"], linewidths=0.45, alpha=0.65)
            ax.clabel(c, inline=True, fontsize=7, fmt="%.2f")
            ax.set_xlabel("Ambient air temperature (°C)")
            ax.set_ylabel("Relative humidity (%)")
            ax.set_title(f"{pol}: {label} selected across T-RH bins", loc="left", fontweight="bold")
            fig.colorbar(cf, ax=ax, label=label)
            fig.savefig(fig_dir / f"decision_heatmap_{fname_tag}_{pol}.png".replace("/", "_"), dpi=280, bbox_inches="tight")
            plt.close(fig)


def generate_additional_figures(out_dir: Path, climate_bins: pd.DataFrame, groupings: dict[str, pd.DataFrame], selected_operations: pd.DataFrame, surrogate: CycleSurrogate, ops: pd.DataFrame, dac_hourly_csv: Path, args) -> None:
    try:
        surf = global_max_productivity_surface(groupings, surrogate, ops, args, out_dir)
        make_surface_and_normalized_figures(surf, out_dir)
    except Exception as exc:
        (out_dir / "figures" / "surface_figures_ERROR.txt").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "figures" / "surface_figures_ERROR.txt").write_text(str(exc), encoding="utf-8")
    try:
        hourly = load_hourly_rows_with_bins(dac_hourly_csv, args)
        make_policy_profile_plots(hourly, selected_operations, out_dir)
        make_lookup_decision_heatmaps(selected_operations, out_dir)
    except Exception as exc:
        (out_dir / "figures" / "operation_profile_figures_ERROR.txt").write_text(str(exc), encoding="utf-8")


def _compute_training_feature_ranges(kpi_csv: Path) -> dict[str, tuple[float, float]]:
    try:
        kpi = pd.read_csv(kpi_csv)
        kpi = canonicalize_feature_columns(canonicalize_operation_columns(kpi))
    except Exception:
        return {}
    ranges = {}
    for col in CANONICAL_FEATURES:
        if col in kpi.columns:
            vals = pd.to_numeric(kpi[col], errors="coerce")
            ranges[col] = (float(vals.min()), float(vals.max()))
    return ranges


def build(args) -> None:
    project_dir = Path(args.project_dir)
    tea_dir = resolve_tea_dir(project_dir, args.tea_dir)
    data_dir = Path(args.data_dir) if args.data_dir else tea_dir / "00_CYCLE_KPI"
    surrogate_dir = Path(args.surrogate_dir) if args.surrogate_dir else tea_dir / "01_CYCLE_SURROGATE"
    out_dir = Path(args.out_dir) if args.out_dir else tea_dir / "02_ANNUAL_DYNAMIC_OPERATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["annual_results", "predictions", "diagnostics", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    kpi_csv = Path(args.kpi_csv) if args.kpi_csv else find_latest_file(
        data_dir,
        [
            "05a_ann_ready_kpi_with_design_rows_*.csv",
            "05a_ann_ready_kpi_with_design_*.csv",
            "05a_success_kpi_with_design_rows_*.csv",
            "05a_success_kpi_with_design_*.csv",
            "*success*kpi*with*design*.csv",
        ],
    )
    dac_hourly_csv = Path(args.dac_hourly_csv) if args.dac_hourly_csv else infer_dac_hourly_csv(project_dir, args.year)
    surrogate_run = Path(args.surrogate_run) if args.surrogate_run else find_latest_surrogate_run(surrogate_dir)

    run_policies = {p.upper() for p in args.policies if str(p).lower() != "all"}
    if not run_policies or "ALL" in run_policies:
        run_policies = {"O0", "O1", "O2", "O3", "O4"}

    print("=" * 100)
    print("02 ANNUAL DYNAMIC OPERATION EVALUATOR - HOURLY/CYCLE PROFILE v8")
    print("=" * 100)
    print(f"Project dir     : {project_dir}")
    print(f"TEA dir         : {tea_dir}")
    print(f"KPI CSV         : {kpi_csv}")
    print(f"DAC hourly CSV  : {dac_hourly_csv}")
    print(f"Surrogate run   : {surrogate_run}")
    print(f"Output dir      : {out_dir}")
    print(f"Policies        : {sorted(run_policies)}")
    print(f"Operation candidates target: {args.max_operation_candidates}")
    print(f"Operation objective: {args.operation_objective}")
    print("=" * 100)

    args._training_feature_ranges = _compute_training_feature_ranges(kpi_csv)
    surrogate = CycleSurrogate(surrogate_run)
    print("[LOAD] Operation candidates")
    ops = load_operation_candidates(kpi_csv, max_candidates=args.max_operation_candidates, random_state=args.random_state)
    ops.to_csv(out_dir / "diagnostics" / "operation_candidates_used.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Operation candidates used: {len(ops):,}")

    print("[LOAD/BUILD] Climate bins")
    climate_bins = build_climate_bins(dac_hourly_csv, args, out_dir)
    groupings = prepare_base_groupings(climate_bins, args)
    make_domain_check(kpi_csv, climate_bins, ops, out_dir)

    selected_all = []
    annual_coarse_all = []

    if "O0" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O0']} - global fixed reference")
        selected_o0 = make_fixed_reference_selected(climate_bins, args)
        selected_all.append(selected_o0)
        # Coarse fallback annual; final annual is normally hourly-based.
        annual_o0 = apply_selected_operations_to_groups(surrogate, groupings["province_bins"], selected_o0, args)
        annual_o0["operation_policy"] = POLICY_LABELS["O0"]
        annual_coarse_all.append(annual_o0)

    if "O1" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O1']}")
        annual_segments, selected_o1 = select_best_per_segment(
            surrogate, groupings["daynight_bins"], ops, segment_cols=["daynight"], args=args, policy_name=POLICY_LABELS["O1"]
        )
        selected_o1["operation_policy"] = POLICY_LABELS["O1"]
        selected_all.append(selected_o1)
        annual_o1 = aggregate_segment_annual(annual_segments, args)
        annual_o1["operation_policy"] = POLICY_LABELS["O1"]
        annual_coarse_all.append(annual_o1)
        annual_segments.to_csv(out_dir / "annual_results" / "annual_operation_O1_day_night_segments.csv", index=False, encoding="utf-8-sig")

    if "O2" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O2']}")
        annual_segments, selected_o2 = select_best_per_segment(
            surrogate, groupings["monthly_bins"], ops, segment_cols=["month"], args=args, policy_name=POLICY_LABELS["O2"]
        )
        selected_o2["operation_policy"] = POLICY_LABELS["O2"]
        selected_all.append(selected_o2)
        annual_o2 = aggregate_segment_annual(annual_segments, args)
        annual_o2["operation_policy"] = POLICY_LABELS["O2"]
        annual_coarse_all.append(annual_o2)
        annual_segments.to_csv(out_dir / "annual_results" / "annual_operation_O2_monthly_segments.csv", index=False, encoding="utf-8-sig")

    if "O3" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O3']} - global climate-bin lookup table")
        _, selected_o3 = select_best_per_bin(
            surrogate,
            groupings["global_bins"],
            ops,
            group_cols_for_bin=["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"],
            args=args,
            policy_name=POLICY_LABELS["O3"],
        )
        selected_o3["operation_policy"] = POLICY_LABELS["O3"]
        selected_all.append(selected_o3)
        annual_o3, _ = apply_global_lookup(surrogate, groupings["province_bins"], selected_o3, args)
        annual_o3["operation_policy"] = POLICY_LABELS["O3"]
        annual_coarse_all.append(annual_o3)

    if "O4" in run_policies:
        print(f"[POLICY] {POLICY_LABELS['O4']} - province-specific hourly/bin-adaptive setting")
        annual_o4, selected_o4 = select_best_per_bin(
            surrogate,
            groupings["province_bins"],
            ops,
            group_cols_for_bin=ID_COLS + ["T_bin", "RH_bin", "P_bin_kPa", "climate_bin_id"],
            args=args,
            policy_name=POLICY_LABELS["O4"],
        )
        selected_o4["operation_policy"] = POLICY_LABELS["O4"]
        selected_all.append(selected_o4)
        annual_o4["operation_policy"] = POLICY_LABELS["O4"]
        annual_coarse_all.append(annual_o4)

    selected_operations = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    selected_path = out_dir / "predictions" / "selected_operations_by_policy.csv"
    selected_operations.to_csv(selected_path, index=False, encoding="utf-8-sig")

    if args.build_hourly_profile:
        print("[LOAD] Full hourly climate rows for hourly/cycle policy profile")
        hourly_df = load_hourly_rows_with_bins(dac_hourly_csv, args)
        annual_summary, domain_by_policy = evaluate_hourly_profiles(hourly_df, selected_operations, surrogate, args, out_dir, run_policies)
        if not domain_by_policy.empty:
            domain_by_policy.to_csv(out_dir / "diagnostics" / "surrogate_domain_violation_by_policy.csv", index=False, encoding="utf-8-sig")
    else:
        annual_summary = pd.concat(annual_coarse_all, ignore_index=True) if annual_coarse_all else pd.DataFrame()

    if not annual_summary.empty and not selected_operations.empty:
        row_counts = selected_operations.groupby(ID_COLS + ["operation_policy"], dropna=False).size().reset_index(name="selected_operation_count")
        unique_counts = selected_operations.groupby(ID_COLS + ["operation_policy"], dropna=False)["operation_id"].nunique().reset_index(name="unique_operation_count")
        annual_summary = annual_summary.merge(row_counts, on=ID_COLS + ["operation_policy"], how="left")
        annual_summary = annual_summary.merge(unique_counts, on=ID_COLS + ["operation_policy"], how="left")
        # O3 has no province-level selected rows. Use global selected count for every province.
        if POLICY_LABELS["O3"] in set(annual_summary["operation_policy"].astype(str)):
            o3_rows = selected_operations[selected_operations["operation_policy"] == POLICY_LABELS["O3"]]
            mask = annual_summary["operation_policy"] == POLICY_LABELS["O3"]
            annual_summary.loc[mask, "selected_operation_count"] = len(o3_rows)
            annual_summary.loc[mask, "unique_operation_count"] = o3_rows["operation_id"].nunique()
        if POLICY_LABELS["O0"] in set(annual_summary["operation_policy"].astype(str)):
            mask = annual_summary["operation_policy"] == POLICY_LABELS["O0"]
            annual_summary.loc[mask, "selected_operation_count"] = 1
            annual_summary.loc[mask, "unique_operation_count"] = 1

    annual_path = out_dir / "annual_results" / "annual_operation_summary_by_province_policy.csv"
    comparison_path = out_dir / "diagnostics" / "operation_policy_comparison.csv"
    config_path = out_dir / "policy_config.json"
    readme_path = out_dir / "README_02_ANNUAL_DYNAMIC_OPERATION.txt"

    annual_summary.to_csv(annual_path, index=False, encoding="utf-8-sig")
    selected_operations.to_csv(selected_path, index=False, encoding="utf-8-sig")
    write_operation_diversity_diagnostics(selected_operations, annual_summary, out_dir)

    if not annual_summary.empty:
        comp_cols = [
            "operation_policy", "annual_CO2_t_per_1000kgads", "productivity_kgCO2_kgads_year",
            "SEC_heat_MWhth_tCO2", "SEC_el_MWhe_tCO2", "SEC_total_MWh_tCO2_before_compression",
            "H2O_CO2_mass_ratio_tH2O_tCO2",
        ]
        comp = annual_summary[[c for c in comp_cols if c in annual_summary.columns]].groupby("operation_policy").agg(["count", "mean", "median", "std", "min", "max"])
        comp.columns = ["_".join(col).strip() for col in comp.columns.values]
        comp = comp.reset_index()
        comp.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(comparison_path, index=False)

    run_config = vars(args).copy()
    # Avoid JSON serialization issue for internal ranges.
    run_config.pop("_training_feature_ranges", None)
    run_config.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_dir),
        "tea_dir": str(tea_dir),
        "kpi_csv": str(kpi_csv),
        "dac_hourly_csv": str(dac_hourly_csv),
        "surrogate_run": str(surrogate_run),
        "out_dir": str(out_dir),
        "surrogate_feature_columns": surrogate.feature_columns,
        "surrogate_target_columns": surrogate.target_columns,
        "policy_labels": POLICY_LABELS,
        "training_feature_ranges": {k: list(v) for k, v in getattr(args, "_training_feature_ranges", {}).items()},
    })
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("[PLOT] Additional operation figures")
    generate_additional_figures(out_dir, climate_bins, groupings, selected_operations, surrogate, ops, dac_hourly_csv, args)

    readme = f"""02_ANNUAL_DYNAMIC_OPERATION output\n\nThis folder contains annual process-side DAC operation results from the cycle surrogate.\n\nPolicies evaluated:\n{sorted(run_policies)}\n\nOperation policies:\n{POLICY_LABELS['O0']} = fixed global static reference: adsorption={args.static_adsorption_time_s}s, heating desorption={args.static_heating_desorption_time_s}s, T_des={args.static_T_des_K}K, T_coolant={args.static_T_coolant_K}K.\n{POLICY_LABELS['O1']} = one operation for day and one for night per province.\n{POLICY_LABELS['O2']} = one operation per month per province.\n{POLICY_LABELS['O3']} = global T-RH-P climate-bin lookup table applied hourly.\n{POLICY_LABELS['O4']} = province-specific T-RH-P adaptive operation applied hourly as a cycle-setting proxy.\n\nMain output files kept for downstream compatibility:\n{annual_path}\n{selected_path}\n{comparison_path}\n{config_path}\n\nAdditional outputs:\n{out_dir / 'predictions' / 'hourly_operation_profile_by_province_policy.csv'}\n{out_dir / 'predictions' / 'cycle_like_operation_profile_by_province_policy.csv'}\n{out_dir / 'diagnostics' / 'unique_operation_count_by_policy.csv'}\n{out_dir / 'diagnostics' / 'operation_id_share_by_policy.csv'}\n{out_dir / 'diagnostics' / 'operation_diversity_by_province.csv'}\n{out_dir / 'diagnostics' / 'surrogate_domain_violation_by_policy.csv'}\n\nFigures:\n- surface_productivity_and_specific_energy_maxprod.png\n- normalized_productivity_and_specific_energy_heatmaps.png\n- west_java_operation_over_year.png\n- west_java_operation_three_sample_weeks.png\n- decision_heatmap_adsorption_time_*.png\n- decision_heatmap_desorption_time_*.png\n"""
    readme_path.write_text(readme, encoding="utf-8")

    print("=" * 100)
    print("02 ANNUAL DYNAMIC OPERATION COMPLETE - HOURLY/CYCLE PROFILE v8")
    print("=" * 100)
    print(f"Saved annual summary      : {annual_path}")
    print(f"Saved selected operations : {selected_path}")
    print(f"Saved comparison summary  : {comparison_path}")
    print(f"Saved config              : {config_path}")
    print(f"Saved figures dir         : {out_dir / 'figures'}")
    print("=" * 100)
    if not annual_summary.empty:
        print(annual_summary.groupby("operation_policy")["annual_CO2_t_per_1000kgads"].describe().to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="Annual dynamic operation evaluator for DAC TVSA cycle surrogate (hourly/cycle profile v8).")
    parser.add_argument("--project-dir", default=r"D:/Ashka/5.DAC/06.PYTHON")
    parser.add_argument("--tea-dir", default=None)
    parser.add_argument("--data-dir", default=None, help="Default: tea_dir/00_CYCLE_KPI")
    parser.add_argument("--surrogate-dir", default=None, help="Default: tea_dir/01_CYCLE_SURROGATE")
    parser.add_argument("--surrogate-run", default=None, help="Specific ANN run folder. If omitted, latest valid run is used.")
    parser.add_argument("--out-dir", default=None, help="Default: tea_dir/02_ANNUAL_DYNAMIC_OPERATION")
    parser.add_argument("--kpi-csv", default=None, help="Corrected 05a success KPI with design CSV. If omitted, latest is used.")
    parser.add_argument("--dac-hourly-csv", default=None, help="DAC hourly input CSV. If omitted, inferred from 00.TEMPORAL_DATA.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
    parser.add_argument("--hourly-eval-chunksize", type=int, default=200_000, help="Rows per chunk when predicting full hourly/cycle profiles.")
    parser.add_argument("--rebuild-climate-bins", action="store_true")
    parser.add_argument("--m-ads-kg", type=float, default=0.13823)
    parser.add_argument("--evacuation-time-s", type=float, default=60.0)
    parser.add_argument("--cooling-time-s", type=float, default=600.0)
    parser.add_argument("--repressurization-time-s", type=float, default=180.0)
    parser.add_argument("--t-bin-c", type=float, default=1.0)
    parser.add_argument("--rh-bin-frac", type=float, default=0.05)
    parser.add_argument("--p-bin-kpa", type=float, default=5.0)
    parser.add_argument("--day-start-hour", type=float, default=6.0)
    parser.add_argument("--day-end-hour", type=float, default=18.0)
    parser.add_argument("--policies", nargs="+", default=["all"], help="Policies to run: all, O0, O1, O2, O3, O4")
    parser.add_argument("--max-operation-candidates", type=int, default=500, help="0 or negative means use all unique operation candidates. Default 500 for operation diversity.")
    parser.add_argument("--operation-batch-size", type=int, default=100, help="Number of operation candidates predicted in one ANN cross-product batch.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-productivity-frac", type=float, default=0.75, help="For segment/month optimization.")
    parser.add_argument("--min-bin-productivity-frac", type=float, default=0.75, help="For bin-level optimization.")
    parser.add_argument("--max-sec-total", type=float, default=50.0)
    parser.add_argument("--max-h2o-co2-ratio", type=float, default=20.0)
    parser.add_argument("--operation-objective", default="balanced_TE_proxy", choices=["min_SEC", "max_productivity", "cost_proxy", "emission_proxy", "balanced_TE_proxy"])
    parser.add_argument("--weight-heat", type=float, default=0.35)
    parser.add_argument("--weight-el", type=float, default=0.35)
    parser.add_argument("--weight-water", type=float, default=0.10)
    parser.add_argument("--weight-prod", type=float, default=0.20)
    parser.add_argument("--domain-penalty-weight", type=float, default=0.30)
    parser.add_argument("--static-adsorption-time-s", type=float, default=2160.0)
    parser.add_argument("--static-heating-desorption-time-s", type=float, default=4650.0)
    parser.add_argument("--static-T-des-K", type=float, default=363.15)
    parser.add_argument("--static-T-coolant-K", type=float, default=303.15)
    parser.add_argument("--build-hourly-profile", action=argparse.BooleanOptionalAction, default=True, help="Build hourly operation profile and aggregate annual summary from hourly predictions.")
    parser.add_argument("--write-hourly-profile", action=argparse.BooleanOptionalAction, default=True, help="Write the full hourly_operation_profile_by_province_policy.csv. Annual hourly aggregation still runs if disabled.")
    parser.add_argument("--price-factors", default="0.5,1.0,1.5,2.0,3.0", help="Comma-separated price-factor multipliers for figure-only sensitivity.")
    return parser.parse_args()

if __name__ == "__main__":
    build(parse_args())
