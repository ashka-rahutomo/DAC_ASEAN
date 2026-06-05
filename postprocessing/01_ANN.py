from __future__ import annotations

"""
01_ANN.py

Train a multi-output ANN cycle surrogate for TVSA DAC process outputs.

Default project layout supported:
    D:/Ashka/5.DAC/06.PYTHON/

Input data expected from corrected 05a post-processing:
    02.TEA_LCOD/00_CYCLE_KPI/*.csv

Output folder:
    02.TEA_LCOD/01_CYCLE_SURROGATE/<run_id>/

Surrogate inputs, canonical 11 features:
    T_ads_K
    RH_frac
    P_Pa
    p_H2O_Pa
    p_CO2_Pa
    q_H2O_GAB_mol_kg
    q_CO2_WADST_mol_kg
    adsorption_time_s
    heating_desorption_time_s
    T_des_K
    T_coolant_K

Surrogate outputs, canonical 9 targets:
    kg_CO2_cycle_corrected
    kg_H2O_cycle_corrected
    Q_heat_kWhth_cycle
    E_total_el_kWhe_cycle
    Q_cool_kWhth_cycle
    E_fan_kWhe_cycle
    E_vacuum_kWhe_cycle
    E_repress_kWhe_cycle
    E_chiller_kWhe_cycle

Notes:
- Targets are log-transformed and standardized before ANN training.
- Inputs are standardized.
- The best model is saved in both .keras and .h5 formats.
- Scalers and metadata are saved with joblib/json for later optimizer loading.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import warnings
import logging

import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow/Keras is required for this script. Install it in your environment, e.g.\n"
        "  python -m pip install tensorflow scikit-learn joblib pandas numpy\n"
        f"Original import error: {exc}"
    )

R = 8.314462618

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

FEATURE_CANDIDATES = {
    # Include _x/_y suffixes because 05a_with_design CSV can contain duplicated columns after merge.
    # Coalescing logic below uses the most complete candidate, not just the first existing column.
    "T_ads_K": ["T_ads_K", "T_ads_K_x", "T_ads_K_y", "T_amb_K", "T_amb_K_x", "T_amb_K_y", "T_K", "temperature_K", "adsorption_temperature_K"],
    "RH_frac": ["RH_frac", "RH_frac_x", "RH_frac_y", "relative_humidity_frac", "relative_humidity_frac_x", "relative_humidity_frac_y", "RH", "humidity_frac"],
    "P_Pa": ["P_Pa", "P_Pa_x", "P_Pa_y", "P_amb_Pa", "P_amb_Pa_x", "P_amb_Pa_y", "pressure_Pa", "PS_Pa"],
    "p_H2O_Pa": ["p_H2O_Pa", "p_H2O_Pa_x", "p_H2O_Pa_y", "pH2O_Pa", "water_partial_pressure_Pa"],
    "p_CO2_Pa": ["p_CO2_Pa", "p_CO2_Pa_x", "p_CO2_Pa_y", "pCO2_Pa", "co2_partial_pressure_Pa"],
    "q_H2O_GAB_mol_kg": ["q_H2O_GAB_mol_kg", "q_H2O_GAB_mol_kg_x", "q_H2O_GAB_mol_kg_y", "q_H2O_mol_kg", "q_H2O_eq_mol_kg"],
    "q_CO2_WADST_mol_kg": ["q_CO2_WADST_mol_kg", "q_CO2_WADST_mol_kg_x", "q_CO2_WADST_mol_kg_y", "q_CO2_mol_kg", "q_CO2_eq_mol_kg"],
    "adsorption_time_s": ["adsorption_time_s", "adsorption_time_s_x", "adsorption_time_s_y", "t_ads_s", "ads_time_s", "ads_time", "t_ads", "adsorption_time", "adsorption_s", "t_adsorption_s", "t_adsorption"],
    "heating_desorption_time_s": ["heating_desorption_time_s", "heating_desorption_time_s_x", "heating_desorption_time_s_y", "desorption_time_s", "desorption_time_s_x", "desorption_time_s_y", "t_des_s", "heating_time_s", "des_time_s", "des_time", "t_des", "desorption_time", "desorption_s", "heating_desorption_s", "t_heating_s"],
    "T_des_K": ["T_des_K", "T_des_K_x", "T_des_K_y", "desorption_temperature_K", "T_regen_K", "T_heating_K"],
    "T_coolant_K": ["T_coolant_K", "T_coolant_K_x", "T_coolant_K_y", "T_cool_K", "cooling_temperature_K", "T_cooling_K"],
}

TARGET_CANDIDATES = {
    "kg_CO2_cycle_corrected": ["kg_CO2_cycle_corrected", "corrected_kg_CO2_cycle", "kg_CO2_cycle"],
    "kg_H2O_cycle_corrected": ["kg_H2O_cycle_corrected", "corrected_kg_H2O_cycle", "kg_H2O_cycle"],
    "Q_heat_kWhth_cycle": ["Q_heat_kWhth_cycle"],
    "E_total_el_kWhe_cycle": ["E_total_el_kWhe_cycle", "E_total_kWhe_cycle", "E_el_kWhe_cycle"],
    "Q_cool_kWhth_cycle": ["Q_cool_kWhth_cycle", "Q_cooling_kWhth_cycle"],
    "E_fan_kWhe_cycle": ["E_fan_kWhe_cycle"],
    "E_vacuum_kWhe_cycle": ["E_vacuum_kWhe_cycle", "E_vac_kWhe_cycle"],
    "E_repress_kWhe_cycle": ["E_repress_kWhe_cycle", "E_repressurization_kWhe_cycle"],
    "E_chiller_kWhe_cycle": ["E_chiller_kWhe_cycle"],
}

# Young/Jajjawi-style GAB/WADST defaults, used only if engineered features are absent.
GAB = {
    "qm_mol_kg": 3.63,
    "C_J_mol": 47_110.0,
    "D_K_inv": 0.023744,
    "F_J_mol": 57_706.0,
    "G_J_mol_K": -47.814,
}

TOTH_DRY = {
    "T0_K": 298.15,
    "qN0_mol_kg": 4.86,
    "w": 0.0,
    "b0_Pa_inv": 2.85e-21,
    "minus_DH0_J_mol": 117_789.0,
    "t0": 0.209,
    "a": 0.523,
}

TOTH_WET = {
    "T0_K": 298.15,
    "qN0_mol_kg": 9.035,
    "w": 0.0,
    "b0_Pa_inv": 1.230e-18,
    "minus_DH0_J_mol": 203_687.0,
    "t0": 0.053,
    "a": 0.053,
}


@dataclass
class TrainConfig:
    project_dir: str
    data_dir: str
    input_csv: str
    out_dir: str
    run_id: str
    target_cases: int
    max_cases: int
    random_state: int
    test_size: float
    val_size_of_trainval: float
    max_trials: int
    full_grid: bool
    epochs: int
    patience: int
    min_delta: float
    default_co2_ppm: float
    wadst_A_mol_kg: float
    min_target_value: float
    max_mass_balance: float | None
    refit_best: bool
    threads: int
    inter_op_threads: int
    requested_workers: int
    workers_used: int
    exclude_score_targets: str


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def coalesce_numeric_columns(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series | None, str]:
    """Return a numeric Series by coalescing all existing candidate columns.

    The first non-null value in the candidate order is used, but candidates are
    sorted by non-null count first. This avoids choosing sparse columns such as
    T_ads_K when T_amb_K is complete in the 05a_with_design CSV.
    """
    cols = existing_columns(df, candidates)
    if not cols:
        return None, ""

    counts = {col: int(pd.to_numeric(df[col], errors="coerce").notna().sum()) for col in cols}
    order_rank = {col: i for i, col in enumerate(cols)}
    cols_sorted = sorted(cols, key=lambda col: (-counts[col], order_rank[col]))

    result = pd.Series(np.nan, index=df.index, dtype="float64")
    used = []
    for col in cols_sorted:
        ser = pd.to_numeric(df[col], errors="coerce")
        before = int(result.notna().sum())
        result = result.fillna(ser)
        after = int(result.notna().sum())
        if after > before:
            used.append(f"{col}({after-before})")

    note = "coalesced from " + ", ".join(used) if used else "all candidates empty: " + ", ".join(cols_sorted)
    return result, note


def fill_missing_numeric(base: pd.Series, fallback: np.ndarray | pd.Series, label: str) -> tuple[pd.Series, str]:
    base_numeric = pd.to_numeric(base, errors="coerce").astype("float64")
    fallback_series = pd.Series(fallback, index=base_numeric.index, dtype="float64")
    missing_before = int(base_numeric.isna().sum())
    out = base_numeric.fillna(fallback_series)
    filled = missing_before - int(out.isna().sum())
    return out, f"filled {filled} missing values using {label}"


def saturation_pressure_water_pa(T_K: np.ndarray) -> np.ndarray:
    T_C = np.asarray(T_K, dtype="float64") - 273.15
    p_hPa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return p_hPa * 100.0


def gab_h2o_loading(T_K: np.ndarray, RH_frac: np.ndarray) -> np.ndarray:
    T_K = np.asarray(T_K, dtype="float64")
    x = np.clip(np.asarray(RH_frac, dtype="float64"), 1e-12, 0.999999)

    qm = GAB["qm_mol_kg"]
    C = GAB["C_J_mol"]
    D = GAB["D_K_inv"]
    F = GAB["F_J_mol"]
    G = GAB["G_J_mol_K"]

    E10_plus = -44.38 * T_K + 57_220.0
    E1 = C - np.exp(D * T_K)
    E2_9 = F + G * T_K

    c_gab = np.exp((E1 - E10_plus) / (R * T_K))
    k_gab = np.exp((E2_9 - E10_plus) / (R * T_K))

    kx = np.clip(k_gab * x, 1e-12, 0.999999)
    denominator = (1.0 - kx) * (1.0 + (c_gab - 1.0) * kx)
    denominator = np.maximum(denominator, 1e-20)
    q_h2o = qm * k_gab * c_gab * x / denominator
    return np.maximum(q_h2o, 0.0)


def toth_loading(T_K: np.ndarray, p_CO2_Pa: np.ndarray, params: dict) -> np.ndarray:
    T_K = np.asarray(T_K, dtype="float64")
    p = np.maximum(np.asarray(p_CO2_Pa, dtype="float64"), 0.0)

    T0 = params["T0_K"]
    qN0 = params["qN0_mol_kg"]
    w = params["w"]
    b0 = params["b0_Pa_inv"]
    minus_DH0 = params["minus_DH0_J_mol"]
    t0 = params["t0"]
    a = params["a"]

    qN = qN0 * np.exp(w * (1.0 - T_K / T0))
    b = b0 * np.exp(minus_DH0 / (R * T_K))
    t = t0 + a * (1.0 - T0 / T_K)
    t = np.clip(t, 1e-6, None)

    bp = np.maximum(b * p, 1e-300)
    q = qN * bp / ((1.0 + np.power(bp, t)) ** (1.0 / t))
    return np.maximum(q, 0.0)


def wadst_co2_loading(T_K: np.ndarray, p_CO2_Pa: np.ndarray, q_H2O_mol_kg: np.ndarray, A: float) -> np.ndarray:
    q_dry = toth_loading(T_K, p_CO2_Pa, TOTH_DRY)
    q_wet = toth_loading(T_K, p_CO2_Pa, TOTH_WET)
    qh = np.maximum(np.asarray(q_H2O_mol_kg, dtype="float64"), 1e-12)
    w_wet = np.exp(-A / qh)
    w_wet = np.clip(w_wet, 0.0, 1.0)
    q = (1.0 - w_wet) * q_dry + w_wet * q_wet
    return np.maximum(q, 0.0)


def detect_default_paths(project_dir: Path) -> tuple[Path, Path]:
    candidate_data_dirs = [
        project_dir.parent / "02.TEA_LCOD" / "00_CYCLE_KPI",
        project_dir / "02.TEA_LCOD" / "00_CYCLE_KPI",
        Path(r"D:/Ashka/5.DAC/02.TEA_LCOD/00_CYCLE_KPI"),
        Path(r"D:/Ashka/5.DAC/06.PYTHON/02.TEA_LCOD/00_CYCLE_KPI"),
    ]

    data_dir = None
    for cand in candidate_data_dirs:
        if cand.exists():
            data_dir = cand
            break

    if data_dir is None:
        # Default to sibling 02.TEA_LCOD even if it does not exist; a clearer error is raised later.
        data_dir = project_dir.parent / "02.TEA_LCOD" / "00_CYCLE_KPI"

    out_dir = data_dir.parent / "01_CYCLE_SURROGATE"
    return data_dir, out_dir


def find_latest_input_csv(data_dir: Path) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in data folder: {data_dir}")

    preferred = [
        p for p in csvs
        if "success" in p.name.lower() and ("with_design" in p.name.lower() or "kpi" in p.name.lower())
    ]
    return preferred[0] if preferred else csvs[0]


def build_canonical_dataframe(
    df_raw: pd.DataFrame,
    default_co2_ppm: float,
    wadst_A_mol_kg: float,
    min_target_value: float,
) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    notes: dict[str, str] = {}

    out = pd.DataFrame(index=df.index)

    # 1) Read non-engineered canonical features using robust coalescing.
    #    This is critical for 05a_with_design files that contain *_x and *_y columns
    #    or sparse columns such as T_ads_K with a complete T_amb_K fallback.
    base_features = [
        "T_ads_K",
        "RH_frac",
        "P_Pa",
        "adsorption_time_s",
        "heating_desorption_time_s",
        "T_des_K",
        "T_coolant_K",
    ]
    for feature in base_features:
        ser, note = coalesce_numeric_columns(df, FEATURE_CANDIDATES[feature])
        if ser is not None:
            out[feature] = ser
            notes[feature] = note

    # Fallbacks and unit conversions for base features.
    if "T_ads_K" not in out.columns or out["T_ads_K"].isna().any():
        if "T_C" in df.columns:
            fallback = to_numeric_series(df, "T_C") + 273.15
            if "T_ads_K" in out.columns:
                out["T_ads_K"], note = fill_missing_numeric(out["T_ads_K"], fallback, "T_C + 273.15")
            else:
                out["T_ads_K"] = fallback
                note = "computed from T_C + 273.15"
            notes["T_ads_K"] = notes.get("T_ads_K", "") + "; " + note

    if "RH_frac" not in out.columns or out["RH_frac"].isna().any():
        rh_percent_col = first_existing_column(df, ["RH_percent", "RH_percent_x", "RH_percent_y", "RH2M", "relative_humidity_percent"])
        if rh_percent_col is not None:
            fallback = to_numeric_series(df, rh_percent_col) / 100.0
            if "RH_frac" in out.columns:
                out["RH_frac"], note = fill_missing_numeric(out["RH_frac"], fallback, f"{rh_percent_col}/100")
            else:
                out["RH_frac"] = fallback
                note = f"computed from {rh_percent_col}/100"
            notes["RH_frac"] = notes.get("RH_frac", "") + "; " + note

    # If RH looks like percent because values exceed 1.5, convert to fraction.
    if "RH_frac" in out.columns:
        rh_max = pd.to_numeric(out["RH_frac"], errors="coerce").max(skipna=True)
        if pd.notna(rh_max) and rh_max > 1.5:
            out["RH_frac"] = out["RH_frac"] / 100.0
            notes["RH_frac"] = notes.get("RH_frac", "") + "; divided by 100 because values looked like percent"
        out["RH_frac"] = out["RH_frac"].clip(lower=0.0, upper=1.0)

    if "P_Pa" not in out.columns or out["P_Pa"].isna().any():
        ps_kpa_col = first_existing_column(df, ["PS_kPa", "PS_kPa_x", "PS_kPa_y", "PS", "pressure_kPa", "P_kPa"])
        if ps_kpa_col is not None:
            fallback = to_numeric_series(df, ps_kpa_col) * 1000.0
            if "P_Pa" in out.columns:
                out["P_Pa"], note = fill_missing_numeric(out["P_Pa"], fallback, f"{ps_kpa_col}*1000")
            else:
                out["P_Pa"] = fallback
                note = f"computed from {ps_kpa_col}*1000"
            notes["P_Pa"] = notes.get("P_Pa", "") + "; " + note

    # 2) Engineered features: read if available, but always fill missing values
    #    using T-RH-P calculations where possible.
    for feature in ["p_H2O_Pa", "p_CO2_Pa", "q_H2O_GAB_mol_kg", "q_CO2_WADST_mol_kg"]:
        ser, note = coalesce_numeric_columns(df, FEATURE_CANDIDATES[feature])
        if ser is not None:
            out[feature] = ser
            notes[feature] = note

    if {"T_ads_K", "RH_frac", "P_Pa"}.issubset(out.columns):
        T_K = out["T_ads_K"].to_numpy(dtype="float64")
        RH = np.clip(out["RH_frac"].to_numpy(dtype="float64"), 0.0, 1.0)
        P = out["P_Pa"].to_numpy(dtype="float64")

        p_sat = saturation_pressure_water_pa(T_K)
        p_h2o_calc = np.minimum(RH * p_sat, 0.99 * P)
        if "p_H2O_Pa" in out.columns:
            out["p_H2O_Pa"], note = fill_missing_numeric(out["p_H2O_Pa"], p_h2o_calc, "T_ads_K, RH_frac, saturation pressure")
        else:
            out["p_H2O_Pa"] = p_h2o_calc
            note = "computed from T_ads_K, RH_frac, and saturation pressure"
        notes["p_H2O_Pa"] = notes.get("p_H2O_Pa", "") + "; " + note

        co2_col = first_existing_column(df, ["CO2_ppm", "CO2_ppm_x", "CO2_ppm_y", "co2_ppm"])
        if co2_col is not None:
            co2_ppm = to_numeric_series(df, co2_col).fillna(default_co2_ppm).to_numpy(dtype="float64")
            co2_note = f"from {co2_col}"
        else:
            co2_ppm = np.full(len(df), default_co2_ppm, dtype="float64")
            co2_note = f"default {default_co2_ppm} ppm"
        p_co2_calc = np.maximum(co2_ppm * 1e-6 * (P - out["p_H2O_Pa"].to_numpy(dtype="float64")), 0.0)
        if "p_CO2_Pa" in out.columns:
            out["p_CO2_Pa"], note = fill_missing_numeric(out["p_CO2_Pa"], p_co2_calc, f"P_Pa, p_H2O_Pa, CO2 ppm ({co2_note})")
        else:
            out["p_CO2_Pa"] = p_co2_calc
            note = f"computed from P_Pa, p_H2O_Pa, and CO2 ppm ({co2_note})"
        notes["p_CO2_Pa"] = notes.get("p_CO2_Pa", "") + "; " + note

        q_h2o_calc = gab_h2o_loading(T_K, RH)
        if "q_H2O_GAB_mol_kg" in out.columns:
            out["q_H2O_GAB_mol_kg"], note = fill_missing_numeric(out["q_H2O_GAB_mol_kg"], q_h2o_calc, "GAB from T_ads_K and RH_frac")
        else:
            out["q_H2O_GAB_mol_kg"] = q_h2o_calc
            note = "computed with GAB from T_ads_K and RH_frac"
        notes["q_H2O_GAB_mol_kg"] = notes.get("q_H2O_GAB_mol_kg", "") + "; " + note

        q_co2_calc = wadst_co2_loading(
            T_K,
            out["p_CO2_Pa"].to_numpy(dtype="float64"),
            out["q_H2O_GAB_mol_kg"].to_numpy(dtype="float64"),
            A=wadst_A_mol_kg,
        )
        if "q_CO2_WADST_mol_kg" in out.columns:
            out["q_CO2_WADST_mol_kg"], note = fill_missing_numeric(out["q_CO2_WADST_mol_kg"], q_co2_calc, f"WADST, A={wadst_A_mol_kg} mol/kg")
        else:
            out["q_CO2_WADST_mol_kg"] = q_co2_calc
            note = f"computed with WADST, A={wadst_A_mol_kg} mol/kg"
        notes["q_CO2_WADST_mol_kg"] = notes.get("q_CO2_WADST_mol_kg", "") + "; " + note

    # 3) Canonical targets. Coalesce as well, so corrected columns are used if present,
    #    with raw columns as fallback.
    for target in CANONICAL_TARGETS:
        ser, note = coalesce_numeric_columns(df, TARGET_CANDIDATES[target])
        if ser is not None:
            out[target] = ser
            notes[target] = note

    missing_features = [c for c in CANONICAL_FEATURES if c not in out.columns]
    missing_targets = [c for c in CANONICAL_TARGETS if c not in out.columns]
    if missing_features or missing_targets:
        message = []
        if missing_features:
            message.append("Missing required features: " + ", ".join(missing_features))
        if missing_targets:
            message.append("Missing required targets: " + ", ".join(missing_targets))
        message.append("Available similar columns: " + ", ".join([c for c in df.columns if any(k in c.lower() for k in ["ads", "des", "time", "cool", "heat", "rh", "pressure", "p_"])]))
        raise ValueError("\n".join(message))

    # Keep useful identifiers/metadata if present.
    for col in ["design_row", "design_row_1based", "case_folder", "case_status", "country_code", "country_name", "province_id", "province_name", "datetime_utc", "datetime_local"]:
        if col in df.columns and col not in out.columns:
            out[col] = df[col]

    # Remove non-finite rows. Allow zero targets by default for quantities like E_repress.
    for col in CANONICAL_FEATURES + CANONICAL_TARGETS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    mask = np.ones(len(out), dtype=bool)
    numeric_block = out[CANONICAL_FEATURES + CANONICAL_TARGETS].to_numpy(dtype="float64")
    mask &= np.isfinite(numeric_block).all(axis=1)

    for target in CANONICAL_TARGETS:
        mask &= out[target].to_numpy(dtype="float64") >= min_target_value

    removed = int((~mask).sum())
    out = out.loc[mask].copy().reset_index(drop=True)
    notes["rows_removed_nonfinite_or_below_min_target"] = str(removed)
    notes["rows_used_after_canonical_filter"] = str(len(out))

    return out, notes


def make_architectures() -> list[tuple[int, ...]]:
    """Architectures from 1 to 3 hidden layers under the user's constraints."""
    archs: list[tuple[int, ...]] = []

    # 1 hidden layer: 16, 32, 64, 128.
    for h in [16, 32, 64, 128]:
        archs.append((h,))

    # 2 hidden layers. Allow 128 only as first layer followed by <=64, no 128-128.
    for h1 in [16, 32, 64]:
        for h2 in [16, 32, 64]:
            archs.append((h1, h2))
    for h2 in [16, 32, 64]:
        archs.append((128, h2))

    # 3 hidden layers: max 64-64-64, values from 16/32/64 only.
    for h1 in [16, 32, 64]:
        for h2 in [16, 32, 64]:
            for h3 in [16, 32, 64]:
                archs.append((h1, h2, h3))

    # Unique while preserving order.
    seen = set()
    unique = []
    for arch in archs:
        if arch not in seen:
            unique.append(arch)
            seen.add(arch)
    return unique


def build_model(
    input_dim: int,
    output_dim: int,
    hidden_layers: tuple[int, ...],
    activation: str,
    learning_rate: float,
    l2_value: float,
) -> keras.Model:
    model = keras.Sequential(name="cycle_ann_surrogate")
    model.add(layers.Input(shape=(input_dim,), name="cycle_features"))

    for i, units in enumerate(hidden_layers, start=1):
        model.add(
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_value) if l2_value > 0 else None,
                name=f"dense_{i}_{units}",
            )
        )

    model.add(layers.Dense(output_dim, activation="linear", name="scaled_log_targets"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def inverse_transform_targets(y_scaled: np.ndarray, y_scaler: StandardScaler, eps: float) -> np.ndarray:
    y_log = y_scaler.inverse_transform(y_scaled)
    y = np.exp(y_log) - eps
    return np.maximum(y, 0.0)


def transform_targets(y_raw: np.ndarray, y_scaler: StandardScaler | None, eps: float, fit: bool = False) -> tuple[np.ndarray, StandardScaler]:
    y_log = np.log(np.maximum(y_raw, eps) + eps)
    if fit:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_log)
        return y_scaled, scaler
    if y_scaler is None:
        raise ValueError("y_scaler must be provided when fit=False")
    return y_scaler.transform(y_log), y_scaler


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    eps = 1e-12
    for j, target in enumerate(targets):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        abs_err = np.abs(yp - yt)
        rel_err = abs_err / np.maximum(np.abs(yt), eps) * 100.0
        rows.append({
            "target": target,
            "n": int(len(yt)),
            "mean_true": float(np.mean(yt)),
            "mean_pred": float(np.mean(yp)),
            "r2": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
            "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "mape_percent": float(np.mean(rel_err)),
            "median_ape_percent": float(np.median(rel_err)),
            "p95_ape_percent": float(np.percentile(rel_err, 95)),
            "max_ape_percent": float(np.max(rel_err)),
        })
    return pd.DataFrame(rows)


def composite_score(metrics_df: pd.DataFrame, score_targets: list[str] | None = None) -> float:
    """Lower is better. Median APE is robust for positive process targets.

    Very small auxiliary outputs such as E_repress_kWhe_cycle can have unstable
    relative error and should not dominate model selection. They can still be
    trained and reported, but excluded from the composite score.
    """
    df = metrics_df.copy()
    if score_targets is not None:
        df = df[df["target"].isin(score_targets)]
    if df.empty:
        df = metrics_df.copy()
    return float(df["median_ape_percent"].mean())




def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ann_run_{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def append_run_metric(run_metrics_csv: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    mode = "a" if run_metrics_csv.exists() else "w"
    header = not run_metrics_csv.exists()
    df.to_csv(run_metrics_csv, mode=mode, header=header, index=False, encoding="utf-8-sig")


def plot_loss_curve(history: dict, save_path: Path, title: str) -> None:
    if not history or "loss" not in history:
        return
    plt.figure(figsize=(9, 6))
    plt.plot(history.get("loss", []), label="train_loss", linewidth=2)
    if "val_loss" in history:
        plt.plot(history.get("val_loss", []), label="val_loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled log MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_lr_curve(epoch_log_csv: Path, save_path: Path, title: str) -> None:
    if not epoch_log_csv.exists():
        return
    df = pd.read_csv(epoch_log_csv)
    lr_col = None
    for c in df.columns:
        cl = c.lower()
        if "learning_rate" in cl or cl == "lr":
            lr_col = c
            break
    if lr_col is None:
        return
    plt.figure(figsize=(9, 6))
    plt.plot(df[lr_col].values, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_parity_multi(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str], save_path: Path, title: str) -> None:
    n = len(targets)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5*ncols, 4.5*nrows), squeeze=False)
    axes = axes.flatten()
    for j, target in enumerate(targets):
        ax = axes[j]
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ax.scatter(yt, yp, s=18, alpha=0.7)
        minv = float(min(np.min(yt), np.min(yp)))
        maxv = float(max(np.max(yt), np.max(yp)))
        if maxv - minv <= 0:
            maxv = minv + 1.0
        pad = 0.05 * (maxv - minv)
        lo, hi = minv - pad, maxv + pad
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float('nan')
        ax.set_title(f"{target}\nR²={r2:.4f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True, linestyle="--", alpha=0.3)
    for k in range(n, len(axes)):
        fig.delaxes(axes[k])
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(save_path, dpi=250)
    plt.close()


def prepare_grid(max_trials: int, full_grid: bool, seed: int) -> list[dict]:
    archs = make_architectures()
    activations = ["relu", "elu"]
    learning_rates = [1e-3, 5e-4]
    l2_values = [0.0, 1e-5, 1e-4]
    batch_sizes = [32, 64]

    all_trials = []
    for arch, activation, lr, l2_val, batch in product(archs, activations, learning_rates, l2_values, batch_sizes):
        all_trials.append({
            "hidden_layers": arch,
            "activation": activation,
            "learning_rate": lr,
            "l2": l2_val,
            "batch_size": batch,
        })

    if full_grid or max_trials <= 0 or max_trials >= len(all_trials):
        return all_trials

    rng = random.Random(seed)
    # Always include a few small canonical architectures, then random sample the rest.
    must_include_archs = [(16,), (32,), (64,), (32, 32), (64, 64), (32, 32, 32), (64, 64, 64), (128, 64)]
    must = [t for t in all_trials if t["hidden_layers"] in must_include_archs and t["activation"] == "relu" and t["learning_rate"] == 1e-3 and t["l2"] in [0.0, 1e-5] and t["batch_size"] == 32]

    remaining = [t for t in all_trials if t not in must]
    rng.shuffle(remaining)
    selected = must[:max_trials]
    if len(selected) < max_trials:
        selected.extend(remaining[: max_trials - len(selected)])
    return selected


def train_one_trial(
    trial: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val_raw: np.ndarray,
    y_scaler: StandardScaler,
    targets: list[str],
    score_targets: list[str],
    epochs: int,
    patience: int,
    min_delta: float,
    eps: float,
    seed: int,
    trial_dir: Path | None = None,
    reduce_lr_patience: int = 12,
    min_lr: float = 1e-6,
) -> tuple[keras.Model, dict, pd.DataFrame, dict, Path | None]:
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = build_model(
        input_dim=X_train.shape[1],
        output_dim=len(targets),
        hidden_layers=trial["hidden_layers"],
        activation=trial["activation"],
        learning_rate=trial["learning_rate"],
        l2_value=trial["l2"],
    )

    csv_log_path = None
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, reduce_lr_patience),
            min_lr=min_lr,
            verbose=0,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    if trial_dir is not None:
        trial_dir.mkdir(parents=True, exist_ok=True)
        csv_log_path = trial_dir / "epoch_log.csv"
        callbacks.append(keras.callbacks.CSVLogger(str(csv_log_path), append=False))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, transform_targets(y_val_raw, y_scaler, eps, fit=False)[0]),
        epochs=epochs,
        batch_size=trial["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )

    val_pred_scaled = model.predict(X_val, verbose=0)
    val_pred_raw = inverse_transform_targets(val_pred_scaled, y_scaler, eps)
    metrics = regression_metrics(y_val_raw, val_pred_raw, targets)

    result = {
        "hidden_layers": "-".join(map(str, trial["hidden_layers"])),
        "n_hidden_layers": len(trial["hidden_layers"]),
        "activation": trial["activation"],
        "learning_rate": trial["learning_rate"],
        "l2": trial["l2"],
        "batch_size": trial["batch_size"],
        "epochs_ran": len(history.history.get("loss", [])),
        "best_val_loss_scaled_log": float(np.min(history.history.get("val_loss", [np.nan]))),
        "final_train_loss_scaled_log": float(history.history.get("loss", [np.nan])[-1]),
        "final_val_loss_scaled_log": float(history.history.get("val_loss", [np.nan])[-1]),
        "val_composite_median_ape_percent": composite_score(metrics, score_targets),
        "score_targets": ";".join(score_targets),
        "val_mean_r2": float(metrics["r2"].mean()),
        "val_mean_mape_percent": float(metrics["mape_percent"].mean()),
        "n_parameters": int(model.count_params()),
    }
    return model, result, metrics, history.history, csv_log_path


def save_load_example(run_dir: Path) -> None:
    text = r'''from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

run_dir = Path(__file__).resolve().parent
model = tf.keras.models.load_model(run_dir / "best_ann_model.keras")
x_scaler = joblib.load(run_dir / "x_scaler.pkl")
y_scaler = joblib.load(run_dir / "y_scaler.pkl")
meta = json.loads((run_dir / "surrogate_metadata.json").read_text(encoding="utf-8"))
features = meta["feature_columns"]
targets = meta["target_columns"]
eps = meta["target_log_epsilon"]

# Example: df must contain the canonical feature columns.
def predict_cycle_outputs(df: pd.DataFrame) -> pd.DataFrame:
    X = df[features].to_numpy(dtype="float64")
    Xs = x_scaler.transform(X)
    y_scaled = model.predict(Xs, verbose=0)
    y_log = y_scaler.inverse_transform(y_scaled)
    y = np.exp(y_log) - eps
    y = np.maximum(y, 0.0)
    return pd.DataFrame(y, columns=targets, index=df.index)
'''
    (run_dir / "load_surrogate_example.py").write_text(text, encoding="utf-8")


def prompt_int(prompt: str, default: int, minimum: int = 1) -> int:
    if not sys.stdin.isatty():
        return default
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(minimum, value)
    except ValueError:
        print(f"Invalid input; using default {default}.")
        return default


def configure_tensorflow_threads(threads: int, inter_op_threads: int | None = None) -> tuple[int, int]:
    threads = max(1, int(threads))
    if inter_op_threads is None:
        inter_op_threads = max(1, min(2, threads // 2 if threads > 1 else 1))
    inter_op_threads = max(1, int(inter_op_threads))

    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(inter_op_threads)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except Exception as exc:
        warnings.warn(f"Could not set TensorFlow thread config after import: {exc}")
    return threads, inter_op_threads


def get_score_targets(exclude_csv: str) -> list[str]:
    exclude = {x.strip() for x in str(exclude_csv).split(",") if x.strip()}
    return [t for t in CANONICAL_TARGETS if t not in exclude]



def main() -> None:
    args = parse_args()

    if args.prompt_resources:
        if args.threads is None:
            args.threads = prompt_int("TensorFlow CPU threads / intra-op threads", default=4, minimum=1)
        if args.workers is None:
            args.workers = prompt_int("Grid-search workers / parallel trials", default=1, minimum=1)
    else:
        args.threads = args.threads or 4
        args.workers = args.workers or 1

    if args.workers > 1:
        print(
            f"[INFO] Requested workers={args.workers}, but this Keras/TensorFlow grid search will run trials sequentially. "
            "Parallel ANN trials often duplicate TensorFlow runtime memory and can be slower/unstable on CPU. "
            "The worker value is recorded in metadata only."
        )

    actual_threads, actual_inter_threads = configure_tensorflow_threads(args.threads, args.inter_op_threads)
    print(f"[INFO] TensorFlow threads: intra_op={actual_threads}, inter_op={actual_inter_threads}; grid workers used=1")

    set_reproducibility(args.random_state)

    project_dir = Path(args.project_dir)
    default_data_dir, default_out_base = detect_default_paths(project_dir)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir
    out_base = Path(args.out_dir) if args.out_dir else default_out_base

    input_csv = Path(args.input_csv) if args.input_csv else find_latest_input_csv(data_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    run_id = args.run_id or datetime.now().strftime("ANN_%Y%m%d_%H%M%S")
    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logging and visualization folders.
    logger = setup_logger(run_dir / "run_log.txt")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    trial_logs_dir = run_dir / "trial_logs"
    trial_logs_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_csv = run_dir / "run_log_metrics.csv"

    print("=" * 100)
    print("01_ANN TVSA CYCLE SURROGATE TRAINING")
    print("=" * 100)
    print(f"Project dir : {project_dir}")
    print(f"Data dir    : {data_dir}")
    print(f"Input CSV   : {input_csv}")
    print(f"Output dir  : {run_dir}")
    print("=" * 100)

    df_raw = pd.read_csv(input_csv)
    df, feature_notes = build_canonical_dataframe(
        df_raw,
        default_co2_ppm=args.default_co2_ppm,
        wadst_A_mol_kg=args.wadst_A_mol_kg,
        min_target_value=args.min_target_value,
    )

    if args.max_mass_balance is not None and "mass_balance_max_abs" in df_raw.columns:
        # Apply by matching after canonical build only if lengths match original filtered index is unavailable.
        # Safer route: rebuild mask on original rows before canonical reset if strict mass balance is needed.
        warnings.warn(
            "--max-mass-balance is requested but canonical filtering resets row indices. "
            "For strict MB filtering, pre-filter the input CSV or keep mass_balance_max_abs in 00_CYCLE_KPI."
        )

    if len(df) < 100:
        warnings.warn(f"Only {len(df)} usable rows found. ANN training may be unreliable.")

    if args.max_cases > 0 and len(df) > args.max_cases:
        df = df.sample(n=args.max_cases, random_state=args.random_state).reset_index(drop=True)
        print(f"[INFO] Sampled {args.max_cases} rows from available dataset for training.")

    print(f"Usable rows : {len(df):,}")
    print(f"Max cases   : {'all usable rows' if args.max_cases == 0 else args.max_cases}")
    print(f"Score excludes: {args.exclude_score_targets if args.exclude_score_targets else 'none'}")
    if len(df) < args.target_cases:
        print(f"[WARNING] Usable rows below target {args.target_cases:,}. Script will still train.")

    X_raw = df[CANONICAL_FEATURES].to_numpy(dtype="float64")
    y_raw = df[CANONICAL_TARGETS].to_numpy(dtype="float64")

    # Split: train/val/test.
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X_raw,
        y_raw,
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )
    X_train, X_val, y_train_raw, y_val_raw, idx_train, idx_val = train_test_split(
        X_trainval,
        y_trainval,
        idx_trainval,
        test_size=args.val_size_of_trainval,
        random_state=args.random_state,
        shuffle=True,
    )

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    X_trainval_s = x_scaler.fit_transform(X_trainval) if args.refit_best else None

    y_train_s, y_scaler = transform_targets(y_train_raw, y_scaler=None, eps=args.target_log_epsilon, fit=True)

    trials = prepare_grid(args.max_trials, args.full_grid, args.random_state)
    logger.info(f"Grid trials: {len(trials)}")

    results = []
    best = None
    best_model = None
    best_metrics = None
    best_history = None
    best_epoch_log_csv = None

    for i, trial in enumerate(trials, start=1):
        logger.info(
            f"[TRIAL {i:03d}/{len(trials):03d}] "
            f"layers={trial['hidden_layers']} activation={trial['activation']} "
            f"lr={trial['learning_rate']} l2={trial['l2']} batch={trial['batch_size']}"
        )
        try:
            trial_dir = trial_logs_dir / f"trial_{i:03d}_{'-'.join(map(str, trial['hidden_layers']))}_{trial['activation']}"
            model, result, metrics, history_dict, epoch_log_csv = train_one_trial(
                trial=trial,
                X_train=X_train_s,
                y_train=y_train_s,
                X_val=X_val_s,
                y_val_raw=y_val_raw,
                y_scaler=y_scaler,
                targets=CANONICAL_TARGETS,
                score_targets=get_score_targets(args.exclude_score_targets),
                epochs=args.epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                eps=args.target_log_epsilon,
                seed=args.random_state + i,
                trial_dir=trial_dir,
                reduce_lr_patience=args.reduce_lr_patience,
                min_lr=args.min_lr,
            )
            result["trial_no"] = i
            result["status"] = "ok"
            result["epoch_log_csv"] = str(epoch_log_csv) if epoch_log_csv is not None else ""
            results.append(result)
            append_run_metric(run_metrics_csv, result)

            if best is None or result["val_composite_median_ape_percent"] < best["val_composite_median_ape_percent"]:
                best = result
                best_model = model
                best_metrics = metrics.copy()
                best_history = history_dict.copy() if history_dict is not None else None
                best_epoch_log_csv = epoch_log_csv
                logger.info(f"  -> new best composite median APE: {best['val_composite_median_ape_percent']:.3f}%")
        except Exception as exc:
            fail_row = {
                "trial_no": i,
                "hidden_layers": "-".join(map(str, trial["hidden_layers"])),
                "activation": trial["activation"],
                "learning_rate": trial["learning_rate"],
                "l2": trial["l2"],
                "batch_size": trial["batch_size"],
                "status": "failed",
                "error": str(exc),
            }
            results.append(fail_row)
            append_run_metric(run_metrics_csv, fail_row)
            logger.info(f"  -> failed: {exc}")

    results_df = pd.DataFrame(results).sort_values(
        by=["status", "val_composite_median_ape_percent"],
        ascending=[True, True],
        na_position="last",
    )
    results_df.to_csv(run_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")

    if best is None or best_model is None:
        raise RuntimeError("No successful ANN trials. Check TensorFlow installation and data columns.")

    logger.info("=" * 100)
    logger.info("BEST VALIDATION MODEL")
    logger.info(json.dumps(best, indent=2))
    logger.info("=" * 100)

    # Optionally refit best architecture on train+val with fresh scalers, preserving test set.
    if args.refit_best:
        logger.info("[REFIT] Re-fitting best architecture on train+validation data.")
        x_scaler = StandardScaler()
        X_trainval_s = x_scaler.fit_transform(X_trainval)
        X_test_s = x_scaler.transform(X_test)
        y_trainval_s, y_scaler = transform_targets(y_trainval, y_scaler=None, eps=args.target_log_epsilon, fit=True)

        # internal small validation split for early stopping during refit
        X_refit_train, X_refit_val, y_refit_train, y_refit_val = train_test_split(
            X_trainval_s,
            y_trainval_s,
            test_size=0.12,
            random_state=args.random_state,
            shuffle=True,
        )

        hidden_layers = tuple(int(x) for x in str(best["hidden_layers"]).split("-"))
        final_model = build_model(
            input_dim=X_trainval_s.shape[1],
            output_dim=len(CANONICAL_TARGETS),
            hidden_layers=hidden_layers,
            activation=best["activation"],
            learning_rate=float(best["learning_rate"]),
            l2_value=float(best["l2"]),
        )
        refit_epoch_log_csv = trial_logs_dir / "best_model_refit_epoch_log.csv"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                min_delta=args.min_delta,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, args.reduce_lr_patience),
                min_lr=args.min_lr,
                verbose=0,
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(str(refit_epoch_log_csv), append=False),
        ]
        history = final_model.fit(
            X_refit_train,
            y_refit_train,
            validation_data=(X_refit_val, y_refit_val),
            epochs=args.epochs,
            batch_size=int(best["batch_size"]),
            verbose=0,
            callbacks=callbacks,
        )
        best_model = final_model
        best["refit_epochs_ran"] = len(history.history.get("loss", []))
        best_history = history.history.copy()
        best_epoch_log_csv = refit_epoch_log_csv

    # Final evaluation on train / validation / held-out test using the final saved model and current scalers.
    X_train_eval_s = x_scaler.transform(X_train)
    X_val_eval_s = x_scaler.transform(X_val)
    X_test_eval_s = x_scaler.transform(X_test)

    y_train_pred = inverse_transform_targets(best_model.predict(X_train_eval_s, verbose=0), y_scaler, args.target_log_epsilon)
    y_val_pred = inverse_transform_targets(best_model.predict(X_val_eval_s, verbose=0), y_scaler, args.target_log_epsilon)
    y_test_pred = inverse_transform_targets(best_model.predict(X_test_eval_s, verbose=0), y_scaler, args.target_log_epsilon)

    # Use the raw target arrays from the original train/validation/test split.
    # The previous version used undefined names y_train and y_val after refit.
    train_metrics = regression_metrics(y_train_raw, y_train_pred, CANONICAL_TARGETS)
    val_metrics = regression_metrics(y_val_raw, y_val_pred, CANONICAL_TARGETS)
    test_metrics = regression_metrics(y_test, y_test_pred, CANONICAL_TARGETS)

    train_metrics.to_csv(run_dir / "metrics_train.csv", index=False, encoding="utf-8-sig")
    val_metrics.to_csv(run_dir / "metrics_val.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(run_dir / "metrics_test.csv", index=False, encoding="utf-8-sig")

    # Backward-compatible filename for previous workflow.
    val_metrics.to_csv(run_dir / "metrics_validation_best_trial.csv", index=False, encoding="utf-8-sig")

    if best_history is not None:
        plot_loss_curve(best_history, figures_dir / "best_model_loss_curve.png", "Best ANN model loss curve")
    if best_epoch_log_csv is not None:
        plot_lr_curve(best_epoch_log_csv, figures_dir / "best_model_learning_rate_curve.png", "Best ANN model learning-rate curve")
    plot_parity_multi(y_train_raw, y_train_pred, CANONICAL_TARGETS, figures_dir / "parity_plot_train.png", "Parity plot - train")
    plot_parity_multi(y_val_raw, y_val_pred, CANONICAL_TARGETS, figures_dir / "parity_plot_val.png", "Parity plot - validation")
    plot_parity_multi(y_test, y_test_pred, CANONICAL_TARGETS, figures_dir / "parity_plot_test.png", "Parity plot - test")

    pred_rows = []
    for row_pos, true_vals, pred_vals in zip(idx_test, y_test, y_test_pred):
        rec = {"canonical_row_index": int(row_pos)}
        # preserve identifiers if available
        for col in ["design_row", "case_folder", "country_code", "country_name", "province_id", "province_name"]:
            if col in df.columns:
                rec[col] = df.loc[row_pos, col]
        for target, tv, pv in zip(CANONICAL_TARGETS, true_vals, pred_vals):
            rec[f"true_{target}"] = tv
            rec[f"pred_{target}"] = pv
            rec[f"ape_percent_{target}"] = abs(pv - tv) / max(abs(tv), 1e-12) * 100.0
        pred_rows.append(rec)
    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # Save model and preprocessing.
    best_model.save(run_dir / "best_ann_model.keras")
    best_model.save(run_dir / "best_ann_model.h5")
    joblib.dump(x_scaler, run_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, run_dir / "y_scaler.pkl")

    dataset_used_path = run_dir / "dataset_used_canonical.csv"
    df.to_csv(dataset_used_path, index=False, encoding="utf-8-sig")

    config = TrainConfig(
        project_dir=str(project_dir),
        data_dir=str(data_dir),
        input_csv=str(input_csv),
        out_dir=str(run_dir),
        run_id=run_id,
        target_cases=args.target_cases,
        max_cases=args.max_cases,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size_of_trainval=args.val_size_of_trainval,
        max_trials=args.max_trials,
        full_grid=args.full_grid,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        default_co2_ppm=args.default_co2_ppm,
        wadst_A_mol_kg=args.wadst_A_mol_kg,
        min_target_value=args.min_target_value,
        max_mass_balance=args.max_mass_balance,
        refit_best=args.refit_best,
        threads=int(actual_threads),
        inter_op_threads=int(actual_inter_threads),
        requested_workers=int(args.workers),
        workers_used=1,
        exclude_score_targets=args.exclude_score_targets,
    )

    metadata = {
        "model_type": "multi_output_ANN_scaled_log_targets",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": CANONICAL_FEATURES,
        "target_columns": CANONICAL_TARGETS,
        "score_target_columns": get_score_targets(args.exclude_score_targets),
        "excluded_from_score": [x.strip() for x in str(args.exclude_score_targets).split(",") if x.strip()],
        "target_log_epsilon": args.target_log_epsilon,
        "best_hyperparameters": best,
        "feature_source_notes": feature_notes,
        "n_rows_used": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "config": asdict(config),
    }
    (run_dir / "surrogate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (run_dir / "best_model_info.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    save_load_example(run_dir)

    readme = f"""01_ANN cycle surrogate training output

Run ID:
{run_id}

Input CSV:
{input_csv}

Rows used:
{len(df)}

Canonical features:
{chr(10).join('- ' + c for c in CANONICAL_FEATURES)}

Canonical targets:
{chr(10).join('- ' + c for c in CANONICAL_TARGETS)}

Best validation hyperparameters:
{json.dumps(best, indent=2)}

Score targets used for model selection:
{chr(10).join('- ' + c for c in get_score_targets(args.exclude_score_targets))}

Resource settings:
- TensorFlow intra-op threads: {actual_threads}
- TensorFlow inter-op threads: {actual_inter_threads}
- Requested workers: {args.workers}
- Workers actually used for ANN trials: 1

Important files:
- best_ann_model.keras
- best_ann_model.h5
- x_scaler.pkl
- y_scaler.pkl
- surrogate_metadata.json
- grid_search_results.csv
- metrics_test.csv
- predictions_test.csv
- load_surrogate_example.py

Use load_surrogate_example.py as a template for optimizer-side loading and prediction.
"""
    (run_dir / "README_01_ANN.txt").write_text(readme, encoding="utf-8")

    print("=" * 100)
    print("TRAINING COMPLETE")
    print(f"Run dir       : {run_dir}")
    print(f"Best .keras   : {run_dir / 'best_ann_model.keras'}")
    print(f"Best .h5      : {run_dir / 'best_ann_model.h5'}")
    print(f"Test metrics  : {run_dir / 'metrics_test.csv'}")
    print("=" * 100)
    print(test_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN TVSA cycle surrogate from corrected 05a KPI data.")
    parser.add_argument("--project-dir", default=r"D:/Ashka/5.DAC/06.PYTHON")
    parser.add_argument("--data-dir", default=None, help="Folder containing corrected 05a KPI CSV. Default: auto-detect 02.TEA_LCOD/00_CYCLE_KPI.")
    parser.add_argument("--input-csv", default=None, help="Specific corrected KPI CSV. Default: latest suitable CSV in data-dir.")
    parser.add_argument("--out-dir", default=None, help="Output base folder. Default: sibling 01_CYCLE_SURROGATE next to data-dir.")
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--target-cases", type=int, default=3000)
    parser.add_argument("--max-cases", type=int, default=0, help="Use up to this many rows. Default 0 = use all usable rows from the CSV.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size-of-trainval", type=float, default=0.1765, help="0.1765 gives ~15% overall validation after 85% trainval split.")

    parser.add_argument("--max-trials", type=int, default=60, help="Default samples 60 architecture/hyperparameter trials. Use 0 with --full-grid for all.")
    parser.add_argument("--full-grid", action="store_true", help="Run the complete grid. This can be slow.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--reduce-lr-patience", type=int, default=12)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--refit-best", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--prompt-resources", action=argparse.BooleanOptionalAction, default=True, help="Ask interactively for TensorFlow threads and grid-search workers in the terminal.")
    parser.add_argument("--threads", type=int, default=None, help="TensorFlow intra-op CPU threads. If omitted and prompting is enabled, asked in terminal.")
    parser.add_argument("--inter-op-threads", type=int, default=None, help="TensorFlow inter-op threads. Default derived from --threads.")
    parser.add_argument("--workers", type=int, default=None, help="Requested grid-search workers. Currently recorded only; Keras trials are run sequentially for stability.")

    parser.add_argument("--target-log-epsilon", type=float, default=1e-12)
    parser.add_argument("--min-target-value", type=float, default=0.0)
    parser.add_argument("--max-mass-balance", type=float, default=None)
    parser.add_argument("--exclude-score-targets", default="E_repress_kWhe_cycle", help="Comma-separated targets excluded from grid-search composite score but still trained/reported.")
    parser.add_argument("--default-co2-ppm", type=float, default=400.0)
    parser.add_argument("--wadst-A-mol-kg", type=float, default=1.523)
    return parser.parse_args()


if __name__ == "__main__":
    main()
