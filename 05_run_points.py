# -*- coding: utf-8 -*-
"""
05_run_surrogate_training_tvsa.py

Run selected surrogate-training TVSA simulations using the detailed 1D TVSA
engine defined in:
    01_jakarta_single_bed_tvsa.py

Typical input from script 04:
    outputs/surrogate_design/surrogate_training_design_300.csv

Main features
-------------
- Can run a selected row range only, e.g. rows 32 to 188.
- Can choose the number of parallel workers.
- Uses engine 01 through dynamic import.
- Enforces detailed full ODE/PDE-MOL mode from script 01:
    * no fast pressure substeps
    * no fast heating split
    * no inert freeze during heating
    * no near-vacuum Tg≈Ts shortcut
    * no nondimensional-state mode by default
- Resume-safe: completed case folders are skipped unless --force-rerun is used.
- Timeout-safe: cases are marked timeout and skipped onward when a step exceeds walltime/nfev limits.
- Per-case outputs are not deleted by default.
- Batch consolidated outputs are written under:
    outputs/surrogate_tvsa/batches/<batch_label>/

Project path:
    D:/Ashka/5.DAC/06.PYTHON

Example commands
----------------
Run rows 32 to 188 with 3 workers:
    python 05_run_surrogate_training_tvsa.py --start-row 32 --end-row 188 --workers 3

Run rows 189 to 292 with 2 workers:
    python 05_run_surrogate_training_tvsa.py --start-row 189 --end-row 292 --workers 2

Run only sensitivity-anchor cases:
    python 05_run_surrogate_training_tvsa.py --design-type sensitivity_anchor --workers 2

Force rerun selected rows:
    python 05_run_surrogate_training_tvsa.py --start-row 32 --end-row 40 --force-rerun
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import importlib.util
import json
import re
import shutil
import sys
import traceback
from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT PATHS
# =============================================================================

PROJECT_DIR_DEFAULT = Path(r"D:/Ashka/5.DAC/06.PYTHON")
SCRIPT01_DEFAULT = PROJECT_DIR_DEFAULT / "01_jakarta_single_bed_tvsa.py"
DESIGN_CSV_DEFAULT = (
    PROJECT_DIR_DEFAULT
    / "outputs"
    / "surrogate_design"
    / "surrogate_training_design_300.csv"
)
OUT_DIR_DEFAULT = PROJECT_DIR_DEFAULT / "outputs" / "surrogate_tvsa"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def safe_name(text: Any, max_len: int = 80) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "NA"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def row_value(row: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def backup_existing_file(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
        shutil.copy2(path, backup)


class StepTimeoutError(RuntimeError):
    """Raised when a TVSA solve step exceeds batch timeout limits."""

    def __init__(
        self,
        *,
        reason: str,
        step_name: str,
        step_walltime_s: float,
        step_nfev: int,
        case_walltime_s: float,
        t_step_s: float | None = None,
    ) -> None:
        self.reason = reason
        self.step_name = step_name
        self.step_walltime_s = float(step_walltime_s)
        self.step_nfev = int(step_nfev)
        self.case_walltime_s = float(case_walltime_s)
        self.t_step_s = None if t_step_s is None else float(t_step_s)
        super().__init__(
            f"{reason} at step={step_name}, "
            f"t_step={self.t_step_s}, "
            f"step_walltime_s={self.step_walltime_s:.1f}, "
            f"step_nfev={self.step_nfev}, "
            f"case_walltime_s={self.case_walltime_s:.1f}"
        )


def append_case_log(case_dir: Path, message: str) -> None:
    """Append a timestamped line to a per-case live progress file."""
    ensure_dir(case_dir)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(case_dir / "live_progress.txt", "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def install_solver_timeout_guard(
    model: Any,
    *,
    case_dir: Path,
    max_step_walltime_s: float | None = None,
    max_nfev_per_step: int | None = None,
    max_case_walltime_s: float | None = None,
) -> dict:
    """
    Wrap model.rhs_solver with per-step walltime and nfev guards.

    This is a soft timeout: it interrupts solve_ivp when RHS is called again.
    It catches the practical batch-stuck cases without changing the physical
    equations. If the solver is stuck inside a low-level linear algebra call,
    a hard subprocess watchdog would be needed.
    """
    limits_active = any([
        max_step_walltime_s is not None and float(max_step_walltime_s) > 0.0,
        max_nfev_per_step is not None and int(max_nfev_per_step) > 0,
        max_case_walltime_s is not None and float(max_case_walltime_s) > 0.0,
    ])
    state = {
        "case_start_time": perf_counter(),
        "current_step": None,
        "step_start_time": None,
        "step_nfev": 0,
        "last_t": None,
        "last_step": "",
        "last_reason": "",
    }
    if not limits_active:
        return state

    original_rhs_solver = model.rhs_solver

    def limited_rhs_solver(t: float, y: np.ndarray, step_name: str) -> np.ndarray:
        now = perf_counter()
        t_float = float(t)

        new_step = (
            state["current_step"] != step_name
            or state["last_t"] is None
            or t_float < float(state["last_t"]) - 1e-9
        )
        if new_step:
            state["current_step"] = step_name
            state["step_start_time"] = now
            state["step_nfev"] = 0
            state["last_t"] = t_float
            state["last_step"] = step_name
            append_case_log(case_dir, f"STEP_START step={step_name}, t_step_s={t_float:.6g}")

        state["step_nfev"] += 1
        state["last_t"] = t_float

        step_wall = now - float(state["step_start_time"])
        case_wall = now - float(state["case_start_time"])

        if max_case_walltime_s is not None and float(max_case_walltime_s) > 0.0:
            if case_wall > float(max_case_walltime_s):
                state["last_reason"] = "case_walltime_timeout"
                raise StepTimeoutError(
                    reason="case_walltime_timeout",
                    step_name=str(step_name),
                    step_walltime_s=step_wall,
                    step_nfev=int(state["step_nfev"]),
                    case_walltime_s=case_wall,
                    t_step_s=t_float,
                )

        if max_step_walltime_s is not None and float(max_step_walltime_s) > 0.0:
            if step_wall > float(max_step_walltime_s):
                state["last_reason"] = "step_walltime_timeout"
                raise StepTimeoutError(
                    reason="step_walltime_timeout",
                    step_name=str(step_name),
                    step_walltime_s=step_wall,
                    step_nfev=int(state["step_nfev"]),
                    case_walltime_s=case_wall,
                    t_step_s=t_float,
                )

        if max_nfev_per_step is not None and int(max_nfev_per_step) > 0:
            if int(state["step_nfev"]) > int(max_nfev_per_step):
                state["last_reason"] = "nfev_timeout"
                raise StepTimeoutError(
                    reason="nfev_timeout",
                    step_name=str(step_name),
                    step_walltime_s=step_wall,
                    step_nfev=int(state["step_nfev"]),
                    case_walltime_s=case_wall,
                    t_step_s=t_float,
                )

        return original_rhs_solver(t, y, step_name)

    model.rhs_solver = limited_rhs_solver
    return state


def load_tvsa_module(script01_path: Path):
    """Dynamically load 01_jakarta_single_bed_tvsa.py."""
    if not script01_path.exists():
        raise FileNotFoundError(f"TVSA model script not found: {script01_path}")

    module_name = "tvsa01_dynamic_for_surrogate"
    spec = importlib.util.spec_from_file_location(module_name, str(script01_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script01_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def add_metadata_to_df(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = df.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def filter_last_cycle(df: pd.DataFrame, last_cycle: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "cycle" in df.columns:
        return df[df["cycle"] == last_cycle].copy()
    return df.copy()


# =============================================================================
# DESIGN READER
# =============================================================================

def canonicalize_design_columns(df: pd.DataFrame, co2_ppm_default: float) -> pd.DataFrame:
    """Make script-04 design columns compatible with script 05."""
    out = df.copy()

    # Create stable 1-based row number before filtering. This is the user's
    # "titik ke berapa" selector.
    if "design_row_1based" not in out.columns:
        out.insert(0, "design_row_1based", np.arange(1, len(out) + 1, dtype=int))

    # Case identity.
    if "case_id" not in out.columns:
        out["case_id"] = out["design_row_1based"].apply(lambda i: f"SURR_{int(i):03d}")
    else:
        out["case_id"] = out["case_id"].astype(str)

    if "design_type" not in out.columns:
        out["design_type"] = "surrogate_training"

    if "run_id" not in out.columns:
        out["run_id"] = out["case_id"].astype(str)

    # Weather columns.
    alias_map = {
        "weather_T_C": ["weather_T_C", "T_C", "representative_T_C"],
        "weather_RH_percent": ["weather_RH_percent", "RH_percent", "representative_RH_percent"],
        "weather_P_Pa": ["weather_P_Pa", "P_Pa", "representative_P_Pa"],
        "weather_CO2_ppm": ["weather_CO2_ppm", "CO2_ppm", "representative_CO2_ppm"],
    }

    for target, aliases in alias_map.items():
        if target not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[target] = out[alias]
                    break

    if "weather_P_Pa" not in out.columns and "PS_kPa" in out.columns:
        out["weather_P_Pa"] = pd.to_numeric(out["PS_kPa"], errors="coerce") * 1000.0

    if "weather_CO2_ppm" not in out.columns:
        out["weather_CO2_ppm"] = co2_ppm_default

    # Operating parameters.
    defaults = {
        "adsorption_time_s": 2160.0,
        "heating_desorption_time_s": 4650.0,
        "T_des_K": 363.15,
        "T_coolant_K": 283.15,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val

    # Optional IDs and metadata.
    if "cluster_id_1based" not in out.columns:
        if "cluster_id" in out.columns:
            out["cluster_id_1based"] = pd.to_numeric(out["cluster_id"], errors="coerce") + 1
        else:
            out["cluster_id_1based"] = np.nan

    if "cluster_id" not in out.columns:
        out["cluster_id"] = pd.to_numeric(out["cluster_id_1based"], errors="coerce") - 1

    for col, default in {
        "country_code": "UNK",
        "country_name": "Unknown",
        "province_id": "",
        "province_name": "Unknown",
        "datetime_utc": "",
        "datetime_local": "",
        "rep_id": "",
        "rep_type": "",
        "climate_source": "",
        "weight_hours": np.nan,
        "weight_fraction": np.nan,
        "n_total_hours": np.nan,
        "p_H2O_Pa": np.nan,
        "p_CO2_Pa": np.nan,
        "y_H2O_molmol": np.nan,
        "y_CO2_molmol": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    numeric_cols = [
        "weather_T_C", "weather_RH_percent", "weather_P_Pa", "weather_CO2_ppm",
        "adsorption_time_s", "heating_desorption_time_s", "T_des_K", "T_coolant_K",
        "cluster_id", "cluster_id_1based", "weight_hours", "weight_fraction", "n_total_hours",
        "p_H2O_Pa", "p_CO2_Pa", "y_H2O_molmol", "y_CO2_molmol",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = [
        "weather_T_C", "weather_RH_percent", "weather_P_Pa",
        "adsorption_time_s", "heating_desorption_time_s", "T_des_K", "T_coolant_K",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Design file missing required columns after canonicalization: {missing}")

    out = out.dropna(subset=required).copy()
    if out.empty:
        raise ValueError("No valid design rows after cleaning.")

    return out


def read_design(
    design_csv: Path,
    co2_ppm_default: float,
    start_row: int | None,
    end_row: int | None,
    max_cases: int | None,
    design_types: list[str] | None,
    only_case_ids: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not design_csv.exists():
        raise FileNotFoundError(
            f"Surrogate design CSV not found:\n{design_csv}\n\n"
            "Run 04_build_surrogate_training_design.py first."
        )

    full = pd.read_csv(design_csv)
    full = canonicalize_design_columns(full, co2_ppm_default=co2_ppm_default)

    selected = full.copy()

    if start_row is not None:
        selected = selected[selected["design_row_1based"] >= int(start_row)].copy()
    if end_row is not None:
        selected = selected[selected["design_row_1based"] <= int(end_row)].copy()

    if design_types:
        allowed = {x.strip() for x in design_types if x.strip()}
        selected = selected[selected["design_type"].astype(str).isin(allowed)].copy()

    if only_case_ids:
        allowed_ids = {str(x).strip() for x in only_case_ids if str(x).strip()}
        selected = selected[selected["case_id"].astype(str).isin(allowed_ids)].copy()

    selected = selected.sort_values("design_row_1based").reset_index(drop=True)
    if max_cases is not None:
        selected = selected.head(int(max_cases)).copy()

    if selected.empty:
        raise ValueError("No design rows selected. Check --start-row, --end-row, --design-type, or --only-case-ids.")

    return full, selected


# =============================================================================
# CASE FOLDERS AND RESUME
# =============================================================================

def case_folder_name(row: dict) -> str:
    row_no = int(row_value(row, "design_row_1based", default=0))
    case_id = safe_name(row_value(row, "case_id", default=f"SURR_{row_no:03d}"), max_len=40)
    design_type = safe_name(row_value(row, "design_type", default="case"), max_len=32)
    country = safe_name(row_value(row, "country_code", default="UNK"), max_len=12)
    province = safe_name(row_value(row, "province_name", "representative_province_name", default="Unknown"), max_len=50)
    return f"{row_no:03d}_{case_id}_{design_type}_{country}_{province}"


def required_case_files(case_dir: Path, n_cycles: int) -> list[Path]:
    suffix = f"cycle{int(n_cycles)}"
    return [
        case_dir / f"summary_{suffix}.csv",
        case_dir / f"product_{suffix}.csv",
        case_dir / f"energy_{suffix}.csv",
        case_dir / f"mass_balance_{suffix}.csv",
        case_dir / f"energy_balance_{suffix}.csv",
        case_dir / f"diagnostics_{suffix}.csv",
        case_dir / f"solver_log_{suffix}.csv",
        case_dir / "model_config.json",
    ]


def is_case_complete(case_dir: Path, n_cycles: int) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in required_case_files(case_dir, n_cycles))


def read_completed_case(case_dir: Path, n_cycles: int, status_row: dict) -> dict:
    suffix = f"cycle{int(n_cycles)}"

    def read_csv_or_empty(name: str) -> list[dict]:
        path = case_dir / f"{name}_{suffix}.csv"
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path).to_dict(orient="records")
        return []

    status = dict(status_row)
    status["status"] = "skipped_complete"
    status["error"] = ""

    return {
        "status": status,
        "summary_last": read_csv_or_empty("summary"),
        "product_last": read_csv_or_empty("product"),
        "energy_last": read_csv_or_empty("energy"),
        "mass_balance_last": read_csv_or_empty("mass_balance"),
        "energy_balance_last": read_csv_or_empty("energy_balance"),
        "diagnostics_last": read_csv_or_empty("diagnostics"),
        "solver_log_last": read_csv_or_empty("solver_log"),
    }


# =============================================================================
# WORKER
# =============================================================================

def build_status_row(row: dict, case_dir: Path, n_cycles: int) -> dict:
    return {
        "design_row_1based": int(row_value(row, "design_row_1based", default=0)),
        "case_id": str(row_value(row, "case_id", default="")),
        "run_id": str(row_value(row, "run_id", default=row_value(row, "case_id", default=""))),
        "design_type": str(row_value(row, "design_type", default="")),
        "cluster_id": row_value(row, "cluster_id", default=np.nan),
        "cluster_id_1based": row_value(row, "cluster_id_1based", default=np.nan),
        "rep_id": row_value(row, "rep_id", default=""),
        "country_code": row_value(row, "country_code", default="UNK"),
        "country_name": row_value(row, "country_name", default="Unknown"),
        "province_id": row_value(row, "province_id", default=""),
        "province_name": row_value(row, "province_name", default="Unknown"),
        "case_dir": str(case_dir),
        "weather_T_C": float(row_value(row, "weather_T_C", default=np.nan)),
        "weather_RH_percent": float(row_value(row, "weather_RH_percent", default=np.nan)),
        "weather_P_Pa": float(row_value(row, "weather_P_Pa", default=np.nan)),
        "weather_CO2_ppm": float(row_value(row, "weather_CO2_ppm", default=np.nan)),
        "adsorption_time_s": float(row_value(row, "adsorption_time_s", default=np.nan)),
        "heating_desorption_time_s": float(row_value(row, "heating_desorption_time_s", default=np.nan)),
        "T_des_K": float(row_value(row, "T_des_K", default=np.nan)),
        "T_coolant_K": float(row_value(row, "T_coolant_K", default=np.nan)),
        "n_cycles": int(n_cycles),
        "n_nodes": int(row_value(row, "n_nodes", default=-1)) if "n_nodes" in row else -1,
        "status": "failed",
        "error": "",
        "timeout_reason": "",
        "timeout_step_name": "",
        "timeout_step_walltime_s": np.nan,
        "timeout_step_nfev": np.nan,
        "timeout_case_walltime_s": np.nan,
        "timeout_t_step_s": np.nan,
    }


def enforce_detailed_mode(numeric: Any, method: str, max_step_s: float) -> None:
    """Force the detailed full ODE/PDE-MOL path in script 01."""
    for attr, value in {
        "use_fast_pressure_steps": False,
        "use_fast_heating_desorption_steps": False,
        "freeze_inert_during_heating_desorption": False,
        "use_near_vacuum_gas_temperature_equilibrium": False,
        "use_nondimensional_state": False,
        "use_jac_sparsity": True,
        "freeze_mass_during_closed_cooling": False,
        "use_total_voidage_for_gas_accumulation": True,
    }.items():
        if hasattr(numeric, attr):
            setattr(numeric, attr, value)

    if hasattr(numeric, "pressure_step_method"):
        numeric.pressure_step_method = str(method)
    if hasattr(numeric, "pressure_step_max_step_s"):
        numeric.pressure_step_max_step_s = float(max_step_s)


def run_one_surrogate_case(task: dict) -> dict:
    row = task["row"]
    script01_path = Path(task["script01_path"])
    cases_dir = Path(task["cases_dir"])
    n_cycles = int(task["n_cycles"])

    case_dir = cases_dir / case_folder_name(row)
    ensure_dir(case_dir)
    status_row = build_status_row(row, case_dir, n_cycles)

    if bool(task.get("resume", True)) and not bool(task.get("force_rerun", False)):
        if is_case_complete(case_dir, n_cycles=n_cycles):
            return read_completed_case(case_dir, n_cycles=n_cycles, status_row=status_row)

    model = None
    timeout_guard_state = None

    try:
        append_case_log(
            case_dir,
            f"CASE_START row={status_row.get('design_row_1based')} case_id={status_row.get('case_id')}"
        )
        tvsa = load_tvsa_module(script01_path)

        weather = tvsa.WeatherConfig(
            T_C=float(row["weather_T_C"]),
            RH_percent=float(row["weather_RH_percent"]),
            P_Pa=float(row["weather_P_Pa"]),
            CO2_ppm=float(row["weather_CO2_ppm"]),
            province_query=str(row_value(row, "province_name", default="surrogate_case")),
            datetime_utc=str(row_value(row, "datetime_utc", default="surrogate_design_point")),
        )

        bed = tvsa.BedConfig()
        if task.get("n_nodes") is not None:
            bed.n_nodes = int(task["n_nodes"])
        status_row["n_nodes"] = int(getattr(bed, "n_nodes", -1))
        ads = tvsa.AdsorbentConfig()
        cycle = tvsa.CycleConfig(n_cycles=n_cycles)
        cycle.adsorption_time_s = float(row["adsorption_time_s"])
        cycle.heating_desorption_time_s = float(row["heating_desorption_time_s"])
        cycle.T_des_K = float(row["T_des_K"])
        cycle.T_coolant_K = float(row["T_coolant_K"])

        # Optional fixed-cycle overrides from CLI.
        if task.get("evac_time_s") is not None:
            cycle.evacuation_time_s = float(task["evac_time_s"])
        if task.get("cool_time_s") is not None:
            cycle.cooling_time_s = float(task["cool_time_s"])
        if task.get("rep_time_s") is not None:
            cycle.repressurization_time_s = float(task["rep_time_s"])

        numeric = tvsa.NumericConfig(
            sample_dt_s=float(task["sample_dt_s"]),
            max_step_s=float(task["max_step_s"]),
            rtol=float(task["rtol"]),
            atol=float(task["atol"]),
            method=str(task["method"]),
        )
        if hasattr(numeric, "node_record_stride_s"):
            numeric.node_record_stride_s = float(task["node_record_stride_s"])

        enforce_detailed_mode(numeric, method=str(task["method"]), max_step_s=float(task["max_step_s"]))

        model = tvsa.TVSABedModel(weather, bed, ads, cycle, numeric)
        timeout_guard_state = install_solver_timeout_guard(
            model,
            case_dir=case_dir,
            max_step_walltime_s=task.get("max_step_walltime_s"),
            max_nfev_per_step=task.get("max_nfev_per_step"),
            max_case_walltime_s=task.get("max_case_walltime_s"),
        )
        result = model.simulate()

        if len(result) != 10:
            raise RuntimeError(
                "Unexpected return format from 01_jakarta_single_bed_tvsa.py. "
                f"Expected 10 returned objects, got {len(result)}."
            )

        (
            profiles,
            nodes_all,
            nodes_last,
            product_last,
            summary_all,
            energy_last,
            mass_balance,
            energy_balance,
            diagnostics,
            solver_log,
        ) = result

        last_cycle = int(cycle.n_cycles)
        suffix = f"cycle{last_cycle}"

        profiles_last = filter_last_cycle(profiles, last_cycle)
        nodes_last = filter_last_cycle(nodes_last, last_cycle)
        product_last = filter_last_cycle(product_last, last_cycle)
        energy_last = filter_last_cycle(energy_last, last_cycle)
        summary_last = filter_last_cycle(summary_all, last_cycle)
        mass_balance_last = filter_last_cycle(mass_balance, last_cycle)
        energy_balance_last = filter_last_cycle(energy_balance, last_cycle)
        diagnostics_last = filter_last_cycle(diagnostics, last_cycle)
        solver_log_last = filter_last_cycle(solver_log, last_cycle)

        metadata = {
            "design_row_1based": int(row_value(row, "design_row_1based", default=0)),
            "case_id": str(row_value(row, "case_id", default="")),
            "run_id": str(row_value(row, "run_id", default=row_value(row, "case_id", default=""))),
            "design_type": str(row_value(row, "design_type", default="")),
            "climate_source": row_value(row, "climate_source", default=""),
            "rep_id": row_value(row, "rep_id", default=""),
            "rep_type": row_value(row, "rep_type", default=""),
            "cluster_id": row_value(row, "cluster_id", default=np.nan),
            "cluster_id_1based": row_value(row, "cluster_id_1based", default=np.nan),
            "weight_hours": row_value(row, "weight_hours", default=np.nan),
            "weight_fraction": row_value(row, "weight_fraction", default=np.nan),
            "n_total_hours": row_value(row, "n_total_hours", default=np.nan),
            "country_code": row_value(row, "country_code", default="UNK"),
            "country_name": row_value(row, "country_name", default="Unknown"),
            "province_id": row_value(row, "province_id", default=""),
            "province_name": row_value(row, "province_name", default="Unknown"),
            "datetime_utc": row_value(row, "datetime_utc", default=""),
            "datetime_local": row_value(row, "datetime_local", default=""),
            "weather_T_C": float(row["weather_T_C"]),
            "weather_RH_percent": float(row["weather_RH_percent"]),
            "weather_P_Pa": float(row["weather_P_Pa"]),
            "weather_CO2_ppm": float(row["weather_CO2_ppm"]),
            "p_H2O_Pa": row_value(row, "p_H2O_Pa", default=np.nan),
            "p_CO2_Pa": row_value(row, "p_CO2_Pa", default=np.nan),
            "y_H2O_molmol": row_value(row, "y_H2O_molmol", default=np.nan),
            "y_CO2_molmol": row_value(row, "y_CO2_molmol", default=np.nan),
            "adsorption_time_s": float(row["adsorption_time_s"]),
            "heating_desorption_time_s": float(row["heating_desorption_time_s"]),
            "T_des_K": float(row["T_des_K"]),
            "T_des_C": float(row["T_des_K"]) - 273.15,
            "T_coolant_K": float(row["T_coolant_K"]),
            "T_coolant_C": float(row["T_coolant_K"]) - 273.15,
            "case_dir": str(case_dir),
            "n_nodes": int(getattr(bed, "n_nodes", -1)),
        }

        summary_last = add_metadata_to_df(summary_last, metadata)
        product_last = add_metadata_to_df(product_last, metadata)
        energy_last = add_metadata_to_df(energy_last, metadata)
        mass_balance_last = add_metadata_to_df(mass_balance_last, metadata)
        energy_balance_last = add_metadata_to_df(energy_balance_last, metadata)
        diagnostics_last = add_metadata_to_df(diagnostics_last, metadata)
        solver_log_last = add_metadata_to_df(solver_log_last, metadata)

        # Save last-cycle-only outputs per case.
        summary_last.to_csv(case_dir / f"summary_{suffix}.csv", index=False, encoding="utf-8-sig")
        product_last.to_csv(case_dir / f"product_{suffix}.csv", index=False, encoding="utf-8-sig")
        energy_last.to_csv(case_dir / f"energy_{suffix}.csv", index=False, encoding="utf-8-sig")
        mass_balance_last.to_csv(case_dir / f"mass_balance_{suffix}.csv", index=False, encoding="utf-8-sig")
        energy_balance_last.to_csv(case_dir / f"energy_balance_{suffix}.csv", index=False, encoding="utf-8-sig")
        diagnostics_last.to_csv(case_dir / f"diagnostics_{suffix}.csv", index=False, encoding="utf-8-sig")
        solver_log_last.to_csv(case_dir / f"solver_log_{suffix}.csv", index=False, encoding="utf-8-sig")

        if bool(task.get("save_detailed", False)):
            profiles_last = add_metadata_to_df(profiles_last, metadata)
            nodes_last = add_metadata_to_df(nodes_last, metadata)
            profiles_last.to_csv(case_dir / f"profiles_{suffix}_long.csv", index=False, encoding="utf-8-sig")
            nodes_last.to_csv(case_dir / f"profiles_{suffix}_nodes.csv", index=False, encoding="utf-8-sig")

        if bool(task.get("make_plots", False)) and hasattr(tvsa, "plot_outputs"):
            try:
                tvsa.plot_outputs(case_dir, profiles, nodes_all, nodes_last, product_last, energy_last, cycle)
            except Exception as plot_exc:
                with open(case_dir / "plot_warning.txt", "w", encoding="utf-8") as f:
                    f.write(str(plot_exc))

        config = {
            "metadata": metadata,
            "weather": asdict(weather),
            "bed": asdict(bed),
            "adsorbent": asdict(ads),
            "cycle": asdict(cycle),
            "numeric": asdict(numeric),
            "task_options": {k: v for k, v in task.items() if k != "row"},
        }
        write_json(case_dir / "model_config.json", config)

        status_row["status"] = "success"
        status_row["error"] = ""
        append_case_log(case_dir, "CASE_SUCCESS")

        return {
            "status": status_row,
            "summary_last": summary_last.to_dict(orient="records"),
            "product_last": product_last.to_dict(orient="records"),
            "energy_last": energy_last.to_dict(orient="records"),
            "mass_balance_last": mass_balance_last.to_dict(orient="records"),
            "energy_balance_last": energy_balance_last.to_dict(orient="records"),
            "diagnostics_last": diagnostics_last.to_dict(orient="records"),
            "solver_log_last": solver_log_last.to_dict(orient="records"),
        }

    except StepTimeoutError as exc:
        status_row["status"] = "timeout"
        status_row["error"] = str(exc)
        status_row["timeout_reason"] = exc.reason
        status_row["timeout_step_name"] = exc.step_name
        status_row["timeout_step_walltime_s"] = exc.step_walltime_s
        status_row["timeout_step_nfev"] = exc.step_nfev
        status_row["timeout_case_walltime_s"] = exc.case_walltime_s
        status_row["timeout_t_step_s"] = exc.t_step_s

        timeout_report = {
            "status": "timeout",
            "status_row": status_row,
            "timeout_guard_state": timeout_guard_state,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(case_dir / "timeout_report.json", timeout_report)
        with open(case_dir / "error_traceback.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        if model is not None and hasattr(model, "solver_log"):
            try:
                pd.DataFrame(model.solver_log).to_csv(
                    case_dir / "partial_solver_log_timeout.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            except Exception:
                pass
        append_case_log(case_dir, f"CASE_TIMEOUT reason={exc.reason} step={exc.step_name}")

        return {
            "status": status_row,
            "summary_last": [],
            "product_last": [],
            "energy_last": [],
            "mass_balance_last": [],
            "energy_balance_last": [],
            "diagnostics_last": [],
            "solver_log_last": [],
        }

    except Exception as exc:
        status_row["status"] = "failed"
        status_row["error"] = str(exc)
        with open(case_dir / "error_traceback.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        if model is not None and hasattr(model, "solver_log"):
            try:
                pd.DataFrame(model.solver_log).to_csv(
                    case_dir / "partial_solver_log_failed.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            except Exception:
                pass
        append_case_log(case_dir, f"CASE_FAILED error={str(exc)}")

        return {
            "status": status_row,
            "summary_last": [],
            "product_last": [],
            "energy_last": [],
            "mass_balance_last": [],
            "energy_balance_last": [],
            "diagnostics_last": [],
            "solver_log_last": [],
        }


# =============================================================================
# MAIN
# =============================================================================

def parse_csv_list(text: str | None) -> list[str] | None:
    if text is None or str(text).strip() == "":
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def make_batch_label(start_row: int | None, end_row: int | None, design_type: str | None, explicit: str | None) -> str:
    if explicit:
        return safe_name(explicit, max_len=80)
    if start_row is None and end_row is None:
        row_part = "all_rows"
    else:
        s = "001" if start_row is None else f"{int(start_row):03d}"
        e = "end" if end_row is None else f"{int(end_row):03d}"
        row_part = f"rows_{s}_{e}"
    dt = safe_name(design_type, max_len=50) if design_type else "all_types"
    return f"{row_part}_{dt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected surrogate-training TVSA cases using script 01 engine.")
    parser.add_argument("--project-dir", type=str, default=str(PROJECT_DIR_DEFAULT))
    parser.add_argument("--script01", type=str, default=str(SCRIPT01_DEFAULT))
    parser.add_argument("--design-csv", type=str, default=str(DESIGN_CSV_DEFAULT))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))

    # Selection controls. Rows are 1-based, inclusive, and refer to the CSV order
    # before filtering. This is intended for distributing batches across computers.
    parser.add_argument("--start-row", type=int, default=None, help="1-based first design row to run, inclusive.")
    parser.add_argument("--end-row", type=int, default=None, help="1-based last design row to run, inclusive.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--design-type", type=str, default=None, help="Comma-separated design_type filter.")
    parser.add_argument("--only-case-ids", type=str, default=None, help="Comma-separated case_id filter.")
    parser.add_argument("--batch-label", type=str, default=None)

    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--co2-ppm", type=float, default=400.0)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--n-nodes", type=int, default=20, help="Axial finite-volume nodes for BedConfig.")

    parser.add_argument("--sample-dt-s", type=float, default=10.0)
    parser.add_argument("--max-step-s", type=float, default=20.0)
    parser.add_argument("--node-record-stride-s", type=float, default=30.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--method", type=str, default="BDF")

    # Batch timeout guards. A case is stopped and marked "timeout" if either
    # one step exceeds the walltime limit or the RHS call count for that step
    # exceeds the nfev limit. Use 0 to disable a guard.
    parser.add_argument("--max-step-walltime-s", type=float, default=5000.0)
    parser.add_argument("--max-nfev-per-step", type=int, default=100000)
    parser.add_argument("--max-case-walltime-s", type=float, default=0.0)

    # Fixed cycle overrides not sampled in script 04.
    parser.add_argument("--evac-time-s", type=float, default=None)
    parser.add_argument("--cool-time-s", type=float, default=None)
    parser.add_argument("--rep-time-s", type=float, default=None)

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--make-plots", action="store_true")

    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    script01_path = Path(args.script01)
    design_csv = Path(args.design_csv)
    out_dir = Path(args.out_dir)
    cases_dir = out_dir / "cases"
    ensure_dir(cases_dir)

    design_types = parse_csv_list(args.design_type)
    only_case_ids = parse_csv_list(args.only_case_ids)

    full_design, selected = read_design(
        design_csv=design_csv,
        co2_ppm_default=args.co2_ppm,
        start_row=args.start_row,
        end_row=args.end_row,
        max_cases=args.max_cases,
        design_types=design_types,
        only_case_ids=only_case_ids,
    )

    batch_label = make_batch_label(args.start_row, args.end_row, args.design_type, args.batch_label)
    batch_dir = out_dir / "batches" / batch_label
    ensure_dir(batch_dir)

    selected_path = batch_dir / f"surrogate_design_selected_{batch_label}.csv"
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    # Build tasks and identify already-complete cases.
    base_task = {
        "project_dir": str(project_dir),
        "script01_path": str(script01_path),
        "cases_dir": str(cases_dir),
        "co2_ppm": args.co2_ppm,
        "n_cycles": args.n_cycles,
        "n_nodes": args.n_nodes,
        "sample_dt_s": args.sample_dt_s,
        "max_step_s": args.max_step_s,
        "node_record_stride_s": args.node_record_stride_s,
        "rtol": args.rtol,
        "atol": args.atol,
        "method": args.method,
        "max_step_walltime_s": None if args.max_step_walltime_s <= 0 else float(args.max_step_walltime_s),
        "max_nfev_per_step": None if args.max_nfev_per_step <= 0 else int(args.max_nfev_per_step),
        "max_case_walltime_s": None if args.max_case_walltime_s <= 0 else float(args.max_case_walltime_s),
        "evac_time_s": args.evac_time_s,
        "cool_time_s": args.cool_time_s,
        "rep_time_s": args.rep_time_s,
        "resume": args.resume,
        "force_rerun": args.force_rerun,
        "save_detailed": args.save_detailed,
        "make_plots": args.make_plots,
    }

    tasks = []
    n_complete_before = 0
    for _, row in selected.iterrows():
        row_dict = row.to_dict()
        task = base_task.copy()
        task["row"] = row_dict
        tasks.append(task)
        cdir = cases_dir / case_folder_name(row_dict)
        if args.resume and not args.force_rerun and is_case_complete(cdir, args.n_cycles):
            n_complete_before += 1

    print("=" * 90)
    print("RUNNING SURROGATE TRAINING TVSA CASES")
    print("=" * 90)
    print(f"Project directory : {project_dir}")
    print(f"TVSA engine       : {script01_path}")
    print(f"Design CSV        : {design_csv}")
    print(f"Output root       : {out_dir}")
    print(f"Cases directory   : {cases_dir}")
    print(f"Batch directory   : {batch_dir}")
    print(f"Batch label       : {batch_label}")
    print(f"Selected rows     : {args.start_row if args.start_row is not None else 'start'} to {args.end_row if args.end_row is not None else 'end'}")
    print(f"Selected cases    : {len(selected)}")
    print(f"Already complete  : {n_complete_before}")
    print(f"Remaining approx. : {len(selected) - n_complete_before if args.resume and not args.force_rerun else len(selected)}")
    print(f"Workers           : {args.workers}")
    print(f"Cycles per case   : {args.n_cycles}")
    print("Model mode        : detailed full ODE/PDE-MOL from script 01")
    print(f"rtol / atol       : {args.rtol} / {args.atol}")
    print(f"max_step_s        : {args.max_step_s}")
    print(f"step timeout s    : {args.max_step_walltime_s if args.max_step_walltime_s > 0 else 'disabled'}")
    print(f"max nfev/step     : {args.max_nfev_per_step if args.max_nfev_per_step > 0 else 'disabled'}")
    print(f"case timeout s    : {args.max_case_walltime_s if args.max_case_walltime_s > 0 else 'disabled'}")
    print(f"Resume            : {args.resume}")
    print(f"Force rerun       : {args.force_rerun}")
    print("=" * 90)

    statuses: list[dict] = []
    summary_rows: list[dict] = []
    product_rows: list[dict] = []
    energy_rows: list[dict] = []
    mass_balance_rows: list[dict] = []
    energy_balance_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    solver_log_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_task = {executor.submit(run_one_surrogate_case, task): task for task in tasks}

        for future in as_completed(future_to_task):
            result = future.result()
            status = result["status"]
            statuses.append(status)

            row_no = status.get("design_row_1based", "?")
            case_id = status.get("case_id", "")
            design_type = status.get("design_type", "")
            stat = status.get("status", "unknown")
            print(f"[{stat.upper()}] Row {row_no}: {case_id} ({design_type})")

            summary_rows.extend(result["summary_last"])
            product_rows.extend(result["product_last"])
            energy_rows.extend(result["energy_last"])
            mass_balance_rows.extend(result["mass_balance_last"])
            energy_balance_rows.extend(result["energy_balance_last"])
            diagnostics_rows.extend(result["diagnostics_last"])
            solver_log_rows.extend(result["solver_log_last"])

    status_df = pd.DataFrame(statuses).sort_values("design_row_1based") if statuses else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    product_df = pd.DataFrame(product_rows)
    energy_df = pd.DataFrame(energy_rows)
    mass_balance_df = pd.DataFrame(mass_balance_rows)
    energy_balance_df = pd.DataFrame(energy_balance_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    solver_log_df = pd.DataFrame(solver_log_rows)

    # Sort consolidated tables for readability if possible.
    for name, df in [
        ("summary", summary_df), ("product", product_df), ("energy", energy_df),
        ("mass_balance", mass_balance_df), ("energy_balance", energy_balance_df),
        ("diagnostics", diagnostics_df), ("solver_log", solver_log_df),
    ]:
        if not df.empty and "design_row_1based" in df.columns:
            if name in {"product", "energy", "solver_log"} and "t_global_s" in df.columns:
                df.sort_values(["design_row_1based", "t_global_s"], inplace=True)
            elif name == "solver_log" and "cycle" in df.columns:
                df.sort_values(["design_row_1based", "cycle"], inplace=True)
            else:
                df.sort_values(["design_row_1based"], inplace=True)

    outputs = {
        "surrogate_tvsa_run_status": status_df,
        "surrogate_tvsa_summary": summary_df,
        "surrogate_tvsa_product": product_df,
        "surrogate_tvsa_energy": energy_df,
        "surrogate_tvsa_mass_balance": mass_balance_df,
        "surrogate_tvsa_energy_balance": energy_balance_df,
        "surrogate_tvsa_diagnostics": diagnostics_df,
        "surrogate_tvsa_solver_log": solver_log_df,
    }

    for stem, df in outputs.items():
        path = batch_dir / f"{stem}_{batch_label}.csv"
        backup_existing_file(path)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    metadata = {
        "project_dir": str(project_dir),
        "script01_path": str(script01_path),
        "design_csv": str(design_csv),
        "selected_design_path": str(selected_path),
        "out_dir": str(out_dir),
        "cases_dir": str(cases_dir),
        "batch_dir": str(batch_dir),
        "batch_label": batch_label,
        "model_mode": "detailed_full_ode_pde_mol",
        "full_design_rows": int(len(full_design)),
        "selected_cases": int(len(selected)),
        "already_complete_before_run": int(n_complete_before),
        "workers": int(args.workers),
        "n_cycles": int(args.n_cycles),
        "sample_dt_s": float(args.sample_dt_s),
        "max_step_s": float(args.max_step_s),
        "node_record_stride_s": float(args.node_record_stride_s),
        "co2_ppm_default": float(args.co2_ppm),
        "method": str(args.method),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_step_walltime_s": float(args.max_step_walltime_s),
        "max_nfev_per_step": int(args.max_nfev_per_step),
        "max_case_walltime_s": float(args.max_case_walltime_s),
        "start_row": args.start_row,
        "end_row": args.end_row,
        "design_type_filter": design_types,
        "only_case_ids": only_case_ids,
        "resume": bool(args.resume),
        "force_rerun": bool(args.force_rerun),
        "save_detailed": bool(args.save_detailed),
        "make_plots": bool(args.make_plots),
        "n_success": int((status_df["status"] == "success").sum()) if not status_df.empty else 0,
        "n_skipped_complete": int((status_df["status"] == "skipped_complete").sum()) if not status_df.empty else 0,
        "n_timeout": int((status_df["status"] == "timeout").sum()) if not status_df.empty else 0,
        "n_failed": int((status_df["status"] == "failed").sum()) if not status_df.empty else 0,
    }
    write_json(batch_dir / f"surrogate_tvsa_metadata_{batch_label}.json", metadata)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Batch directory : {batch_dir}")
    print(f"Run status      : {batch_dir / f'surrogate_tvsa_run_status_{batch_label}.csv'}")
    print(f"Main summary    : {batch_dir / f'surrogate_tvsa_summary_{batch_label}.csv'}")
    print(f"Cases directory : {cases_dir}")

    if not status_df.empty:
        print("\nSTATUS SUMMARY")
        print(status_df["status"].value_counts().to_string())

    if not summary_df.empty:
        useful_cols = [
            "design_row_1based", "case_id", "design_type", "country_code", "province_name",
            "weather_T_C", "weather_RH_percent", "adsorption_time_s", "heating_desorption_time_s",
            "T_des_K", "T_coolant_K", "kg_CO2_cycle", "kg_H2O_cycle",
            "CO2_H2O_mol_ratio", "Q_heat_kWhth_cycle", "E_total_el_kWhe_cycle",
            "specific_heat_MWhth_tCO2", "specific_electricity_MWhe_tCO2",
            "annual_tCO2_per_bed",
        ]
        useful_cols = [c for c in useful_cols if c in summary_df.columns]
        print("\nSURROGATE TVSA SUMMARY PREVIEW")
        print(summary_df[useful_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
