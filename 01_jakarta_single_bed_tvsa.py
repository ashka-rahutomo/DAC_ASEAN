from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

R = 8.314462618  # J/mol/K

COMPONENTS = ("CO2", "H2O", "N2", "O2")
ADS_COMPONENTS = ("CO2", "H2O")
IDX = {name: i for i, name in enumerate(COMPONENTS)}
AIDX = {name: i for i, name in enumerate(ADS_COMPONENTS)}

MW = {
    "CO2": 44.0095e-3,      # kg/mol
    "H2O": 18.01528e-3,
    "N2": 28.0134e-3,
    "O2": 31.9988e-3,
}

CP_GAS_MOLAR = {
    "CO2": 37.1,            # J/mol/K, engineering constant near ambient
    "H2O": 33.6,
    "N2": 29.1,
    "O2": 29.4,
}

# Molecular diffusivities in air from Jajjawi SI Table 3.
# SI reports cm2/s; converted here to m2/s.
DM = {
    "CO2": 0.1381e-4,
    "H2O": 0.2178e-4,
    "N2": 0.1788e-4,
    "O2": 0.1820e-4,
}


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class PathConfig:
    project_dir: Path = Path(r"D:/Ashka/5.DAC/06.PYTHON")
    nasa_hourly_csv: Path = Path(
        r"D:/Ashka/5.DAC/00.TEMPORAL_DATA/NASA_POWER/ASEAN_province_hourly_NASA_POWER_full_2025.csv"
    )
    output_subdir: str = "outputs/jakarta"


@dataclass
class WeatherConfig:
    # Fallback values if NASA POWER input is unavailable.
    T_C: float = 27.0
    RH_percent: float = 75.0
    P_Pa: float = 101_000.0
    CO2_ppm: float = 400.0
    province_query: str = "Jakarta"
    datetime_utc: str | None = None


@dataclass
class BedConfig:
    # Jajjawi small-scale cylindrical packed bed: D = 10 cm, bed height = 2 cm.
    n_nodes: int = 20
    bed_length_m: float = 0.020
    bed_diameter_m: float = 0.100

    # Lewatit VP OC 1065 properties from Jajjawi SI.
    epsilon_b: float = 0.400
    epsilon_p: float = 0.238  # retained for documentation; LDF absorbs pore detail
    rho_bulk_kg_m3_bed: float = 880.0
    particle_radius_m: float = 0.0007
    cp_solid_J_kgK: float = 1.58e3
    lambda_solid_W_mK: float = 0.43

    # Gas/solid heat transfer and conductivity.
    lambda_gas_W_mK: float = 0.030
    h_gs_W_m2K: float = 26.749

    # Heat-exchanger area density and overall coefficient from Jajjawi SI.
    a_hx_m2_m3: float = 40.0
    U_hx_W_m2K: float = 300.0  # used as wall-jacket coefficient

    # Wall model closure assumptions.
    wall_density_kg_m3: float = 7850.0
    wall_cp_J_kgK: float = 500.0
    wall_thickness_m: float = 0.0010
    h_solid_wall_W_m2K: float = 80.0
    U_loss_W_m2K: float = 0.0  # adiabatic by default, consistent with Jajjawi column assumption

    # Explicit jacket-fluid holdup closure.
    jacket_holdup_kg_per_m3_bed: float = 25.0


@dataclass
class AdsorbentConfig:
    # Dry Toth branch.
    dry_qmax_mol_kg: float = 4.86
    dry_b0_Pa_inv: float = 2.85e-21
    dry_minus_dH_J_mol: float = 117_789.0
    dry_t0: float = 0.209
    dry_a: float = 0.523

    # Wet-site Toth branch.
    wet_qmax_mol_kg: float = 9.035
    wet_b0_Pa_inv: float = 1.230e-18
    wet_minus_dH_J_mol: float = 203_687.0
    wet_t0: float = 0.053
    wet_a: float = 0.053

    T0_K: float = 298.15
    wadst_A_mol_kg: float = 1.523

    # GAB water model.
    gab_qm_mol_kg: float = 3.63
    gab_C_J_mol: float = 47_110.0
    gab_D_K_inv: float = 0.023744
    gab_F_J_mol: float = 57_706.0
    gab_G_J_molK: float = -47.814

    # LDF kinetic coefficients from Jajjawi SI.
    k_ldf_co2_s: float = 0.003
    k_ldf_h2o_s: float = 0.0086

    # Heat of adsorption from Jajjawi SI.
    dH_ads_CO2_J_mol: float = -60.0e3
    dH_ads_H2O_J_mol: float = -49.0e3

    # Adsorbed phase heat capacities from Jajjawi SI.
    cp_ads_CO2_J_molK: float = 88.0
    cp_ads_H2O_J_molK: float = 75.0


@dataclass
class CycleConfig:
    # Default is 3 cycles; last cycle is used for detailed CSV/report.
    n_cycles: int = 3

    # Cycle durations. Desorption time refers only to heating desorption.
    adsorption_time_s: float = 36.0 * 60.0
    evacuation_time_s: float = 60.0
    heating_desorption_time_s: float = 77.5 * 60.0
    cooling_time_s: float = 600.0
    repressurization_time_s: float = 180.0
    desorption_co2_cutoff_mol_s: float = 1.0e-6
    cooling_max_sorbent_temperature_K: float = 348.15  # 75 degC

    T_des_K: float = 363.15
    T_coolant_K: float = 283.15
    P_vac_Pa: float = 0.1e5
    P_high_default_Pa: float = 1.013e5

    # Jajjawi airflow = 6.2 L/s for D = 10 cm bed.
    air_volumetric_flow_m3_s_ref: float = 6.2e-3

    # Jacket medium from Jajjawi SI.
    heating_fluid_mdot_kg_s: float = 0.01
    cooling_fluid_mdot_kg_s: float = 0.01
    jacket_cp_J_kgK: float = 4.1e3

    # KPI parameters from Jajjawi SI.
    eta_fan: float = 0.75
    eta_vacuum: float = 0.70
    COP_chiller: float = 3.0

    # Practical pump/valve limits for numerical stability.
    max_product_superficial_velocity_m_s: float = 0.05
    max_repress_superficial_velocity_m_s: float = 0.10

    # Smooth valve/control ramps. These keep the fixed cycle schedule, but avoid
    # instantaneous pressure/flow boundary shocks that make evacuation and
    # repressurization unnecessarily stiff.
    evacuation_valve_ramp_time_s: float = 20.0
    desorption_product_valve_ramp_time_s: float = 0.0
    repressurization_valve_ramp_time_s: float = 30.0

    # Repressurization pressure control. Feed flow is smoothly reduced as the
    # bed-average pressure approaches ambient pressure to avoid pressure overshoot.
    repressurization_pressure_stop_margin_Pa: float = 250.0

    # Evacuation/product pressure control. Product withdrawal is reduced once the
    # bed is close to vacuum to avoid overdraw, negative species inventories, and
    # unnecessary BDF step rejection. During desorption, the same factor reopens
    # when desorbed gas raises the bed pressure above the vacuum margin.
    evacuation_pressure_stop_margin_Pa: float = 500.0


@dataclass
class NumericConfig:
    sample_dt_s: float = 10.0
    max_step_s: float = 20.0
    rtol: float = 1e-3
    atol: float = 1e-7
    method: str = "BDF"

    # Optional reduced pressure equalization for short evacuation/repressurization steps.
    # Default is False for the detailed-validation path: evacuation and
    # repressurization are solved by the full dynamic ODE/PDE-MOL system.
    # Turn this on only for accelerated comparison/debugging.
    use_fast_pressure_steps: bool = False
    pressure_step_method: str = "BDF"
    pressure_step_max_step_s: float = 20.0

    # Stronger solver conditioning: solve scaled states internally.
    # Physical equations remain dimensional; only the state vector seen by BDF
    # is nondimensional. This usually improves Jacobian conditioning more than
    # vector tolerances alone.
    use_nondimensional_state: bool = False
    C_scale_floor_mol_m3: float = 1e-4
    q_scale_floor_mol_kg: float = 0.1
    T_scale_K: float = 300.0

    # Smooth floors reduce nonsmooth RHS behavior from hard np.maximum/np.clip.
    # Hard finite guards are still retained only for NaN/inf and extreme states.
    smooth_floor_rel_width: float = 1e-6
    smooth_floor_abs_width: float = 1e-12
    smooth_temperature_width_K: float = 0.5

    # State-specific absolute tolerances. This does not change the physical
    # units or equations; it only gives the implicit solver realistic error
    # scales for concentration, loading, and temperature states.
    use_vector_atol: bool = True
    atol_C_mol_m3: float = 1e-6
    atol_q_mol_kg: float = 1e-7
    atol_T_K: float = 1e-4

    min_concentration_mol_m3: float = 1e-12
    min_temperature_K: float = 250.0
    max_temperature_K: float = 450.0
    min_loading_mol_kg: float = 0.0

    axial_dispersion_pe: float = 2.0
    longitudinal_dispersivity_m: float = 0.002

    # Aspen Adsorption distinguishes inter-particle voidage for transport from
    # total bed voidage for gas accumulation. Enable this by default.
    use_total_voidage_for_gas_accumulation: bool = True

    # Relative tolerance for cycle-to-cycle KPI convergence diagnostics.
    css_relative_tolerance: float = 0.001

    # Store axial node records less frequently to reduce memory and CSV/plot overhead.
    node_record_stride_s: float = 30.0

    # Gas-capacity regularization for vacuum steps.
    # This prevents near-empty gas nodes from making epsilon*Ctot*Cp nearly zero,
    # which otherwise creates very large dTg/dt and BDF Jacobian overflow.
    gas_pressure_floor_fraction_of_vac: float = 0.5
    gas_pressure_floor_min_Pa: float = 1000.0

    # Gas thermal regularization pressure. This is used only in the gas energy
    # denominator, so near-vacuum gas temperature does not become numerically
    # singular when gas holdup is tiny.
    gas_thermal_pressure_floor_Pa: float = 5000.0

    # Optional heating-step runtime reduction options.
    # Default is False for the detailed-validation path: all gas species remain
    # dynamic during heating desorption.
    freeze_inert_during_heating_desorption: bool = False

    # Optional fast operator-split heating desorption mode. Default is False for
    # the detailed-validation path: heating desorption is solved by the full
    # dynamic ODE/PDE-MOL system and product is from outlet-face flux.
    use_fast_heating_desorption_steps: bool = False
    fast_heating_step_s: float = 50.0

    # Optional near-vacuum gas temperature quasi-equilibrium. Default is False
    # for detailed validation; Tg is solved as an independent dynamic state.
    use_near_vacuum_gas_temperature_equilibrium: bool = False
    gas_temperature_eq_pressure_threshold_Pa: float = 15000.0
    gas_temperature_eq_tau_s: float = 60.0

    # Numerical regularization for the explicit jacket-fluid state.
    jacket_tau_min_s: float = 120.0

    # Smooth inlet switching from hot to cold jacket fluid without splitting the cycle step.
    jacket_inlet_ramp_time_s: float = 180.0

    # Robust closed-cooling option: no external mass crosses the bed during this
    # step, and the gas holdup under vacuum is tiny. Freezing gas/loadings during
    # closed cooling removes the artificial numerical fight between LDF adsorption
    # and a near-empty gas phase, with minimal effect on product-cycle KPI.
    freeze_mass_during_closed_cooling: bool = False

    # Use a smoothed effective cooling-fluid boundary instead of solving a very
    # small jacket holdup as a fast state during closed cooling.
    robust_cooling_jacket_boundary: bool = True

    # Sparse-Jacobian structure for BDF/Radau. This does not change the equations;
    # it only tells the solver that the 1D finite-volume model is locally coupled.
    # Empirically, sparse finite-difference grouping can be slower/less stable
    # for this clipped, piecewise TVSA RHS. Keep it available but off by default.
    use_jac_sparsity: bool = True

    # Hard finite-safety caps used only to prevent BDF trial states from
    # generating NaN/inf during numerical Jacobian evaluation.
    max_pressure_regularization_Pa: float = 2.0e5
    max_abs_rhs_value: float = 1.0e8


# =============================================================================
# THERMODYNAMIC AND PHYSICAL FUNCTIONS
# =============================================================================


def smooth_floor(x: np.ndarray | float, floor: np.ndarray | float, rel_width: float = 1e-6, abs_width: float = 1e-12) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    f_arr = np.asarray(floor, dtype=float)
    w = np.maximum(abs_width, rel_width * np.maximum(np.abs(f_arr), 1.0))
    z = (x_arr - f_arr) / w
    transition = f_arr + w * np.log1p(np.exp(np.clip(z, -50.0, 50.0)))
    return np.where(z > 50.0, x_arr, np.where(z < -50.0, f_arr, transition))


def smooth_ceiling(x: np.ndarray | float, ceiling: np.ndarray | float, rel_width: float = 1e-6, abs_width: float = 1e-12) -> np.ndarray:
    """Smooth approximation of min(x, ceiling), exact away from the ceiling."""
    c_arr = np.asarray(ceiling, dtype=float)
    return c_arr - smooth_floor(
        c_arr - np.asarray(x, dtype=float),
        0.0,
        rel_width=rel_width,
        abs_width=abs_width,
    )

def smooth_clip(x: np.ndarray | float, lower: np.ndarray | float, upper: np.ndarray | float, rel_width: float = 1e-6, abs_width: float = 1e-12) -> np.ndarray:
    """Smooth approximation of clip(x, lower, upper)."""
    return smooth_ceiling(
        smooth_floor(x, lower, rel_width=rel_width, abs_width=abs_width),
        upper,
        rel_width=rel_width,
        abs_width=abs_width,
    )


def saturation_pressure_water_pa(T_K: np.ndarray | float) -> np.ndarray:
    """Saturation vapor pressure of water over liquid water using Magnus equation."""
    T_K_arr = np.asarray(T_K, dtype=float)
    T_C = T_K_arr - 273.15
    p_hPa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return p_hPa * 100.0


def feed_composition_from_weather(
    T_K: float,
    RH_frac: float,
    P_Pa: float,
    CO2_ppm: float,
) -> dict[str, float]:
    """Convert T-RH-P into humid-air feed composition."""
    RH = float(np.clip(RH_frac, 0.0, 0.999999))
    p_sat = float(saturation_pressure_water_pa(T_K))
    p_H2O = min(RH * p_sat, 0.99 * P_Pa)

    x_CO2 = CO2_ppm * 1e-6
    p_CO2 = max(x_CO2 * (P_Pa - p_H2O), 0.0)

    y_H2O = p_H2O / P_Pa
    y_CO2 = p_CO2 / P_Pa
    y_O2 = 0.21 * max(1.0 - y_CO2 - y_H2O, 0.0)
    y_N2 = max(1.0 - y_CO2 - y_H2O - y_O2, 0.0)

    y_sum = y_CO2 + y_H2O + y_N2 + y_O2
    return {
        "CO2": y_CO2 / y_sum,
        "H2O": y_H2O / y_sum,
        "N2": y_N2 / y_sum,
        "O2": y_O2 / y_sum,
        "p_sat_H2O_Pa": p_sat,
        "p_H2O_Pa": p_H2O,
        "p_CO2_Pa": p_CO2,
    }


def mixture_mw_kg_mol(y: np.ndarray) -> np.ndarray:
    return (
        y[..., IDX["CO2"]] * MW["CO2"]
        + y[..., IDX["H2O"]] * MW["H2O"]
        + y[..., IDX["N2"]] * MW["N2"]
        + y[..., IDX["O2"]] * MW["O2"]
    )


def mixture_cp_molar_J_molK(y: np.ndarray) -> np.ndarray:
    return (
        y[..., IDX["CO2"]] * CP_GAS_MOLAR["CO2"]
        + y[..., IDX["H2O"]] * CP_GAS_MOLAR["H2O"]
        + y[..., IDX["N2"]] * CP_GAS_MOLAR["N2"]
        + y[..., IDX["O2"]] * CP_GAS_MOLAR["O2"]
    )


def gas_viscosity_air_sutherland(T_K: np.ndarray | float) -> np.ndarray:
    """Approximate humid-air viscosity with Sutherland's law for air."""
    T = np.asarray(T_K, dtype=float)
    mu0 = 1.716e-5
    T0 = 273.15
    S = 111.0
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


def gab_h2o_loading(T_K: np.ndarray, RH_frac: np.ndarray, ads: AdsorbentConfig) -> np.ndarray:
    """Temperature-dependent GAB water isotherm."""
    T = np.asarray(T_K, dtype=float)
    x = np.clip(np.asarray(RH_frac, dtype=float), 1e-12, 0.999999)

    E10_plus = -44.38 * T + 57_220.0
    E1 = ads.gab_C_J_mol - np.exp(ads.gab_D_K_inv * T)
    E2_9 = ads.gab_F_J_mol + ads.gab_G_J_molK * T

    c_gab = np.exp((E1 - E10_plus) / (R * T))
    k_gab = np.exp((E2_9 - E10_plus) / (R * T))

    kx = np.clip(k_gab * x, 1e-12, 0.999999)
    denom = (1.0 - kx) * (1.0 + (c_gab - 1.0) * kx)
    denom = np.maximum(denom, 1e-20)

    q = ads.gab_qm_mol_kg * k_gab * c_gab * x / denom
    return np.maximum(q, 0.0)


def toth_loading(
    T_K: np.ndarray,
    p_CO2_Pa: np.ndarray,
    qmax: float,
    b0_Pa_inv: float,
    minus_dH_J_mol: float,
    t0: float,
    a: float,
    T0_K: float,
) -> np.ndarray:
    """Temperature-dependent Toth isotherm."""
    T = np.asarray(T_K, dtype=float)
    p = np.maximum(np.asarray(p_CO2_Pa, dtype=float), 0.0)

    b = b0_Pa_inv * np.exp(minus_dH_J_mol / (R * T))
    t = t0 + a * (1.0 - T0_K / T)
    t = np.clip(t, 1e-6, None)

    bp = np.maximum(b * p, 1e-300)
    q = qmax * bp / ((1.0 + np.power(bp, t)) ** (1.0 / t))
    return np.maximum(q, 0.0)


def wadst_co2_loading(
    T_K: np.ndarray,
    p_CO2_Pa: np.ndarray,
    q_H2O_mol_kg: np.ndarray,
    ads: AdsorbentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-average dual-site Toth model for humid CO2 loading."""
    q_dry = toth_loading(
        T_K=T_K,
        p_CO2_Pa=p_CO2_Pa,
        qmax=ads.dry_qmax_mol_kg,
        b0_Pa_inv=ads.dry_b0_Pa_inv,
        minus_dH_J_mol=ads.dry_minus_dH_J_mol,
        t0=ads.dry_t0,
        a=ads.dry_a,
        T0_K=ads.T0_K,
    )
    q_wet = toth_loading(
        T_K=T_K,
        p_CO2_Pa=p_CO2_Pa,
        qmax=ads.wet_qmax_mol_kg,
        b0_Pa_inv=ads.wet_b0_Pa_inv,
        minus_dH_J_mol=ads.wet_minus_dH_J_mol,
        t0=ads.wet_t0,
        a=ads.wet_a,
        T0_K=ads.T0_K,
    )

    qh = np.maximum(np.asarray(q_H2O_mol_kg, dtype=float), 1e-12)
    w_wet = np.clip(np.exp(-ads.wadst_A_mol_kg / qh), 0.0, 1.0)
    w_dry = 1.0 - w_wet
    q = w_dry * q_dry + w_wet * q_wet
    return np.maximum(q, 0.0), q_dry, q_wet, w_wet, w_dry


def equilibrium_loadings_from_gas_and_solid(
    C: np.ndarray,
    Tg: np.ndarray,
    Ts: np.ndarray,
    ads: AdsorbentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate q*_CO2 and q*_H2O from gas concentrations, gas temperature,
    and sorbent temperature.

    Partial pressures are gas-phase quantities and are therefore computed from
    C_i * R * Tg. The isotherm temperature is the sorbent/surface temperature
    Ts. RH_surface is used for GAB driving force; RH_gas is retained as a
    diagnostic for gas-phase supersaturation.
    """
    C_pos = np.maximum(C, 1e-20)
    Tg_safe = np.maximum(np.asarray(Tg, dtype=float), 1e-9)
    Ts_safe = np.maximum(np.asarray(Ts, dtype=float), 1e-9)

    p_CO2 = C_pos[:, IDX["CO2"]] * R * Tg_safe
    p_H2O = C_pos[:, IDX["H2O"]] * R * Tg_safe

    p_sat_surface = np.maximum(saturation_pressure_water_pa(Ts_safe), 1e-12)
    p_sat_gas = np.maximum(saturation_pressure_water_pa(Tg_safe), 1e-12)
    RH_surface_raw = p_H2O / p_sat_surface
    RH_gas_raw = p_H2O / p_sat_gas
    RH_for_GAB = np.clip(RH_surface_raw, 1e-12, 0.999999)

    q_H2O = gab_h2o_loading(Ts_safe, RH_for_GAB, ads)
    q_CO2, q_dry, q_wet, w_wet, _ = wadst_co2_loading(Ts_safe, p_CO2, q_H2O, ads)
    return q_CO2, q_H2O, q_dry, q_wet, w_wet, RH_surface_raw, RH_gas_raw


def equilibrium_loadings(
    C: np.ndarray,
    T: np.ndarray,
    ads: AdsorbentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper for cases where Tg and Ts are intentionally
    assumed equal. New detailed-model calls should use
    equilibrium_loadings_from_gas_and_solid(C, Tg, Ts, ads).
    """
    q_CO2, q_H2O, q_dry, q_wet, w_wet, RH_surface_raw, _ = equilibrium_loadings_from_gas_and_solid(
        C, T, T, ads
    )
    return q_CO2, q_H2O, q_dry, q_wet, w_wet, RH_surface_raw


# =============================================================================
# MODEL CLASS
# =============================================================================

class TVSABedModel:
    def __init__(
        self,
        weather: WeatherConfig,
        bed: BedConfig,
        ads: AdsorbentConfig,
        cycle: CycleConfig,
        numeric: NumericConfig,
    ) -> None:
        self.weather = weather
        self.bed = bed
        self.ads = ads
        self.cycle = cycle
        self.numeric = numeric

        self.N = bed.n_nodes
        self.z = np.linspace(0.0, bed.bed_length_m, self.N)
        self.dz = self.z[1] - self.z[0]
        self.area = math.pi * (bed.bed_diameter_m / 2.0) ** 2
        self.volume = self.area * bed.bed_length_m
        self.node_volume = self.volume / self.N
        self.m_ads_kg = bed.rho_bulk_kg_m3_bed * self.volume
        self.d_p = 2.0 * bed.particle_radius_m
        self.epsilon_inter = float(bed.epsilon_b)
        self.epsilon_total = float(bed.epsilon_b + (1.0 - bed.epsilon_b) * bed.epsilon_p)
        self.epsilon_gas_accum = (
            self.epsilon_total
            if bool(getattr(numeric, "use_total_voidage_for_gas_accumulation", True))
            else self.epsilon_inter
        )

        self.a_gs = 6.0 * (1.0 - bed.epsilon_b) / self.d_p
        self.A_hx_node = bed.a_hx_m2_m3 * self.node_volume
        self.jacket_mass_node = bed.jacket_holdup_kg_per_m3_bed * self.node_volume

        # Wall mass approximated from heat exchanger area density and wall thickness.
        self.wall_area_node = bed.a_hx_m2_m3 * self.node_volume
        self.wall_mass_node = bed.wall_density_kg_m3 * bed.wall_thickness_m * self.wall_area_node

        self.T_amb_K = weather.T_C + 273.15
        self.RH_frac = np.clip(weather.RH_percent / 100.0, 0.0, 0.999999)
        self.P_amb_Pa = weather.P_Pa
        self.feed = feed_composition_from_weather(
            self.T_amb_K,
            self.RH_frac,
            self.P_amb_Pa,
            weather.CO2_ppm,
        )
        self.y_feed = np.array([self.feed[c] for c in COMPONENTS], dtype=float)
        self.Ctot_feed = self.P_amb_Pa / (R * self.T_amb_K)
        self.C_feed = self.y_feed * self.Ctot_feed

        # Jajjawi superficial velocity from 6.2 L/s and D = 10 cm.
        self.u_ads_m_s = cycle.air_volumetric_flow_m3_s_ref / self.area
        self.Vdot_air_m3_s = self.u_ads_m_s * self.area
        self.ndot_feed_mol_s = self.Vdot_air_m3_s * self.Ctot_feed

        self.solver_log: list[dict] = []
        self.fast_substep_diagnostics: list[dict] = []
        self.jac_sparsity = self.build_jac_sparsity() if self.numeric.use_jac_sparsity else None
        self.state_scale = self.build_state_scale_vector() if self.numeric.use_nondimensional_state else np.ones(
            self.N * len(COMPONENTS) + self.N * len(ADS_COMPONENTS) + 4 * self.N,
            dtype=float,
        )
        if self.numeric.use_nondimensional_state:
            self.atol_solver = self.numeric.atol
        else:
            self.atol_solver = self.build_atol_vector() if self.numeric.use_vector_atol else self.numeric.atol

    def build_atol_vector(self) -> np.ndarray:
        N = self.N
        num = self.numeric
        return np.concatenate([
            np.full(N * len(COMPONENTS), num.atol_C_mol_m3, dtype=float),
            np.full(N * len(ADS_COMPONENTS), num.atol_q_mol_kg, dtype=float),
            np.full(N, num.atol_T_K, dtype=float),  # Tg
            np.full(N, num.atol_T_K, dtype=float),  # Ts
            np.full(N, num.atol_T_K, dtype=float),  # Tw
            np.full(N, num.atol_T_K, dtype=float),  # Tj
        ])


    def build_state_scale_vector(self) -> np.ndarray:
        N = self.N
        num = self.numeric

        C_comp_scale = np.maximum(np.abs(self.C_feed), num.C_scale_floor_mol_m3)

        C0 = np.tile(self.C_feed, (N, 1))
        T0 = np.full(N, self.T_amb_K)
        qco2, qh2o, *_ = equilibrium_loadings_from_gas_and_solid(C0, T0, T0, self.ads)
        q_comp_scale = np.array([
            max(float(np.nanmean(np.abs(qco2))), num.q_scale_floor_mol_kg),
            max(float(np.nanmean(np.abs(qh2o))), num.q_scale_floor_mol_kg),
        ])

        return np.concatenate([
            np.tile(C_comp_scale, N),
            np.tile(q_comp_scale, N),
            np.full(N, num.T_scale_K, dtype=float),
            np.full(N, num.T_scale_K, dtype=float),
            np.full(N, num.T_scale_K, dtype=float),
            np.full(N, num.T_scale_K, dtype=float),
        ])

    def to_solver_state(self, y_dim: np.ndarray) -> np.ndarray:
        """Convert dimensional state to the solver state."""
        if not self.numeric.use_nondimensional_state:
            return y_dim
        return y_dim / self.state_scale

    def from_solver_state(self, y_solver: np.ndarray) -> np.ndarray:
        """Convert solver state to dimensional state."""
        if not self.numeric.use_nondimensional_state:
            return y_solver
        return y_solver * self.state_scale

    def from_solver_state_matrix(self, y_solver_matrix: np.ndarray) -> np.ndarray:
        """Convert solve_ivp sol.y matrix to dimensional states."""
        if not self.numeric.use_nondimensional_state:
            return y_solver_matrix
        return y_solver_matrix * self.state_scale[:, None]

    def rhs_solver(self, t: float, y_solver: np.ndarray, step_name: str) -> np.ndarray:
        y_dim = self.from_solver_state(y_solver)
        dy_dim = self.rhs(t, y_dim, step_name)
        if not self.numeric.use_nondimensional_state:
            return dy_dim
        return dy_dim / self.state_scale

    def solver_method_for_step(self, step_name: str) -> str:
        """Choose the time integrator for each cycle step."""
        if step_name in {"evacuation", "repressurization"}:
            return self.numeric.pressure_step_method
        return self.numeric.method

    def solver_max_step_for_step(self, step_name: str) -> float:
        """Step-specific maximum time step."""
        if step_name in {"evacuation", "repressurization"}:
            return min(self.numeric.max_step_s, self.numeric.pressure_step_max_step_s)
        return self.numeric.max_step_s

    def state_indices_for_node(self, i: int) -> list[int]:
        """Return all state-vector indices associated with axial node i."""
        N = self.N
        nC = N * len(COMPONENTS)
        nq = N * len(ADS_COMPONENTS)
        base_Tg = nC + nq
        base_Ts = base_Tg + N
        base_Tw = base_Ts + N
        base_Tj = base_Tw + N

        idx = []
        idx.extend(range(i * len(COMPONENTS), (i + 1) * len(COMPONENTS)))
        idx.extend(range(nC + i * len(ADS_COMPONENTS), nC + (i + 1) * len(ADS_COMPONENTS)))
        idx.append(base_Tg + i)
        idx.append(base_Ts + i)
        idx.append(base_Tw + i)
        idx.append(base_Tj + i)
        return idx

    def build_jac_sparsity(self):
        N = self.N
        n_state = N * len(COMPONENTS) + N * len(ADS_COMPONENTS) + 4 * N
        S = lil_matrix((n_state, n_state), dtype=bool)

        for i in range(N):
            rows = self.state_indices_for_node(i)
            cols = []
            for j in range(max(0, i - 1), min(N, i + 2)):
                cols.extend(self.state_indices_for_node(j))

            # Jacket plug-flow has one-direction coupling from previous jacket node.
            if i > 0:
                cols.extend(self.state_indices_for_node(i - 1))

            for r in rows:
                S[r, cols] = True

        return S.tocsr()

    # -------------------------------------------------------------------------
    # State vector packing
    # -------------------------------------------------------------------------

    def pack(self, C: np.ndarray, q: np.ndarray, Tg: np.ndarray, Ts: np.ndarray, Tw: np.ndarray, Tj: np.ndarray) -> np.ndarray:
        return np.concatenate([C.ravel(), q.ravel(), Tg, Ts, Tw, Tj])

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = self.N
        nC = N * len(COMPONENTS)
        nq = N * len(ADS_COMPONENTS)
        C = y[:nC].reshape((N, len(COMPONENTS)))
        q = y[nC:nC+nq].reshape((N, len(ADS_COMPONENTS)))
        base = nC + nq
        Tg = y[base:base+N]
        Ts = y[base+N:base+2*N]
        Tw = y[base+2*N:base+3*N]
        Tj = y[base+3*N:base+4*N]
        return C, q, Tg, Ts, Tw, Tj

    # -------------------------------------------------------------------------
    # Initial state
    # -------------------------------------------------------------------------

    def initial_state(self) -> np.ndarray:
        C0 = np.tile(self.C_feed, (self.N, 1))
        Tg0 = np.full(self.N, self.T_amb_K)
        Ts0 = np.full(self.N, self.T_amb_K)
        Tw0 = np.full(self.N, self.T_amb_K)
        Tj0 = np.full(self.N, self.T_amb_K)

        qco2, qh2o, *_ = equilibrium_loadings_from_gas_and_solid(C0, Tg0, Ts0, self.ads)
        q0 = np.column_stack([qco2, qh2o])

        return self.pack(C0, q0, Tg0, Ts0, Tw0, Tj0)

    def regularize_gas_state(self, C: np.ndarray, Tg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num = self.numeric
        relw = num.smooth_floor_rel_width
        absw = num.smooth_floor_abs_width

        Tg_arr = np.asarray(Tg, dtype=float)
        Tg_arr = np.nan_to_num(
            Tg_arr,
            nan=self.T_amb_K,
            posinf=num.max_temperature_K,
            neginf=num.min_temperature_K,
        )
        Tg_safe = smooth_clip(
            Tg_arr,
            num.min_temperature_K,
            num.max_temperature_K,
            rel_width=0.0,
            abs_width=max(num.smooth_temperature_width_K, 1e-9),
        )

        C_arr = np.asarray(C, dtype=float)
        P_cap = max(num.max_pressure_regularization_Pa, 1.2 * self.P_amb_Pa)
        C_component_cap = P_cap / (R * num.min_temperature_K)

        C_arr = np.nan_to_num(
            C_arr,
            nan=num.min_concentration_mol_m3,
            posinf=C_component_cap,
            neginf=num.min_concentration_mol_m3,
        )
        C_safe = smooth_clip(
            C_arr,
            num.min_concentration_mol_m3,
            C_component_cap,
            rel_width=relw,
            abs_width=absw,
        )

        Ctot_raw = np.sum(C_safe, axis=1)
        Ctot_raw = np.nan_to_num(
            Ctot_raw,
            nan=0.0,
            posinf=C_component_cap * len(COMPONENTS),
            neginf=0.0,
        )
        Ctot_raw = smooth_floor(
            Ctot_raw,
            len(COMPONENTS) * num.min_concentration_mol_m3,
            rel_width=relw,
            abs_width=absw,
        )

        ygas = C_safe / Ctot_raw[:, None]
        ygas = np.nan_to_num(ygas, nan=0.0, posinf=0.0, neginf=0.0)
        ysum = np.sum(ygas, axis=1)
        bad_y = (~np.isfinite(ysum)) | (ysum <= 0.0)
        if np.any(bad_y):
            ygas[bad_y, :] = self.y_feed[None, :]
            ysum[bad_y] = 1.0
        ygas = ygas / np.maximum(ysum[:, None], 1e-30)

        Ctot = smooth_floor(
            Ctot_raw,
            len(COMPONENTS) * num.min_concentration_mol_m3,
            rel_width=relw,
            abs_width=absw,
        )
        Ctot = np.nan_to_num(
            Ctot,
            nan=len(COMPONENTS) * num.min_concentration_mol_m3,
            posinf=C_component_cap * len(COMPONENTS),
            neginf=len(COMPONENTS) * num.min_concentration_mol_m3,
        )

        C_safe = ygas * Ctot[:, None]
        ygas = C_safe / np.maximum(Ctot[:, None], 1e-30)
        ysum = np.sum(ygas, axis=1)
        ygas = ygas / np.maximum(ysum[:, None], 1e-30)
        C_safe = ygas * Ctot[:, None]

        return C_safe, Tg_safe, Ctot, ygas

    # -------------------------------------------------------------------------
    # Cycle and jacket
    # -------------------------------------------------------------------------

    def step_schedule(self) -> list[dict[str, float | str]]:
        c = self.cycle
        return [
            {"name": "adsorption", "duration_s": c.adsorption_time_s},
            {"name": "evacuation", "duration_s": c.evacuation_time_s},
            {"name": "heating_desorption", "duration_s": c.heating_desorption_time_s},
            {"name": "closed_cooling", "duration_s": c.cooling_time_s},
            {"name": "repressurization", "duration_s": c.repressurization_time_s},
        ]

    def jacket_inlet_temperature(self, step_name: str, t_step_s: float = 0.0) -> float:
        """Jacket inlet temperature with finite switching ramp."""
        tau = max(float(self.numeric.jacket_inlet_ramp_time_s), 0.0)
        t = max(float(t_step_s), 0.0)
        if step_name == "heating_desorption":
            if tau > 0.0:
                return self.cycle.T_des_K - (self.cycle.T_des_K - self.T_amb_K) * __import__('math').exp(-t / tau)
            return self.cycle.T_des_K
        if step_name == "closed_cooling":
            if tau > 0.0:
                return self.cycle.T_coolant_K + (self.cycle.T_des_K - self.cycle.T_coolant_K) * __import__('math').exp(-t / tau)
            return self.cycle.T_coolant_K
        return self.T_amb_K

    def jacket_mdot(self, step_name: str) -> float:
        if step_name == "heating_desorption":
            return self.cycle.heating_fluid_mdot_kg_s
        if step_name == "closed_cooling":
            return self.cycle.cooling_fluid_mdot_kg_s
        return 0.0

    @staticmethod
    def smoothstep01(x: float) -> float:
        """C1 smooth opening function from 0 to 1."""
        s = float(np.clip(x, 0.0, 1.0))
        return s * s * (3.0 - 2.0 * s)

    def valve_opening(self, step_name: str, t_step_s: float) -> float:
        """Smooth valve opening for stiff pressure-boundary steps."""
        if step_name == "evacuation":
            ramp = self.cycle.evacuation_valve_ramp_time_s
        elif step_name == "heating_desorption":
            ramp = self.cycle.desorption_product_valve_ramp_time_s
        elif step_name == "repressurization":
            ramp = self.cycle.repressurization_valve_ramp_time_s
        else:
            return 1.0

        if ramp <= 0.0:
            return 1.0
        return self.smoothstep01(float(t_step_s) / float(ramp))

    def repressurization_pressure_factor(self, P_nodes_Pa: np.ndarray) -> float:
        Pavg = float(np.mean(P_nodes_Pa))
        margin = max(float(self.cycle.repressurization_pressure_stop_margin_Pa), 0.0)
        if Pavg >= self.P_amb_Pa - margin:
            return 0.0

        denom = max(self.P_amb_Pa - self.cycle.P_vac_Pa, 1.0)
        factor = (self.P_amb_Pa - margin - Pavg) / denom
        return float(np.clip(factor, 0.0, 1.0))


    def product_pressure_factor(self, P_nodes_Pa: np.ndarray, step_name: str) -> float:
        if step_name not in {"evacuation", "heating_desorption"}:
            return 1.0

        margin = max(float(self.cycle.evacuation_pressure_stop_margin_Pa), 0.0)
        # Use Pmax rather than Pavg so local desorbed gas can still leave.
        Pdrive = float(np.max(P_nodes_Pa))
        P_low = self.cycle.P_vac_Pa + margin
        denom = max(self.P_amb_Pa - P_low, 1.0)
        raw = (Pdrive - P_low) / denom
        return self.smoothstep01(raw)

    def near_vacuum_temperature_mask(self, Ctot: np.ndarray, Tg: np.ndarray, step_name: str) -> np.ndarray:
        if not self.numeric.use_near_vacuum_gas_temperature_equilibrium:
            return np.zeros_like(Ctot, dtype=bool)
        if step_name not in {"heating_desorption", "closed_cooling"}:
            return np.zeros_like(Ctot, dtype=bool)
        P_nodes = np.asarray(Ctot, dtype=float) * R * np.asarray(Tg, dtype=float)
        return P_nodes <= float(self.numeric.gas_temperature_eq_pressure_threshold_Pa)

    # -------------------------------------------------------------------------
    # Ergun and finite-volume fluxes
    # -------------------------------------------------------------------------

    def ergun_velocity_from_gradient(self, gradP_Pa_m: np.ndarray, mu: np.ndarray, rho: np.ndarray) -> np.ndarray:
        eps = self.bed.epsilon_b
        dp = self.d_p
        A = 150.0 * mu * (1.0 - eps) ** 2 / (dp ** 2 * eps ** 3)
        B = 1.75 * rho * (1.0 - eps) / (dp * eps ** 3)

        S = -gradP_Pa_m
        sign = np.sign(S)
        Sabs = np.abs(S)

        # B*u^2 + A*u - S = 0 for magnitude.
        u_abs = np.where(
            B > 1e-30,
            (-A + np.sqrt(np.maximum(A * A + 4.0 * B * Sabs, 0.0))) / (2.0 * B),
            Sabs / np.maximum(A, 1e-30),
        )
        return sign * np.maximum(u_abs, 0.0)

    def face_velocities(self, C: np.ndarray, Tg: np.ndarray, step_name: str, t_step_s: float = 0.0) -> np.ndarray:
        N = self.N
        u_faces = np.zeros(N + 1)

        C_safe, Tg_safe, Ctot, ygas = self.regularize_gas_state(C, Tg)
        P = Ctot * R * Tg_safe
        mu_nodes = gas_viscosity_air_sutherland(Tg_safe)
        mw_nodes = mixture_mw_kg_mol(ygas)
        rho_nodes = P * mw_nodes / (R * Tg_safe)

        P = np.nan_to_num(P, nan=self.P_amb_Pa, posinf=self.P_amb_Pa, neginf=self.cycle.P_vac_Pa)
        mu_nodes = np.nan_to_num(mu_nodes, nan=1.8e-5, posinf=1.8e-5, neginf=1.8e-5)
        rho_nodes = np.nan_to_num(rho_nodes, nan=1.0, posinf=10.0, neginf=1e-6)

        if step_name == "adsorption":
            u_faces[:] = self.u_ads_m_s
            return u_faces

        if step_name in {"evacuation", "heating_desorption"}:
            u_faces[0] = 0.0

            if N > 1:
                gradP = (P[1:] - P[:-1]) / self.dz
                mu = 0.5 * (mu_nodes[:-1] + mu_nodes[1:])
                rho = 0.5 * (rho_nodes[:-1] + rho_nodes[1:])
                u_faces[1:N] = self.ergun_velocity_from_gradient(gradP, mu, rho)

            gradP_out = (self.cycle.P_vac_Pa - P[-1]) / (0.5 * self.dz)
            u_faces[-1] = self.ergun_velocity_from_gradient(
                np.array([gradP_out]),
                np.array([mu_nodes[-1]]),
                np.array([rho_nodes[-1]]),
            )[0]

            opening = self.valve_opening(step_name, t_step_s)
            pctrl = self.product_pressure_factor(P, step_name)
            u_faces = np.clip(u_faces, 0.0, self.cycle.max_product_superficial_velocity_m_s)
            u_faces[-1] *= opening * pctrl
            return u_faces

        if step_name == "closed_cooling":
            return u_faces

        if step_name == "repressurization":
            gradP_in = (P[0] - self.P_amb_Pa) / (0.5 * self.dz)
            u_faces[0] = self.ergun_velocity_from_gradient(
                np.array([gradP_in]),
                np.array([mu_nodes[0]]),
                np.array([rho_nodes[0]]),
            )[0]

            if N > 1:
                gradP = (P[1:] - P[:-1]) / self.dz
                mu = 0.5 * (mu_nodes[:-1] + mu_nodes[1:])
                rho = 0.5 * (rho_nodes[:-1] + rho_nodes[1:])
                u_faces[1:N] = self.ergun_velocity_from_gradient(gradP, mu, rho)

            u_faces[-1] = 0.0
            opening = self.valve_opening(step_name, t_step_s)
            pctrl = self.repressurization_pressure_factor(P)
            u_faces = np.clip(u_faces, 0.0, self.cycle.max_repress_superficial_velocity_m_s)
            u_faces[0] *= opening * pctrl
            return u_faces

        return u_faces

    def component_convective_fluxes(self, C: np.ndarray, Tg: np.ndarray, step_name: str) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        C_safe, Tg_safe, _, _ = self.regularize_gas_state(C, Tg)
        u_faces = self.face_velocities(C_safe, Tg_safe, step_name)
        flux = np.zeros((N + 1, len(COMPONENTS)))

        for f in range(N + 1):
            u = u_faces[f]
            if abs(u) < 1e-15:
                continue

            if u >= 0:
                if f == 0:
                    # Feed boundary active for adsorption/repressurization, otherwise closed.
                    if step_name in {"adsorption", "repressurization"}:
                        Cup = self.C_feed
                    else:
                        Cup = np.zeros(len(COMPONENTS))
                else:
                    Cup = C_safe[f - 1]
            else:
                if f == N:
                    Cup = C_safe[-1]
                else:
                    Cup = C_safe[f]

            flux[f, :] = u * Cup

        return flux, u_faces

    def component_convective_fluxes_given_u(self, C_safe: np.ndarray, u_faces: np.ndarray, step_name: str) -> np.ndarray:
        N = self.N
        ncomp = len(COMPONENTS)
        flux = np.zeros((N + 1, ncomp), dtype=float)

        u = np.asarray(u_faces, dtype=float)
        pos = u >= 0.0
        active = np.abs(u) >= 1e-15

        if N > 1:
            idx = np.arange(1, N)
            pos_int = pos[idx]
            Cup = np.where(pos_int[:, None], C_safe[idx - 1, :], C_safe[idx, :])
            flux[idx, :] = u[idx, None] * Cup

        # Left boundary.
        if active[0]:
            if u[0] >= 0.0:
                Cup0 = self.C_feed if step_name in {"adsorption", "repressurization"} else np.zeros(ncomp)
            else:
                Cup0 = C_safe[0]
            flux[0, :] = u[0] * Cup0

        # Right boundary.
        if active[N]:
            if u[N] >= 0.0:
                CupN = C_safe[-1]
            else:
                CupN = C_safe[-1]
            flux[N, :] = u[N] * CupN

        # Optional speed/conditioning approximation: after evacuation, inert
        # gas holdup is small and does not control CO2/H2O productivity. Freeze
        # inert transport during heating to avoid stiff, nearly-empty N2/O2 states.
        if step_name == "heating_desorption" and self.numeric.freeze_inert_during_heating_desorption:
            flux[:, IDX["N2"]] = 0.0
            flux[:, IDX["O2"]] = 0.0

        return flux


    def axial_dispersion_face_coeffs(self, u_faces: np.ndarray) -> np.ndarray:
        u_abs = np.abs(np.asarray(u_faces, dtype=float))
        Dm_vec = np.array([DM[comp] for comp in COMPONENTS], dtype=float)
        D_face = Dm_vec[None, :] + self.numeric.longitudinal_dispersivity_m * u_abs[:, None]
        D_face += (u_abs[:, None] * self.d_p / max(self.numeric.axial_dispersion_pe, 1e-12))
        return np.maximum(D_face, Dm_vec[None, :])

    def component_diffusive_fluxes(
        self,
        C: np.ndarray,
        Tg: np.ndarray,
        step_name: str,
        u_faces: np.ndarray | None = None,
    ) -> np.ndarray:
        N = self.N
        C_safe, _, _, _ = self.regularize_gas_state(C, Tg)
        flux = np.zeros((N + 1, len(COMPONENTS)), dtype=float)
        if u_faces is None:
            u_faces = self.face_velocities(C_safe, Tg, step_name)
        D_face = self.axial_dispersion_face_coeffs(u_faces)

        if step_name == "adsorption":
            grad_left = (C_safe[0, :] - self.C_feed) / (0.5 * self.dz)
            flux[0, :] = -self.epsilon_inter * D_face[0, :] * grad_left

        if N > 1:
            grad_internal = (C_safe[1:, :] - C_safe[:-1, :]) / self.dz
            flux[1:N, :] = -self.epsilon_inter * D_face[1:N, :] * grad_internal

        # Optional inert freeze during heating.
        if step_name == "heating_desorption" and self.numeric.freeze_inert_during_heating_desorption:
            flux[:, IDX["N2"]] = 0.0
            flux[:, IDX["O2"]] = 0.0

        return flux


    def gas_enthalpy_fluxes(self, C: np.ndarray, Tg: np.ndarray, step_name: str) -> tuple[np.ndarray, np.ndarray]:
        N = self.N
        comp_flux, u_faces = self.component_convective_fluxes(C, Tg, step_name)
        h_flux = np.zeros(N + 1)

        for f in range(N + 1):
            ndot_area = float(np.sum(comp_flux[f, :]))
            if ndot_area <= 0:
                continue

            y_face = comp_flux[f, :] / ndot_area

            if f == 0 and step_name in {"adsorption", "repressurization"}:
                T_face = self.T_amb_K
            elif f == N:
                T_face = Tg[-1]
            else:
                T_face = Tg[max(f - 1, 0)]

            cp_molar = float(mixture_cp_molar_J_molK(y_face))
            h_flux[f] = ndot_area * cp_molar * T_face

        return h_flux, u_faces

    def gas_enthalpy_fluxes_from_component_flux(
        self,
        comp_flux: np.ndarray,
        Tg_safe: np.ndarray,
        step_name: str,
    ) -> np.ndarray:
        N = self.N
        ndot_area = np.sum(comp_flux, axis=1)
        h_flux = np.zeros(N + 1, dtype=float)
        active = ndot_area > 0.0
        if not np.any(active):
            return h_flux

        y_face = np.zeros_like(comp_flux)
        y_face[active, :] = comp_flux[active, :] / ndot_area[active, None]
        ysum = np.sum(y_face, axis=1)
        bad = active & ((~np.isfinite(ysum)) | (ysum <= 0.0))
        if np.any(bad):
            y_face[bad, :] = self.y_feed[None, :]
            ysum[bad] = 1.0
        y_face[active, :] = y_face[active, :] / np.maximum(ysum[active, None], 1e-30)

        T_face = np.empty(N + 1, dtype=float)
        T_face[0] = self.T_amb_K if step_name in {"adsorption", "repressurization"} else Tg_safe[0]
        if N > 1:
            T_face[1:N] = Tg_safe[:N - 1]
        T_face[N] = Tg_safe[-1]

        cp_molar = mixture_cp_molar_J_molK(y_face)
        h_flux[active] = ndot_area[active] * cp_molar[active] * T_face[active]
        return h_flux


    # -------------------------------------------------------------------------
    # Core ODE
    # -------------------------------------------------------------------------

    def rhs(self, t: float, y: np.ndarray, step_name: str) -> np.ndarray:
        N = self.N
        b = self.bed
        a = self.ads
        num = self.numeric

        C, q, Tg, Ts, Tw, Tj = self.unpack(y)

        C_safe, Tg_safe, Ctot, ygas = self.regularize_gas_state(C, Tg)
        q_arr = np.nan_to_num(q, nan=num.min_loading_mol_kg, posinf=100.0, neginf=num.min_loading_mol_kg)
        q_safe = smooth_floor(
            q_arr,
            num.min_loading_mol_kg,
            rel_width=num.smooth_floor_rel_width,
            abs_width=max(num.smooth_floor_abs_width, 1e-9),
        )
        Ts_safe = smooth_clip(Ts, num.min_temperature_K, num.max_temperature_K, rel_width=0.0, abs_width=max(num.smooth_temperature_width_K, 1e-9))
        Tw_safe = smooth_clip(Tw, num.min_temperature_K, num.max_temperature_K, rel_width=0.0, abs_width=max(num.smooth_temperature_width_K, 1e-9))
        Tj_safe = smooth_clip(Tj, num.min_temperature_K, num.max_temperature_K, rel_width=0.0, abs_width=max(num.smooth_temperature_width_K, 1e-9))

        P_nodes = Ctot * R * Tg_safe
        near_vac_Tg_eq = self.near_vacuum_temperature_mask(Ctot, Tg_safe, step_name)
        # Use Ts as the effective gas temperature in near-vacuum thermal/flow
        # calculations; gas holdup is tiny, so Tg as an independent thermal
        # state is numerically expensive and weakly observable.
        Tg_flow = np.where(near_vac_Tg_eq, Ts_safe, Tg_safe)

        cp_molar = mixture_cp_molar_J_molK(ygas)

        P_thermal_floor_Pa = max(
            num.gas_thermal_pressure_floor_Pa,
            num.gas_pressure_floor_min_Pa,
        )
        cp_vol_floor = (P_thermal_floor_Pa / (R * Tg_safe)) * cp_molar
        cp_vol = np.maximum(Ctot * cp_molar, cp_vol_floor)

        # Equilibrium and LDF kinetics. Partial pressures are computed with Tg;
        # isotherm loading uses sorbent temperature Ts.
        qeq_co2, qeq_h2o, _, _, _, RH_surface_raw, RH_gas_raw = equilibrium_loadings_from_gas_and_solid(
            C_safe, Tg_safe, Ts_safe, a
        )
        dqdt = np.zeros_like(q_safe)
        dqdt[:, AIDX["CO2"]] = a.k_ldf_co2_s * (qeq_co2 - q_safe[:, AIDX["CO2"]])
        dqdt[:, AIDX["H2O"]] = a.k_ldf_h2o_s * (qeq_h2o - q_safe[:, AIDX["H2O"]])

        if step_name == "closed_cooling" and self.numeric.freeze_mass_during_closed_cooling:
            dqdt[:, :] = 0.0

        # Gas-phase mass balance with face fluxes.
        # Velocity and convective flux are computed once and reused for enthalpy.
        u_faces = self.face_velocities(C_safe, Tg_flow, step_name, t)
        conv_flux = self.component_convective_fluxes_given_u(C_safe, u_faces, step_name)
        diff_flux = self.component_diffusive_fluxes(C_safe, Tg_flow, step_name, u_faces=u_faces)
        total_flux = conv_flux + diff_flux

        eps_accum = max(self.epsilon_gas_accum, 1e-12)
        dCdt = -(total_flux[1:, :] - total_flux[:-1, :]) / (self.dz * eps_accum)

        # Adsorption/desorption source/sink for adsorbing components.
        dCdt[:, IDX["CO2"]] += -(b.rho_bulk_kg_m3_bed / eps_accum) * dqdt[:, AIDX["CO2"]]
        dCdt[:, IDX["H2O"]] += -(b.rho_bulk_kg_m3_bed / eps_accum) * dqdt[:, AIDX["H2O"]]

        if step_name == "heating_desorption" and self.numeric.freeze_inert_during_heating_desorption:
            dCdt[:, IDX["N2"]] = 0.0
            dCdt[:, IDX["O2"]] = 0.0

        if step_name == "closed_cooling" and self.numeric.freeze_mass_during_closed_cooling:
            dCdt[:, :] = 0.0

        # Gas energy balance.
        h_flux = self.gas_enthalpy_fluxes_from_component_flux(conv_flux, Tg_flow, step_name)
        div_h = (h_flux[1:] - h_flux[:-1]) / self.dz
        dTgdt = -div_h / np.maximum(self.epsilon_gas_accum * cp_vol, 1e-9)

        # Gas axial conduction.
        d2Tg = np.zeros(N)
        d2Tg[0] = (Tg_flow[1] - 2.0 * Tg_flow[0] + (self.T_amb_K if step_name == "adsorption" else Tg_flow[0])) / (self.dz ** 2)
        d2Tg[1:-1] = (Tg_flow[2:] - 2.0 * Tg_flow[1:-1] + Tg_flow[:-2]) / (self.dz ** 2)
        d2Tg[-1] = (Tg_flow[-2] - Tg_flow[-1]) / (self.dz ** 2)
        dTgdt += b.lambda_gas_W_mK * d2Tg / np.maximum(self.epsilon_gas_accum * cp_vol, 1e-9)

        # Gas-solid heat transfer.
        hgs_term = b.h_gs_W_m2K * self.a_gs
        dTgdt += hgs_term * (Ts_safe - Tg_flow) / np.maximum(self.epsilon_gas_accum * cp_vol, 1e-9)

        if np.any(near_vac_Tg_eq):
            tau_g = max(float(self.numeric.gas_temperature_eq_tau_s), 1e-9)
            dTgdt[near_vac_Tg_eq] = (Ts_safe[near_vac_Tg_eq] - Tg_safe[near_vac_Tg_eq]) / tau_g

        # Solid energy balance.
        d2Ts = np.zeros(N)
        d2Ts[0] = (Ts_safe[1] - Ts_safe[0]) / (self.dz ** 2)
        d2Ts[1:-1] = (Ts_safe[2:] - 2.0 * Ts_safe[1:-1] + Ts_safe[:-2]) / (self.dz ** 2)
        d2Ts[-1] = (Ts_safe[-2] - Ts_safe[-1]) / (self.dz ** 2)

        solid_heat_capacity = b.rho_bulk_kg_m3_bed * b.cp_solid_J_kgK
        cp_ads_eff = (
            b.rho_bulk_kg_m3_bed
            * (
                q_safe[:, AIDX["CO2"]] * a.cp_ads_CO2_J_molK
                + q_safe[:, AIDX["H2O"]] * a.cp_ads_H2O_J_molK
            )
        )
        solid_heat_capacity_eff = np.maximum(solid_heat_capacity + cp_ads_eff, 1e-9)

        dTsdt = b.lambda_solid_W_mK * d2Ts / solid_heat_capacity_eff
        dTsdt += hgs_term * (Tg_safe - Ts_safe) / solid_heat_capacity_eff

        heat_ads_W_m3 = (
            -a.dH_ads_CO2_J_mol * b.rho_bulk_kg_m3_bed * dqdt[:, AIDX["CO2"]]
            -a.dH_ads_H2O_J_mol * b.rho_bulk_kg_m3_bed * dqdt[:, AIDX["H2O"]]
        )
        dTsdt += heat_ads_W_m3 / solid_heat_capacity_eff

        # Solid-wall heat transfer.
        q_sw_W_node = b.h_solid_wall_W_m2K * self.wall_area_node * (Tw_safe - Ts_safe)
        dTsdt += q_sw_W_node / (solid_heat_capacity_eff * self.node_volume)

        # Wall energy balance.
        if step_name == "closed_cooling" and self.numeric.robust_cooling_jacket_boundary:
            Tj_for_wall = np.full(N, self.jacket_inlet_temperature(step_name, t))
        else:
            Tj_for_wall = Tj_safe
        q_wj_W_node = b.U_hx_W_m2K * self.wall_area_node * (Tj_for_wall - Tw_safe)
        q_loss_W_node = b.U_loss_W_m2K * self.wall_area_node * (Tw_safe - self.T_amb_K)
        wall_cap_node = max(self.wall_mass_node * b.wall_cp_J_kgK, 1e-12)
        dTwdt = (-q_sw_W_node + q_wj_W_node - q_loss_W_node) / wall_cap_node

        # Jacket plug-flow model.
        Tj_inlet = self.jacket_inlet_temperature(step_name, t)
        mdot_j = self.jacket_mdot(step_name)
        cp_j = self.cycle.jacket_cp_J_kgK
        jacket_cap_node_raw = max(self.jacket_mass_node * cp_j, 1e-12)
        if mdot_j > 0.0:
            # Lower-bound the unresolved jacket-fluid time constant. This removes
            # the near-zero residence-time stiffness that was causing closed_cooling
            # to fail with required step size below machine spacing.
            jacket_cap_node = max(
                jacket_cap_node_raw,
                mdot_j * cp_j * max(self.numeric.jacket_tau_min_s, 0.0),
                1e-12,
            )
        else:
            jacket_cap_node = jacket_cap_node_raw
        dTjdt = np.zeros(N)

        if mdot_j > 0.0:
            if step_name == "closed_cooling" and self.numeric.robust_cooling_jacket_boundary:
                tau_j = max(self.numeric.jacket_tau_min_s, 1e-9)
                dTjdt[:] = (Tj_inlet - Tj_safe) / tau_j
            else:
                Tin_vec = np.empty(N, dtype=float)
                Tin_vec[0] = Tj_inlet
                if N > 1:
                    Tin_vec[1:] = Tj_safe[:-1]
                advective_W = mdot_j * cp_j * (Tin_vec - Tj_safe)
                exchange_W = -q_wj_W_node
                dTjdt[:] = (advective_W + exchange_W) / jacket_cap_node
        else:
            dTjdt[:] = 0.0

        dy = self.pack(dCdt, dqdt, dTgdt, dTsdt, dTwdt, dTjdt)

        # Final RHS finite guard. This prevents BDF/Radau numerical Jacobian
        # from receiving NaN/inf derivatives.
        dy = np.nan_to_num(
            dy,
            nan=0.0,
            posinf=self.numeric.max_abs_rhs_value,
            neginf=-self.numeric.max_abs_rhs_value,
        )
        dy = np.clip(
            dy,
            -self.numeric.max_abs_rhs_value,
            self.numeric.max_abs_rhs_value,
        )

        return dy

    # -------------------------------------------------------------------------
    # Derived quantities and inventories
    # -------------------------------------------------------------------------

    def inventory(self, C: np.ndarray, q: np.ndarray, Tg: np.ndarray, Ts: np.ndarray, Tw: np.ndarray, Tj: np.ndarray) -> dict[str, float]:
        C_safe, Tg_safe, Ctot, ygas = self.regularize_gas_state(C, Tg)

        q_safe = np.nan_to_num(
            q,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        q_safe = np.maximum(q_safe, 0.0)

        Ts_safe = np.nan_to_num(
            Ts,
            nan=self.T_amb_K,
            posinf=self.numeric.max_temperature_K,
            neginf=self.numeric.min_temperature_K,
        )
        Ts_safe = np.clip(Ts_safe, self.numeric.min_temperature_K, self.numeric.max_temperature_K)

        Tw_safe = np.nan_to_num(
            Tw,
            nan=self.T_amb_K,
            posinf=self.numeric.max_temperature_K,
            neginf=self.numeric.min_temperature_K,
        )
        Tw_safe = np.clip(Tw_safe, self.numeric.min_temperature_K, self.numeric.max_temperature_K)

        Tj_safe = np.nan_to_num(
            Tj,
            nan=self.T_amb_K,
            posinf=self.numeric.max_temperature_K,
            neginf=self.numeric.min_temperature_K,
        )
        Tj_safe = np.clip(Tj_safe, self.numeric.min_temperature_K, self.numeric.max_temperature_K)

        gas_mol = np.sum(C_safe * self.epsilon_gas_accum * self.node_volume, axis=0)
        solid_mol = np.zeros(len(COMPONENTS))
        solid_mol[IDX["CO2"]] = np.sum(q_safe[:, AIDX["CO2"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)
        solid_mol[IDX["H2O"]] = np.sum(q_safe[:, AIDX["H2O"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)

        cp_vol = Ctot * mixture_cp_molar_J_molK(ygas)
        U_gas = float(np.sum(self.epsilon_gas_accum * cp_vol * Tg_safe * self.node_volume))
        U_solid = float(np.sum(self.bed.rho_bulk_kg_m3_bed * self.bed.cp_solid_J_kgK * Ts_safe * self.node_volume))
        U_wall = float(np.sum(self.wall_mass_node * self.bed.wall_cp_J_kgK * Tw_safe))
        U_jacket = float(np.sum(self.jacket_mass_node * self.cycle.jacket_cp_J_kgK * Tj_safe))
        
        return {
            "gas_CO2_mol": gas_mol[IDX["CO2"]],
            "gas_H2O_mol": gas_mol[IDX["H2O"]],
            "gas_N2_mol": gas_mol[IDX["N2"]],
            "gas_O2_mol": gas_mol[IDX["O2"]],
            "solid_CO2_mol": solid_mol[IDX["CO2"]],
            "solid_H2O_mol": solid_mol[IDX["H2O"]],
            "solid_N2_mol": 0.0,
            "solid_O2_mol": 0.0,
            "U_gas_J": U_gas,
            "U_solid_J": U_solid,
            "U_wall_J": U_wall,
            "U_jacket_J": U_jacket,
            "U_total_proxy_J": U_gas + U_solid + U_wall + U_jacket,
        }

    def derived_rates(self, y: np.ndarray, step_name: str, t_step_s: float = 0.0) -> dict[str, np.ndarray | float]:
        C, q, Tg, Ts, Tw, Tj = self.unpack(y)

        C_safe, Tg_safe, Ctot, ygas = self.regularize_gas_state(C, Tg)
        q_safe = smooth_floor(
            np.nan_to_num(q, nan=self.numeric.min_loading_mol_kg, posinf=100.0, neginf=self.numeric.min_loading_mol_kg),
            self.numeric.min_loading_mol_kg,
            rel_width=self.numeric.smooth_floor_rel_width,
            abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9),
        )
        Ts_safe = smooth_clip(Ts, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
        Tw_safe = smooth_clip(Tw, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
        Tj_safe = smooth_clip(Tj, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))

        P = Ctot * R * Tg_safe

        u_faces = self.face_velocities(C_safe, Tg_safe, step_name, t_step_s)
        conv_flux = self.component_convective_fluxes_given_u(C_safe, u_faces, step_name)
        Dax_faces = self.axial_dispersion_face_coeffs(u_faces)

        feed_flux = conv_flux[0, :] if step_name in {"adsorption", "repressurization"} else np.zeros(len(COMPONENTS))
        vent_flux = conv_flux[-1, :] if step_name == "adsorption" else np.zeros(len(COMPONENTS))
        # Detailed-validation product definition: product is the outlet-face
        # convective flux through the product/vacuum boundary. Inventory loss is
        # used only as a mass-balance diagnostic, not as a product shortcut.
        product_flux = conv_flux[-1, :] if step_name in {"evacuation", "heating_desorption"} else np.zeros(len(COMPONENTS))

        ndot_feed = feed_flux * self.area
        ndot_vent = vent_flux * self.area
        ndot_product = product_flux * self.area

        ndot_product_total = float(np.sum(ndot_product))
        if ndot_product_total > 0:
            yprod = ndot_product / ndot_product_total
            Tprod = float(Tg_safe[-1])
            Pprod = float(max(self.cycle.P_vac_Pa, min(P[-1], self.P_amb_Pa)))
        else:
            yprod = np.zeros(len(COMPONENTS))
            Tprod = float(Tg_safe[-1])
            Pprod = float(P[-1])

        # Ergun pressure drop for reporting and fan.
        mu = float(np.mean(gas_viscosity_air_sutherland(Tg_safe)))
        mwmix = mixture_mw_kg_mol(ygas)
        rho = P * mwmix / (R * Tg_safe)
        rho_avg = float(np.mean(rho))
        eps = self.bed.epsilon_b
        dPdz = (
            150.0 * mu * (1.0 - eps) ** 2 / (self.d_p ** 2 * eps ** 3) * self.u_ads_m_s
            + 1.75 * rho_avg * (1.0 - eps) / (self.d_p * eps ** 3) * abs(self.u_ads_m_s) * self.u_ads_m_s
        )
        dP_bed = max(dPdz * self.bed.bed_length_m, 0.0)

        # Jacket utility duty and bed-side heat transfer are reported separately.
        mdot_j = self.jacket_mdot(step_name)
        Tj_in = self.jacket_inlet_temperature(step_name, t_step_s)
        Tj_out = float(Tj_safe[-1])

        if step_name == "closed_cooling" and self.numeric.robust_cooling_jacket_boundary:
            Tj_for_wall = np.full(self.N, Tj_in)
        else:
            Tj_for_wall = Tj_safe

        q_wj_W_node = self.bed.U_hx_W_m2K * self.wall_area_node * (Tj_for_wall - Tw_safe)
        Q_heat_to_bed_W = float(np.sum(np.maximum(q_wj_W_node, 0.0)))
        Q_cool_from_bed_W = float(np.sum(np.maximum(-q_wj_W_node, 0.0)))

        Q_heat_utility_W = 0.0
        Q_cool_utility_W = 0.0
        if step_name == "heating_desorption" and mdot_j > 0:
            Q_heat_utility_W = max(mdot_j * self.cycle.jacket_cp_J_kgK * (Tj_in - Tj_out), 0.0)
        if step_name == "closed_cooling" and mdot_j > 0:
            Q_cool_utility_W = max(mdot_j * self.cycle.jacket_cp_J_kgK * (Tj_out - Tj_in), 0.0)

        Q_heat_W = Q_heat_to_bed_W if step_name == "heating_desorption" else 0.0
        Q_cool_W = Q_cool_from_bed_W if step_name == "closed_cooling" else 0.0

        W_fan = 0.0
        if step_name == "adsorption":
            W_fan = self.Vdot_air_m3_s * dP_bed / max(self.cycle.eta_fan, 1e-9)

        W_vac = 0.0
        if step_name in {"evacuation", "heating_desorption"} and ndot_product_total > 0:
            W_vac = (
                ndot_product_total
                * R
                * Tprod
                / max(self.cycle.eta_vacuum, 1e-9)
                * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
            )

        W_repress = 0.0
        if step_name == "repressurization":
            ndot_feed_total = float(np.sum(ndot_feed))
            if ndot_feed_total > 0:
                Vdot_rep = ndot_feed_total * R * self.T_amb_K / self.P_amb_Pa
                W_repress = Vdot_rep * dP_bed / max(self.cycle.eta_fan, 1e-9)

        qeq_co2, qeq_h2o, _, _, _, RH_raw, RH_gas_raw = equilibrium_loadings_from_gas_and_solid(C_safe, Tg_safe, Ts_safe, self.ads)

        # Diagnostic heat of adsorption/desorption based on the current LDF rate.
        # Positive Q_ads_W means heat released to the bed; negative means heat
        # consumed by desorption.
        dqdt_diag = np.zeros_like(q_safe)
        dqdt_diag[:, AIDX["CO2"]] = self.ads.k_ldf_co2_s * (qeq_co2 - q_safe[:, AIDX["CO2"]])
        dqdt_diag[:, AIDX["H2O"]] = self.ads.k_ldf_h2o_s * (qeq_h2o - q_safe[:, AIDX["H2O"]])
        if step_name == "closed_cooling" and self.numeric.freeze_mass_during_closed_cooling:
            dqdt_diag[:, :] = 0.0
        Q_ads_W = float(np.sum((
            -self.ads.dH_ads_CO2_J_mol * self.bed.rho_bulk_kg_m3_bed * dqdt_diag[:, AIDX["CO2"]]
            -self.ads.dH_ads_H2O_J_mol * self.bed.rho_bulk_kg_m3_bed * dqdt_diag[:, AIDX["H2O"]]
        ) * self.node_volume))

        cp_vec = np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float)
        H_feed_W = float(np.sum(ndot_feed * cp_vec * self.T_amb_K))
        H_vent_W = float(np.sum(ndot_vent * cp_vec * Tg_safe[-1]))
        H_product_W = float(np.sum(ndot_product * cp_vec * Tprod))

        inv = self.inventory(C_safe, q_safe, Tg_safe, Ts_safe, Tw_safe, Tj_safe)

        # Diagnostics.
        clip_conc = int(np.sum(C < self.numeric.min_concentration_mol_m3))
        clip_q = int(np.sum(q < self.numeric.min_loading_mol_kg))
        clip_T_low = int(np.sum(np.r_[Tg, Ts, Tw, Tj] < self.numeric.min_temperature_K))
        clip_T_high = int(np.sum(np.r_[Tg, Ts, Tw, Tj] > self.numeric.max_temperature_K))
        n_supersat = int(np.sum(RH_raw > 1.0))
        supersat_excess = float(np.sum(np.maximum(RH_raw - 1.0, 0.0)))

        return {
            "P_node_Pa": P,
            "Tg_node_K": Tg_safe,
            "Ts_node_K": Ts_safe,
            "Tw_node_K": Tw_safe,
            "Tj_node_K": Tj_safe,
            "q_CO2_node_mol_kg": q_safe[:, AIDX["CO2"]],
            "q_H2O_node_mol_kg": q_safe[:, AIDX["H2O"]],
            "qeq_CO2_node_mol_kg": qeq_co2,
            "qeq_H2O_node_mol_kg": qeq_h2o,
            "RH_raw_node": RH_raw,
            "u_faces_m_s": u_faces,
            "Pavg_Pa": float(np.mean(P)),
            "Pmin_Pa": float(np.min(P)),
            "Pmax_Pa": float(np.max(P)),
            "Tg_avg_K": float(np.mean(Tg_safe)),
            "Ts_avg_K": float(np.mean(Ts_safe)),
            "Tw_avg_K": float(np.mean(Tw_safe)),
            "Tj_avg_K": float(np.mean(Tj_safe)),
            "q_CO2_avg_mol_kg": float(np.mean(q_safe[:, AIDX["CO2"]])),
            "q_H2O_avg_mol_kg": float(np.mean(q_safe[:, AIDX["H2O"]])),
            "max_RH_raw": float(np.max(RH_raw)),
            "n_supersat_nodes": n_supersat,
            "supersat_excess_sum": supersat_excess,
            "clip_concentration_count": clip_conc,
            "clip_loading_count": clip_q,
            "clip_temperature_low_count": clip_T_low,
            "clip_temperature_high_count": clip_T_high,
            "dP_bed_Pa": dP_bed,
            "feed_CO2_mol_s": ndot_feed[IDX["CO2"]],
            "feed_H2O_mol_s": ndot_feed[IDX["H2O"]],
            "feed_N2_mol_s": ndot_feed[IDX["N2"]],
            "feed_O2_mol_s": ndot_feed[IDX["O2"]],
            "vent_CO2_mol_s": ndot_vent[IDX["CO2"]],
            "vent_H2O_mol_s": ndot_vent[IDX["H2O"]],
            "vent_N2_mol_s": ndot_vent[IDX["N2"]],
            "vent_O2_mol_s": ndot_vent[IDX["O2"]],
            "product_CO2_mol_s": ndot_product[IDX["CO2"]],
            "product_H2O_mol_s": ndot_product[IDX["H2O"]],
            "product_N2_mol_s": ndot_product[IDX["N2"]],
            "product_O2_mol_s": ndot_product[IDX["O2"]],
            "ndot_product_mol_s": ndot_product_total,
            "T_product_K": Tprod,
            "P_product_Pa": Pprod,
            "y_CO2_product": yprod[IDX["CO2"]],
            "y_H2O_product": yprod[IDX["H2O"]],
            "y_N2_product": yprod[IDX["N2"]],
            "y_O2_product": yprod[IDX["O2"]],
            "Q_heat_W": Q_heat_W,
            "Q_cool_W": Q_cool_W,
            "Q_heat_utility_W": Q_heat_utility_W,
            "Q_cool_utility_W": Q_cool_utility_W,
            "Q_heat_to_bed_W": Q_heat_to_bed_W,
            "Q_cool_from_bed_W": Q_cool_from_bed_W,
            "W_fan_W": W_fan,
            "W_vac_W": W_vac,
            "W_repress_W": W_repress,
            "H_feed_W": H_feed_W,
            "H_vent_W": H_vent_W,
            "H_product_W": H_product_W,
            "Q_ads_W": Q_ads_W,
            "Ts_min_K": float(np.min(Ts_safe)),
            "Ts_max_K": float(np.max(Ts_safe)),
            "q_CO2_min_mol_kg": float(np.min(q_safe[:, AIDX["CO2"]])),
            "q_CO2_max_mol_kg": float(np.max(q_safe[:, AIDX["CO2"]])),
            "q_H2O_min_mol_kg": float(np.min(q_safe[:, AIDX["H2O"]])),
            "q_H2O_max_mol_kg": float(np.max(q_safe[:, AIDX["H2O"]])),
            "qeq_CO2_avg_mol_kg": float(np.mean(qeq_co2)),
            "qeq_CO2_min_mol_kg": float(np.min(qeq_co2)),
            "qeq_CO2_max_mol_kg": float(np.max(qeq_co2)),
            "qeq_H2O_avg_mol_kg": float(np.mean(qeq_h2o)),
            "RH_gas_raw_max": float(np.max(RH_gas_raw)),
            "Dax_CO2_mean_m2_s": float(np.mean(Dax_faces[:, IDX["CO2"]])),
            "Dax_H2O_mean_m2_s": float(np.mean(Dax_faces[:, IDX["H2O"]])),
            "Dax_N2_mean_m2_s": float(np.mean(Dax_faces[:, IDX["N2"]])),
            "Dax_O2_mean_m2_s": float(np.mean(Dax_faces[:, IDX["O2"]])),
            "Pe_CO2_effective": float(abs(self.u_ads_m_s) * self.bed.bed_length_m / max(float(np.mean(Dax_faces[:, IDX["CO2"]])), 1e-30)),
            "Pe_H2O_effective": float(abs(self.u_ads_m_s) * self.bed.bed_length_m / max(float(np.mean(Dax_faces[:, IDX["H2O"]])), 1e-30)),
            **inv,
        }

    def fast_pressure_step(
        self,
        y0_dim: np.ndarray,
        step_name: str,
        duration_s: float,
        t_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], int]:
        C0, q0, Tg0, Ts0, Tw0, Tj0 = self.unpack(y0_dim)
        C0_safe, Tg_safe, Ctot0, ygas0 = self.regularize_gas_state(C0, Tg0)
        Ts_safe = smooth_clip(
            Ts0,
            self.numeric.min_temperature_K,
            self.numeric.max_temperature_K,
            rel_width=0.0,
            abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9),
        )
        q_prev = smooth_floor(
            np.nan_to_num(q0, nan=0.0, posinf=100.0, neginf=0.0),
            self.numeric.min_loading_mol_kg,
            rel_width=self.numeric.smooth_floor_rel_width,
            abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9),
        )

        if step_name == "evacuation":
            Ctot_target = self.cycle.P_vac_Pa / (R * Tg_safe)
            Ctarget = ygas0 * Ctot_target[:, None]
        elif step_name == "repressurization":
            Ctot_target = self.P_amb_Pa / (R * Tg_safe)
            Ctarget = self.y_feed[None, :] * Ctot_target[:, None]
        else:
            raise ValueError(f"fast_pressure_step called for invalid step: {step_name}")

        duration = max(float(duration_s), 1e-12)
        dt_internal = max(float(self.numeric.pressure_step_max_step_s), 1e-9)

        internal = np.arange(0.0, duration + dt_internal, dt_internal)
        internal = internal[internal <= duration]
        if len(internal) == 0 or internal[0] != 0.0:
            internal = np.insert(internal, 0, 0.0)
        if internal[-1] < duration:
            internal = np.append(internal, duration)

        # Make sure all reporting times are also internal update times.
        t_grid = np.unique(np.concatenate([internal, np.asarray(t_eval, dtype=float), np.array([0.0, duration])]))
        t_grid = t_grid[(t_grid >= -1e-12) & (t_grid <= duration + 1e-12)]
        t_grid[0] = 0.0
        t_grid[-1] = duration

        def pressure_path_C(tloc: float) -> np.ndarray:
            sfrac = float(np.clip(float(tloc) / duration, 0.0, 1.0))
            alpha = self.smoothstep01(sfrac)
            return C0_safe + alpha * (Ctarget - C0_safe)

        def total_inventory_vec(C: np.ndarray, q: np.ndarray) -> np.ndarray:
            gas = np.sum(C * self.bed.epsilon_b * self.node_volume, axis=0)
            solid = np.zeros(len(COMPONENTS))
            solid[IDX["CO2"]] = np.sum(q[:, AIDX["CO2"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)
            solid[IDX["H2O"]] = np.sum(q[:, AIDX["H2O"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)
            return gas + solid

        ymat = np.zeros((len(y0_dim), len(t_eval)), dtype=float)
        overrides: list[dict[str, float]] = []
        t_eval_list = [float(x) for x in t_eval]
        eval_pos = {round(float(t), 10): i for i, t in enumerate(t_eval_list)}

        C_prev = pressure_path_C(0.0)
        inv_prev = total_inventory_vec(C_prev, q_prev)

        zero_rates = {f"feed_{c}_mol_s": 0.0 for c in COMPONENTS}
        zero_rates.update({f"vent_{c}_mol_s": 0.0 for c in COMPONENTS})
        zero_rates.update({f"product_{c}_mol_s": 0.0 for c in COMPONENTS})
        zero_rates.update({
            "ndot_product_mol_s": 0.0,
            "y_CO2_product": 0.0,
            "y_H2O_product": 0.0,
            "y_N2_product": 0.0,
            "y_O2_product": 0.0,
            "W_vac_W": 0.0,
            "W_repress_W": 0.0,
            "H_feed_W": 0.0,
            "H_vent_W": 0.0,
            "H_product_W": 0.0,
            "Q_ads_W": 0.0,
        })
        for _s in ("feed", "vent", "product"):
            for _c in COMPONENTS:
                zero_rates[f"{_s}_{_c}_mol_interval"] = 0.0

        # Record t=0.
        if round(0.0, 10) in eval_pos:
            k0 = eval_pos[round(0.0, 10)]
            ymat[:, k0] = self.pack(C_prev, q_prev, Tg_safe, Ts0, Tw0, Tj0)

        last_rates = zero_rates.copy()

        for j in range(1, len(t_grid)):
            t_old = float(t_grid[j - 1])
            t_new = float(t_grid[j])
            dt = max(t_new - t_old, 1e-12)

            # LDF loading update over this pressure substep using the previous
            # gas state.
            qeq_co2, qeq_h2o, *_ = equilibrium_loadings(C_prev, Ts_safe, self.ads)
            q_new = q_prev.copy()
            q_new[:, AIDX["CO2"]] = qeq_co2 + (q_prev[:, AIDX["CO2"]] - qeq_co2) * math.exp(-self.ads.k_ldf_co2_s * dt)
            q_new[:, AIDX["H2O"]] = qeq_h2o + (q_prev[:, AIDX["H2O"]] - qeq_h2o) * math.exp(-self.ads.k_ldf_h2o_s * dt)
            q_new = smooth_floor(
                q_new,
                self.numeric.min_loading_mol_kg,
                rel_width=self.numeric.smooth_floor_rel_width,
                abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9),
            )

            C_new = pressure_path_C(t_new)
            inv_new = total_inventory_vec(C_new, q_new)
            delta_total = inv_new - inv_prev

            rates = zero_rates.copy()
            if step_name == "evacuation":
                out_rate = np.maximum(-delta_total / dt, 0.0)
                for comp in COMPONENTS:
                    rates[f"product_{comp}_mol_s"] = float(out_rate[IDX[comp]])
                    rates[f"product_{comp}_mol_interval"] = float(out_rate[IDX[comp]] * dt)
                ndot = float(np.sum(out_rate))
                rates["ndot_product_mol_s"] = ndot
                rates["H_product_W"] = float(np.sum(out_rate * np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float) * float(np.mean(Tg_safe))))
                if ndot > 0.0:
                    yprod = out_rate / ndot
                    rates["y_CO2_product"] = float(yprod[IDX["CO2"]])
                    rates["y_H2O_product"] = float(yprod[IDX["H2O"]])
                    rates["y_N2_product"] = float(yprod[IDX["N2"]])
                    rates["y_O2_product"] = float(yprod[IDX["O2"]])
                    Tprod = float(np.mean(Tg_safe))
                    rates["W_vac_W"] = (
                        ndot * R * Tprod
                        / max(self.cycle.eta_vacuum, 1e-9)
                        * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
                    )

            elif step_name == "repressurization":
                in_rate = np.maximum(delta_total / dt, 0.0)
                for comp in COMPONENTS:
                    rates[f"feed_{comp}_mol_s"] = float(in_rate[IDX[comp]])
                    rates[f"feed_{comp}_mol_interval"] = float(in_rate[IDX[comp]] * dt)
                ndot_in = float(np.sum(in_rate))
                rates["H_feed_W"] = float(np.sum(in_rate * np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float) * self.T_amb_K))
                if ndot_in > 0.0:
                    rates["W_repress_W"] = (
                        ndot_in * R * self.T_amb_K
                        / max(self.cycle.eta_fan, 1e-9)
                        * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
                    )

            key = round(t_new, 10)
            if key in eval_pos:
                kk = eval_pos[key]
                ymat[:, kk] = self.pack(C_new, q_new, Tg_safe, Ts0, Tw0, Tj0)
                last_rates = rates.copy()
                # Store temporarily in a dictionary keyed by report index.
                pass

            # Attach rates to exact reporting points reached in this interval.
            for t_report in t_eval_list:
                if t_old < t_report <= t_new + 1e-12:
                    # Ensure state is available even for non-grid roundoff cases.
                    kk = eval_pos[round(float(t_report), 10)]
                    if not np.any(ymat[:, kk]):
                        C_rep = pressure_path_C(float(t_report))
                        ymat[:, kk] = self.pack(C_rep, q_new, Tg_safe, Ts0, Tw0, Tj0)

            C_prev = C_new
            q_prev = q_new
            inv_prev = inv_new

        # Build overrides in t_eval order by finite-difference inventory over
        # reporting intervals.
        prev_C = None
        prev_q = None
        prev_t = None
        prev_inv = None
        for k, tloc in enumerate(t_eval_list):
            if k == 0:
                overrides.append(zero_rates.copy())
                Ck, qk, *_ = self.unpack(ymat[:, k])
                prev_C, prev_q, prev_t = Ck, qk, tloc
                prev_inv = total_inventory_vec(Ck, qk)
                continue

            Ck, qk, *_ = self.unpack(ymat[:, k])
            inv_k = total_inventory_vec(Ck, qk)
            dt = max(tloc - float(prev_t), 1e-12)
            delta_total = inv_k - prev_inv

            rates = zero_rates.copy()
            if step_name == "evacuation":
                out_rate = np.maximum(-delta_total / dt, 0.0)
                for comp in COMPONENTS:
                    rates[f"product_{comp}_mol_s"] = float(out_rate[IDX[comp]])
                    rates[f"product_{comp}_mol_interval"] = float(out_rate[IDX[comp]] * dt)
                ndot = float(np.sum(out_rate))
                rates["ndot_product_mol_s"] = ndot
                rates["H_product_W"] = float(np.sum(out_rate * np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float) * float(np.mean(Tg_safe))))
                if ndot > 0.0:
                    yprod = out_rate / ndot
                    rates["y_CO2_product"] = float(yprod[IDX["CO2"]])
                    rates["y_H2O_product"] = float(yprod[IDX["H2O"]])
                    rates["y_N2_product"] = float(yprod[IDX["N2"]])
                    rates["y_O2_product"] = float(yprod[IDX["O2"]])
                    rates["W_vac_W"] = (
                        ndot * R * float(np.mean(Tg_safe))
                        / max(self.cycle.eta_vacuum, 1e-9)
                        * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
                    )
            elif step_name == "repressurization":
                in_rate = np.maximum(delta_total / dt, 0.0)
                for comp in COMPONENTS:
                    rates[f"feed_{comp}_mol_s"] = float(in_rate[IDX[comp]])
                    rates[f"feed_{comp}_mol_interval"] = float(in_rate[IDX[comp]] * dt)
                ndot_in = float(np.sum(in_rate))
                rates["H_feed_W"] = float(np.sum(in_rate * np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float) * self.T_amb_K))
                if ndot_in > 0.0:
                    rates["W_repress_W"] = (
                        ndot_in * R * self.T_amb_K
                        / max(self.cycle.eta_fan, 1e-9)
                        * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
                    )

            overrides.append(rates)
            prev_inv = inv_k
            prev_t = tloc

        n_internal_steps = max(1, len(t_grid) - 1)
        return t_eval, ymat, overrides, n_internal_steps

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------


    def fast_heating_desorption_step(
        self,
        y0_dim: np.ndarray,
        duration_s: float,
        t_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], int]:
        """
        It keeps the main CO2/H2O physics active:
        - WADST/GAB equilibrium,
        - LDF desorption for CO2 and H2O,
        - heat of desorption,
        - wall/jacket heat transfer,

        Simplifications:
        - N2/O2 are frozen during heating by default;
        - gas pressure is kept near the vacuum boundary;
        - gas temperature is relaxed toward solid temperature in near-vacuum.
        """
        C0, q0, Tg0, Ts0, Tw0, Tj0 = self.unpack(y0_dim)
        C_safe, Tg_safe, Ctot, ygas = self.regularize_gas_state(C0, Tg0)
        q_prev = smooth_floor(
            np.nan_to_num(q0, nan=0.0, posinf=100.0, neginf=0.0),
            self.numeric.min_loading_mol_kg,
            rel_width=self.numeric.smooth_floor_rel_width,
            abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9),
        )
        Ts_prev = smooth_clip(Ts0, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
        Tw_prev = smooth_clip(Tw0, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
        Tj_prev = smooth_clip(Tj0, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))

        duration = max(float(duration_s), 1e-12)
        dt_internal = max(float(self.numeric.fast_heating_step_s), 1e-9)
        internal = np.arange(0.0, duration + dt_internal, dt_internal)
        internal = internal[internal <= duration]
        if len(internal) == 0 or internal[0] != 0.0:
            internal = np.insert(internal, 0, 0.0)
        if internal[-1] < duration:
            internal = np.append(internal, duration)

        t_grid = np.unique(np.concatenate([internal, np.asarray(t_eval, dtype=float), np.array([0.0, duration])]))
        t_grid = t_grid[(t_grid >= -1e-12) & (t_grid <= duration + 1e-12)]
        t_grid[0] = 0.0
        t_grid[-1] = duration

        ymat = np.zeros((len(y0_dim), len(t_eval)), dtype=float)
        t_eval_list = [float(x) for x in t_eval]
        eval_pos = {round(float(t), 10): i for i, t in enumerate(t_eval_list)}

        def gas_from_product_generation(C_old: np.ndarray, prod_mol: np.ndarray, Tg_ref: np.ndarray) -> np.ndarray:
            # Keep gas near vacuum; let CO2/H2O reflect generated product ratio.
            Ctot_vac = self.cycle.P_vac_Pa / (R * np.maximum(Tg_ref, self.numeric.min_temperature_K))
            y = np.zeros(len(COMPONENTS), dtype=float)
            total = float(np.sum(prod_mol))
            if total > 0.0:
                y[:] = prod_mol / total
            else:
                # fallback to previous CO2/H2O only, with inert optionally frozen
                Csum = np.sum(C_old, axis=1)
                yprev = np.mean(C_old / np.maximum(Csum[:, None], 1e-30), axis=0)
                y[:] = yprev
            if self.numeric.freeze_inert_during_heating_desorption:
                y[IDX["N2"]] = 0.0
                y[IDX["O2"]] = 0.0
            ysum = float(np.sum(y))
            if ysum <= 0.0:
                y[IDX["CO2"]] = 0.5
                y[IDX["H2O"]] = 0.5
                ysum = 1.0
            y = y / ysum
            return y[None, :] * Ctot_vac[:, None]

        def total_inventory_vec(C: np.ndarray, q: np.ndarray) -> np.ndarray:
            gas = np.sum(C * self.bed.epsilon_b * self.node_volume, axis=0)
            solid = np.zeros(len(COMPONENTS))
            solid[IDX["CO2"]] = np.sum(q[:, AIDX["CO2"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)
            solid[IDX["H2O"]] = np.sum(q[:, AIDX["H2O"]] * self.bed.rho_bulk_kg_m3_bed * self.node_volume)
            return gas + solid

        zero_rates = {f"feed_{c}_mol_s": 0.0 for c in COMPONENTS}
        zero_rates.update({f"vent_{c}_mol_s": 0.0 for c in COMPONENTS})
        zero_rates.update({f"product_{c}_mol_s": 0.0 for c in COMPONENTS})
        zero_rates.update({
            "ndot_product_mol_s": 0.0,
            "y_CO2_product": 0.0,
            "y_H2O_product": 0.0,
            "y_N2_product": 0.0,
            "y_O2_product": 0.0,
            "W_vac_W": 0.0,
            "W_repress_W": 0.0,
            "Q_heat_W": 0.0,
            "Q_heat_utility_W": 0.0,
            "H_feed_W": 0.0,
            "H_vent_W": 0.0,
            "H_product_W": 0.0,
            "Q_ads_W": 0.0,
        })
        for _s in ("feed", "vent", "product"):
            for _c in COMPONENTS:
                zero_rates[f"{_s}_{_c}_mol_interval"] = 0.0

        # Start from near-vacuum gas at the current CO2/H2O composition.
        C_prev = C_safe.copy()
        if self.numeric.freeze_inert_during_heating_desorption:
            C_prev[:, IDX["N2"]] = 0.0
            C_prev[:, IDX["O2"]] = 0.0
            Csum = np.sum(C_prev, axis=1)
            bad = Csum <= 0.0
            if np.any(bad):
                C_prev[bad, IDX["CO2"]] = 0.5
                C_prev[bad, IDX["H2O"]] = 0.5
                Csum = np.sum(C_prev, axis=1)
            C_prev = C_prev / np.maximum(Csum[:, None], 1e-30)
            C_prev = C_prev * (self.cycle.P_vac_Pa / (R * np.maximum(Ts_prev, self.numeric.min_temperature_K)))[:, None]

        if round(0.0, 10) in eval_pos:
            ymat[:, eval_pos[round(0.0, 10)]] = self.pack(C_prev, q_prev, Ts_prev, Ts_prev, Tw_prev, Tj_prev)

        inv_start_heating = total_inventory_vec(C_prev, q_prev)
        product_cum_prev = np.zeros(len(COMPONENTS), dtype=float)
        prev_inv_report = inv_start_heating.copy()
        report_states = {0.0: (C_prev.copy(), q_prev.copy(), Ts_prev.copy(), Ts_prev.copy(), Tw_prev.copy(), Tj_prev.copy(), zero_rates.copy())}

        for j in range(1, len(t_grid)):
            t_old = float(t_grid[j - 1])
            t_new = float(t_grid[j])
            dt = max(t_new - t_old, 1e-12)

            qeq_co2, qeq_h2o, *_ = equilibrium_loadings(C_prev, Ts_prev, self.ads)
            q_old = q_prev.copy()
            q_new = q_prev.copy()
            exp_co2 = math.exp(-self.ads.k_ldf_co2_s * dt)
            exp_h2o = math.exp(-self.ads.k_ldf_h2o_s * dt)
            q_new[:, AIDX["CO2"]] = qeq_co2 + (q_prev[:, AIDX["CO2"]] - qeq_co2) * exp_co2
            q_new[:, AIDX["H2O"]] = qeq_h2o + (q_prev[:, AIDX["H2O"]] - qeq_h2o) * exp_h2o
            q_new = smooth_floor(q_new, self.numeric.min_loading_mol_kg, rel_width=self.numeric.smooth_floor_rel_width, abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9))

            dqdt = (q_new - q_old) / dt

            # Preliminary desorbed amount from adsorbed-phase swing. This is used
            # only to set the near-vacuum gas composition.
            ads_loss_guess = np.zeros(len(COMPONENTS), dtype=float)
            ads_loss_guess[IDX["CO2"]] = max(float(np.sum((q_old[:, AIDX["CO2"]] - q_new[:, AIDX["CO2"]]) * self.bed.rho_bulk_kg_m3_bed * self.node_volume)), 0.0)
            ads_loss_guess[IDX["H2O"]] = max(float(np.sum((q_old[:, AIDX["H2O"]] - q_new[:, AIDX["H2O"]]) * self.bed.rho_bulk_kg_m3_bed * self.node_volume)), 0.0)

            inv_before = total_inventory_vec(C_prev, q_old)

            # Keep the gas holdup fixed during the fast heating split.
            C_new = C_prev.copy()
            inv_after_prethermal = total_inventory_vec(C_new, q_new)

            # Conservative cumulative product accounting: during heating there
            # is no feed and no vent. 
            product_cum_target = np.maximum(inv_start_heating - inv_after_prethermal, 0.0)
            if self.numeric.freeze_inert_during_heating_desorption:
                product_cum_target[IDX["N2"]] = 0.0
                product_cum_target[IDX["O2"]] = 0.0
            prod_mol = np.maximum(product_cum_target - product_cum_prev, 0.0)
            product_cum_prev = product_cum_prev + prod_mol

            # Cumulative mass-balance diagnostic.
            substep_residual = inv_start_heating - inv_after_prethermal - product_cum_prev
            self.fast_substep_diagnostics.append({
                "step": "heating_desorption",
                "t_old_s": t_old,
                "t_new_s": t_new,
                "dt_s": dt,
                "CO2_inventory_before_mol": float(inv_before[IDX["CO2"]]),
                "CO2_inventory_after_mol": float(inv_after_prethermal[IDX["CO2"]]),
                "CO2_product_mol_substep": float(prod_mol[IDX["CO2"]]),
                "CO2_residual_mol_substep": float(substep_residual[IDX["CO2"]]),
                "H2O_inventory_before_mol": float(inv_before[IDX["H2O"]]),
                "H2O_inventory_after_mol": float(inv_after_prethermal[IDX["H2O"]]),
                "H2O_product_mol_substep": float(prod_mol[IDX["H2O"]]),
                "H2O_residual_mol_substep": float(substep_residual[IDX["H2O"]]),
            })

            # Thermal update: solid/wall/jacket only; Tg follows Ts near vacuum.
            Tj_in = self.jacket_inlet_temperature("heating_desorption", t_new)
            mdot_j = self.jacket_mdot("heating_desorption")
            cp_j = self.cycle.jacket_cp_J_kgK

            Tin_vec = np.empty(self.N, dtype=float)
            Tin_vec[0] = Tj_in
            if self.N > 1:
                Tin_vec[1:] = Tj_prev[:-1]

            # keep same steady-state jacket behavior as ODE but explicit/stable.
            tau_j = max(self.numeric.jacket_tau_min_s, 1e-9)
            Tj_new = Tj_prev + dt * (Tin_vec - Tj_prev) / tau_j

            q_sw_W_node = self.bed.h_solid_wall_W_m2K * self.wall_area_node * (Tw_prev - Ts_prev)
            q_wj_W_node = self.bed.U_hx_W_m2K * self.wall_area_node * (Tj_new - Tw_prev)

            solid_heat_capacity = self.bed.rho_bulk_kg_m3_bed * self.bed.cp_solid_J_kgK
            cp_ads_eff = self.bed.rho_bulk_kg_m3_bed * (
                q_new[:, AIDX["CO2"]] * self.ads.cp_ads_CO2_J_molK
                + q_new[:, AIDX["H2O"]] * self.ads.cp_ads_H2O_J_molK
            )
            solid_cap_eff = np.maximum(solid_heat_capacity + cp_ads_eff, 1e-9)

            heat_ads_W_m3 = (
                -self.ads.dH_ads_CO2_J_mol * self.bed.rho_bulk_kg_m3_bed * dqdt[:, AIDX["CO2"]]
                -self.ads.dH_ads_H2O_J_mol * self.bed.rho_bulk_kg_m3_bed * dqdt[:, AIDX["H2O"]]
            )
            dTsdt = q_sw_W_node / (solid_cap_eff * self.node_volume) + heat_ads_W_m3 / solid_cap_eff
            Ts_new = Ts_prev + dt * dTsdt
            Ts_new = np.clip(Ts_new, self.numeric.min_temperature_K, self.numeric.max_temperature_K)

            wall_cap_node = max(self.wall_mass_node * self.bed.wall_cp_J_kgK, 1e-12)
            dTwdt = (-q_sw_W_node + q_wj_W_node) / wall_cap_node
            Tw_new = Tw_prev + dt * dTwdt
            Tw_new = np.clip(Tw_new, self.numeric.min_temperature_K, self.numeric.max_temperature_K)

            rates = zero_rates.copy()
            out_rate = prod_mol / dt
            for comp in COMPONENTS:
                rates[f"product_{comp}_mol_s"] = float(out_rate[IDX[comp]])
                rates[f"product_{comp}_mol_interval"] = float(prod_mol[IDX[comp]])
            ndot = float(np.sum(out_rate))
            rates["ndot_product_mol_s"] = ndot
            cp_vec = np.array([CP_GAS_MOLAR[c] for c in COMPONENTS], dtype=float)
            rates["H_product_W"] = float(np.sum(out_rate * cp_vec * float(np.mean(Ts_prev))))
            if ndot > 0.0:
                yprod = out_rate / ndot
                rates["y_CO2_product"] = float(yprod[IDX["CO2"]])
                rates["y_H2O_product"] = float(yprod[IDX["H2O"]])
                rates["y_N2_product"] = float(yprod[IDX["N2"]])
                rates["y_O2_product"] = float(yprod[IDX["O2"]])
                rates["W_vac_W"] = (
                    ndot * R * float(np.mean(Ts_prev))
                    / max(self.cycle.eta_vacuum, 1e-9)
                    * math.log(max(self.P_amb_Pa / self.cycle.P_vac_Pa, 1.0))
                )

            rates["Q_ads_W"] = float(np.sum(heat_ads_W_m3 * self.node_volume))
            rates["Q_heat_W"] = float(np.sum(np.maximum(q_wj_W_node, 0.0)))
            rates["Q_heat_utility_W"] = max(mdot_j * cp_j * (Tj_in - float(Tj_new[-1])), 0.0)

            for t_report in t_eval_list:
                if t_old < t_report <= t_new + 1e-12:
                    frac = (float(t_report) - t_old) / dt
                    frac = float(np.clip(frac, 0.0, 1.0))
                    C_rep = C_prev + frac * (C_new - C_prev)
                    q_rep = q_prev + frac * (q_new - q_prev)
                    Ts_rep = Ts_prev + frac * (Ts_new - Ts_prev)
                    Tw_rep = Tw_prev + frac * (Tw_new - Tw_prev)
                    Tj_rep = Tj_prev + frac * (Tj_new - Tj_prev)
                    kk = eval_pos[round(float(t_report), 10)]
                    ymat[:, kk] = self.pack(C_rep, q_rep, Ts_rep, Ts_rep, Tw_rep, Tj_rep)
                    report_states[float(t_report)] = (C_rep, q_rep, Ts_rep, Ts_rep, Tw_rep, Tj_rep, rates.copy())

            C_prev, q_prev, Ts_prev, Tw_prev, Tj_prev = C_new, q_new, Ts_new, Tw_new, Tj_new

        overrides = []
        for k, tloc in enumerate(t_eval_list):
            entry = report_states.get(float(tloc), None)
            if entry is None:
                Ck, qk, Tgk, Tsk, Twk, Tjk = self.unpack(ymat[:, k])
                rates = zero_rates.copy()
            else:
                *_, rates = entry
            overrides.append(rates)

        n_internal_steps = max(1, len(t_grid) - 1)
        return t_eval, ymat, overrides, n_internal_steps

    def simulate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        y0 = self.initial_state()
        all_records: list[dict] = []
        node_records_all: list[dict] = []
        node_records_last: list[dict] = []
        cycle_summaries: list[dict] = []
        mass_balance_rows: list[dict] = []
        energy_balance_rows: list[dict] = []
        diagnostics_rows: list[dict] = []

        t_global_start = 0.0
        last_cycle = int(self.cycle.n_cycles)

        for cyc in range(1, self.cycle.n_cycles + 1):
            print(f"[CYCLE {cyc}/{self.cycle.n_cycles}] start", flush=True)
            cycle_start_time = t_global_start
            cycle_records_indices = []

            C0, q0, Tg0, Ts0, Tw0, Tj0 = self.unpack(y0)
            inv_start = self.inventory(
                np.maximum(C0, self.numeric.min_concentration_mol_m3),
                np.maximum(q0, 0.0),
                np.clip(Tg0, self.numeric.min_temperature_K, self.numeric.max_temperature_K),
                np.clip(Ts0, self.numeric.min_temperature_K, self.numeric.max_temperature_K),
                np.clip(Tw0, self.numeric.min_temperature_K, self.numeric.max_temperature_K),
                np.clip(Tj0, self.numeric.min_temperature_K, self.numeric.max_temperature_K),
            )

            for step in self.step_schedule():
                step_name = str(step["name"])
                duration = float(step["duration_s"])
                if duration <= 0:
                    continue

                print(f"  - Step: {step_name}, duration={duration:.1f} s", flush=True)

                t_eval = np.arange(0.0, duration + self.numeric.sample_dt_s, self.numeric.sample_dt_s)
                t_eval = t_eval[t_eval <= duration]
                if len(t_eval) == 0 or t_eval[-1] < duration:
                    t_eval = np.append(t_eval, duration)

                rate_overrides = None

                if (
                    self.numeric.use_fast_pressure_steps
                    and step_name in {"evacuation", "repressurization"}
                ):
                    sol_t, sol_y_dim, rate_overrides, n_internal_steps = self.fast_pressure_step(
                        y0, step_name, duration, t_eval
                    )
                    step_method = "FAST_PRESSURE_SUBSTEP"
                    step_max_step = self.numeric.pressure_step_max_step_s
                    sol = SimpleNamespace(
                        t=sol_t,
                        success=True,
                        message=(
                            "fast pressure substep model with LDF loading update; "
                            "nfev reports internal pressure substeps, not solve_ivp RHS calls"
                        ),
                        nfev=n_internal_steps,
                        njev=0,
                        nlu=0,
                    )
                elif (
                    self.numeric.use_fast_heating_desorption_steps
                    and step_name == "heating_desorption"
                ):
                    sol_t, sol_y_dim, rate_overrides, n_internal_steps = self.fast_heating_desorption_step(
                        y0, duration, t_eval
                    )
                    step_method = "FAST_HEATING_SPLIT"
                    step_max_step = self.numeric.fast_heating_step_s
                    sol = SimpleNamespace(
                        t=sol_t,
                        success=True,
                        message=(
                            "fast operator-split heating desorption; "
                            "nfev reports internal heating substeps, not solve_ivp RHS calls"
                        ),
                        nfev=n_internal_steps,
                        njev=0,
                        nlu=0,
                    )
                else:
                    y0_solver = self.to_solver_state(y0)
                    step_method = self.solver_method_for_step(step_name)
                    step_max_step = self.solver_max_step_for_step(step_name)
                    solve_kwargs = dict(
                        fun=lambda t, y, s=step_name: self.rhs_solver(t, y, s),
                        t_span=(0.0, duration),
                        y0=y0_solver,
                        method=step_method,
                        t_eval=t_eval,
                        max_step=step_max_step,
                        rtol=self.numeric.rtol,
                        atol=self.atol_solver,
                    )
                    if step_method in {"BDF", "Radau"} and self.jac_sparsity is not None:
                        solve_kwargs["jac_sparsity"] = self.jac_sparsity
                    sol = solve_ivp(**solve_kwargs)
                    sol_y_dim = self.from_solver_state_matrix(sol.y)

                print(
                    f"    done: success={sol.success}, nfev={getattr(sol, 'nfev', None)}, "
                    f"njev={getattr(sol, 'njev', None)}, nlu={getattr(sol, 'nlu', None)}",
                    flush=True,
                )

                self.solver_log.append({
                    "cycle": cyc,
                    "step": step_name,
                    "method": step_method,
                    "max_step_s": step_max_step,
                    "success": bool(sol.success),
                    "message": sol.message,
                    "nfev": getattr(sol, "nfev", np.nan),
                    "njev": getattr(sol, "njev", np.nan),
                    "nlu": getattr(sol, "nlu", np.nan),
                    "n_time_points": len(sol.t),
                })

                if not sol.success:
                    raise RuntimeError(f"ODE solver failed at cycle {cyc}, step {step_name}: {sol.message}")

                for k, t_local in enumerate(sol.t):
                    t_abs = t_global_start + float(t_local)
                    yk = sol_y_dim[:, k]
                    d = self.derived_rates(yk, step_name, float(t_local))
                    if rate_overrides is not None:
                        d.update(rate_overrides[k])

                    record = {
                        "cycle": cyc,
                        "step": step_name,
                        "t_global_s": t_abs,
                        "t_cycle_s": t_abs - cycle_start_time,
                        "t_step_s": float(t_local),
                        "Pavg_Pa": d["Pavg_Pa"],
                        "Pmin_Pa": d["Pmin_Pa"],
                        "Pmax_Pa": d["Pmax_Pa"],
                        "Tg_avg_K": d["Tg_avg_K"],
                        "Ts_avg_K": d["Ts_avg_K"],
                        "Tw_avg_K": d["Tw_avg_K"],
                        "Tj_avg_K": d["Tj_avg_K"],
                        "q_CO2_avg_mol_kg": d["q_CO2_avg_mol_kg"],
                        "q_H2O_avg_mol_kg": d["q_H2O_avg_mol_kg"],
                        "Ts_max_K": d.get("Ts_max_K", np.nan),
                        "q_CO2_min_mol_kg": d.get("q_CO2_min_mol_kg", np.nan),
                        "q_CO2_max_mol_kg": d.get("q_CO2_max_mol_kg", np.nan),
                        "qeq_CO2_avg_mol_kg": d.get("qeq_CO2_avg_mol_kg", np.nan),
                        "qeq_CO2_min_mol_kg": d.get("qeq_CO2_min_mol_kg", np.nan),
                        "qeq_CO2_max_mol_kg": d.get("qeq_CO2_max_mol_kg", np.nan),
                        "max_RH_raw": d["max_RH_raw"],
                        "max_RH_gas_raw": d.get("RH_gas_raw_max", np.nan),
                        "Dax_CO2_mean_m2_s": d.get("Dax_CO2_mean_m2_s", np.nan),
                        "Dax_H2O_mean_m2_s": d.get("Dax_H2O_mean_m2_s", np.nan),
                        "Dax_N2_mean_m2_s": d.get("Dax_N2_mean_m2_s", np.nan),
                        "Dax_O2_mean_m2_s": d.get("Dax_O2_mean_m2_s", np.nan),
                        "Pe_CO2_effective": d.get("Pe_CO2_effective", np.nan),
                        "Pe_H2O_effective": d.get("Pe_H2O_effective", np.nan),
                        "n_supersat_nodes": d["n_supersat_nodes"],
                        "supersat_excess_sum": d["supersat_excess_sum"],
                        "clip_concentration_count": d["clip_concentration_count"],
                        "clip_loading_count": d["clip_loading_count"],
                        "clip_temperature_low_count": d["clip_temperature_low_count"],
                        "clip_temperature_high_count": d["clip_temperature_high_count"],
                        "dP_bed_Pa": d["dP_bed_Pa"],
                        "feed_CO2_mol_s": d["feed_CO2_mol_s"],
                        "feed_H2O_mol_s": d["feed_H2O_mol_s"],
                        "feed_N2_mol_s": d["feed_N2_mol_s"],
                        "feed_O2_mol_s": d["feed_O2_mol_s"],
                        "vent_CO2_mol_s": d["vent_CO2_mol_s"],
                        "vent_H2O_mol_s": d["vent_H2O_mol_s"],
                        "vent_N2_mol_s": d["vent_N2_mol_s"],
                        "vent_O2_mol_s": d["vent_O2_mol_s"],
                        "product_CO2_mol_s": d["product_CO2_mol_s"],
                        "product_H2O_mol_s": d["product_H2O_mol_s"],
                        "product_N2_mol_s": d["product_N2_mol_s"],
                        "product_O2_mol_s": d["product_O2_mol_s"],
                        "ndot_product_mol_s": d["ndot_product_mol_s"],
                        "T_product_K": d["T_product_K"],
                        "P_product_Pa": d["P_product_Pa"],
                        "y_CO2_product": d["y_CO2_product"],
                        "y_H2O_product": d["y_H2O_product"],
                        "y_N2_product": d["y_N2_product"],
                        "y_O2_product": d["y_O2_product"],
                        "Q_heat_W": d["Q_heat_W"],
                        "Q_cool_W": d["Q_cool_W"],
                        "Q_heat_utility_W": d["Q_heat_utility_W"],
                        "Q_cool_utility_W": d["Q_cool_utility_W"],
                        "Q_heat_to_bed_W": d["Q_heat_to_bed_W"],
                        "Q_cool_from_bed_W": d["Q_cool_from_bed_W"],
                        "W_fan_W": d["W_fan_W"],
                        "W_vac_W": d["W_vac_W"],
                        "W_repress_W": d["W_repress_W"],
                        "H_feed_W": d.get("H_feed_W", 0.0),
                        "H_vent_W": d.get("H_vent_W", 0.0),
                        "H_product_W": d.get("H_product_W", 0.0),
                        "Q_ads_W": d.get("Q_ads_W", 0.0),
                        "feed_CO2_mol_interval": d.get("feed_CO2_mol_interval", np.nan),
                        "feed_H2O_mol_interval": d.get("feed_H2O_mol_interval", np.nan),
                        "feed_N2_mol_interval": d.get("feed_N2_mol_interval", np.nan),
                        "feed_O2_mol_interval": d.get("feed_O2_mol_interval", np.nan),
                        "vent_CO2_mol_interval": d.get("vent_CO2_mol_interval", np.nan),
                        "vent_H2O_mol_interval": d.get("vent_H2O_mol_interval", np.nan),
                        "vent_N2_mol_interval": d.get("vent_N2_mol_interval", np.nan),
                        "vent_O2_mol_interval": d.get("vent_O2_mol_interval", np.nan),
                        "product_CO2_mol_interval": d.get("product_CO2_mol_interval", np.nan),
                        "product_H2O_mol_interval": d.get("product_H2O_mol_interval", np.nan),
                        "product_N2_mol_interval": d.get("product_N2_mol_interval", np.nan),
                        "product_O2_mol_interval": d.get("product_O2_mol_interval", np.nan),
                        "U_total_proxy_J": d["U_total_proxy_J"],
                    }
                    all_records.append(record)
                    cycle_records_indices.append(len(all_records) - 1)

                    if int(round(t_abs)) % max(int(round(self.numeric.node_record_stride_s)), 1) == 0:
                        Pnode = d["P_node_Pa"]
                        Tgnode = d["Tg_node_K"]
                        Tsnode = d["Ts_node_K"]
                        Twnode = d["Tw_node_K"]
                        Tjnode = d["Tj_node_K"]
                        qco2 = d["q_CO2_node_mol_kg"]
                        qh2o = d["q_H2O_node_mol_kg"]
                        RHraw = d["RH_raw_node"]

                        for i in range(self.N):
                            nrec = {
                                "cycle": cyc,
                                "step": step_name,
                                "t_global_s": t_abs,
                                "t_cycle_s": t_abs - cycle_start_time,
                                "t_step_s": float(t_local),
                                "node": i + 1,
                                "z_m": self.z[i],
                                "P_Pa": Pnode[i],
                                "Tg_K": Tgnode[i],
                                "Ts_K": Tsnode[i],
                                "Tw_K": Twnode[i],
                                "Tj_K": Tjnode[i],
                                "q_CO2_mol_kg": qco2[i],
                                "q_H2O_mol_kg": qh2o[i],
                                "RH_raw": RHraw[i],
                            }
                            node_records_all.append(nrec)
                            if cyc == last_cycle:
                                node_records_last.append(nrec)

                # Advance initial condition and floor/clip between steps.
                y0 = sol_y_dim[:, -1].copy()
                C, q, Tg, Ts, Tw, Tj = self.unpack(y0)
                C, Tg, _, _ = self.regularize_gas_state(C, Tg)
                q = smooth_floor(
                    q,
                    self.numeric.min_loading_mol_kg,
                    rel_width=self.numeric.smooth_floor_rel_width,
                    abs_width=max(self.numeric.smooth_floor_abs_width, 1e-9),
                )
                Ts = smooth_clip(Ts, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
                Tw = smooth_clip(Tw, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
                Tj = smooth_clip(Tj, self.numeric.min_temperature_K, self.numeric.max_temperature_K, rel_width=0.0, abs_width=max(self.numeric.smooth_temperature_width_K, 1e-9))
                y0 = self.pack(C, q, Tg, Ts, Tw, Tj)

                t_global_start += duration

            df_cycle = pd.DataFrame([all_records[i] for i in cycle_records_indices])
            cycle_summaries.append(self.summarize_cycle(df_cycle, cyc))

            C1, q1, Tg1, Ts1, Tw1, Tj1 = self.unpack(y0)
            inv_end = self.inventory(C1, q1, Tg1, Ts1, Tw1, Tj1)

            mb = self.mass_balance_cycle(df_cycle, inv_start, inv_end, cyc)
            eb = self.energy_balance_cycle(df_cycle, inv_start, inv_end, cyc)
            dg = self.diagnostics_cycle(df_cycle, cyc)

            mass_balance_rows.append(mb)
            energy_balance_rows.append(eb)
            diagnostics_rows.append(dg)

        profiles = pd.DataFrame(all_records)
        nodes_all = pd.DataFrame(node_records_all)
        nodes_last = pd.DataFrame(node_records_last)
        summary_all = pd.DataFrame(cycle_summaries)
        summary_all = self.add_cycle_convergence_diagnostics(summary_all)
        product_last = profiles[profiles["cycle"] == last_cycle].copy()
        energy_last = self.energy_timeseries(product_last)
        mass_balance = pd.DataFrame(mass_balance_rows)
        energy_balance = pd.DataFrame(energy_balance_rows)
        diagnostics = pd.DataFrame(diagnostics_rows)
        solver_log = pd.DataFrame(self.solver_log)

        return profiles, nodes_all, nodes_last, product_last, summary_all, energy_last, mass_balance, energy_balance, diagnostics, solver_log

    # -------------------------------------------------------------------------
    # Summaries and diagnostics
    # -------------------------------------------------------------------------

    def add_cycle_convergence_diagnostics(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Add cycle-to-cycle convergence diagnostics to summary rows."""
        if summary is None or summary.empty:
            return summary
        out = summary.copy().sort_values("cycle").reset_index(drop=True)
        tol = float(getattr(self.numeric, "css_relative_tolerance", 0.05))
        metrics = {
            "kg_CO2_cycle": "delta_CO2_product_prev_percent",
            "kg_H2O_cycle": "delta_H2O_product_prev_percent",
            "Q_heat_utility_J_cycle": "delta_Qheat_utility_prev_percent",
            "E_el_engineering_J_cycle": "delta_Eel_engineering_prev_percent",
        }
        for src, dst in metrics.items():
            if src in out.columns:
                prev = out[src].shift(1)
                out[dst] = 100.0 * (out[src] - prev) / prev.abs().replace(0.0, np.nan)
            else:
                out[dst] = np.nan
        delta_cols = list(metrics.values())
        out["css_max_abs_delta_percent"] = out[delta_cols].abs().max(axis=1, skipna=True)
        out["css_pass_flag"] = (out["css_max_abs_delta_percent"] <= 100.0 * tol) & (out["cycle"] > 1)
        return out

    def integrate_col(self, df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) < 2:
            return 0.0
        return float(np.trapezoid(df[col].to_numpy(), df["t_global_s"].to_numpy()))

    def integrate_col_where(self, df: pd.DataFrame, col: str, mask: pd.Series | np.ndarray) -> float:
        """Integrate a rate column over contiguous True blocks of a mask."""
        if col not in df.columns or len(df) < 2:
            return 0.0
        mask_arr = np.asarray(mask, dtype=bool)
        if not np.any(mask_arr):
            return 0.0
        total = 0.0
        idx = np.where(mask_arr)[0]
        if len(idx) < 2:
            return 0.0
        # Split into contiguous index blocks so unrelated steps are not bridged.
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for block in np.split(idx, splits):
            if len(block) >= 2:
                total += float(np.trapezoid(df[col].to_numpy()[block], df["t_global_s"].to_numpy()[block]))
        return total

    def amount_from_rate_or_intervals(self, df: pd.DataFrame, stream: str, comp: str) -> float:
        rate_col = f"{stream}_{comp}_mol_s"
        interval_col = f"{stream}_{comp}_mol_interval"
        if interval_col not in df.columns:
            return self.integrate_col(df, rate_col)
        interval = pd.to_numeric(df[interval_col], errors="coerce")
        exact_sum = float(interval.fillna(0.0).sum())
        ode_mask = interval.isna()
        return exact_sum + self.integrate_col_where(df, rate_col, ode_mask)

    def summarize_cycle(self, df: pd.DataFrame, cycle: int) -> dict:
        duration_s = df["t_cycle_s"].max() - df["t_cycle_s"].min()

        n_prod = {c: self.amount_from_rate_or_intervals(df, "product", c) for c in COMPONENTS}
        n_feed = {c: self.amount_from_rate_or_intervals(df, "feed", c) for c in COMPONENTS}
        n_vent = {c: self.amount_from_rate_or_intervals(df, "vent", c) for c in COMPONENTS}

        n_CO2 = n_prod["CO2"]
        n_H2O = n_prod["H2O"]
        m_CO2 = n_CO2 * MW["CO2"]
        m_H2O = n_H2O * MW["H2O"]

        E_fan_J = self.integrate_col(df, "W_fan_W")
        E_vac_J = self.integrate_col(df, "W_vac_W")
        E_repress_J = self.integrate_col(df, "W_repress_W")
        # Bed-side and utility-side heat duties are kept separate. Legacy Q_heat_W
        # remains bed-side; the Jajjawi/Aspen-like KPI uses utility duty.
        Q_heat_bed_J = self.integrate_col(df, "Q_heat_to_bed_W") if "Q_heat_to_bed_W" in df.columns else self.integrate_col(df, "Q_heat_W")
        Q_cool_bed_J = self.integrate_col(df, "Q_cool_from_bed_W") if "Q_cool_from_bed_W" in df.columns else self.integrate_col(df, "Q_cool_W")
        Q_heat_J = self.integrate_col(df, "Q_heat_W")
        Q_cool_J = self.integrate_col(df, "Q_cool_W")
        Q_heat_utility_J = self.integrate_col(df, "Q_heat_utility_W") if "Q_heat_utility_W" in df.columns else Q_heat_bed_J
        Q_cool_utility_J = self.integrate_col(df, "Q_cool_utility_W") if "Q_cool_utility_W" in df.columns else Q_cool_bed_J
        E_chiller_J = Q_cool_utility_J / max(self.cycle.COP_chiller, 1e-9)
        E_el_jajjawi_like_J = E_fan_J + E_vac_J
        E_el_engineering_J = E_fan_J + E_vac_J + E_repress_J + E_chiller_J
        E_el_J = E_el_engineering_J

        heat_delivery_eff = Q_heat_bed_J / Q_heat_utility_J if Q_heat_utility_J > 0 else np.nan

        # Fixed-time cycle diagnostics for desorption and cooling quality.
        des_df = df[df["step"] == "heating_desorption"].copy()
        ads_df = df[df["step"] == "adsorption"].copy()
        cool_df = df[df["step"] == "closed_cooling"].copy()
        cutoff = float(self.cycle.desorption_co2_cutoff_mol_s)
        if not des_df.empty:
            des_last = des_df.iloc[-1]
            co2_flow_end_des = float(des_last.get("product_CO2_mol_s", np.nan))
            h2o_flow_end_des = float(des_last.get("product_H2O_mol_s", np.nan))
            q_co2_avg_end_des = float(des_last.get("q_CO2_avg_mol_kg", np.nan))
            q_co2_max_end_des = float(des_last.get("q_CO2_max_mol_kg", np.nan))
            q_co2_min_possible_des = float(des_last.get("qeq_CO2_avg_mol_kg", np.nan))
            above_cutoff = des_df["product_CO2_mol_s"].to_numpy(dtype=float) > cutoff
            if np.any(above_cutoff):
                last_above_t = float(des_df.loc[above_cutoff, "t_step_s"].max())
                wasted_des_time = max(float(self.cycle.heating_desorption_time_s) - last_above_t, 0.0) if co2_flow_end_des <= cutoff else 0.0
            else:
                wasted_des_time = np.nan
            cutoff_reached = bool(co2_flow_end_des <= cutoff) if np.isfinite(co2_flow_end_des) else False
        else:
            co2_flow_end_des = h2o_flow_end_des = q_co2_avg_end_des = q_co2_max_end_des = q_co2_min_possible_des = np.nan
            wasted_des_time = np.nan
            cutoff_reached = False

        if not ads_df.empty:
            q_co2_avg_end_ads = float(ads_df.iloc[-1].get("q_CO2_avg_mol_kg", np.nan))
        else:
            q_co2_avg_end_ads = np.nan
        denom_regen = q_co2_avg_end_ads - q_co2_min_possible_des
        regeneration_fraction = (
            (q_co2_avg_end_ads - q_co2_avg_end_des) / denom_regen
            if np.isfinite(denom_regen) and abs(denom_regen) > 1e-12
            else np.nan
        )

        if not cool_df.empty:
            max_Ts_end_cooling = float(cool_df.iloc[-1].get("Ts_max_K", cool_df.iloc[-1].get("Ts_avg_K", np.nan)))
        else:
            max_Ts_end_cooling = np.nan
        cooling_pass_75C = bool(max_Ts_end_cooling <= self.cycle.cooling_max_sorbent_temperature_K) if np.isfinite(max_Ts_end_cooling) else False

        prod_duration_s = self.cycle.evacuation_time_s + self.cycle.heating_desorption_time_s

        # Flow-weighted product average temperature and pressure.
        npt = df["ndot_product_mol_s"].to_numpy()
        if np.sum(npt) > 0:
            avg_Tprod = float(np.trapezoid(npt * df["T_product_K"].to_numpy(), df["t_global_s"]) / np.trapezoid(npt, df["t_global_s"]))
            avg_Pprod = float(np.trapezoid(npt * df["P_product_Pa"].to_numpy(), df["t_global_s"]) / np.trapezoid(npt, df["t_global_s"]))
        else:
            avg_Tprod = np.nan
            avg_Pprod = np.nan

        out = {
            "cycle": cycle,
            "cycle_time_s": duration_s,
            "adsorption_time_s": self.cycle.adsorption_time_s,
            "evacuation_time_s": self.cycle.evacuation_time_s,
            "heating_desorption_time_s": self.cycle.heating_desorption_time_s,
            "cooling_time_s": self.cycle.cooling_time_s,
            "repressurization_time_s": self.cycle.repressurization_time_s,
            "n_CO2_product_mol_cycle": n_prod["CO2"],
            "n_H2O_product_mol_cycle": n_prod["H2O"],
            "n_N2_product_mol_cycle": n_prod["N2"],
            "n_O2_product_mol_cycle": n_prod["O2"],
            "n_product_total_mol_cycle": sum(n_prod.values()),
            "n_CO2_feed_mol_cycle": n_feed["CO2"],
            "n_H2O_feed_mol_cycle": n_feed["H2O"],
            "n_N2_feed_mol_cycle": n_feed["N2"],
            "n_O2_feed_mol_cycle": n_feed["O2"],
            "n_CO2_vent_mol_cycle": n_vent["CO2"],
            "n_H2O_vent_mol_cycle": n_vent["H2O"],
            "n_N2_vent_mol_cycle": n_vent["N2"],
            "n_O2_vent_mol_cycle": n_vent["O2"],
            "kg_CO2_cycle": m_CO2,
            "kg_H2O_cycle": m_H2O,
            "CO2_H2O_mol_ratio": n_CO2 / n_H2O if n_H2O > 0 else np.nan,
            "CO2_H2O_mass_ratio": m_CO2 / m_H2O if m_H2O > 0 else np.nan,
            "avg_ndot_CO2_product_mol_s": n_CO2 / prod_duration_s,
            "avg_ndot_H2O_product_mol_s": n_H2O / prod_duration_s,
            "avg_ndot_N2_product_mol_s": n_prod["N2"] / prod_duration_s,
            "avg_ndot_O2_product_mol_s": n_prod["O2"] / prod_duration_s,
            "avg_product_T_K": avg_Tprod,
            "avg_product_P_Pa": avg_Pprod,
            "Q_heat_J_cycle": Q_heat_utility_J,  # primary Aspen/Jajjawi-like thermal KPI
            "Q_heat_bed_J_cycle": Q_heat_bed_J,
            "Q_cool_J_cycle": Q_cool_utility_J,
            "Q_cool_bed_J_cycle": Q_cool_bed_J,
            "Q_heat_legacy_bed_rate_J_cycle": Q_heat_J,
            "Q_cool_legacy_bed_rate_J_cycle": Q_cool_J,
            "Q_heat_utility_J_cycle": Q_heat_utility_J,
            "Q_cool_utility_J_cycle": Q_cool_utility_J,
            "heat_delivery_efficiency": heat_delivery_eff,
            "E_fan_J_cycle": E_fan_J,
            "E_vacuum_J_cycle": E_vac_J,
            "E_repress_J_cycle": E_repress_J,
            "E_chiller_J_cycle": E_chiller_J,
            "E_el_jajjawi_like_J_cycle": E_el_jajjawi_like_J,
            "E_el_engineering_J_cycle": E_el_engineering_J,
            "E_total_el_J_cycle": E_el_engineering_J,
            "E_total_jajjawi_like_J_cycle": Q_heat_utility_J + E_el_jajjawi_like_J,
            "E_total_engineering_J_cycle": Q_heat_utility_J + E_el_engineering_J,
            "Q_heat_kWhth_cycle": Q_heat_utility_J / 3.6e6,
            "Q_heat_bed_kWhth_cycle": Q_heat_bed_J / 3.6e6,
            "Q_cool_kWhth_cycle": Q_cool_utility_J / 3.6e6,
            "Q_cool_bed_kWhth_cycle": Q_cool_bed_J / 3.6e6,
            "Q_heat_utility_kWhth_cycle": Q_heat_utility_J / 3.6e6,
            "Q_cool_utility_kWhth_cycle": Q_cool_utility_J / 3.6e6,
            "E_fan_kWhe_cycle": E_fan_J / 3.6e6,
            "E_vacuum_kWhe_cycle": E_vac_J / 3.6e6,
            "E_repress_kWhe_cycle": E_repress_J / 3.6e6,
            "E_chiller_kWhe_cycle": E_chiller_J / 3.6e6,
            "E_el_jajjawi_like_kWhe_cycle": E_el_jajjawi_like_J / 3.6e6,
            "E_el_engineering_kWhe_cycle": E_el_engineering_J / 3.6e6,
            "E_total_el_kWhe_cycle": E_el_engineering_J / 3.6e6,
            "productivity_kgCO2_day": m_CO2 / duration_s * 86400.0 if duration_s > 0 else np.nan,
            "productivity_kgCO2_kgads_h": m_CO2 / self.m_ads_kg * 3600.0 / duration_s if duration_s > 0 else np.nan,
            "annual_tCO2_per_bed": m_CO2 * (365.0 * 24.0 * 3600.0 / duration_s) / 1000.0 if duration_s > 0 else np.nan,
            "specific_heat_MWhth_tCO2": (Q_heat_utility_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_heat_utility_MWhth_tCO2": (Q_heat_utility_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_heat_bed_MWhth_tCO2": (Q_heat_bed_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_electricity_MWhe_tCO2": (E_el_engineering_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_electricity_jajjawi_like_MWhe_tCO2": (E_el_jajjawi_like_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_electricity_engineering_MWhe_tCO2": (E_el_engineering_J / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_total_jajjawi_like_MWh_tCO2_before_compression": ((Q_heat_utility_J + E_el_jajjawi_like_J) / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_total_engineering_MWh_tCO2_before_compression": ((Q_heat_utility_J + E_el_engineering_J) / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "specific_total_MWh_tCO2_before_compression": ((Q_heat_utility_J + E_el_engineering_J) / 3.6e9) / (m_CO2 / 1000.0) if m_CO2 > 0 else np.nan,
            "co2_product_flow_end_desorption_mol_s": co2_flow_end_des,
            "h2o_product_flow_end_desorption_mol_s": h2o_flow_end_des,
            "q_CO2_avg_end_adsorption_mol_kg": q_co2_avg_end_ads,
            "q_CO2_avg_end_desorption_mol_kg": q_co2_avg_end_des,
            "q_CO2_max_end_desorption_mol_kg": q_co2_max_end_des,
            "q_CO2_min_possible_desorption_mol_kg": q_co2_min_possible_des,
            "regeneration_fraction": regeneration_fraction,
            "max_Ts_end_cooling_K": max_Ts_end_cooling,
            "cooling_pass_75C_flag": cooling_pass_75C,
            "desorption_cutoff_reached_flag": cutoff_reached,
            "wasted_desorption_time_s": wasted_des_time,
            "m_ads_kg": self.m_ads_kg,
            "bed_volume_m3": self.volume,
            "superficial_velocity_m_s": self.u_ads_m_s,
            "feed_molar_flow_mol_s": self.ndot_feed_mol_s,
            "feed_volumetric_flow_m3_s": self.Vdot_air_m3_s,
            "T_amb_K": self.T_amb_K,
            "RH_frac": self.RH_frac,
            "P_amb_Pa": self.P_amb_Pa,
            "CO2_ppm": self.weather.CO2_ppm,
        }
        return out

    def mass_balance_cycle(self, df: pd.DataFrame, inv_start: dict, inv_end: dict, cycle: int) -> dict:
        out = {"cycle": cycle}
        for comp in COMPONENTS:
            feed = self.amount_from_rate_or_intervals(df, "feed", comp)
            vent = self.amount_from_rate_or_intervals(df, "vent", comp)
            prod = self.amount_from_rate_or_intervals(df, "product", comp)
            inv0 = inv_start[f"gas_{comp}_mol"] + inv_start[f"solid_{comp}_mol"]
            inv1 = inv_end[f"gas_{comp}_mol"] + inv_end[f"solid_{comp}_mol"]
            residual = feed - vent - prod - (inv1 - inv0)
            denom = max(abs(feed), abs(vent) + abs(prod) + abs(inv1 - inv0), 1e-12)
            out[f"{comp}_feed_mol"] = feed
            out[f"{comp}_vent_mol"] = vent
            out[f"{comp}_product_mol"] = prod
            out[f"{comp}_delta_inventory_mol"] = inv1 - inv0
            out[f"{comp}_residual_mol"] = residual
            out[f"{comp}_relative_residual"] = residual / denom
        return out

    def energy_balance_cycle(self, df: pd.DataFrame, inv_start: dict, inv_end: dict, cycle: int) -> dict:
        Q_heat = self.integrate_col(df, "Q_heat_W")
        Q_cool = self.integrate_col(df, "Q_cool_W")
        Q_heat_utility = self.integrate_col(df, "Q_heat_utility_W") if "Q_heat_utility_W" in df.columns else Q_heat
        Q_cool_utility = self.integrate_col(df, "Q_cool_utility_W") if "Q_cool_utility_W" in df.columns else Q_cool
        E_fan = self.integrate_col(df, "W_fan_W")
        E_vac = self.integrate_col(df, "W_vac_W")
        E_rep = self.integrate_col(df, "W_repress_W")
        H_feed = self.integrate_col(df, "H_feed_W") if "H_feed_W" in df.columns else 0.0
        H_vent = self.integrate_col(df, "H_vent_W") if "H_vent_W" in df.columns else 0.0
        H_product = self.integrate_col(df, "H_product_W") if "H_product_W" in df.columns else 0.0
        Q_ads = self.integrate_col(df, "Q_ads_W") if "Q_ads_W" in df.columns else 0.0
        dU = inv_end["U_total_proxy_J"] - inv_start["U_total_proxy_J"]

        residual = Q_heat - Q_cool + E_fan + E_vac + E_rep + H_feed - H_vent - H_product + Q_ads - dU
        denom = max(
            abs(Q_heat) + abs(Q_cool) + abs(E_fan) + abs(E_vac) + abs(E_rep)
            + abs(H_feed) + abs(H_vent) + abs(H_product) + abs(Q_ads) + abs(dU),
            1e-12,
        )
        return {
            "cycle": cycle,
            "Q_heat_J": Q_heat,
            "Q_cool_J": Q_cool,
            "Q_heat_utility_J": Q_heat_utility,
            "Q_cool_utility_J": Q_cool_utility,
            "E_fan_J": E_fan,
            "E_vacuum_J": E_vac,
            "E_repress_J": E_rep,
            "E_el_jajjawi_like_J": E_fan + E_vac,
            "E_el_engineering_J": E_fan + E_vac + E_rep + Q_cool_utility / max(self.cycle.COP_chiller, 1e-9),
            "H_feed_J": H_feed,
            "H_vent_J": H_vent,
            "H_product_J": H_product,
            "Q_ads_J": Q_ads,
            "delta_U_total_proxy_J": dU,
            "energy_proxy_residual_J": residual,
            "energy_proxy_relative_residual": residual / denom,
        }

    def diagnostics_cycle(self, df: pd.DataFrame, cycle: int) -> dict:
        return {
            "cycle": cycle,
            "max_RH_raw_cycle": float(df["max_RH_raw"].max()),
            "n_supersat_node_samples_cycle": int(df["n_supersat_nodes"].sum()),
            "supersaturation_excess_sum_cycle": float(df["supersat_excess_sum"].sum()),
            "clip_concentration_count_cycle": int(df["clip_concentration_count"].sum()),
            "clip_loading_count_cycle": int(df["clip_loading_count"].sum()),
            "clip_temperature_low_count_cycle": int(df["clip_temperature_low_count"].sum()),
            "clip_temperature_high_count_cycle": int(df["clip_temperature_high_count"].sum()),
            "min_Pavg_Pa": float(df["Pavg_Pa"].min()),
            "max_Pavg_Pa": float(df["Pavg_Pa"].max()),
            "min_Ts_avg_K": float(df["Ts_avg_K"].min()),
            "max_Ts_avg_K": float(df["Ts_avg_K"].max()),
            "min_q_CO2_avg_mol_kg": float(df["q_CO2_avg_mol_kg"].min()),
            "max_q_CO2_avg_mol_kg": float(df["q_CO2_avg_mol_kg"].max()),
        }

    def energy_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[
            "cycle", "step", "t_global_s", "t_cycle_s",
            "Q_heat_W", "Q_cool_W", "Q_heat_utility_W", "Q_cool_utility_W",
            "Q_heat_to_bed_W", "Q_cool_from_bed_W",
            "W_fan_W", "W_vac_W", "W_repress_W",
            "product_CO2_mol_s", "product_H2O_mol_s",
        ]].copy()
        out["W_chiller_W"] = out["Q_cool_utility_W"] / max(self.cycle.COP_chiller, 1e-9)
        out["W_el_jajjawi_like_W"] = out["W_fan_W"] + out["W_vac_W"]
        out["W_total_el_W"] = out["W_fan_W"] + out["W_vac_W"] + out["W_repress_W"] + out["W_chiller_W"]
        out["W_el_engineering_W"] = out["W_total_el_W"]
        return out


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_jakarta_weather_from_nasa(path: Path, cfg: WeatherConfig) -> WeatherConfig:
    """Load representative Jakarta condition from user's NASA POWER CSV."""
    if not path.exists():
        warnings.warn(f"NASA POWER CSV not found. Fallback weather is used: {path}")
        return cfg

    usecols = [
        "country_code", "country_name", "province_id", "province_name",
        "datetime_utc", "T2M", "RH2M", "PS"
    ]

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        pname = chunk["province_name"].astype(str)
        mask = pname.str.contains(cfg.province_query, case=False, na=False)
        mask |= pname.str.contains("DKI", case=False, na=False)
        mask |= pname.str.contains("Jakarta", case=False, na=False)
        sub = chunk.loc[mask].copy()
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        warnings.warn("No Jakarta province row found in NASA POWER CSV. Fallback weather is used.")
        return cfg

    df = pd.concat(chunks, ignore_index=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    if cfg.datetime_utc:
        target = pd.to_datetime(cfg.datetime_utc)
        row = df.iloc[(df["datetime_utc"] - target).abs().argsort().iloc[0]]
        T_C = float(row["T2M"])
        RH = float(row["RH2M"])
        P_Pa = float(row["PS"]) * 1000.0
        dt = str(row["datetime_utc"])
    else:
        T_C = float(pd.to_numeric(df["T2M"], errors="coerce").mean())
        RH = float(pd.to_numeric(df["RH2M"], errors="coerce").mean())
        P_Pa = float(pd.to_numeric(df["PS"], errors="coerce").mean()) * 1000.0
        dt = "annual_mean"

    return WeatherConfig(
        T_C=T_C,
        RH_percent=RH,
        P_Pa=P_Pa,
        CO2_ppm=cfg.CO2_ppm,
        province_query=cfg.province_query,
        datetime_utc=dt,
    )


# =============================================================================
# OUTPUT AND PLOTTING
# =============================================================================

def ensure_output_dir(paths: PathConfig) -> Path:
    out_dir = paths.project_dir / paths.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(out_dir: Path, weather: WeatherConfig, bed: BedConfig, ads: AdsorbentConfig, cycle: CycleConfig, numeric: NumericConfig) -> None:
    config = {
        "weather": asdict(weather),
        "bed": asdict(bed),
        "adsorbent": asdict(ads),
        "cycle": asdict(cycle),
        "numeric": asdict(numeric),
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def apply_origin_like_style(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_line(out_dir: Path, df: pd.DataFrame, x: str, ys: list[str], labels: list[str], ylabel: str, title: str, output: str, scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for y, lab in zip(ys, labels):
        if y in df.columns:
            ax.plot(df[x].to_numpy(), df[y].to_numpy() * scale, linewidth=1.4, label=lab)
    ax.set_xlabel("Global time (s)" if x == "t_global_s" else "Cycle time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_origin_like_style(ax)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / output, dpi=300)
    plt.close(fig)


def plot_node_lines(out_dir: Path, nodes: pd.DataFrame, x: str, y: str, ylabel: str, title: str, output: str, scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for node, sub in nodes.groupby("node"):
        label = f"Node {node}" if int(node) in {1, 5, 10, 15, 20} else None
        ax.plot(sub[x].to_numpy(), sub[y].to_numpy() * scale, linewidth=0.9, alpha=0.85, label=label)
    ax.set_xlabel("Global time (s)" if x == "t_global_s" else "Cycle time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_origin_like_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8, ncol=5, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / output, dpi=300)
    plt.close(fig)


def plot_dual_product(out_dir: Path, df: pd.DataFrame, x: str, title: str, output: str) -> None:
    fig, ax1 = plt.subplots(figsize=(9.0, 5.2))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df[x].to_numpy(), df["product_CO2_mol_s"].to_numpy(), linewidth=1.4, label="CO2 product")
    l2, = ax2.plot(df[x].to_numpy(), df["product_H2O_mol_s"].to_numpy(), linewidth=1.4, linestyle="--", label="H2O product")

    ax1.set_xlabel("Global time (s)" if x == "t_global_s" else "Cycle time (s)")
    ax1.set_ylabel("CO2 product flow (mol/s)")
    ax2.set_ylabel("H2O product flow (mol/s)")
    ax1.set_title(title)
    apply_origin_like_style(ax1)
    ax2.tick_params(direction="in", top=True, right=True)
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / output, dpi=300)
    plt.close(fig)


def plot_outputs(out_dir: Path, profiles: pd.DataFrame, nodes_all: pd.DataFrame, nodes_last: pd.DataFrame, product_last: pd.DataFrame, energy_last: pd.DataFrame, cycle: CycleConfig) -> None:
    # 20-cycle/global-time line plots based on bed-average records.
    plot_line(
        out_dir, profiles, "t_global_s", ["Pavg_Pa"], ["Bed-average pressure"],
        "Pressure (bar)", "Pressure profile, all cycles", "pressure_profile_all_cycles.png", scale=1e-5
    )
    plot_line(
        out_dir, profiles, "t_global_s", ["Ts_avg_K"], ["Bed-average solid temperature"],
        "Solid temperature (K)", "Solid temperature profile, all cycles", "temperature_profile_all_cycles.png"
    )
    plot_line(
        out_dir, profiles, "t_global_s", ["q_CO2_avg_mol_kg"], ["Average CO2 loading"],
        "CO2 loading (mol/kg)", "CO2 solid loading, all cycles", "solid_loading_co2_all_cycles.png"
    )
    plot_dual_product(out_dir, profiles, "t_global_s", "Product CO2 and H2O flow, all cycles", "product_co2_h2o_all_cycles.png")

    # Last-cycle axial-node line plots.
    plot_node_lines(
        out_dir, nodes_last, "t_cycle_s", "P_Pa",
        "Pressure (bar)", f"Pressure profile, cycle {cycle.n_cycles}", "pressure_profile_last_cycle.png", scale=1e-5
    )
    plot_node_lines(
        out_dir, nodes_last, "t_cycle_s", "Ts_K",
        "Solid temperature (K)", f"Solid temperature profile, cycle {cycle.n_cycles}", "temperature_profile_last_cycle.png"
    )
    plot_node_lines(
        out_dir, nodes_last, "t_cycle_s", "q_CO2_mol_kg",
        "CO2 loading (mol/kg)", f"CO2 solid loading, cycle {cycle.n_cycles}", "solid_loading_co2_last_cycle.png"
    )
    plot_dual_product(out_dir, product_last, "t_cycle_s", f"Product CO2 and H2O flow, cycle {cycle.n_cycles}", "product_co2_h2o_last_cycle.png")

    # Energy breakdown last cycle.
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(energy_last["t_cycle_s"], energy_last["Q_heat_W"] / 1000.0, label="Heating duty")
    ax.plot(energy_last["t_cycle_s"], energy_last["W_vac_W"] / 1000.0, label="Vacuum pump")
    ax.plot(energy_last["t_cycle_s"], energy_last["W_fan_W"] / 1000.0, label="Fan")
    ax.plot(energy_last["t_cycle_s"], energy_last["W_chiller_W"] / 1000.0, label="Chiller")
    ax.set_xlabel("Cycle time (s)")
    ax.set_ylabel("Power / duty (kW)")
    ax.set_title(f"Energy-related rates, cycle {cycle.n_cycles}")
    apply_origin_like_style(ax)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "energy_breakdown_last_cycle.png", dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jakarta 1D TVSA DAC single-bed model.")
    parser.add_argument("--project-dir", type=str, default=r"D:/Ashka/5.DAC/06.PYTHON")
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--co2-ppm", type=float, default=400.0)
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--datetime-utc", type=str, default=None)
    parser.add_argument("--ads-time-s", type=float, default=None)
    parser.add_argument("--evac-time-s", type=float, default=None)
    parser.add_argument("--des-time-s", type=float, default=None)
    parser.add_argument("--cool-time-s", type=float, default=None)
    parser.add_argument("--rep-time-s", type=float, default=None)
    parser.add_argument("--Tdes-K", type=float, default=None)
    parser.add_argument("--Tcool-K", type=float, default=None)
    parser.add_argument("--sample-dt-s", type=float, default=10.0)
    parser.add_argument("--max-step-s", type=float, default=20.0)
    parser.add_argument("--node-record-stride-s", type=float, default=30.0)
    parser.add_argument("--jacket-tau-min-s", type=float, default=120.0)
    parser.add_argument("--jacket-inlet-ramp-time-s", type=float, default=180.0)
    parser.add_argument("--evac-valve-ramp-time-s", type=float, default=20.0)
    parser.add_argument("--des-product-valve-ramp-time-s", type=float, default=0.0)
    parser.add_argument("--rep-valve-ramp-time-s", type=float, default=30.0)
    parser.add_argument("--rep-pressure-stop-margin-Pa", type=float, default=250.0)
    parser.add_argument("--evac-pressure-stop-margin-Pa", type=float, default=500.0)
    parser.add_argument("--use-nondimensional-state", action="store_true")
    parser.add_argument("--no-nondimensional-state", action="store_true")
    parser.add_argument("--no-vector-atol", action="store_true")
    parser.add_argument("--atol-C", type=float, default=1e-6)
    parser.add_argument("--atol-q", type=float, default=1e-7)
    parser.add_argument("--atol-T", type=float, default=1e-4)
    parser.add_argument("--gas-thermal-pressure-floor-Pa", type=float, default=5000.0)
    parser.add_argument("--no-freeze-mass-cooling", action="store_true")
    parser.add_argument("--no-robust-cooling-jacket", action="store_true")
    parser.add_argument("--no-jac-sparsity", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--method", type=str, default="BDF")
    parser.add_argument("--pressure-step-method", type=str, default="BDF")
    parser.add_argument("--pressure-step-max-step-s", type=float, default=20.0)
    # Detailed-validation path: full dynamic ODE/PDE-MOL is the default.
    # Fast modes are retained only for explicit comparison/debugging.
    parser.add_argument("--use-fast-pressure-steps", action="store_true")
    parser.add_argument("--use-fast-heating-desorption", action="store_true")
    parser.add_argument("--no-fast-pressure-steps", action="store_true")  # backward-compatible no-op when default is detailed
    parser.add_argument("--no-fast-heating-desorption", action="store_true")  # backward-compatible no-op when default is detailed
    parser.add_argument("--freeze-inert-heating", action="store_true")
    parser.add_argument("--use-near-vacuum-tg-equilibrium", action="store_true")
    parser.add_argument("--fast-heating-step-s", type=float, default=50.0)
    parser.add_argument("--profile", action="store_true", help="Run cProfile and save top expensive functions to outputs/jakarta/profile_top_functions.txt")
    parser.add_argument("--profile-top", type=int, default=40)
    args = parser.parse_args()

    paths = PathConfig(project_dir=Path(args.project_dir))
    weather = WeatherConfig(CO2_ppm=args.co2_ppm, datetime_utc=args.datetime_utc)
    weather = load_jakarta_weather_from_nasa(paths.nasa_hourly_csv, weather)

    bed = BedConfig(n_nodes=args.n_nodes)
    ads = AdsorbentConfig()
    cycle = CycleConfig(
        n_cycles=args.n_cycles,
        evacuation_valve_ramp_time_s=args.evac_valve_ramp_time_s,
        desorption_product_valve_ramp_time_s=args.des_product_valve_ramp_time_s,
        repressurization_valve_ramp_time_s=args.rep_valve_ramp_time_s,
        repressurization_pressure_stop_margin_Pa=args.rep_pressure_stop_margin_Pa,
        evacuation_pressure_stop_margin_Pa=args.evac_pressure_stop_margin_Pa,
    )
    numeric = NumericConfig(
        sample_dt_s=args.sample_dt_s,
        max_step_s=args.max_step_s,
        node_record_stride_s=args.node_record_stride_s,
        jacket_tau_min_s=args.jacket_tau_min_s,
        jacket_inlet_ramp_time_s=args.jacket_inlet_ramp_time_s,
        freeze_mass_during_closed_cooling=not args.no_freeze_mass_cooling,
        robust_cooling_jacket_boundary=not args.no_robust_cooling_jacket,
        use_jac_sparsity=False if args.no_jac_sparsity else NumericConfig.use_jac_sparsity,
        use_nondimensional_state=(args.use_nondimensional_state and not args.no_nondimensional_state),
        use_vector_atol=not args.no_vector_atol,
        atol_C_mol_m3=args.atol_C,
        atol_q_mol_kg=args.atol_q,
        atol_T_K=args.atol_T,
        gas_thermal_pressure_floor_Pa=args.gas_thermal_pressure_floor_Pa,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
        pressure_step_method=args.pressure_step_method,
        pressure_step_max_step_s=args.pressure_step_max_step_s,
        use_fast_pressure_steps=(args.use_fast_pressure_steps and not args.no_fast_pressure_steps),
        freeze_inert_during_heating_desorption=args.freeze_inert_heating,
        use_fast_heating_desorption_steps=(args.use_fast_heating_desorption and not args.no_fast_heating_desorption),
        fast_heating_step_s=args.fast_heating_step_s,
        use_near_vacuum_gas_temperature_equilibrium=args.use_near_vacuum_tg_equilibrium,
        gas_temperature_eq_pressure_threshold_Pa=15000.0,
        gas_temperature_eq_tau_s=60.0,
    )

    if args.ads_time_s is not None:
        cycle.adsorption_time_s = float(args.ads_time_s)
    if args.evac_time_s is not None:
        cycle.evacuation_time_s = float(args.evac_time_s)
    if args.des_time_s is not None:
        cycle.heating_desorption_time_s = float(args.des_time_s)
    if args.cool_time_s is not None:
        cycle.cooling_time_s = float(args.cool_time_s)
    if args.rep_time_s is not None:
        cycle.repressurization_time_s = float(args.rep_time_s)
    if args.Tdes_K is not None:
        cycle.T_des_K = float(args.Tdes_K)
    if args.Tcool_K is not None:
        cycle.T_coolant_K = float(args.Tcool_K)

    out_dir = ensure_output_dir(paths)

    prof = None
    if args.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()

    save_config(out_dir, weather, bed, ads, cycle, numeric)

    model = TVSABedModel(weather, bed, ads, cycle, numeric)

    print("=" * 90)
    print("RUNNING JAKARTA 1D TVSA DAC MODEL")
    print("=" * 90)
    print(f"Project directory : {paths.project_dir}")
    print(f"Output directory  : {out_dir}")
    print(f"Weather           : T={weather.T_C:.3f} °C, RH={weather.RH_percent:.3f} %, P={weather.P_Pa:.1f} Pa")
    print(f"CO2               : {weather.CO2_ppm:.1f} ppm")
    print(f"Bed               : L={bed.bed_length_m} m, D={bed.bed_diameter_m} m, nodes={bed.n_nodes}")
    print(f"Adsorbent mass    : {model.m_ads_kg:.6f} kg")
    print(f"Superficial vel.  : {model.u_ads_m_s:.6f} m/s")
    print(f"Cycle count       : {cycle.n_cycles}")
    print(f"Cycle times       : ads={cycle.adsorption_time_s:.1f}s, evac={cycle.evacuation_time_s:.1f}s, "
          f"des={cycle.heating_desorption_time_s:.1f}s, cool={cycle.cooling_time_s:.1f}s, "
          f"rep={cycle.repressurization_time_s:.1f}s")
    print(f"Valve ramps       : evac={cycle.evacuation_valve_ramp_time_s:.1f}s, "
          f"des_prod={cycle.desorption_product_valve_ramp_time_s:.1f}s, "
          f"rep={cycle.repressurization_valve_ramp_time_s:.1f}s")
    print(f"Nondim state      : {numeric.use_nondimensional_state}")
    print(f"Fast pressure     : {numeric.use_fast_pressure_steps} (False = full dynamic ODE/PDE-MOL)")
    print(f"Pressure fallback : {numeric.pressure_step_method}, max_step={numeric.pressure_step_max_step_s:g} s")
    print(f"Vector atol       : {numeric.use_vector_atol} "
          f"(C={numeric.atol_C_mol_m3:g}, q={numeric.atol_q_mol_kg:g}, T={numeric.atol_T_K:g})")
    print(f"Jac sparsity      : {numeric.use_jac_sparsity}")
    print(f"Heating shortcuts : fast split={numeric.use_fast_heating_desorption_steps} "
          f"(dt={numeric.fast_heating_step_s:g}s), "
          f"freeze inert={numeric.freeze_inert_during_heating_desorption}, "
          f"Tg≈Ts near vacuum={numeric.use_near_vacuum_gas_temperature_equilibrium} "
          f"(P<thr {numeric.gas_temperature_eq_pressure_threshold_Pa:g} Pa)")
    if numeric.use_nondimensional_state:
        print("WARNING           : nondimensional state is ON; use only for comparison, not default production run.")
    if (not numeric.use_fast_pressure_steps) and numeric.pressure_step_method.upper().startswith("RK"):
        print("WARNING           : explicit RK pressure fallback can be extremely slow for evacuation/repressurization.")
    print("=" * 90)

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
    ) = model.simulate()

    last_cycle = cycle.n_cycles
    suffix = f"cycle{last_cycle}"
    profiles_last = profiles[profiles["cycle"] == last_cycle].copy()
    summary_last = summary_all[summary_all["cycle"] == last_cycle].copy()

    # Last-cycle CSV outputs.
    profiles_last.to_csv(out_dir / f"profiles_{suffix}_long.csv", index=False, encoding="utf-8-sig")
    nodes_last.to_csv(out_dir / f"profiles_{suffix}_nodes.csv", index=False, encoding="utf-8-sig")
    product_last.to_csv(out_dir / f"product_{suffix}.csv", index=False, encoding="utf-8-sig")
    energy_last.to_csv(out_dir / f"energy_{suffix}.csv", index=False, encoding="utf-8-sig")
    summary_last.to_csv(out_dir / f"summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    # Diagnostic/full-cycle outputs.
    summary_all.to_csv(out_dir / "summary_all_cycles.csv", index=False, encoding="utf-8-sig")
    mass_balance.to_csv(out_dir / f"mass_balance_{suffix}.csv", index=False, encoding="utf-8-sig")
    energy_balance.to_csv(out_dir / f"energy_balance_{suffix}.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / f"diagnostics_{suffix}.csv", index=False, encoding="utf-8-sig")
    solver_log.to_csv(out_dir / "solver_log.csv", index=False, encoding="utf-8-sig")
    if getattr(model, "fast_substep_diagnostics", None):
        pd.DataFrame(model.fast_substep_diagnostics).to_csv(
            out_dir / "fast_substep_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_outputs(out_dir, profiles, nodes_all, nodes_last, product_last, energy_last, cycle)

    print("\nSUMMARY - LAST CYCLE")
    print(summary_last.T.to_string(header=False))
    print("\nSAVED OUTPUTS")
    print(f"- {out_dir / f'profiles_{suffix}_long.csv'}")
    print(f"- {out_dir / f'profiles_{suffix}_nodes.csv'}")
    print(f"- {out_dir / f'product_{suffix}.csv'}")
    print(f"- {out_dir / f'energy_{suffix}.csv'}")
    print(f"- {out_dir / f'summary_{suffix}.csv'}")
    print(f"- {out_dir / f'mass_balance_{suffix}.csv'}")
    print(f"- {out_dir / f'energy_balance_{suffix}.csv'}")
    print(f"- {out_dir / f'diagnostics_{suffix}.csv'}")
    print(f"- {out_dir / 'solver_log.csv'}")
    if getattr(model, "fast_substep_diagnostics", None):
        print(f"- {out_dir / 'fast_substep_diagnostics.csv'}")

    if args.profile and prof is not None:
        prof.disable()
        import io
        import pstats
        profile_stream = io.StringIO()
        stats = pstats.Stats(prof, stream=profile_stream).strip_dirs().sort_stats("cumtime")
        stats.print_stats(max(int(args.profile_top), 1))
        profile_path = out_dir / "profile_top_functions.txt"
        profile_path.write_text(profile_stream.getvalue(), encoding="utf-8")
        print(f"- {profile_path}")

    print("=" * 90)


if __name__ == "__main__":
    main()
