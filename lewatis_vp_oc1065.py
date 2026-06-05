# -*- coding: utf-8 -*-
"""
lewatis_vp_oc1065.py

Adsorbent-property and adsorption-equilibrium module for Lewatit VP OC 1065.

Intended location in the user's project:
    D:/Ashka/5.DAC/06.PYTHON/adsorbents/lewatis_vp_oc1065.py

Purpose
-------
This module is designed to be imported by:
    01_jakarta_single_bed_tvsa.py

It contains only adsorbent- and mixture-property definitions needed by the
full-Python TVSA DAC model:
- molecular constants for CO2, H2O, N2, O2
- Lewatit VP OC 1065 material properties
- GAB water isotherm
- dry/wet Toth CO2 branches
- WADST CO2-H2O co-adsorption model
- LDF kinetic rates for CO2 and H2O
- helper functions for humid-air feed composition and equilibrium loading

Reference priority for parameters:
    1. Jajjawi et al. weather-dependent TVSA DAC model and SI
    2. Young et al. WADST/GAB model as reported in Jajjawi SI
    3. Explicit engineering fallback assumptions where a Python implementation
       needs a numerical closure not disclosed by Aspen Adsorption.

Note on spelling
----------------
The commercial resin is Lewatit VP OC 1065. The file name follows the user's
requested spelling: lewatis_vp_oc1065.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


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
    "H2O": 18.01528e-3,     # kg/mol
    "N2": 28.0134e-3,       # kg/mol
    "O2": 31.9988e-3,       # kg/mol
}

CP_GAS_MOLAR = {
    # Engineering constant heat capacities around ambient DAC conditions.
    "CO2": 37.1,            # J/mol/K
    "H2O": 33.6,            # J/mol/K, vapor
    "N2": 29.1,             # J/mol/K
    "O2": 29.4,             # J/mol/K
}

DM = {
    # Molecular diffusivities in air from Jajjawi SI Table 3.
    # SI reports cm2/s; converted here to m2/s.
    "CO2": 0.1381e-4,
    "H2O": 0.2178e-4,
    "N2": 0.1788e-4,
    "O2": 0.1820e-4,
}


# =============================================================================
# ADSORBENT CONFIGURATION
# =============================================================================

@dataclass
class AdsorbentConfig:
    """
    Baseline adsorbent and isotherm configuration for Lewatit VP OC 1065.

    Units:
    - loadings: mol/kg adsorbent
    - pressures: Pa
    - temperature: K
    - heat of adsorption: J/mol
    - heat capacity: J/(kg K) or J/(mol K), as named
    """

    name: str = "Lewatit VP OC 1065"
    source_note: str = "Jajjawi SI / Young WADST-GAB parameters"

    # -------------------------------------------------------------------------
    # CO2 WADST model: dry Toth branch
    # -------------------------------------------------------------------------
    dry_qmax_mol_kg: float = 4.86
    # Jajjawi/Young table lists 2.85e-18 in kPa-type pressure basis in some
    # reproductions. The Python model uses Pa, hence 2.85e-21 Pa^-1.
    dry_b0_Pa_inv: float = 2.85e-21
    dry_minus_dH_J_mol: float = 117_789.0
    dry_t0: float = 0.209
    dry_a: float = 0.523

    # -------------------------------------------------------------------------
    # CO2 WADST model: wet-site Toth branch
    # -------------------------------------------------------------------------
    wet_qmax_mol_kg: float = 9.035
    wet_b0_Pa_inv: float = 1.230e-18
    wet_minus_dH_J_mol: float = 203_687.0
    wet_t0: float = 0.053
    wet_a: float = 0.053

    T0_K: float = 298.15
    # Jajjawi SI reports the WADST water-loading transition parameter as 1.523.
    # Some previous local scripts used 1.532; this module follows Jajjawi SI.
    wadst_A_mol_kg: float = 1.523

    # -------------------------------------------------------------------------
    # H2O GAB model
    # -------------------------------------------------------------------------
    gab_qm_mol_kg: float = 3.63
    gab_C_J_mol: float = 47_110.0
    gab_D_K_inv: float = 0.023744
    gab_F_J_mol: float = 57_706.0
    gab_G_J_molK: float = -47.814

    # -------------------------------------------------------------------------
    # LDF kinetic coefficients
    # -------------------------------------------------------------------------
    k_ldf_co2_s: float = 0.003
    k_ldf_h2o_s: float = 0.0086

    # -------------------------------------------------------------------------
    # Adsorption energetics
    # Negative values mean exothermic adsorption.
    # -------------------------------------------------------------------------
    dH_ads_CO2_J_mol: float = -60.0e3
    dH_ads_H2O_J_mol: float = -49.0e3

    cp_ads_CO2_J_molK: float = 88.0
    cp_ads_H2O_J_molK: float = 75.0

    # -------------------------------------------------------------------------
    # Lewatit VP OC 1065 particle/solid properties from Jajjawi SI Table 4
    # These are repeated here so the adsorbent file can be called independently
    # by future scripts. The 01 script may still keep bed geometry separately.
    # -------------------------------------------------------------------------
    particle_radius_m: float = 0.0007
    inter_particle_voidage_m3_m3: float = 0.400
    intra_particle_voidage_m3_m3: float = 0.238
    bulk_solid_density_kg_m3_bed: float = 880.0
    cp_solid_J_kgK: float = 1.58e3
    thermal_conductivity_solid_W_mK: float = 0.43

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_adsorbent_config() -> AdsorbentConfig:
    """Return the default Lewatit VP OC 1065 configuration."""
    return AdsorbentConfig()


# =============================================================================
# HUMID-AIR AND GAS-PHASE HELPERS
# =============================================================================

def saturation_pressure_water_pa(T_K: np.ndarray | float) -> np.ndarray:
    """
    Saturation vapor pressure of water over liquid water using a Magnus equation.

    Parameters
    ----------
    T_K:
        Temperature in K.

    Returns
    -------
    np.ndarray
        Saturation pressure in Pa.
    """
    T_arr = np.asarray(T_K, dtype=float)
    T_C = T_arr - 273.15
    p_hPa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return p_hPa * 100.0


def feed_composition_from_weather(
    T_K: float,
    RH_frac: float,
    P_Pa: float,
    CO2_ppm: float = 400.0,
) -> dict[str, float]:
    """
    Convert T-RH-P into humid-air gas composition.

    CO2 is treated as a fixed dry-air mole fraction applied to the dry-air
    portion of the gas:
        p_CO2 = x_CO2 * (P - p_H2O)

    Returns mole fractions of CO2, H2O, N2, O2 and selected partial pressures.
    """
    T = float(T_K)
    P = float(P_Pa)
    RH = float(np.clip(RH_frac, 0.0, 0.999999))

    p_sat = float(saturation_pressure_water_pa(T))
    p_H2O = min(RH * p_sat, 0.99 * P)

    x_CO2 = float(CO2_ppm) * 1e-6
    p_CO2 = max(x_CO2 * (P - p_H2O), 0.0)

    y_H2O = p_H2O / P
    y_CO2 = p_CO2 / P
    dry_air_remaining = max(1.0 - y_CO2 - y_H2O, 0.0)

    y_O2 = 0.21 * dry_air_remaining
    y_N2 = dry_air_remaining - y_O2

    y_sum = y_CO2 + y_H2O + y_N2 + y_O2
    return {
        "CO2": y_CO2 / y_sum,
        "H2O": y_H2O / y_sum,
        "N2": y_N2 / y_sum,
        "O2": y_O2 / y_sum,
        "p_sat_H2O_Pa": p_sat,
        "p_H2O_Pa": p_H2O,
        "p_CO2_Pa": p_CO2,
        "RH_frac": RH,
    }


def mixture_mw_kg_mol(y: np.ndarray) -> np.ndarray:
    """Mixture molecular weight from mole fractions, kg/mol."""
    y_arr = np.asarray(y, dtype=float)
    return (
        y_arr[..., IDX["CO2"]] * MW["CO2"]
        + y_arr[..., IDX["H2O"]] * MW["H2O"]
        + y_arr[..., IDX["N2"]] * MW["N2"]
        + y_arr[..., IDX["O2"]] * MW["O2"]
    )


def mixture_cp_molar_J_molK(y: np.ndarray) -> np.ndarray:
    """Mixture molar heat capacity, J/mol/K."""
    y_arr = np.asarray(y, dtype=float)
    return (
        y_arr[..., IDX["CO2"]] * CP_GAS_MOLAR["CO2"]
        + y_arr[..., IDX["H2O"]] * CP_GAS_MOLAR["H2O"]
        + y_arr[..., IDX["N2"]] * CP_GAS_MOLAR["N2"]
        + y_arr[..., IDX["O2"]] * CP_GAS_MOLAR["O2"]
    )


def gas_viscosity_air_sutherland(T_K: np.ndarray | float) -> np.ndarray:
    """Approximate humid-air viscosity with Sutherland's law for air."""
    T = np.asarray(T_K, dtype=float)
    mu0 = 1.716e-5  # Pa.s
    T0 = 273.15     # K
    S = 111.0       # K
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


# =============================================================================
# GAB / TOTH / WADST EQUILIBRIUM FUNCTIONS
# =============================================================================

def gab_h2o_loading(
    T_K: np.ndarray | float,
    RH_frac: np.ndarray | float,
    ads: AdsorbentConfig | None = None,
) -> np.ndarray:
    """
    Temperature-dependent GAB water isotherm.

    q_H2O = qm*k*c*x / [(1 - k*x)(1 + (c - 1)k*x)]

    Parameters
    ----------
    T_K:
        Gas/solid temperature in K.
    RH_frac:
        Relative humidity fraction, 0-1.
    ads:
        AdsorbentConfig. Default: Lewatit VP OC 1065.

    Returns
    -------
    np.ndarray
        Water loading in mol/kg adsorbent.
    """
    if ads is None:
        ads = AdsorbentConfig()

    T = np.asarray(T_K, dtype=float)
    x = np.clip(np.asarray(RH_frac, dtype=float), 1e-12, 0.999999)

    E10_plus = -44.38 * T + 57_220.0
    E1 = ads.gab_C_J_mol - np.exp(ads.gab_D_K_inv * T)
    E2_9 = ads.gab_F_J_mol + ads.gab_G_J_molK * T

    c_gab = np.exp((E1 - E10_plus) / (R * T))
    k_gab = np.exp((E2_9 - E10_plus) / (R * T))

    kx = np.clip(k_gab * x, 1e-12, 0.999999)
    denominator = (1.0 - kx) * (1.0 + (c_gab - 1.0) * kx)
    denominator = np.maximum(denominator, 1e-20)

    q_h2o = ads.gab_qm_mol_kg * k_gab * c_gab * x / denominator
    return np.maximum(q_h2o, 0.0)


def gab_h2o_details(
    T_K: np.ndarray | float,
    RH_frac: np.ndarray | float,
    ads: AdsorbentConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return GAB loading plus intermediate c, k, E1, E2_9 values."""
    if ads is None:
        ads = AdsorbentConfig()

    T = np.asarray(T_K, dtype=float)
    x = np.clip(np.asarray(RH_frac, dtype=float), 1e-12, 0.999999)

    E10_plus = -44.38 * T + 57_220.0
    E1 = ads.gab_C_J_mol - np.exp(ads.gab_D_K_inv * T)
    E2_9 = ads.gab_F_J_mol + ads.gab_G_J_molK * T

    c_gab = np.exp((E1 - E10_plus) / (R * T))
    k_gab = np.exp((E2_9 - E10_plus) / (R * T))
    kx = np.clip(k_gab * x, 1e-12, 0.999999)
    denominator = (1.0 - kx) * (1.0 + (c_gab - 1.0) * kx)
    denominator = np.maximum(denominator, 1e-20)
    q = np.maximum(ads.gab_qm_mol_kg * k_gab * c_gab * x / denominator, 0.0)

    return {
        "q_H2O_GAB_mol_kg": q,
        "c_gab": c_gab,
        "k_gab": k_gab,
        "E10_plus_J_mol": E10_plus,
        "E1_J_mol": E1,
        "E2_9_J_mol": E2_9,
    }


def toth_loading(
    T_K: np.ndarray | float,
    p_CO2_Pa: np.ndarray | float,
    qmax: float,
    b0_Pa_inv: float,
    minus_dH_J_mol: float,
    t0: float,
    a: float,
    T0_K: float,
) -> np.ndarray:
    """
    Temperature-dependent Toth CO2 loading.

    q = qmax*b(T)*p / [1 + (b(T)*p)^t(T)]^(1/t(T))

    Jajjawi/Young report -DeltaH as a positive parameter, so the affinity is
    implemented as:
        b(T) = b0 * exp(minus_dH/(R*T))
    """
    T = np.asarray(T_K, dtype=float)
    p = np.maximum(np.asarray(p_CO2_Pa, dtype=float), 0.0)

    b = b0_Pa_inv * np.exp(minus_dH_J_mol / (R * T))
    t = t0 + a * (1.0 - T0_K / T)
    t = np.clip(t, 1e-6, None)

    bp = np.maximum(b * p, 1e-300)
    q = qmax * bp / ((1.0 + np.power(bp, t)) ** (1.0 / t))
    return np.maximum(q, 0.0)


def dry_toth_co2_loading(
    T_K: np.ndarray | float,
    p_CO2_Pa: np.ndarray | float,
    ads: AdsorbentConfig | None = None,
) -> np.ndarray:
    """Dry-branch Toth CO2 loading, mol/kg."""
    if ads is None:
        ads = AdsorbentConfig()
    return toth_loading(
        T_K=T_K,
        p_CO2_Pa=p_CO2_Pa,
        qmax=ads.dry_qmax_mol_kg,
        b0_Pa_inv=ads.dry_b0_Pa_inv,
        minus_dH_J_mol=ads.dry_minus_dH_J_mol,
        t0=ads.dry_t0,
        a=ads.dry_a,
        T0_K=ads.T0_K,
    )


def wet_site_toth_co2_loading(
    T_K: np.ndarray | float,
    p_CO2_Pa: np.ndarray | float,
    ads: AdsorbentConfig | None = None,
) -> np.ndarray:
    """Wet-site Toth CO2 loading, mol/kg."""
    if ads is None:
        ads = AdsorbentConfig()
    return toth_loading(
        T_K=T_K,
        p_CO2_Pa=p_CO2_Pa,
        qmax=ads.wet_qmax_mol_kg,
        b0_Pa_inv=ads.wet_b0_Pa_inv,
        minus_dH_J_mol=ads.wet_minus_dH_J_mol,
        t0=ads.wet_t0,
        a=ads.wet_a,
        T0_K=ads.T0_K,
    )


def wadst_co2_loading(
    T_K: np.ndarray | float,
    p_CO2_Pa: np.ndarray | float,
    q_H2O_mol_kg: np.ndarray | float,
    ads: AdsorbentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted-average dual-site Toth CO2-H2O co-adsorption model.

    q_CO2 = (1 - exp(-A/q_H2O))*q_dry + exp(-A/q_H2O)*q_wet

    Returns
    -------
    tuple
        q_wadst, q_dry, q_wet, wet_fraction, dry_fraction
    """
    if ads is None:
        ads = AdsorbentConfig()

    q_dry = dry_toth_co2_loading(T_K, p_CO2_Pa, ads)
    q_wet = wet_site_toth_co2_loading(T_K, p_CO2_Pa, ads)

    qh = np.maximum(np.asarray(q_H2O_mol_kg, dtype=float), 1e-12)
    wet_fraction = np.clip(np.exp(-ads.wadst_A_mol_kg / qh), 0.0, 1.0)
    dry_fraction = 1.0 - wet_fraction

    q_wadst = dry_fraction * q_dry + wet_fraction * q_wet
    return np.maximum(q_wadst, 0.0), q_dry, q_wet, wet_fraction, dry_fraction


def equilibrium_loadings(
    C: np.ndarray,
    Tg: np.ndarray,
    ads: AdsorbentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate q*_CO2 and q*_H2O from gas concentrations and temperature.

    Parameters
    ----------
    C:
        Gas concentration array, shape (N, 4), in mol/m3 gas.
        Column order: CO2, H2O, N2, O2.
    Tg:
        Node temperature array, shape (N,), in K.
    ads:
        AdsorbentConfig. Default: Lewatit VP OC 1065.

    Returns
    -------
    tuple
        q_CO2_WADST, q_H2O_GAB, q_CO2_dry, q_CO2_wet_site, wet_fraction
    """
    if ads is None:
        ads = AdsorbentConfig()

    C_arr = np.asarray(C, dtype=float)
    T_arr = np.asarray(Tg, dtype=float)

    C_pos = np.maximum(C_arr, 1e-20)
    Ctot = np.maximum(np.sum(C_pos, axis=1), 1e-12)

    p_CO2 = C_pos[:, IDX["CO2"]] * R * T_arr
    p_H2O = C_pos[:, IDX["H2O"]] * R * T_arr
    p_sat = np.maximum(saturation_pressure_water_pa(T_arr), 1e-12)
    RH = np.clip(p_H2O / p_sat, 1e-12, 0.999999)

    q_H2O = gab_h2o_loading(T_arr, RH, ads)
    q_CO2, q_dry, q_wet, wet_fraction, dry_fraction = wadst_co2_loading(
        T_arr,
        p_CO2,
        q_H2O,
        ads,
    )

    return q_CO2, q_H2O, q_dry, q_wet, wet_fraction


def equilibrium_from_weather(
    T_K: np.ndarray | float,
    RH_frac: np.ndarray | float,
    P_Pa: np.ndarray | float,
    CO2_ppm: float = 400.0,
    ads: AdsorbentConfig | None = None,
) -> pd.DataFrame:
    """
    Compute feed partial pressures and GAB/WADST equilibrium loadings from weather.

    This is useful for checking the hourly NASA POWER input pipeline before the
    dynamic TVSA simulation is run.
    """
    if ads is None:
        ads = AdsorbentConfig()

    T = np.asarray(T_K, dtype=float)
    RH = np.clip(np.asarray(RH_frac, dtype=float), 0.0, 0.999999)
    P = np.asarray(P_Pa, dtype=float)

    p_sat = saturation_pressure_water_pa(T)
    p_H2O = np.minimum(RH * p_sat, 0.99 * P)
    p_CO2 = np.maximum(float(CO2_ppm) * 1e-6 * (P - p_H2O), 0.0)

    q_H2O = gab_h2o_loading(T, RH, ads)
    q_CO2, q_dry, q_wet, wet_fraction, dry_fraction = wadst_co2_loading(T, p_CO2, q_H2O, ads)

    return pd.DataFrame({
        "CO2_ppm": float(CO2_ppm),
        "T_K": np.ravel(T),
        "T_C": np.ravel(T - 273.15),
        "RH_frac": np.ravel(RH),
        "P_Pa": np.ravel(P),
        "p_sat_H2O_Pa": np.ravel(p_sat),
        "p_H2O_Pa": np.ravel(p_H2O),
        "p_CO2_Pa": np.ravel(p_CO2),
        "q_H2O_GAB_mol_kg": np.ravel(q_H2O),
        "q_CO2_dry_Toth_mol_kg": np.ravel(q_dry),
        "q_CO2_wet_site_Toth_mol_kg": np.ravel(q_wet),
        "WADST_wet_fraction": np.ravel(wet_fraction),
        "WADST_dry_fraction": np.ravel(dry_fraction),
        "q_CO2_WADST_mol_kg": np.ravel(q_CO2),
    })


# =============================================================================
# KINETICS AND ADSORPTION HEAT HELPERS
# =============================================================================

def ldf_rates(
    q_CO2_mol_kg: np.ndarray,
    q_H2O_mol_kg: np.ndarray,
    qeq_CO2_mol_kg: np.ndarray,
    qeq_H2O_mol_kg: np.ndarray,
    ads: AdsorbentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear-driving-force adsorption/desorption rates.

    dq_k/dt = k_LDF,k * (q*_k - q_k)
    """
    if ads is None:
        ads = AdsorbentConfig()

    dqdt_CO2 = ads.k_ldf_co2_s * (np.asarray(qeq_CO2_mol_kg) - np.asarray(q_CO2_mol_kg))
    dqdt_H2O = ads.k_ldf_h2o_s * (np.asarray(qeq_H2O_mol_kg) - np.asarray(q_H2O_mol_kg))
    return dqdt_CO2, dqdt_H2O


def adsorption_heat_source_W_per_kg_solid(
    dqdt_CO2_mol_kg_s: np.ndarray,
    dqdt_H2O_mol_kg_s: np.ndarray,
    ads: AdsorbentConfig | None = None,
) -> np.ndarray:
    """
    Heat source from adsorption/desorption per kg solid, W/kg solid.

    Positive means heat release into the solid phase.
    Negative means heat consumed by desorption.
    """
    if ads is None:
        ads = AdsorbentConfig()

    return (
        -ads.dH_ads_CO2_J_mol * np.asarray(dqdt_CO2_mol_kg_s)
        -ads.dH_ads_H2O_J_mol * np.asarray(dqdt_H2O_mol_kg_s)
    )


def effective_solid_heat_capacity_J_m3K(
    q_CO2_mol_kg: np.ndarray,
    q_H2O_mol_kg: np.ndarray,
    rho_bulk_kg_m3_bed: float | None = None,
    ads: AdsorbentConfig | None = None,
) -> np.ndarray:
    """
    Effective volumetric heat capacity of solid + adsorbed phase, J/m3bed/K.
    """
    if ads is None:
        ads = AdsorbentConfig()
    if rho_bulk_kg_m3_bed is None:
        rho_bulk_kg_m3_bed = ads.bulk_solid_density_kg_m3_bed

    qco2 = np.maximum(np.asarray(q_CO2_mol_kg, dtype=float), 0.0)
    qh2o = np.maximum(np.asarray(q_H2O_mol_kg, dtype=float), 0.0)

    cp_base = rho_bulk_kg_m3_bed * ads.cp_solid_J_kgK
    cp_ads = rho_bulk_kg_m3_bed * (
        qco2 * ads.cp_ads_CO2_J_molK
        + qh2o * ads.cp_ads_H2O_J_molK
    )
    return np.maximum(cp_base + cp_ads, 1e-9)


# =============================================================================
# SMALL STANDALONE CHECK
# =============================================================================

def self_check() -> pd.DataFrame:
    """Run a small equilibrium check at representative DAC conditions."""
    T_list_K = np.array([293.15, 298.15, 303.15, 313.15])
    RH_list = np.array([0.2, 0.5, 0.7, 0.9])
    P_list = np.full_like(T_list_K, 101_325.0)
    return equilibrium_from_weather(T_list_K, RH_list, P_list, CO2_ppm=400.0)


if __name__ == "__main__":
    pd.set_option("display.max_columns", 30)
    print(self_check().to_string(index=False))
