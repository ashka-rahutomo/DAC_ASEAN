# DAC_ASEAN
This repository contains a full-Python dynamic 1D finite-volume TVSA Direct Air Capture model for Lewatit VP OC 1065. The model is intended for process-level simulation, surrogate dataset generation, and comparison against Aspen Adsorption-style TVSA results.

## 1. Main Limitations
- The cycle still uses fixed adsorption, desorption, cooling, and repressurization durations for surrogate generation. Desorption and cooling cut-off indicators are reported as diagnostics, not as the default stopping criteria.
- Product gas is reported as wet pre-compression gas containing CO2, H2O, N2, and O2. Dry-basis or post-condenser CO2 purity must be calculated separately.
- N2 and O2 are treated as inert gases.
- Thermophysical properties are simplified using constant or approximate values.
- The heat-exchanger and jacket model is an explicit Python approximation and may differ from Aspen Adsorption's internal heat-transfer implementation.
- Numerical results should be checked using mass-balance residuals, energy-balance diagnostics, cooling-temperature flags, desorption cut-off flags, and cycle-to-cycle convergence indicators.
- Experimental validation is not included in this script.

## 2. Included Equations and Models
The script includes the following governing equations and sub-models:
- Ideal gas law
- Humid-air feed composition from temperature, relative humidity, pressure, and CO2 ppm
- Water saturation pressure correlation
- WADST CO2-H2O co-adsorption isotherm
- GAB H2O adsorption isotherm
- Linear Driving Force kinetics for CO2 and H2O
- 1D gas-phase species balance
- Axial convection
- Component-wise axial dispersion
- Gas-solid adsorption/desorption source term
- Total bed voidage gas accumulation correction
- Gas-phase energy balance
- Solid-phase energy balance
- Wall energy balance
- Jacket-fluid energy balance
- Adsorption/desorption heat effect
- Adsorbed-phase heat-capacity contribution
- Ergun pressure-drop equation
- Fan electricity calculation
- Vacuum pump electricity calculation
- Repressurization electricity calculation
- Heat-exchanger bed-side heat duty
- Heat-exchanger utility-side heat duty
- Cooling duty and chiller electricity
- Cycle-level CO2 and H2O product accounting
- Wet product composition accounting
- Mass-balance residual calculation
- Energy-balance proxy diagnostic
- Desorption cut-off diagnostic
- Cooling-temperature diagnostic
- Cycle-to-cycle convergence diagnostic
- Productivity and specific energy consumption calculation
