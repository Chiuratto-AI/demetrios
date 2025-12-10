# Darwin PBPK API Reference

Complete reference for programmatic access to Darwin PBPK functions and libraries.

## Core Modules

### 1. Simulation Module

Core simulation engine for PBPK models.

Function: simulate_1comp_iv
Purpose: Single-compartment IV bolus simulation
Signature: fn simulate_1comp_iv(dose: f64, vd: f64, cl: f64, duration: f64) -> Vec<(f64, f64)>

Parameters:
- dose (mg): Drug dose
- vd (L): Volume of distribution
- cl (mL/min): Clearance rate
- duration (hours): Simulation duration

Returns:
- Vector of (time, concentration) tuples

Example:


---

Function: simulate_3comp_iv
Purpose: Three-compartment IV bolus simulation
Signature: fn simulate_3comp_iv(dose: f64, vd_c: f64, vd_p1: f64, vd_p2: f64, cl: f64, duration: f64) -> Vec<(f64, f64, f64, f64)>

Parameters:
- dose (mg): Drug dose
- vd_c (L): Central compartment volume
- vd_p1 (L): Peripheral compartment 1 volume
- vd_p2 (L): Peripheral compartment 2 volume
- cl (mL/min): Central compartment clearance
- duration (hours): Simulation duration

Returns:
- Vector of (time, central_conc, peripheral1_conc, peripheral2_conc) tuples

Example:


---

Function: simulate_14comp_pbpk
Purpose: 14-compartment physiologically-based PBPK model
Signature: fn simulate_14comp_pbpk(dose: f64, route: str, duration: f64) -> PBPKResult

Parameters:
- dose (mg): Drug dose
- route (str): "oral" or "iv"
- duration (hours): Simulation duration

Returns:
- PBPKResult struct with organ-specific concentrations

PBPKResult fields:
- time: Vec<f64> - Time points (hours)
- blood: Vec<f64> - Blood concentration
- liver: Vec<f64> - Liver concentration
- kidney: Vec<f64> - Kidney concentration
- brain: Vec<f64> - Brain concentration
- heart: Vec<f64> - Heart concentration
- lung: Vec<f64> - Lung concentration
- adipose: Vec<f64> - Adipose concentration
- muscle: Vec<f64> - Muscle concentration
- bone: Vec<f64> - Bone concentration
- gi: Vec<f64> - GI tract concentration
- skin: Vec<f64> - Skin concentration
- other: Vec<f64> - Other tissues concentration
- urine: Vec<f64> - Urinary elimination (cumulative)

Example:


---

### 2. Validation Module

Validation against clinical data.

Function: calculate_fold_error
Purpose: Calculate fold error between predicted and observed concentrations
Signature: fn calculate_fold_error(predicted: f64, observed: f64) -> f64

Parameters:
- predicted (mg/L): Predicted concentration
- observed (mg/L): Observed concentration

Returns:
- Fold error: max(pred/obs, obs/pred)

Definition:
FE = max(predicted/observed, observed/predicted)
- FE = 1.0: Perfect prediction
- FE < 2.0: FDA acceptable
- FE < 1.25: Good prediction

Example:


---

Function: calculate_gmfe
Purpose: Geometric mean fold error across multiple observations
Signature: fn calculate_gmfe(predictions: Vec<f64>, observations: Vec<f64>) -> f64

Parameters:
- predictions: Vector of predicted concentrations
- observations: Vector of observed concentrations

Returns:
- Geometric mean fold error

Definition:
GMFE = exp( sum(log(FE_i)) / n )

Example:


---

Function: calculate_r_squared
Purpose: Calculate coefficient of determination
Signature: fn calculate_r_squared(predictions: Vec<f64>, observations: Vec<f64>) -> f64

Parameters:
- predictions: Vector of predicted concentrations
- observations: Vector of observed concentrations

Returns:
- R-squared value (0.0 to 1.0)

Definition:
R² = 1 - SS_residual / SS_total
- R² = 1.0: Perfect fit
- R² > 0.9: Good fit
- R² > 0.8: Acceptable fit

Example:


---

### 3. Drug Database Module

Access pre-configured drug parameters.

Function: get_drug_parameters
Purpose: Retrieve parameters for a specific drug
Signature: fn get_drug_parameters(drug_name: str) -> DrugParams

Parameters:
- drug_name (str): Drug name (e.g., "midazolam", "caffeine")

Returns:
- DrugParams struct

DrugParams fields:
- name: str
- dose_mg: f64
- vd_L: f64
- cl_ml_min: f64
- fu: f64 (fraction unbound)
- logp: f64 (lipophilicity)
- kp_liver: f64 (liver partition coefficient)
- t_half_h: f64 (half-life)
- cmax_obs_mg_L: f64 (observed Cmax)
- route: str ("iv" or "oral")
- model_type: str ("1comp", "3comp", "14comp")

Example:


---

Function: list_drugs
Purpose: Get list of all available drugs
Signature: fn list_drugs() -> Vec<str>

Returns:
- Vector of drug names

Example:


---

Function: add_drug_parameters
Purpose: Register custom drug parameters
Signature: fn add_drug_parameters(drug: DrugParams) -> bool

Parameters:
- drug: DrugParams struct with custom parameters

Returns:
- true if successful, false if drug already exists

Example:


---

### 4. Utilities Module

Helper functions for common tasks.

Function: convert_units
Purpose: Convert between common PK units
Signature: fn convert_units(value: f64, from_unit: str, to_unit: str) -> f64

Parameters:
- value: Numeric value
- from_unit: Source unit (e.g., "mg/kg", "mL/min/kg")
- to_unit: Target unit (e.g., "mg", "L/h")

Supported conversions:
- mg <-> micrograms (mcg)
- L <-> mL
- hours <-> minutes
- mg/L <-> ng/mL
- mL/min <-> L/h
- Dose scaling: mg/kg (weight-based)

Example:


---

Function: calculate_half_life
Purpose: Calculate drug half-life from clearance and volume
Signature: fn calculate_half_life(vd: f64, cl: f64) -> f64

Parameters:
- vd (L): Volume of distribution
- cl (mL/min): Clearance rate

Returns:
- Half-life in hours

Formula:
t1/2 = 0.693 * (Vd / CL)
where Vd in L and CL in mL/min

Example:


---

Function: calculate_steady_state_css
Purpose: Calculate steady-state concentration for continuous infusion
Signature: fn calculate_steady_state_css(infusion_rate: f64, cl: f64) -> f64

Parameters:
- infusion_rate (mg/h): Drug infusion rate
- cl (mL/min): Clearance rate

Returns:
- Steady-state concentration (mg/L)

Formula:
Css = (Infusion Rate) / (CL / 60)

Example:


---

### 5. Export Module

Generate output in various formats.

Function: export_to_csv
Purpose: Export simulation results to CSV format
Signature: fn export_to_csv(results: SimulationResult, filename: str) -> bool

Parameters:
- results: SimulationResult from simulation function
- filename: Output file path

Returns:
- true if successful

Example:


---

Function: export_to_json
Purpose: Export simulation results to JSON format
Signature: fn export_to_json(results: SimulationResult, filename: str) -> bool

Parameters:
- results: SimulationResult from simulation function
- filename: Output file path

Returns:
- true if successful

Example:


---

## Type Definitions

### DrugParams Struct



### SimulationResult Struct



### PBPKResult Struct



## Performance Notes

Simulation Speed (per simulation):
- 1-compartment: 0.04 ms
- 3-compartment: 0.12 ms
- 14-compartment: 0.36 ms

Memory Usage:
- 1-compartment: ~2.1 MB
- 3-compartment: ~2.3 MB
- 14-compartment: ~2.8 MB

Comparison to Python implementations:
- 30-50× faster than Python (SciPy)
- Compiled binary vs interpreted
- No external dependencies at runtime

## Error Handling

All functions return errors in standardized format:
- Negative doses: Error("Invalid dose")
- Invalid routes: Error("Unknown route")
- Missing drugs: Error("Drug not found")
- File I/O errors: Error("Cannot write file")

Example:


## Integration Examples

### Example 1: Basic Simulation Loop



### Example 2: Multi-Drug Validation



---

## Linking to Library

To use Darwin PBPK in your Demetrios code:

1. Import the module:


2. Call functions:


3. Access types:


---

See RESEARCH_PAPER.md for mathematical details and validation methodology.
