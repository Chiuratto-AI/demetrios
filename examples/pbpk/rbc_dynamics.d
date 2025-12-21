// ===========================================================================
// RBC DYNAMICS WITH HEMATOPOIESIS FEEDBACK
// ===========================================================================
//
// Port of Darwin PBPK's rbc_dynamics_integrated.jl to Demetrios.
//
// This module implements a closed-loop erythropoiesis model with:
// - EPO-regulated RBC production
// - Bone marrow progenitor dynamics (BFU-E, CFU-E, reticulocytes)
// - Age-structured RBC population
// - Drug binding to RBCs with competition
// - Hematological toxicity assessment
//
// Critical for modeling:
// - Anemia/polycythemia drug effects
// - Hemoglobin-bound drug transport
// - ESA (erythropoiesis-stimulating agent) dosing
// - Chemotherapy-induced anemia
//
// References:
// - Bélair et al. (1995) - Age-structured RBC model
// - Krzyzanski et al. (2006) - EPO-erythropoiesis PD
// - Friberg et al. (2002) - Myelosuppression model
//
// Author: Demetrios Chiuratto Agourakis
// Date: December 2025
// Version: 1.0.0
// ===========================================================================

module rbc_dynamics

import darwin_pbpk_14comp::{Drug, Patient}

// =============================================================================
// UNIT DEFINITIONS
// =============================================================================

unit cells_per_L       // RBC count
unit g_per_dL          // Hemoglobin concentration
unit mIU_per_mL        // EPO concentration
unit days              // Time for slow dynamics
unit per_day           // Rate constants

// Derived
unit cells_per_L_per_day = cells_per_L / days

// =============================================================================
// PHYSIOLOGICAL CONSTANTS
// =============================================================================

/// Reference hematological values (healthy adult)
pub const HEMATOLOGY_REF = struct {
    // RBC counts
    rbc_count_male: cells_per_L = 5.0e12,
    rbc_count_female: cells_per_L = 4.5e12,

    // Hemoglobin
    hgb_male: g_per_dL = 15.5,
    hgb_female: g_per_dL = 13.5,

    // Hematocrit
    hct_male: f64 = 0.45,
    hct_female: f64 = 0.40,

    // RBC lifespan
    rbc_lifespan: days = 120.0,

    // Mean corpuscular hemoglobin
    mch: f64 = 29.0,  // pg

    // EPO
    epo_baseline: mIU_per_mL = 10.0,
    epo_half_life: h = 6.0,

    // Reticulocyte fraction
    retic_fraction: f64 = 0.01,  // 1% normally

    // Blood volume
    blood_volume_male: L = 5.0,
    blood_volume_female: L = 4.5,
};

/// Bone marrow kinetic parameters
pub const MARROW_KINETICS = struct {
    // Progenitor transit times
    bfu_e_transit: days = 7.0,      // BFU-E to CFU-E
    cfu_e_transit: days = 3.0,      // CFU-E to proerythroblast
    erythroblast_transit: days = 5.0, // Erythroblast to reticulocyte
    retic_maturation: days = 3.0,   // Reticulocyte in marrow

    // Amplification factors
    bfu_e_amplification: f64 = 8.0,    // Divisions
    cfu_e_amplification: f64 = 4.0,
    erythroblast_amplification: f64 = 16.0,

    // Apoptosis rates (baseline)
    bfu_e_apoptosis: per_day = 0.1,
    cfu_e_apoptosis: per_day = 0.15,

    // EPO sensitivity
    epo_ec50: mIU_per_mL = 20.0,
    epo_emax: f64 = 5.0,  // Maximum fold increase
};

// =============================================================================
// REFINEMENT TYPES
// =============================================================================

// Hemoglobin must be physiologically valid
type ValidHgb = { hgb: g_per_dL | hgb > 3.0 && hgb < 25.0 }

// Hematocrit 0-1 (typically 0.20-0.60)
type ValidHct = { hct: f64 | hct > 0.10 && hct < 0.70 }

// EPO concentration (can be very high with ESA therapy)
type ValidEPO = { epo: mIU_per_mL | epo > 0.0 && epo < 100000.0 }

// RBC count
type ValidRBC = { rbc: cells_per_L | rbc > 1.0e11 && rbc < 1.0e13 }

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// EPO regulation state
pub struct EPOState {
    /// Current EPO concentration
    pub concentration: Knowledge[ValidEPO, epsilon >= 0.80],

    /// Endogenous production rate
    pub production_rate: mIU_per_mL_per_day,

    /// Exogenous EPO (if any)
    pub exogenous: mIU_per_mL,

    /// Renal function factor (0-1, reduced in CKD)
    pub renal_factor: f64,
}

/// Bone marrow progenitor state
pub struct ProgenitorState {
    /// BFU-E (burst-forming unit erythroid)
    pub bfu_e: cells_per_L,

    /// CFU-E (colony-forming unit erythroid)
    pub cfu_e: cells_per_L,

    /// Proerythroblasts
    pub proerythroblast: cells_per_L,

    /// Erythroblasts (combined stages)
    pub erythroblast: cells_per_L,

    /// Marrow reticulocytes
    pub marrow_retic: cells_per_L,
}

/// Age-structured RBC population
/// Uses discrete age bins (e.g., 10-day bins over 120-day lifespan)
pub struct RBCPopulation {
    /// RBC count in each age bin
    pub age_bins: [cells_per_L; 12],  // 12 bins × 10 days = 120 days

    /// Total circulating RBCs
    pub total_rbc: ValidRBC,

    /// Circulating reticulocytes
    pub reticulocytes: cells_per_L,

    /// Mean RBC age
    pub mean_age: days,
}

/// Drug binding to RBCs
pub struct RBCDrugBinding {
    /// Drug name
    pub drug: string,

    /// Blood:plasma ratio
    pub bp_ratio: f64,

    /// Fraction bound to RBCs (vs plasma)
    pub f_rbc: f64,

    /// Binding site on RBC
    pub binding_site: RBCBindingSite,

    /// Binding affinity (if saturable)
    pub kd: Option<uM>,

    /// Maximum binding capacity
    pub bmax: Option<mg_per_L>,
}

pub enum RBCBindingSite {
    Hemoglobin,
    CarbonAnhydrase,
    Band3Protein,
    Membrane,
    Cytosol,
}

/// Complete erythropoiesis state
pub struct ErythropoiesisState {
    /// EPO regulation
    pub epo: EPOState,

    /// Bone marrow progenitors
    pub progenitors: ProgenitorState,

    /// Circulating RBCs
    pub rbc_population: RBCPopulation,

    /// Hemoglobin concentration
    pub hemoglobin: Knowledge[ValidHgb, epsilon >= 0.90],

    /// Hematocrit
    pub hematocrit: Knowledge[ValidHct, epsilon >= 0.90],

    /// Drug binding state
    pub drug_binding: Option<RBCDrugBinding>,
}

/// Hematological toxicity assessment
pub enum HematoxicityGrade {
    None,      // Hgb >= 10 g/dL
    Grade1,    // Hgb 10.0-9.0 g/dL
    Grade2,    // Hgb 8.0-9.0 g/dL
    Grade3,    // Hgb 6.5-8.0 g/dL
    Grade4,    // Hgb < 6.5 g/dL (transfusion required)
}

// =============================================================================
// EPO REGULATION FUNCTIONS
// =============================================================================

/// Kidney oxygen sensing and EPO production
/// EPO production increases exponentially as Hgb decreases
pub fn epo_production_rate(
    hemoglobin: ValidHgb,
    renal_factor: f64,
    baseline_hgb: g_per_dL,
) -> mIU_per_mL_per_day {
    // Exponential relationship with hypoxia
    let hgb_ratio = hemoglobin / baseline_hgb;
    let hypoxia_stimulus = (-2.5 * (hgb_ratio - 1.0)).exp();

    // Scale by renal function (CKD reduces EPO production)
    let rate = 100.0 * hypoxia_stimulus * renal_factor;

    rate.clamp(1.0, 10000.0)
}

/// EPO effect on erythropoiesis (Emax model)
pub fn epo_effect(
    epo: ValidEPO,
    ec50: mIU_per_mL,
    emax: f64,
) -> f64 {
    // Fold increase in progenitor proliferation
    1.0 + (emax - 1.0) * epo / (ec50 + epo)
}

// =============================================================================
// PROGENITOR DYNAMICS
// =============================================================================

/// Compute progenitor derivatives (ODE right-hand side)
pub fn progenitor_ode(
    state: &ProgenitorState,
    epo: ValidEPO,
    drug_effect: f64,  // 0-1, fraction of normal (1 = no effect)
) -> ProgenitorDerivatives {
    let epo_stim = epo_effect(epo, MARROW_KINETICS.epo_ec50, MARROW_KINETICS.epo_emax);

    // BFU-E dynamics: production - transit - apoptosis
    let bfu_e_production = 1.0e10 * epo_stim * drug_effect;  // cells/L/day
    let bfu_e_transit_out = state.bfu_e / MARROW_KINETICS.bfu_e_transit;
    let bfu_e_apoptosis = state.bfu_e * MARROW_KINETICS.bfu_e_apoptosis;
    let d_bfu_e = bfu_e_production - bfu_e_transit_out - bfu_e_apoptosis;

    // CFU-E dynamics: amplified input from BFU-E - transit
    let cfu_e_input = bfu_e_transit_out * MARROW_KINETICS.bfu_e_amplification * epo_stim;
    let cfu_e_transit_out = state.cfu_e / MARROW_KINETICS.cfu_e_transit;
    let cfu_e_apoptosis = state.cfu_e * MARROW_KINETICS.cfu_e_apoptosis * (1.0 / drug_effect);
    let d_cfu_e = cfu_e_input - cfu_e_transit_out - cfu_e_apoptosis;

    // Proerythroblast dynamics
    let proeryth_input = cfu_e_transit_out * MARROW_KINETICS.cfu_e_amplification;
    let proeryth_transit = state.proerythroblast / 1.0;  // 1 day transit
    let d_proerythroblast = proeryth_input - proeryth_transit;

    // Erythroblast dynamics (multiple stages combined)
    let eryth_input = proeryth_transit;
    let eryth_transit = state.erythroblast / MARROW_KINETICS.erythroblast_transit;
    let d_erythroblast = eryth_input - eryth_transit;

    // Marrow reticulocyte dynamics
    let retic_input = eryth_transit * MARROW_KINETICS.erythroblast_amplification;
    let retic_release = state.marrow_retic / MARROW_KINETICS.retic_maturation;
    let d_marrow_retic = retic_input - retic_release;

    ProgenitorDerivatives {
        d_bfu_e,
        d_cfu_e,
        d_proerythroblast,
        d_erythroblast,
        d_marrow_retic,
        retic_release,
    }
}

pub struct ProgenitorDerivatives {
    pub d_bfu_e: cells_per_L_per_day,
    pub d_cfu_e: cells_per_L_per_day,
    pub d_proerythroblast: cells_per_L_per_day,
    pub d_erythroblast: cells_per_L_per_day,
    pub d_marrow_retic: cells_per_L_per_day,
    pub retic_release: cells_per_L_per_day,  // Output to circulation
}

// =============================================================================
// AGE-STRUCTURED RBC DYNAMICS
// =============================================================================

/// Update age-structured RBC population
/// Uses McKendrick-von Foerster equation discretized into age bins
pub fn update_rbc_population(
    population: &mut RBCPopulation,
    retic_input: cells_per_L_per_day,
    drug_induced_hemolysis: f64,  // Fraction hemolyzed per day
    dt: days,
) {
    // Shift cells to older bins (aging)
    for i in (1..12).rev() {
        population.age_bins[i] = population.age_bins[i-1];
    }

    // New reticulocytes enter youngest bin
    population.age_bins[0] = retic_input * dt;

    // Apply senescent removal (oldest bin dies)
    population.age_bins[11] = 0.0;

    // Apply drug-induced hemolysis uniformly
    if drug_induced_hemolysis > 0.0 {
        for i in 0..12 {
            population.age_bins[i] *= (1.0 - drug_induced_hemolysis * dt);
        }
    }

    // Update totals
    population.total_rbc = population.age_bins.iter().sum();
    population.reticulocytes = population.age_bins[0] + population.age_bins[1];

    // Calculate mean age
    let mut age_sum = 0.0;
    for i in 0..12 {
        let bin_age = (i as f64 + 0.5) * 10.0;  // Center of 10-day bin
        age_sum += population.age_bins[i] * bin_age;
    }
    population.mean_age = age_sum / population.total_rbc;
}

// =============================================================================
// DRUG-RBC INTERACTIONS
// =============================================================================

/// Calculate drug concentration in RBCs
pub fn drug_concentration_in_rbc(
    c_plasma: mg_per_L,
    bp_ratio: f64,
    hematocrit: ValidHct,
) -> mg_per_L {
    // C_blood = C_plasma * BP_ratio
    // C_RBC = (C_blood - C_plasma * (1-Hct)) / Hct
    let c_blood = c_plasma * bp_ratio;
    let c_rbc = (c_blood - c_plasma * (1.0 - hematocrit)) / hematocrit;
    c_rbc.max(0.0)
}

/// Saturable hemoglobin binding (e.g., carbon monoxide, oxygen)
pub fn hemoglobin_saturation(
    ligand_conc: mg_per_L,
    kd: mg_per_L,
    n_hill: f64,  // Hill coefficient for cooperativity
) -> f64 {
    // Hill equation: Y = [L]^n / (Kd + [L]^n)
    let l_n = ligand_conc.pow(n_hill);
    l_n / (kd + l_n)
}

// =============================================================================
// TOXICITY ASSESSMENT
// =============================================================================

/// Classify anemia severity per CTCAE
pub fn classify_anemia(hemoglobin: ValidHgb) -> HematoxicityGrade {
    if hemoglobin >= 10.0 {
        HematoxicityGrade::None
    } else if hemoglobin >= 9.0 {
        HematoxicityGrade::Grade1
    } else if hemoglobin >= 8.0 {
        HematoxicityGrade::Grade2
    } else if hemoglobin >= 6.5 {
        HematoxicityGrade::Grade3
    } else {
        HematoxicityGrade::Grade4
    }
}

/// Calculate hemoglobin from RBC count and MCH
pub fn calculate_hemoglobin(rbc_count: ValidRBC, mch: f64) -> ValidHgb {
    // Hgb (g/dL) = RBC (cells/L) × MCH (pg) × 1e-13
    let hgb = rbc_count * mch * 1.0e-13;
    hgb.clamp(3.0, 25.0)
}

/// Calculate hematocrit from RBC and MCV
pub fn calculate_hematocrit(rbc_count: ValidRBC, mcv: f64) -> ValidHct {
    // Hct = RBC (cells/L) × MCV (fL) × 1e-15
    let hct = rbc_count * mcv * 1.0e-15;
    hct.clamp(0.10, 0.70)
}

// =============================================================================
// COMPLETE ERYTHROPOIESIS SIMULATION
// =============================================================================

/// Simulate erythropoiesis over time with drug effects
pub fn simulate_erythropoiesis(
    patient: &Patient,
    drug: Option<&Drug>,
    drug_myelotoxicity: f64,  // 0-1, 1 = no effect
    duration: days,
    dt: days,
) -> ErythropoiesisTimeCourse with Alloc {
    // Initialize state based on patient
    let is_male = patient.sex == Sex::Male;
    let baseline_hgb = if is_male { HEMATOLOGY_REF.hgb_male } else { HEMATOLOGY_REF.hgb_female };
    let baseline_rbc = if is_male { HEMATOLOGY_REF.rbc_count_male } else { HEMATOLOGY_REF.rbc_count_female };

    let mut state = ErythropoiesisState {
        epo: EPOState {
            concentration: Knowledge::new(HEMATOLOGY_REF.epo_baseline, 0.85, Provenance::source("baseline")),
            production_rate: 100.0,
            exogenous: 0.0,
            renal_factor: 1.0,
        },
        progenitors: ProgenitorState {
            bfu_e: 1.0e10,
            cfu_e: 2.0e10,
            proerythroblast: 5.0e10,
            erythroblast: 1.0e11,
            marrow_retic: 5.0e10,
        },
        rbc_population: RBCPopulation {
            age_bins: [baseline_rbc / 12.0; 12],
            total_rbc: baseline_rbc,
            reticulocytes: baseline_rbc * HEMATOLOGY_REF.retic_fraction,
            mean_age: 60.0,
        },
        hemoglobin: Knowledge::new(baseline_hgb, 0.95, Provenance::source("baseline")),
        hematocrit: Knowledge::new(if is_male { HEMATOLOGY_REF.hct_male } else { HEMATOLOGY_REF.hct_female }, 0.95, Provenance::source("baseline")),
        drug_binding: None,
    };

    let mut time_course: Vec<(days, ValidHgb, ValidRBC, HematoxicityGrade)> = vec![];

    let n_steps = (duration / dt) as i32;

    for step in 0..=n_steps {
        let t = step as f64 * dt;

        // Record current state
        let grade = classify_anemia(state.hemoglobin.value);
        time_course.push((t, state.hemoglobin.value, state.rbc_population.total_rbc, grade));

        if step < n_steps {
            // Update EPO (fast dynamics, but we integrate daily)
            let epo_prod = epo_production_rate(
                state.hemoglobin.value,
                state.epo.renal_factor,
                baseline_hgb,
            );
            state.epo.concentration = Knowledge::new(
                epo_prod * 0.1,  // Steady-state approximation
                0.80,
                Provenance::derived("epo_regulation"),
            );

            // Progenitor dynamics with drug effect
            let derivs = progenitor_ode(
                &state.progenitors,
                state.epo.concentration.value,
                drug_myelotoxicity,
            );

            // Update progenitors
            state.progenitors.bfu_e += derivs.d_bfu_e * dt;
            state.progenitors.cfu_e += derivs.d_cfu_e * dt;
            state.progenitors.proerythroblast += derivs.d_proerythroblast * dt;
            state.progenitors.erythroblast += derivs.d_erythroblast * dt;
            state.progenitors.marrow_retic += derivs.d_marrow_retic * dt;

            // Clamp to non-negative
            state.progenitors.bfu_e = state.progenitors.bfu_e.max(0.0);
            state.progenitors.cfu_e = state.progenitors.cfu_e.max(0.0);
            state.progenitors.erythroblast = state.progenitors.erythroblast.max(0.0);
            state.progenitors.marrow_retic = state.progenitors.marrow_retic.max(0.0);

            // Update RBC population
            update_rbc_population(
                &mut state.rbc_population,
                derivs.retic_release,
                0.0,  // No drug-induced hemolysis in this example
                dt,
            );

            // Update Hgb and Hct
            state.hemoglobin = Knowledge::new(
                calculate_hemoglobin(state.rbc_population.total_rbc, HEMATOLOGY_REF.mch),
                0.90,
                Provenance::derived("rbc_to_hgb"),
            );
            state.hematocrit = Knowledge::new(
                calculate_hematocrit(state.rbc_population.total_rbc, 90.0),  // MCV ~90 fL
                0.90,
                Provenance::derived("rbc_to_hct"),
            );
        }
    }

    ErythropoiesisTimeCourse {
        times: time_course.iter().map(|(t, _, _, _)| *t).collect(),
        hemoglobin: time_course.iter().map(|(_, h, _, _)| *h).collect(),
        rbc_count: time_course.iter().map(|(_, _, r, _)| *r).collect(),
        toxicity_grades: time_course.iter().map(|(_, _, _, g)| g.clone()).collect(),
        nadir_hgb: time_course.iter().map(|(_, h, _, _)| *h).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        nadir_time: time_course.iter().min_by(|(_, h1, _, _), (_, h2, _, _)| h1.partial_cmp(h2).unwrap()).map(|(t, _, _, _)| *t).unwrap(),
        recovery_time: None,  // Would calculate when Hgb returns to baseline
    }
}

pub struct ErythropoiesisTimeCourse {
    pub times: Vec<days>,
    pub hemoglobin: Vec<ValidHgb>,
    pub rbc_count: Vec<ValidRBC>,
    pub toxicity_grades: Vec<HematoxicityGrade>,
    pub nadir_hgb: ValidHgb,
    pub nadir_time: days,
    pub recovery_time: Option<days>,
}

// =============================================================================
// MAIN EXAMPLE
// =============================================================================

pub fn main() with IO, Alloc {
    println("=== RBC Dynamics with Hematopoiesis Feedback ===")
    println("Closed-loop erythropoiesis model in Demetrios")
    println("")

    // Create patient
    let patient = Patient {
        id: "SUBJ001",
        weight: 70.0 : kg,
        age: 50.0,
        sex: Sex::Male,
        egfr: 90.0 : mL_per_min,
        liver_function: LiverFunction::Normal,
    };

    println("Patient: Male, 70 kg, age 50")
    println("Baseline Hgb: {} g/dL", HEMATOLOGY_REF.hgb_male)
    println("")

    // Simulate chemotherapy-induced anemia
    println("Simulating 30-day chemotherapy with myelotoxicity...")
    println("")

    let result = simulate_erythropoiesis(
        &patient,
        None,
        0.5,    // 50% reduction in progenitor production
        30.0,   // 30 days
        1.0,    // Daily steps
    );

    println("Results:")
    println("  Nadir Hgb: {:.1} g/dL at day {:.0}", result.nadir_hgb, result.nadir_time)
    println("  Nadir Grade: {:?}", classify_anemia(result.nadir_hgb))
    println("")

    println("Day-by-day Hgb (first 10 days):")
    for i in 0..10.min(result.times.len()) {
        let grade = &result.toxicity_grades[i];
        println("  Day {}: {:.1} g/dL [{:?}]", result.times[i], result.hemoglobin[i], grade)
    }
    println("")

    println("Key features demonstrated:")
    println("  - EPO-mediated feedback regulation")
    println("  - Bone marrow progenitor kinetics")
    println("  - Age-structured RBC population")
    println("  - CTCAE toxicity grading")
    println("")
    println("=== Simulation Complete ===")
}
