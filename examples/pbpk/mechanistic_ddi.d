// ===========================================================================
// MECHANISTIC DDI MODULE - Drug-Drug Interaction Modeling in Demetrios
// ===========================================================================
//
// Port of Darwin PBPK's MechanisticDDI.jl to Demetrios, demonstrating:
// - Unit-safe IVIVE calculations
// - Epistemic types for uncertainty tracking
// - Refinement types for kinetic parameter validation
// - Effect system for probabilistic DDI prediction
//
// Features:
// - In vitro to in vivo extrapolation (IVIVE)
// - Competitive, noncompetitive, mixed inhibition
// - Mechanism-based inhibition (MBI / time-dependent)
// - CYP enzyme induction (PXR/CAR pathway)
// - Transporter DDI (OATP, P-gp, BCRP)
// - Monte Carlo uncertainty quantification
//
// References:
// - FDA Guidance: Clinical DDI Studies (2020)
// - EMA Guideline: Investigation of DDI (2012)
// - Rowland-Yeo et al. (2011) IVIVE scaling factors
//
// Author: Demetrios Chiuratto Agourakis
// Date: December 2025
// Version: 1.0.0
// ===========================================================================

module mechanistic_ddi

import darwin_pbpk_14comp::{Drug, Patient, PBPK14Params}

// =============================================================================
// UNIT DEFINITIONS FOR DDI
// =============================================================================

unit uM               // micromolar concentration
unit pmol             // picomole
unit mg_protein       // mg microsomal protein
unit pmol_per_min     // velocity
unit per_min          // rate constant (1/min)
unit per_h            // rate constant (1/h)
unit cells_per_mL     // hepatocyte concentration
unit g_liver          // grams of liver

// Derived units
unit pmol_per_min_per_pmol_CYP = pmol_per_min / pmol    // Intrinsic clearance per CYP
unit mL_per_min_per_mg = mL / min / mg_protein          // CLint per mg protein
unit uL_per_min_per_pmol = uL / min / pmol              // CLint per pmol CYP

// =============================================================================
// REFINEMENT TYPES FOR KINETIC PARAMETERS
// =============================================================================

// Ki must be positive and physiologically reasonable (nM to mM range)
type ValidKi = { ki: uM | ki > 0.0001 && ki < 10000.0 }

// Km must be positive (typically nM to mM)
type ValidKm = { km: uM | km > 0.001 && km < 5000.0 }

// Vmax must be positive
type ValidVmax = { vmax: pmol_per_min | vmax > 0.0 && vmax < 100000.0 }

// Fraction metabolized 0-1
type FractionMetabolized = { fm: f64 | fm >= 0.0 && fm <= 1.0 }

// Inactivation rate for MBI (typically 0.001 - 1.0 min⁻¹)
type ValidKinact = { kinact: per_min | kinact > 0.0 && kinact < 5.0 }

// Fold induction (1 = no change, typically 1-40x)
type FoldInduction = { fold: f64 | fold >= 1.0 && fold <= 100.0 }

// AUC ratio (DDI magnitude)
type AUCRatio = { ratio: f64 | ratio > 0.0 && ratio < 100.0 }

// =============================================================================
// CONSTANTS - PHYSIOLOGICAL SCALING FACTORS
// =============================================================================

/// Physiological parameters for IVIVE (70kg reference adult)
pub const PHYSIOLOGY = struct {
    // Liver parameters
    liver_weight: g_liver = 1800.0,
    liver_blood_flow: mL_per_min = 1450.0,       // Qh
    hepatocellularity: f64 = 120e6,               // HPGL (cells/g liver)
    mppgl: mg_protein = 40.0,                     // mg microsomal protein per g liver

    // Intestinal parameters
    gut_weight: g = 1650.0,
    gut_blood_flow: mL_per_min = 300.0,
    enterocyte_number: f64 = 3.0e11,
    gut_volume: mL = 250.0,                       // For [I]g calculation

    // Blood/plasma
    plasma_volume: mL = 3000.0,
    blood_volume: mL = 5000.0,
};

/// CYP enzyme abundance in liver (pmol/mg protein)
/// Reference: Achour et al. (2014)
pub const CYP_ABUNDANCE = struct {
    CYP3A4:  pmol_per_mg = 137.0,
    CYP3A5:  pmol_per_mg = 12.0,
    CYP2D6:  pmol_per_mg = 10.0,
    CYP2C9:  pmol_per_mg = 96.0,
    CYP2C19: pmol_per_mg = 14.0,
    CYP2C8:  pmol_per_mg = 24.0,
    CYP1A2:  pmol_per_mg = 52.0,
    CYP2B6:  pmol_per_mg = 17.0,
    CYP2E1:  pmol_per_mg = 49.0,
};

/// CYP enzyme half-lives for MBI recovery (hours)
pub const CYP_HALFLIFE = struct {
    CYP3A4:  h = 36.0,
    CYP2D6:  h = 51.0,
    CYP2C9:  h = 104.0,
    CYP2C19: h = 26.0,
    CYP1A2:  h = 39.0,
    CYP2C8:  h = 23.0,
    CYP2B6:  h = 32.0,
};

/// Transporter abundance (pmol/mg protein)
/// Reference: Prasad et al. (2016)
pub const TRANSPORTER_ABUNDANCE = struct {
    OATP1B1: pmol_per_mg = 2.94,
    OATP1B3: pmol_per_mg = 1.35,
    OATP2B1: pmol_per_mg = 0.89,
    P_gp:    pmol_per_mg = 2.50,
    BCRP:    pmol_per_mg = 1.80,
    MRP2:    pmol_per_mg = 3.20,
    OCT1:    pmol_per_mg = 1.50,
};

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/// CYP enzyme identifiers
pub enum CYPEnzyme {
    CYP3A4,
    CYP3A5,
    CYP2D6,
    CYP2C9,
    CYP2C19,
    CYP2C8,
    CYP1A2,
    CYP2B6,
    CYP2E1,
}

/// Transporter identifiers
pub enum Transporter {
    OATP1B1,
    OATP1B3,
    OATP2B1,
    P_gp,
    BCRP,
    MRP2,
    OCT1,
    MATE1,
}

/// DDI inhibition mechanism
pub enum InhibitionType {
    Competitive,
    Noncompetitive,
    Uncompetitive,
    Mixed { alpha: f64 },           // alpha = Ki'/Ki
    MechanismBased { kinact: per_min, ki_mbi: uM },
}

/// DDI severity classification (FDA guidance)
pub enum DDISeverity {
    None,           // AUC ratio < 1.25
    Weak,           // 1.25 <= AUC ratio < 2
    Moderate,       // 2 <= AUC ratio < 5
    Strong,         // AUC ratio >= 5
    Contraindicated,// AUC ratio >= 10 (clinical concern)
}

/// IVIVE parameters for a compound
pub struct IVIVEParams {
    /// Fraction unbound in plasma
    pub fu_plasma: Knowledge[f64, epsilon >= 0.85],

    /// Fraction unbound in blood
    pub fu_blood: Knowledge[f64, epsilon >= 0.80],

    /// Fraction unbound in microsomes
    pub fu_mic: Knowledge[f64, epsilon >= 0.75],

    /// Fraction unbound in hepatocytes
    pub fu_hep: Knowledge[f64, epsilon >= 0.75],

    /// Blood:plasma ratio
    pub rb: Knowledge[f64, epsilon >= 0.80],

    /// LogP (lipophilicity)
    pub logp: f64,

    /// Molecular weight
    pub mw: f64,
}

/// Enzyme kinetics for a substrate-enzyme pair
pub struct EnzymeKinetics {
    /// Target enzyme
    pub enzyme: CYPEnzyme,

    /// Michaelis constant
    pub km: Knowledge[ValidKm, epsilon >= 0.80],

    /// Maximum velocity
    pub vmax: Knowledge[ValidVmax, epsilon >= 0.75],

    /// Intrinsic clearance (CLint = Vmax/Km)
    pub clint: mL_per_min_per_mg,

    /// Fraction metabolized by this pathway
    pub fm: Knowledge[FractionMetabolized, epsilon >= 0.70],

    /// Inter-system extrapolation factor
    pub isef: f64,
}

/// Inhibitor parameters
pub struct InhibitorParams {
    /// Drug name
    pub name: string,

    /// Mechanism of inhibition
    pub inhibition_type: InhibitionType,

    /// Inhibition constant
    pub ki: Knowledge[ValidKi, epsilon >= 0.80],

    /// Target enzymes
    pub target_enzymes: Vec<CYPEnzyme>,

    /// Target transporters
    pub target_transporters: Vec<Transporter>,

    /// Unbound Cmax at steady state
    pub cmax_unbound: Knowledge[uM, epsilon >= 0.85],

    /// Gut concentration ([I]g = Dose / 250mL)
    pub i_gut: uM,

    /// Unbound hepatic inlet concentration
    pub i_hepatic_inlet: uM,
}

/// Inducer parameters
pub struct InducerParams {
    /// Drug name
    pub name: string,

    /// Maximum fold induction per enzyme
    pub emax: Map<CYPEnzyme, FoldInduction>,

    /// EC50 per enzyme
    pub ec50: Map<CYPEnzyme, uM>,

    /// Induction pathway
    pub pathway: InductionPathway,

    /// Unbound Cmax
    pub cmax_unbound: Knowledge[uM, epsilon >= 0.85],

    /// Onset delay (days)
    pub onset_delay: f64,
}

pub enum InductionPathway {
    PXR,    // Pregnane X receptor (CYP3A4, CYP2C9)
    CAR,    // Constitutive androstane receptor (CYP2B6)
    AhR,    // Aryl hydrocarbon receptor (CYP1A2)
}

/// DDI prediction result with full epistemic tracking
pub struct DDIPredictionResult {
    /// Perpetrator drug
    pub perpetrator: string,

    /// Victim drug
    pub victim: string,

    /// Predicted AUC ratio with uncertainty
    pub auc_ratio: Knowledge[AUCRatio, epsilon >= 0.70],

    /// Predicted Cmax ratio
    pub cmax_ratio: Knowledge[f64, epsilon >= 0.70],

    /// Change in oral clearance
    pub cl_ratio: f64,

    /// Primary mechanism
    pub mechanism: InhibitionType,

    /// DDI severity classification
    pub severity: DDISeverity,

    /// FDA/EMA classification string
    pub regulatory_class: string,

    /// Monte Carlo uncertainty bounds
    pub uncertainty: DDIUncertainty,

    /// Per-enzyme contributions
    pub cyp_contributions: Map<CYPEnzyme, f64>,

    /// Per-transporter contributions
    pub transporter_contributions: Map<Transporter, f64>,

    /// Full provenance chain
    pub provenance: Provenance,
}

/// Monte Carlo uncertainty quantification
pub struct DDIUncertainty {
    pub mean_ratio: f64,
    pub sd_ratio: f64,
    pub ci_90_lower: f64,
    pub ci_90_upper: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub p_exceeds_2fold: f64,
    pub p_exceeds_5fold: f64,
    pub n_simulations: i32,
}

// =============================================================================
// IVIVE FUNCTIONS
// =============================================================================

/// Calculate fraction unbound in microsomes from plasma binding
/// Reference: Austin et al. (2002)
pub fn calculate_fu_mic(
    fu_plasma: f64,
    logp: f64,
) -> Knowledge[f64, epsilon >= 0.75] {
    // Austin equation
    let log_fu_mic_fu_inc = 0.53 * logp - 0.49;
    let fu_mic_fu_inc = 10.0.pow(log_fu_mic_fu_inc);
    let fu_mic = 1.0 / (1.0 + fu_mic_fu_inc * (1.0/fu_plasma - 1.0));

    Knowledge::new(
        value: fu_mic.clamp(0.001, 1.0),
        confidence: 0.75,
        provenance: Provenance::derived("austin_fu_mic"),
    )
}

/// Scale intrinsic clearance from in vitro to in vivo
/// CLint,in_vivo = CLint,in_vitro × MPPGL × Liver_weight × ISEF
pub fn scale_clint_ivive(
    clint_in_vitro: mL_per_min_per_mg,
    fu_mic: f64,
    fu_plasma: f64,
    isef: f64,
) -> Knowledge[L_per_h, epsilon >= 0.70] with Alloc {
    let mppgl = PHYSIOLOGY.mppgl;
    let liver_weight = PHYSIOLOGY.liver_weight;

    // Correct for microsomal binding
    let clint_unbound = clint_in_vitro / fu_mic;

    // Scale to whole liver
    let clint_liver = clint_unbound * mppgl * liver_weight * isef;

    // Convert mL/min to L/h
    let clint_lph = clint_liver * 0.06;

    // Apply plasma binding
    let clint_plasma = clint_lph * fu_plasma;

    Knowledge::new(
        value: clint_plasma,
        confidence: 0.70,
        provenance: Provenance::derived("ivive_clint"),
    )
}

/// Calculate hepatic clearance using well-stirred model
pub fn hepatic_clearance_well_stirred(
    clint: L_per_h,
    qh: L_per_h,
    fu_blood: f64,
    rb: f64,
) -> L_per_h {
    let fu_b = fu_blood / rb;
    let clh = (qh * fu_b * clint) / (qh + fu_b * clint);
    clh
}

// =============================================================================
// INHIBITION FUNCTIONS
// =============================================================================

/// Competitive inhibition: enzyme binds either substrate or inhibitor
/// R = 1 + [I]u / Ki
pub fn competitive_inhibition(
    i_unbound: uM,
    ki: ValidKi,
) -> f64 {
    1.0 + i_unbound / ki
}

/// Noncompetitive inhibition: inhibitor binds to allosteric site
/// R = 1 + [I]u / Ki
pub fn noncompetitive_inhibition(
    i_unbound: uM,
    ki: ValidKi,
) -> f64 {
    1.0 + i_unbound / ki
}

/// Mixed inhibition: affects both binding and catalysis
/// R = (1 + [I]u/Ki) * (1 + [I]u/(alpha*Ki)) / (1 + [I]u/(alpha*Ki))
pub fn mixed_inhibition(
    i_unbound: uM,
    ki: ValidKi,
    alpha: f64,
) -> f64 {
    let ki_prime = ki * alpha;
    (1.0 + i_unbound / ki) / (1.0 + i_unbound / ki_prime)
}

/// Mechanism-based inhibition (time-dependent, irreversible)
/// R = 1 + (kinact * [I]u) / (kdeg * (KI + [I]u))
///
/// This is the CRITICAL DDI mechanism - accounts for enzyme inactivation
/// that persists after inhibitor washout.
pub fn mechanism_based_inhibition(
    i_unbound: uM,
    kinact: ValidKinact,
    ki_mbi: uM,
    kdeg: per_h,           // Enzyme degradation rate = ln(2)/t½
) -> Knowledge[f64, epsilon >= 0.65] with Prob {
    let kdeg_per_min = kdeg / 60.0;
    let r_mbi = 1.0 + (kinact * i_unbound) / (kdeg_per_min * (ki_mbi + i_unbound));

    Knowledge::new(
        value: r_mbi,
        confidence: 0.65,  // MBI has higher uncertainty
        provenance: Provenance::derived("mechanism_based_inhibition"),
    )
}

/// Multi-inhibitor saturation: when multiple inhibitors compete for same enzyme
pub fn multi_inhibitor_saturation(
    inhibitors: &[InhibitorParams],
    enzyme: CYPEnzyme,
) -> f64 {
    // Sum of all inhibition terms
    let mut total_r = 1.0;

    for inh in inhibitors {
        if inh.target_enzymes.contains(&enzyme) {
            let r_i = match inh.inhibition_type {
                InhibitionType::Competitive => competitive_inhibition(inh.cmax_unbound.value, inh.ki.value),
                InhibitionType::Noncompetitive => noncompetitive_inhibition(inh.cmax_unbound.value, inh.ki.value),
                InhibitionType::Mixed { alpha } => mixed_inhibition(inh.cmax_unbound.value, inh.ki.value, alpha),
                InhibitionType::MechanismBased { kinact, ki_mbi } => {
                    let kdeg = 0.693 / get_cyp_halflife(enzyme);
                    mechanism_based_inhibition(inh.cmax_unbound.value, kinact, ki_mbi, kdeg).value
                }
            };
            total_r += r_i - 1.0;  // Add incremental inhibition
        }
    }

    total_r
}

fn get_cyp_halflife(enzyme: CYPEnzyme) -> h {
    match enzyme {
        CYPEnzyme::CYP3A4 => CYP_HALFLIFE.CYP3A4,
        CYPEnzyme::CYP2D6 => CYP_HALFLIFE.CYP2D6,
        CYPEnzyme::CYP2C9 => CYP_HALFLIFE.CYP2C9,
        CYPEnzyme::CYP2C19 => CYP_HALFLIFE.CYP2C19,
        CYPEnzyme::CYP1A2 => CYP_HALFLIFE.CYP1A2,
        CYPEnzyme::CYP2C8 => CYP_HALFLIFE.CYP2C8,
        CYPEnzyme::CYP2B6 => CYP_HALFLIFE.CYP2B6,
        _ => 36.0 : h,  // Default
    }
}

// =============================================================================
// INDUCTION FUNCTIONS
// =============================================================================

/// Calculate induction magnitude using Emax model
/// Fold_induction = 1 + (Emax - 1) * [I]u / (EC50 + [I]u)
pub fn calculate_induction_magnitude(
    i_unbound: uM,
    emax: FoldInduction,
    ec50: uM,
) -> FoldInduction {
    let fold = 1.0 + (emax - 1.0) * i_unbound / (ec50 + i_unbound);
    fold
}

/// Net effect when both inhibition and induction occur
/// This is common for drugs like ritonavir (inhibits + induces CYP3A4)
pub fn net_effect_inhibition_induction(
    r_inhibition: f64,
    fold_induction: f64,
) -> Knowledge[f64, epsilon >= 0.60] {
    // Net effect: induction reduces enzyme, inhibition blocks remaining
    let net = r_inhibition / fold_induction;

    Knowledge::new(
        value: net,
        confidence: 0.60,  // High uncertainty with dual mechanisms
        provenance: Provenance::derived("net_inhibition_induction"),
    )
}

// =============================================================================
// TRANSPORTER DDI FUNCTIONS
// =============================================================================

/// OATP inhibition (hepatic uptake transporter)
/// Critical for statins, methotrexate, repaglinide
pub fn oatp_inhibition(
    i_unbound: uM,
    ki: ValidKi,
) -> f64 {
    // R = 1 + [I]u,inlet / Ki
    1.0 + i_unbound / ki
}

/// P-glycoprotein inhibition (efflux transporter)
/// Affects oral absorption and tissue distribution
pub fn pgp_inhibition(
    i_gut: uM,
    ic50: uM,
) -> f64 {
    // Gut lumen concentration is relevant
    1.0 + i_gut / ic50
}

/// BCRP inhibition
pub fn bcrp_inhibition(
    i_gut: uM,
    ic50: uM,
) -> f64 {
    1.0 + i_gut / ic50
}

/// Combined CYP + transporter DDI prediction
pub fn combined_cyp_transporter_ddi(
    cyp_r: f64,
    transporter_r: f64,
    fm_cyp: f64,
    fm_transporter: f64,
) -> Knowledge[f64, epsilon >= 0.65] {
    // AUC ratio = 1 / ((fm_CYP/R_CYP) + (fm_transporter/R_transporter) + (1-fm_CYP-fm_transporter))
    let denominator = (fm_cyp / cyp_r) + (fm_transporter / transporter_r) + (1.0 - fm_cyp - fm_transporter);
    let auc_ratio = 1.0 / denominator;

    Knowledge::new(
        value: auc_ratio,
        confidence: 0.65,
        provenance: Provenance::derived("combined_cyp_transporter_ddi"),
    )
}

// =============================================================================
// MONTE CARLO DDI PREDICTION
// =============================================================================

/// Monte Carlo DDI prediction with uncertainty propagation
pub fn monte_carlo_ddi_prediction(
    victim: &EnzymeKinetics,
    perpetrator: &InhibitorParams,
    n_simulations: i32,
) -> DDIUncertainty with Prob, Alloc {
    let mut auc_ratios: Vec<f64> = vec![];

    for _ in 0..n_simulations {
        // Sample from parameter distributions
        let ki_sample = sample(LogNormal(
            perpetrator.ki.value.ln(),
            0.3,  // CV ~30%
        ));

        let fm_sample = sample(Beta(
            victim.fm.value * 10.0,
            (1.0 - victim.fm.value) * 10.0,
        ));

        let i_sample = sample(LogNormal(
            perpetrator.cmax_unbound.value.ln(),
            0.25,  // CV ~25%
        ));

        // Calculate DDI for this sample
        let r = competitive_inhibition(i_sample, ki_sample);
        let auc_ratio = 1.0 / ((fm_sample / r) + (1.0 - fm_sample));

        auc_ratios.push(auc_ratio);
    }

    // Calculate statistics
    let mean_ratio = auc_ratios.mean();
    let sd_ratio = auc_ratios.std();

    auc_ratios.sort();
    let n = n_simulations as usize;

    DDIUncertainty {
        mean_ratio,
        sd_ratio,
        ci_90_lower: auc_ratios[(0.05 * n as f64) as usize],
        ci_90_upper: auc_ratios[(0.95 * n as f64) as usize],
        ci_95_lower: auc_ratios[(0.025 * n as f64) as usize],
        ci_95_upper: auc_ratios[(0.975 * n as f64) as usize],
        p_exceeds_2fold: auc_ratios.iter().filter(|r| **r >= 2.0).count() as f64 / n as f64,
        p_exceeds_5fold: auc_ratios.iter().filter(|r| **r >= 5.0).count() as f64 / n as f64,
        n_simulations,
    }
}

/// Classify DDI severity per FDA guidance
pub fn classify_ddi_severity(auc_ratio: f64) -> DDISeverity {
    if auc_ratio < 1.25 {
        DDISeverity::None
    } else if auc_ratio < 2.0 {
        DDISeverity::Weak
    } else if auc_ratio < 5.0 {
        DDISeverity::Moderate
    } else if auc_ratio < 10.0 {
        DDISeverity::Strong
    } else {
        DDISeverity::Contraindicated
    }
}

// =============================================================================
// MAIN DDI PREDICTION FUNCTION
// =============================================================================

/// Complete mechanistic DDI prediction with epistemic tracking
pub fn predict_ddi_mechanistic(
    victim_kinetics: &[EnzymeKinetics],
    perpetrator: &InhibitorParams,
    n_mc_samples: i32,
) -> DDIPredictionResult with Prob, Alloc, IO {
    // Calculate R values for each enzyme
    let mut cyp_contributions: Map<CYPEnzyme, f64> = Map::new();
    let mut total_inhibited_clearance = 0.0;
    let mut total_fm = 0.0;

    for kinetics in victim_kinetics {
        total_fm += kinetics.fm.value;

        if perpetrator.target_enzymes.contains(&kinetics.enzyme) {
            let r = match perpetrator.inhibition_type {
                InhibitionType::Competitive =>
                    competitive_inhibition(perpetrator.cmax_unbound.value, perpetrator.ki.value),
                InhibitionType::MechanismBased { kinact, ki_mbi } => {
                    let kdeg = 0.693 / get_cyp_halflife(kinetics.enzyme);
                    mechanism_based_inhibition(perpetrator.cmax_unbound.value, kinact, ki_mbi, kdeg).value
                }
                _ => competitive_inhibition(perpetrator.cmax_unbound.value, perpetrator.ki.value),
            };

            let inhibited_contribution = kinetics.fm.value / r;
            total_inhibited_clearance += inhibited_contribution;
            cyp_contributions.insert(kinetics.enzyme, 1.0 / r);
        } else {
            total_inhibited_clearance += kinetics.fm.value;
            cyp_contributions.insert(kinetics.enzyme, 1.0);
        }
    }

    // Non-CYP clearance
    let non_cyp_fm = 1.0 - total_fm;
    total_inhibited_clearance += non_cyp_fm;

    // AUC ratio = 1 / total_inhibited_clearance
    let auc_ratio = 1.0 / total_inhibited_clearance;

    // Monte Carlo for uncertainty
    let uncertainty = monte_carlo_ddi_prediction(
        &victim_kinetics[0],  // Primary pathway
        perpetrator,
        n_mc_samples,
    );

    // Classify severity
    let severity = classify_ddi_severity(auc_ratio);
    let regulatory_class = match severity {
        DDISeverity::None => "No clinically significant interaction expected",
        DDISeverity::Weak => "Weak inhibitor - consider dose adjustment",
        DDISeverity::Moderate => "Moderate inhibitor - dose reduction recommended",
        DDISeverity::Strong => "Strong inhibitor - use lowest effective dose or avoid",
        DDISeverity::Contraindicated => "CONTRAINDICATED - do not co-administer",
    };

    // Determine confidence from inputs
    let base_conf = perpetrator.ki.confidence
        .min(perpetrator.cmax_unbound.confidence)
        .min(victim_kinetics[0].fm.confidence);

    DDIPredictionResult {
        perpetrator: perpetrator.name.clone(),
        victim: "substrate".to_string(),
        auc_ratio: Knowledge::new(
            value: auc_ratio,
            confidence: base_conf * 0.90,
            provenance: Provenance::derived("mechanistic_ddi_prediction"),
        ),
        cmax_ratio: Knowledge::new(
            value: auc_ratio.sqrt(),  // Approximation
            confidence: base_conf * 0.85,
            provenance: Provenance::derived("cmax_ratio_estimate"),
        ),
        cl_ratio: 1.0 / auc_ratio,
        mechanism: perpetrator.inhibition_type.clone(),
        severity,
        regulatory_class: regulatory_class.to_string(),
        uncertainty,
        cyp_contributions,
        transporter_contributions: Map::new(),
        provenance: Provenance::merged([
            perpetrator.provenance(),
            Provenance::source("fda_ddi_guidance_2020"),
        ]),
    }
}

// =============================================================================
// EXAMPLE: KETOCONAZOLE + MIDAZOLAM DDI
// =============================================================================

pub fn main() with IO, Alloc, Prob {
    println("=== Mechanistic DDI Prediction in Demetrios ===")
    println("Example: Ketoconazole (inhibitor) + Midazolam (victim)")
    println("")

    // Define victim substrate kinetics (Midazolam via CYP3A4)
    let midazolam_kinetics = [
        EnzymeKinetics {
            enzyme: CYPEnzyme::CYP3A4,
            km: Knowledge::new(4.0 : uM, 0.90, Provenance::source("literature")),
            vmax: Knowledge::new(1500.0 : pmol_per_min, 0.85, Provenance::source("hep_incubation")),
            clint: 375.0 : mL_per_min_per_mg,
            fm: Knowledge::new(0.95, 0.90, Provenance::source("clinical_ddi")),
            isef: 1.0,
        },
    ];

    // Define perpetrator inhibitor (Ketoconazole)
    let ketoconazole = InhibitorParams {
        name: "Ketoconazole",
        inhibition_type: InhibitionType::Competitive,
        ki: Knowledge::new(0.015 : uM, 0.92, Provenance::source("enzyme_assay")),
        target_enzymes: vec![CYPEnzyme::CYP3A4, CYPEnzyme::CYP3A5],
        target_transporters: vec![],
        cmax_unbound: Knowledge::new(0.024 : uM, 0.88, Provenance::source("clinical_pk")),
        i_gut: 800.0 : uM,  // 200mg / 250mL
        i_hepatic_inlet: 0.5 : uM,
    };

    // Predict DDI
    let result = predict_ddi_mechanistic(&midazolam_kinetics, &ketoconazole, 1000);

    // Display results
    println("Perpetrator: {}", result.perpetrator)
    println("Mechanism: Competitive CYP3A4 inhibition")
    println("")
    println("Predicted DDI:")
    println("  AUC ratio: {:.2}x (ε={:.2})", result.auc_ratio.value, result.auc_ratio.confidence)
    println("  Cmax ratio: {:.2}x", result.cmax_ratio.value)
    println("  CL ratio: {:.2}", result.cl_ratio)
    println("")
    println("Severity: {:?}", result.severity)
    println("FDA Classification: {}", result.regulatory_class)
    println("")
    println("Monte Carlo Uncertainty (n={}):", result.uncertainty.n_simulations)
    println("  Mean AUC ratio: {:.2}", result.uncertainty.mean_ratio)
    println("  90% CI: [{:.2}, {:.2}]", result.uncertainty.ci_90_lower, result.uncertainty.ci_90_upper)
    println("  P(AUC > 2-fold): {:.1}%", result.uncertainty.p_exceeds_2fold * 100.0)
    println("  P(AUC > 5-fold): {:.1}%", result.uncertainty.p_exceeds_5fold * 100.0)
    println("")
    println("=== DDI Prediction Complete ===")
}
