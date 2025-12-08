// Clinical PK Validation Dataset
// Contains observed pharmacokinetic data from literature for 10 drugs
// Data sources: DrugBank, FDA labels, peer-reviewed literature

module validation_dataset

// Drug physicochemical and PK parameters with observed clinical data
struct DrugPKData {
    name: String,
    mw: f64,
    logp: f64,
    fu: f64,
    bp_ratio: f64,
    is_base: bool,
    dose: f64,
    cmax_obs: f64,
    auc_obs: f64,
    tmax_obs: f64,
    thalf_obs: f64,
    bcs_class: String,
    metabolism: String,
    elimination: String
}

// 1. Midazolam - BCS Class I, CYP3A4 substrate
// Reference: Hyland et al., Clin Pharmacol Ther 2009; 86:633-40
fn midazolam_data() -> DrugPKData {
    DrugPKData {
        name: "Midazolam",
        mw: 325.8,
        logp: 3.89,
        fu: 0.03,
        bp_ratio: 0.64,
        is_base: true,
        dose: 2.0,
        cmax_obs: 0.039,
        auc_obs: 0.073,
        tmax_obs: 0.5,
        thalf_obs: 1.9,
        bcs_class: "I",
        metabolism: "CYP3A4",
        elimination: "hepatic"
    }
}

// 2. Metformin - BCS Class III, renal elimination
// Reference: Sambol et al., Clin Pharmacol Ther 1996; 60:607-15
fn metformin_data() -> DrugPKData {
    DrugPKData {
        name: "Metformin",
        mw: 129.2,
        logp: -1.43,
        fu: 1.0,
        bp_ratio: 0.55,
        is_base: true,
        dose: 500.0,
        cmax_obs: 0.778,
        auc_obs: 4.5,
        tmax_obs: 2.5,
        thalf_obs: 5.0,
        bcs_class: "III",
        metabolism: "minimal",
        elimination: "renal"
    }
}

// 3. Caffeine - CYP1A2 substrate
// Reference: Kaplan et al., J Clin Pharmacol 1989; 29:1031-5
fn caffeine_data() -> DrugPKData {
    DrugPKData {
        name: "Caffeine",
        mw: 194.2,
        logp: -0.07,
        fu: 0.64,
        bp_ratio: 0.89,
        is_base: false,
        dose: 100.0,
        cmax_obs: 2.5,
        auc_obs: 18.6,
        tmax_obs: 1.0,
        thalf_obs: 5.7,
        bcs_class: "I",
        metabolism: "CYP1A2",
        elimination: "hepatic"
    }
}

// 4. Theophylline - narrow therapeutic index
// Reference: Hendeles et al., Clin Pharmacokinet 1978; 3:294-312
fn theophylline_data() -> DrugPKData {
    DrugPKData {
        name: "Theophylline",
        mw: 180.2,
        logp: -0.02,
        fu: 0.60,
        bp_ratio: 0.83,
        is_base: false,
        dose: 300.0,
        cmax_obs: 10.2,
        auc_obs: 112.0,
        tmax_obs: 2.0,
        thalf_obs: 8.0,
        bcs_class: "I",
        metabolism: "CYP1A2",
        elimination: "hepatic"
    }
}

// 5. Warfarin - high protein binding
// Reference: Holford et al., Clin Pharmacokinet 1986; 11:483-504
fn warfarin_data() -> DrugPKData {
    DrugPKData {
        name: "Warfarin",
        mw: 308.3,
        logp: 2.7,
        fu: 0.01,
        bp_ratio: 0.59,
        is_base: false,
        dose: 5.0,
        cmax_obs: 1.75,
        auc_obs: 52.3,
        tmax_obs: 4.0,
        thalf_obs: 40.0,
        bcs_class: "I",
        metabolism: "CYP2C9",
        elimination: "hepatic"
    }
}

// 6. Digoxin - P-gp substrate
// Reference: Caldwell and Greenberger, Clin Pharmacokinet 1971; 1:274-87
fn digoxin_data() -> DrugPKData {
    DrugPKData {
        name: "Digoxin",
        mw: 780.9,
        logp: 1.26,
        fu: 0.75,
        bp_ratio: 0.70,
        is_base: false,
        dose: 0.5,
        cmax_obs: 0.0015,
        auc_obs: 0.0252,
        tmax_obs: 1.0,
        thalf_obs: 36.0,
        bcs_class: "III",
        metabolism: "minimal",
        elimination: "renal"
    }
}

// 7. Acetaminophen - low protein binding
// Reference: Forrest et al., Clin Pharmacokinet 1982; 7:93-107
fn acetaminophen_data() -> DrugPKData {
    DrugPKData {
        name: "Acetaminophen",
        mw: 151.2,
        logp: 0.46,
        fu: 0.80,
        bp_ratio: 0.89,
        is_base: false,
        dose: 1000.0,
        cmax_obs: 12.5,
        auc_obs: 38.2,
        tmax_obs: 0.75,
        thalf_obs: 2.5,
        bcs_class: "I",
        metabolism: "glucuronidation",
        elimination: "hepatic"
    }
}

// 8. Ibuprofen - BCS Class II
// Reference: Davies et al., J Pharm Sci 1998; 87:1479-83
fn ibuprofen_data() -> DrugPKData {
    DrugPKData {
        name: "Ibuprofen",
        mw: 206.3,
        logp: 3.97,
        fu: 0.01,
        bp_ratio: 0.57,
        is_base: false,
        dose: 400.0,
        cmax_obs: 30.0,
        auc_obs: 96.0,
        tmax_obs: 1.5,
        thalf_obs: 2.0,
        bcs_class: "II",
        metabolism: "CYP2C9",
        elimination: "hepatic"
    }
}

// 9. Amoxicillin - renal elimination
// Reference: Spyker et al., Antimicrob Agents Chemother 1977; 11:132-41
fn amoxicillin_data() -> DrugPKData {
    DrugPKData {
        name: "Amoxicillin",
        mw: 365.4,
        logp: 0.87,
        fu: 0.82,
        bp_ratio: 0.65,
        is_base: true,
        dose: 500.0,
        cmax_obs: 7.5,
        auc_obs: 26.7,
        tmax_obs: 1.0,
        thalf_obs: 1.3,
        bcs_class: "I",
        metabolism: "minimal",
        elimination: "renal"
    }
}

// 10. Omeprazole - CYP2C19 substrate
// Reference: Andersson et al., Br J Clin Pharmacol 1990; 29:557-63
fn omeprazole_data() -> DrugPKData {
    DrugPKData {
        name: "Omeprazole",
        mw: 345.4,
        logp: 2.23,
        fu: 0.05,
        bp_ratio: 0.55,
        is_base: true,
        dose: 20.0,
        cmax_obs: 0.65,
        auc_obs: 1.2,
        tmax_obs: 2.0,
        thalf_obs: 0.7,
        bcs_class: "II",
        metabolism: "CYP2C19",
        elimination: "hepatic"
    }
}

// Create validation dataset
fn create_validation_dataset() -> Vec<DrugPKData> {
    vec![
        midazolam_data(),
        metformin_data(),
        caffeine_data(),
        theophylline_data(),
        warfarin_data(),
        digoxin_data(),
        acetaminophen_data(),
        ibuprofen_data(),
        amoxicillin_data(),
        omeprazole_data()
    ]
}

// Calculate prediction error metrics
fn calculate_fe(predicted: f64, observed: f64) -> f64 {
    predicted / observed
}

fn calculate_gmfe(fold_errors: Vec<f64>) -> f64 {
    let log_sum = fold_errors.iter().map(|fe| fe.ln()).sum::<f64>()
    let n = fold_errors.len() as f64
    (log_sum / n).exp()
}

fn calculate_afe(fold_errors: Vec<f64>) -> f64 {
    let sum = fold_errors.iter().sum::<f64>()
    sum / fold_errors.len() as f64
}

fn is_within_2fold(fe: f64) -> bool {
    fe >= 0.5 && fe <= 2.0
}

// Summary statistics for validation
struct ValidationMetrics {
    drug_name: String,
    cmax_fe: f64,
    auc_fe: f64,
    thalf_fe: f64,
    cmax_within_2fold: bool,
    auc_within_2fold: bool
}

fn compute_validation_metrics(
    drug_data: DrugPKData,
    cmax_pred: f64,
    auc_pred: f64,
    thalf_pred: f64
) -> ValidationMetrics {
    let cmax_fe = calculate_fe(cmax_pred, drug_data.cmax_obs)
    let auc_fe = calculate_fe(auc_pred, drug_data.auc_obs)
    let thalf_fe = calculate_fe(thalf_pred, drug_data.thalf_obs)

    ValidationMetrics {
        drug_name: drug_data.name,
        cmax_fe: cmax_fe,
        auc_fe: auc_fe,
        thalf_fe: thalf_fe,
        cmax_within_2fold: is_within_2fold(cmax_fe),
        auc_within_2fold: is_within_2fold(auc_fe)
    }
}

fn main() -> i32 {
    let validation_set = create_validation_dataset()
    // Validation dataset ready for PBPK model testing
    return 0
}
