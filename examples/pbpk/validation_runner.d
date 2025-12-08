// Darwin PBPK Validation Runner
// 10 drugs - comparing predicted vs observed

struct DrugParams {
    mw: f64,
    logp: f64,
    fu: f64,
    bp_ratio: f64,
    is_base: bool
}

struct PKParams {
    cl: f64@L_per_h,
    vd: f64@L,
    ka: f64,
    f_oral: f64
}

fn calc_cmax(dose: f64@mg, vd: f64@L, f: f64, bp: f64) -> f64@mg_per_L {
    let amt = dose * f
    let c = amt / vd
    return c / bp
}

fn calc_auc(dose: f64@mg, cl: f64@L_per_h, f: f64) -> f64@mg_h_per_L {
    return dose * f / cl
}

fn calc_thalf(vd: f64@L, cl: f64@L_per_h) -> f64@h {
    return 0.693 * vd / cl
}

fn fold_error(pred: f64, obs: f64) -> f64 {
    let ratio = pred / obs
    if ratio >= 1.0 {
        return ratio
    }
    return obs / pred
}

fn is_within_2fold(pred: f64, obs: f64) -> bool {
    let fe = fold_error(pred, obs)
    if fe <= 2.0 {
        return true
    }
    return false
}

fn main() -> i32 {
    println("==========================================")
    println("  DARWIN PBPK VALIDATION - DEMETRIOS")
    println("  10 Drug Clinical Dataset")
    println("==========================================")
    println("")

    let total_cmax = 0
    let total_auc = 0
    let total_thalf = 0
    let pass_cmax = 0
    let pass_auc = 0
    let pass_thalf = 0

    // 1. MIDAZOLAM
    println("1. MIDAZOLAM (CYP3A4)")
    let dose1: f64@mg = 2.0
    let vd1: f64@L = 77.0
    let cl1: f64@L_per_h = 27.0
    let f1 = 0.44
    let bp1 = 0.64
    let cmax1 = calc_cmax(dose1, vd1, f1, bp1)
    let auc1 = calc_auc(dose1, cl1, f1)
    let thalf1 = calc_thalf(vd1, cl1)
    let cmax1_obs = 0.039
    let auc1_obs = 0.073
    let thalf1_obs = 1.9
    println("  Cmax pred/obs:")
    println(cmax1)
    println(cmax1_obs)
    println("  FE Cmax:")
    println(fold_error(cmax1, cmax1_obs))
    println("")

    // 2. METFORMIN
    println("2. METFORMIN (Renal)")
    let dose2: f64@mg = 500.0
    let vd2: f64@L = 400.0
    let cl2: f64@L_per_h = 350.0
    let f2 = 0.50
    let bp2 = 0.55
    let cmax2 = calc_cmax(dose2, vd2, f2, bp2)
    let auc2 = calc_auc(dose2, cl2, f2)
    let thalf2 = calc_thalf(vd2, cl2)
    let cmax2_obs = 0.778
    let auc2_obs = 4.5
    let thalf2_obs = 5.0
    println("  Cmax pred/obs:")
    println(cmax2)
    println(cmax2_obs)
    println("  FE Cmax:")
    println(fold_error(cmax2, cmax2_obs))
    println("")

    // 3. CAFFEINE
    println("3. CAFFEINE (CYP1A2)")
    let dose3: f64@mg = 100.0
    let vd3: f64@L = 40.0
    let cl3: f64@L_per_h = 6.0
    let f3 = 0.97
    let bp3 = 0.89
    let cmax3 = calc_cmax(dose3, vd3, f3, bp3)
    let auc3 = calc_auc(dose3, cl3, f3)
    let thalf3 = calc_thalf(vd3, cl3)
    let cmax3_obs = 2.5
    let auc3_obs = 18.6
    let thalf3_obs = 5.7
    println("  Cmax pred/obs:")
    println(cmax3)
    println(cmax3_obs)
    println("  FE Cmax:")
    println(fold_error(cmax3, cmax3_obs))
    println("")

    // 4. THEOPHYLLINE
    println("4. THEOPHYLLINE (NTI)")
    let dose4: f64@mg = 300.0
    let vd4: f64@L = 35.0
    let cl4: f64@L_per_h = 2.8
    let f4 = 0.96
    let bp4 = 0.83
    let cmax4 = calc_cmax(dose4, vd4, f4, bp4)
    let auc4 = calc_auc(dose4, cl4, f4)
    let thalf4 = calc_thalf(vd4, cl4)
    let cmax4_obs = 10.2
    let auc4_obs = 112.0
    let thalf4_obs = 8.0
    println("  Cmax pred/obs:")
    println(cmax4)
    println(cmax4_obs)
    println("  FE Cmax:")
    println(fold_error(cmax4, cmax4_obs))
    println("")

    // 5. WARFARIN
    println("5. WARFARIN (High PPB)")
    let dose5: f64@mg = 5.0
    let vd5: f64@L = 10.0
    let cl5: f64@L_per_h = 0.2
    let f5 = 0.93
    let bp5 = 0.58
    let cmax5 = calc_cmax(dose5, vd5, f5, bp5)
    let auc5 = calc_auc(dose5, cl5, f5)
    let thalf5 = calc_thalf(vd5, cl5)
    let cmax5_obs = 0.9
    let auc5_obs = 40.0
    let thalf5_obs = 37.0
    println("  Cmax pred/obs:")
    println(cmax5)
    println(cmax5_obs)
    println("  FE Cmax:")
    println(fold_error(cmax5, cmax5_obs))
    println("")

    // 6. DIGOXIN
    println("6. DIGOXIN (P-gp)")
    let dose6: f64@mg = 0.25
    let vd6: f64@L = 500.0
    let cl6: f64@L_per_h = 10.0
    let f6 = 0.70
    let bp6 = 0.95
    let cmax6 = calc_cmax(dose6, vd6, f6, bp6)
    let auc6 = calc_auc(dose6, cl6, f6)
    let thalf6 = calc_thalf(vd6, cl6)
    let cmax6_obs = 0.0018
    let auc6_obs = 0.045
    let thalf6_obs = 36.0
    println("  Cmax pred/obs:")
    println(cmax6)
    println(cmax6_obs)
    println("  FE Cmax:")
    println(fold_error(cmax6, cmax6_obs))
    println("")

    // 7. ACETAMINOPHEN
    println("7. ACETAMINOPHEN (Low PPB)")
    let dose7: f64@mg = 1000.0
    let vd7: f64@L = 60.0
    let cl7: f64@L_per_h = 20.0
    let f7 = 0.88
    let bp7 = 1.0
    let cmax7 = calc_cmax(dose7, vd7, f7, bp7)
    let auc7 = calc_auc(dose7, cl7, f7)
    let thalf7 = calc_thalf(vd7, cl7)
    let cmax7_obs = 15.0
    let auc7_obs = 50.0
    let thalf7_obs = 2.5
    println("  Cmax pred/obs:")
    println(cmax7)
    println(cmax7_obs)
    println("  FE Cmax:")
    println(fold_error(cmax7, cmax7_obs))
    println("")

    // 8. IBUPROFEN
    println("8. IBUPROFEN (BCS II)")
    let dose8: f64@mg = 400.0
    let vd8: f64@L = 10.0
    let cl8: f64@L_per_h = 3.5
    let f8 = 0.80
    let bp8 = 0.55
    let cmax8 = calc_cmax(dose8, vd8, f8, bp8)
    let auc8 = calc_auc(dose8, cl8, f8)
    let thalf8 = calc_thalf(vd8, cl8)
    let cmax8_obs = 25.0
    let auc8_obs = 110.0
    let thalf8_obs = 2.1
    println("  Cmax pred/obs:")
    println(cmax8)
    println(cmax8_obs)
    println("  FE Cmax:")
    println(fold_error(cmax8, cmax8_obs))
    println("")

    // 9. AMOXICILLIN
    println("9. AMOXICILLIN (Renal)")
    let dose9: f64@mg = 500.0
    let vd9: f64@L = 20.0
    let cl9: f64@L_per_h = 15.0
    let f9 = 0.80
    let bp9 = 0.83
    let cmax9 = calc_cmax(dose9, vd9, f9, bp9)
    let auc9 = calc_auc(dose9, cl9, f9)
    let thalf9 = calc_thalf(vd9, cl9)
    let cmax9_obs = 8.0
    let auc9_obs = 18.0
    let thalf9_obs = 1.3
    println("  Cmax pred/obs:")
    println(cmax9)
    println(cmax9_obs)
    println("  FE Cmax:")
    println(fold_error(cmax9, cmax9_obs))
    println("")

    // 10. OMEPRAZOLE
    println("10. OMEPRAZOLE (CYP2C19)")
    let dose10: f64@mg = 20.0
    let vd10: f64@L = 25.0
    let cl10: f64@L_per_h = 30.0
    let f10 = 0.40
    let bp10 = 0.55
    let cmax10 = calc_cmax(dose10, vd10, f10, bp10)
    let auc10 = calc_auc(dose10, cl10, f10)
    let thalf10 = calc_thalf(vd10, cl10)
    let cmax10_obs = 0.5
    let auc10_obs = 1.2
    let thalf10_obs = 1.0
    println("  Cmax pred/obs:")
    println(cmax10)
    println(cmax10_obs)
    println("  FE Cmax:")
    println(fold_error(cmax10, cmax10_obs))
    println("")

    println("==========================================")
    println("  VALIDATION COMPLETE")
    println("  Unit-safe PBPK in Demetrios")
    println("==========================================")

    return 0
}
