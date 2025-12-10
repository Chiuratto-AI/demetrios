// Darwin PBPK Drug Parameters Database
// Clinical pharmacokinetic data from literature

// Drug 1: Midazolam (CYP3A4 substrate)
// Source: FDA NDA 021466, Kharasch et al. 2011
fn midazolam() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 2.0          // mg IV
    let vd = 77.0           // L
    let cl = 30.0           // L/h
    let fu = 0.03           // fraction unbound
    let kp_liver = 3.5
    let t_half = 1.5        // hours
    let cmax_obs = 0.0385   // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 2: Caffeine (CYP1A2 substrate)
// Source: Arnaud 2011 (doi:10.1016/j.fct.2011.06.069)
fn caffeine() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 200.0        // mg oral
    let vd = 35.0           // L
    let cl = 5.0            // L/h
    let fu = 1.0            // fully unbound
    let kp_liver = 1.2
    let t_half = 5.0        // hours
    let cmax_obs = 8.0      // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 3: Metformin (OCT2 substrate, renal clearance)
// Source: Tucker 1981, Scheen 1996
fn metformin() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 500.0        // mg oral
    let vd = 65.0           // L
    let cl = 35.0           // L/h (renal dominant)
    let fu = 0.0            // not protein bound
    let kp_liver = 0.5
    let t_half = 3.0        // hours
    let cmax_obs = 2.5      // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 4: Ibuprofen (CYP2C9 substrate)
// Source: Davies 1998
fn ibuprofen() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 400.0        // mg oral
    let vd = 8.0            // L (very highly protein bound)
    let cl = 5.0            // L/h
    let fu = 0.01           // 99% protein bound
    let kp_liver = 2.0
    let t_half = 2.0        // hours
    let cmax_obs = 30.0     // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 5: Diazepam (CYP3A4 + CYP2C19)
// Source: Mandema 1992
fn diazepam() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 10.0         // mg oral
    let vd = 120.0          // L (highly lipophilic)
    let cl = 0.5            // L/h (slow elimination)
    let fu = 0.01
    let kp_liver = 5.0
    let t_half = 48.0       // hours (very long!)
    let cmax_obs = 0.5      // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 6: Omeprazole (CYP2C19 substrate)
// Source: Andersson 1998
fn omeprazole() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 20.0         // mg oral
    let vd = 40.0           // L
    let cl = 30.0           // L/h
    let fu = 0.05
    let kp_liver = 2.5
    let t_half = 1.0        // hours
    let cmax_obs = 0.8      // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 7: Warfarin (CYP2C9 substrate)
// Source: O'Reilly et al. 1992
fn warfarin() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 5.0          // mg oral
    let vd = 11.0           // L (highly protein bound)
    let cl = 0.13           // L/h (very slow)
    let fu = 0.01           // 99% protein bound
    let kp_liver = 3.0
    let t_half = 37.0       // hours
    let cmax_obs = 1.0      // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 8: Digoxin (P-gp substrate)
// Source: Kersting et al. 1996
fn digoxin() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 0.5          // mg IV loading dose
    let vd = 500.0          // L (very large - tissue binding)
    let cl = 7.0            // L/h (renal + hepatic)
    let fu = 0.75           // mostly unbound
    let kp_liver = 0.8
    let t_half = 40.0       // hours
    let cmax_obs = 0.001    // mg/L (very low - narrow therapeutic window)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 9: Atorvastatin (OATP1B1 substrate)
// Source: Yamazaki et al. 2005
fn atorvastatin() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 40.0         // mg oral
    let vd = 381.0          // L
    let cl = 172.0          // L/h (high clearance)
    let fu = 0.05
    let kp_liver = 8.0      // hepatic uptake
    let t_half = 14.0       // hours (active metabolites longer)
    let cmax_obs = 0.06     // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

// Drug 10: Morphine (glucuronidation)
// Source: Gong et al. 1991
fn morphine() -> (f64, f64, f64, f64, f64, f64, f64) {
    let dose = 10.0         // mg IV
    let vd = 150.0          // L
    let cl = 70.0           // L/h (hepatic glucuronidation)
    let fu = 0.35
    let kp_liver = 1.5
    let t_half = 2.0        // hours
    let cmax_obs = 0.07     // mg/L (observed)
    return (dose, vd, cl, fu, kp_liver, t_half, cmax_obs)
}

fn main() -> i32 {
    // Validate all 10 drugs compile
    let (d1, v1, cl1, fu1, kp1, t1, c1) = midazolam()
    let (d2, v2, cl2, fu2, kp2, t2, c2) = caffeine()
    let (d3, v3, cl3, fu3, kp3, t3, c3) = metformin()
    let (d4, v4, cl4, fu4, kp4, t4, c4) = ibuprofen()
    let (d5, v5, cl5, fu5, kp5, t5, c5) = diazepam()
    let (d6, v6, cl6, fu6, kp6, t6, c6) = omeprazole()
    let (d7, v7, cl7, fu7, kp7, t7, c7) = warfarin()
    let (d8, v8, cl8, fu8, kp8, t8, c8) = digoxin()
    let (d9, v9, cl9, fu9, kp9, t9, c9) = atorvastatin()
    let (d10, v10, cl10, fu10, kp10, t10, c10) = morphine()
    
    return 10
}
