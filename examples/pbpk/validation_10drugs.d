// Darwin PBPK 10-Drug Validation Campaign

fn simulate_single_comp(dose: f64, vd: f64, cl: f64, fu: f64, t_hours: f64) -> f64 {
    let k = cl / vd
    let dt = 0.001
    let mut c = dose / vd
    let mut t = 0.0
    
    while t < t_hours {
        c = c * (1.0 - k * fu * dt)
        t = t + dt
    }
    
    return c
}

fn fold_error(predicted: f64, observed: f64) -> f64 {
    if predicted > observed {
        return predicted / observed
    } else {
        return observed / predicted
    }
}

fn main() -> i32 {
    // Drug 1: Midazolam (CYP3A4)
    let c_mid = simulate_single_comp(2.0, 77.0, 30.0, 0.03, 2.0)
    let fe_mid = fold_error(c_mid, 0.01191)
    
    // Drug 2: Caffeine (CYP1A2)
    let c_caf = simulate_single_comp(200.0, 35.0, 5.0, 1.0, 1.0)
    let fe_caf = fold_error(c_caf, 8.0)
    
    // Drug 3: Metformin (OCT2)
    let c_met = simulate_single_comp(500.0, 65.0, 35.0, 0.0, 1.0)
    let fe_met = fold_error(c_met, 2.5)
    
    // Drug 4: Ibuprofen (CYP2C9)
    let c_ibu = simulate_single_comp(400.0, 8.0, 5.0, 0.01, 1.0)
    let fe_ibu = fold_error(c_ibu, 30.0)
    
    // Drug 5: Diazepam (CYP3A4+2C19)
    let c_dia = simulate_single_comp(10.0, 120.0, 0.5, 0.01, 4.0)
    let fe_dia = fold_error(c_dia, 0.5)
    
    // Drug 6: Omeprazole (CYP2C19)
    let c_ome = simulate_single_comp(20.0, 40.0, 30.0, 0.05, 1.0)
    let fe_ome = fold_error(c_ome, 0.8)
    
    // Drug 7: Warfarin (CYP2C9)
    let c_war = simulate_single_comp(5.0, 11.0, 0.13, 0.01, 12.0)
    let fe_war = fold_error(c_war, 1.0)
    
    // Drug 8: Digoxin (P-gp)
    let c_dig = simulate_single_comp(0.5, 500.0, 7.0, 0.75, 6.0)
    let fe_dig = fold_error(c_dig, 0.001)
    
    // Drug 9: Atorvastatin (OATP1B1)
    let c_ato = simulate_single_comp(40.0, 381.0, 172.0, 0.05, 2.0)
    let fe_ato = fold_error(c_ato, 0.06)
    
    // Drug 10: Morphine (glucuronidation)
    let c_mor = simulate_single_comp(10.0, 150.0, 70.0, 0.35, 2.0)
    let fe_mor = fold_error(c_mor, 0.07)
    
    // Count within 2-fold
    let mut count_2fold = 0
    
    if fe_mid < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_caf < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_met < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_ibu < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_dia < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_ome < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_war < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_dig < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_ato < 2.0 {
        count_2fold = count_2fold + 1
    }
    if fe_mor < 2.0 {
        count_2fold = count_2fold + 1
    }
    
    return count_2fold
}
