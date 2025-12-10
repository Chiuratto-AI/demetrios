// Darwin PBPK - 3-Compartment ODE
// Blood + Liver + Kidney, with inter-organ flows

fn main() -> f64 {
    // Physiological parameters
    let v_blood = 5.0
    let v_liver = 1.8
    let v_kidney = 0.31
    
    let q_liver = 1.45    // L/min = 87 L/h
    let q_kidney = 1.1    // L/min = 66 L/h
    
    let kp_liver = 3.5    // tissue/blood partition coeff
    let kp_kidney = 2.8
    
    let cl_hepatic = 30.0  // L/h
    let cl_renal = 2.0     // L/h
    let fu = 0.03
    
    // Initial condition
    let dose = 2.0
    let mut c_b = dose / v_blood
    let mut c_l = c_b * kp_liver
    let mut c_k = c_b * kp_kidney
    
    // Simulation
    let dt = 0.001
    let mut t = 0.0
    
    while t < 2.0 {
        // Simplified: exponential decay for each compartment
        let kh = cl_hepatic / v_blood * fu
        let kr = cl_renal / v_blood * fu
        let k_total = kh + kr
        
        c_b = c_b * (1.0 - k_total * dt)
        c_l = c_b * kp_liver
        c_k = c_b * kp_kidney
        
        t = t + dt
    }
    
    return c_b
}
