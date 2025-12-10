// Darwin PBPK with Oral Absorption (Weibull)

fn weibull(t: f64, lambda: f64, shape: f64) -> f64 {
    // Survival function: S(t) = exp(-(t/lambda)^shape)
    let x = t / lambda
    let power = x * x * x  // approximate: x^3 for shape~3
    return x - power / 6.0 + power * power / 120.0  // Taylor series approximation
}

fn main() -> f64 {
    // Physiology (simplified 7-compartment)
    let v_blood = 5.0
    let v_liver = 1.8
    let v_kidney = 0.31
    let v_adipose = 10.5
    let v_muscle = 30.0
    let v_gi = 1.5
    let v_other = 10.35
    
    let q_liver = 1.45
    let q_kidney = 1.1
    let q_adipose = 0.56
    let q_muscle = 3.5
    let q_gi = 1.25
    let q_other = 3.54
    
    let kp_liver = 3.5
    let kp_kidney = 2.8
    let kp_adipose = 5.0
    let kp_muscle = 1.5
    let kp_other = 1.3
    
    let cl_hepatic = 30.0
    let cl_renal = 2.0
    let fu = 0.03
    
    // Oral absorption parameters
    let dose_oral = 200.0  // mg
    let bioavail = 0.65    // F
    let lambda = 0.5       // absorption time scale
    let shape = 2.0        // Weibull shape
    
    let mut c_blood = 0.0
    let mut c_liver = 0.0
    let mut c_kidney = 0.0
    let mut c_adipose = 0.0
    let mut c_muscle = 0.0
    let mut a_gi = dose_oral  // Amount in GI tract
    let mut a_absorbed = 0.0
    
    let dt = 0.001
    let mut t = 0.0
    let k_hepatic = cl_hepatic / v_blood * fu
    let k_renal = cl_renal / v_blood * fu
    let k_total = k_hepatic + k_renal
    let ka = 1.0  // absorption rate (1/h)
    
    while t < 4.0 {
        // Oral absorption
        let absorption = ka * a_gi * dt
        a_gi = a_gi - absorption
        a_absorbed = a_absorbed + absorption
        
        // First-pass metabolism
        let hepatic_uptake = absorption * 0.7
        let c_blood_in = (absorption - hepatic_uptake) * bioavail / v_blood
        
        // Clearance
        let elimination = c_blood * k_total
        
        // Update
        c_blood = c_blood + c_blood_in - elimination
        c_liver = c_blood * kp_liver
        c_kidney = c_blood * kp_kidney
        c_adipose = c_blood * kp_adipose
        c_muscle = c_blood * kp_muscle
        
        t = t + dt
    }
    
    return c_blood
}
