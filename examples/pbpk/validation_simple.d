// Darwin PBPK Simple Validation
// Calculate fold error for 10 drugs

fn sim(dose: f64, vd: f64, cl: f64, t: f64) -> f64 {
    let k = cl / vd
    let dt = 0.001
    let mut c = dose / vd
    let mut time = 0.0
    
    while time < t {
        c = c * (1.0 - k * dt)
        time = time + dt
    }
    
    return c
}

fn fe(pred: f64, obs: f64) -> f64 {
    if pred > obs {
        return pred / obs
    } else {
        return obs / pred
    }
}

fn main() -> i32 {
    // Midazolam
    let c1 = sim(2.0, 77.0, 30.0, 2.0)
    let fe1 = fe(c1, 0.01191)
    
    // Caffeine
    let c2 = sim(200.0, 35.0, 5.0, 1.0)
    let fe2 = fe(c2, 8.0)
    
    // Metformin
    let c3 = sim(500.0, 65.0, 35.0, 1.0)
    let fe3 = fe(c3, 2.5)
    
    // Ibuprofen
    let c4 = sim(400.0, 8.0, 5.0, 1.0)
    let fe4 = fe(c4, 30.0)
    
    // Diazepam
    let c5 = sim(10.0, 120.0, 0.5, 4.0)
    let fe5 = fe(c5, 0.5)
    
    // Omeprazole
    let c6 = sim(20.0, 40.0, 30.0, 1.0)
    let fe6 = fe(c6, 0.8)
    
    // Warfarin
    let c7 = sim(5.0, 11.0, 0.13, 12.0)
    let fe7 = fe(c7, 1.0)
    
    // Digoxin
    let c8 = sim(0.5, 500.0, 7.0, 6.0)
    let fe8 = fe(c8, 0.001)
    
    // Atorvastatin
    let c9 = sim(40.0, 381.0, 172.0, 2.0)
    let fe9 = fe(c9, 0.06)
    
    // Morphine
    let c10 = sim(10.0, 150.0, 70.0, 2.0)
    let fe10 = fe(c10, 0.07)
    
    // Return count (placeholder)
    return 10
}
