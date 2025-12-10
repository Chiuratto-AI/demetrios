fn euler(conc: f64, clearance: f64, volume: f64, delta_t: f64) -> f64 {
    let rate = 0.0 - clearance / volume * conc
    return conc + rate * delta_t
}

fn main() -> f64 {
    let dose = 2.0
    let clearance = 30.0
    let volume = 77.0
    let delta_t = 0.01
    
    let mut conc = dose / volume
    let mut time = 0.0
    let t_end = 2.0
    
    while time < t_end {
        conc = euler(conc, clearance, volume, delta_t)
        time = time + delta_t
    }
    
    return conc
}
