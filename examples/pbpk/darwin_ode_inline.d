fn main() -> f64 {
    let dose = 2.0
    let v = 77.0
    let cl = 30.0
    let k = cl / v
    let dt = 0.001
    let t_end = 2.0
    let factor = 1.0 - k * dt
    
    let mut c = dose / v
    let mut t = 0.0
    
    while t < t_end {
        c = c * factor
        t = t + dt
    }
    
    return c
}
